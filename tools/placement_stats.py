"""1단계 놓침이 **병변 탓인가 창 배치 탓인가** — 폴리곤으로 가릅니다 (GPU 0).

    uv run --extra train python tools/placement_stats.py
    uv run --extra train python tools/placement_stats.py --chunk chunk_VL01

왜 이걸 재나
------------
`rho(병변 크기, 1단계 점수) = -0.201` 로 **큰 병변을 더 놓칩니다** (STEP 17·18).
STEP 20 이 "창이 작아서" 를 기각했고(f448 로 넓히니 8.0% → 9.1% 로 악화),
남은 설명이 둘이었습니다 — 그리고 **아직 안 갈렸습니다** (CLAUDE.md 결론표).

이제 갈릴 수 있는 이유: **매니페스트 100% 행에 폴리곤(병변 외곽선)이 있고**,
`bbox` 는 그 폴리곤의 외접사각형입니다 (VL01 4,000행 오차 **0px**, 표준편차 0).
지금까지 우리가 쓴 "병변 크기" 는 전부 네모였는데, 길쭉하거나 굽은 병변에서
네모는 실제 병변보다 훨씬 크고 **중심이 빈 곳에 놓입니다.**

    H-A 병변    큰 병변이 원래 어렵다 → 창이 병변으로 **꽉 찰수록** 더 놓침
    H-B' 창배치  네모 중심이 병변을 벗어난다 → 창에 병변이 **적을수록** 더 놓침

같은 값(`occ` = 창 안 병변 점유율)에 정반대를 예측합니다. 판정 규칙은
`experiments.stage1_placement_report()` 에 **미리** 박혀 있습니다 (작업 규칙 2).

⚠️ 이건 **관찰**입니다. H-B' 가 지지돼도 "창을 옮기면 낫다" 는 아직 아닙니다 —
   그건 폴리곤 무게중심으로 다시 잘라 재는 개입 실험이 말합니다 (`--intervene`).
⚠️ holdout 은 열지 않습니다 (06 만 엽니다). 판정은 val 로.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import crop, env, experiments, labels, split, stages  # noqa: E402

DEFAULT_MANIFEST = ROOT / "manifest_final.parquet"


def _longside(row) -> float:
    b = crop._box4(row.get("bbox"))
    return max(b[2] - b[0], b[3] - b[1]) if b else float("nan")


def build(df: pd.DataFrame, tag: str, exp: str, chunk: str) -> pd.DataFrame:
    """val 병변 행에 1단계 점수와 폴리곤 기하를 붙입니다."""
    view = stages.to_stage1(crop.switch_tag(df, tag, verbose=False))
    _, va = split.get_fold(view, 0)
    va = va.reset_index(drop=True)

    npz = env.work_root() / "checkpoints" / exp / f"logits_{chunk}_val.npz"
    if not npz.exists():
        raise SystemExit(
            f"[X] 1단계 로짓이 없습니다: {npz}\n"
            f"    tools/local_logits.py --stages 1 --chunk {chunk.replace('chunk_','')} 로 먼저 뽑으세요.")
    z = np.load(npz, allow_pickle=False)
    lg, y = z["logits"], z["y"]
    if len(lg) != len(va):
        raise SystemExit(
            f"[X] 행 수가 다릅니다 — 로짓 {len(lg):,} vs val {len(va):,}.\n"
            f"    매니페스트가 그 로짓을 만든 것과 다릅니다 (VL01 단독 45,885행 vs 전체 365,428행).")

    p = np.exp(lg - lg.max(1, keepdims=True))
    va["score"] = p[:, 1] / p.sum(1)
    va["is_lesion"] = y == 1

    les = va[va["is_lesion"]].copy()
    geo = les.apply(lambda r: crop.polygon_in_window(r, tag), axis=1, result_type="expand")
    les = pd.concat([les.reset_index(drop=True), geo.reset_index(drop=True)], axis=1)
    les["box_px"] = les.apply(_longside, axis=1)
    les["poly_px"] = np.sqrt(les["polygon"].apply(crop.polygon_area))
    return les.dropna(subset=["occ", "box_px"]).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--chunk", default="chunk_VL01")
    ap.add_argument("--tag", default="f320")
    ap.add_argument("--exp", default=None, help="비우면 stage1_threshold.json 이 지정한 것")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    env.load_prepared()
    rec = json.loads((env.work_root() / "stage1_threshold.json").read_text(encoding="utf-8"))
    exp = a.exp or rec["stage1_exp"]
    thr = rec["threshold"]
    print(f"[모델] {exp}\n[임계값] {thr:.4f}  (target recall {rec['target_recall']})")

    df = labels.load(a.manifest)
    df = df[df["chunk"] == a.chunk].reset_index(drop=True)
    print(f"[데이터] {a.chunk} {len(df):,}행")

    les = build(df, a.tag, exp, a.chunk)
    les["miss"] = les["score"] < thr
    print(f"[val 병변] {len(les):,}행 · 놓침 {int(les['miss'].sum())}건 "
          f"({les['miss'].mean():.1%})")

    # ── 네모가 병변을 얼마나 부풀리나 ────────────────────────────────
    print(f"\n네모(bbox)는 폴리곤의 외접사각형입니다. 그 사이의 빈 곳:")
    print(f"  slack = 1 - 폴리곤넓이/네모넓이   중앙값 {les['slack'].median():.3f}"
          f"  (사분위 {les['slack'].quantile(.25):.3f} ~ {les['slack'].quantile(.75):.3f})")
    print(f"  네모 중심이 병변 **밖**인 비율: {(~les['center_on'].astype(bool)).mean():.1%}")
    print(f"  창 안 병변 점유율 occ  중앙값 {les['occ'].median():.3f}")
    print(f"  병변이 창에 다 담긴 비율(captured=1): {(les['captured'] >= 0.999).mean():.1%}")

    print("\n네모 중심이 병변 안/밖 별 놓침률")
    print(les.groupby(les["center_on"].astype(bool), observed=True)["miss"]
          .agg(n="size", 놓침률="mean").to_string())

    # ── 사전등록 판정 (공선성이면 스스로 거부합니다) ──────────────────
    verdict = experiments.stage1_placement_report(
        les[["occ", "miss", "box_px", "score"]])

    # ── 크기를 고정한 모양 검사 — 위가 거부될 때 실제로 가르는 쪽 ──────
    les["ecc"] = les["bbox"].apply(
        lambda b: (lambda q: max(q[2] - q[0], q[3] - q[1])
                   / max(min(q[2] - q[0], q[3] - q[1]), 1e-6))(crop._box4(b))
        if crop._box4(b) else float("nan"))
    shape = experiments.stage1_shape_report(
        les[["box_px", "score", "slack", "center_off", "ecc"]])

    print("\n크기 4분위 x 모양 2분할 놓침률 (모양이 크기 안에서 움직이나)")
    les["size_q"] = pd.qcut(les["box_px"], 4, labels=["S1", "S2", "S3", "S4"])
    for v in ("slack", "center_off", "ecc"):
        les[f"{v}_h"] = les.groupby("size_q", observed=True)[v].transform(
            lambda s: pd.qcut(s, 2, labels=["낮음", "높음"], duplicates="drop"))
        t = les.pivot_table(index="size_q", columns=f"{v}_h", values="miss",
                            aggfunc="mean", observed=True)
        print(f"  --- {v} ---")
        print(t.to_string(float_format=lambda x: f"{x:.3f}"))

    out = {"step": "STEP 24 — 1단계 놓침: 병변인가 창 배치인가",
           "chunk": a.chunk, "tag": a.tag, "exp": exp, "threshold": thr,
           "n_lesion_val": int(len(les)), "n_miss": int(les["miss"].sum()),
           "slack_median": float(les["slack"].median()),
           "center_off_poly_frac": float((~les["center_on"].astype(bool)).mean()),
           "occ_median": float(les["occ"].median()),
           "verdict": verdict, "shape": shape}
    p = Path(a.out) if a.out else env.work_root() / "reports" / "step24_placement.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float),
                 encoding="utf-8")
    print(f"\n저장: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
