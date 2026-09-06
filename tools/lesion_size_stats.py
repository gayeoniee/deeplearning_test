"""1단계가 놓치는 병변은 **큰 병변**인가 — 고정 창(f320)과 병변 크기의 관계.

    uv run --extra train python tools/lesion_size_stats.py
    uv run --extra train python tools/lesion_size_stats.py \
        --from-saved data/work/reports/false_alarm_stats.parquet \
        --manifest data/work/manifests/manifest_final.parquet

왜 이 도구가 있나
-----------------
STEP 16 에서 1단계가 A4(농포) 병변의 **12%** 를 "괜찮아요" 로 내보냈습니다
(A3 의 6배). 처음 세운 가설은 "A4 병변이 제일 작아서 잘 안 보인다" 였는데,
로컬 실측에서 **정반대**가 나왔습니다 — 모든 클래스에서 **큰 병변일수록**
1단계 점수가 낮습니다.

⚠️ 이 도구는 그 관찰을 재현 가능하게 만들 뿐, "원인" 을 말하지 않습니다.
   아래 두 가지가 아직 안 갈렸습니다:

   ① **창 채움** — `f320` 은 bbox 중심에 **고정 320px** 창을 놓습니다. 병변이
      크면 창이 병변으로 꽉 차서 **주변 정상 피부가 안 보입니다.** 모델이
      "주변과의 대비" 로 이상을 판정한다면 여기서 무너집니다
   ② **큰 병변 자체가 어렵다** — 창과 무관하게 넓고 경계가 흐린 병변이
      원래 애매하다 (라벨러가 대충 큰 네모를 그렸을 수도)

   **가르는 법**: 2단계는 `m2.5`(병변 크기에 **비례**하는 창)를 씁니다.
   거기서는 채움 비율이 1/2.5 로 **항상 같습니다.** 그러니
   · 2단계에 크기 효과가 **없으면** → ① 창 채움
   · 2단계에도 **있으면**        → ② 병변 자체
   2단계 로짓이 생기면 같은 검사를 그쪽에 돌립니다.

읽는 법
-------
`occ` = 병변 긴 변 ÷ 1단계 창 크기(320px).

    occ 0.3  창의 30% 만 병변. 주변 정상 피부가 넉넉히 보임
    occ 1.0  병변이 창을 꽉 채움
    occ 2.0  병변이 창보다 2배 커서 **병변 한가운데만** 보임
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402

# ── 사전등록 판정 기준 (결과를 보고 바꾸지 않습니다 — 작업 규칙 2) ──────
#   창을 넘치는 쪽(occ>1)의 놓침률이 안 넘치는 쪽의 몇 배여야 "효과 있음" 인가.
#   실행 간 잡음이 크므로 배수로 잡고, 클래스마다 같은 방향인지도 같이 봅니다.
OVERFLOW_RATIO_STRONG = 1.5
#   부분상관이 이만큼은 돼야 "선명도의 그림자가 아니다" 로 봅니다.
PARTIAL_RHO_MIN = 0.15
#   클래스별 검사를 할 최소 표본 (이보다 적으면 건너뜁니다)
MIN_N = 20

STAGE1_WINDOW_PX = 320          # f320. `crop.fixed_of_tag()` 와 같은 값이어야 합니다

OCC_BINS = [0, .25, .5, .75, 1.0, 1.5, 2.0, np.inf]
OCC_LABELS = ["<0.25", "0.25-0.5", "0.5-0.75", "0.75-1.0",
              "1.0-1.5", "1.5-2.0", ">2.0"]


def _ratio(bad: float, good: float) -> float:
    """놓침률의 배수. ⚠️ 분모가 0 이면 nan 이 아니라 **inf** 입니다.

    한쪽이 하나도 안 틀렸는데 다른 쪽이 틀렸으면 그건 "효과가 가장 센" 경우지
    "잴 수 없는" 경우가 아닙니다. nan 으로 두면 판정이 조용히 '미지지' 로
    떨어집니다 — 검사가 잡아낸 자리입니다.
    """
    if good > 0:
        return bad / good
    return float("inf") if bad > 0 else float("nan")


def bbox_long_side(bbox_series) -> np.ndarray:
    """매니페스트의 `bbox` 는 xyxy 를 담은 **문자열**입니다 (list 가 아닙니다)."""
    b = np.array([json.loads(s) if isinstance(s, str) else list(s)
                  for s in bbox_series], dtype=float)
    return np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])


def partial_spearman(df: pd.DataFrame, x: str, y: str, ctrl: str,
                     deg: int = 3) -> float:
    """`ctrl` 을 3차 다항으로 걷어낸 뒤의 순위 상관.

    `tools/false_alarm_stats.py` 의 `residualize` 와 같은 방식입니다 —
    그쪽은 AUROC 용이고 여기는 연속값 상관용이라 계산만 따로 둡니다.
    """
    from scipy.stats import spearmanr
    R = df[[x, y, ctrl]].rank()
    rx = R[x] - np.poly1d(np.polyfit(R[ctrl], R[x], deg))(R[ctrl])
    ry = R[y] - np.poly1d(np.polyfit(R[ctrl], R[y], deg))(R[ctrl])
    return float(spearmanr(rx, ry).statistic)


def attach(saved: Path, manifest: Path, window_px: int) -> pd.DataFrame:
    """저장된 1단계 점수에 매니페스트의 bbox 를 붙입니다."""
    fa = pd.read_parquet(saved)
    need = {"image_path", "score", "said_abnormal", "is_normal"}
    missing = need - set(fa.columns)
    if missing:
        raise SystemExit(f"[X] {saved} 에 {sorted(missing)} 이 없습니다.")

    mf = pd.read_parquet(manifest, columns=["image_path", "label", "bbox"])
    # ⚠️ `label` 이 양쪽에 있으면 _x/_y 로 갈라집니다. 저장본의 label 은
    #    A7/ABNORMAL 로 뭉개져 있으므로 **매니페스트 쪽을 씁니다.**
    d = fa.drop(columns=["label"], errors="ignore").merge(
        mf, on="image_path", how="left")

    # ⚠️ 붙일 값이 0장이면 **멈춥니다.** 조용히 NaN 이 되면 없는 걸 재면서
    #    분석이 끝까지 돕니다 (CLAUDE.md 함정).
    lost = int(d["label"].isna().sum())
    if lost:
        raise SystemExit(
            f"[X] {lost:,}/{len(d):,} 행이 매니페스트에 없습니다 — 다른 데이터입니다.\n"
            f"    저장본     {saved}\n    매니페스트 {manifest}")

    d["long"] = bbox_long_side(d["bbox"])
    d["occ"] = d["long"] / window_px
    return d


def report(d: pd.DataFrame, window_px: int) -> dict:
    from scipy.stats import spearmanr
    les = d[~d["is_normal"]].copy()
    les["bin"] = pd.cut(les["occ"], OCC_BINS, labels=OCC_LABELS)
    out: dict = {"n_lesion": len(les), "window_px": window_px}

    print(f"\n=== 1단계 놓침률 vs 창 채움 (병변 {len(les):,}장, 창 {window_px}px) ===")
    print(f"{'occ':>10}{'n':>7}{'놓침':>6}{'놓침률':>8}{'점수 중앙값':>12}")
    rows = []
    for k, s in les.groupby("bin", observed=True):
        r = float((~s["said_abnormal"]).mean())
        print(f"{str(k):>10}{len(s):>7}{int((~s['said_abnormal']).sum()):>6}"
              f"{r:>8.1%}{s['score'].median():>12.3f}")
        rows.append({"occ": str(k), "n": len(s), "miss_rate": r})
    out["by_occ"] = rows

    sub, ovr = les[les["occ"] <= 1], les[les["occ"] > 1]
    r_sub = float((~sub["said_abnormal"]).mean())
    r_ovr = float((~ovr["said_abnormal"]).mean())
    ratio = _ratio(r_ovr, r_sub)
    print(f"\n  창 안 (occ<=1): n={len(sub):,}  놓침률 {r_sub:.1%}")
    print(f"  창 밖 (occ>1) : n={len(ovr):,}  놓침률 {r_ovr:.1%}   -> {ratio:.1f}배")
    out.update(miss_within=r_sub, miss_overflow=r_ovr, overflow_ratio=ratio)

    print(f"\n=== 클래스별로도 같은 방향인가 (최소 {MIN_N}장) ===")
    print(f"{'':4}{'occ<=1 n':>10}{'놓침률':>8}{'occ>1 n':>9}{'놓침률':>8}{'배수':>7}")
    per: dict = {}
    for c in sorted(les["label"].unique()):
        s = les[les["label"] == c]
        a, o = s[s["occ"] <= 1], s[s["occ"] > 1]
        if len(a) < MIN_N or len(o) < MIN_N:
            continue
        ra, ro = float((~a["said_abnormal"]).mean()), float((~o["said_abnormal"]).mean())
        k = _ratio(ro, ra)
        per[c] = k
        print(f"{c:4}{len(a):>10}{ra:>8.1%}{len(o):>9}{ro:>8.1%}{k:>7.1f}x")
    out["per_class_ratio"] = per

    print("\n=== 크기 효과가 선명도(blur)의 그림자인가 ===")
    if "blur" in les.columns:
        raw = float(spearmanr(les["long"], les["score"]).statistic)
        par = partial_spearman(les, "long", "score", "blur")
        print(f"  rho(크기, 점수)          = {raw:+.3f}")
        print(f"  rho(크기, 점수 | 선명도) = {par:+.3f}")
        out.update(rho_size_score=raw, rho_partial=par)
        for c in sorted(les["label"].unique()):
            s = les[les["label"] == c]
            if len(s) < MIN_N:
                continue
            print(f"    {c}: {partial_spearman(s, 'long', 'score', 'blur'):+.3f}")
    else:
        print("  (blur 열이 없어 건너뜁니다)")

    print("\n=== 판정 ===")
    strong = bool(ratio >= OVERFLOW_RATIO_STRONG)
    same_dir = bool(per) and all(v >= 1.0 for v in per.values())
    not_shadow = abs(out.get("rho_partial", 0.0)) >= PARTIAL_RHO_MIN
    for name, ok, why in [
        ("창 밖 놓침률 배수", strong, f"{ratio:.1f}배 >= {OVERFLOW_RATIO_STRONG}"),
        ("클래스 방향 일치", same_dir, "모든 클래스에서 배수 >= 1.0"),
        ("선명도의 그림자 아님", not_shadow, f"|부분상관| >= {PARTIAL_RHO_MIN}"),
    ]:
        print(f"  {'[O]' if ok else '[X]'} {name:<18} ({why})")
    v = "크기 효과 있음" if (strong and same_dir and not_shadow) else "미지지"
    out["verdict"] = v
    print(f"\n  -> {v}")
    print("\n⚠️ 이건 **1단계(f320, 고정 창)** 얘기입니다. 창 채움 때문인지 병변"
          " 자체가\n   어려워서인지는 2단계(m2.5, 비례 창)로 같은 검사를 해야 갈립니다.")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-saved",
                    default="data/work/reports/false_alarm_stats.parquet",
                    help="false_alarm_stats.py 가 --save 로 남긴 파일")
    ap.add_argument("--manifest",
                    default="data/work/manifests/manifest_final.parquet")
    ap.add_argument("--window-px", type=int, default=STAGE1_WINDOW_PX)
    ap.add_argument("--out", default=None, help="판정을 JSON 으로 저장할 경로")
    a = ap.parse_args(argv)

    saved, manifest = ROOT / a.from_saved, ROOT / a.manifest
    for p in (saved, manifest):
        if not p.exists():
            raise SystemExit(f"[X] {p} 가 없습니다.")
    print(f"저장본     {saved}\n매니페스트 {manifest}")

    d = attach(saved, manifest, a.window_px)
    out = report(d, a.window_px)

    if a.out:
        p = ROOT / a.out
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n저장: {p}")


if __name__ == "__main__":
    main()
