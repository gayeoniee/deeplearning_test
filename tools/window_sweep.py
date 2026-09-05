"""1단계 고정 창 **크기** 2판 — `f320` vs `f448` (STEP 20).

    uv run --extra train python tools/window_sweep.py

묻는 것 하나
------------
STEP 18 에서 1단계가 **큰 병변일수록 놓친다**는 걸 찾았고 (rho −0.201, 6개 클래스
전부 음수), 2단계(비례 창)에는 그 효과가 없어서 원인이 **고정 창이 병변으로
꽉 차는 것**으로 좁혀졌습니다. 창 안 놓침 1.8% vs 창 밖 4.6% (2.5배).

그럼 **창을 키우면 낫나?** 전체 병변의 23.2% 가 320px 창을 넘고, 448 로 키우면
12.4% 로 줍니다. 대신 작은 병변이 그만큼 작아집니다 — 맞바꿈이라 재봐야 합니다.

판정은 `experiments.stage1_window_report()` 에 **미리** 박혀 있습니다
(노트북·스크립트 셀은 결과를 보고 고칠 수 있으므로 — 작업 규칙 3).

⚠️ **VL01 청크만** 씁니다 (39,508행). TL01·TL02 원본이 이 PC 에 없습니다.
   1단계에는 오히려 깔끔합니다 — VL01 안에서 정상 20,555 / 이상 18,953 로
   거의 균형입니다. 다만 **순위가 전체 데이터와 같다는 보장은 없습니다.**

⚠️ 창을 **비례**로 바꾸면 안 됩니다 — STEP 9-A 에서 ROI 크롭이 정답을 흘려
   AUROC 가 0.8272 → 0.9477 로 갈렸습니다. 고정인 채 **크기만** 바꿉니다.
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
import torch                                                    # noqa: E402

from src import crop, evaluate, experiments, labels, split, stages, train   # noqa: E402
from src.config import CLASSES_STAGE1                           # noqa: E402

BASE_TAG = "f320"          # 지금 확정된 1단계 크롭 (STEP 9-A)
REF_WINDOW_PX = 320        # ⚠️ '넘침' 의 정의는 **기준 창**에 고정합니다


def _long_side(v) -> float:
    a = np.asarray(json.loads(v) if isinstance(v, str) else v, dtype=float)
    return float(max(a[2] - a[0], a[3] - a[1]))


def extra_metrics(exp: str, va: pd.DataFrame, threshold: float,
                  boot: int = 1000) -> dict:
    """캐시된 val 로짓으로 **넘친 병변 / 안 넘친 병변**의 놓침률을 가릅니다."""
    z = np.load(train.ckpt_dir(exp) / "logits_val.npz", allow_pickle=False)
    logits, y = z["logits"], z["y"]
    if len(y) != len(va):
        raise SystemExit(f"[X] 행 수가 안 맞습니다: 로짓 {len(y)} vs va {len(va)}. "
                         "크롭이 빠진 행이 있는지 확인하세요.")
    score = stages.stage1_scores(logits)
    said = score >= threshold

    d = va.reset_index(drop=True).copy()
    d["long"] = d["bbox"].apply(_long_side)
    d["said"] = said
    d["is_lesion"] = stages.binary_targets(y).astype(bool)

    les, nor = d[d.is_lesion], d[~d.is_lesion]
    ovf, wit = les[les["long"] > REF_WINDOW_PX], les[les["long"] <= REF_WINDOW_PX]

    # 놓침률의 CI 는 "그 부분집합 안에서 병변을 얼마나 잡나" = recall 의 CI 입니다.
    yy = np.zeros(len(ovf), dtype=int)          # 전부 병변(클래스 0)
    pp = np.where(ovf["said"].to_numpy(), 0, 1)  # 잡았으면 0, 놓쳤으면 1
    _, lo, hi = evaluate.bootstrap_ci(yy, pp, metric="recall", cls=0, n=boot)

    return {
        "false_alarm": float(nor["said"].mean()),
        "overflow_miss": float((~ovf["said"]).mean()),
        "within_miss": float((~wit["said"]).mean()),
        "overflow_miss_ci": float((hi - lo) / 2),
        "n_overflow": int(len(ovf)), "n_within": int(len(wit)),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="manifest_final.parquet")
    ap.add_argument("--chunk", default="chunk_VL01")
    ap.add_argument("--tags", default="f320,f448")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--subset", type=float, default=1.0)
    ap.add_argument("--n-robust", type=int, default=2000)
    ap.add_argument("--out", default="data/work/reports/step20_window.json")
    a = ap.parse_args(argv)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        raise SystemExit("[X] GPU 가 없습니다. 1단계 2판 학습은 CPU 로는 며칠 걸립니다.")
    print(f"GPU {torch.cuda.get_device_name(0)}")

    df = labels.load(ROOT / a.manifest)
    df = df[df["chunk"] == a.chunk].reset_index(drop=True)
    print(f"{a.chunk} {len(df):,}행 / 개체 {df['animal_id'].nunique():,}마리")

    tags = [t.strip() for t in a.tags.split(",")]
    have = crop.available_tags()
    missing = [t for t in tags if t not in have]
    if missing:
        raise SystemExit(f"[X] 크롭 태그가 없습니다: {missing}  (있는 것: {have})")

    runs = []
    for tag in tags:
        print(f"\n{'#' * 70}\n 판 {tag}\n{'#' * 70}")
        view = stages.to_stage1(crop.switch_tag(df, tag, verbose=False))
        split.verify(view, fold=0, strict=True)
        r = experiments.train_and_measure(
            view, stage=1, img_size=384, crop_tag=tag, device=dev,
            model_name="effnetv2_s", finetune="moderate", aug="photometric",
            epochs=a.epochs, subset_frac=a.subset,
            measure_robust=False,        # 1단계 판정은 배율이 아니라 화질입니다
            measure_blur=True, n_robust=a.n_robust)

        _, va = split.get_fold(view, 0)
        r.update(extra_metrics(r["exp_name"], va, r["threshold"]))
        print(f"  AUROC {r['auroc']:.4f} · 헛알림 {r['false_alarm']:.1%} · "
              f"넘친것 놓침 {r['overflow_miss']:.1%} (n={r['n_overflow']:,}) · "
              f"안넘친것 {r['within_miss']:.1%} (n={r['n_within']:,})")
        runs.append(r)

    verdict = experiments.stage1_window_report(runs, base_crop=BASE_TAG)

    p = ROOT / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(
        {"step": "STEP 20 — 1단계 고정 창 크기", "chunk": a.chunk,
         "ref_window_px": REF_WINDOW_PX, "epochs": a.epochs,
         "subset_frac": a.subset, "verdict": verdict,
         "runs": [{k: v for k, v in r.items() if k != "report"} for r in runs]},
        indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"\n저장: {p}")


if __name__ == "__main__":
    main()
