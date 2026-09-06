"""2단계 크롭 — 비례 창(`m2.5`) vs 고정 창(`f320`/`f448`) (STEP 22).

    uv run --extra train python tools/stage2_size_sweep.py

묻는 것 하나
------------
STEP 21: VL01 안에서 A4 병변이 A1 보다 **1.68배** 큽니다. A1·A4 를 둘 다 가진
개 72마리로 좁혀도 **1.88배**(Wilcoxon p=8.1e-05)라 "다른 개를 다르게 찍었다"
로는 설명이 안 됩니다.

그런데 `m2.5` 는 창 = 병변 × 2.5 라 병변이 언제나 프레임의 40% 를 차지합니다:

    A1 117px → 창 292px → 384 입력에서 154px
    A4 196px → 창 490px → 384 입력에서 154px   ← 절대 크기가 지워집니다

**고정 창은 그 크기를 남깁니다.** 그럼 A4↔A1 이 덜 헷갈릴까요?
STEP 16 혼동행렬에서 A4→A1 이 0.25 로 **대각선과 동률**이었습니다.

⚠️ **좋아져도 바로 채택할 수 없습니다.** 절대 크기를 쓰면 배율 교란에 약해지고,
   배포에서 보호자의 촬영 거리는 통제되지 않습니다. 판정이 **세 갈래**인 이유입니다
   (`experiments.stage2_size_report`).

⚠️ **VL01 만** 씁니다 (병변 18,953장). A1 4,051 · A4 1,526 이 다 있어서 이
   질문에는 충분하고, 크롭 세 종류가 이미 로컬에 있어 **재크롭이 필요 없습니다.**

⚠️ 백본은 **`effnetv2_s`** 입니다 — 축은 크롭이고 `convnextv2_base` 는 4.6배
   느려 3판이 8시간이 됩니다. 효과가 크면 확정 백본으로 다시 확인합니다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402

from src import crop, evaluate, experiments, labels, split, stages, train   # noqa: E402
from src.config import CLASSES                                  # noqa: E402

BASE_TAG = "m2.5"          # 지금 확정된 2단계 크롭 (STEP 4C)
FOCUS, OTHER = "A4", "A1"  # STEP 16 에서 A4→A1 이 대각선과 동률


def extra_metrics(exp: str, boot: int = 1000) -> dict:
    """캐시된 val 로짓으로 관심 클래스 recall·CI 와 A4→A1 혼동률을 냅니다."""
    z = np.load(train.ckpt_dir(exp) / "logits_val.npz", allow_pickle=False)
    y, pred = z["y"], z["logits"].argmax(1)
    fi, oi = CLASSES.index(FOCUS), CLASSES.index(OTHER)
    m = y == fi
    if not m.any():
        raise SystemExit(f"[X] val 에 {FOCUS} 가 없습니다 — 분할을 확인하세요.")
    _, lo, hi = evaluate.bootstrap_ci(y, pred, metric="recall", cls=fi, n=boot)
    return {
        "focus_recall": float((pred[m] == fi).mean()),
        "focus_recall_ci": float((hi - lo) / 2),
        # A4 중 A1 이라고 불린 비율 — 혼동행렬의 그 칸
        "f2o_rate": float((pred[m] == oi).mean()),
        "n_focus": int(m.sum()),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="manifest_final.parquet")
    ap.add_argument("--chunk", default="chunk_VL01")
    ap.add_argument("--tags", default="m2.5,f320,f448")
    ap.add_argument("--model", default="effnetv2_s")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--n-robust", type=int, default=2000)
    ap.add_argument("--out", default="data/work/reports/step22_stage2_size.json")
    a = ap.parse_args(argv)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        raise SystemExit("[X] GPU 가 없습니다. 3판 학습은 CPU 로는 며칠 걸립니다.")
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
        view = stages.to_stage2(crop.switch_tag(df, tag, verbose=False))
        split.verify(view, fold=0, strict=True)
        r = experiments.train_and_measure(
            view, stage=2, img_size=384, crop_tag=tag, device=dev,
            model_name=a.model, finetune="moderate", aug="default",
            epochs=a.epochs, subset_frac=1.0,
            measure_robust=True,          # ★ 배율 하락이 이 실험의 핵심 가드입니다
            measure_blur=False, n_robust=a.n_robust)
        r.update(extra_metrics(r["exp_name"]))
        print(f"  macro-F1 {r['macro_f1']:.4f} · {FOCUS} recall {r['focus_recall']:.3f}"
              f" (±{r['focus_recall_ci']:.3f}, n={r['n_focus']:,})"
              f" · {FOCUS}→{OTHER} {r['f2o_rate']:.1%}"
              f" · 배율 하락 {r.get('scale_drop', float('nan')):.1%}")
        runs.append(r)

    verdict = experiments.stage2_size_report(runs, base_crop=BASE_TAG,
                                             focus=FOCUS, other=OTHER)
    p = ROOT / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(
        {"step": "STEP 22 — 2단계 비례 창 vs 고정 창", "chunk": a.chunk,
         "model": a.model, "epochs": a.epochs, "focus": FOCUS, "other": OTHER,
         "verdict": verdict,
         "runs": [{k: v for k, v in r.items() if k != "report"} for r in runs]},
        indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"\n저장: {p}")


if __name__ == "__main__":
    main()
