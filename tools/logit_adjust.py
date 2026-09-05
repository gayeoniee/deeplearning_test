"""저장된 로짓에 **사전확률(prior)을 빼서** A4 가 살아나는지 봅니다 — 재학습 없음.

    uv run --extra train python tools/logit_adjust.py

무엇을 가르려는 것인가
----------------------
STEP 16 혼동행렬에서 소수 클래스(A4·A5·A6)의 오답이 다수 클래스(A1·A2·A3)로
흐릅니다. 설명이 둘인데 **처방이 정반대**입니다:

| | 설명 | 맞다면 |
|---|---|---|
| **prior** | 표현은 있는데 학습 분포가 흔한 클래스 쪽으로 밀어둔 것 | 로짓만 고치면 됨 (**공짜**) |
| **표현 부재** | A4 를 볼 줄 모름 | 크롭·데이터·손실을 바꿔야 함 (비쌈) |

혼동행렬만으로는 못 가릅니다 — **둘 다 똑같이 "흩어짐" 으로 보입니다.**
그래서 개입해 봅니다. 학습 분포를 로짓에서 빼는 건 한 줄이고 몇 초입니다:

    logit_c  ←  logit_c − τ · log π_c        (π = 학습셋 클래스 비율)

τ=0 이 지금 모델, τ=1 이 "prior 를 완전히 걷어낸" 상태입니다.
**prior 가 원인이면 A4 recall 이 크게 오르고, 아니면 거의 안 움직입니다.**

⚠️ 공짜로 좋아지는 게 아닙니다. 소수 클래스를 밀어 올리면 다수 클래스 recall 과
   소수 클래스 precision 이 내려갑니다. 그래서 **macro-F1 이 심판**입니다.

⚠️ **보정 전 로짓**에 걸어야 합니다. `calibrate.apply()` 는 확률을 돌려주므로
   거기에 걸면 softmax 가 두 번 먹습니다 (이 리포가 ECE 0.03 → 0.20 으로 겪음).
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

from src import errors                                          # noqa: E402
from src.evaluate import bootstrap_ci                           # noqa: E402
from src.experiments import MACRO_F1_NOISE                      # noqa: E402

# ── 사전등록 판정 (결과를 보고 바꾸지 않습니다 — 작업 규칙 2) ────────────
#   채택 후보가 되려면 macro-F1 이 잡음 밖에서 올라야 합니다.
ADOPT_MACRO_F1_GAIN = MACRO_F1_NOISE          # 0.02, experiments.py 와 같은 값
#   "prior 가 주원인" 으로 인정하려면 A4 의 쏠림 lift 가 여기까지 내려와야 합니다.
PRIOR_EXPLAINED_LIFT = 1.10
#   A4 recall 개선은 **그 실행에서 계산한 CI 반폭**보다 커야 인정합니다.
TAU_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def train_prior(manifest: Path, classes: list[str]) -> np.ndarray:
    """모델이 실제로 학습한 분포. **val 이 아니라 train** 입니다."""
    from src import crop, split, stages

    df = pd.read_parquet(manifest, columns=["label", "fold", "is_holdout",
                                            "image_path", "crop_path", "crop_tag",
                                            "crop_rel", "is_normal"])
    s2 = stages.to_stage2(df)
    tr, _ = split.get_fold(s2, 0)
    cnt = tr["label"].value_counts()
    n = np.array([cnt.get(c, 0) for c in classes], dtype=float)
    if (n == 0).any():
        raise SystemExit(f"[X] 학습셋에 없는 클래스가 있습니다: "
                         f"{[c for c, v in zip(classes, n) if v == 0]}")
    return n / n.sum()


def sweep(logits: np.ndarray, y: np.ndarray, prior: np.ndarray,
          classes: list[str], taus=TAU_GRID) -> list[dict]:
    from sklearn.metrics import f1_score, precision_score, recall_score

    logp = np.log(prior)
    rows = []
    for t in taus:
        pred = (logits - t * logp).argmax(1)
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        np.add.at(cm, (y, pred), 1)
        disp = {d["cls"]: d for d in errors.class_dispersion(
            {"confusion": cm.tolist(), "classes": classes}, show=False)}
        rows.append({
            "tau": t,
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "recall": recall_score(y, pred, average=None, zero_division=0,
                                   labels=range(len(classes))),
            "precision": precision_score(y, pred, average=None, zero_division=0,
                                         labels=range(len(classes))),
            "lift": {c: disp[c]["lift"] for c in classes},
            "pred": pred,
        })
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="data/work/reports/step18_local")
    ap.add_argument("--manifest", default="manifest_final.parquet")
    ap.add_argument("--focus", default="A4", help="주목할 클래스")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    d = ROOT / a.dir
    z = np.load(d / "stage2_logits.npz", allow_pickle=False)
    logits, y = z["logits"], z["y"]
    classes = [str(c) for c in z["classes"]]
    print(f"로짓 {logits.shape} · 클래스 {classes}")
    print("평가셋 클래스별 장수:",
          {c: int((y == i).sum()) for i, c in enumerate(classes)})

    prior = train_prior(ROOT / a.manifest, classes)
    print("\n학습셋 prior (모델이 본 분포):")
    for c, p in zip(classes, prior):
        print(f"  {c}  {p:6.2%}")

    rows = sweep(logits, y, prior, classes)
    fi = classes.index(a.focus)

    print(f"\n=== τ 를 올리며 (τ=0 이 지금 모델) ===")
    print(f"{'τ':>5}{'macro-F1':>10}" + "".join(f"{c:>8}" for c in classes)
          + f"{a.focus + ' lift':>10}")
    for r in rows:
        lf = r["lift"][a.focus]
        lfs = "—" if lf != lf else f"{lf:.2f}"
        print(f"{r['tau']:>5.1f}{r['macro_f1']:>10.4f}"
              + "".join(f"{v:>8.3f}" for v in r["recall"]) + f"{lfs:>10}")
    print("  (가운데 여섯 칸은 클래스별 recall)")

    base, best = rows[0], max(rows, key=lambda r: r["macro_f1"])
    print(f"\n=== 판정 (기준 τ=0) ===")
    dF = best["macro_f1"] - base["macro_f1"]
    print(f"  macro-F1  {base['macro_f1']:.4f} → {best['macro_f1']:.4f} "
          f"(τ={best['tau']:.1f}, Δ{dF:+.4f})   문턱 +{ADOPT_MACRO_F1_GAIN}")

    r0, _, _ = bootstrap_ci(y, base["pred"], metric="recall", cls=fi, n=a.boot)
    r1, lo1, hi1 = bootstrap_ci(y, best["pred"], metric="recall", cls=fi, n=a.boot)
    half = (hi1 - lo1) / 2
    print(f"  {a.focus} recall {r0:.3f} → {r1:.3f} (Δ{r1 - r0:+.3f})  "
          f"이 실행의 CI 반폭 ±{half:.3f}")
    print(f"  {a.focus} precision {base['precision'][fi]:.3f} → "
          f"{best['precision'][fi]:.3f}")
    l0, l1 = base["lift"][a.focus], best["lift"][a.focus]
    print(f"  {a.focus} 쏠림 lift {l0:.2f} → {l1:.2f}   문턱 ≤{PRIOR_EXPLAINED_LIFT}")

    ok_f1 = dF >= ADOPT_MACRO_F1_GAIN
    ok_rec = (r1 - r0) > half
    ok_lift = (l1 == l1) and l1 <= PRIOR_EXPLAINED_LIFT
    for name, ok in [("macro-F1 이 잡음 밖에서 상승", ok_f1),
                     (f"{a.focus} recall 개선 > CI 반폭", ok_rec),
                     ("쏠림이 우연 수준으로 내려옴", ok_lift)]:
        print(f"  {'[O]' if ok else '[X]'} {name}")

    if ok_f1 and ok_rec:
        v = "채택 후보 — prior 가 상당 부분입니다"
    elif ok_rec and not ok_f1:
        v = "부분적 — recall 은 오르나 macro-F1 이 안 따라옵니다 (precision 을 잃음)"
    else:
        v = "기각 — prior 가 아닙니다. 표현 쪽(크롭·데이터)으로 갑니다"
    print(f"\n  → {v}")
    print("\n⚠️ VL01 부분집합에서 잰 값입니다. 방향을 보는 용도이고, 채택은 전체"
          " val 에서 다시 확인해야 합니다.")

    if a.out:
        p = ROOT / a.out
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "classes": classes, "prior": prior.tolist(), "focus": a.focus,
            "verdict": v, "best_tau": best["tau"],
            "macro_f1": {"base": base["macro_f1"], "best": best["macro_f1"]},
            "focus_recall": {"base": r0, "best": r1, "ci_half": half},
            "focus_lift": {"base": l0, "best": l1},
            "rows": [{k: (v2.tolist() if isinstance(v2, np.ndarray) else v2)
                      for k, v2 in r.items() if k != "pred"} for r in rows],
        }, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
        print(f"\n저장: {p}")


if __name__ == "__main__":
    main()
