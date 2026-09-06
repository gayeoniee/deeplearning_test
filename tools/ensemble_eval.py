"""여러 크롭 뷰의 **저장된 로짓을 평균**해 앙상블을 평가합니다 (GPU 0, 재학습 0).

    # 로컬 VL01
    uv run --extra train python tools/ensemble_eval.py \
        --logits m2.5=data/work/checkpoints/stage2_convnextv2_base_m2.5_384_n121k_moderate/logits_chunk_VL01_val.npz \
        --logits f320=data/work/checkpoints/stage2_effnetv2_s_f320_384_moderate/logits_val.npz

    # 캐글 STEP 23 산출물을 받아온 뒤 (전체 데이터 확인)
    uv run --extra train python tools/ensemble_eval.py \
        --logits m2.5=~/Downloads/step23/m2.5_logits_val.npz \
        --logits f320=~/Downloads/step23/f320_logits_val.npz \
        --out data/work/reports/step25_ensemble_full.json

왜 이게 되나
------------
STEP 22·23 에서 `m2.5`(비례 창)와 `f320`(고정 창)이 **각각 단독으로는** 상대를
못 이겼습니다. 그런데 둘이 **다른 실수**를 한다면 확률을 평균했을 때 둘 다를
이깁니다 — 실제로 VL01 에서 서로 보완하는 몫이 27.0% 였습니다.

⚠️ **판정 기준은 `experiments.stage2_ensemble_report()` 에 미리 박혀 있습니다**
   (관문 4개). 특히 네 번째 `ENS_NO_SCATTER` 는 STEP 23 에서 제 기준에 뚫려
   있던 구멍을 막은 것입니다 — "짝 혼동이 줄었다" 는 **흩어짐**으로도 만들어집니다.

⚠️ 로짓 파일들은 **같은 val 행을 같은 순서로** 담고 있어야 합니다. 이 도구는
   `y` 배열을 대조해 다르면 **멈춥니다** (조용히 엉뚱한 걸 더하면 아무도 모릅니다).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import experiments  # noqa: E402
from src.config import CLASSES  # noqa: E402


def _softmax(a: np.ndarray) -> np.ndarray:
    e = np.exp(a - a.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def _stats(p: np.ndarray, y: np.ndarray, fi: int, oi: int) -> dict:
    from sklearn.metrics import f1_score, precision_recall_fscore_support

    pred = p.argmax(1)
    m = y == fi
    rec = precision_recall_fscore_support(y, pred, labels=range(len(CLASSES)),
                                          zero_division=0)[1]
    return {"macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "accuracy": float((pred == y).mean()),
            "recall": {c: float(rec[i]) for i, c in enumerate(CLASSES)},
            "focus_recall": float((pred[m] == fi).mean()),
            "focus_to_other": float((pred[m] == oi).mean()),
            "focus_to_rest": float(1 - (pred[m] == fi).mean() - (pred[m] == oi).mean()),
            "n_focus": int(m.sum())}


def _paired_ci(pb, pc, y, n_boot=2000, seed=0):
    """차이의 95% CI — **같은 행을 같이 뽑습니다**(paired). 두 팔을 따로 뽑으면
    상관을 무시해 CI 가 과하게 넓어집니다."""
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(seed)
    a, b = pb.argmax(1), pc.argmax(1)
    d = [f1_score(y[i], b[i], average="macro", zero_division=0)
         - f1_score(y[i], a[i], average="macro", zero_division=0)
         for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))]
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logits", action="append", required=True,
                    metavar="이름=경로", help="여러 번 주세요. **첫 번째가 기준선**입니다")
    ap.add_argument("--focus", default="A4")
    ap.add_argument("--other", default="A1")
    ap.add_argument("--scale-drop", nargs=2, type=float, default=None,
                    metavar=("기준", "앙상블"),
                    help="배율 하락 (관문 2). 없으면 그 관문은 '못 잼' 으로 남습니다")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    arms, y = {}, None
    for spec in a.logits:
        if "=" not in spec:
            raise SystemExit(f"[X] --logits 는 '이름=경로' 형식입니다 — {spec!r}")
        name, path = spec.split("=", 1)
        z = np.load(Path(path).expanduser(), allow_pickle=False)
        if y is None:
            y = z["y"]
        elif not np.array_equal(y, z["y"]):
            raise SystemExit(
                f"[X] '{name}' 의 라벨 배열이 첫 번째와 다릅니다 "
                f"({len(z['y']):,}행 vs {len(y):,}행).\n"
                f"    같은 val 행을 같은 순서로 담은 로짓만 더할 수 있습니다.\n"
                f"    (전체 데이터 판과 VL01 판을 섞으려 한 것 아닌지 확인하세요.)")
        arms[name] = _softmax(z["logits"])
        print(f"  [{name}] {path}  {z['logits'].shape}")

    fi, oi = CLASSES.index(a.focus), CLASSES.index(a.other)
    print(f"\nval {len(y):,}행 · {a.focus} {int((y==fi).sum()):,}장\n")

    names = list(arms)
    base_name = names[0]
    print(f"{'':16}{'macro-F1':>10}{'accuracy':>10}{f'{a.focus} recall':>12}"
          f"{f'→{a.other}':>9}{'→그밖':>9}")
    single = {}
    for n in names:
        s = _stats(arms[n], y, fi, oi)
        single[n] = s
        print(f"{n:16}{s['macro_f1']:>10.4f}{s['accuracy']:>10.4f}"
              f"{s['focus_recall']:>12.3f}{s['focus_to_other']:>9.1%}"
              f"{s['focus_to_rest']:>9.1%}")

    ens_p = np.mean([arms[n] for n in names], 0)
    ens = _stats(ens_p, y, fi, oi)
    print(f"{'앙상블(' + str(len(names)) + '팔)':16}{ens['macro_f1']:>10.4f}"
          f"{ens['accuracy']:>10.4f}{ens['focus_recall']:>12.3f}"
          f"{ens['focus_to_other']:>9.1%}{ens['focus_to_rest']:>9.1%}")

    # 겹침 — 앙상블이 먹을 재료가 있나
    if len(names) == 2:
        pa, pb_ = arms[names[0]].argmax(1), arms[names[1]].argmax(1)
        only = ((pa == y) & (pb_ != y)).mean() + ((pa != y) & (pb_ == y)).mean()
        print(f"\n[겹침] 한쪽만 맞히는 몫 {only:.1%} · 둘 다 틀림 "
              f"{((pa != y) & (pb_ != y)).mean():.1%}")
        print("       한쪽만 맞히는 몫이 작으면 앙상블이 먹을 게 없습니다.")

    base = dict(single[base_name])
    cand = dict(ens)
    cand["d_macro_f1_ci"] = _paired_ci(arms[base_name], ens_p, y)
    if a.scale_drop:
        base["scale_drop"], cand["scale_drop"] = a.scale_drop
    else:
        base["scale_drop"] = cand["scale_drop"] = None
        print("\n⚠️ `--scale-drop` 을 안 줘서 관문 2(배율)를 **못 쟀습니다.**"
              "   판정은 채택 후보가 될 수 없습니다.")

    verdict = experiments.stage2_ensemble_report(base, cand, focus=a.focus, other=a.other)
    out = {"step": "STEP 25 — 2단계 크롭 뷰 앙상블", "n_val": int(len(y)),
           "arms": names, "baseline": base_name, "single": single, "ensemble": ens,
           "scale_drop_measured": bool(a.scale_drop), "verdict": verdict}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float),
                               encoding="utf-8")
        print(f"\n저장: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
