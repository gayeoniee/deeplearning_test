"""A5(미란·궤양) 하향을 어떻게 줄일 것인가 — 저장된 확률만으로, 추론 0회.

    uv run python tools/a5_downgrade.py
    uv run python tools/a5_downgrade.py --arrays data/work/reports/step33_arrays_holdout.npz

무엇을 재나 — STEP 35 의 네 판을 그대로 재현합니다:

  판 A  같은 커버리지에서 재보기
        "앙상블이 A5 를 망가뜨렸나?" → 아니오. 말을 더 해서 그런 것입니다.
        ⚠️ STEP 34 가 A6 경보를 푼 방법과 **같습니다** — 비율만 보고 모델을
           탓하면 두 번째로 같은 실수를 합니다.

  판 B  팔별로 따로 재보기 ("한 팔이 범인인가?")

  판 C  ★ 하향 방지 규칙 — 급한 쪽 확률 합이 문턱을 넘으면 덜 급한 묶음은
        답 후보에서 뺍니다. **채택 후보.**

  판 D  비용 행렬로 규칙을 유도 (argmax 대신 기대비용 최소)

★ 판정 기준은 `src/experiments.py` 에 상수로 있습니다 (작업 규칙 2).
⚠️ **문턱은 val 에서 고르고 holdout 은 확인만** 합니다 — 두 배열을 다 주면
   자동으로 그렇게 합니다.

배열 파일이 담고 있어야 하는 것 (노트북 11·12 가 저장합니다):
    p1        (n,)     1단계 이상 확률
    p2_0..2   (n, 6)   2단계 팔별 확률 (이름 순)
    truth     (n,)     실제 라벨 — 헛알림은 'A7'
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import MORPH_GROUP_KEEP_A6  # noqa: E402
from src.experiments import (  # noqa: E402
    A5_RULE_COVERAGE_LOSS_MAX,
    A5_RULE_UNDER_GAIN_MIN,
    ARM_SAME_COVERAGE_TOL,
    NAMING_TARGET_ERROR,
)

G4 = ["융기·발진", "표면 변화", "미란·궤양", "결절·종괴"]
#: 묶음의 긴급도. 묶음 안에서 **높은 쪽**을 씁니다 (보수적으로).
TIER = {"융기·발진": 1, "표면 변화": 1, "미란·궤양": 2, "결절·종괴": 2}
LOW = [g for g in G4 if TIER[g] == 1]
HIGH = [g for g in G4 if TIER[g] == 2]
C6 = ["A1", "A2", "A3", "A4", "A5", "A6"]

DEFAULT_VAL = ROOT / "data/work/reports/step31_arrays_full.npz"
DEFAULT_HO = ROOT / "data/work/reports/step33_arrays_holdout.npz"


# ---------------------------------------------------------------- 불러오기
def load(path: Path):
    d = np.load(path, allow_pickle=True)
    need = {"p1", "p2_0", "p2_1", "p2_2", "truth"}
    if not need <= set(d.files):
        raise KeyError(f"{path.name} 에 {sorted(need - set(d.files))} 가 없습니다")
    truth = np.asarray([str(t) for t in d["truth"]])
    arms = [d["p2_0"], d["p2_1"], d["p2_2"]]
    names = [str(x) for x in d["arm_names"]] if "arm_names" in d.files else \
        ["팔0", "팔1", "팔2"]
    true_g = np.asarray(
        ["정상" if t == "A7" else MORPH_GROUP_KEEP_A6[t] for t in truth])
    return d["p1"], arms, true_g, names


def fold(p2: np.ndarray) -> np.ndarray:
    """6종 확률을 4묶음으로 **더해서** 접습니다 (argmax 를 묶는 게 아니라)."""
    P = np.zeros((len(p2), 4))
    for j, c in enumerate(C6):
        P[:, G4.index(MORPH_GROUP_KEEP_A6[c])] += p2[:, j]
    return P


# ---------------------------------------------------------------- 재기
def speak(p1, P, true_g, *, block=None, k=None):
    """확신도 내림차순으로 누적 오답률이 목표를 넘지 않는 지점까지만 말합니다.

    block 을 주면 **급한 쪽 확률 합이 그 값 이상일 때 덜 급한 묶음을 후보에서
    뺍니다** (판 C). k 를 주면 커버리지를 그 장수로 **고정**합니다 (판 A).
    """
    Q = P if block is None else np.where(
        ((P[:, 2] + P[:, 3]) >= block)[:, None] & np.isin(G4, LOW)[None, :], -1.0, P)
    said = np.asarray([G4[i] for i in Q.argmax(1)])
    conf = p1 * np.clip(Q.max(1), 0, None)
    order = np.argsort(-conf)
    if k is None:
        err = np.cumsum((true_g != said)[order]) / np.arange(1, len(conf) + 1)
        ok = np.flatnonzero(err <= NAMING_TARGET_ERROR)
        k = int(ok[-1]) + 1 if len(ok) else 0
    sp = np.zeros(len(conf), bool)
    sp[order[:k]] = True
    return said, sp, k


def a5_under(said, sp, true_g):
    """말한 A5 중 **덜 급하다고 부른** 비율."""
    m = (true_g == "미란·궤양") & sp
    return (int(m.sum()),
            float((m & np.isin(said, LOW)).sum() / m.sum()) if m.sum() else 0.0)


def total_under(said, sp, true_g):
    m = sp & (true_g != "정상")
    if not m.any():
        return 0.0
    return float((m & np.isin(true_g, HIGH) & np.isin(said, LOW)).sum() / m.sum())


def line(tag, p1, P, true_g, **kw):
    said, sp, k = speak(p1, P, true_g, **kw)
    n5, a5 = a5_under(said, sp, true_g)
    return dict(tag=tag, cov=float(sp.mean()), n5=n5, a5=a5,
                tot=total_under(said, sp, true_g), k=k)


def show(rows, title):
    print(f"\n{title}")
    print(f"  {'':30}{'커버리지':>10}{'A5 말함':>9}{'A5 하향':>10}{'전체 하향':>11}")
    for r in rows:
        print(f"  {r['tag']:30}{r['cov']:>10.1%}{r['n5']:>9,}"
              f"{r['a5']:>10.1%}{r['tot']:>11.1%}")


# ---------------------------------------------------------------- 판 D
def cost_pick(p1, P, c_under, *, c_name=1.0, c_over=1.0):
    """기대비용이 가장 낮은 묶음. argmin Σ L(g,g') p(g'|x).

    ⚠️ '정상인데 이름을 말함' 비용은 네 후보에 **똑같이** 붙어 argmin 에서
       상쇄됩니다 — 정상 사진이면 어느 이름이든 똑같이 틀리니 당연합니다.
       넣어도 아무 일도 안 하므로 넣지 않습니다.
    """
    L = np.zeros((4, 4))
    for i, g in enumerate(G4):
        for j, gp in enumerate(G4):
            if g == gp:
                continue
            d = TIER[gp] - TIER[g]
            L[i, j] = c_under * d if d > 0 else (c_over * -d if d < 0 else c_name)
    risk = P @ L.T
    return np.asarray([G4[i] for i in risk.argmin(1)])


def cost_line(tag, p1, P, true_g, c_under):
    said = cost_pick(p1, P, c_under)
    conf = p1 * P.max(1)
    order = np.argsort(-conf)
    err = np.cumsum((true_g != said)[order]) / np.arange(1, len(conf) + 1)
    ok = np.flatnonzero(err <= NAMING_TARGET_ERROR)
    k = int(ok[-1]) + 1 if len(ok) else 0
    sp = np.zeros(len(conf), bool)
    sp[order[:k]] = True
    n5, a5 = a5_under(said, sp, true_g)
    return dict(tag=tag, cov=float(sp.mean()), n5=n5, a5=a5,
                tot=total_under(said, sp, true_g), k=k)


# ---------------------------------------------------------------- 본체
def run(path: Path, *, pick_from: float | None):
    p1, arms, true_g, names = load(path)
    REL, ENS = fold(arms[0]), fold(np.mean(arms, axis=0))
    base = line("3팔 앙상블 (지금)", p1, ENS, true_g)
    rel = line("릴리스 단독", p1, REL, true_g)

    print("=" * 74)
    print(f"■ {path.name}   n={len(p1):,}   팔: {', '.join(names)}")
    show([rel, base], "기준")

    # ---- 판 A --------------------------------------------------------
    same = line(f"앙상블 @{rel['k']:,}장", p1, ENS, true_g, k=rel["k"])
    show([rel, same, base], "판 A · 같은 장수로 맞춰서 (선택 효과인가?)")
    gap = abs(same["a5"] - rel["a5"])
    print(f"  → 차이 {gap:+.1%}p — "
          + ("**선택 효과입니다. 모델이 나빠진 게 아닙니다.**"
             if gap <= ARM_SAME_COVERAGE_TOL else "★ 선택 효과로 설명 안 됩니다"))

    # ---- 판 B --------------------------------------------------------
    show([line(nm, p1, fold(a), true_g) for nm, a in zip(names, arms)],
         "판 B · 팔별 (한 팔이 범인인가?)")
    vals = [line(nm, p1, fold(a), true_g)["a5"] for nm, a in zip(names, arms)]
    print(f"  → 폭 {max(vals) - min(vals):.1%}p "
          f"(10%p 미만이면 '한 팔 탓' 기각)")

    # ---- 판 C --------------------------------------------------------
    rows, passing = [], []
    for t in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
        r = line(f"문턱 {t:.2f}", p1, ENS, true_g, block=t)
        gain, loss = base["a5"] - r["a5"], base["cov"] - r["cov"]
        r["ok"] = gain >= A5_RULE_UNDER_GAIN_MIN and loss <= A5_RULE_COVERAGE_LOSS_MAX
        r["t"] = t
        rows.append(r)
        if r["ok"]:
            passing.append(r)
    show([base] + rows, "판 C · ★ 하향 방지 규칙 (급한 쪽 합이 문턱 이상이면 "
                        "덜 급한 묶음을 후보에서 뺌)")
    for r in rows:
        if r["ok"]:
            print(f"  통과: 문턱 {r['t']:.2f}  커버리지 {r['cov'] - base['cov']:+.1%}p  "
                  f"A5 하향 {r['a5'] - base['a5']:+.1%}p")
    if not passing:
        print("  → 통과한 문턱이 없습니다")

    # ---- 판 D --------------------------------------------------------
    show([base] + [cost_line(f"비용 C_under={c}", p1, ENS, true_g, c)
                   for c in (2, 3, 5, 8, 12)],
         "판 D · 비용 행렬로 규칙 유도 (문턱 규칙과 같은 곡선인가?)")
    print("  ⚠️ 커버리지가 다르면 비교가 안 됩니다 — 같은 구간끼리 보세요")

    if pick_from is not None:
        r = line(f"문턱 {pick_from:.2f} (val 에서 고름)", p1, ENS, true_g,
                 block=pick_from)
        print(f"\n★ val 에서 고른 문턱 {pick_from:.2f} 을 여기 적용:")
        print(f"  커버리지 {r['cov']:.1%} ({r['cov'] - base['cov']:+.1%}p) · "
              f"A5 하향 {r['a5']:.1%} ({r['a5'] - base['a5']:+.1%}p) · "
              f"전체 하향 {r['tot']:.1%}")
    return passing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays", type=Path, default=None,
                    help="하나만 볼 때. 안 주면 val 에서 고르고 holdout 으로 확인합니다")
    ap.add_argument("--val", type=Path, default=DEFAULT_VAL)
    ap.add_argument("--holdout", type=Path, default=DEFAULT_HO)
    a = ap.parse_args()

    if a.arrays:
        run(a.arrays, pick_from=None)
        return

    if not a.val.exists():
        raise SystemExit(f"[X] {a.val} 가 없습니다. --arrays 로 하나만 보세요.")
    passing = run(a.val, pick_from=None)
    if not passing:
        print("\n[X] val 에서 통과한 문턱이 없어 holdout 을 안 봅니다.")
        return
    # 통과한 것 중 **커버리지를 제일 적게 잃는** 쪽
    best = min(passing, key=lambda r: -r["cov"])
    print(f"\n{'=' * 74}\n★ val 에서 고른 문턱: {best['t']:.2f}")
    if a.holdout.exists():
        run(a.holdout, pick_from=best["t"])
    else:
        print(f"[!] {a.holdout} 가 없어 확인을 건너뜁니다")


if __name__ == "__main__":
    main()
