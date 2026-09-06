"""두 무리의 차이를 재는 값들 — **여기가 유일한 정의입니다.**

왜 별도 모듈인가
----------------
같은 세 값(`d` · `AUROC` · 판정 문턱)을 두 곳에서 씁니다:

* `tools/false_alarm_stats.py` — 헛알림 난 사진이 무엇과 다른가
* `tools/class_overlap.py`     — A4 와 A1 이 데이터에서 무엇이 다른가

따로 적으면 **갈라지고, 갈라져도 아무도 모릅니다.** 이 리포는 크롭 창에서
이미 같은 실패를 했습니다 (`crop.crop_window`). 그래서 정의를 하나만 둡니다.

읽는 법
-------
* `d`      = 퍼짐으로 나눈 평균 차이 (Cohen's d). |d| < 0.2 는 사실상 차이 없음
* `AUROC`  = 그 값 **하나만** 보고 한쪽을 골라낼 수 있나.
  0.5 = 못 함. **0.5 미만이면 방향이 반대**이고 세기는 |AUROC − 0.5| 입니다

⚠️ **`d` 하나만 보면 속습니다.** 퍼짐이 아주 작으면 실제 차이가 없다시피 해도
   `d` 가 커집니다 (`bright` 가 0.5001 vs 0.4981 인데 d=0.69 였습니다).
   그래서 판정은 **d 와 AUROC 를 둘 다** 넘어야 합니다.

이 문턱은 **결과를 보기 전에** 못 박은 값입니다 (CLAUDE.md 작업 규칙 2).
"""

from __future__ import annotations

# 판정 문턱 — 둘 **다** 넘어야 "차이 있음"
D_STRONG, A_STRONG = 0.5, 0.15
D_WEAK, A_WEAK = 0.2, 0.06


def cohens_d(x, y) -> float:
    import numpy as np

    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    s = np.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1))
                / (len(x) + len(y) - 2))
    return float((x.mean() - y.mean()) / s) if s else float("nan")


def auroc(pos, neg) -> float:
    """값 하나만으로 pos 를 neg 와 가를 수 있나 (Mann-Whitney U)."""
    import numpy as np

    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = allv.argsort().argsort().astype(float) + 1
    # 동점 처리 — 각질처럼 값이 뭉치는 경우가 있어서 필요합니다
    order = np.sort(allv)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and order[j + 1] == order[i]:
            j += 1
        if j > i:
            r[(allv >= order[i]) & (allv <= order[j])] = (i + j) / 2 + 1
        i = j + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def verdict(d: float, a: float) -> str:
    import math

    if math.isnan(d) or math.isnan(a):
        return "못 잼"
    da = abs(a - 0.5)
    if abs(d) >= D_STRONG and da >= A_STRONG:
        return "**차이 있음**"
    if abs(d) >= D_WEAK and da >= A_WEAK:
        return "조금"
    return "차이 없음"
