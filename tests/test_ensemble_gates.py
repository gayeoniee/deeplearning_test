"""2단계 앙상블 채택 관문 (STEP 25) — 결과를 보고 기준을 못 바꾸게 못 박습니다.

    uv run --extra train python tests/test_ensemble_gates.py

왜 이 검사가 필요한가
--------------------
STEP 23 에서 제 사전등록 기준에 **구멍**이 있었습니다. "짝 혼동(A4→A1)이 줄
것" 은 통과했는데, 줄어든 만큼 정답이 아니라 **나머지 클래스로 흩어졌습니다**
(그 밖 36.8% → 42.1%). 행선지 하나만 보면 악화를 개선으로 읽습니다.

그래서 `stage2_ensemble_report` 에 네 번째 관문 `ENS_NO_SCATTER` 를 넣었고,
**이 검사가 그 관문이 실제로 작동하는지** 확인합니다 — 즉 "macro-F1 은
올랐는데 흩어진" 가짜 후보를 코드가 걸러내는가.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import experiments as E  # noqa: E402

ok = fail = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {name}")
    else:
        fail += 1
        print(f"  FAIL  {name}  {detail}")


BASE = {"macro_f1": 0.5805, "scale_drop": 0.26,
        "recall": {"A1": .555, "A2": .752, "A3": .788, "A4": .356, "A5": .557, "A6": .571},
        "focus_to_other": 0.209, "focus_to_rest": 0.436}


def cand(**kw):
    d = {"macro_f1": 0.6384, "d_macro_f1_ci": [0.0397, 0.0751], "scale_drop": 0.28,
         "recall": {"A1": .573, "A2": .775, "A3": .802, "A4": .540, "A5": .600, "A6": .587},
         "focus_to_other": 0.138, "focus_to_rest": 0.322}
    d.update(kw)
    return d


print("\n[1] 상수가 코드에 박혀 있다 (작업 규칙 2)")
check("ENS_MIN_GAIN = MACRO_F1_NOISE", E.ENS_MIN_GAIN == E.MACRO_F1_NOISE == 0.02)
check("ENS_SCALE_TOL_PP = SCALE_DROP_REJECT_PP",
      E.ENS_SCALE_TOL_PP == E.SCALE_DROP_REJECT_PP == 0.12)
check("ENS_NO_CLASS_LOSS = 0.03", E.ENS_NO_CLASS_LOSS == 0.03)
check("ENS_NO_SCATTER = 0.0", E.ENS_NO_SCATTER == 0.0)

print("\n[2] 네 관문을 다 통과하면 후보")
r = E.stage2_ensemble_report(BASE, cand())
check("판정이 '채택 후보'", r["verdict"] == "채택 후보", r["verdict"])
check("관문 4개를 다 보고한다", len(r["gates"]) == 4, str(len(r["gates"])))

print("\n[3] ★ STEP 23 의 구멍 — macro-F1 은 올랐는데 **흩어진** 경우")
scattered = cand(focus_to_other=0.10,      # 짝 혼동은 줄었고
                 focus_to_rest=0.50)       # 나머지로 흩어졌습니다
r = E.stage2_ensemble_report(BASE, scattered)
check("흩어지면 macro-F1 이 올라도 **기각**", r["verdict"] == "기각", r["verdict"])
check("걸린 관문이 '4. 흩어짐'", not r["gates"]["4. 흩어짐"]["pass"])
check("짝 혼동만 보면 통과했을 것 (구멍이 실재했음을 확인)",
      scattered["focus_to_other"] < BASE["focus_to_other"])

print("\n[4] 나머지 관문도 실제로 막는가")
r = E.stage2_ensemble_report(BASE, cand(macro_f1=0.5905, d_macro_f1_ci=[0.001, 0.02]))
check("이득이 잡음(0.02) 미만이면 기각", r["verdict"] == "기각")

r = E.stage2_ensemble_report(BASE, cand(d_macro_f1_ci=[-0.01, 0.12]))
check("차이 CI 가 0 을 넘으면 기각 (평균이 커도)", r["verdict"] == "기각")

r = E.stage2_ensemble_report(BASE, cand(scale_drop=0.26 + 0.13))
check("배율 하락이 12%p 넘게 나빠지면 기각", r["verdict"] == "기각")

r = E.stage2_ensemble_report(BASE, cand(
    recall={"A1": .573, "A2": .775, "A3": .788 - 0.04, "A4": .540, "A5": .600, "A6": .587}))
check("어느 한 클래스라도 recall 이 0.03 넘게 떨어지면 기각", r["verdict"] == "기각")

print("\n[5] 경계값 — 딱 문턱이면 통과 (부등호 방향)")
r = E.stage2_ensemble_report(BASE, cand(scale_drop=0.26 + 0.12))
check("배율 악화가 정확히 12%p 면 통과", r["gates"]["2. 배율 하락"]["pass"])
r = E.stage2_ensemble_report(BASE, cand(focus_to_rest=BASE["focus_to_rest"]))
check("흩어짐이 그대로면 통과 (늘어야 기각)", r["gates"]["4. 흩어짐"]["pass"])

print("\n" + "=" * 60)
print(f" 통과 {ok} / {ok + fail}")
sys.exit(1 if fail else 0)
