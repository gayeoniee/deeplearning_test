"""STEP 28 — 알갱이 판정 기준을 못 박습니다 (작업 규칙 2).

제일 중요한 감시: **굵게 말하면 커버리지는 공짜로 오릅니다.** 극단적으로
"이상이 있습니다" 한 가지만 말하면 커버리지 100% 입니다. 그래서 커버리지
문턱만 두면 무슨 묶음이든 통과합니다 — `UNDER_TRIAGE_MAX` 가 그걸 막습니다.
"""

import inspect
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import experiments as E
from src.config import CLASSES, MORPH_GROUP, URGENCY_TIER

ok = fail = 0


def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {name}")
    else:
        fail += 1
        print(f"  FAIL  {name}" + (f"\n        {detail}" if detail else ""))


print("[1] 묶음이 임상 문서와 맞는가 (데이터로 만든 게 아님)")
check("여섯 클래스가 다 등급을 가진다", set(URGENCY_TIER) == set(CLASSES))
check("여섯 클래스가 다 형태 묶음을 가진다", set(MORPH_GROUP) == set(CLASSES))
check("등급은 0·1·2 세 개", sorted(set(URGENCY_TIER.values())) == [0, 1, 2])
check("A6·A5 가 가장 급하다 (조기 진료)",
      URGENCY_TIER["A6"] == URGENCY_TIER["A5"] == 2)
check("A1·A2 가 관찰", URGENCY_TIER["A1"] == URGENCY_TIER["A2"] == 0)
# 임상 해설의 '🟢 비교적 안전한 혼동' = A1↔A4, A5↔A6
check("A1·A4 가 같은 형태 묶음 (안전한 혼동)",
      MORPH_GROUP["A1"] == MORPH_GROUP["A4"])
check("A5·A6 가 같은 형태 묶음 (안전한 혼동)",
      MORPH_GROUP["A5"] == MORPH_GROUP["A6"])
check("형태 묶음은 셋", len(set(MORPH_GROUP.values())) == 3)
# 🔴 위험한 혼동 A6→A2 는 등급이 2 → 0 이라 '하향' 으로 잡혀야 합니다
check("A6→A2 는 긴급도 하향으로 잡힌다", URGENCY_TIER["A6"] > URGENCY_TIER["A2"])
check("A5→A1 은 긴급도 하향으로 잡힌다", URGENCY_TIER["A5"] > URGENCY_TIER["A1"])

print("\n[2] ★ 굵게 말해 커버리지만 올리면 통과 못 한다")
n = 1000
rng = np.random.default_rng(0)
# 전부 '관찰' 이라고 말하는 모델. 틀린 적이 없다고 우기려고 wrong=False 로 둡니다.
tier_true = rng.integers(0, 3, n)
lazy = E.granularity_report("전부 관찰이라고 말함", {
    "conf": np.linspace(0, 1, n), "wrong": np.zeros(n, bool),
    "tier_true": tier_true, "tier_said": np.zeros(n, int)})
check("커버리지는 100% 로 나온다", lazy["coverage"] == 1.0)
check("그래도 기각된다 (긴급도 하향)", lazy["verdict"].startswith("기각"))
check("하향 비율이 문턱을 넘는다", lazy["under_triage"] > E.UNDER_TRIAGE_MAX)

print("\n[3] 반대 방향(과하게 급하다고 말함)은 통과한다 — 다만 **값이 찍혀야** 한다")
eager = E.granularity_report("전부 조기진료라고 말함", {
    "conf": np.linspace(0, 1, n), "wrong": np.zeros(n, bool),
    "tier_true": tier_true, "tier_said": np.full(n, 2)})
check("하향이 0", eager["under_triage"] == 0.0)
check("통과 (과잉은 위험하지 않다)", eager["verdict"] == "논의할 가치 있음")
# ★ 과잉을 안 찍으면 "하향 0%" 만 보고 안심하게 됩니다. 실제로 권고안(4군)의
#   과잉이 49.3% 인 걸 이 값을 안 찍던 동안 몰랐습니다.
check("과잉도 함께 보고된다", "over_triage" in eager)
check("전부 급하다고 하면 과잉이 크다", eager["over_triage"] > 0.5)
check("하향과 과잉은 다른 값", eager["over_triage"] != eager["under_triage"])

print("\n[4] 정상 사진(-1)은 하향 계산에서 빠진다")
r = E.granularity_report("정상만", {
    "conf": np.linspace(0, 1, n), "wrong": np.zeros(n, bool),
    "tier_true": np.full(n, -1), "tier_said": np.zeros(n, int)})
check("하향 0 (낮출 긴급도가 없다)", r["under_triage"] == 0.0)

print("\n[5] ★ 하향 문턱에 **닻이 있는가** (STEP 34)")
# 처음엔 근거 없이 박은 말뚝이었습니다. 이제 기준은 "1단계가 이미 받아들이는
# 하향보다 작아야 한다" 입니다 — 그 관계가 깨지면 여기서 실패합니다.
check("1단계 놓침 비율이 상수로 있다", hasattr(E, "STAGE1_MISS_RATE"))
check("2단계 하향 문턱 < 1단계 놓침 (더 작아야 한다)",
      E.UNDER_TRIAGE_MAX < E.STAGE1_MISS_RATE,
      f"{E.UNDER_TRIAGE_MAX} vs {E.STAGE1_MISS_RATE}")
check("1단계 놓침이 실측값과 맞다 (holdout recall 0.9430)",
      abs(E.STAGE1_MISS_RATE - (1 - 0.9430)) < 0.002,
      f"{E.STAGE1_MISS_RATE} vs {1 - 0.9430:.4f}")

print("\n[6] 커버리지 문턱은 이름 판정과 같은 값")
check("50% 로 같다", E.GRANULARITY_MIN_COVERAGE == E.NAMING_MIN_COVERAGE)
check("커버리지가 낮으면 기각",
      E.granularity_report("못 맞힘", {
          "conf": np.linspace(0, 1, n), "wrong": np.ones(n, bool),
          "tier_true": np.full(n, -1),
          "tier_said": np.zeros(n, int)})["verdict"].startswith("기각"))

print("\n[6] 길이가 안 맞으면 멈춘다")
try:
    E.granularity_report("x", {"conf": [1, 2], "wrong": [True],
                               "tier_true": [0, 0], "tier_said": [0, 0]})
    check("길이 불일치를 잡는다", False)
except ValueError:
    check("길이 불일치를 잡는다", True)

print("\n[7] ★ 클래스별 하향이 사라지지 않는가 (STEP 35)")
# 전체 평균은 분모가 커서 **한 클래스가 나빠도 묻힙니다.**
# 실측(holdout, 계열 4군 3팔): 전체 3.7% 로 관문을 통과하는데
# 미란·궤양 하나만 보면 **43.8%** 였습니다. 관문이 이걸 못 봤습니다.
# → over_triage 와 같은 처방입니다: **관문으로 안 쓰되 값을 찍습니다.**
_rng = np.random.default_rng(0)
_n = 1000
_wrong = _rng.random(_n) < 0.15
_conf = np.where(_wrong, _rng.random(_n) * 0.5, 0.5 + _rng.random(_n) * 0.5)
_tt, _ts = np.ones(_n, int), np.ones(_n, int)
_cls = np.where(np.arange(_n) % 5 == 0, "미란·궤양", "표면 변화")
_tt[(_cls == "미란·궤양") & (np.arange(_n) % 2 == 0)] = 2   # A5 절반만 하향
_r = E.granularity_report("클래스별 시험", {
    "conf": _conf, "wrong": _wrong, "tier_true": _tt, "tier_said": _ts,
    "true_class": _cls})

check("보고에 under_by_class 가 있다", "under_by_class" in _r)
check("보고에 under_worst 가 있다", "under_worst" in _r)
check("★ 한 클래스가 나쁜 것을 잡아낸다 (최악이 전체의 3배 넘음)",
      _r["under_worst"] > _r["under_triage"] * 3,
      f"전체 {_r['under_triage']:.1%} / 최악 {_r['under_worst']:.1%}")
check("최악 클래스 이름을 짚는다", _r["under_worst_class"] == "미란·궤양",
      str(_r["under_worst_class"]))
check("전체 평균만 보면 통과해 보인다 (이게 구멍이었습니다)",
      _r["under_triage"] <= E.UNDER_TRIAGE_MAX * 3,
      f"{_r['under_triage']:.1%}")

_r2 = E.granularity_report("클래스 없이", {
    "conf": _conf, "wrong": _wrong, "tier_true": _tt, "tier_said": _ts})
check("true_class 가 없으면 조용히 빈 dict (기존 호출자 보호)",
      _r2["under_by_class"] == {} and _r2["under_worst"] is None)

check("★ 문턱을 안 건 것이 **의도**라고 코드에 남아 있다",
      E.UNDER_TRIAGE_PER_CLASS_GATE is None)
check("찍기만 하는 최소 표본이 상수로 있다",
      isinstance(E.UNDER_TRIAGE_PER_CLASS_MIN_N, int))

print("\n[8] ★ 하향 방지 규칙이 서빙에 살아 있는가 (STEP 35)")
# 급한 쪽 합이 문턱 이상이면 **덜 급한 묶음을 답 후보에서 뺍니다.**
# holdout: 커버리지 67.9 → 66.5%(−1.3%p), A5 하향 43.8 → 36.8%(−7.0%p).
import os as _os  # noqa: E402

_os.environ["DOG_SKIN_SHOW_GROUP"] = "1"
import importlib  # noqa: E402

import src.message as _MSG  # noqa: E402
from src.config import (DOWNGRADE_BLOCK_MIN, MORPH_GROUP_KEEP_A6,  # noqa: E402
                        URGENT_GROUPS)

importlib.reload(_MSG)
from src.agent import lesion_group as _lg  # noqa: E402

check("문턱이 상수로 있다 (val 에서 고름)", DOWNGRADE_BLOCK_MIN == 0.25,
      str(DOWNGRADE_BLOCK_MIN))
check("급한 쪽 묶음 이름이 MORPH_GROUP_KEEP_A6 과 글자 그대로 맞다",
      set(URGENT_GROUPS) <= set(MORPH_GROUP_KEEP_A6.values()),
      f"{URGENT_GROUPS} vs {sorted(set(MORPH_GROUP_KEEP_A6.values()))}")

# 융기·발진 0.60 으로 1등인데 급한 쪽이 0.30 → **덜 급한 쪽을 말하면 안 됩니다**
_blocked = _lg([("A1", .6), ("A5", .25), ("A6", .05), ("A2", .1)], 0.95)
check("★ 급한 쪽이 문턱을 넘으면 덜 급한 묶음을 말하지 않는다",
      _blocked is None or _blocked["name"] in URGENT_GROUPS,
      str(_blocked))
# 급한 쪽 0.20 (미달) → 예전대로 1등을 말합니다
_ok = _lg([("A1", .7), ("A5", .15), ("A6", .05), ("A2", .1)], 0.95)
check("문턱 미달이면 예전대로 1등을 말한다",
      _ok is not None and _ok["name"] == "융기·발진", str(_ok))
# 급한 쪽이 애초에 낮으면 규칙이 개입하지 않습니다
_free = _lg([("A2", .6), ("A3", .2), ("A1", .1), ("A5", .05), ("A6", .05)], 0.9)
check("급한 쪽이 낮으면 규칙이 개입 안 한다",
      _free is not None and _free["name"] == "표면 변화", str(_free))

# ★ 규칙이 **한 곳**에만 있어야 합니다 — 두 화면이 다른 계열을 말하면 안 됩니다
_line_src = inspect.getsource(_MSG.lesion_group_line)
check("★ message 가 묶음 선택을 다시 계산하지 않는다 (출처 하나)",
      "lesion_group" in _line_src and "MORPH_GROUP_KEEP_A6" not in _line_src,
      "message.py 가 자기 argmax 를 갖고 있으면 규칙이 갈립니다")
check("두 경로가 같은 답을 낸다",
      (_blocked is None) == (_MSG.lesion_group_line(
          [("A1", .6), ("A5", .25), ("A6", .05), ("A2", .1)], 0.95) == ""))

print("\n" + "=" * 60)
print(f" 통과 {ok} / {ok + fail}")
raise SystemExit(1 if fail else 0)
