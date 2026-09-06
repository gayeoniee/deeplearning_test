"""STEP 27 판정 기준을 못 박습니다 — 결과를 보고 문턱을 바꿀 수 없게 (작업 규칙 2).

특히 감시하는 것: **오답의 정의에 헛알림이 들어가는가.** STEP 11 이 이걸 빼고
재서 "커버리지 18.2%" 를 냈고, 우리는 그 숫자로 화면 규격을 정했습니다.
정상 사진에 병변 이름을 붙이면 **무조건 오답**이어야 합니다.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import experiments as E

ok = fail = 0


def check(name, cond):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {name}")
    else:
        fail += 1
        print(f"  FAIL  {name}")


print("[1] 문턱이 코드에 박혀 있는가")
check("목표 오답률 20% (STEP 11 과 같은 값)", E.NAMING_TARGET_ERROR == 0.20)
check("논의 문턱 50%", E.NAMING_MIN_COVERAGE == 0.50)
check("닫음 문턱 30%", E.NAMING_CLOSE_COVERAGE == 0.30)
check("닫음 < 논의", E.NAMING_CLOSE_COVERAGE < E.NAMING_MIN_COVERAGE)
check("A6 안전 관문이 있다", 0 < E.NAMING_A6_MISS_MAX <= 0.5)

print("\n[2] 완벽한 모델은 커버리지 100%")
n = 1000
r = E.naming_report({"conf": np.linspace(0, 1, n), "wrong": np.zeros(n, bool),
                     "is_a6": np.zeros(n, bool), "said_a6": np.zeros(n, bool)})
check("오답이 없으면 커버리지 1.0", r["coverage"] == 1.0)
check("판정이 '논의할 가치 있음'", r["verdict"] == "논의할 가치 있음")

print("\n[3] 전부 틀리면 커버리지 0")
r = E.naming_report({"conf": np.linspace(0, 1, n), "wrong": np.ones(n, bool),
                     "is_a6": np.zeros(n, bool), "said_a6": np.zeros(n, bool)})
check("커버리지 0", r["coverage"] == 0.0)
check("판정이 '축을 닫음'", r["verdict"] == "축을 닫음")

print("\n[4] ★ 헛알림을 오답으로 안 세면 판정이 뒤집힌다 (STEP 11 의 구멍)")
# 절반이 헛알림(정상인데 이름 붙음)이고, 병변 쪽은 다 맞힌 모델.
half = n // 2
wrong_honest = np.zeros(n, bool)
wrong_honest[:half] = True           # 앞 절반 = 헛알림 → 무조건 오답
conf = np.linspace(1, 0, n)          # 헛알림이 오히려 확신도가 높은 최악의 경우
honest = E.naming_report({"conf": conf, "wrong": wrong_honest,
                          "is_a6": np.zeros(n, bool), "said_a6": np.zeros(n, bool)})
naive = E.naming_report({"conf": conf[half:], "wrong": wrong_honest[half:],
                         "is_a6": np.zeros(half, bool), "said_a6": np.zeros(half, bool)})
check("정직한 쪽이 축을 닫는다", honest["verdict"] == "축을 닫음")
check("헛알림을 빼면 100% 로 보인다", naive["coverage"] == 1.0)
check("두 판정이 다르다 — 분모를 틀리면 결론이 뒤집힙니다",
      honest["verdict"] != naive["verdict"])

print("\n[5] ★ A6 안전 관문은 다른 관문을 통과해도 이긴다")
# 커버리지는 훌륭한데 실제 A6 을 전부 다른 이름으로 부르는 모델.
w = np.zeros(n, bool)
is_a6 = np.zeros(n, bool)
is_a6[::20] = True                   # 5% 가 A6
r = E.naming_report({"conf": np.linspace(0, 1, n), "wrong": w,
                     "is_a6": is_a6, "said_a6": np.zeros(n, bool)})
check("커버리지가 100% 여도", r["coverage"] == 1.0)
check("A6 을 다 놓치면 안전 관문 실패", r["verdict"].startswith("안전 관문 실패"))

print("\n[6] 길이가 안 맞으면 조용히 넘어가지 않는다")
try:
    E.naming_report({"conf": [0.1, 0.2], "wrong": [True],
                     "is_a6": [False, False], "said_a6": [False, False]})
    check("길이 불일치를 잡는다", False)
except ValueError:
    check("길이 불일치를 잡는다", True)

print("\n" + "=" * 60)
print(f" 통과 {ok} / {ok + fail}")
raise SystemExit(1 if fail else 0)
