"""experiments.stage1_window_report — 1단계 고정 창 크기 판정이 안 흔들리나.

⚠️ 요점은 **판정 기준을 못 박는 것**입니다. 결과를 보고 문턱을 고치면
무슨 숫자가 나와도 성공담이 됩니다 (CLAUDE.md 규칙 2).

    uv run --extra train python tests/test_window_sweep.py
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import experiments as E                               # noqa: E402

ok = fail = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {name}")
    else:
        fail += 1
        print(f"  FAIL  {name}  {detail}")


def run(*runs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        v = E.stage1_window_report(list(runs))
    return v, buf.getvalue()


BASE = dict(crop_tag="f320", auroc=0.9504, blur_drop=0.033, false_alarm=0.212,
            overflow_miss=0.046, within_miss=0.018, overflow_miss_ci=0.010,
            minutes=60, converged=True)

print("\n[1] 문턱이 그대로인가")
check("WINDOW_FALSE_ALARM_TOL_PP == 0.02", E.WINDOW_FALSE_ALARM_TOL_PP == 0.02,
      str(E.WINDOW_FALSE_ALARM_TOL_PP))
check("AUROC_NOISE == 0.01 (재사용)", E.AUROC_NOISE == 0.01, str(E.AUROC_NOISE))
check("BLUR_NOISE_PP == 0.05 (재사용)", E.BLUR_NOISE_PP == 0.05, str(E.BLUR_NOISE_PP))
check("1차 기준은 고정값이 아니라 CI", E.WINDOW_MIN_OVERFLOW_GAIN == "ci",
      str(E.WINDOW_MIN_OVERFLOW_GAIN))

print("\n[2] 다 좋으면 채택")
v, _ = run(BASE, dict(BASE, crop_tag="f448", auroc=0.9498, overflow_miss=0.028,
                      within_miss=0.020, false_alarm=0.215, blur_drop=0.040))
check("채택된다", v["verdict"] == "채택: f448", v["verdict"])

print("\n[3] 1차(넘친 병변 놓침)를 못 넘으면 기각 — 점수보다 이게 먼저")
v, _ = run(BASE, dict(BASE, crop_tag="f448", auroc=0.9600,      # AUROC 는 올랐는데
                      overflow_miss=0.045, within_miss=0.015,   # 개선이 CI 안
                      false_alarm=0.200, blur_drop=0.030))
check("AUROC 가 올라도 1차를 못 넘으면 기각",
      v["candidates"][0]["verdict"].startswith("기각"), v["candidates"][0]["verdict"])

print("\n[4] 개선폭이 CI 반폭보다 작으면 안 됩니다 (잡음에 안 속게)")
v, _ = run(BASE, dict(BASE, crop_tag="f448", overflow_miss=0.040,   # -0.006
                      overflow_miss_ci=0.010))                      # CI ±0.010
check("0.006 개선 < CI 0.010 → 기각", v["candidates"][0]["verdict"].startswith("기각"),
      str(v["candidates"][0]))
v, _ = run(BASE, dict(BASE, crop_tag="f448", overflow_miss=0.040,
                      overflow_miss_ci=0.003))                      # CI 를 좁히면
check("같은 개선폭이라도 CI 가 좁으면 통과", not v["candidates"][0]["verdict"].startswith("기각"),
      str(v["candidates"][0]))

print("\n[5] 1차는 넘겼는데 다른 데서 잃으면 '맞바꿈'")
for name, over in [("AUROC", dict(auroc=0.9350)),
                   ("흐림 하락", dict(blur_drop=0.120)),
                   ("헛알림", dict(false_alarm=0.250))]:
    v, _ = run(BASE, dict(BASE, crop_tag="f448", overflow_miss=0.020, **over))
    check(f"{name} 가 나빠지면 맞바꿈", v["candidates"][0]["verdict"].startswith("맞바꿈"),
          v["candidates"][0]["verdict"])

print("\n[6] 안전장치")
v, _ = run(dict(BASE, crop_tag="f448"))          # 기준이 없음
check("기준 크롭이 없으면 판정 안 함", "verdict" not in v or v.get("verdict") is None,
      str(v.keys()))
v, out = run(dict(BASE, converged=False),
             dict(BASE, crop_tag="f448", overflow_miss=0.020))
check("기준이 미수렴이면 경고", "수렴하지 않았습니다" in out)
v, _ = run()
check("빈 입력이면 빈 dict", v == {})

print("\n[7] '넘친 것' 정의가 기준 창에 고정된다고 화면에 적히나")
_, out = run(BASE, dict(BASE, crop_tag="f448", overflow_miss=0.020))
check("정의 경고가 출력된다", "f320 기준" in out and "같은 사진 집합" in out)

print("\n" + "=" * 60)
print(f" 통과 {ok} / {ok + fail}")
sys.exit(1 if fail else 0)
