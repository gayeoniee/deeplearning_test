"""tools/lesion_size_stats.py — 계산이 맞나 · 문턱이 안 흔들리나.

⚠️ 요점은 **판정 문턱을 못 박는 것**입니다. 결과를 보고 문턱을 고치면
무슨 숫자가 나와도 성공담이 됩니다 (CLAUDE.md 규칙 2).

    uv run --extra train python tests/test_lesion_size_stats.py
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402

import lesion_size_stats as L                                   # noqa: E402

ok = fail = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {name}")
    else:
        fail += 1
        print(f"  FAIL  {name}  {detail}")


print("\n[1] 사전등록 문턱이 그대로인가 (2026-09-05 실측 때 쓴 값)")
check("OVERFLOW_RATIO_STRONG == 1.5", L.OVERFLOW_RATIO_STRONG == 1.5,
      str(L.OVERFLOW_RATIO_STRONG))
check("PARTIAL_RHO_MIN == 0.15", L.PARTIAL_RHO_MIN == 0.15, str(L.PARTIAL_RHO_MIN))
check("MIN_N == 20", L.MIN_N == 20, str(L.MIN_N))
check("STAGE1_WINDOW_PX == 320 (f320 과 같아야 함)", L.STAGE1_WINDOW_PX == 320,
      str(L.STAGE1_WINDOW_PX))

print("\n[2] bbox 파싱 — 매니페스트는 xyxy 를 **문자열**로 담습니다")
s = pd.Series(['[264.0, 589.0, 592.0, 1061.0]', '[0, 0, 10, 4]'])
got = L.bbox_long_side(s)
check("문자열 xyxy 에서 긴 변", np.allclose(got, [472.0, 10.0]), str(got))
check("list 도 받는다", np.allclose(L.bbox_long_side(pd.Series([[0, 0, 3, 9]])), [9.0]))

print("\n[3] 부분상관 — 통제변수가 전부일 때 0 으로 내려가나")
rng = np.random.default_rng(0)
c = rng.normal(size=4000)
d = pd.DataFrame({"ctrl": c, "x": c + rng.normal(0, .05, 4000),
                  "y": c + rng.normal(0, .05, 4000)})
raw_ok = abs(L.partial_spearman(d, "x", "y", "ctrl")) < 0.15
check("x,y 가 ctrl 로만 얽혀 있으면 부분상관 ~0", raw_ok,
      f"{L.partial_spearman(d, 'x', 'y', 'ctrl'):+.3f}")
d2 = pd.DataFrame({"ctrl": c, "x": rng.normal(size=4000)})
d2["y"] = d2["x"]
check("ctrl 과 무관한 진짜 상관은 살아남는다",
      L.partial_spearman(d2, "x", "y", "ctrl") > 0.8,
      f"{L.partial_spearman(d2, 'x', 'y', 'ctrl'):+.3f}")

print("\n[4] 붙일 값이 0장이면 **멈추나** (조용한 NaN 금지 — CLAUDE.md 함정)")
tmp = Path(ROOT / "data" / "work" / "reports")
tmp.mkdir(parents=True, exist_ok=True)
p_saved, p_mf = tmp / "_t_saved.parquet", tmp / "_t_mf.parquet"
pd.DataFrame({"image_path": ["a", "b"], "score": [.9, .1],
              "said_abnormal": [True, False], "is_normal": [False, False],
              "blur": [.1, .2]}).to_parquet(p_saved)
pd.DataFrame({"image_path": ["zzz"], "label": ["A1"],
              "bbox": ['[0,0,10,10]']}).to_parquet(p_mf)
try:
    L.attach(p_saved, p_mf, 320)
    check("매니페스트에 없는 행이면 SystemExit", False, "안 멈췄습니다")
except SystemExit as e:
    check("매니페스트에 없는 행이면 SystemExit", "행이" in str(e), str(e)[:60])
finally:
    p_saved.unlink(missing_ok=True)
    p_mf.unlink(missing_ok=True)

print("\n[5] 판정이 방향을 제대로 읽나 — 큰 병변이 더 놓치게 만든 가짜 데이터")
n = 600
big = np.r_[np.full(n, 100.0), np.full(n, 640.0)]        # occ 0.31 / 2.0
miss = np.r_[np.zeros(n, bool), np.ones(n, bool)]        # 큰 쪽만 놓침
d3 = pd.DataFrame({
    "is_normal": np.zeros(2 * n, bool),
    "label": ["A1"] * n + ["A1"] * n,
    "long": big, "occ": big / 320.0,
    "said_abnormal": ~miss,
    "score": np.r_[np.full(n, .9), np.full(n, .1)],
    "blur": rng.normal(0, 1, 2 * n),
})
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    r = L.report(d3, 320)
check("배수가 문턱을 넘으면 '크기 효과 있음'", r["verdict"] == "크기 효과 있음",
      r["verdict"])
check("창 밖 놓침률이 100%", abs(r["miss_overflow"] - 1.0) < 1e-9,
      str(r["miss_overflow"]))
check("크기-점수 상관이 음수 (큰 쪽 점수가 낮음)", r["rho_size_score"] < -0.5,
      f"{r['rho_size_score']:+.3f}")

print("\n[6] 효과가 없으면 '미지지' 로 떨어지나")
d4 = d3.copy()
d4["said_abnormal"] = True                                # 아무도 안 놓침
d4["score"] = .9
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    r4 = L.report(d4, 320)
check("놓침이 없으면 '크기 효과 있음' 이 안 나온다",
      r4["verdict"] == "미지지", r4["verdict"])

print("\n" + "=" * 60)
print(f" 통과 {ok} / {ok + fail}")
sys.exit(1 if fail else 0)
