"""src/errors.py — 오답 분석이 혼동행렬을 제대로 읽는가.

숫자를 손으로 만든 혼동행렬로 확인합니다. 여기가 틀리면 "헛알림의 89% 가
A1/A2 로 간다" 같은 문장이 통째로 거짓말이 됩니다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import errors                                            # noqa: E402
from src.stages import PIPELINE_CLASSES                           # noqa: E402

ok = fail = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {name}")
    else:
        fail += 1
        print(f"  FAIL  {name}  {detail}")


# A1..A6, A7 = 7x7. 손으로 만든 값 — 합계를 눈으로 검산할 수 있게 작게 둡니다.
CM = np.zeros((7, 7), dtype=int)
for i in range(6):
    CM[i, i] = 10                      # 각 병변 10장씩 맞힘
CM[0, 1] = 5                           # A1 → A2  5장
CM[5, 6] = 7                           # A6 → A7  7장 (놓친 병변)
CM[1, 6] = 3                           # A2 → A7  3장
CM[6, 6] = 100                         # 정상 100장 맞힘
CM[6, 0] = 40                          # 정상 → A1  40장  ← 헛알림 최다
CM[6, 1] = 25                          # 정상 → A2  25장
CM[6, 3] = 5                           # 정상 → A4   5장
REP = {"confusion": CM.tolist(), "classes": PIPELINE_CLASSES}

print("\n[1] confusion_pairs — 대각선 밖만, 장수 순")
pairs = errors.confusion_pairs(REP, top=4, show=True)
check("대각선을 안 셉니다", all(d["true"] != d["pred"] for d in pairs))
check("장수 내림차순", [d["n"] for d in pairs] == sorted((d["n"] for d in pairs), reverse=True),
      str([d["n"] for d in pairs]))
check("1위가 A7→A1 40장", pairs[0] == {"true": "A7", "pred": "A1", "n": 40,
                                       "share_of_true": 40 / 170},
      str(pairs[0]))
check("0인 칸은 안 나옵니다", all(d["n"] > 0 for d in pairs))
# 대각선 밖 0 아닌 칸: A1→A2, A2→A7, A6→A7, A7→A1, A7→A2, A7→A4 = 6
check("칸 개수 6개", len(pairs) == 6, str(len(pairs)))

print("\n[2] false_alarm_targets — 정상 행만")
fa = errors.false_alarm_targets(REP, show=True)
check("정상 전체 170", fa["normal_total"] == 170, str(fa["normal_total"]))
check("헛알림 70", fa["false_alarms"] == 70, str(fa["false_alarms"]))
check("A1 40 / A2 25 / A4 5", (fa["by_class"]["A1"], fa["by_class"]["A2"],
                               fa["by_class"]["A4"]) == (40, 25, 5), str(fa["by_class"]))
check("행선지 합 = 헛알림", sum(fa["by_class"].values()) == fa["false_alarms"])
check("A7 자기 자신은 안 셉니다", "A7" not in fa["by_class"])

print("\n[3] miss_sources — 정상이라고 안심시킨 것")
ms = errors.miss_sources(REP, show=True)
check("놓친 병변 10장", ms["total"] == 10, str(ms["total"]))
check("A6 7장 / A2 3장", (ms["by_class"]["A6"], ms["by_class"]["A2"]) == (7, 3),
      str(ms["by_class"]))
check("A6 놓침 비율 7/17", abs(ms["rate_by_class"]["A6"] - 7 / 17) < 1e-9,
      str(ms["rate_by_class"]["A6"]))
check("A3 은 하나도 안 놓침", ms["by_class"]["A3"] == 0)

print("\n[4] 행렬 크기가 안 맞으면 조용히 넘어가지 않습니다")
try:
    errors.confusion_pairs({"confusion": [[1, 2], [3, 4]], "classes": PIPELINE_CLASSES})
    check("ValueError 를 냅니다", False, "예외가 없었습니다")
except ValueError:
    check("ValueError 를 냅니다", True)

print("\n[5] contact_sheet")
tmp = Path("/tmp/kagsim_errors"); tmp.mkdir(exist_ok=True)
from PIL import Image
paths = []
for i in range(8):
    p = tmp / f"{i}.jpg"
    Image.fromarray(np.full((64, 64, 3), i * 30 % 255, dtype=np.uint8)).save(p)
    paths.append(str(p))
df = pd.DataFrame({"crop_path": paths, "note": [f"n{i}" for i in range(8)]})
picks = errors.contact_sheet(df, n=6, cols=3, title="테스트",
                             save_to=tmp / "sheet.png", show=False)
check("6장을 골랐습니다", picks is not None and len(picks) == 6,
      str(None if picks is None else len(picks)))
check("파일이 생겼습니다", (tmp / "sheet.png").exists())
check("빈 df 는 None (죽지 않음)",
      errors.contact_sheet(df.iloc[:0], show=False) is None)
check("경로가 전부 None 이어도 None",
      errors.contact_sheet(pd.DataFrame({"crop_path": [None, None]}), show=False) is None)

print("\n[6] by_group — 부위별로 쪼개기")
gdf = pd.DataFrame({
    # 발바닥 10장 중 8장이 헛알림, 등 10장 중 1장
    "label_orig": ["A7"] * 20 + ["A2"] * 5,
    "pred": (["A2"] * 8 + ["A7"] * 2) + (["A2"] * 1 + ["A7"] * 9) + ["A2"] * 5,
    "region": ["발바닥"] * 10 + ["등"] * 10 + ["등"] * 5,
})
r = errors.by_group(gdf, col="region", show=True)
check("전체 헛알림률 9/20", abs(r["baseline"] - 9 / 20) < 1e-9, str(r["baseline"]))
tbl = {d["region"]: d for d in r["table"]}
check("발바닥 8/10", (tbl["발바닥"]["헛알림"], tbl["발바닥"]["n"]) == (8, 10), str(tbl["발바닥"]))
check("등 1/10 (병변 5장은 안 셈)", (tbl["등"]["헛알림"], tbl["등"]["n"]) == (1, 10),
      str(tbl["등"]))
check("전체대비 = 비율 − 기준", abs(tbl["발바닥"]["전체대비"] - (0.8 - 0.45)) < 1e-9)
check("헛알림 많은 순", [d["region"] for d in r["table"]] == ["발바닥", "등"])
check("컬럼이 없으면 빈 dict", errors.by_group(gdf.drop(columns=["region"]),
                                              col="region") == {})
check("값이 전부 비어도 빈 dict",
      errors.by_group(gdf.assign(region=""), col="region") == {})


print("\n[6] class_dispersion — 멘토 지적을 수치로")
check("문턱이 그대로 (0.85 / 0.60 / 0.50 / 1.25)",
      (errors.DISPERSION_SCATTER, errors.DISPERSION_PAIRWISE,
       errors.PAIRWISE_TOP_SHARE, errors.PRIOR_LIFT_SUSPECT) == (0.85, 0.60, 0.50, 1.25))

_cls = ["A", "B", "C", "D"]
_cm4 = [[900, 20, 20, 60],       # A: 흔한 클래스
        [300, 700, 0, 0],        # B: 오답 전부 A 로 → 짝 혼동
        [40, 100, 60, 100],      # C: 세 곳으로 흩어짐
        [10, 10, 10, 970]]
_by = {d["cls"]: d for d in errors.class_dispersion(
    {"confusion": _cm4, "classes": _cls}, show=False)}
check("B 는 짝 혼동 (H 낮고 한 곳에 몰림)", _by["B"]["shape"] == "짝 혼동",
      f"H={_by['B']['h_norm']:.3f} top={_by['B']['top_share']:.2f}")
check("B 의 최다 행선지는 A", _by["B"]["top_pred"] == "A")
check("C 는 흩어짐", _by["C"]["shape"] == "흩어짐", f"H={_by['C']['h_norm']:.3f}")
check("recall 이 대각선/행합", abs(_by["B"]["recall"] - 700 / 1000) < 1e-9)

# lift 는 **우연 기대값으로 나눠져야** 합니다. 날 것의 비를 쓰면 제일 드문
# 클래스가 모델과 무관하게 무한대가 나옵니다 (A6 에서 실제로 겪었습니다).
# 지지대를 일부러 다르게: A 1000 > B 800 > C 500 > D 300.
# C 를 봅니다 — 더 흔한 쪽(A·B)이 2개, 덜 흔한 쪽(D)이 1개라 우연 기대값이 2/3.
_cm_rare = [[970, 10, 10, 10],           # A 1000
            [10, 770, 10, 10],           # B  800
            [20, 20, 440, 20],           # C  500 — 오답이 A·B·D 로 완전 균등
            [10, 10, 10, 270]]           # D  300
_rr = {d["cls"]: d for d in errors.class_dispersion(
    {"confusion": _cm_rare, "classes": _cls}, show=False)}
check("오답이 균등하면 lift 가 1 근처", abs(_rr["C"]["lift"] - 1.0) < 0.05,
      f"lift={_rr['C']['lift']:.3f} exp={_rr['C']['exp_larger']:.3f}")
check("균등하면 prior 의심 아님", _rr["C"]["prior_suspect"] is False)
check("lift 가 무한대가 되지 않는다",
      all(d["lift"] != float("inf") for d in _rr.values()))
# ⚠️ 제일 흔한 A 와 제일 드문 D 는 lift 를 **잴 수 없습니다** (nan).
check("제일 흔한 클래스는 lift 가 nan", _rr["A"]["lift"] != _rr["A"]["lift"])
check("제일 드문 클래스도 lift 가 nan", _rr["D"]["lift"] != _rr["D"]["lift"])
check("잴 수 없으면 prior 의심도 아님",
      _rr["A"]["prior_suspect"] is False and _rr["D"]["prior_suspect"] is False)

_cm_bias = [[970, 10, 10, 10],
            [10, 770, 10, 10],
            [60, 0, 440, 0],             # C 의 오답이 전부 제일 흔한 A 로
            [10, 10, 10, 270]]
_rb = {d["cls"]: d for d in errors.class_dispersion(
    {"confusion": _cm_bias, "classes": _cls}, show=False)}
check("흔한 쪽으로만 흐르면 lift 가 문턱 위", _rb["C"]["lift"] > 1.25,
      f"lift={_rb['C']['lift']:.3f}")
check("그때 prior 의심", _rb["C"]["prior_suspect"] is True)
check("오답이 없는 클래스는 '오답 없음'",
      errors.class_dispersion({"confusion": [[5, 0], [0, 5]], "classes": ["A", "B"]},
                              show=False)[0]["shape"] == "오답 없음")

print("\n" + "=" * 60)
print(f" 통과 {ok} / {ok + fail}")
sys.exit(1 if fail else 0)
