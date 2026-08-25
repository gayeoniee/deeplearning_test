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

print("\n[6] subsample — GPU 없이 돌릴 때 표본 줄이기")
big = pd.DataFrame({"label_orig": (["A7"] * 600 + ["A1"] * 300 + ["A6"] * 100),
                    "x": range(1000)})
small = errors.subsample(big, 300, verbose=False)
check("300장 근처로 줄었습니다", 290 <= len(small) <= 310, str(len(small)))
_share = (small["label_orig"].value_counts(normalize=True)
          - big["label_orig"].value_counts(normalize=True)).abs().max()
check("클래스 비율이 유지됩니다 (오차 2%p 이내)", _share < 0.02, f"{_share:.3f}")
check("n=None 이면 그대로", errors.subsample(big, None, verbose=False) is big)
check("이미 작으면 그대로", errors.subsample(big, 5000, verbose=False) is big)
check("같은 seed 면 같은 표본",
      errors.subsample(big, 300, verbose=False)["x"].tolist()
      == errors.subsample(big, 300, verbose=False)["x"].tolist())
check("seed 가 다르면 다른 표본",
      errors.subsample(big, 300, seed=1, verbose=False)["x"].tolist()
      != errors.subsample(big, 300, seed=2, verbose=False)["x"].tolist())

print("\n" + "=" * 60)
print(f" 통과 {ok} / {ok + fail}")
sys.exit(1 if fail else 0)
