"""폴리곤 기하 — 넓이·무게중심·창 자르기 (STEP 24).

    uv run python tests/test_polygon_geometry.py

왜 필요한가
-----------
`bbox` 는 폴리곤의 **외접사각형**입니다 (VL01 4,000행에서 오차 0px, 표준편차 0).
지금까지 재 온 "병변 크기" 는 전부 그 네모였는데, 길쭉하거나 굽은 병변에서
네모는 실제 병변보다 훨씬 큽니다. 그 차이를 재는 함수들이라 **틀리면 조용히
틀립니다** — 넓이가 반만 나와도 상관계수는 그럴듯하게 나옵니다.

그리고 `stage1_placement_report` 의 **공선성 잠금장치**를 못 박습니다.
2026-09-06 에 실제로 밟은 함정입니다: 층을 가르는 `occ` 가 병변 크기와
rho +0.995 로 붙어 있는데 코드가 아무 말 없이 "H-A" 판정을 찍었습니다.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import crop, experiments  # noqa: E402

ok = fail = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {name}")
    else:
        fail += 1
        print(f"  FAIL  {name}  {detail}")


def close(a, b, tol=1e-6):
    return abs(a - b) < tol


SQ = [[0, 0], [10, 0], [10, 10], [0, 10]]
TRI = [[0, 0], [10, 0], [0, 10]]

print("\n[1] 넓이 — 신발끈")
check("정사각형 100", close(crop.polygon_area(SQ), 100.0))
check("삼각형 50", close(crop.polygon_area(TRI), 50.0))
check("점 순서를 뒤집어도 같다 (부호 없음)",
      close(crop.polygon_area(SQ[::-1]), 100.0))
check("점이 2개면 0", crop.polygon_area([[0, 0], [1, 1]]) == 0.0)
check("None 이면 0", crop.polygon_area(None) == 0.0)
check("JSON 문자열도 읽는다 (parquet 왕복)",
      close(crop.polygon_area("[[0,0],[10,0],[10,10],[0,10]]"), 100.0))
check("NaN 이 섞이면 0", crop.polygon_area([[0, 0], [1, float("nan")], [2, 2]]) == 0.0)

print("\n[2] 무게중심")
check("정사각형은 한가운데", crop.polygon_centroid(SQ) == (5.0, 5.0))
c = crop.polygon_centroid(TRI)
check("삼각형은 꼭짓점 평균 (10/3, 10/3)", close(c[0], 10 / 3) and close(c[1], 10 / 3))
flat = [[0, 0], [10, 0], [5, 0]]      # 넓이 0 — 직선으로 눌림
c2 = crop.polygon_centroid(flat)
check("넓이 0 이면 꼭짓점 평균으로 물러선다 (0 나눗셈 안 남)",
      c2 is not None and close(c2[0], 5.0) and close(c2[1], 0.0))
check("None 이면 None", crop.polygon_centroid(None) is None)

print("\n[3] 점이 안에 있나")
check("한가운데는 안", crop.point_in_polygon(5, 5, SQ))
check("밖은 밖", not crop.point_in_polygon(15, 5, SQ))
check("멀리 위도 밖", not crop.point_in_polygon(5, -5, SQ))
U = [[0, 0], [10, 0], [10, 10], [7, 10], [7, 3], [3, 3], [3, 10], [0, 10]]  # ㄷ 모양
check("오목한 폴리곤의 파인 곳은 밖", not crop.point_in_polygon(5, 7, U))
check("오목한 폴리곤의 아랫부분은 안", crop.point_in_polygon(5, 1, U))

print("\n[4] 창으로 자르기 (Sutherland-Hodgman)")
check("완전히 안이면 그대로",
      close(crop.polygon_area(crop._clip_to_rect(SQ, (-5, -5, 20, 20))), 100.0))
check("사분의 일만 겹치면 25",
      close(crop.polygon_area(crop._clip_to_rect(SQ, (5, 5, 20, 20))), 25.0))
check("절반이면 50",
      close(crop.polygon_area(crop._clip_to_rect(SQ, (0, 5, 10, 20))), 50.0))
check("안 겹치면 빈 것", len(crop._clip_to_rect(SQ, (50, 50, 60, 60))) < 3)
check("오목한 것도 넓이가 맞다 (ㄷ 모양 전체 = 100 - 4*7 = 72)",
      close(crop.polygon_area(crop._clip_to_rect(U, (-1, -1, 11, 11))), 72.0))

print("\n[5] polygon_in_window — 창이 병변을 얼마나 담았나")
row = {"polygon": SQ, "bbox": [0, 0, 10, 10], "img_w": 100, "img_h": 100,
       "crop_tag": "f320"}
g = crop.polygon_in_window(row, "f320")
# f320 은 100x100 이미지에선 창이 이미지 크기(100)로 잘립니다
check("occ = 병변 100 / 창 100*100", close(g["occ"], 100.0 / (100 * 100), 1e-9))
check("captured = 1 (병변이 다 들어옴)", close(g["captured"], 1.0, 1e-9))
check("slack = 0 (폴리곤이 네모를 꽉 채움)", close(g["slack"], 0.0, 1e-9))
check("center_off = 0", close(g["center_off"], 0.0, 1e-9))
check("center_on = True", g["center_on"] is True)

# 길쭉한 대각선 병변 — 네모는 크지만 실제 병변은 가늘다
diag = [[0, 0], [2, 0], [10, 8], [10, 10], [8, 10], [0, 2]]
row2 = {"polygon": diag, "bbox": [0, 0, 10, 10], "img_w": 100, "img_h": 100}
g2 = crop.polygon_in_window(row2, "f320")
check("대각선 병변은 slack 이 크다 (>0.5)", g2["slack"] > 0.5,
      f"slack={g2['slack']:.3f}")
check("그래도 네모 중심은 병변 안 (대각선이 중심을 지남)", g2["center_on"] is True)

check("폴리곤이 없으면 nan (0 으로 채우지 않는다)",
      crop.polygon_in_window({"bbox": [0, 0, 1, 1], "img_w": 10, "img_h": 10},
                             "f320")["occ"] != crop.polygon_in_window(
          {"bbox": [0, 0, 1, 1], "img_w": 10, "img_h": 10}, "f320")["occ"])

print("\n[6] 판정 상수가 코드에 박혀 있다 (작업 규칙 2)")
check("PLACEMENT_MAX_COLLINEARITY = 0.80",
      experiments.PLACEMENT_MAX_COLLINEARITY == 0.80)
check("PLACEMENT_MIN_MISS = 40", experiments.PLACEMENT_MIN_MISS == 40)
check("PLACEMENT_RATIO_HB = 2.0", experiments.PLACEMENT_RATIO_HB == 2.0)
check("PLACEMENT_RATIO_HA = 0.5", experiments.PLACEMENT_RATIO_HA == 0.5)
check("SHAPE_RHO_MIN = 0.10", experiments.SHAPE_RHO_MIN == 0.10)

print("\n[7] 공선성 잠금장치 — 실제로 밟은 함정")
import numpy as np  # noqa: E402

rng = np.random.default_rng(0)
n = 600
size = rng.uniform(50, 800, n)
rows_collinear = [{"occ": s / 1000.0, "box_px": s, "miss": bool(rng.random() < 0.20),
                   "score": float(rng.random())} for s in size]
r = experiments.stage1_placement_report(rows_collinear, boot=200)
check("occ 가 크기와 같으면 판정을 **거부**한다",
      r["verdict"] == "판정 불가(공선성)", r["verdict"])

rows_free = [{"occ": float(rng.random()), "box_px": s,
              "miss": bool(rng.random() < 0.20), "score": float(rng.random())}
             for s in size]
r2 = experiments.stage1_placement_report(rows_free, boot=200)
check("occ 가 크기와 무관하면 판정한다", r2["verdict"] != "판정 불가(공선성)", r2["verdict"])

r3 = experiments.stage1_placement_report(
    [{"occ": float(rng.random()), "box_px": 1.0, "miss": False, "score": 0.5}] * 30,
    boot=50)
check("놓침이 40건 미만이면 판정하지 않는다", r3["verdict"] == "표본 부족")

print("\n" + "=" * 60)
print(f" 통과 {ok} / {ok + fail}")
sys.exit(1 if fail else 0)
