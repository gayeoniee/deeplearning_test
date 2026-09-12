"""저장된 `m2.5` 크롭 **안에서** 병변을 보존하는 random 창 (STEP 53 — STEP 49 의 확대판).

STEP 49 파일럿은 원본에서 bbox 근처 창을 흔들었습니다. 캐글엔 원본이 없고 `m2.5` 크롭(13GB)만 있는데,
그 크롭은 **bbox 긴 변 × 2.5 를 bbox 중심에** 놓고 자른 것이라 병변이 항상 크롭 중앙의 **1/2.5 = 40%** 상자 안에
있습니다 (`crop.expand_box`). 그래서 원본 없이도 "병변을 다 담는 범위 안에서 창을 흔드는" 학습이 됩니다.

    창 변 = 크롭 변 × U(lo, hi)  (기본 0.42~1.0: 병변 상자 + 5% 여유 ~ 크롭 전체)
    위치  = 중앙 40% 상자를 다 담는 범위에서 무작위

⚠️ 파일럿의 `safe`(×1.0~1.25, bbox 기준)와 **정의가 다릅니다** — 배포 입력이 사진 전체(포화)라 그쪽으로 넓혔습니다.
사전등록에 그 차이를 적습니다. 검증은 창을 흔들지 않습니다 (크롭 그대로).
"""
from __future__ import annotations

import random

INNER_FRAC = 1.0 / 2.5          # m2.5 크롭 안에서 병변 상자가 차지하는 변 비율


def sample_window_in_crop(width: int, height: int, *, side_range=(0.42, 1.0), inner_frac: float = INNER_FRAC,
                          rng: random.Random | None = None) -> tuple[int, int, int, int]:
    """(x1, y1, x2, y2). 중앙 `inner_frac` 상자를 항상 포함합니다. 크롭이 정사각이 아니면 짧은 변 기준."""
    rng = rng or random
    if width <= 0 or height <= 0:
        raise ValueError("crop dimensions must be positive")
    if not (0 < side_range[0] <= side_range[1] <= 1.0) or side_range[0] < inner_frac:
        raise ValueError("side_range must be within (inner_frac, 1]")
    base = min(width, height)
    side = int(round(base * rng.uniform(*side_range)))
    side = max(side, int(round(base * inner_frac)) + 1)
    cx, cy = width / 2, height / 2
    half = base * inner_frac / 2
    # 창이 중앙 상자를 담으려면: x1 <= cx-half 이고 x1+side >= cx+half
    lo_x, hi_x = max(0, int(round(cx + half - side))), min(width - side, int(round(cx - half)))
    lo_y, hi_y = max(0, int(round(cy + half - side))), min(height - side, int(round(cy - half)))
    if lo_x > hi_x or lo_y > hi_y:                       # 창이 너무 작아 못 담으면 중앙에 둠
        x1, y1 = int(cx - side / 2), int(cy - side / 2)
    else:
        x1, y1 = rng.randint(lo_x, hi_x), rng.randint(lo_y, hi_y)
    return x1, y1, x1 + side, y1 + side
