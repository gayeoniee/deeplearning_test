"""STEP 53 — m2.5 크롭 안 병변 보존 창.  uv run python -m pytest -q tests/test_safe_window.py"""
import random

import pytest

from src.safe_window import INNER_FRAC, sample_window_in_crop


def test_window_always_contains_central_lesion_box():
    for W, H in [(512, 512), (400, 400), (640, 480), (97, 97)]:
        base = min(W, H); half = base * INNER_FRAC / 2
        for seed in range(300):
            x1, y1, x2, y2 = sample_window_in_crop(W, H, rng=random.Random(seed))
            assert 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H and x2 - x1 == y2 - y1
            assert x1 <= W/2 - half + 1 and x2 >= W/2 + half - 1 and y1 <= H/2 - half + 1 and y2 >= H/2 + half - 1


def test_window_varies_in_size_and_position():
    ws = {sample_window_in_crop(512, 512, rng=random.Random(s)) for s in range(50)}
    sides = {w[2] - w[0] for w in ws}
    assert len(ws) > 30 and min(sides) < 300 < max(sides)
    assert sample_window_in_crop(512, 512, side_range=(1.0, 1.0)) == (0, 0, 512, 512)   # 상한 = 크롭 전체


def test_rejects_ranges_that_could_cut_the_lesion():
    with pytest.raises(ValueError):
        sample_window_in_crop(512, 512, side_range=(0.3, 1.0))
