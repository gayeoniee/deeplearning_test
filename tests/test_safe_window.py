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


def test_dataset_applies_window_only_when_given(tmp_path):
    """SkinDataset 은 window 를 준 경우에만 자르고, build_loaders 는 학습 쪽에만 답니다."""
    import numpy as np
    import pandas as pd
    from PIL import Image
    from src import data
    from src.config import CFG

    p = tmp_path / "a.jpg"
    Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8)).save(p)
    df = pd.DataFrame({"crop_path": [str(p)] * 4, "label": ["A1"] * 4})
    ident = lambda im: im  # noqa: E731
    plain = data.SkinDataset(df, ident)[0][0]
    windowed = data.SkinDataset(df, ident, window=data.SafeWindow((0.42, 0.42)))[0][0]
    assert plain.size == (512, 512)
    assert 214 <= windowed.size[0] <= 216 and windowed.size[0] == windowed.size[1]

    cfg = CFG(img_size=32, batch_size=2, num_workers=0, train_window_side=(0.42, 1.0))
    _, _, ds_tr, ds_va = data.build_loaders(df, df, cfg)
    assert isinstance(ds_tr.window, data.SafeWindow) and ds_va.window is None
    assert CFG.from_dict({**cfg.to_dict(), "train_window_side": [0.5, 1.0]}).train_window_side == (0.5, 1.0)


def test_exp_name_gets_safe_suffix_and_still_parses():
    from src import agent, train
    name = "stage2_effnetv2_s_m2.5_384_moderate_safe0.42"
    assert train.model_key_from_exp(name) == "effnetv2_s"
    assert agent.crop_tag_from_exp(name) == "m2.5"
