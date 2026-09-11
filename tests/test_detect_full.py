"""STEP 50 — 전체 원본 검출 데이터셋 · 코랩 노트북 19 · 끝단 커버리지 도구.

    uv run --extra train python -m pytest -q tests/test_detect_full.py
"""
import ast
import json
from pathlib import Path

from tools import make_detect_dataset_full as mk


def test_largest_box_dedups_pairs_and_picks_biggest():
    boxes = json.dumps([[10, 10, 20, 20], [10, 10, 20, 20], [0, 0, 50, 40], [0, 0, 50, 40], [5, 5, 5, 9]])
    assert mk.largest_box(boxes) == (0.0, 0.0, 50.0, 40.0)
    assert mk.largest_box(json.dumps([[3, 3, 3, 3]])) is None


def test_geometry_is_deterministic_and_keeps_box_inside():
    boxes = json.dumps([[578.0, 611.0, 638.0, 721.0]])
    a = mk.geometry('abc.jpg', 1920, 1080, boxes)
    b = mk.geometry('abc.jpg', 1920, 1080, boxes)
    assert a == b and a is not None
    wx, wy, side, t = a
    assert 480 <= side <= 1080 and all(0 <= v <= 1 for v in t) and t[2] > t[0] and t[3] > t[1]
    assert mk.geometry('abd.jpg', 1920, 1080, boxes) != a          # 이름이 다르면 창도 다릅니다
    # 창이 사진 안: 창 크기와 위치가 원본을 넘지 않습니다
    assert 0 <= wx and wx + side <= 1920 + 1e-6 and 0 <= wy and wy + side <= 1080 + 1e-6


def test_geometry_rejects_lesions_too_large_for_the_window():
    huge = json.dumps([[0, 0, 1900, 1070]])
    assert mk.geometry('x.jpg', 1920, 1080, huge) is None


def test_colab_notebook_19_compiles_and_bundles_current_src():
    nb = json.loads(Path('notebooks/19_병변_검출기_전체데이터_colab.ipynb').read_text())
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            compile(''.join(c['source']), '<nb>', 'exec')
    source = ''.join(nb['cells'][1]['source'])
    assign = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'FILES' for t in n.targets))
    for name, content in ast.literal_eval(assign.value).items():
        assert content == Path(name).read_text(), f'{name} 스냅샷이 현재 소스와 다릅니다 — 생성기를 다시 돌리세요'
    train_cell = ''.join(nb['cells'][3]['source'])
    assert 'T.Resize((IMG, IMG))' in train_cell and 'T.CenterCrop' not in train_cell   # STEP 42 의 가장자리 절단 금지
    assert 'fold != 0' in ''.join(nb['cells'][2]['source'])                          # fold 0 = 검증
    fetch_cell = ''.join(nb['cells'][2]['source'])
    assert 'is_holdout' not in fetch_cell + train_cell                                # holdout 은 데이터에 없음
