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


def test_multibox_labels_keep_inside_boxes_and_clip():
    from tools.detect_multibox_labels import boxes_in_window
    boxes = json.dumps([[100, 100, 200, 200], [100, 100, 200, 200], [900, 900, 1000, 1000], [480, 480, 560, 560]])
    out = boxes_in_window(boxes, wx=0, wy=0, side=500)
    assert [0.2, 0.2, 0.4, 0.4] in [[round(v, 3) for v in b] for b in out]      # 안에 있는 것 (중복은 하나로)
    assert not any(b[0] > 0.9 for b in out)                                      # 밖에 있는 것은 제외
    assert all(0 <= v <= 1 for b in out for v in b)                              # 경계로 자름
    assert len(out) == 1                                                         # 걸친 것(480~560, 창 안 4%) 은 80% 미만이라 제외


def test_colab_notebook_20_compiles_bundles_src_and_targets_dfine():
    nb = json.loads(Path('notebooks/20_병변_검출_DFINE_colab.ipynb').read_text())
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            compile(''.join(c['source']), '<nb>', 'exec')
    source = ''.join(nb['cells'][1]['source'])
    assign = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'FILES' for t in n.targets))
    for name, content in ast.literal_eval(assign.value).items():
        assert content == Path(name).read_text(), f'{name} 스냅샷이 현재 소스와 다릅니다 — 생성기를 다시 돌리세요'
    assert "MODEL_ID = 'ustc-community/dfine-large-obj2coco-e25'" in source
    fetch, train = ''.join(nb['cells'][2]['source']), ''.join(nb['cells'][3]['source'])
    assert 'boxes_multi.parquet' in fetch and 'fold != 0' in fetch
    assert 'num_labels=1' in train and "rep['off_median'] < best" in train        # 1클래스 · best 는 중심 오차
    assert 'is_holdout' not in fetch + train


def test_kaggle_notebook_21_compiles_and_trains_with_normals_as_empty_targets():
    nb = json.loads(Path('notebooks/21_검출기_1단계후보_kaggle.ipynb').read_text())
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            compile(''.join(c['source']), '<nb>', 'exec')
    source = ''.join(nb['cells'][1]['source'])
    assign = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'FILES' for t in n.targets))
    for name, content in ast.literal_eval(assign.value).items():
        assert content == Path(name).read_text(), f'{name} 스냅샷이 현재 소스와 다릅니다'
    fetch, train = ''.join(nb['cells'][2]['source']), ''.join(nb['cells'][3]['source'])
    assert "normals['boxes'] = [[] for _ in range(len(normals))]" in fetch          # 정상 = 정답 네모 0개
    assert 'torch.zeros(0, 4)' in train and 'GradScaler' in train and 'deadline' in train   # 빈 정답 · fp16 · 시간 예산
    assert "rep['auroc'] > best" in train                                             # 1단계 후보 — best 는 AUROC
    assert 'is_holdout' not in fetch + train
