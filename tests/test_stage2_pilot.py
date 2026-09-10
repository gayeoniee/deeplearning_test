"""`--profile stage2` (STEP 49) — A7 을 빼고 병변 6종, 매 epoch clean + 위치 교란.

    uv run --extra train python -m pytest -q tests/test_stage2_pilot.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from src.config import CLASSES
from tests.test_safe_crop_pilot import packaged, source_manifest, options  # noqa: F401
from tools import kaggle_safe_crop as pilot


def test_stage2_drops_normal_after_package_check(packaged):
    tr, va, metadata = pilot.load_data(packaged, 'stage2')
    assert set(tr.label) <= set(CLASSES) and set(va.label) <= set(CLASSES)
    assert 'A7' not in set(tr.original_label) | set(va.original_label)
    assert len(tr) == sum(metadata['counts'][c]['train'] for c in CLASSES)
    assert len(va) == sum(metadata['counts'][c]['val'] for c in CLASSES)
    tr1, _, _ = pilot.load_data(packaged)                     # 1단계 경로는 그대로
    assert set(tr1.label) == {'A7', 'ABNORMAL'}


def test_shift_moves_fixed_roi_and_clips_at_edge():
    x0, y0, x1, y1 = pilot.roi_window(1920, 1080, [[450, 250, 480, 290]])
    side = x1 - x0
    from PIL import Image
    image = Image.new('RGB', (1920, 1080))
    row = type('Row', (), {'boxes': '[[450, 250, 480, 290]]'})()
    ds = pilot.ShiftedROIDataset.__new__(pilot.ShiftedROIDataset)
    shifted = pilot.ShiftedROIDataset.prepare_image(ds, image, row)
    assert shifted.size == (side, y1 - y0)
    # 경계 근처: 이동이 잘려도 창 크기는 유지됩니다.
    row_edge = type('Row', (), {'boxes': '[[1800, 950, 1919, 1079]]'})()
    edge = pilot.ShiftedROIDataset.prepare_image(ds, image, row_edge)
    assert edge.size[0] >= 320 and edge.size[1] >= 320


def test_stage2_paired_run_resume_and_metrics(packaged, tmp_path):
    args = options(packaged, tmp_path / 'stage2')
    args.profile = 'stage2'
    result = pilot.run(args)
    assert result['completed_epochs'] == {'fixed': 1, 'safe': 1}
    comparison = result['comparison']
    for key in ['macro_f1', 'group4_accuracy', 'A6_recall', 'shift_macro_f1', 'shift_drop_rel']:
        assert key in comparison, key
    protocol = json.loads((args.out / 'protocol.json').read_text())
    assert protocol['task'] == 'stage2' and protocol['classes'] == list(CLASSES)
    saved = np.load(args.out / 'fixed_epoch1_val.npz')
    assert saved['probability'].shape == (protocol['rows']['val'], len(CLASSES))
    assert 'A7' not in set(saved['original_label'])
    assert np.allclose(saved['probability'].sum(1), 1, atol=1e-4)
    before = (args.out / 'fixed_last.pt').stat().st_mtime_ns
    assert pilot.run(args)['common_epoch'] == 1          # 이어받기: 다시 안 돎
    assert (args.out / 'fixed_last.pt').stat().st_mtime_ns == before
    assert (tmp_path / 'stage2_crop_pilot_resume.zip').exists()


def test_stage2_metrics_group4_from_probability_sum():
    n = len(CLASSES)
    target = np.arange(n)
    probability = np.full((n, n), 0.02)
    probability[np.arange(n), np.arange(n)] = 1 - 0.02*(n-1)
    scores = pilot.metrics_stage2(probability, target)
    assert scores['macro_f1'] == 1.0 and scores['group4_accuracy'] == 1.0
    assert scores['A6_recall'] == 1.0 and scores['A4_to_A1'] == 0.0
    # A4 를 A1 로 부르면 6종은 틀리지만 계열(솟아오른 변화)은 맞습니다.
    a1, a4 = CLASSES.index('A1'), CLASSES.index('A4')
    probability[a4] = 0.02
    probability[a4, a1] = 1 - 0.02*(n-1)
    scores = pilot.metrics_stage2(probability, target)
    assert scores['A4_recall'] == 0.0 and scores['A4_to_A1'] == 1.0
    assert scores['group4_accuracy'] == 1.0 and scores['accuracy'] < 1.0


def test_stage2_notebook_embedded_runner(packaged, tmp_path):
    path = Path('notebooks/18_2단계_병변보존_crop_파일럿.ipynb')
    notebook = json.loads(path.read_text())
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), '<notebook>', 'exec')
    source = ''.join(notebook['cells'][1]['source'])
    assert 'HOURS = 0.9' in source and "'--profile', 'stage2'" in ''.join(notebook['cells'][3]['source'])
    import ast
    assignment = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'FILES' for t in n.targets))
    files = ast.literal_eval(assignment.value)
    for name, content in files.items():
        assert content == Path(name).read_text(), f'{name} 스냅샷이 현재 소스와 다릅니다 — 생성기를 다시 돌리세요'
        dest = tmp_path / 'code' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)
    subprocess.run([sys.executable, str(tmp_path/'code/tools/kaggle_safe_crop.py'),
                    '--data', str(packaged), '--out', str(tmp_path/'output'),
                    '--profile', 'stage2', '--smoke', '--epochs', '1', '--workers', '0'],
                   check=True, capture_output=True)
    assert (tmp_path / 'stage2_crop_pilot_resume.zip').exists()
