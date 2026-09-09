import random
import json
import subprocess
import sys
from pathlib import Path

from src.roi_data import roi_window
from tests.test_safe_crop_pilot import packaged, source_manifest, options
from tools import kaggle_safe_crop as pilot


def test_roi_retains_all_boxes_at_edges_and_large_spans():
    for boxes in ([[0, 0, 20, 40]], [[900.2, 500.1, 1919.8, 1079.9]],
                  [[2, 2, 20, 20], [1800, 950, 1920, 1080]],
                  [[450, 250, 480, 290]]):
        for augment in (False, True):
            for seed in range(100):
                x, y, r, b = roi_window(1920, 1080, boxes, augment=augment,
                                       rng=random.Random(seed))
                assert 0 <= x < r <= 1920 and 0 <= y < b <= 1080
                assert all(x <= a and y <= c and r >= d and b >= e for a,c,d,e in boxes)
    assert roi_window(80, 48, [], augment=True) == (0, 0, 80, 48)
    box = [[450, 250, 480, 290]]
    assert len({roi_window(1920, 1080, box, augment=True, rng=random.Random(i))
                for i in range(10)}) > 1


def test_roi_paired_run_and_resume(packaged, tmp_path):
    args = options(packaged, tmp_path / 'roi')
    args.profile = 'roi'
    result = pilot.run(args)
    assert result['completed_epochs'] == {'fixed': 1, 'safe': 1}
    checkpoint = args.out / 'fixed_last.pt'
    before = checkpoint.stat().st_mtime_ns
    assert pilot.run(args)['common_epoch'] == 1
    assert checkpoint.stat().st_mtime_ns == before
    assert (tmp_path / 'roi_crop_pilot_resume.zip').exists()


def test_notebook_embedded_runner(packaged, tmp_path):
    notebook = json.loads(Path('notebooks/03i_ROI_crop_비교.ipynb').read_text())
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), '<notebook>', 'exec')
    source = ''.join(notebook['cells'][1]['source'])
    import ast
    assignment = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'FILES' for t in n.targets))
    files = ast.literal_eval(assignment.value)
    for name, content in files.items():
        assert content == Path(name).read_text()
        dest = tmp_path / 'code' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)
    subprocess.run([sys.executable, str(tmp_path/'code/tools/kaggle_safe_crop.py'),
                    '--data', str(packaged), '--out', str(tmp_path/'output'),
                    '--profile', 'roi', '--smoke', '--epochs', '1', '--workers', '0'],
                   check=True, capture_output=True)
