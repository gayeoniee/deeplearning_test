import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import shutil
import sys
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from tools.package_safe_crop_pilot import package, select_pilot
from tools import kaggle_safe_crop as pilot


@pytest.fixture
def source_manifest(tmp_path):
    archive = tmp_path / 'source.zip'
    rows = []
    with zipfile.ZipFile(archive, 'w') as z:
        for label in range(1, 8):
            for part, count in [('train', 4), ('val', 2), ('holdout', 1)]:
                for index in range(count):
                    name = f'{label}_{part}_{index}'
                    image = Image.new('RGB', (80, 48), (label*25, index*40, len(part)*15))
                    data = io.BytesIO()
                    image.save(data, format='JPEG')
                    raw = data.getvalue()
                    z.writestr(name+'.jpg', raw)
                    z.writestr(name+'.json', json.dumps({'name': name}))
                    rows.append(dict(sha256=hashlib.sha256(raw).hexdigest(), label=f'A{label}',
                                     group=f'{label}_{part}', animal_id=f'{label}_{part}',
                                     img_w=80, img_h=48, boxes='[[20, 10, 40, 30]]',
                                     source_chunk='TL01', is_holdout=part == 'holdout',
                                     fold=0 if part == 'val' else 1, zip_path=str(archive),
                                     zip_member=name+'.jpg', json_member=name+'.json',
                                     image_path=f'{archive}!{name}.jpg'))
    path = tmp_path / 'manifest.parquet'
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


@pytest.fixture
def packaged(source_manifest, tmp_path):
    path = tmp_path / 'pilot.zip'
    package(source_manifest, path, train_size=14, val_size=7)
    root = tmp_path / 'data'
    with zipfile.ZipFile(path) as z:
        z.extractall(root)
    return root


def options(root, output, hours=1):
    return argparse.Namespace(data=root, out=output, hours=hours, epochs=1,
                              batch_size=4, workers=0, smoke=True)


def test_portable_package_preserves_bytes_and_split(packaged, source_manifest):
    source = pd.read_parquet(source_manifest)
    selected = pd.read_parquet(packaged / 'pilot_manifest.parquet')
    assert selected.pilot_split.value_counts().to_dict() == {'train': 14, 'val': 7}
    assert set(selected.sha256) <= set(source[~source.is_holdout].sha256)
    assert selected.groupby('pilot_split').label.nunique().eq(7).all()
    assert not set(selected[selected.pilot_split == 'train'].group) & set(selected[selected.pilot_split == 'val'].group)
    for row in selected.itertuples():
        assert hashlib.sha256((packaged / row.image_path).read_bytes()).hexdigest() == row.sha256
    assert not list(packaged.rglob('.env'))
    assert 'source.zip' not in (packaged / 'pilot_package.json').read_text()


def test_selection_rejects_existing_leakage(source_manifest):
    df = pd.read_parquet(source_manifest)
    df['animal_id'] = 'shared'
    with pytest.raises(ValueError, match='overlap'):
        select_pilot(df, 14, 7)


def test_interruption_resume_and_common_initialization(packaged, tmp_path, monkeypatch):
    output = tmp_path / 'run'
    args = options(packaged, output)
    original_epoch = pilot.one_epoch

    def interrupt_safe(*positional, **kwargs):
        if positional[6] == 'safe':
            raise RuntimeError('simulated interruption')
        return original_epoch(*positional, **kwargs)

    monkeypatch.setattr(pilot, 'one_epoch', interrupt_safe)
    with pytest.raises(RuntimeError, match='simulated'):
        pilot.run(args)
    assert (output / 'random_last.pt').exists()
    assert not (output / 'safe_last.pt').exists()
    random_mtime = (output / 'random_last.pt').stat().st_mtime_ns
    assert json.loads((output / 'session.json').read_text())['stop_reason'] == 'error_or_interruption'
    monkeypatch.setattr(pilot, 'one_epoch', original_epoch)
    summary = pilot.run(args)
    assert summary['common_epoch'] == 1
    assert (output / 'random_last.pt').stat().st_mtime_ns == random_mtime
    states = [pilot.load_checkpoint(output / f'{mode}_last.pt') for mode in ['random', 'safe']]
    digest = hashlib.sha256((output / 'initial.pt').read_bytes()).hexdigest()
    assert all(s['initial_sha256'] == digest for s in states)
    assert all(len(s['history']) == 1 for s in states)
    for state in states:
        assert state['optimizer']['state']
        assert all(torch.isfinite(v).all() for v in state['model'].values())
    assert (output.parent / 'safe_crop_pilot_resume.zip').exists()
    assert pilot.run(args)['completed_epochs'] == {'random': 1, 'safe': 1}
    changed = options(packaged, output)
    changed.batch_size = 8
    with pytest.raises(ValueError, match='protocol differs'):
        pilot.run(changed)


def test_budget_expiry_keeps_valid_empty_report(packaged, tmp_path):
    output = tmp_path / 'expired'
    summary = pilot.run(options(packaged, output, hours=1e-9))
    assert summary['common_epoch'] == 0
    assert pd.read_csv(output / 'history.csv').empty
    assert not (output / 'random_last.pt').exists()


def test_packaged_cli_works_outside_repository(packaged, tmp_path):
    command = [sys.executable, str(packaged / 'tools/kaggle_safe_crop.py'),
               '--data', str(packaged), '--out', str(tmp_path / 'cli'),
               '--smoke', '--epochs', '1', '--workers', '0', '--batch-size', '4']
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / 'cli/comparison.json').read_text())
    assert report['common_epoch'] == 1


def test_notebook_cells_and_resume_zip_end_to_end(packaged, tmp_path, monkeypatch):
    notebook = Path(__file__).resolve().parents[1] / 'notebooks/14_원본_병변보존_crop_파일럿.ipynb'
    cells = json.loads(notebook.read_text())['cells']
    input_root = tmp_path / 'input'
    shutil.copytree(packaged, input_root / 'dataset')
    actual_run = subprocess.run

    def run_cpu_smoke(command, **kwargs):
        if '-c' in command:
            return subprocess.CompletedProcess(command, 0, stdout='Simulated GPU preflight passed\n', stderr='')
        return actual_run([*command, '--smoke'], timeout=90, **kwargs)

    monkeypatch.setattr(subprocess, 'run', run_cpu_smoke)
    for attempt in [1, 2]:
        output = tmp_path / f'work{attempt}' / 'safe_crop_pilot'
        namespace = {'display': lambda value: None}
        for cell in cells:
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(cell['source'])
            source = source.replace("Path('/kaggle/input')", f'Path({str(input_root)!r})')
            source = source.replace("Path('/kaggle/working/safe_crop_pilot')", f'Path({str(output)!r})')
            source = source.replace("Path('/kaggle/temp/safe_crop_pilot_data')", f'Path({str(tmp_path / "scratch")!r})')
            source = source.replace('EPOCHS = 5', 'EPOCHS = 1').replace('WORKERS = 2', 'WORKERS = 0')
            source = source.replace('assert torch.cuda.is_available(),', 'assert True,')
            source = source.replace('torch.cuda.get_device_name(0)', "'CPU smoke test'")
            exec(compile(source, str(notebook), 'exec'), namespace)
        assert json.loads((output / 'comparison.json').read_text())['common_epoch'] == 1
        if attempt == 1:
            shutil.copy(output.parent / 'safe_crop_pilot_resume.zip', input_root / 'safe_crop_pilot_resume.zip')


def test_notebook_stops_before_data_when_cuda_kernel_is_unsupported(monkeypatch):
    notebook = Path(__file__).resolve().parents[1] / 'notebooks/14_원본_병변보존_crop_파일럿.ipynb'
    source = ''.join(json.loads(notebook.read_text())['cells'][1]['source'])
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda index: 'Tesla P100')
    commands = []

    def fail_probe(command, **kwargs):
        commands.append(command)
        assert command[2] == '-c'
        assert 'resnet50(weights=None)' in command[3]
        assert '.backward()' in command[3]
        assert 'for amp in (False, True)' in command[3]
        assert kwargs['timeout'] == 120
        return subprocess.CompletedProcess(command, 1, stdout='GPU: Tesla P100\n',
                                           stderr='CUDA error: no kernel image is available')

    monkeypatch.setattr(subprocess, 'run', fail_probe)
    with pytest.raises(RuntimeError, match='GPU 연산 사전 검사 실패'):
        exec(compile(source, str(notebook), 'exec'), {})
    assert len(commands) == 1  # No weight download or training subprocess started.
