import io
import json
import pickle
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from src.config import CFG
from src.original_data import OriginalDataset, build_original_loaders, letterbox


@pytest.fixture
def originals(tmp_path):
    archive = tmp_path / 'original.zip'
    image = Image.new('RGB', (160, 90), (240, 50, 30))
    raw = io.BytesIO()
    image.save(raw, format='JPEG')
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('one.jpg', raw.getvalue())
    return pd.DataFrame([dict(image_path=f'{archive}!one.jpg', zip_path=str(archive),
                              zip_member='one.jpg', img_w=160, img_h=90,
                              boxes=json.dumps([[0, 0, 160, 90]]), label='A1')])


def test_validation_identical_for_every_training_mode(originals):
    cfg = CFG(img_size=64)
    values = []
    for mode in ['safe', 'random', 'full']:
        ds = OriginalDataset(originals, cfg, mode=mode)
        x, y = ds[0]
        assert y == 0 and x.shape == (3, 64, 64)
        assert torch.equal(x, ds[0][0])
        values.append(x)
        ds.close()
    assert all(torch.equal(values[0], v) for v in values)


def test_safe_full_box_survives_to_tensor_and_worker_pickle(originals):
    cfg = CFG(img_size=64, hflip=0, vflip=0, color_jitter=0, hue_jitter=0)
    ds = OriginalDataset(originals, cfg, train=True)
    eval_ds = OriginalDataset(originals, cfg)
    assert torch.equal(ds[0][0], eval_ds[0][0])
    restored = pickle.loads(pickle.dumps(ds))
    assert torch.equal(ds[0][0], restored[0][0])
    for dataset in [ds, eval_ds, restored]:
        dataset.close()


def test_missing_image_raises_instead_of_gray_training_sample(originals):
    originals.loc[0, 'zip_member'] = 'missing.jpg'
    with pytest.raises(KeyError):
        OriginalDataset(originals, CFG())[0]


def test_letterbox_keeps_edge_content():
    array = np.zeros((10, 20, 3), dtype=np.uint8)
    array[:, 0] = [255, 0, 0]
    array[:, -1] = [0, 255, 0]
    result = np.array(letterbox(Image.fromarray(array), 20))
    assert (result[5:15, 0] == [255, 0, 0]).all()
    assert (result[5:15, -1] == [0, 255, 0]).all()


def test_occluding_batch_augmentation_rejected(originals):
    with pytest.raises(ValueError, match='mixup/cutmix'):
        build_original_loaders(originals, originals, CFG(cutmix_alpha=1))


def test_real_backward_and_spawn_worker(originals):
    cfg = CFG(img_size=32, batch_size=1, num_workers=1, balance_strategy='none')
    from torch.utils.data import DataLoader
    ds = OriginalDataset(originals, cfg, train=True)
    ds[0]  # Ensure an open archive is not serialized to the spawned worker.
    loader = DataLoader(ds, batch_size=1, num_workers=1, multiprocessing_context='spawn')
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 7, 3),
                                torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
    for x, y in loader:
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        assert torch.isfinite(loss) and model[0].weight.grad.abs().sum() > 0
    ds.close()


def test_existing_trainer_accepts_original_loader(originals, tmp_path, monkeypatch):
    from src import train
    rows = pd.concat([originals, originals], ignore_index=True)
    rows['label'] = ['A1', 'A2']
    cfg = CFG(img_size=32, batch_size=2, num_workers=0, epochs=1,
              warmup_epochs=0, amp=False, ema_decay=0, random_erasing=0,
              balance_strategy='none')
    dl_tr, dl_va, ds_tr, ds_va = build_original_loaders(
        rows, rows, cfg, classes=['A1', 'A2'])
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 2, 3),
                                torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
    monkeypatch.setattr(train, 'ckpt_dir', lambda exp: tmp_path)
    result = train.fit(model, dl_tr, dl_va, cfg, ds_train=ds_tr,
                       device='cpu', exp_name='integration_test', persist=False,
                       resume=False, verbose=False)
    assert len(result.history) == 1
    assert (tmp_path / 'best.pt').exists() and (tmp_path / 'last.pt').exists()
    assert json.loads((tmp_path / 'result.json').read_text())['completed']
    ds_tr.close()
    ds_va.close()
