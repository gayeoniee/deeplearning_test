"""원본 ZIP 을 직접 읽는 학습 로더 — full / random / safe 를 같은 조건으로 (STEP 44).

★ **검증은 세 방법 모두 원본 전체 letterbox 로 동일**합니다. 다른 것은 학습
입력뿐이라, 차이가 나면 crop 방식으로 좁혀집니다.

보존 crop 은 리사이즈 **전에** 원본 좌표에서 자릅니다. 그 뒤 변환은 남긴
ROI 를 자르거나 회전·이동·지우지 않습니다.
"""
from __future__ import annotations

import io
import json
import os
import random
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from src.safe_crop import sample_window


def letterbox(image, size):
    """가로세로비를 지키고 화각을 통째로 남깁니다."""
    image = ImageOps.contain(image, (size, size), Image.Resampling.BILINEAR)
    result = Image.new('RGB', (size, size), (128, 128, 128))
    result.paste(image, ((size-image.width)//2, (size-image.height)//2))
    return result


class OriginalDataset(Dataset):
    def __init__(self, df, cfg, *, train=False, mode='safe', classes=None,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                 scale=(0.35, 1.0), ratio=(0.85, 1.18), padding=0.05):
        if mode not in ('safe', 'random', 'full'):
            raise ValueError(f'Unknown crop mode: {mode}')
        self.df = df.reset_index(drop=True).copy()
        self.cfg, self.train, self.mode = cfg, train, mode
        self.scale, self.ratio, self.padding = scale, ratio, padding
        self.classes = classes or [f'A{i}' for i in range(1, 8)]
        mapping = {c: i for i, c in enumerate(self.classes)}
        if not self.df.label.isin(self.classes).all():
            raise ValueError('Unknown labels in original dataset')
        self.targets = self.df.label.map(mapping).to_numpy(dtype=np.int64)
        # 로짓 지문이 쓰는 경로에 전처리 버전을 넣습니다 — 설정이 다른 실행에 이어붙지 않게.
        self.paths = [f'original-letterbox-v1:{p}' for p in self.df.image_path]
        self._archives = {}
        self._pid = os.getpid()
        self.normalize = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        self.color = T.ColorJitter(brightness=cfg.color_jitter,
                                   contrast=cfg.color_jitter,
                                   saturation=cfg.color_jitter*0.5,
                                   hue=cfg.hue_jitter)

    def __len__(self):
        return len(self.df)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_archives'] = {}
        return state

    def close(self):
        for z in self._archives.values():
            z.close()
        self._archives = {}

    def __getitem__(self, index):
        # ZIP 핸들은 seek 를 쓰므로 워커끼리 **절대** 공유하면 안 됩니다.
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        row = self.df.iloc[index]
        path = row.zip_path
        if isinstance(path, str) and path:
            if path not in self._archives:
                self._archives[path] = zipfile.ZipFile(path)
            raw = self._archives[path].read(row.zip_member)
        else:
            raw = Path(row.image_path).read_bytes()
        with Image.open(io.BytesIO(raw)) as im:
            image = im.convert('RGB')
        if image.size != (row.img_w, row.img_h):
            raise ValueError(f'Original dimensions changed: {row.image_path}')
        image = self.prepare_image(image, row)
        if self.train:
            if self.mode == 'safe':
                boxes = json.loads(row.boxes) if isinstance(row.boxes, str) else row.boxes
                image = image.crop(sample_window(*image.size, boxes, scale=self.scale,
                                                ratio=self.ratio, padding=self.padding))
            elif self.mode == 'random':
                top, left, h, w = T.RandomResizedCrop.get_params(image, self.scale, self.ratio)
                image = image.crop((left, top, left+w, top+h))
            if random.random() < self.cfg.hflip:
                image = ImageOps.mirror(image)
            if random.random() < self.cfg.vflip:
                image = ImageOps.flip(image)
            image = self.color(image)
        x = self.normalize(letterbox(image, self.cfg.img_size))
        return x, int(self.targets[index])

    def prepare_image(self, image, row):
        return image


def build_original_loaders(train_df, val_df, cfg, model=None, *, mode='safe',
                           classes=None, scale=(0.35, 1.0), padding=0.05):
    """`src.train.fit` 에 그대로 물립니다. 배치 가림(occlusion)은 명시적으로 거부합니다."""
    if cfg.mixup_alpha or cfg.cutmix_alpha:
        raise ValueError('Original crop comparison requires mixup/cutmix disabled')
    if cfg.balance_strategy not in ('none', 'class_weight', 'weighted_sampler'):
        raise ValueError('Unsupported original-image sampler')
    pretrained = getattr(model, 'pretrained_cfg', {}) or {}
    common = dict(classes=classes, mean=pretrained.get('mean', (0.485, 0.456, 0.406)),
                  std=pretrained.get('std', (0.229, 0.224, 0.225)),
                  mode=mode, scale=scale, padding=padding)
    ds_tr = OriginalDataset(train_df, cfg, train=True, **common)
    ds_va = OriginalDataset(val_df, cfg, train=False, **common)
    sampler = None
    if cfg.balance_strategy == 'weighted_sampler':
        from src.data import weighted_sampler
        sampler = weighted_sampler(ds_tr)
    nw = cfg.resolved_num_workers()
    options = dict(num_workers=nw, pin_memory=torch.cuda.is_available(),
                   persistent_workers=nw > 0)
    # RAM 에 쌓이는 배치를 제한합니다 — 워커마다 ZIP 목차까지 물고 있습니다.
    if nw:
        options['prefetch_factor'] = 2
    bs = cfg.resolved_batch_size()
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=sampler is None,
                       sampler=sampler, drop_last=False, **options)
    dl_va = DataLoader(ds_va, batch_size=bs, shuffle=False, **options)
    return dl_tr, dl_va, ds_tr, ds_va
