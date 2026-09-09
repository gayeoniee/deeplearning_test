"""원본 사진 짝 비교 — 기존 학습기에 그대로 물립니다 (STEP 44).

    python -m tools.train_safe_crop --mode safe   --task stage1
    python -m tools.train_safe_crop --mode random --task stage1
    python -m tools.train_safe_crop --mode full   --task stage1

세 실행이 **고정된 새 분할**과 원본 전체 검증을 공유합니다.
⚠️ 과거 크롭 지표와 **비교 금지** — 분할이 다릅니다.
⚠️ `--smoke` 는 입출력과 역전파만 봅니다. 모델 품질이 아닙니다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from src import env, split
from src.config import CFG, CLASSES, CLASSES_STAGE1
from src.original_data import build_original_loaders


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('data/work/safe_crop_v1'))
    p.add_argument('--raw', type=Path, help='Rebase archives on another machine')
    p.add_argument('--mode', choices=['safe', 'random', 'full'], default='safe')
    p.add_argument('--task', choices=['stage1', 'stage2', 'seven'], default='stage1')
    p.add_argument('--model', default='resnet50')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--scale-low', type=float, default=0.35)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    env.set_seed(args.seed)
    path = args.root / 'manifest.parquet'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    df = pd.read_parquet(path)
    if args.raw:
        locations = {}
        for chunk in df.source_chunk.unique():
            matches = list(args.raw.rglob(f'{chunk}.zip'))
            if len(matches) != 1:
                raise ValueError(f'Expected one {chunk}.zip below --raw')
            locations[chunk] = str(matches[0].resolve())
        df['zip_path'] = df.source_chunk.map(locations)
    if args.task == 'stage1':
        classes = CLASSES_STAGE1
        df['label'] = df.label.where(df.label == 'A7', 'ABNORMAL')
    elif args.task == 'stage2':
        df = df[df.label != 'A7'].copy()
        classes = CLASSES
    else:
        classes = [f'A{i}' for i in range(1, 8)]
    tr, va = split.get_fold(df, 0)
    if args.smoke:
        tr = tr.groupby('label', group_keys=False).head(4)
        va = va.groupby('label', group_keys=False).head(2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not args.smoke and device == 'cpu':
        raise RuntimeError('Full training requires CUDA here; use --smoke for local verification')
    model_key = ''.join(c if c.isalnum() else '_' for c in args.model)
    name = f'original_v1_{args.task}_{args.mode}_{model_key}_s{args.seed}_a{args.scale_low}_{digest[:10]}'
    cfg = CFG(model_name=args.model, seed=args.seed, epochs=args.epochs,
              batch_size=args.batch_size, num_workers=args.workers,
              img_size=64 if args.smoke else 288, exp_name=name,
              rrc_scale=(args.scale_low, 1.0), rotate_deg=0,
              random_erasing=0, mixup_alpha=0, cutmix_alpha=0,
              amp=device == 'cuda')
    if args.smoke:
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, stride=2),
                                    torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1),
                                    torch.nn.Flatten(), torch.nn.Linear(8, len(classes)))
    else:
        from src import models
        model = models.build(args.model, len(classes), img_size=cfg.img_size)
    dl_tr, dl_va, ds_tr, ds_va = build_original_loaders(
        tr, va, cfg, model, mode=args.mode, classes=classes, scale=cfg.rrc_scale)
    try:
        if args.smoke:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            model.train()
            losses = []
            for x, y in dl_tr:
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(model(x), y)
                if not torch.isfinite(loss):
                    raise AssertionError('Nonfinite loss')
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            model.eval()
            with torch.no_grad():
                for x, y in dl_va:
                    assert torch.isfinite(model(x)).all()
            result = {'smoke_only': True, 'mode': args.mode, 'task': args.task,
                      'train_rows': len(ds_tr), 'val_rows': len(ds_va),
                      'batches': len(losses), 'finite_losses': losses,
                      'manifest_sha256': digest, 'device': device}
            (args.root / f'smoke_{args.task}_{args.mode}.json').write_text(json.dumps(result, indent=2))
            print(json.dumps(result, indent=2))
        else:
            from src import train
            provenance = {'manifest_sha256': digest, 'mode': args.mode,
                          'scale': cfg.rrc_scale, 'ratio': [0.85, 1.18],
                          'padding': 0.05, 'task': args.task,
                          'validation': 'original full-frame letterbox', 'cfg': cfg.to_dict()}
            source_root = Path(__file__).resolve().parents[1] / 'src'
            provenance['preprocessing_sha256'] = hashlib.sha256(
                (source_root / 'safe_crop.py').read_bytes() +
                (source_root / 'original_data.py').read_bytes()).hexdigest()
            output = train.ckpt_dir(name) / 'original_crop_protocol.json'
            if output.exists() and json.loads(output.read_text()) != json.loads(json.dumps(provenance)):
                raise ValueError('Existing run uses different protocol; choose a new run configuration')
            output.write_text(json.dumps(provenance, indent=2))
            train.fit(model, dl_tr, dl_va, cfg, ds_train=ds_tr,
                      exp_name=name, device=device, persist=False)
    finally:
        ds_tr.close()
        ds_va.close()


if __name__ == '__main__':
    main()
