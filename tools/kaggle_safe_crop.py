"""시간 예산이 있는 1단계 짝 비교 — epoch 경계에서 이어받습니다 (STEP 44).

    python tools/kaggle_safe_crop.py --data DATASET_ROOT --out OUTPUT_DIR

이 실행기와 `src/` 스냅샷은 파일럿 데이터에 **같이 담겨** 나갑니다 —
캐글에서 git clone 도 API 키도 필요 없게.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG
from src.original_data import OriginalDataset


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + '.part')
    torch.save(value, temporary)
    temporary.replace(path)


def load_checkpoint(path, device='cpu'):
    return torch.load(path, map_location=device, weights_only=True)


def load_data(root):
    metadata = json.loads((root / 'pilot_package.json').read_text())
    raw = (root / 'pilot_manifest.parquet').read_bytes()
    if hashlib.sha256(raw).hexdigest() != metadata['pilot_manifest_sha256']:
        raise ValueError('Pilot manifest differs from the packaged version')
    for name, digest in metadata['code_sha256'].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Packaged source changed: {name}')
    df = pd.read_parquet(root / 'pilot_manifest.parquet')
    if not df.pilot_split.isin(['train', 'val']).all():
        raise ValueError('Unexpected split: holdout must not be included')
    for member in df.image_path:
        relative = Path(member)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Nonportable image path')
    df['image_path'] = df.image_path.map(lambda p: str(root / p))
    df['original_label'] = df.label
    df['label'] = df.label.where(df.label == 'A7', 'ABNORMAL')
    tr, va = (df[df.pilot_split == name].reset_index(drop=True) for name in ['train', 'val'])
    for col in ['group', 'animal_id', 'sha256']:
        if set(tr[col]) & set(va[col]):
            raise ValueError(f'Train/validation overlap in {col}')
    if len(tr) != metadata['train_rows'] or len(va) != metadata['val_rows']:
        raise ValueError('Pilot row counts changed')
    return tr, va, metadata


def make_model(smoke, pretrained=False, profile="original"):
    if profile == "roi" and not smoke:
        import timm
        return timm.create_model("tf_efficientnetv2_s.in21k_ft_in1k", pretrained=pretrained,
                                 num_classes=2, drop_rate=0.2, drop_path_rate=0.1)
    if smoke:
        return torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, stride=2),
                                   torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1),
                                   torch.nn.Flatten(), torch.nn.Linear(8, 2))
    from torchvision.models import resnet50, ResNet50_Weights
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    return model


def metrics(probability, target, original_labels):
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(target, prediction, labels=[0, 1]).ravel()
    result = {'macro_f1': float(f1_score(target, prediction, labels=[0, 1], average='macro', zero_division=0)),
              'balanced_accuracy': float(balanced_accuracy_score(target, prediction)),
              'abnormal_recall': float(tp/max(tp+fn, 1)),
              'normal_specificity': float(tn/max(tn+fp, 1)),
              'auroc': float(roc_auc_score(target, probability)) if len(set(target)) == 2 else None}
    for label in sorted(set(original_labels)):
        mask = original_labels == label
        result[f'{label}_correct_rate'] = float((prediction[mask] == target[mask]).mean())
    return result


def one_epoch(model, optimizer, scaler, tr, va, cfg, mode, epoch, device,
              deadline, max_train_batches=None):
    """마감을 넘기면 None — 부르는 쪽이 직전에 커밋된 epoch 를 그대로 유지합니다."""
    seed_everything(cfg.seed + epoch)
    model.train()
    dataset = OriginalDataset
    if getattr(cfg, 'experiment_profile', 'original') == 'roi':
        from src.roi_data import ROIDataset
        dataset = ROIDataset
    ds_tr = dataset(tr, cfg, train=True, mode=mode, classes=['A7', 'ABNORMAL'])
    ds_va = dataset(va, cfg, train=False, mode=mode, classes=['A7', 'ABNORMAL'])
    generator = torch.Generator().manual_seed(cfg.seed + epoch)
    options = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                   pin_memory=device == 'cuda')
    if cfg.num_workers:
        options['prefetch_factor'] = 2
    dl_tr = DataLoader(ds_tr, shuffle=True, generator=generator, **options)
    dl_va = DataLoader(ds_va, shuffle=False, **options)
    counts = np.bincount(ds_tr.targets, minlength=2)
    weights = torch.tensor(len(ds_tr)/(2*np.maximum(counts, 1)), dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    started = time.monotonic()
    losses, trained = 0.0, 0
    try:
        for batch, (x, y) in enumerate(dl_tr):
            if time.monotonic() >= deadline:
                return None
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=device == 'cuda'):
                loss = criterion(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses += float(loss.detach()) * len(y)
            trained += len(y)
            if (batch+1) % 100 == 0:
                print(f'{mode} epoch {epoch+1}: {trained}/{len(ds_tr)} photos', flush=True)
            if max_train_batches and batch+1 >= max_train_batches:
                break
        model.eval()
        probabilities, targets = [], []
        with torch.no_grad():
            for x, y in dl_va:
                if time.monotonic() >= deadline:
                    return None
                with torch.autocast(device_type=device, enabled=device == 'cuda'):
                    logits = model(x.to(device))
                probabilities.extend(logits.float().softmax(1)[:, 1].cpu().tolist())
                targets.extend(y.tolist())
        if device == 'cuda':
            torch.cuda.synchronize()
        result = metrics(np.asarray(probabilities), np.asarray(targets), va.original_label.to_numpy())
        result.update(epoch=epoch+1, mode=mode, train_loss=losses/max(trained, 1),
                      elapsed_sec=time.monotonic()-started, train_rows=trained, val_rows=len(va))
        return result, np.asarray(probabilities)
    finally:
        ds_tr.close()
        ds_va.close()


def write_comparison(output, histories):
    records = [r for history in histories.values() for r in history]
    table = pd.DataFrame(records) if records else pd.DataFrame(columns=['mode', 'epoch', 'macro_f1'])
    table.to_csv(output / 'history.csv', index=False)
    baseline_mode = next(m for m in histories if m != 'safe')
    common = min(len(histories[baseline_mode]), len(histories['safe']))
    summary = {'completed_epochs': {k: len(v) for k, v in histories.items()},
               'common_epoch': common, 'comparison': None,
               'note': 'Exploratory stage-1 pilot; compare only the same epoch. No holdout evaluation.'}
    if common:
        baseline, safe = (histories[m][common-1] for m in [baseline_mode, 'safe'])
        summary['comparison'] = {key: {baseline_mode: baseline[key], 'safe': safe[key],
                                       f'safe_minus_{baseline_mode}': safe[key]-baseline[key]}
                                 for key in ['macro_f1', 'abnormal_recall', 'normal_specificity', 'balanced_accuracy']}
    (output / 'comparison.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def export_resume(output):
    # 사진과 데이터셋은 캐글 출력에 안 넣습니다 — 작은 실행 산출물만.
    profile = json.loads((output / 'protocol.json').read_text()).get('profile', 'original')
    archive = output.parent / ('roi_crop_pilot_resume.zip' if profile == 'roi' else 'safe_crop_pilot_resume.zip')
    temporary = archive.with_suffix('.zip.part')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_STORED) as z:
        for path in sorted(output.rglob('*')):
            if path.is_file() and not path.name.endswith('.part'):
                z.write(path, path.relative_to(output))
    temporary.replace(archive)
    return archive


def run(args):
    started = time.monotonic()
    profile = getattr(args, 'profile', 'original')
    modes = ['fixed', 'safe'] if profile == 'roi' else ['random', 'safe']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda' and not args.smoke:
        raise RuntimeError('Select a GPU accelerator; use --smoke only for local tests')
    if not (0 < args.hours <= 7.5) or not (1 <= args.epochs <= 5):
        raise ValueError('Pilot budget: 0 < hours <= 7.5 and 1 <= epochs <= 5')
    args.out.mkdir(parents=True, exist_ok=True)
    tr, va, metadata = load_data(args.data)
    if args.smoke:
        tr = tr.groupby('original_label', group_keys=False).head(2).reset_index(drop=True)
        va = va.groupby('original_label', group_keys=False).head(1).reset_index(drop=True)
    cfg = CFG(seed=42, img_size=32 if args.smoke else (384 if profile == 'roi' else 288), batch_size=args.batch_size,
              num_workers=args.workers, epochs=args.epochs, lr=3e-4,
              rotate_deg=0, random_erasing=0, amp=device == 'cuda')
    protocol = {'version': 1, 'manifest_sha256': metadata['pilot_manifest_sha256'],
                'code_sha256': metadata['code_sha256'], 'smoke_only': args.smoke,
                'model': 'tiny' if args.smoke else 'torchvision_resnet50_IMAGENET1K_V2',
                'cfg': cfg.to_dict(), 'crop_scale': [0.35, 1.0], 'padding': 0.05,
                'validation': 'full-frame letterbox', 'task': 'stage1',
                'schedule': 'paired alternating epochs; fixed cosine horizon',
                'torch_version': str(torch.__version__)}
    if profile == 'roi':
        import timm
        from PIL import ImageFile
        cfg.model_name = 'tf_efficientnetv2_s.in21k_ft_in1k'
        cfg.experiment_profile = 'roi'
        protocol.update(profile='roi', version=2,
                        cfg=cfg.to_dict(),
                        load_truncated_images=ImageFile.LOAD_TRUNCATED_IMAGES,
                        model='tiny' if args.smoke else 'tf_efficientnetv2_s.in21k_ft_in1k',
                        timm_version=timm.__version__, validation='fixed lesion ROI letterbox',
                        crop_scale=None, roi_min_side=320, roi_side_multiplier=[1.0, 1.25],
                        modes=modes, augmentation='shared flip/color only; no rotation/erasing',
                        missing_boxes='full frame in both arms',
                        runtime_code_sha256={name: hashlib.sha256(
                            (Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
                            for name in ['tools/kaggle_safe_crop.py', 'src/roi_data.py',
                                         'src/original_data.py', 'src/safe_crop.py', 'src/config.py']})
    protocol_path = args.out / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != json.loads(json.dumps(protocol)):
            raise ValueError('Resume protocol differs; keep the same data, settings and source snapshot')
    else:
        if any(args.out.iterdir()):
            raise ValueError('Nonempty output directory has no protocol')
        protocol_path.write_text(json.dumps(protocol, indent=2))
    initial_path = args.out / 'initial.pt'
    if not initial_path.exists():
        if any(args.out.glob('*_last.pt')):
            raise ValueError('Missing common initialization in resumed run')
        seed_everything(cfg.seed)
        initial = make_model(args.smoke, pretrained=not args.smoke, profile=profile)
        atomic_save(initial.state_dict(), initial_path)
        del initial
    initial_digest = hashlib.sha256(initial_path.read_bytes()).hexdigest()
    histories = {}
    for mode in modes:
        last = args.out / f'{mode}_last.pt'
        state = load_checkpoint(last) if last.exists() else None
        if state and state['initial_sha256'] != initial_digest:
            raise ValueError('Initialization mismatch')
        histories[mode] = state['history'] if state else []
    # 체크포인트·내보내기에 10분을 남깁니다. --hours 는 이번 호출의 벽시계 시간입니다.
    reserve = min(600, args.hours*3600*0.1)
    deadline = started + args.hours*3600 - reserve
    reason = 'completed'
    try:
        for epoch in range(args.epochs):
            order = modes if epoch % 2 == 0 else list(reversed(modes))
            for mode in order:
                if len(histories[mode]) > epoch:
                    continue
                if len(histories[mode]) != epoch:
                    raise ValueError('Noncontiguous checkpoint history')
                observed = [r['elapsed_sec'] for h in histories.values() for r in h]
                estimate = max(observed[-4:]) * 1.35 if observed else 0
                pending_in_pair = sum(len(history) <= epoch for history in histories.values())
                if time.monotonic() + estimate * pending_in_pair >= deadline:
                    reason = 'time_budget_before_epoch'
                    return write_comparison(args.out, histories)
                model = make_model(args.smoke, profile=profile).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
                scaler = torch.amp.GradScaler('cuda', enabled=device == 'cuda')
                last = args.out / f'{mode}_last.pt'
                if last.exists():
                    state = load_checkpoint(last, device)
                    model.load_state_dict(state['model'])
                    optimizer.load_state_dict(state['optimizer'])
                    scaler.load_state_dict(state['scaler'])
                    del state
                else:
                    model.load_state_dict(load_checkpoint(initial_path))
                # epoch 로 색인한 스케줄이라 중단된 epoch 를 다시 돌려도 결정론적입니다.
                lr = cfg.lr * (0.1 + 0.9*(1+math.cos(math.pi*epoch/cfg.epochs))/2)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                result = one_epoch(model, optimizer, scaler, tr, va, cfg, mode, epoch,
                                   device, deadline)
                if result is None:
                    reason = 'time_budget_during_epoch_replay_required'
                    return write_comparison(args.out, histories)
                record, probabilities = result
                histories[mode].append(record)
                atomic_save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                             'scaler': scaler.state_dict(), 'history': histories[mode],
                             'initial_sha256': initial_digest}, last)
                if record['macro_f1'] >= max(r['macro_f1'] for r in histories[mode]):
                    atomic_save({'model': model.state_dict(), 'epoch': epoch+1,
                                 'initial_sha256': initial_digest}, args.out / f'{mode}_best.pt')
                # epoch 마다 작은 예측을 남겨야 이어받은 뒤에도 **공통 epoch** 비교가 됩니다.
                np.savez_compressed(args.out / f'{mode}_epoch{epoch+1}_val.npz',
                                    probability=probabilities, sha256=va.sha256.to_numpy(dtype=str),
                                    original_label=va.original_label.to_numpy(dtype=str))
                print(json.dumps(record, indent=2), flush=True)
                print(f'Measured epoch {record["elapsed_sec"]/60:.1f} min; '
                      f'10 epochs at this rate ≈ {record["elapsed_sec"]*10/3600:.2f} h (estimate)', flush=True)
                write_comparison(args.out, histories)
                del model, optimizer, scaler
                if device == 'cuda':
                    torch.cuda.empty_cache()
        return write_comparison(args.out, histories)
    except BaseException:
        reason = 'error_or_interruption'
        raise
    finally:
        (args.out / 'session.json').write_text(json.dumps({
            'stop_reason': reason, 'elapsed_sec': time.monotonic()-started,
            'budget_hours': args.hours, 'device': device, 'smoke_only': args.smoke,
            'initial_sha256': initial_digest}, indent=2))
        print('Resume bundle:', export_resume(args.out), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--hours', type=float, default=7.0)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--profile', choices=['original', 'roi'], default='original')
    parser.add_argument('--smoke', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
