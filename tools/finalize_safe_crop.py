"""새 분할을 고정하고, **학습 행만으로** crop 후보를 감사합니다 (STEP 44)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import pandas as pd

from src.config import CFG
from src.safe_crop import sample_window
from src import split


def finalize(root):
    destination = root / 'manifest.parquet'
    if destination.exists():
        raise FileExistsError('Frozen manifest already exists; refusing to replace it')
    frames = []
    for name in ['TL01', 'TL02', 'VL01']:
        report = json.loads((root / f'{name}_audit.json').read_text())
        if not report['verify_images']:
            raise ValueError('Run preparation with --verify-images before freezing')
        frames.append(pd.read_parquet(root / f'{name}.parquet'))
    df = pd.concat(frames, ignore_index=True)
    if df.sha256.isna().any():
        raise ValueError('Missing verified image hashes')
    input_rows = len(df)
    conflict = df.groupby('sha256').label.nunique()
    conflicts = set(conflict[conflict > 1].index)
    rejected = df[df.sha256.isin(conflicts)]
    rejected.to_parquet(root / 'conflicting_labels.parquet', index=False)
    df = df[~df.sha256.isin(conflicts)].reset_index(drop=True)
    # 원본 기록은 다 남기되 JPEG 바이트가 같은 것만 한 그룹으로 묶습니다.
    # ⚠️ 유사 이미지 중복(pHash)은 여기서 주장하지 않습니다.
    df['dup_cluster'] = pd.factorize(df.sha256, sort=True)[0]
    df = split.assign(df, CFG())
    # 중복 제거는 그룹을 만든 **뒤에** — 개체 ID 의 전이 관계를 살립니다.
    before = len(df)
    df = df.drop_duplicates('sha256').reset_index(drop=True)
    split.verify(df, fold=0, strict=True)
    for fold in range(5):
        tr, va = split.get_fold(df, fold)
        if set(tr.sha256) & set(va.sha256):
            raise AssertionError('Exact duplicate leakage')
        if set(tr.label) != {f'A{i}' for i in range(1, 8)} or set(va.label) != set(tr.label):
            raise ValueError(f'Missing classes in fold {fold}')
    df.to_parquet(destination, index=False)
    report = {'input_rows': input_rows, 'conflicting_label_rows': len(rejected),
              'duplicate_rows_removed': before-len(df), 'final_rows': len(df),
              'holdout_rows': int(df.is_holdout.sum()),
              'label_counts': df.label.value_counts().to_dict(),
              'source_counts': df.source_chunk.value_counts().to_dict(),
              'grouping': 'surrogate animal ID union exact JPEG SHA256; near duplicates not checked',
              'historical_split': False}
    (root / 'split_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return df


def retention(boxes, window):
    x, y, r, b = window
    return min(max(0, min(r, x2)-max(x, x1))*max(0, min(b, y2)-max(y, y1)) /
               ((x2-x1)*(y2-y1)) for x1, y1, x2, y2 in boxes)


def audit(root, df):
    from PIL import Image, ImageDraw
    from torchvision.transforms import RandomResizedCrop
    import torch
    from src.crop import _open_source

    tr, _ = split.get_fold(df, 0)
    sample = tr.groupby('label', group_keys=False).sample(n=200, random_state=42)
    rows = []
    for low in [0.35, 0.5, 0.7]:
        rng = random.Random(42)
        torch.manual_seed(42)
        for row in sample.itertuples():
            boxes = json.loads(row.boxes)
            # get_params 는 이미지 크기만 씁니다 — 원본을 디코딩할 필요가 없습니다.
            canvas = Image.new('L', (row.img_w, row.img_h))
            for repeat in range(5):
                window = sample_window(row.img_w, row.img_h, boxes,
                                       scale=(low, 1.0), rng=rng)
                t, l, h, w = RandomResizedCrop.get_params(canvas, (low, 1.0), (0.85, 1.18))
                for mode, win in [('safe', window), ('random', (l, t, l+w, t+h))]:
                    x, y, r, b = win
                    retained = retention(boxes, win)
                    if mode == 'safe' and retained < 1-1e-9:
                        raise AssertionError('Safe crop removed an annotation')
                    rows.append({'scale_low': low, 'mode': mode, 'label': row.label,
                                 'full_fallback': win == (0, 0, row.img_w, row.img_h),
                                 'area_fraction': (r-x)*(b-y)/(row.img_w*row.img_h),
                                 'min_roi_retention': retained,
                                 'roi_cut': retained < 1-1e-9})
    measures = pd.DataFrame(rows)
    measures.to_parquet(root / 'crop_trials.parquet', index=False)
    summary = measures.groupby(['scale_low', 'mode', 'label']).agg(
        proposals=('roi_cut', 'size'), roi_cut_rate=('roi_cut', 'mean'),
        fallback_rate=('full_fallback', 'mean'), mean_area=('area_fraction', 'mean'),
        min_retention=('min_roi_retention', 'min')).reset_index()
    summary.to_csv(root / 'crop_summary.csv', index=False)
    print(summary.to_string(index=False), flush=True)
    # 클래스당 한 장: 주석이 있는 원본, 그리고 크롭 두 개.
    sheet = Image.new('RGB', (960, 7*210), 'white')
    rng = random.Random(17)
    for i, (_, row) in enumerate(sample.groupby('label').first().iterrows()):
        with _open_source(row) as im:
            original = im.convert('RGB')
        boxes = json.loads(row.boxes)
        annotated = original.copy()
        draw = ImageDraw.Draw(annotated)
        for box in boxes:
            draw.rectangle(box, outline='red', width=5)
        for j in range(3):
            if j == 0:
                pic = annotated.copy()
            else:
                win = sample_window(*original.size, boxes, scale=(0.35, 1.0), rng=rng)
                pic = annotated.crop(win)
            pic.thumbnail((320, 185))
            sheet.paste(pic, (j*320, i*210+20))
        ImageDraw.Draw(sheet).text((5, i*210+3), f'A{i+1}: original | safe crop 1 | safe crop 2', fill='black')
    sheet.save(root / 'preview.jpg')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('data/work/safe_crop_v1'))
    p.add_argument('--audit-only', action='store_true')
    args = p.parse_args()
    df = pd.read_parquet(args.root / 'manifest.parquet') if args.audit_only else finalize(args.root)
    audit(args.root, df)


if __name__ == '__main__':
    main()
