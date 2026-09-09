"""Build a portable, byte-preserving Kaggle pilot; never include holdout."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from PIL import Image


def stratified_sample(df, size, seed):
    counts = df.label.value_counts().sort_index()
    if size < len(counts) or size > len(df):
        raise ValueError('Sample size must cover all classes and fit in its split')
    # Largest remainder allocation preserves original class proportions.
    expected = counts * size / len(df)
    quotas = np.floor(expected).astype(int)
    for label in (expected-quotas).sort_values(ascending=False, kind='stable').index[:size-int(quotas.sum())]:
        quotas[label] += 1
    if (quotas == 0).any():
        raise ValueError('Too few samples to represent every original class')
    parts = [df[df.label == label].sample(n=int(n), random_state=seed)
             for label, n in quotas.items()]
    return pd.concat(parts).sort_values('sha256').reset_index(drop=True)


def select_pilot(df, train_size=20000, val_size=4000, seed=42):
    dev = df[~df.is_holdout]
    tr = stratified_sample(dev[dev.fold != 0], train_size, seed)
    va = stratified_sample(dev[dev.fold == 0], val_size, seed)
    for column in ['group', 'animal_id', 'sha256']:
        if set(tr[column]) & set(va[column]):
            raise ValueError(f'Train/validation overlap: {column}')
    return pd.concat([tr.assign(pilot_split='train'), va.assign(pilot_split='val')], ignore_index=True)


def package(manifest, destination, train_size=20000, val_size=4000):
    if destination.exists() or destination.with_suffix('.zip.part').exists():
        raise FileExistsError(f'Refusing to overwrite {destination} or its partial file')
    selected = select_pilot(pd.read_parquet(manifest), train_size, val_size)
    portable = selected[['sha256', 'label', 'group', 'animal_id', 'img_w', 'img_h',
                         'boxes', 'source_chunk', 'pilot_split']].copy()
    portable['source_member'] = selected.zip_member
    portable['image_path'] = 'images/' + portable.sha256 + '.jpg'
    portable['zip_path'] = None
    portable['zip_member'] = None
    report = {'schema_version': 1, 'seed': 42,
              'parent_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
              'train_rows': train_size, 'val_rows': val_size, 'holdout_rows': 0,
              'selection': 'proportional A1-A7 sampling within frozen fold 0 train/val',
              'image_policy': 'original JPEG bytes, no resize or recompression',
              'counts': pd.crosstab(portable.pilot_split, portable.label).to_dict(),
              'train_groups': int(portable[portable.pilot_split == 'train'].group.nunique()),
              'val_groups': int(portable[portable.pilot_split == 'val'].group.nunique())}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.zip.part')
    total_bytes = 0
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED) as output:
        done = 0
        for archive, records in selected.groupby('zip_path'):
            with zipfile.ZipFile(archive) as source:
                for row in records.itertuples():
                    raw = source.read(row.zip_member)
                    if hashlib.sha256(raw).hexdigest() != row.sha256:
                        raise ValueError(f'Original JPEG changed: {row.zip_member}')
                    try:
                        with Image.open(io.BytesIO(raw)) as image:
                            image.convert('RGB').load()
                    except OSError as exc:
                        raise ValueError(f'JPEG decode failed: {row.zip_member}: {exc}') from exc
                    output.writestr(f'images/{row.sha256}.jpg', raw)
                    output.writestr(f'annotations/{row.sha256}.json', source.read(row.json_member))
                    total_bytes += len(raw)
                    done += 1
                    if done % 2000 == 0:
                        print(f'{done:,}/{len(selected):,} photos; {total_bytes/1024**3:.2f} GiB', flush=True)
        buffer = io.BytesIO()
        portable.to_parquet(buffer, index=False)
        manifest_bytes = buffer.getvalue()
        output.writestr('pilot_manifest.parquet', manifest_bytes)
        report['pilot_manifest_sha256'] = hashlib.sha256(manifest_bytes).hexdigest()
        report['jpeg_bytes'] = total_bytes
        repo = Path(__file__).resolve().parents[1]
        files = sorted((repo / 'src').glob('*.py')) + [repo / 'tools/kaggle_safe_crop.py']
        report['code_sha256'] = {}
        for path in files:
            name = path.relative_to(repo).as_posix()
            content = path.read_bytes()
            output.writestr(name, content)
            report['code_sha256'][name] = hashlib.sha256(content).hexdigest()
        output.writestr('pilot_package.json', json.dumps(report, ensure_ascii=False, indent=2))
        output.writestr('README.md', (repo / 'docs/kaggle_safe_crop_pilot.md').read_bytes())
    # Read the completed ZIP once to check every output entry's CRC.
    with zipfile.ZipFile(temporary) as output:
        bad = output.testzip()
        if bad:
            raise ValueError(f'Output CRC failure: {bad}')
    temporary.rename(destination)
    destination.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'COMPLETE {destination} ({destination.stat().st_size/1024**3:.2f} GiB)', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('data/work/safe_crop_v1/manifest.parquet'))
    parser.add_argument('--out', type=Path, default=Path('data/packages/dogskin_safe_crop_pilot.zip'))
    args = parser.parse_args()
    package(args.manifest, args.out)


if __name__ == '__main__':
    main()
