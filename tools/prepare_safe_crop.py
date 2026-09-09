"""원본 아카이브를 감사하고 **격리된** 전체-주석 매니페스트를 만듭니다 (STEP 44).

    python -m tools.prepare_safe_crop [--verify-images]

기존 크롭·매니페스트는 **건드리지 않습니다**. 의심스러운 주석은 지우지 않고
격리해 `*_audit.json` 에 남깁니다 — 이름 체계가 바뀐 것일 수도 있어서
라벨이 틀렸다고 단정하지 않습니다.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
import random
import zipfile

import pandas as pd
from PIL import Image

from src.labels import parse_record_561
from src.safe_crop import annotation_boxes, sample_window


def prepare(archive, out, verify_images=False):
    target = out / f"{archive.stem}.parquet"
    report_path = out / f"{archive.stem}_audit.json"
    if target.exists() or report_path.exists():
        raise FileExistsError(f"Output exists: {target}; use another --out directory")
    counts = Counter()
    rows, issues = [], []
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        names = {i.filename for i in infos}
        if len(names) != len(infos):
            raise ValueError(f"Duplicate ZIP member names: {archive}")
        counts.update({"jpg_members": sum(n.lower().endswith('.jpg') for n in names),
                       "json_members": sum(n.lower().endswith('.json') for n in names)})
        for info in infos:
            n = info.filename
            if not n.lower().endswith('.json'):
                continue
            counts['json_processed'] += 1
            if counts['json_processed'] % 10000 == 0:
                print(archive.stem, dict(counts), flush=True)
            try:
                rec = json.loads(z.read(info).decode('utf-8-sig'))
                if not isinstance(rec, dict):
                    raise ValueError('Expected one JSON record')
                meta = rec.get('metaData', {})
                if meta.get('species') != 'D' or '/일반카메라/' not in n:
                    counts['outside_scope'] += 1
                    continue
                parsed = parse_record_561(rec, n)
                if parsed is None or parsed['label'] not in [f'A{k}' for k in range(1, 8)]:
                    raise ValueError('Missing or invalid label')
                if parsed['synthetic']:
                    counts['synthetic_excluded'] += 1
                    continue
                member = str(Path(n).with_suffix('.jpg'))
                if member not in names:
                    raise ValueError('No same-directory JPG for JSON')
                if Path(str(meta.get('Raw data ID', ''))).name != Path(member).name:
                    raise ValueError('JSON image ID does not match JPG')
                w, h = parsed['img_w'], parsed['img_h']
                if not w or not h:
                    raise ValueError('Missing image dimensions')
                boxes = annotation_boxes(rec)
                # 전체 사진으로 물러설 사진이라도 box 는 전부 검사합니다.
                sample_window(w, h, boxes, rng=random.Random(0), attempts=1)
                if not boxes:
                    raise ValueError('Missing ROI annotations (including normal ROI)')
                if any(not str(meta.get(k, '')).strip() for k in
                       ('breed', 'age', 'gender', 'date')):
                    raise ValueError('Incomplete surrogate grouping metadata')
                payload_hash = None
                if verify_images:
                    raw = z.read(member)  # ZipFile verifies CRC for each read.
                    payload_hash = hashlib.sha256(raw).hexdigest()
                    with Image.open(io.BytesIO(raw)) as im:
                        if im.size != (w, h):
                            raise ValueError('Image dimensions differ from JSON')
                        im.verify()
                    # ⚠️ JPEG 헤더·ZIP CRC 검사는 픽셀까지 안 봅니다 — 손상 5장이 여기를 빠져나갔습니다.
                    with Image.open(io.BytesIO(raw)) as im:
                        im.convert('RGB').load()
                    counts['images_verified'] += 1
                rows.append({
                    'image_path': f'{archive.resolve()}!{member}',
                    'zip_path': str(archive.resolve()), 'zip_member': member,
                    'json_member': n, 'source_chunk': archive.stem,
                    'label': parsed['label'], 'animal_id': parsed['animal_id'],
                    'img_w': w, 'img_h': h, 'boxes': json.dumps(boxes),
                    'sha256': payload_hash,
                })
                counts['accepted'] += 1
                counts[f'label_{parsed["label"]}'] += 1
            except (ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile) as exc:
                counts['quarantined'] += 1
                issues.append({'member': n, 'error': f'{type(exc).__name__}: {exc}'})
    pd.DataFrame(rows).to_parquet(target, index=False)
    report_path.write_text(json.dumps({
        'archive': str(archive.resolve()), 'archive_bytes': archive.stat().st_size,
        'verify_images': verify_images, 'counts': dict(counts), 'issues': issues,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print('DONE', archive.stem, dict(counts), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, default=Path('data/raw'))
    p.add_argument('--out', type=Path, default=Path('data/work/safe_crop_v1'))
    p.add_argument('--verify-images', action='store_true')
    p.add_argument('--chunk', choices=['TL01', 'TL02', 'VL01'])
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ([args.chunk] if args.chunk else ['TL01', 'TL02', 'VL01']):
        archives = list(args.raw.rglob(name + '.zip'))
        if len(archives) != 1:
            raise ValueError(f'Expected exactly one {name}.zip; found {len(archives)}')
        prepare(archives[0], args.out, args.verify_images)


if __name__ == '__main__':
    main()
