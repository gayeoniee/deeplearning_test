"""검출 학습용 데이터셋 — **전체 원본**(TL01·TL02·VL01, holdout 제외)에서 (STEP 50).

    uv run python tools/make_detect_dataset_full.py --out data/work/detect_full          # 전부
    uv run python tools/make_detect_dataset_full.py --out /tmp/x --limit 200            # 먼저 작게

STEP 42 의 `make_detect_dataset.py` 와 **같은 자극**입니다 — 앱처럼 가까이:

    창 = bbox 긴 변 × U(3.5, 6.0), 최소 480px, 사진 밖으로 안 나감, 위치 무작위
    저장 512×512 JPEG, 정답 네모는 그 창 기준 0~1

다른 점 셋:
  * 원본이 세 청크 다 있으니 **비 holdout 병변 전부**(약 15.7만 장)를 담습니다
  * 분할은 매니페스트의 `fold`(개체 단위) 를 그대로 — **fold 0 = 검증**. 청크로 가르면
    안 됩니다: 같은 개체가 VL01 과 TL01·TL02 에 걸쳐 있습니다 (940 중 ~900)
  * 4 프로세스 병렬 · 행마다 sha256 으로 시드 → 끊겨도 같은 결과로 이어갑니다

출력은 캐글 Dataset 크기에 맞춰 **조각(shard)** 으로 나눕니다. `boxes.parquet` 은
조각마다 복사해 둡니다 (어느 하나만 붙여도 표가 있게).

⚠️ AI Hub 재배포 금지 — 캐글 업로드는 반드시 Private.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import sys
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PX = 512
QUALITY = 88
_ZIPS: dict[str, zipfile.ZipFile] = {}


def largest_box(boxes_json: str):
    """매니페스트 `boxes` 는 병변 하나당 두 번 적혀 있습니다 — 중복을 걷고 가장 큰 것."""
    boxes = {tuple(float(v) for v in b) for b in json.loads(boxes_json)}
    boxes = [b for b in boxes if b[2] > b[0] and b[3] > b[1]]
    if not boxes:
        return None
    return max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))


def geometry(name, img_w, img_h, boxes_json):
    """창과 정답 좌표 — **원본을 안 열고** 계산합니다 (매니페스트의 img_w/img_h 를 믿습니다).

    시드가 행 이름에 박혀 있어 워커와 부모가 같은 값을 냅니다. 그래서 끊겼다가 이어갈 때
    이미 저장된 사진의 좌표를 다시 그리지 않고도 되찾습니다.
    """
    b0 = largest_box(boxes_json)
    if b0 is None:
        return None
    rng = random.Random(int(hashlib.sha256(name.encode()).hexdigest()[:12], 16))
    FW, FH = img_w, img_h
    long0 = max(b0[2]-b0[0], b0[3]-b0[1])
    side = min(max(long0*rng.uniform(3.5, 6.0), 480.0), FW, FH)
    if side < long0*1.6:
        return None
    lo_x, hi_x = max(0.0, b0[2]-side), min(b0[0], FW-side)
    lo_y, hi_y = max(0.0, b0[3]-side), min(b0[1], FH-side)
    if lo_x > hi_x or lo_y > hi_y:
        return None
    wx, wy = rng.uniform(lo_x, hi_x), rng.uniform(lo_y, hi_y)
    t = [(b0[0]-wx)/side, (b0[1]-wy)/side, (b0[2]-wx)/side, (b0[3]-wy)/side]
    if not all(0.0 <= v <= 1.0 for v in t) or t[2] <= t[0] or t[3] <= t[1]:
        return None
    return wx, wy, side, t


def one(task):
    """한 행 → ('ok', name, x1, y1, x2, y2) 또는 None. 워커 프로세스에서 돕니다."""
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True   # 원본에 끝이 잘린 JPEG 가 섞여 있습니다 (캐글 러너와 같은 처방)
    name, zip_path, member, img_w, img_h, boxes_json, out_path = task
    g = geometry(name, img_w, img_h, boxes_json)
    if g is None:
        return None
    wx, wy, side, t = g
    if Path(out_path).exists():
        return ('ok', name, *t)
    if zip_path not in _ZIPS:
        _ZIPS[zip_path] = zipfile.ZipFile(zip_path)
    try:
        blob = _ZIPS[zip_path].read(member)
    except KeyError:
        return None
    with Image.open(io.BytesIO(blob)) as full:
        if full.size != (img_w, img_h):
            raise ValueError(f'Manifest size differs from the image: {member}')
        im = full.convert('RGB').crop((int(wx), int(wy), int(wx+side), int(wy+side))).resize((PX, PX))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path + '.part'
    im.save(tmp, format='JPEG', quality=QUALITY)
    os.replace(tmp, out_path)
    im.close()
    return ('ok', name, *t)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path,
                    default=Path('/Users/gayeon/deeplearning_test/data/work/safe_crop_v1/manifest.parquet'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--shards', type=int, default=3)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--normals', type=int, default=0, help='정상(A7) 창을 이만큼 추가 — 정답 네모 없는 음성 (STEP 52). 병변은 안 담음')
    a = ap.parse_args()

    m = pd.read_parquet(a.manifest)
    if a.normals:
        # ★ 정상(A7)도 라벨 네모가 100% 있습니다 — 병변과 같은 창 규칙을 쓰고, 검출 정답만 비웁니다.
        m = m[(m.label == 'A7') & (~m.is_holdout) & (m.fold >= 0)].copy()
        if a.normals > 0:
            m = m.sample(a.normals, random_state=52).copy()   # -1 이면 전부
    else:
        m = m[(m.label != 'A7') & (~m.is_holdout) & (m.fold >= 0)].copy()
    m = m.sort_values('sha256').reset_index(drop=True)          # 결정론적 순서
    if a.limit:
        m = m.sample(a.limit, random_state=0).reset_index(drop=True)
    m['name'] = m.sha256.str[:16] + '.jpg'
    m['shard'] = (m.index % a.shards).astype(int)
    print(f'■ 후보 {len(m):,}장 · 청크 {m.source_chunk.value_counts().to_dict()} · fold {m.fold.value_counts().sort_index().to_dict()}', flush=True)
    tasks = [(r.name, r.zip_path, r.zip_member, int(r.img_w), int(r.img_h), r.boxes,
              str(a.out / f'shard{r.shard}' / 'images' / r.name)) for r in m.itertuples()]
    t0 = time.perf_counter()
    got = {}
    with Pool(a.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(one, tasks, chunksize=16)):
            if res and res[0] == 'ok':
                got[res[1]] = res[2:]
            if (i+1) % 2000 == 0:
                el = time.perf_counter()-t0
                print(f'   {i+1:,}/{len(tasks):,}  담음 {len(got):,}  {el/60:.1f}분  '
                      f'남은 예상 {el/(i+1)*(len(tasks)-i-1)/60:.0f}분', flush=True)
    rows = m[m.name.isin(got)].copy()
    coords = pd.DataFrame.from_dict(got, orient='index', columns=['x1', 'y1', 'x2', 'y2'])
    rows = rows.join(coords, on='name')
    table = rows[['name', 'shard', 'x1', 'y1', 'x2', 'y2', 'label', 'group', 'fold', 'source_chunk', 'sha256']]
    table = table.rename(columns={'name': 'image'}).reset_index(drop=True)
    a.out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(a.out / 'boxes.parquet', index=False)
    for k in range(a.shards):
        (a.out / f'shard{k}').mkdir(exist_ok=True)
        table.to_parquet(a.out / f'shard{k}' / 'boxes.parquet', index=False)
    report = {'rows': int(len(table)), 'candidates': int(len(m)), 'px': PX, 'quality': QUALITY,
              'stimulus': 'window = bbox long side x U(3.5,6.0), min 480px, random position, resized 512',
              'split': 'manifest fold (group-aware); fold 0 = validation', 'holdout_rows': 0,
              'by_fold': table.fold.value_counts().sort_index().to_dict(),
              'by_chunk': table.source_chunk.value_counts().to_dict(),
              'by_label': table.label.value_counts().to_dict(),
              'groups': int(table.group.nunique()),
              'manifest_sha256': hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
              'elapsed_min': (time.perf_counter()-t0)/60}
    (a.out / 'dataset.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False, indent=1))
    for k in range(a.shards):
        size = sum(p.stat().st_size for p in (a.out / f'shard{k}').rglob('*.jpg')) / 1e9
        print(f'   shard{k}: {int((table.shard == k).sum()):,}장 · {size:.2f} GB')


if __name__ == '__main__':
    main()
