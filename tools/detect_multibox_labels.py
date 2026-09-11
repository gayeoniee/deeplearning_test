"""STEP 51 — 이미 뽑아둔 검출 창(512px) 에 **창 안의 모든 병변 네모**를 라벨로 붙입니다.

    uv run python tools/detect_multibox_labels.py --out data/work/detect_full/boxes_multi.parquet

`make_detect_dataset_full.py` 의 창은 행 이름으로 시드가 박혀 있어 **원본을 안 열고** 같은 창을
되찾을 수 있습니다. 그 창에 매니페스트의 모든 네모(중복 제거)를 투영해, 80% 이상 창 안에 드는
것을 남기고 창 경계로 자릅니다. object detection 학습용 — STEP 42·50 은 가장 큰 것 하나만 썼습니다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.make_detect_dataset_full import geometry  # noqa: E402

MIN_INSIDE = 0.8


def boxes_in_window(boxes_json, wx, wy, side):
    out = []
    for b in {tuple(float(v) for v in x) for x in json.loads(boxes_json)}:
        if b[2] <= b[0] or b[3] <= b[1]:
            continue
        area = (b[2]-b[0])*(b[3]-b[1])
        cx1, cy1 = max(b[0], wx), max(b[1], wy)
        cx2, cy2 = min(b[2], wx+side), min(b[3], wy+side)
        if cx2 <= cx1 or cy2 <= cy1 or (cx2-cx1)*(cy2-cy1) < MIN_INSIDE*area:
            continue
        out.append([(cx1-wx)/side, (cy1-wy)/side, (cx2-wx)/side, (cy2-wy)/side])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, default=Path('/Users/gayeon/deeplearning_test/data/work/safe_crop_v1/manifest.parquet'))
    ap.add_argument('--boxes', type=Path, default=Path('/Users/gayeon/deeplearning_test/data/work/detect_full/boxes.parquet'))
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    table = pd.read_parquet(a.boxes)
    m = pd.read_parquet(a.manifest).set_index('sha256')
    rows = []
    for r in table.itertuples():
        src = m.loc[r.sha256]
        g = geometry(r.image, int(src.img_w), int(src.img_h), src.boxes)
        assert g is not None
        wx, wy, side, t = g
        multi = boxes_in_window(src.boxes, wx, wy, side)
        assert any(abs(b[0]-t[0]) < 1e-6 and abs(b[3]-t[3]) < 1e-6 for b in multi), r.image   # 가장 큰 것은 반드시 포함
        rows.append({'image': r.image, 'shard': r.shard, 'boxes': json.dumps(multi), 'n_boxes': len(multi),
                     'label': r.label, 'group': r.group, 'fold': r.fold, 'source_chunk': r.source_chunk})
    out = pd.DataFrame(rows)
    out.to_parquet(a.out, index=False)
    print(f'{len(out):,}행 · 네모 수 분포 {out.n_boxes.value_counts().sort_index().to_dict()}')


if __name__ == '__main__':
    main()
