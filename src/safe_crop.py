"""원본 사진에서 **주석된 병변을 하나도 안 자르는** random crop (STEP 44).

멘토 제안 — *"병변이 최대한 안 잘리는 범위로 random crop"*. 저장된 크롭
(`m2.5`·`f320`)은 이미 병변을 가운데 놓고 자른 결과라 여기서 못 씁니다.
좌표는 **원본 JSON** 에서 옵니다 — 매니페스트의 첫 box 하나가 아닙니다.

torch 를 안 씁니다: 로컬 전처리와 학습 양쪽에서 그대로 부릅니다.
"""
from __future__ import annotations

import math
import random
import re


def annotation_boxes(record):
    """box 와 polygon 외접값을 전부 모읍니다 — 병변이 여러 개인 사진 포함."""
    boxes = []
    for item in record.get("labelingInfo") or []:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("box"), dict):
            locations = item["box"].get("location")
            if not locations:
                raise ValueError("Empty box annotation")
            for loc in locations if isinstance(locations, list) else [locations]:
                try:
                    x, y, w, h = (float(loc[k]) for k in ('x', 'y', 'width', 'height'))
                except (KeyError, TypeError, ValueError):
                    raise ValueError("Malformed box annotation")
                boxes.append([x, y, x+w, y+h])
        if isinstance(item.get("polygon"), dict):
            locations = item["polygon"].get("location")
            if not locations:
                raise ValueError("Empty polygon annotation")
            for loc in locations if isinstance(locations, list) else [locations]:
                if not isinstance(loc, dict):
                    raise ValueError("Malformed polygon annotation")
                indices = {int(k[1:]) for k in loc if re.fullmatch(r'[xy]\d+', k)}
                if len(indices) < 3 or indices != set(range(1, max(indices)+1)):
                    raise ValueError("Malformed polygon annotation")
                try:
                    pts = [(float(loc[f'x{i}']), float(loc[f'y{i}'])) for i in sorted(indices)]
                except (KeyError, TypeError, ValueError):
                    raise ValueError("Malformed polygon coordinate")
                if not all(math.isfinite(v) for p in pts for v in p):
                    raise ValueError("Nonfinite polygon coordinate")
                xs, ys = zip(*pts)
                boxes.append([min(xs), min(ys), max(xs), max(ys)])
    return boxes


def sample_window(width, height, boxes, *, scale=(0.35, 1.0),
                  ratio=(0.85, 1.18), padding=0.05, attempts=50, rng=None):
    """모든 box 를 여유와 함께 담는 창을 뽑습니다. `scale` 은 **원본 면적 대비 비율**.

    요청한 크기·비율로 병변을 다 담을 수 없으면 **원본 전체**로 물러섭니다.
    주석이 없는 사진도 원본 전체입니다 — 배경 아무 곳이나 자르지 않습니다.
    좌표가 잘못됐으면 **실패시킵니다**: 조용히 잘못된 크롭으로 학습하는 것보다
    멈추는 게 낫습니다.
    """
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    if not (0 < scale[0] <= scale[1] <= 1):
        raise ValueError("scale must satisfy 0 < low <= high <= 1")
    if not (0 < ratio[0] <= ratio[1]) or padding < 0 or attempts < 1:
        raise ValueError("Invalid ratio, padding or attempts")
    rng = rng or random
    full = (0, 0, width, height)
    protected = []
    for box in boxes:
        if len(box) != 4 or not all(math.isfinite(v) for v in box):
            raise ValueError("Invalid lesion box")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("Lesion box outside original image")
        dx, dy = (x2 - x1) * padding, (y2 - y1) * padding
        protected.append((max(0, x1-dx), max(0, y1-dy),
                          min(width, x2+dx), min(height, y2+dy)))
    if not protected:
        return full
    left = min(b[0] for b in protected)
    top = min(b[1] for b in protected)
    right = max(b[2] for b in protected)
    bottom = max(b[3] for b in protected)
    for _ in range(attempts):
        area = width * height * rng.uniform(*scale)
        aspect = math.exp(rng.uniform(math.log(ratio[0]), math.log(ratio[1])))
        w, h = round(math.sqrt(area * aspect)), round(math.sqrt(area / aspect))
        if not (1 <= w <= width and 1 <= h <= height):
            continue
        xmin, xmax = max(0, math.ceil(right-w)), min(width-w, math.floor(left))
        ymin, ymax = max(0, math.ceil(bottom-h)), min(height-h, math.floor(top))
        if xmin <= xmax and ymin <= ymax:
            x, y = rng.randint(xmin, xmax), rng.randint(ymin, ymax)
            return x, y, x+w, y+h
    return full


def crop_original(image, record, *, rng=None, **kwargs):
    """PIL 크롭과 그 원본 좌표 창을 돌려줍니다 — 리사이즈는 안 합니다."""
    window = sample_window(*image.size, annotation_boxes(record), rng=rng, **kwargs)
    return image.crop(window), window
