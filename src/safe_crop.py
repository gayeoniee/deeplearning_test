"""Original-image random crops retaining every annotated lesion.

Coordinates must come from the original JSON, not the first-box legacy manifest.
No torch dependency: usable during local preprocessing and training alike.
"""
from __future__ import annotations

import math
import random
import re


def annotation_boxes(record):
    """Collect boxes and polygon bounds, including multiple lesions."""
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
    """Sample an image-area crop containing all boxes plus relative padding.

    If the requested size/aspect cannot contain the lesions, retain the entire
    image. Missing annotations also retain the entire image (not a random
    background patch). Invalid coordinates fail rather than silently training
    on a mislabeled crop. `scale` is crop area / original image area.
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
    """Return PIL crop and its original-coordinate window; no resizing."""
    window = sample_window(*image.size, annotation_boxes(record), rng=rng, **kwargs)
    return image.crop(window), window
