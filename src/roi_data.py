"""고정 ROI vs 병변 근처 random ROI — 검증은 둘 다 고정 ROI (STEP 44 후속)."""
import json
import math
import random

from src.original_data import OriginalDataset
from src.safe_crop import sample_window


def roi_window(width, height, boxes, *, augment=False, rng=None):
    rng = rng or random
    # 좌표 검사는 safe_crop 것을 그대로 씁니다. 주석이 없으면 두 방법 모두 원본 전체.
    sample_window(width, height, boxes, attempts=1, rng=random.Random(0))
    if not boxes:
        return (0, 0, width, height)
    left = max(0, min(b[0] - (b[2]-b[0])*.05 for b in boxes))
    top = max(0, min(b[1] - (b[3]-b[1])*.05 for b in boxes))
    right = min(width, max(b[2] + (b[2]-b[0])*.05 for b in boxes))
    bottom = min(height, max(b[3] + (b[3]-b[1])*.05 for b in boxes))
    side = max(320, math.ceil(right)-math.floor(left), math.ceil(bottom)-math.floor(top))
    if augment:
        side = math.ceil(side * rng.uniform(1.0, 1.25))
    w, h = min(width, side), min(height, side)
    xmin, xmax = max(0, math.ceil(right-w)), min(width-w, math.floor(left))
    ymin, ymax = max(0, math.ceil(bottom-h)), min(height-h, math.floor(top))
    if augment:
        x, y = rng.randint(xmin, xmax), rng.randint(ymin, ymax)
    else:
        x = min(xmax, max(xmin, round((left+right-w)/2)))
        y = min(ymax, max(ymin, round((top+bottom-h)/2)))
    return x, y, x+w, y+h


class ROIDataset(OriginalDataset):
    def __init__(self, *args, mode='fixed', **kwargs):
        if mode not in ('fixed', 'safe'):
            raise ValueError(mode)
        self.roi_mode = mode
        super().__init__(*args, mode='full', **kwargs)

    def prepare_image(self, image, row):
        boxes = json.loads(row.boxes) if isinstance(row.boxes, str) else row.boxes
        return image.crop(roi_window(*image.size, boxes,
                                    augment=self.train and self.roi_mode == 'safe'))
