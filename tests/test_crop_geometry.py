"""크롭 좌표 재현 테스트.

왜 필요한가:
  학습은 크롭만 올린 Colab 에서 합니다. 그런데 Grad-CAM 게이트는
  "CAM 이 병변 위에 있는가" 를 재야 하고, 그러려면 **크롭 안에서 병변이 어디인지**
  알아야 합니다. 원본이 없으니 `crop.bbox_in_crop()` 이 크롭 창을 다시 계산해
  병변 위치를 되찾습니다. 그 계산이 틀리면 게이트가 조용히 무의미해집니다
  — 통과도 실패도 다 믿을 수 없게 됩니다.

    python tests/test_crop_geometry.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import crop                       # noqa: E402
from src.config import CFG                 # noqa: E402


def test_margin_of_tag():
    assert crop.margin_of_tag("m1.5") == 1.5
    assert crop.margin_of_tag("m2.5") == 2.5
    assert crop.margin_of_tag("full") == 0.0
    assert crop.margin_of_tag(None) == 0.0
    assert crop.margin_of_tag("m엉뚱") == 0.0


def test_bbox_in_crop_is_centered_and_scaled():
    """이미지 가운데 200x200 병변, margin 1.5 → 병변이 크롭 가운데 2/3 를 차지."""
    row = {"bbox": [860, 440, 1060, 640], "img_w": 1920, "img_h": 1080,
           "crop_tag": "m1.5"}
    rel = crop.bbox_in_crop(row, cfg=CFG(crop_min_px=64))
    assert rel is not None

    w, h = rel[2] - rel[0], rel[3] - rel[1]
    assert abs(w - 1 / 1.5) < 0.02, f"폭 비율 {w:.3f} — 기대 {1/1.5:.3f}"
    assert abs(h - 1 / 1.5) < 0.02
    cx, cy = (rel[0] + rel[2]) / 2, (rel[1] + rel[3]) / 2
    assert abs(cx - 0.5) < 0.02 and abs(cy - 0.5) < 0.02, "병변이 크롭 중앙이 아닙니다"


def test_bbox_in_crop_wider_margin_gives_smaller_box():
    row = {"bbox": [860, 440, 1060, 640], "img_w": 1920, "img_h": 1080}
    a = crop.bbox_in_crop(row, tag="m1.5")
    b = crop.bbox_in_crop(row, tag="m2.5")
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])          # noqa: E731
    assert area(b) < area(a), "margin 을 키웠는데 병변 비중이 안 줄었습니다"
    # 2.5배 넓히면 면적 비중은 (1/2.5)^2 ≈ 0.16
    assert abs(area(b) - (1 / 2.5) ** 2) < 0.02


def test_bbox_in_crop_handles_edge_lesion():
    """왼쪽 위 모서리 병변 — expand_box 가 창을 안쪽으로 밀어넣습니다."""
    row = {"bbox": [0, 0, 100, 100], "img_w": 1920, "img_h": 1080, "crop_tag": "m1.5"}
    rel = crop.bbox_in_crop(row)
    assert rel is not None
    assert all(0.0 <= v <= 1.0 for v in rel), f"정규화 좌표가 범위를 벗어남: {rel}"
    # 창이 왼쪽 위에 붙었으므로 병변도 왼쪽 위에 붙어 있어야 합니다
    assert rel[0] < 0.05 and rel[1] < 0.05


def test_bbox_in_crop_full_tag_uses_center_square():
    """'full' 은 중앙 정사각 크롭이었습니다 — 좌우가 잘려 나갑니다."""
    row = {"bbox": [800, 400, 1000, 600], "img_w": 1920, "img_h": 1080}
    rel = crop.bbox_in_crop(row, tag="full")
    assert rel is not None
    # 중앙 1080x1080 창의 좌상단은 x=420 → 병변 x1 = (800-420)/1080
    assert abs(rel[0] - (800 - 420) / 1080) < 1e-6
    assert abs(rel[1] - 400 / 1080) < 1e-6

    # 왼쪽 끝 병변은 중앙 정사각 크롭 밖이라 계산 불가여야 합니다
    off = crop.bbox_in_crop({"bbox": [0, 400, 200, 600], "img_w": 1920, "img_h": 1080},
                            tag="full")
    assert off is None, "크롭 밖 병변을 유효한 좌표로 돌려줬습니다"


def test_bbox_in_crop_rejects_bad_rows():
    assert crop.bbox_in_crop({"bbox": None, "img_w": 100, "img_h": 100}, tag="m1.5") is None
    assert crop.bbox_in_crop({"bbox": [1, 2, 3], "img_w": 100, "img_h": 100}, tag="m1.5") is None
    assert crop.bbox_in_crop({"bbox": [1, 2, 3, 4]}, tag="m1.5") is None          # 크기 없음
    assert crop.bbox_in_crop({"bbox": [1, 2, 3, 4], "img_w": 0, "img_h": 0},
                             tag="m1.5") is None
    # JSON 문자열로 저장된 bbox 도 읽어야 합니다 (parquet 왕복 후 흔한 형태)
    ok = crop.bbox_in_crop({"bbox": "[860, 440, 1060, 640]", "img_w": 1920,
                            "img_h": 1080}, tag="m1.5")
    assert ok is not None


def test_bbox_in_crop_matches_expand_box():
    """expand_box 와 같은 창을 쓰는지 직접 대조 — 두 코드가 갈라지면 게이트가 거짓말합니다."""
    bbox = [300.0, 200.0, 460.0, 420.0]
    W, H, margin = 1280, 960, 1.5
    win = crop.expand_box(bbox, W, H, margin=margin, min_px=CFG().crop_min_px)
    rel = crop.bbox_in_crop({"bbox": bbox, "img_w": W, "img_h": H}, tag="m1.5")

    ww, wh = win[2] - win[0], win[3] - win[1]
    expected = [(bbox[0] - win[0]) / ww, (bbox[1] - win[1]) / wh,
                (bbox[2] - win[0]) / ww, (bbox[3] - win[1]) / wh]
    for a, b in zip(rel, expected):
        assert abs(a - b) < 1e-9


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    fails = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:
            fails += 1
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - fails}/{len(fns)} passed")
    sys.exit(1 if fails else 0)
