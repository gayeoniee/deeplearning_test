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


def test_polygon_maps_into_crop():
    """polygon 도 bbox 와 같은 창으로 옮겨져야 합니다."""
    bbox = [860, 440, 1060, 640]
    poly = [[880, 460], [1040, 460], [1040, 620], [880, 620]]
    g = crop.geometry_in_crop(
        {"bbox": bbox, "polygon": poly, "img_w": 1920, "img_h": 1080}, tag="m1.5")
    assert g["bbox"] is not None and g["polygon"] is not None
    assert len(g["polygon"]) == 4
    assert all(0.0 <= v <= 1.0 for p in g["polygon"] for v in p)
    # polygon 이 bbox 안쪽에 있었으니 옮긴 뒤에도 안쪽이어야 합니다
    xs = [p[0] for p in g["polygon"]]; ys = [p[1] for p in g["polygon"]]
    assert min(xs) >= g["bbox"][0] - 1e-9 and max(xs) <= g["bbox"][2] + 1e-9
    assert min(ys) >= g["bbox"][1] - 1e-9 and max(ys) <= g["bbox"][3] + 1e-9


def test_polygon_absent_is_none():
    g = crop.geometry_in_crop({"bbox": [10, 10, 50, 50], "img_w": 200, "img_h": 200},
                              tag="m1.5")
    assert g["polygon"] is None
    # 점이 2개뿐이면 다각형이 아니므로 버립니다
    g2 = crop.geometry_in_crop({"bbox": [10, 10, 50, 50], "polygon": [[11, 11], [20, 20]],
                                "img_w": 200, "img_h": 200}, tag="m1.5")
    assert g2["polygon"] is None


def test_crop_rel_survives_windows_to_linux():
    """★ 실제로 당한 버그: Windows 에서 만든 crop_rel 은 역슬래시입니다.

    'm1.5\\ab\\file.jpg' 를 리눅스에서 그냥 이어붙이면 파일명 한 덩어리가 되어
    45,885개 전부 "파일 없음" 이 됩니다. 그런데 switch_tag 는 경로를 다시 계산하니
    잘 되어서, 두 함수가 서로 다른 답을 내는 상태가 됩니다.
    """
    import tempfile

    from src import labels

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "crops"
        target = root / "m1.5" / "ab" / "img_deadbeef.jpg"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"x")

        import os

        old = os.environ.get("DOG_SKIN_WORK")
        os.environ["DOG_SKIN_WORK"] = td
        try:
            import pandas as pd

            for rel in ("m1.5/ab/img_deadbeef.jpg",          # 리눅스에서 만든 것
                        "m1.5\\ab\\img_deadbeef.jpg"):        # Windows 에서 만든 것
                out = labels.rebase_paths(pd.DataFrame([{"crop_rel": rel}]))
                p = out["crop_path"].iloc[0]
                assert Path(p).exists(), f"rebase 실패: {rel!r} → {p!r}"
        finally:
            if old is None:
                os.environ.pop("DOG_SKIN_WORK", None)
            else:
                os.environ["DOG_SKIN_WORK"] = old


def test_fixed_box_is_scale_invariant():
    """★ 고정 픽셀 창의 핵심 성질: 병변 크기가 달라도 창 크기가 같습니다.

    실측 문제: A1 박스 0.47% vs A6 3.08% (6.5배). margin 크롭은 그 비율을 그대로
    확대 배율 차이로 바꿔서, 모델이 피부 대신 배율을 세게 만듭니다.
    """
    small = [900, 500, 940, 540]        # 40px 병변
    large = [900, 500, 1160, 760]       # 260px 병변
    W, H, SIDE = 1920, 1080, 320

    ws = crop.fixed_box(small, W, H, SIDE)
    wl = crop.fixed_box(large, W, H, SIDE)
    for w in (ws, wl):
        assert (w[2] - w[0]) == SIDE and (w[3] - w[1]) == SIDE, f"창 크기가 {SIDE} 가 아님: {w}"

    # margin 크롭은 반대로 크게 달라야 합니다 (그게 문제의 원인)
    ms = crop.expand_box(small, W, H, margin=1.5, min_px=64)
    ml = crop.expand_box(large, W, H, margin=1.5, min_px=64)
    assert (ml[2] - ml[0]) > 3 * (ms[2] - ms[0]), "margin 크롭의 배율 차이가 재현되지 않음"


def test_fixed_box_centers_on_lesion_and_stays_in_bounds():
    W, H, SIDE = 1920, 1080, 320
    b = [900, 500, 1000, 600]
    x1, y1, x2, y2 = crop.fixed_box(b, W, H, SIDE)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    assert abs(cx - 950) <= 1 and abs(cy - 550) <= 1, "병변 중심에서 벗어남"

    # 모서리 병변: 크기를 줄이지 않고 창을 안쪽으로 밀어야 합니다
    for corner in ([0, 0, 40, 40], [W - 40, H - 40, W, H]):
        w = crop.fixed_box(corner, W, H, SIDE)
        assert w[0] >= 0 and w[1] >= 0 and w[2] <= W and w[3] <= H, f"경계 이탈: {w}"
        assert (w[2] - w[0]) == SIDE and (w[3] - w[1]) == SIDE, f"모서리에서 창이 줄어듦: {w}"


def test_fixed_box_clamps_to_image_when_side_too_big():
    """창이 이미지보다 크면 이미지 크기로 줄입니다 (정사각 유지)."""
    w = crop.fixed_box([900, 500, 1000, 600], 1920, 1080, 4000)
    assert (w[2] - w[0]) == (w[3] - w[1]) == 1080


def test_crop_window_understands_fixed_tag():
    """'f320' 태그로 저장된 크롭도 좌표 복원이 되어야 합니다 (CAM 게이트용)."""
    row = {"bbox": [900, 500, 1000, 600], "img_w": 1920, "img_h": 1080, "crop_tag": "f320"}
    w = crop.crop_window(row)
    assert (w[2] - w[0]) == 320
    rel = crop.bbox_in_crop(row)
    assert rel is not None
    # 100px 병변 / 320px 창 → 폭의 약 31%, 중앙에
    assert abs((rel[2] - rel[0]) - 100 / 320) < 0.01
    assert abs((rel[0] + rel[2]) / 2 - 0.5) < 0.01


def test_window_is_single_source_of_truth():
    """_crop_one 과 crop_window 가 같은 함수를 쓰는지 (갈라지면 게이트가 거짓말)."""
    import inspect

    src_crop_one = inspect.getsource(crop._crop_one)
    src_window_fn = inspect.getsource(crop.crop_window)
    assert "_window(" in src_crop_one, "_crop_one 이 _window 를 쓰지 않습니다"
    assert "_window(" in src_window_fn, "crop_window 가 _window 를 쓰지 않습니다"
    # 창 계산을 직접 하고 있으면 안 됩니다
    assert "expand_box(" not in src_crop_one, "_crop_one 이 창을 직접 계산합니다"


def test_shortcut_baseline_feature_sets():
    """★ 하한선은 **크롭에 보이는 특징만** 써야 합니다.

    실제로 당한 문제: 종횡비·병변 개수·원본 해상도까지 넣었더니 하한선이 부풀려졌고,
    CNN 이 넘어야 할 선을 실제보다 높게 잡을 뻔했습니다. 크롭은 정사각형이라
    종횡비가 사라지고, 박스 하나만 자르니 병변 개수도 화면에 없습니다.
    """
    import numpy as np
    import pandas as pd

    from src import split
    from src.config import CFG, CLASSES, NORMAL_LABEL

    assert set(crop.FEATURE_SETS) == {"scale_only", "all"}
    scale = crop.FEATURE_SETS["scale_only"]
    for hidden in ("aspect", "n_lesion", "img_w", "img_h"):
        assert hidden not in scale, f"'{hidden}' 은 크롭에 안 보이는데 scale_only 에 있습니다"
    assert "area_ratio" in scale and "win_side" in scale
    assert set(scale) < set(crop.FEATURE_SETS["all"])

    # 두 집합 모두 실제로 돌아가야 합니다
    rng = np.random.default_rng(0)
    rows = []
    for i, lab in enumerate(([NORMAL_LABEL] * 60) + [c for c in CLASSES for _ in range(30)]):
        side = 120 + 40 * CLASSES.index(lab) if lab in CLASSES else 150
        rows.append({"label": lab, "animal_id": f"G{i // 6}", "phash": str(i),
                     "img_w": 1920, "img_h": 1080, "crop_tag": "m1.5",
                     "bbox": [600, 300, 600 + side, 300 + side],
                     "n_lesion": int(rng.integers(1, 4)), "synthetic": False,
                     "crop_path": f"/x/{i}.jpg", "image_path": f"/x/{i}.jpg"})
    df = split.assign(pd.DataFrame(rows), CFG(), verbose=False)

    for fs in ("scale_only", "all"):
        out = crop.shortcut_baseline(df, cfg=CFG(img_size=224), features=fs, verbose=False)
        assert "stage1_auroc_metadata_only" in out
        assert 0.0 <= out["stage2_macro_f1_metadata_only"] <= 1.0

    try:
        crop.shortcut_baseline(df, features="없는집합", verbose=False)
    except KeyError as e:
        assert "모르는 특징 집합" in str(e)
    else:
        raise AssertionError("모르는 특징 집합을 통과시켰습니다")


def test_audit_bounds_verdict_uses_rate_not_absolute_px():
    """★ 경계 이탈 판정은 절대 px 이 아니라 건수 비율로.

    실측에서 4/45,885건이 최대 135px 벗어났는데, 절대 px 기준(50px)으로는
    🚨 좌표 오류로 오판했습니다. 좌표 해석이 틀렸으면 수천 건이 어긋납니다.
    """
    import tempfile

    import pandas as pd

    with tempfile.TemporaryDirectory() as td:
        df = _audit_frame(tmpdir=td, n_each=40)          # 정상 행 다수
        bad = df.iloc[[0]].copy()
        bad["bbox"] = [[846, 975, 1131, 1215]]           # 아래로 135px 초과
        bad["img_w"], bad["img_h"] = 1920, 1080
        df.loc[:, "img_w"], df.loc[:, "img_h"] = 1920, 1080
        r = crop.audit(pd.concat([df, bad], ignore_index=True), n_sample=30)

    assert r["boxes_out_of_bounds"] == 1
    # 라벨러 오차로 판정되어야 하는 조건: 드물고(1% 미만) 이탈량도 작음(30% 미만)
    assert r["boxes_out_of_bounds_frac"] < 0.01, "드문 이탈인데 비율이 높게 계산됨"
    assert r["box_overflow_max_rel"] < 0.3, "이미지 높이 대비 이탈 비율이 잘못 계산됨"

    # 반대 경우: 계통적으로 어긋나면 좌표 오류로 판정해야 합니다
    import tempfile as _tf

    with _tf.TemporaryDirectory() as td2:
        df2 = _audit_frame(tmpdir=td2, n_each=4)          # 작은 데이터
        df2.loc[:, "img_w"], df2.loc[:, "img_h"] = 1920, 1080
        df2["bbox"] = [[100, 100, 3000, 2500]] * len(df2)  # 전부 이미지 밖
        r2 = crop.audit(df2, n_sample=10)
    assert r2["boxes_out_of_bounds_frac"] > 0.5, "계통적 이탈을 못 잡았습니다"
    assert r2["box_overflow_max_rel"] > 0.3


def test_audit_flags_within_lesion_spread():
    """정상/병변 배율은 같아도 병변끼리 다르면 2단계 지름길입니다."""
    import tempfile

    import numpy as np
    import pandas as pd
    from PIL import Image

    from src.config import CLASSES, NORMAL_LABEL

    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        rows = []
        # A1 은 작고 A6 은 크게, 정상은 그 중간 → (a) 통과 / (b) 걸림
        sides = {c: 200 for c in CLASSES}
        sides["A1"] = 120
        sides["A6"] = 620
        sides[NORMAL_LABEL] = 205
        for i, lab in enumerate([NORMAL_LABEL] * 24 + [c for c in CLASSES for _ in range(12)]):
            cp = root / "crops" / "m1.5" / f"{i}.jpg"
            cp.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(
                rng.normal(128, 40, (96, 96, 3)).clip(0, 255).astype("uint8")).save(cp)
            s = sides[lab]
            rows.append({"image_name": f"{i}.jpg", "label": lab, "crop_path": str(cp),
                         "crop_tag": "m1.5", "image_path": f"/absent/{i}.jpg",
                         "img_w": 1920, "img_h": 1080,
                         "bbox": [600, 300, 600 + s, 300 + s],
                         "area_ratio": s * s / (1920 * 1080)})
        r = crop.audit(pd.DataFrame(rows), n_sample=40)

    assert 0.67 <= r["area_ratio_normal_over_lesion"] <= 1.5, "정상/병변은 같아야 하는 설정"
    assert r["area_spread_within_lesions"] > 2.0, "병변 간 배율 격차를 못 잡았습니다"
    assert r["area_largest_class"] == "A6" and r["area_smallest_class"] == "A1"


def _audit_frame(normal_side=200, lesion_side=200, normal_noise=40, lesion_noise=40,
                 tmpdir=None, n_each=12):
    """감사용 합성 데이터. 정상/병변의 박스 크기와 화질을 따로 조절합니다."""
    import numpy as np
    import pandas as pd
    from PIL import Image

    from src.config import CLASSES, NORMAL_LABEL

    rng = np.random.default_rng(0)
    root = Path(tmpdir)
    rows = []
    labs = [NORMAL_LABEL] * (n_each * 2) + [c for c in CLASSES for _ in range(n_each)]
    for i, lab in enumerate(labs):
        cp = root / "crops" / "m1.5" / f"{i}.jpg"
        cp.parent.mkdir(parents=True, exist_ok=True)
        noise = normal_noise if lab == NORMAL_LABEL else lesion_noise
        arr = rng.normal(128, noise, (128, 128, 3)).clip(0, 255).astype("uint8")
        Image.fromarray(arr).save(cp)
        side = normal_side if lab == NORMAL_LABEL else lesion_side
        rows.append({"image_name": f"{i}.jpg", "label": lab, "crop_path": str(cp),
                     "crop_tag": "m1.5", "image_path": f"/absent/{i}.jpg",
                     "img_w": 1920, "img_h": 1080,
                     "bbox": [700, 300, 700 + side, 300 + side],
                     "area_ratio": side * side / (1920 * 1080)})
    return pd.DataFrame(rows)


def test_audit_flags_scale_shortcut():
    """★ 정상 박스가 병변 박스보다 훨씬 크면 크롭 배율이 정답을 흘립니다."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        df = _audit_frame(normal_side=520, lesion_side=200, tmpdir=td)
        r = crop.audit(df, n_sample=40)
    gap = r["area_ratio_normal_over_lesion"]
    assert gap > 1.5, f"배율 격차 {gap:.2f} — 지름길을 못 잡았습니다"
    # 노트북 03 이 이 값으로 1단계 크롭을 정합니다
    assert ("full" if (gap > 1.5 or gap < 0.67) else "m1.5") == "full"


def test_audit_passes_when_scales_match():
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        df = _audit_frame(normal_side=210, lesion_side=200, tmpdir=td)
        r = crop.audit(df, n_sample=40)
    gap = r["area_ratio_normal_over_lesion"]
    assert 0.67 <= gap <= 1.5, f"배율이 같은데 격차 {gap:.2f} 로 보고됨 (거짓 경보)"


def test_audit_flags_sharpness_shortcut():
    """정상만 흐리면 모델이 '화질' 로 맞힐 수 있습니다."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        df = _audit_frame(normal_noise=2, lesion_noise=45, tmpdir=td)
        r = crop.audit(df, n_sample=80)
    assert "blur_ratio_normal_over_lesion" in r
    assert r["blur_ratio_normal_over_lesion"] < 0.62, \
        "정상만 흐린데 화질 지름길을 못 잡았습니다"


def test_audit_detects_label_conflict_and_bad_boxes():
    import tempfile

    import pandas as pd

    with tempfile.TemporaryDirectory() as td:
        df = _audit_frame(tmpdir=td)
        # 같은 파일명에 다른 라벨 + 경계를 벗어난 박스
        bad = df.iloc[[0]].copy()
        bad["label"] = "A5"
        bad["bbox"] = [[1900, 1000, 2400, 1500]]
        df = pd.concat([df, bad], ignore_index=True)
        r = crop.audit(df, n_sample=30)
    assert r["conflicting_images"] >= 1, "라벨 충돌을 못 잡았습니다"
    assert r["boxes_out_of_bounds"] >= 1, "경계 이탈 박스를 못 잡았습니다"


def test_blur_score_orders_sharp_above_flat():
    import numpy as np
    from PIL import Image

    flat = Image.fromarray(np.full((64, 64, 3), 128, "uint8"))
    noisy = Image.fromarray(
        np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8))
    assert crop._blur_score(flat) < crop._blur_score(noisy)
    assert crop._blur_score(flat) < 1.0     # 완전히 균일한 면은 0 에 가까움


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
