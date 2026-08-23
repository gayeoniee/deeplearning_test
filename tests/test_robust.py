"""견고성 검사 회귀 테스트.

이 검사의 존재 이유는 "배율에 의존하는 모델을 잡아내기" 입니다.
그러니 테스트는 **일부러 배율에 의존하는 모델**을 만들어서
검사가 그걸 실제로 잡는지 확인해야 합니다.
검사가 조용히 통과만 시켜주면, 지름길이 있어도 없다고 보고하게 됩니다.

    python tests/test_robust.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import robust                                    # noqa: E402
from src.config import CFG                                # noqa: E402

IMG = 64
CLASSES2 = ["A", "B"]


# ──────────────────────────────────────────────────────────────
# 교란 변환 자체
# ──────────────────────────────────────────────────────────────
def _checker(size: int, cell: int) -> Image.Image:
    """cell 픽셀 격자무늬 — 배율이 바뀌면 격자 주기가 바뀝니다."""
    a = np.indices((size, size)).sum(0) // cell % 2
    return Image.fromarray((a * 255).astype(np.uint8)).convert("RGB")


def test_zoom_view_outputs_exact_size():
    im = _checker(200, 8)
    for z in (0.4, 0.71, 1.0, 1.41, 2.5):
        out = robust.ZoomView(IMG, z)(im)
        assert out.size == (IMG, IMG), f"z={z} 에서 크기 {out.size}"


def _center_square(size: int, side: int) -> Image.Image:
    """검은 배경 가운데 흰 정사각형 — 배율이 바뀌면 흰 면적이 바뀝니다."""
    a = np.zeros((size, size), dtype=np.uint8)
    o = (size - side) // 2
    a[o:o + side, o:o + side] = 255
    return Image.fromarray(a).convert("RGB")


def _white_frac(pil) -> float:
    return float((np.asarray(pil.convert("L")) > 127).mean())


def test_zoom_view_actually_changes_scale():
    """z 를 키우면 가운데 물체가 화면에서 더 크게 보여야 합니다.

    (앞서 격자무늬의 고주파로 재려 했는데, z<1 에서 격자가 앨리어싱으로
     뭉개져 신호가 사라집니다. 물체 면적으로 재는 게 확실합니다.)
    """
    im = _center_square(400, 100)          # 화면의 6.25%
    fracs = [_white_frac(robust.ZoomView(IMG, z)(im)) for z in (0.5, 1.0, 2.0)]
    assert fracs[0] < fracs[1] < fracs[2], f"배율에 따라 커지지 않음: {fracs}"
    # 2배 확대면 면적은 대략 4배
    assert fracs[2] > fracs[1] * 2.5, f"확대량이 부족: {fracs}"


def test_zoom_view_small_z_pads_without_error():
    """z<1 은 크롭 밖 픽셀이 필요합니다 — 반사 패딩으로 버텨야 합니다.

    작은 입력 + 극단적 축소에서 무한 루프나 예외가 나지 않는지가 핵심입니다.
    """
    for size in (40, 80, 300):
        for z in (0.1, 0.25, 0.5):
            out = robust.ZoomView(IMG, z)(_center_square(size, size // 2))
            assert out.size == (IMG, IMG), f"size={size} z={z} → {out.size}"
            # 완전히 균일해지면 패딩이 망가진 것 (흰 사각형이 남아 있어야 함)
            f = _white_frac(out)
            assert 0.0 < f < 1.0, f"size={size} z={z} 에서 내용이 사라짐 (흰 비율 {f})"


def test_shift_view_moves_content():
    im = _checker(200, 8)
    base = np.asarray(robust.ShiftView(IMG, 0.0)(im), dtype=float)
    moved = np.asarray(robust.ShiftView(IMG, 0.3)(im), dtype=float)
    assert base.shape == moved.shape == (IMG, IMG, 3)
    assert np.abs(base - moved).mean() > 1.0, "이동이 적용되지 않았습니다"


def test_shift_view_zero_is_identity_size():
    out = robust.ShiftView(IMG, 0.0)(_checker(200, 8))
    assert out.size == (IMG, IMG)


# ──────────────────────────────────────────────────────────────
# 검사가 지름길 모델을 잡아내는가
#
# ⚠️ 여기서 쓰는 가짜 모델은 **절대 임계값**을 써야 합니다.
#    두 클래스를 서로 비교하는 규칙(예: "둘 중 고주파가 큰 쪽")은 배율이 바뀌어도
#    순서가 유지되므로 배율에 의존하지 않습니다 — 검사가 잡을 것도 없습니다.
#    실제 CNN 이 배우는 건 "겉보기 크기가 이 정도면 A6" 같은 절대 규칙입니다.
# ──────────────────────────────────────────────────────────────
class SizeReader(nn.Module):
    """겉보기 크기(밝은 면적)만 보고 절대 임계값으로 찍는 모델.

    실제 상황의 축소판: 크롭 배율이 클래스마다 다르면 병변의 겉보기 크기가
    클래스와 상관되고, 모델은 "이 정도 크기면 A6" 를 배웁니다.
    그런데 사용자가 다른 거리에서 찍으면 그 규칙이 무너집니다.
    """

    def __init__(self, thresh: float):
        super().__init__()
        self.thresh = float(thresh)

    def forward(self, x):
        m = x.mean(dim=(1, 2, 3)) - self.thresh
        return torch.stack([-m, m], dim=1) * 8.0


class Constant(nn.Module):
    """항상 같은 답. 배율을 바꿔도 점수가 안 변합니다 (= 견고하지만 쓸모없음)."""

    def forward(self, x):
        out = torch.zeros(x.shape[0], 2, device=x.device)
        out[:, 0] = 5.0
        return out


def _size_frame(tmpdir, side_a=40, side_b=120, n=24):
    """A = 작은 흰 사각형, B = 큰 흰 사각형.

    즉 **겉보기 크기만으로 구분되는** 데이터. 우리 데이터의 A1 vs A6 를 흉내낸 것입니다.
    """
    root = Path(tmpdir)
    rows = []
    for i in range(n):
        lab = "A" if i % 2 == 0 else "B"
        p = root / f"{i}.png"
        _center_square(256, side_a if lab == "A" else side_b).save(p)
        rows.append({"label": lab, "crop_path": str(p), "image_path": str(p),
                     "crop_tag": "m1.5", "img_w": 256, "img_h": 256,
                     "bbox": [64, 64, 192, 192]})
    return pd.DataFrame(rows)


def _cfg():
    return CFG(img_size=IMG, batch_size=8, num_workers=0)


def _fit_threshold(df, cfg) -> float:
    """z=1 에서 두 클래스를 가르는 임계값. 가짜 모델을 데이터에 맞춰 둡니다."""
    from torchvision import transforms as T

    from src.data import IMAGENET_MEAN, IMAGENET_STD

    tf = T.Compose([robust.ZoomView(cfg.img_size, 1.0), T.ToTensor(),
                    T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    per = {}
    for _, r in df.iterrows():
        with Image.open(r["crop_path"]) as im:
            x = tf(im.convert("RGB"))
        per.setdefault(r["label"], []).append(float(x.mean()))
    means = {k: float(np.mean(v)) for k, v in per.items()}
    return (means["A"] + means["B"]) / 2


def test_scale_stress_flags_scale_dependent_model():
    with tempfile.TemporaryDirectory() as td:
        df = _size_frame(td)
        cfg = _cfg()
        model = SizeReader(_fit_threshold(df, cfg))

        # 먼저 z=1 에서는 잘 맞혀야 합니다 (그래야 하락이 의미 있음)
        base = robust.scale_stress(model, df, cfg, CLASSES2, zooms=(1.0,),
                                   n=None, device="cpu")
        assert base[robust._zoom_name(1.0)]["macro_f1"] > 0.9, "가짜 모델이 기준부터 못 맞힘"

        out = robust.scale_stress(model, df, cfg, CLASSES2, zooms=(0.4, 1.0, 2.5),
                                  n=None, device="cpu")
    s = out["_summary"]
    assert s["rel_drop"] > 0.30, (
        f"겉보기 크기에만 의존하는 모델인데 하락이 {s['rel_drop']:.1%} 뿐 "
        "— 검사가 못 잡습니다")


def test_scale_stress_passes_scale_invariant_model():
    """거짓 경보 확인 — 배율과 무관한 모델은 통과해야 합니다."""
    with tempfile.TemporaryDirectory() as td:
        df = _size_frame(td)
        out = robust.scale_stress(Constant(), df, _cfg(), CLASSES2,
                                  zooms=(0.4, 1.0, 2.5), n=None, device="cpu")
    assert out["_summary"]["rel_drop"] < 1e-6, "배율 무관 모델에 하락을 보고했습니다"


def test_scale_stress_needs_baseline_zoom():
    """zooms 에 1.0 이 없으면 하락폭을 못 냅니다 — 조용히 넘어가지 않아야 합니다."""
    with tempfile.TemporaryDirectory() as td:
        df = _size_frame(td)
        out = robust.scale_stress(Constant(), df, _cfg(), CLASSES2,
                                  zooms=(0.5, 2.0), n=None, device="cpu")
    assert "_summary" not in out


def test_shift_stress_runs_and_summarizes():
    with tempfile.TemporaryDirectory() as td:
        df = _size_frame(td)
        cfg = _cfg()
        model = SizeReader(_fit_threshold(df, cfg))
        out = robust.shift_stress(model, df, cfg, CLASSES2, fracs=(0.0, 0.25),
                                  n=None, device="cpu")
    assert "_summary" in out
    assert robust._shift_name(0.0) in out and robust._shift_name(0.25) in out


def test_report_covers_both():
    with tempfile.TemporaryDirectory() as td:
        df = _size_frame(td)
        out = robust.report(Constant(), df, _cfg(), CLASSES2, n=None, device="cpu")
    assert set(out) == {"scale", "shift"}


# ──────────────────────────────────────────────────────────────
# full 크롭 손실
# ──────────────────────────────────────────────────────────────
def test_full_crop_loss_counts_lost_lesions():
    from src import crop
    from src.config import NORMAL_LABEL

    # 1920x1080 → 중앙 정사각은 x 420~1500.
    rows = [
        # 중앙: 전부 보임
        {"label": "A2", "bbox": [900, 400, 1000, 500], "img_w": 1920, "img_h": 1080},
        # 왼쪽 끝: 완전히 사라짐
        {"label": "A5", "bbox": [0, 400, 100, 500], "img_w": 1920, "img_h": 1080},
        # 오른쪽 끝: 완전히 사라짐
        {"label": "A6", "bbox": [1800, 400, 1900, 500], "img_w": 1920, "img_h": 1080},
        # 경계에 걸침: 절반쯤
        {"label": "A1", "bbox": [370, 400, 470, 500], "img_w": 1920, "img_h": 1080},
        {"label": NORMAL_LABEL, "bbox": [900, 400, 1000, 500],
         "img_w": 1920, "img_h": 1080},
    ]
    out = crop.full_crop_loss(pd.DataFrame(rows), tag="full")

    assert out["n"] == 5
    # 병변 4개 중 2개가 사라짐
    assert abs(out["lesion_mostly_gone"] - 0.5) < 1e-6
    assert abs(out["stage1_recall_ceiling"] - 0.5) < 1e-6


def test_full_crop_loss_no_loss_when_all_centered():
    from src import crop

    rows = [{"label": "A2", "bbox": [900, 400, 1000, 500],
             "img_w": 1920, "img_h": 1080} for _ in range(10)]
    out = crop.full_crop_loss(pd.DataFrame(rows), tag="full")
    assert out["lesion_mostly_gone"] == 0.0
    assert out["stage1_recall_ceiling"] == 1.0
    assert out["fully_visible"] == 1.0


def test_full_crop_loss_fixed_tag_keeps_lesion():
    """f320 은 병변 중심으로 자르니 손실이 없어야 합니다 — full 과의 핵심 차이."""
    from src import crop

    rows = [
        {"label": "A5", "bbox": [0, 400, 100, 500], "img_w": 1920, "img_h": 1080},
        {"label": "A6", "bbox": [1800, 400, 1900, 500], "img_w": 1920, "img_h": 1080},
    ]
    out = crop.full_crop_loss(pd.DataFrame(rows), tag="f320")
    assert out["lesion_mostly_gone"] == 0.0, "f320 인데 병변을 잃었습니다"
    assert out["stage1_recall_ceiling"] == 1.0


# ──────────────────────────────────────────────────────────────
# 증강 프리셋
# ──────────────────────────────────────────────────────────────
def test_aug_presets_change_scale_range():
    """프리셋이 배율 범위를 의도한 방향으로 바꾸는가 (넓히기 / **좁히기** 둘 다)."""
    from src.config import AUG_PRESETS, with_aug

    base = CFG(exp_name="s2")
    lo_default = base.rrc_scale[0]

    # 넓히는 쪽 — 확대 범위를 키웁니다
    for name in ("scale_robust", "zoom_both"):
        c = with_aug(base, name)
        assert c.rrc_scale[0] < lo_default, f"{name} 이 배율 범위를 넓히지 않습니다"
        assert c.exp_name.endswith(name), "실험 이름에 프리셋이 안 붙었습니다"

    # ★ 좁히는 쪽 (멘토 피드백 1번) — 넓히기만 두 번 실패해서 추가한 방향입니다
    narrow = with_aug(base, "narrow")
    assert narrow.rrc_scale[0] > lo_default, "narrow 가 배율 범위를 좁히지 않습니다"

    # 축소를 가르치는 프리셋은 affine_scale 하한이 1 미만이어야 합니다.
    # RandomResizedCrop 은 축소를 못 하므로 이게 없으면 축소를 못 배웁니다.
    for name in ("zoom_both", "zoom_mild", "zoom_shift", "kitchen_sink"):
        c = with_aug(base, name)
        assert c.affine_scale and c.affine_scale[0] < 1.0, \
            f"{name} 에 축소(affine_scale<1)가 없습니다"

    # zoom_mild 는 zoom_both 보다 완만해야 합니다
    assert with_aug(base, "zoom_mild").affine_scale[0] > with_aug(base, "zoom_both").affine_scale[0]

    # default 는 아무것도 안 바꿈
    assert with_aug(base, "default").to_dict() == base.to_dict()
    assert "default" in AUG_PRESETS


def test_aug_preset_rejects_unknown():
    from src.config import aug_preset

    try:
        aug_preset("없는프리셋")
    except KeyError as e:
        assert "모르는 프리셋" in str(e)
    else:
        raise AssertionError("모르는 프리셋을 통과시켰습니다")


def test_aug_preset_actually_changes_transform():
    """프리셋이 **실제 변환에** 반영되는지 (설정만 바뀌고 안 쓰이면 무의미).

    albumentations 든 torchvision 이든 같은 불변식을 확인합니다.
    """
    from src.config import with_aug
    from src.data import build_transforms

    base = CFG(img_size=IMG, exp_name="t")

    def rrc_scale(tf):
        # albumentations 는 어댑터 안에 Compose 가 들어 있습니다
        ops = getattr(getattr(tf, "tf", tf), "transforms", [])
        for op in ops:
            if type(op).__name__ == "RandomResizedCrop":
                return tuple(op.scale)
        return None

    a = rrc_scale(build_transforms(base, train=True))
    wide = rrc_scale(build_transforms(with_aug(base, "zoom_both"), train=True))
    narrow = rrc_scale(build_transforms(with_aug(base, "narrow"), train=True))

    assert a and wide and narrow, "RandomResizedCrop 을 못 찾았습니다"
    assert wide[0] < a[0], "zoom_both 가 변환에 반영되지 않았습니다"
    assert narrow[0] > a[0], "narrow 가 변환에 반영되지 않았습니다"


def test_train_aug_is_random_per_call():
    """증강은 **학습 중 실시간 랜덤**이어야 합니다 (멘토 피드백 7번).

    같은 이미지를 두 번 넣어 다른 결과가 나와야 합니다. 미리 만들어 캐시하는
    구조라면 여기서 걸립니다.
    """
    import numpy as np
    from PIL import Image

    from src.data import build_transforms

    tf = build_transforms(CFG(img_size=IMG, exp_name="t"), train=True)
    im = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
    a, b = tf(im), tf(im)
    assert a.shape == b.shape
    assert not np.allclose(a.numpy(), b.numpy()), "같은 입력에 같은 결과 — 랜덤이 아닙니다"


def test_val_transform_is_deterministic():
    """검증은 반대로 **결정론적**이어야 합니다.

    여기에 랜덤이 섞이면 점수가 흔들리고 로짓 캐시 지문도 무의미해집니다.
    albumentations 로 옮기면서 검증 경로를 안 건드렸는지 확인합니다.
    """
    import numpy as np
    from PIL import Image

    from src.data import build_transforms

    tf = build_transforms(CFG(img_size=IMG, exp_name="t"), train=False)
    im = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
    assert np.allclose(tf(im).numpy(), tf(im).numpy()), "검증 변환에 랜덤이 섞였습니다"


# ──────────────────────────────────────────────────────────────
# 파인튜닝 강도 프리셋
#
# ⚠️ 우리는 처음부터 fine-tuning 이었습니다 (freeze_backbone 은 호출되지 않음).
#    프리셋이 정하는 건 "할까 말까" 가 아니라 "백본을 얼마나 움직일까" 입니다.
# ──────────────────────────────────────────────────────────────
def test_ft_presets_scale_backbone_lr():
    from src import models
    from src.config import FT_PRESETS, with_finetune

    base = CFG(model_name="resnet18", lr=3e-4, exp_name="s2")
    seen = {}
    for name in FT_PRESETS:
        m = models.build("resnet18", 6, pretrained=False, verbose=False)
        c = with_finetune(base, name)
        groups = models.param_groups(m, c)
        lrs = sorted({g["lr"] for g in groups})
        seen[name] = (min(lrs), max(lrs),
                      sum(p.numel() for p in m.parameters() if p.requires_grad))

    # 백본 lr 순서: linear_probe(동결) < conservative < moderate
    assert seen["conservative"][0] < seen["moderate"][0], "moderate 가 백본을 더 안 움직임"
    # aggressive 는 백본과 헤드 lr 이 같아야 합니다
    assert abs(seen["aggressive"][0] - seen["aggressive"][1]) < 1e-12


def test_linear_probe_actually_freezes_backbone():
    """lr 0 으로 두는 것과 requires_grad 를 끄는 건 결과는 같지만 속도가 다릅니다.

    얼리지 않으면 백본까지 역전파해 계산을 낭비합니다 — 대조군 실험이 느려집니다.
    """
    from src import models
    from src.config import CFG as _CFG

    m = models.build("resnet18", 6, pretrained=False, verbose=False)
    total = sum(p.numel() for p in m.parameters())
    models.param_groups(m, _CFG(model_name="resnet18", backbone_lr_mult=0.0))
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)

    assert trainable < total * 0.01, f"백본이 안 얼었습니다 ({trainable:,}/{total:,})"
    assert trainable > 0, "헤드까지 얼렸습니다"


def test_finetune_default_is_not_frozen():
    """★ 기본 설정은 fine-tuning 이어야 합니다 (linear probe 가 아니라).

    freeze_backbone() 이 실수로 호출되거나 기본값이 0 이 되면
    조용히 linear probe 로 바뀌어 성능이 크게 떨어집니다.
    """
    from src import models
    from src.config import CFG as _CFG

    c = _CFG(model_name="resnet18")
    assert c.backbone_lr_mult > 0, "기본 설정이 백본을 얼리고 있습니다"

    m = models.build("resnet18", 6, pretrained=False, verbose=False)
    total = sum(p.numel() for p in m.parameters())
    models.param_groups(m, c)
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert trainable > total * 0.9, "기본 설정에서 백본이 학습되지 않습니다"


# ──────────────────────────────────────────────────────────────
# 화질 교란 — "흐리면 정상" 지름길을 잡아내는가
#
# 실측: A7 무증상의 선명도 중앙값 50, 병변 274. 정상 사진이 계통적으로 흐립니다.
# 모델이 그걸 단서로 쓰면 val 점수는 멀쩡한데 개체가 바뀌면 무너집니다
# (STEP 5: 1단계 AUROC 0.8143 → holdout 0.7412).
#
# 그러니 여기서도 **일부러 화질에 의존하는 모델**을 만들어, 검사가 잡는지 봅니다.
# ──────────────────────────────────────────────────────────────
def _sharpness(pil) -> float:
    """인접 픽셀 차이의 분산 — 라플라시안 분산의 값싼 대용."""
    a = np.asarray(pil.convert("L"), dtype=float)
    return float(np.var(np.diff(a, axis=0)))


def test_blur_view_monotonically_softens():
    im = _checker(200, 6)
    vals = [_sharpness(robust.BlurView(IMG, r)(im)) for r in (0.0, 1.0, 2.0, 3.0)]
    assert vals[0] > vals[1] > vals[2], f"흐림이 단조롭게 안 먹습니다: {vals}"
    assert all(robust.BlurView(IMG, r)(im).size == (IMG, IMG) for r in (0, 1, 2, 3))


class SharpnessReader(nn.Module):
    """**선명도만** 보고 찍는 모델 — 우리가 의심하는 그 지름길의 축소판.

    또렷하면 B(병변), 흐리면 A(정상). 화질 교란을 걸면 전부 A 로 무너져야 합니다.
    """

    def __init__(self, thresh: float):
        super().__init__()
        self.thresh = float(thresh)

    def forward(self, x):
        # 세로 인접 픽셀 차이의 분산 = 고주파 성분 (배치별 스칼라)
        d = x[:, :, 1:, :] - x[:, :, :-1, :]
        m = d.flatten(1).var(dim=1) - self.thresh
        return torch.stack([-m, m], dim=1) * 40.0


def _sharpness_frame(tmpdir, n=24):
    """A = 흐린 사진, B = 또렷한 사진. **화질만으로 구분되는** 데이터."""
    root = Path(tmpdir)
    rows = []
    for i in range(n):
        lab = "A" if i % 2 == 0 else "B"
        im = _checker(256, 6)
        if lab == "A":                       # 정상 = 대충 찍어서 흐림
            from PIL import ImageFilter
            im = im.filter(ImageFilter.GaussianBlur(radius=3.0))
        p = root / f"s{i}.png"
        im.save(p)
        rows.append({"label": lab, "crop_path": str(p), "image_path": str(p),
                     "crop_tag": "full", "img_w": 256, "img_h": 256,
                     "bbox": [64, 64, 192, 192]})
    return pd.DataFrame(rows)


def _fit_sharpness_threshold(df, cfg) -> float:
    from torchvision import transforms as T

    from src.data import IMAGENET_MEAN, IMAGENET_STD

    tf = T.Compose([robust.BlurView(cfg.img_size, 0.0), T.ToTensor(),
                    T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    per: dict[str, list[float]] = {}
    for _, r in df.iterrows():
        with Image.open(r["crop_path"]) as im:
            x = tf(im.convert("RGB"))
        d = x[:, 1:, :] - x[:, :-1, :]
        per.setdefault(r["label"], []).append(float(d.flatten().var()))
    means = {k: float(np.mean(v)) for k, v in per.items()}
    return (means["A"] + means["B"]) / 2


def test_blur_stress_flags_a_quality_shortcut():
    """화질로 찍는 모델은 흐림 교란에서 크게 무너져야 합니다."""
    with tempfile.TemporaryDirectory() as td:
        df = _sharpness_frame(td)
        cfg = _cfg()
        model = SharpnessReader(_fit_sharpness_threshold(df, cfg))
        out = robust.blur_stress(model, df, cfg, CLASSES2,
                                 radii=(0.0, 1.0, 2.0, 3.0), n=len(df), device="cpu")
        s = out["_summary"]
        assert s["baseline"] > 0.8, f"기준 조건에서부터 못 맞힙니다: {s}"
        assert s["rel_drop"] > 0.20, f"화질 지름길인데 하락 {s['rel_drop']:.1%} 뿐입니다"


def test_blur_stress_passes_a_quality_independent_model():
    """화질을 안 쓰는 모델은 통과해야 합니다 (검사가 아무나 잡으면 쓸모없음)."""
    with tempfile.TemporaryDirectory() as td:
        df = _size_frame(td)
        cfg = _cfg()
        model = SizeReader(_fit_threshold(df, cfg))   # 밝은 면적으로 찍는 모델
        out = robust.blur_stress(model, df, cfg, CLASSES2,
                                 radii=(0.0, 1.0, 2.0), n=len(df), device="cpu")
        s = out["_summary"]
        assert s["rel_drop"] < 0.15, f"화질과 무관한데 하락 {s['rel_drop']:.1%}"


def test_blur_condition_names_are_readable():
    with tempfile.TemporaryDirectory() as td:
        df = _size_frame(td, n=8)
        out = robust.blur_stress(SizeReader(0.0), df, _cfg(), CLASSES2,
                                 radii=(0.0, 2.0), n=8, device="cpu")
        assert "원본(선명)" in out and "흐림(r=2)" in out, list(out)


if __name__ == "__main__":
    import io
    from contextlib import redirect_stdout

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    fails = 0
    for fn in fns:
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:
            fails += 1
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - fails}/{len(fns)} passed")
    sys.exit(1 if fails else 0)