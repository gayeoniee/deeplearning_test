"""Dataset / DataLoader / 증강.

증강 설계에서 한 가지만 기억하세요:

  ⚠️ 피부 병변은 "색과 질감"이 곧 라벨입니다.
     일반 이미지 분류에서 쓰는 강한 ColorJitter(0.4)를 그대로 쓰면
     A3(과다색소침착)의 어두운 색을 밝게 만들어 A1 처럼 보이게 합니다.
     즉 증강이 라벨을 파괴합니다. 그래서 여기 기본값은 의도적으로 약합니다.
     (docs/basics/06_과적합_정규화_데이터증강.md 참고)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.config import CFG, CLASSES

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ──────────────────────────────────────────────────────────────
# 변환
# ──────────────────────────────────────────────────────────────
class _AlbumentationsAdapter:
    """PIL 을 받아 albumentations 를 태우고 텐서를 돌려줍니다.

    SkinDataset 은 PIL 을 넘겨주고 torchvision 은 PIL 을 먹지만,
    albumentations 는 numpy(HWC, uint8)를 먹습니다. 그 사이를 잇습니다.
    """

    def __init__(self, tf):
        self.tf = tf

    def __call__(self, pil):
        import numpy as np

        return self.tf(image=np.asarray(pil.convert("RGB")))["image"]


def _albumentations_train(cfg: CFG, mean, std):
    """학습용 실시간 증강 (albumentations).

    ⚠️ **학습 경로에만** 씁니다. 검증은 torchvision 그대로 둡니다 —
    검증 전처리를 바꾸면 지금까지 쌓은 기준선(1단계 0.8192 / 2단계 0.5697)과
    비교가 안 되고, 로짓 캐시 지문도 전부 무효가 됩니다.

    None 을 돌려주면 호출자가 torchvision 으로 되돌아갑니다.
    """
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError:
        return None

    S = cfg.img_size
    ops = [
        # RandomResizedCrop 은 **잘라서 확대**만 합니다 (축소 불가).
        A.RandomResizedCrop(size=(S, S), scale=tuple(cfg.rrc_scale),
                            ratio=(0.85, 1.18)),
    ]

    # 축소까지 배우려면 이미지를 실제로 줄이고 여백을 채워야 합니다.
    # shift_limit 은 위치 교란(실측 20.6% 하락) 대응입니다.
    if cfg.affine_scale or cfg.shift_limit or cfg.rotate_deg:
        lo, hi = cfg.affine_scale if cfg.affine_scale else (1.0, 1.0)
        ops.append(A.Affine(
            scale=(lo, hi),
            translate_percent=(-cfg.shift_limit, cfg.shift_limit) if cfg.shift_limit else None,
            rotate=(-cfg.rotate_deg, cfg.rotate_deg) if cfg.rotate_deg else None,
            border_mode=0, fill=0, p=1.0 if (cfg.affine_scale or cfg.shift_limit) else 0.5,
        ))

    if cfg.hflip:
        ops.append(A.HorizontalFlip(p=cfg.hflip))
    if cfg.vflip:
        ops.append(A.VerticalFlip(p=cfg.vflip))

    # ⚠️ 색은 조심합니다 — A3(과다색소침착)은 색 자체가 라벨입니다.
    if cfg.color_jitter or cfg.hue_jitter:
        ops.append(A.ColorJitter(
            brightness=cfg.color_jitter, contrast=cfg.color_jitter,
            saturation=cfg.color_jitter * 0.5, hue=cfg.hue_jitter, p=0.7))

    # 촬영 조건 흉내 — 보호자 사진은 흐리고 압축돼 있습니다.
    if cfg.clahe_p:
        ops.append(A.CLAHE(clip_limit=2.0, p=cfg.clahe_p))
    if cfg.blur_p:
        ops.append(A.OneOf([A.GaussianBlur(blur_limit=(3, 7)),
                            A.MotionBlur(blur_limit=(3, 7))], p=cfg.blur_p))
    if cfg.noise_p:
        ops.append(A.GaussNoise(p=cfg.noise_p))
    if cfg.jpeg_p:
        ops.append(A.ImageCompression(quality_range=(40, 90), p=cfg.jpeg_p))

    ops += [A.Normalize(mean=mean, std=std), ToTensorV2()]
    if cfg.random_erasing:
        ops.insert(-1, A.CoarseDropout(p=cfg.random_erasing))
    return _AlbumentationsAdapter(A.Compose(ops))


def build_transforms(cfg: CFG, train: bool, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    from torchvision import transforms as T

    if not train:
        # 검증/추론은 항상 결정론적으로. 여기에 증강이 섞이면 점수가 흔들립니다.
        return T.Compose([
            T.Resize(int(cfg.img_size * 1.14)),
            T.CenterCrop(cfg.img_size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

    # ★ 학습 증강은 albumentations 우선 (torchvision 보다 빠르고 기법이 많습니다).
    #   없으면 아래 torchvision 경로로 조용히 되돌아갑니다.
    alb = _albumentations_train(cfg, mean, std)
    if alb is not None:
        return alb

    ops = [T.RandomResizedCrop(cfg.img_size, scale=cfg.rrc_scale, ratio=(0.85, 1.18))]
    # ⚠️ RandomResizedCrop 은 잘라서 **확대**만 합니다 (가장 축소돼도 이미지 전체).
    #    보호자가 멀리서 찍은 사진 = 병변이 작게 보이는 경우를 배우려면
    #    이미지를 실제로 **줄이고 여백을 채우는** 변환이 따로 필요합니다.
    if cfg.affine_scale:
        ops.append(T.RandomAffine(degrees=0, scale=tuple(cfg.affine_scale)))
    if cfg.hflip:
        ops.append(T.RandomHorizontalFlip(cfg.hflip))
    if cfg.vflip:
        ops.append(T.RandomVerticalFlip(cfg.vflip))
    if cfg.rotate_deg:
        ops.append(T.RandomRotation(cfg.rotate_deg))
    if cfg.randaugment_n:
        ops.append(T.RandAugment(num_ops=cfg.randaugment_n, magnitude=cfg.randaugment_m))
    if cfg.color_jitter or cfg.hue_jitter:
        ops.append(T.ColorJitter(
            brightness=cfg.color_jitter,
            contrast=cfg.color_jitter,
            saturation=cfg.color_jitter * 0.5,   # 채도는 더 약하게
            hue=cfg.hue_jitter,
        ))
    ops += [T.ToTensor(), T.Normalize(mean, std)]
    if cfg.random_erasing:
        ops.append(T.RandomErasing(p=cfg.random_erasing, scale=(0.02, 0.12)))
    return T.Compose(ops)


def transforms_for_model(cfg: CFG, model, train: bool):
    """timm 모델의 사전학습 통계(mean/std)에 맞춥니다.

    SigLIP/CLIP 계열은 ImageNet 통계와 mean/std 가 다릅니다.
    여기를 안 맞추면 사전학습 가중치의 이점을 상당 부분 날립니다.
    """
    mean, std = IMAGENET_MEAN, IMAGENET_STD
    cfgd = getattr(model, "pretrained_cfg", None) or {}
    if isinstance(cfgd, dict):
        mean = tuple(cfgd.get("mean", mean))
        std = tuple(cfgd.get("std", std))
    return build_transforms(cfg, train, mean, std)


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────
class SkinDataset(Dataset):
    """매니페스트 DataFrame 하나로 동작하는 Dataset.

    path_col 을 'crop_path' 로 두면 ROI 크롭본, 'image_path' 면 원본을 씁니다.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform=None,
        path_col: str = "crop_path",
        label_col: str = "label",
        classes: list[str] | None = None,
        return_index: bool = False,
        draft_size: int | None = None,
    ):
        self.classes = classes or CLASSES
        # JPEG 은 1/2, 1/4, 1/8 크기로 **디코딩 단계에서** 줄일 수 있습니다(DCT 스케일링).
        # 512px 크롭을 224 로 쓸 거면 512 전체를 푸는 건 낭비입니다.
        # draft_size 를 주면 그 크기 이상이 되는 가장 작은 배율로 풉니다.
        # ⚠️ 학습용에는 쓰지 마세요 — RandomResizedCrop 이 확대할 여지를 줄입니다.
        #    검증/추론 변환은 Resize(img_size*1.14) 로 어차피 줄이므로 손실이 없습니다.
        self.draft_size = draft_size
        self.cls2idx = {c: i for i, c in enumerate(self.classes)}
        self.df = df[df[path_col].notna()].reset_index(drop=True)
        self.paths = self.df[path_col].tolist()
        self.transform = transform
        self.return_index = return_index

        if label_col in self.df.columns:
            self.targets = np.array([
                self.cls2idx.get(str(v), -1) for v in self.df[label_col]
            ])
        else:
            self.targets = np.full(len(self.df), -1)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        from PIL import Image

        try:
            with Image.open(self.paths[i]) as im:
                if self.draft_size:
                    im.draft("RGB", (self.draft_size, self.draft_size))
                img = im.convert("RGB")
        except Exception:
            img = Image.new("RGB", (256, 256), (128, 128, 128))
        x = self.transform(img) if self.transform else torch.from_numpy(
            np.array(img).transpose(2, 0, 1)
        ).float() / 255
        y = int(self.targets[i])
        return (x, y, i) if self.return_index else (x, y)


# ──────────────────────────────────────────────────────────────
# 불균형 대응
# ──────────────────────────────────────────────────────────────
def class_counts(ds: SkinDataset) -> np.ndarray:
    c = np.zeros(len(ds.classes), dtype=np.int64)
    for t in ds.targets:
        if 0 <= t < len(c):
            c[t] += 1
    return c


def class_weights(ds: SkinDataset, scheme: str = "inverse_sqrt") -> torch.Tensor:
    """손실 함수에 넣을 클래스 가중치.

    inverse(1/n)는 희소 클래스를 너무 과하게 밀어 학습이 불안정해지기 쉬워서,
    기본은 완만한 1/sqrt(n) 을 씁니다.
    """
    c = class_counts(ds).astype(np.float64)
    c[c == 0] = 1
    w = 1.0 / np.sqrt(c) if scheme == "inverse_sqrt" else 1.0 / c
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def weighted_sampler(ds: SkinDataset) -> WeightedRandomSampler:
    """희소 클래스를 더 자주 뽑는 샘플러.

    class_weight 와 동시에 쓰면 보정이 이중으로 걸립니다. 둘 중 하나만 쓰세요.
    """
    c = class_counts(ds).astype(np.float64)
    c[c == 0] = 1
    per = 1.0 / c
    w = np.array([per[t] if 0 <= t < len(per) else 0.0 for t in ds.targets])
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(w), replacement=True)


def hair_sampler(ds: SkinDataset, alpha: float = 1.0,
                 target_class: str = "A7", cache: Path | None = None,
                 verbose: bool = True) -> WeightedRandomSampler:
    """**털처럼 가는 선이 많은 정상 사진**을 더 자주 뽑습니다.

    왜 — 헛알림 실측 (`docs/results/헛알림_사진통계_실측.md`)
    -------------------------------------------------------
    정상인데 "병원 가보세요" 가 나온 사진 1,234장은 **`hair` 가 큽니다**
    (AUROC 0.749, d 0.80). 그리고 그 값은 견종(0.739)·부위(0.727)·촬영거리
    (0.747) 어느 것으로도 설명되지 않는 **사진 한 장의 성질**입니다.

    사진 단위 값이라 **샘플러 가중치로 그대로 들어갑니다.** 새 데이터도
    새 라벨도 필요 없습니다 — 이미 있는 정상 사진 중 **어려운 것**을 더
    자주 보여줘서 "털 ≠ 병변" 을 배우게 합니다.

    ★ 총량을 안 바꿉니다 — 이게 설계의 핵심입니다
    ---------------------------------------------
    가중치를 그냥 올리면 **정상 사진이 전체적으로 더 많이 뽑혀서**
    클래스 균형이 같이 바뀝니다. 그러면 좋아져도 "털 가중치 덕분" 인지
    "정상을 더 봐서" 인지 못 가릅니다 — 교란(confound)입니다.

    그래서 **클래스별 총 가중치를 1로 다시 맞춥니다.** 바뀌는 건
    *정상 안에서 누가 더 뽑히나* 뿐이고, 정상:이상 비율은 그대로입니다.

    Args:
        alpha: 세기. `0` 이면 균등(= 아무것도 안 함), `1` 이면 `hair` 최상위가
            최하위보다 **2배** 자주 뽑힙니다. 순위(백분위)를 쓰므로
            `hair` 의 단위·분포에 안 흔들립니다.
        target_class: 가중치를 걸 클래스. 기본은 정상(`A7`) — 헛알림이
            정상 쪽 오류이기 때문입니다. 놓친 병변에서는 `hair` 가
            신호가 없었습니다 (6개 값 전부 문턱 아래).
    """
    import numpy as np

    from src import texture

    w = np.ones(len(ds), dtype=np.float64)
    if alpha <= 0:
        if verbose:
            print("[data] hair_sampler alpha=0 — 균등 샘플링과 같습니다.")
        return WeightedRandomSampler(torch.as_tensor(w), len(w), replacement=True)

    if target_class not in ds.classes:
        raise ValueError(f"'{target_class}' 가 클래스에 없습니다: {ds.classes}")
    tgt = ds.classes.index(target_class)
    targets = np.asarray(ds.targets)
    mask = targets == tgt
    if mask.sum() < 2:
        raise ValueError(f"'{target_class}' 표본이 {int(mask.sum())}장뿐입니다.")

    paths = [str(pp) for pp in np.asarray(ds.paths)[mask]]
    hair = texture.hair_index(paths, cache=cache, verbose=verbose)

    # 순위 → 0~1 백분위. 값의 단위·치우침에 안 흔들립니다.
    order = hair.argsort().argsort().astype(np.float64)
    pct = order / max(len(order) - 1, 1)
    w[mask] = 1.0 + alpha * pct

    # ★ 클래스별 총량을 원래대로 — 이걸 빼면 클래스 균형이 같이 바뀝니다
    for t in np.unique(targets):
        m = targets == t
        w[m] *= m.sum() / w[m].sum()

    if verbose:
        q = np.quantile(hair, [0, .25, .5, .75, 1])
        print(f"[data] hair 가중 샘플러 alpha={alpha:g} · '{target_class}' "
              f"{int(mask.sum()):,}장")
        print(f"[data]   hair 분위 {np.round(q, 4).tolist()}")
        print(f"[data]   뽑힐 확률 최저 {w[mask].min():.3f} ~ 최고 "
              f"{w[mask].max():.3f} (평균 {w[mask].mean():.3f})")
        for t in np.unique(targets):
            m = targets == t
            print(f"[data]   {ds.classes[t]:<10} 총 가중치 {w[m].sum():>9,.1f} "
                  f"(장수 {int(m.sum()):,}) ← 총량 보존")
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                 len(w), replacement=True)


# ──────────────────────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────────────────────
def build_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: CFG,
    model=None,
    path_col: str = "crop_path",
    classes: list[str] | None = None,
) -> tuple[DataLoader, DataLoader, SkinDataset, SkinDataset]:
    tf_tr = transforms_for_model(cfg, model, True) if model else build_transforms(cfg, True)
    tf_va = transforms_for_model(cfg, model, False) if model else build_transforms(cfg, False)

    ds_tr = SkinDataset(train_df, tf_tr, path_col, classes=classes)
    # 검증은 어차피 Resize(img_size*1.14) 로 줄이므로 그 크기로 디코딩합니다.
    # full 크롭(512px)에서 매 에폭 검증이 2~3배 빨라지고, 결과는 사실상 같습니다.
    ds_va = SkinDataset(val_df, tf_va, path_col, classes=classes,
                        draft_size=int(cfg.img_size * 1.14))

    bs = cfg.resolved_batch_size()
    if cfg.balance_strategy == "weighted_sampler":
        sampler = weighted_sampler(ds_tr)
    elif cfg.balance_strategy == "hair_weighted":
        from src import env as _env

        sampler = hair_sampler(
            ds_tr, alpha=getattr(cfg, "hair_alpha", 1.0),
            # 캐시를 두면 실험을 여러 번 돌려도 hair 를 한 번만 잽니다
            cache=_env.work_root() / "reports" / "hair_index.parquet")
    else:
        sampler = None

    common = dict(
        num_workers=cfg.resolved_num_workers(),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.resolved_num_workers() > 0,
    )
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=sampler is None,
                       sampler=sampler, drop_last=True, **common)
    dl_va = DataLoader(ds_va, batch_size=bs * 2, shuffle=False, **common)

    print(f"[data] train {len(ds_tr):,} / val {len(ds_va):,}  batch={bs}  "
          f"balance={cfg.balance_strategy}")
    print(f"[data] 클래스별 학습 수: {dict(zip(ds_tr.classes, class_counts(ds_tr).tolist()))}")
    return dl_tr, dl_va, ds_tr, ds_va


def eval_loader(
    df: pd.DataFrame,
    cfg: CFG,
    model=None,
    path_col: str = "crop_path",
    classes: list[str] | None = None,
    batch_mult: int = 2,
) -> tuple[DataLoader, SkinDataset]:
    """평가 전용 로더 하나. **행 순서를 보존**합니다 (shuffle=False).

    2단계 파이프라인을 이어붙여 평가할 때 반드시 필요합니다:
    1단계 모델과 2단계 모델을 **같은 행, 같은 순서**로 돌려야 두 출력을 짝지을
    수 있습니다. 순서가 어긋나면 점수가 조용히 엉망이 됩니다 — 에러도 안 납니다.

    돌려주는 ds.df 를 정답의 출처로 쓰세요. SkinDataset 이 경로가 없는 행을
    걸러내므로, 원본 df 를 정답으로 쓰면 한 칸씩 밀릴 수 있습니다.
    """
    tf = transforms_for_model(cfg, model, False) if model else build_transforms(cfg, False)
    ds = SkinDataset(df, tf, path_col, classes=classes,
                     draft_size=int(cfg.img_size * 1.14))
    nw = cfg.resolved_num_workers()
    dl = DataLoader(
        ds,
        batch_size=max(cfg.resolved_batch_size() * batch_mult, 1),
        shuffle=False,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=4 if nw > 0 else None,
    )
    if len(ds) != len(df):
        print(f"⚠️ {len(df) - len(ds):,}행이 제외됐습니다 (크롭 파일 없음). "
              "정답은 ds.df 에서 가져오세요.")
    return dl, ds


# ──────────────────────────────────────────────────────────────
# Mixup / CutMix
# ──────────────────────────────────────────────────────────────
def mixup_cutmix(x: torch.Tensor, y: torch.Tensor, cfg: CFG, n_classes: int):
    """(mixed_x, y_a, y_b, lam) 을 돌려줍니다. 꺼져 있으면 lam=1.

    의료 이미지에서 Mixup 은 논쟁적입니다 — 존재하지 않는 병변 조합을
    만들어내기 때문입니다. 켜고/끄고 둘 다 실험해서 비교하세요.
    """
    if cfg.mixup_alpha <= 0 and cfg.cutmix_alpha <= 0:
        return x, y, y, 1.0

    use_cut = cfg.cutmix_alpha > 0 and (cfg.mixup_alpha <= 0 or np.random.rand() < 0.5)
    alpha = cfg.cutmix_alpha if use_cut else cfg.mixup_alpha
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)

    if use_cut:
        _, _, H, W = x.shape
        r = np.sqrt(1 - lam)
        cw, ch = int(W * r), int(H * r)
        cx, cy = np.random.randint(W), np.random.randint(H)
        x1, y1 = np.clip(cx - cw // 2, 0, W), np.clip(cy - ch // 2, 0, H)
        x2, y2 = np.clip(cx + cw // 2, 0, W), np.clip(cy + ch // 2, 0, H)
        x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    else:
        x = lam * x + (1 - lam) * x[idx]

    return x, y, y[idx], lam
