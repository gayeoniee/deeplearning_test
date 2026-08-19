"""프로젝트 전역 설정 — 모든 "숫자"는 여기 한 곳에 모읍니다.

노트북에서 하이퍼파라미터를 직접 고치지 마세요. 여기서 고치고 노트북은 읽기만 하면
어떤 설정으로 어떤 결과가 나왔는지 나중에 추적할 수 있습니다.

    from src.config import CFG, CLASSES
    cfg = CFG()                          # 기본값
    cfg = CFG(img_size=384, epochs=20)   # 일부만 바꾸기
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# ──────────────────────────────────────────────────────────────
# 클래스 정의
#
# ⚠️ 매우 중요: 이 라벨들은 "병명"이 아니라 "병변의 형태(morphology)"입니다.
#   A2 가 나왔다고 "이 강아지는 지루성 피부염" 이 아니라
#   "비듬·각질 형태의 병변이 보인다" 까지가 모델이 말할 수 있는 전부입니다.
#   같은 병변 형태가 여러 질환에서 나오고, 같은 질환이 여러 형태로 나타납니다.
#   → docs/data/병변_6종_임상_해설.md 참고
# ──────────────────────────────────────────────────────────────
CLASSES: list[str] = ["A1", "A2", "A3", "A4", "A5", "A6"]

CLASS_KO: dict[str, str] = {
    "A1": "구진·플라크",
    "A2": "비듬·각질·상피성잔고리",
    "A3": "태선화·과다색소침착",
    "A4": "농포·여드름",
    "A5": "미란·궤양",
    "A6": "결절·종괴",
}

CLASS_EN: dict[str, str] = {
    "A1": "Papule / Plaque",
    "A2": "Scale / Crust / Epidermal collarette",
    "A3": "Lichenification / Hyperpigmentation",
    "A4": "Pustule / Acne",
    "A5": "Erosion / Ulcer",
    "A6": "Nodule / Mass",
}

# 1단계(정상/이상) 이진 분류용 라벨. 무증상 데이터 존재 여부는 STEP 2 스캔으로 확정합니다.
NORMAL_LABEL = "A0"
CLASS_KO[NORMAL_LABEL] = "무증상(정상)"
CLASS_EN[NORMAL_LABEL] = "Normal / Asymptomatic"

# 병변별 임상적 긴급도 힌트 — "의심된다"의 톤을 조절할 때 씁니다.
# 진단이 아니라 안내 문구의 강도를 정하는 용도일 뿐입니다.
URGENCY_HINT: dict[str, str] = {
    "A1": "관찰",
    "A2": "관찰",
    "A3": "만성 경과 가능 — 진료 권장",
    "A4": "감염 동반 가능 — 진료 권장",
    "A5": "피부 장벽 손상 — 조기 진료 권장",
    "A6": "종양 감별 필요 — 조기 진료 권장",
}

# AI Hub 데이터셋 식별자
AIHUB_DATASET_KEY = "561"
AIHUB_DATASET_NAME = "반려동물 피부 질환 데이터"

# 우리가 쓸 데이터 범위 (사용자 결정: 반려견 + 일반카메라만)
INCLUDE_SPECIES = ["반려견"]
EXCLUDE_SPECIES = ["반려묘"]
INCLUDE_CAMERA = ["일반카메라"]
EXCLUDE_CAMERA = ["더모스코프"]  # 보호자가 만들 수 없는 입력이므로 제외


# ──────────────────────────────────────────────────────────────
# 실행 설정
# ──────────────────────────────────────────────────────────────
@dataclass
class CFG:
    # --- 재현성 ---
    seed: int = 42
    deterministic: bool = False

    # --- 데이터 ---
    img_size: int = 288
    crop_margin: float = 1.5          # ROI 크롭 여유 배율. 1.0=박스 딱 맞게, 2.0=주변 2배
    crop_min_px: int = 64             # 이보다 작은 병변 박스는 버림 (노이즈)
    save_crop_size: int = 512         # 디스크에 저장할 크롭 해상도 (학습 시 img_size 로 리사이즈)
    save_crop_quality: int = 92

    # --- 중복 제거 ---
    phash_size: int = 16              # phash 비트 크기 (16 → 256bit, 기본 8보다 정밀)
    dedup_hamming: int = 6            # 이 거리 이하면 near-duplicate 로 봄

    # --- 분할 ---
    n_folds: int = 5
    use_fold: int = 0                 # 단일 실험에서 검증에 쓸 fold
    holdout_ratio: float = 0.15       # 최종 1회만 보는 테스트셋 비율 (개체 단위)

    # --- 모델 ---
    model_name: str = "tf_efficientnetv2_s.in21k_ft_in1k"
    pretrained: bool = True
    drop_rate: float = 0.2
    drop_path_rate: float = 0.1

    # --- 학습 ---
    epochs: int = 15
    batch_size: int = 0               # 0 이면 env.suggest_batch_size() 로 자동
    grad_accum: int = 1
    lr: float = 3e-4
    backbone_lr_mult: float = 0.1     # 백본은 헤드보다 낮은 lr (전이학습 관례)
    weight_decay: float = 0.05
    warmup_epochs: int = 2
    label_smoothing: float = 0.1
    amp: bool = True
    ema_decay: float = 0.999          # 0 이면 EMA 끔
    clip_grad_norm: float = 1.0
    num_workers: int = 2
    early_stop_patience: int = 5
    monitor: str = "macro_f1"         # ⚠️ accuracy 아님. 불균형 데이터에서 accuracy 는 거짓말을 합니다.

    # --- 증강 ---
    # ⚠️ 피부 병변은 "색과 질감"이 곧 라벨입니다.
    #    강한 색상 증강은 A3(과다색소침착)을 A1 처럼 만들어 라벨을 파괴합니다.
    #    아래 값은 일반 이미지 분류 기본값보다 의도적으로 약하게 잡았습니다.
    rrc_scale: tuple[float, float] = (0.7, 1.0)
    hflip: float = 0.5
    vflip: float = 0.0                # 피부 사진은 위아래 뒤집기가 부자연스러움
    rotate_deg: int = 15
    color_jitter: float = 0.1         # brightness/contrast/saturation 공통 강도 (약하게!)
    hue_jitter: float = 0.02          # 색조는 특히 조심 — 거의 건드리지 않음
    randaugment_n: int = 0            # 0 이면 끔. 켜려면 2 권장
    randaugment_m: int = 7
    mixup_alpha: float = 0.0          # 실험용. 켜면 0.2 권장
    cutmix_alpha: float = 0.0
    random_erasing: float = 0.15

    # --- 불균형 대응 ---
    balance_strategy: str = "class_weight"   # "none" | "class_weight" | "weighted_sampler"
    focal_gamma: float = 0.0                 # >0 이면 focal loss 사용

    # --- 평가 / 안전장치 ---
    tta_hflip: bool = True
    calibrate: bool = True                   # 온도 스케일링
    target_recall_stage1: float = 0.95       # 1단계 정상/이상: 놓치지 않는 게 우선
    abstain_threshold: float = 0.45          # 최고확률이 이 미만이면 "판단 어려움"
    topk_report: int = 3                     # 사용자에게 상위 몇 개까지 보여줄지

    # --- 로깅 ---
    exp_name: str = "baseline"
    log_every: int = 50
    use_wandb: bool = False

    def resolved_batch_size(self) -> int:
        if self.batch_size > 0:
            return self.batch_size
        from src import env

        scale = _infer_scale(self.model_name)
        return env.suggest_batch_size(self.img_size, scale)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        import json

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return p

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CFG":
        """알 수 없는 키는 무시하고 만듭니다 (설정 파일이 구버전이어도 동작)."""
        data = dict(data)
        # tuple 필드 복원 (JSON 은 list 로 저장됨)
        if isinstance(data.get("rrc_scale"), list):
            data["rrc_scale"] = tuple(data["rrc_scale"])
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def load(cls, path: str | Path) -> "CFG":
        import json

        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _infer_scale(model_name: str) -> str:
    n = model_name.lower()
    for tag in ("nano", "tiny", "small", "base", "large", "huge"):
        if tag in n:
            return {"nano": "tiny", "huge": "large"}.get(tag, tag)
    # efficientnet_b0..b7 같은 이름 처리
    if "_b0" in n or "_b1" in n or "_s." in n or n.endswith("_s"):
        return "small"
    if "_b2" in n or "_b3" in n or "_m." in n:
        return "base"
    return "base"


# ──────────────────────────────────────────────────────────────
# 모델 라인업 — STEP 4 에서 순서대로 돌립니다.
#
# timm 모델명은 버전마다 바뀝니다. src/models.py 가 실행 시점에
# timm.list_models() 로 존재를 검증하고, 없으면 fallback 을 씁니다.
# ──────────────────────────────────────────────────────────────
@dataclass
class ModelSpec:
    key: str
    timm_name: str
    fallbacks: list[str] = field(default_factory=list)
    img_size: int = 288
    scale: str = "base"
    note: str = ""


MODEL_ZOO: list[ModelSpec] = [
    ModelSpec(
        key="resnet50",
        timm_name="resnet50.a1_in1k",
        fallbacks=["resnet50"],
        img_size=224, scale="base",
        note="기준선. 다른 모든 수치는 이것과 비교해서 읽습니다.",
    ),
    ModelSpec(
        key="effnetv2_s",
        timm_name="tf_efficientnetv2_s.in21k_ft_in1k",
        fallbacks=["tf_efficientnetv2_s", "efficientnet_b3"],
        img_size=300, scale="small",
        note="가볍고 강함. 모바일 배포 1순위 후보.",
    ),
    ModelSpec(
        key="convnextv2_base",
        timm_name="convnextv2_base.fcmae_ft_in22k_in1k",
        fallbacks=["convnextv2_tiny.fcmae_ft_in22k_in1k", "convnext_base.fb_in22k_ft_in1k"],
        img_size=288, scale="base",
        note="현대 CNN 최강급. 질감(texture) 표현이 좋아 피부에 잘 맞을 가능성이 큼.",
    ),
    ModelSpec(
        key="swinv2_base",
        timm_name="swinv2_base_window12to16_192to256.ms_in22k_ft_in1k",
        fallbacks=["swinv2_base_window8_256.ms_in1k", "swin_base_patch4_window7_224.ms_in22k_ft_in1k"],
        img_size=256, scale="base",
        note="계층적 ViT. 지역 패턴 + 전역 문맥을 같이 봄.",
    ),
    ModelSpec(
        key="eva02_base",
        timm_name="eva02_base_patch14_448.mim_in22k_ft_in22k_in1k",
        fallbacks=["eva02_small_patch14_336.mim_in22k_ft_in1k", "vit_base_patch16_224.augreg2_in21k_ft_in1k"],
        img_size=336, scale="base",
        note="정확도 상한 확인용. T4 에서는 무거우니 배치 작게 + grad_accum 사용.",
    ),
    ModelSpec(
        key="siglip2_base",
        timm_name="vit_base_patch16_siglip_256.v2_webli",
        fallbacks=["vit_base_patch16_siglip_224.webli", "vit_base_patch16_clip_224.openai"],
        img_size=256, scale="base",
        note="대규모 이미지-텍스트 사전학습 백본. 소량 데이터에서 특히 강한 편.",
    ),
]

MODEL_BY_KEY = {m.key: m for m in MODEL_ZOO}
