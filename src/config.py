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

# 1단계(정상/이상) 이진 분류용 라벨.
# ✅ 실물 확인 결과 무증상 데이터가 존재합니다 — metaData.lesions == "A7".
#    유증상 26,191 / 무증상 28,042 로 거의 반반이라 2단계 모델이 가능합니다.
#    ⚠️ 무증상 이미지가 'A1_구진_플라크' 같은 폴더 안에 들어 있습니다.
#       폴더명이 아니라 metaData.lesions 를 봐야 합니다.
NORMAL_LABEL = "A7"
CLASS_KO[NORMAL_LABEL] = "무증상(정상)"
CLASS_EN[NORMAL_LABEL] = "Normal / Asymptomatic"

# 1단계에서 쓰는 클래스 (정상 vs 이상)
CLASSES_STAGE1 = [NORMAL_LABEL, "ABNORMAL"]
# 2단계에서 쓰는 클래스 (병변 6종) = CLASSES

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
    # >0 이면 margin 대신 **고정 픽셀 창**으로 자릅니다 (병변 중심, 항상 같은 크기).
    # margin 크롭은 병변 크기에 따라 확대 배율이 달라져 그 배율이 정답을 흘립니다.
    # → src/crop.py 의 fixed_box() 설명, docs/cautions/08 참고
    crop_fixed_px: int = 0
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
    backbone_lr_mult: float = 0.1     # 백본은 헤드보다 낮은 lr (파인튜닝 관례)
    weight_decay: float = 0.05
    warmup_epochs: int = 2
    label_smoothing: float = 0.1
    amp: bool = True
    ema_decay: float = 0.999          # 0 이면 EMA 끔
    clip_grad_norm: float = 1.0
    num_workers: int = -1        # -1 = CPU 코어 수에 맞춰 자동
    early_stop_patience: int = 5
    monitor: str = "macro_f1"         # ⚠️ accuracy 아님. 불균형 데이터에서 accuracy 는 거짓말을 합니다.

    # --- 증강 ---
    # ⚠️ 피부 병변은 "색과 질감"이 곧 라벨입니다.
    #    강한 색상 증강은 A3(과다색소침착)을 A1 처럼 만들어 라벨을 파괴합니다.
    #    아래 값은 일반 이미지 분류 기본값보다 의도적으로 약하게 잡았습니다.
    rrc_scale: tuple[float, float] = (0.7, 1.0)
    # ⚠️ RandomResizedCrop 은 **축소를 못 합니다.** 이미지의 일부를 잘라 확대할 뿐이라
    #    가장 축소된 경우가 "이미지 전체"(검증 대비 약 0.88배)이고, rrc_scale 하한을
    #    낮추면 **확대 쪽만** 넓어집니다. 실측:
    #        default(0.70,1.0)      → 0.88x ~ 1.05x
    #        scale_robust(0.35,1.0) → 0.88x ~ 1.49x
    #    그런데 배율 교란 검사는 0.5x·0.71x 를 묻습니다. 훈련에서 **한 번도 안 본**
    #    구간이라, rrc_scale 을 아무리 넓혀도 그 하락은 안 줄어듭니다 (실측 확인).
    #    축소를 배우려면 이미지를 줄여 여백을 채우는 affine 변환이 필요합니다.
    affine_scale: tuple[float, float] | None = None   # 예: (0.5, 1.3) — 축소 포함
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

    # --- albumentations 전용 노브 ---
    # torchvision 에는 없거나 느린 것들. albumentations 가 없으면 전부 무시됩니다.
    # ⚠️ 촬영 조건(흐림·노이즈·조명)을 흉내 내는 쪽입니다. 실측: 정상 사진의
    #    선명도 중앙값이 39, 병변은 271 — 모델이 화질로 맞힐 여지가 있어서
    #    흐림을 훈련에 넣어두면 그 지름길을 막는 효과도 기대할 수 있습니다.
    blur_p: float = 0.0               # 가우시안/모션 블러 확률
    noise_p: float = 0.0              # 센서 노이즈 확률
    jpeg_p: float = 0.0               # JPEG 압축 열화 (보호자 사진은 대개 압축됨)
    clahe_p: float = 0.0              # 국소 대비 보정 — 질감을 살리는 방향
    shift_limit: float = 0.0          # 평행이동 비율 (위치 교란 20.6% 대응)

    # --- 불균형 대응 ---
    balance_strategy: str = "class_weight"
    # "none" | "class_weight" | "weighted_sampler" | "hair_weighted"
    #   hair_weighted — 털처럼 가는 선이 많은 **정상** 사진을 더 자주 뽑습니다.
    #   헛알림 실측(AUROC 0.749)에서 나온 값이고, 클래스 총량은 보존합니다.
    hair_alpha: float = 1.0   # 0=끔, 1=최상위가 최하위보다 2배 자주
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
        # 백본별 메모리 보정 — timm 이름으로 MODEL_ZOO 를 되짚습니다
        mf = 1.0
        for spec in MODEL_ZOO:
            if spec.timm_name == self.model_name or spec.key == self.model_name:
                mf = spec.mem_factor
                break
        return env.suggest_batch_size(self.img_size, scale, mem_factor=mf)

    def resolved_num_workers(self) -> int:
        """-1 이면 CPU 코어 수에 맞춰 정합니다."""
        if self.num_workers >= 0:
            return self.num_workers
        from src import env

        return env.suggest_workers()

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
        if isinstance(data.get("affine_scale"), list):
            data["affine_scale"] = tuple(data["affine_scale"])
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
# 증강 프리셋
#
# 기본값은 의도적으로 약합니다 — 피부 병변은 색과 질감이 곧 라벨이라
# 강한 색상 증강은 A3(과다색소침착)를 A1 처럼 만들어 버립니다.
#
# 그런데 **배율**은 사정이 다릅니다. 크롭이 병변 박스에 맞춰 잘리기 때문에
# 배율이 클래스마다 다르고(실측 A1 0.47% ~ A6 3.08%, 6.5배), 모델이 그걸
# 단서로 쓸 수 있습니다. 실사용에서 배율은 무작위이므로 그건 무너집니다.
#
# ⚠️ 기본 rrc_scale=(0.7, 1.0) 은 면적 1.43배 범위(선형 1.2배)입니다.
#    막아야 할 격차가 선형 2.5배인데 이건 아무 효과가 없습니다.
#
# ⚠️ 넓은 배율 증강에는 **여유가 있는 크롭**이 필요합니다.
#    m1.5 처럼 딱 붙은 크롭에 0.15 를 걸면 병변이 화면에서 잘려 나가
#    라벨이 깨집니다. m2.5 나 f512 같은 넉넉한 크롭을 base 로 쓰세요.
#    효과 판정은 src/robust.py 의 scale_stress() 로 합니다.
# ──────────────────────────────────────────────────────────────
AUG_PRESETS: dict[str, dict] = {
    # ── 기준 ────────────────────────────────────────────────────
    "default": {},

    # ── ① 폭을 **줄이는** 쪽 (멘토 피드백 1번) ──────────────────
    # 지금까지 넓히기만 두 번 시도해 둘 다 실패했습니다. 반대 방향은 안 해봤습니다.
    # 좁히면 과제가 쉬워져 점수가 오를 수 있고, 대신 견고성은 나빠질 수 있습니다.
    # 좁히기 축에 **점을 두 개** 찍습니다. 하나만 찍으면 "이겼다" 는 알아도
    # "더 좁혀야 하나" 를 몰라서 또 한 판 돌려야 합니다.
    #     0.70(default) → 0.85(narrow) → 0.92(narrower)
    #   계속 좋아지면 더 좁히고, narrow 가 최고면 거기가 최적점이고,
    #   셋이 비슷하면 이 축은 상관없다는 뜻입니다. 한 번에 결론이 납니다.
    "narrow": {"rrc_scale": (0.85, 1.0), "rotate_deg": 10, "random_erasing": 0.0},
    "narrower": {"rrc_scale": (0.92, 1.0), "rotate_deg": 5, "random_erasing": 0.0},

    # ── ② 축소를 가르치는 쪽 (2단계 최악 조건이 0.5x) ───────────
    # affine 이 이미지를 실제로 줄이고 여백을 채웁니다. RRC 는 축소를 못 합니다.
    "zoom_both": {                       # 세게 — 224px 에서는 실패했던 설정
        "rrc_scale": (0.5, 1.0),
        "affine_scale": (0.45, 1.25),
        "rotate_deg": 20,
        "random_erasing": 0.25,
    },
    "zoom_mild": {                       # 완만하게 — ①과 ②의 절충
        "rrc_scale": (0.75, 1.0),
        "affine_scale": (0.70, 1.15),
        "rotate_deg": 12,
    },

    # ── ③ 위치 교란 대응 (실측 20.6% 하락) ─────────────────────
    "shift": {"shift_limit": 0.15, "rotate_deg": 15},

    # ── ④ 촬영 조건 흉내 (멘토 피드백 4번) ─────────────────────
    # 배율은 기본값 그대로 두고 화질만 흔듭니다.
    # 실측: 정상 사진 선명도 중앙값 39 vs 병변 271 — 모델이 화질로 맞힐 여지가
    # 있어서, 흐림을 훈련에 넣으면 그 지름길을 막는 효과도 기대할 수 있습니다.
    "photometric": {"blur_p": 0.3, "noise_p": 0.25, "jpeg_p": 0.3, "clahe_p": 0.2},

    # ── ⑤ 조합 ─────────────────────────────────────────────────
    # ⚠️ 아래 둘은 정의만 남겨둡니다 — 03b 스윕에서는 뺐습니다.
    #    zoom_shift 는 zoom_mild + shift 결과로 대충 예상되고,
    #    kitchen_sink 는 "너무 세면 나빠진다" 를 확인하는 대조군이라 기대값이 낮습니다.
    "zoom_shift": {                      # 축소 + 이동 (두 교란을 같이)
        "rrc_scale": (0.75, 1.0),
        "affine_scale": (0.70, 1.15),
        "shift_limit": 0.12,
        "rotate_deg": 15,
    },
    "kitchen_sink": {                    # 전부 — 과한 게 해로운지 확인하는 상한선
        "rrc_scale": (0.6, 1.0),
        "affine_scale": (0.55, 1.25),
        "shift_limit": 0.15,
        "rotate_deg": 20,
        "blur_p": 0.25, "noise_p": 0.2, "jpeg_p": 0.25,
        "random_erasing": 0.25,
    },

    # ── 과거 실험용 (결론 남음. 참고로만 둡니다) ────────────────
    # 확대만 넓힙니다. 2단계 최악 조건이 축소라 여기엔 효과가 없습니다.
    "scale_robust": {"rrc_scale": (0.35, 1.0), "rotate_deg": 20, "random_erasing": 0.25},
}


# 파인튜닝 강도 프리셋
#
# ⚠️ 용어 정리 — 자주 헷갈립니다.
#   "전이학습(transfer learning)" 은 우산 개념입니다. 그 안에 두 방식이 있습니다:
#     · linear probe   백본을 얼리고 헤드만 학습 (freeze_backbone(True))
#     · fine-tuning    백본까지 같이 학습 ← **우리는 처음부터 이쪽입니다**
#   `freeze_backbone()` 은 정의만 되어 있고 어디서도 호출하지 않습니다.
#
# 그러면 남는 질문은 "파인튜닝을 할까?" 가 아니라 "얼마나 세게 할까?" 입니다.
# 그걸 정하는 게 backbone_lr_mult 입니다:
#
#     헤드 lr   = cfg.lr                        (랜덤 초기화라 빨리 배워야 함)
#     백본 lr   = cfg.lr × backbone_lr_mult     (사전학습 지식을 지키려고 낮춤)
#
# 기본 0.1 은 "ImageNet 특징이 이미 쓸만하다" 는 전제입니다. 그런데 우리 과제는
# 물체 인식이 아니라 **피부 질감·색의 미세한 구분**이라 도메인 격차가 큽니다.
# 백본이 3e-5 로 움직이면 12 에폭 동안 거의 제자리입니다.
#
# 실측 근거 (VL01 2단계): train 1.352 / val 1.474 — 학습 데이터조차 잘 못 맞춥니다.
# 과적합이 아니라 **덜 배운** 상태이고, 백본 lr 이 유력한 원인입니다.
# ──────────────────────────────────────────────────────────────
FT_PRESETS: dict[str, dict] = {
    # 지금까지 쓰던 설정 (비교 기준)
    "conservative": {"backbone_lr_mult": 0.1},

    # 백본을 3배 더 움직입니다. 도메인 격차가 클 때의 표준적인 선택.
    "moderate": {"backbone_lr_mult": 0.3, "warmup_epochs": 3},

    # 백본과 헤드를 같은 lr 로. 격차가 아주 클 때 가장 좋을 수 있지만
    # 사전학습 지식을 잃을 위험(catastrophic forgetting)이 있어 warmup 을 길게 둡니다.
    "aggressive": {"backbone_lr_mult": 1.0, "warmup_epochs": 4, "lr": 1e-4},

    # 백본을 얼리고 헤드만. 우리 데이터(1.5만장)에는 부족하지만,
    # "백본 적응이 실제로 기여하는가" 를 재는 대조군으로 유용합니다.
    "linear_probe": {"backbone_lr_mult": 0.0},
}


def ft_preset(name: str) -> dict:
    """파인튜닝 강도 프리셋 → CFG 오버라이드 사전."""
    if name not in FT_PRESETS:
        raise KeyError(f"모르는 프리셋: {name}. 가능: {sorted(FT_PRESETS)}")
    return dict(FT_PRESETS[name])


def with_finetune(cfg: "CFG", name: str) -> "CFG":
    """cfg 에 파인튜닝 강도 프리셋을 얹은 새 CFG."""
    d = {**cfg.to_dict(), **ft_preset(name)}
    if name != "conservative":
        d["exp_name"] = f"{cfg.exp_name}_{name}"
    return CFG.from_dict(d)


def aug_preset(name: str) -> dict:
    """프리셋 이름 → CFG 오버라이드 사전.

        cfg = CFG(**{**CFG(model_name="resnet50").to_dict(), **aug_preset("scale_robust")})
    """
    if name not in AUG_PRESETS:
        raise KeyError(f"모르는 프리셋: {name}. 가능: {sorted(AUG_PRESETS)}")
    return dict(AUG_PRESETS[name])


def with_aug(cfg: "CFG", name: str) -> "CFG":
    """cfg 에 증강 프리셋을 얹은 새 CFG. 실험 이름에 프리셋을 붙여 둡니다."""
    over = aug_preset(name)
    d = {**cfg.to_dict(), **over}
    if name != "default":
        d["exp_name"] = f"{cfg.exp_name}_{name}"
    return CFG.from_dict(d)


# ──────────────────────────────────────────────────────────────
# 노트북 버전
# ──────────────────────────────────────────────────────────────
# ⚠️ 노트북 셀은 `git pull` 로 갱신되지 않습니다. Colab/Kaggle 에 올린 .ipynb 는
#    다시 import 하기 전까지 그대로입니다. src/ 만 매번 최신이 됩니다.
#    그래서 셀을 고칠 때마다 이 값을 올리고, 노트북 첫 셀이 자기가 들고 있는
#    값과 비교해 **낡았으면 바로 알립니다.** (몇 시간 뒤에 알게 되면 늦습니다)
NOTEBOOK_VERSION = "2026-09-04.1"

# ★ 채택된 2단계 크롭. STEP 4C(비교) → 4D(재기준선) 에서 확정했습니다.
#   노트북 05 가 불러온 체크포인트의 크롭이 이것과 다르면 **멈춥니다** —
#   실제로 예전 실행(m1.5)의 출력을 붙이고 그대로 진행할 뻔했습니다.
#   그러면 촬영 가이드·보정·임계값이 전부 버린 설정 기준으로 나옵니다.
ADOPTED_STAGE2_CROP = "m2.5"

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
    # 배치 추천을 몇 배로 줄일지. env.suggest_batch_size 의 공식이 **ResNet 기준**
    # 이라 트랜스포머 계열의 활성값 메모리를 모릅니다. 실측으로 정했습니다:
    #   T4(14.6GB) · 256px · 배치 32 에서
    #     swinv2_base   → OOM
    #     siglip2_base  → 14.3GB (98%, EMA 붙으면 터짐)
    #   같은 조건의 convnextv2_base 는 384px 배치 12 에서 8.9GB 로 멀쩡했습니다.
    # 그래서 어텐션 계열은 0.4 로 잡습니다 (32 → 12).
    mem_factor: float = 1.0
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
        mem_factor=0.4,
        note="계층적 ViT. 지역 패턴 + 전역 문맥을 같이 봄.",
    ),
    ModelSpec(
        key="eva02_base",
        timm_name="eva02_base_patch14_448.mim_in22k_ft_in22k_in1k",
        fallbacks=["eva02_small_patch14_336.mim_in22k_ft_in1k", "vit_base_patch16_224.augreg2_in21k_ft_in1k"],
        img_size=336, scale="base",
        mem_factor=0.4,
        note="정확도 상한 확인용. T4 에서는 무거우니 배치 작게 + grad_accum 사용.",
    ),
    ModelSpec(
        key="siglip2_base",
        timm_name="vit_base_patch16_siglip_256.v2_webli",
        fallbacks=["vit_base_patch16_siglip_224.webli", "vit_base_patch16_clip_224.openai"],
        img_size=256, scale="base",
        mem_factor=0.4,
        note="대규모 이미지-텍스트 사전학습 백본. 소량 데이터에서 특히 강한 편.",
    ),
]

MODEL_BY_KEY = {m.key: m for m in MODEL_ZOO}
