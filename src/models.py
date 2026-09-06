"""timm 백본 팩토리.

timm 모델 이름은 버전마다 바뀝니다. "1년 전 블로그에서 본 이름"이 지금은
없어서 노트북이 죽는 일이 흔합니다. 그래서 여기서는 **실행 시점에** 존재를
확인하고, 없으면 fallback 으로 조용히 갈아탑니다 (경고는 찍습니다).

    from src import models
    m = models.build("convnextv2_base", n_classes=6)
    models.available()          # 이 환경에서 실제로 쓸 수 있는 모델 목록
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from src.config import MODEL_BY_KEY, MODEL_ZOO, CFG, ModelSpec


# ──────────────────────────────────────────────────────────────
# 이름 해석
# ──────────────────────────────────────────────────────────────
def _exists(name: str) -> bool:
    import timm

    base = name.split(".")[0]
    try:
        if name in timm.list_models(pretrained=True):
            return True
        # 태그 없이 등록된 경우
        return base in timm.list_models()
    except Exception:
        return False


def resolve(spec: ModelSpec | str, verbose: bool = True) -> str:
    """spec 의 timm_name 이 없으면 fallback 을 순서대로 시도합니다."""
    import timm

    if isinstance(spec, str):
        spec = MODEL_BY_KEY.get(spec, ModelSpec(key=spec, timm_name=spec))

    for cand in [spec.timm_name, *spec.fallbacks]:
        if _exists(cand):
            if cand != spec.timm_name and verbose:
                print(f"⚠️ [models] '{spec.timm_name}' 없음 → '{cand}' 로 대체 "
                      f"(timm {timm.__version__})")
            return cand

    # 마지막 방어선: 접두어가 같은 아무 모델
    base = spec.timm_name.split(".")[0].split("_")[0]
    alt = [m for m in timm.list_models(pretrained=True) if m.startswith(base)]
    if alt:
        print(f"⚠️ [models] '{spec.timm_name}' 및 fallback 모두 없음 → '{alt[0]}' 사용")
        return alt[0]
    raise ValueError(
        f"'{spec.timm_name}' 을(를) 찾을 수 없습니다. `pip install -U timm` 후 다시 시도하거나 "
        f"timm.list_models('*{base}*') 로 이름을 확인하세요."
    )


def available(verbose: bool = True) -> list[dict]:
    """MODEL_ZOO 중 이 환경에서 실제로 쓸 수 있는 것들."""
    rows = []
    for s in MODEL_ZOO:
        try:
            name = resolve(s, verbose=False)
            ok = True
        except Exception:
            name, ok = "—", False
        rows.append({"key": s.key, "requested": s.timm_name, "resolved": name,
                     "ok": ok, "img_size": s.img_size, "note": s.note})
    if verbose:
        import timm

        print(f"timm {timm.__version__}\n")
        for r in rows:
            mark = "✅" if r["ok"] else "❌"
            same = "" if r["resolved"] == r["requested"] else f"  →  {r['resolved']}"
            print(f"{mark} {r['key']:<16} {r['requested']}{same}")
            print(f"    {r['note']}")
    return rows


# ──────────────────────────────────────────────────────────────
# 생성
# ──────────────────────────────────────────────────────────────
def build(
    spec: ModelSpec | str,
    n_classes: int,
    pretrained: bool = True,
    drop_rate: float = 0.2,
    drop_path_rate: float = 0.1,
    img_size: int | None = None,
    verbose: bool = True,
) -> nn.Module:
    """timm 모델을 만들고 분류 헤드를 n_classes 로 갈아 끼웁니다."""
    import timm

    if isinstance(spec, str):
        spec = MODEL_BY_KEY.get(spec, ModelSpec(key=spec, timm_name=spec))
    name = resolve(spec, verbose=verbose)

    kwargs: dict = dict(pretrained=pretrained, num_classes=n_classes, drop_rate=drop_rate)
    # drop_path 를 지원 안 하는 모델이 있어 실패 시 빼고 재시도합니다.
    try:
        model = timm.create_model(name, drop_path_rate=drop_path_rate, **kwargs)
    except TypeError:
        model = timm.create_model(name, **kwargs)
    except Exception as exc:
        if "img_size" in str(exc) and img_size:
            model = timm.create_model(name, img_size=img_size, **kwargs)
        else:
            raise

    if verbose:
        n_par = sum(p.numel() for p in model.parameters()) / 1e6
        pc = getattr(model, "pretrained_cfg", {}) or {}
        print(f"[models] {name}  |  {n_par:.1f}M params  |  "
              f"입력 {pc.get('input_size', '?')}  mean={tuple(round(x,3) for x in pc.get('mean', ()))}")
    return model


# ──────────────────────────────────────────────────────────────
# 파라미터 그룹 (layer-wise LR)
# ──────────────────────────────────────────────────────────────
def param_groups(model: nn.Module, cfg: CFG) -> list[dict]:
    """백본은 낮은 lr, 새로 만든 헤드는 높은 lr.

    사전학습된 백본은 이미 좋은 특징을 뽑고 있는데 높은 lr 로 흔들면
    그 지식을 망가뜨립니다(catastrophic forgetting). 헤드는 랜덤 초기화라
    빠르게 배워야 하고요. 파인튜닝의 기본 관례입니다.
    (백본을 얼리는 linear probe 에는 해당하지 않습니다 — 거기선 백본 lr 이 아예 없습니다)

    BatchNorm/LayerNorm/bias 에는 weight decay 를 걸지 않습니다 — 통상적인 관례로,
    이들에 decay 를 걸면 정규화 통계가 왜곡됩니다.
    """
    head_names = ("head", "fc", "classifier", "last_linear")
    bb_lr = cfg.lr * cfg.backbone_lr_mult

    # backbone_lr_mult == 0 은 "백본을 얼린다"(linear probe)는 뜻입니다.
    # lr 0 으로 두면 결과는 같지만 백본까지 역전파해 계산을 낭비합니다.
    # requires_grad 를 꺼서 실제로 건너뛰게 합니다.
    freeze_bb = cfg.backbone_lr_mult == 0

    groups = {
        "head_decay": {"params": [], "lr": cfg.lr, "weight_decay": cfg.weight_decay},
        "head_nodecay": {"params": [], "lr": cfg.lr, "weight_decay": 0.0},
        "bb_decay": {"params": [], "lr": bb_lr, "weight_decay": cfg.weight_decay},
        "bb_nodecay": {"params": [], "lr": bb_lr, "weight_decay": 0.0},
    }
    for n, p in model.named_parameters():
        is_head = any(h in n for h in head_names)
        if freeze_bb and not is_head:
            p.requires_grad = False
            continue
        if not p.requires_grad:
            continue
        no_decay = p.ndim <= 1 or n.endswith(".bias")
        key = ("head_" if is_head else "bb_") + ("nodecay" if no_decay else "decay")
        groups[key]["params"].append(p)

    return [g for g in groups.values() if g["params"]]


def freeze_backbone(model: nn.Module, freeze: bool = True) -> None:
    """헤드만 먼저 학습하는 warmup 용. 보통 1~2 에폭만 얼렸다 풉니다."""
    head_names = ("head", "fc", "classifier", "last_linear")
    for n, p in model.named_parameters():
        if not any(h in n for h in head_names):
            p.requires_grad = not freeze


# ──────────────────────────────────────────────────────────────
# EMA
# ──────────────────────────────────────────────────────────────
class ModelEMA:
    """가중치의 지수이동평균.

    학습 후반에 가중치가 최적점 주변을 진동할 때, 그 궤적의 평균이
    어느 한 순간의 가중치보다 대체로 더 좋습니다. 공짜로 0.3~1%p 정도 얻습니다.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        import copy

        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for e, m in zip(self.ema.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.detach(), alpha=1 - d)
            else:
                e.copy_(m)


# ──────────────────────────────────────────────────────────────
# 앙상블
# ──────────────────────────────────────────────────────────────
@dataclass
class Ensemble:
    """여러 모델의 logit 을 평균냅니다.

    서로 다른 구조(CNN + ViT)를 섞을 때 가장 효과가 큽니다 — 틀리는 방식이
    다르기 때문입니다. 같은 모델을 seed 만 바꿔 섞으면 이득이 작습니다.
    """

    models: list[nn.Module]
    weights: list[float] | None = None

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weights or [1.0] * len(self.models)
        s = sum(w)
        out = None
        for m, wi in zip(self.models, w):
            logit = m(x)
            out = logit * (wi / s) if out is None else out + logit * (wi / s)
        return out

    def eval(self):
        for m in self.models:
            m.eval()
        return self

    def to(self, device):
        for m in self.models:
            m.to(device)
        return self


def load_checkpoint(path: str, spec: ModelSpec | str, n_classes: int,
                    device: str | None = None) -> nn.Module:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("ema") or ckpt.get("model") or ckpt
    model = build(spec, n_classes, pretrained=False, verbose=False)
    try:
        missing, unexpected = model.load_state_dict(state, strict=False)
    except RuntimeError as exc:
        # ⚠️ strict=False 는 **키 불일치**만 봐줍니다. 텐서 **모양**이 다르면 터집니다.
        #    거의 항상 원인은 하나 — 다른 백본의 가중치를 부은 것입니다.
        #    실제로 effnetv2_s 체크포인트를 resnet50 껍데기에 붓다가 죽었습니다.
        want = getattr(spec, "key", spec)
        raise RuntimeError(
            f"\n체크포인트를 '{want}' 에 못 넣었습니다 — **백본이 다릅니다.**\n"
            f"  파일: {path}\n\n"
            "  체크포인트 폴더 이름에 어떤 백본으로 학습했는지 적혀 있습니다:\n"
            "      stage1_effnetv2_s_full_384_moderate  →  effnetv2_s\n"
            "      stage2_resnet50_m2.5_384_moderate    →  resnet50\n"
            "  `train.model_key_from_exp(<폴더이름>)` 이 그걸 읽어 줍니다.\n\n"
            f"  원본 오류: {str(exc).splitlines()[0]}"
        ) from exc
    if missing or unexpected:
        warnings.warn(f"state_dict 불일치 — missing={len(missing)}, unexpected={len(unexpected)}")
    return model.to(device).eval()


# ──────────────────────────────────────────────────────────────
# 임베딩 (마지막 직전 특징)
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def embed(model: nn.Module, loader, device: str = "cuda",
          verbose: bool = True) -> np.ndarray:
    """분류 헤드 **직전**의 특징 벡터를 뽑습니다. 행 순서는 로더 그대로입니다.

    왜 필요한가
    -----------
    STEP 18 에서 "A4 는 결정 규칙이 아니라 **표현**의 문제" 까지 왔습니다
    (로짓 보정으로 prior 를 걷어내도 macro-F1 이 안 오름). 그 다음 질문은
    **어떻게** 실패하느냐입니다:

        · A4 가 특징 공간에서 **뭉쳐 있는데** 분류기가 못 쓰는 것  → 헤드·손실 문제
        · A4 가 애초에 **흩어져 있는 것**                        → 특징·데이터 문제

    앞이면 싸게 고쳐지고 뒤면 크롭·데이터를 손봐야 합니다. 로짓만으로는
    구분이 안 되고, 임베딩을 봐야 갈립니다.

    ⚠️ timm 모델마다 헤드 모양이 달라서 `forward_features` → `forward_head(
    pre_logits=True)` 순서로 부릅니다. 그게 안 되는 모델은 헤드를 `Identity`
    로 갈아 끼우고 한 번 통과시킵니다 (원본은 안 건드립니다 — 복원합니다).
    """
    model = model.to(device).eval()
    feats: list[np.ndarray] = []

    def _pre_logits(x):
        f = model.forward_features(x)
        return model.forward_head(f, pre_logits=True)

    use_head = True
    try:                                    # 한 배치로 먼저 확인
        probe = next(iter(loader))[0][:1].to(device)
        _pre_logits(probe)
    except Exception:
        use_head = False

    saved = None
    if not use_head:
        # ⚠️ 헤드 이름은 모델마다 다릅니다 — param_groups 와 **같은 목록**을 씁니다.
        for name in ("head", "fc", "classifier", "last_linear"):
            if hasattr(model, name):
                saved = (name, getattr(model, name))
                setattr(model, name, nn.Identity())
                break
        if saved is None:
            raise RuntimeError(
                "임베딩을 못 뽑습니다 — forward_head(pre_logits=True) 도 안 되고 "
                f"헤드 속성({('head','fc','classifier','last_linear')})도 없습니다.")

    try:
        for batch in loader:
            x = batch[0].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                                enabled=device.startswith("cuda")):
                f = _pre_logits(x) if use_head else model(x)
            if f.ndim > 2:                  # (B,C,H,W) 가 나오면 평균 풀링
                f = f.mean(dim=tuple(range(2, f.ndim)))
            feats.append(f.float().cpu().numpy())
    finally:
        if saved is not None:
            setattr(model, saved[0], saved[1])

    out = np.concatenate(feats, 0)
    if verbose:
        print(f"[models] 임베딩 {out.shape}  (헤드 {'통과' if use_head else 'Identity 교체'})")
    return out
