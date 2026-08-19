"""Grad-CAM — 모델이 "어디를 보고" 그렇게 답했는지 확인.

이건 예쁜 그림을 만드는 기능이 아니라 **필수 검증 게이트**입니다.

왜냐하면:
  이 데이터는 병변이 이미지의 5% 미만인 경우가 대부분이고 배경이 제각각입니다.
  모델이 병변이 아니라 "진료대 바닥무늬", "털 색", "조명"을 보고 맞히는 일이
  충분히 일어날 수 있습니다. 그리고 그 단서가 클래스와 상관관계가 있으면
  **검증 점수까지 잘 나옵니다.** 숫자만 보면 절대 못 잡습니다.

그래서 규칙: macro-F1 이 아무리 좋아도 Grad-CAM 이 병변을 안 보고 있으면 실패로 간주.

    from src import explain
    explain.grid(model, df_val, n=8)
"""

from __future__ import annotations

import json

import numpy as np
import torch

from src.config import CFG, CLASS_KO


# ──────────────────────────────────────────────────────────────
# 타깃 레이어 찾기
# ──────────────────────────────────────────────────────────────
def find_target_layer(model: torch.nn.Module) -> list:
    """모델 구조에 상관없이 CAM 을 걸 마지막 특징 레이어를 찾습니다."""
    # timm 모델은 대부분 이 중 하나를 가지고 있습니다
    for attr in ("stages", "blocks", "layer4", "features", "layers"):
        mod = getattr(model, attr, None)
        if mod is not None and len(list(mod.children())) > 0:
            return [list(mod.children())[-1]]
    # 마지막 방어선: 컨볼루션/노름 계열 마지막 모듈
    cands = [m for m in model.modules()
             if isinstance(m, (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.LayerNorm))]
    return [cands[-1]] if cands else []


def _reshape_for_vit(tensor, height, width):
    """ViT 계열은 토큰 시퀀스라 2D 로 되돌려야 합니다."""
    if tensor.dim() == 3:
        n_tokens = tensor.size(1)
        side = int(round(n_tokens ** 0.5))
        if side * side != n_tokens:      # CLS 토큰이 있는 경우
            tensor = tensor[:, 1:, :]
            side = int(round(tensor.size(1) ** 0.5))
        return tensor.reshape(tensor.size(0), side, side, tensor.size(2)).permute(0, 3, 1, 2)
    return tensor


def _is_transformer(model) -> bool:
    name = type(model).__name__.lower()
    return any(k in name for k in ("vit", "eva", "swin", "beit", "deit"))


# ──────────────────────────────────────────────────────────────
# 실행
# ──────────────────────────────────────────────────────────────
def cam_for(model, img_tensor: torch.Tensor, target_class: int | None = None,
            device: str = "cuda") -> np.ndarray:
    """단일 이미지의 CAM 히트맵 (H, W) 0~1."""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    layers = find_target_layer(model)
    if not layers:
        raise RuntimeError("CAM 을 걸 레이어를 찾지 못했습니다.")

    kwargs = {}
    if _is_transformer(model):
        kwargs["reshape_transform"] = _reshape_for_vit

    cam = GradCAM(model=model, target_layers=layers, **kwargs)
    targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
    out = cam(input_tensor=img_tensor.unsqueeze(0).to(device), targets=targets)
    return out[0]


def overlay(img_np: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    import cv2

    h = cv2.resize(heat, (img_np.shape[1], img_np.shape[0]))
    h = (h - h.min()) / (h.ptp() + 1e-8)
    color = cv2.applyColorMap(np.uint8(255 * h), cv2.COLORMAP_JET)[:, :, ::-1] / 255.0
    return np.clip((1 - alpha) * img_np + alpha * color, 0, 1)


def grid(model, df, cfg: CFG | None = None, n: int = 8, classes: list[str] | None = None,
         path_col: str = "crop_path", device: str = "cuda", seed: int = 0,
         only_correct: bool | None = None) -> None:
    """여러 장을 한 화면에 놓고 CAM 을 확인합니다.

    only_correct=False 로 두고 **틀린 예측**을 보는 게 특히 유용합니다.
    어디를 보고 틀렸는지가 다음 개선의 힌트입니다.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    from src.config import CLASSES
    from src.data import transforms_for_model

    cfg = cfg or CFG()
    classes = classes or CLASSES
    tf = transforms_for_model(cfg, model, train=False)
    model = model.to(device).eval()

    sub = df[df[path_col].notna()]
    if only_correct is not None and "pred" in sub.columns:
        sub = sub[(sub["pred"] == sub["label"]) == only_correct]
    picks = sub.sample(min(n, len(sub)), random_state=seed)
    if len(picks) == 0:
        print("보여줄 샘플이 없습니다.")
        return

    cols = min(4, len(picks))
    rows = (len(picks) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.7 * rows))
    axes = [axes] if len(picks) == 1 else list(np.array(axes).flat)

    for ax, (_, r) in zip(axes, picks.iterrows()):
        try:
            with Image.open(r[path_col]) as im:
                pil = im.convert("RGB").resize((cfg.img_size, cfg.img_size))
            x = tf(pil)
            with torch.no_grad():
                logit = model(x.unsqueeze(0).to(device))
                p = torch.softmax(logit.float(), 1)[0]
            pi = int(p.argmax())
            heat = cam_for(model, x, pi, device)
            ax.imshow(overlay(np.array(pil) / 255.0, heat))
            true = str(r.get("label", "?"))
            mark = "✓" if true == classes[pi] else "✗"
            ax.set_title(f"{mark} 정답 {true} / 예측 {classes[pi]} ({p[pi]:.2f})", fontsize=8.5)
        except Exception as exc:
            ax.text(0.5, 0.5, str(exc)[:60], ha="center", wrap=True, fontsize=7)
        ax.axis("off")
    for ax in axes[len(picks):]:
        ax.axis("off")
    plt.tight_layout(); plt.show()

    print("\n" + "─" * 60)
    print(" 이 그림에서 확인할 것:")
    print("  ✅ 빨간 영역이 병변 위에 있는가?")
    print("  ❌ 배경(바닥, 손, 목줄, 털만 있는 곳)에 몰려 있지 않은가?")
    print("  ❌ 이미지 가장자리나 모서리에 붙어 있지 않은가? (전형적인 shortcut 학습)")
    print("\n 배경을 보고 있다면 macro-F1 이 높아도 그 모델은 실패입니다.")
    print(" → 크롭 margin 을 줄이거나, 배경 증강을 강화하거나, 세그멘테이션 마스킹을 검토하세요.")
    print("─" * 60)


def lesion_overlap_score(model, df, cfg: CFG | None = None, n: int = 200,
                         device: str = "cuda", seed: int = 0) -> dict:
    """CAM 이 실제 병변 박스와 얼마나 겹치는지 **수치로** 잽니다.

    눈으로 보는 것보다 객관적이라 모델 비교에 쓸 수 있습니다.
    원본 이미지 좌표계의 bbox 가 있어야 하므로 path_col='image_path' 기준입니다.
    """
    from PIL import Image

    from src.data import transforms_for_model

    cfg = cfg or CFG()
    tf = transforms_for_model(cfg, model, train=False)
    model = model.to(device).eval()

    sub = df[df["bbox"].notna() & df["image_path"].notna()]
    picks = sub.sample(min(n, len(sub)), random_state=seed)
    scores: list[float] = []

    for _, r in picks.iterrows():
        try:
            with Image.open(r["image_path"]) as im:
                W, H = im.size
                pil = im.convert("RGB").resize((cfg.img_size, cfg.img_size))
            x = tf(pil)
            with torch.no_grad():
                pi = int(model(x.unsqueeze(0).to(device)).argmax())
            heat = cam_for(model, x, pi, device)

            b = r["bbox"]
            if isinstance(b, str):
                b = json.loads(b)
            hh, hw = heat.shape
            x1 = int(np.clip(b[0] / W * hw, 0, hw - 1)); x2 = int(np.clip(b[2] / W * hw, 1, hw))
            y1 = int(np.clip(b[1] / H * hh, 0, hh - 1)); y2 = int(np.clip(b[3] / H * hh, 1, hh))
            if x2 <= x1 or y2 <= y1:
                continue
            total = heat.sum()
            if total <= 0:
                continue
            inside = heat[y1:y2, x1:x2].sum()
            box_frac = ((x2 - x1) * (y2 - y1)) / (hh * hw)
            # 박스가 큰 이미지는 우연히 겹칠 확률도 높으므로 면적으로 정규화
            scores.append(float((inside / total) / max(box_frac, 1e-6)))
        except Exception:
            continue

    if not scores:
        print("계산할 샘플이 없습니다 (bbox 가 필요합니다).")
        return {}

    arr = np.array(scores)
    out = {"n": len(arr), "mean_lift": float(arr.mean()), "median_lift": float(np.median(arr)),
           "frac_above_1": float((arr > 1).mean())}
    print(f"\n[CAM–병변 정렬도] n={out['n']}")
    print(f"  평균 lift {out['mean_lift']:.2f} (1.0 = 우연 수준, 클수록 병변을 잘 봄)")
    print(f"  중앙값 {out['median_lift']:.2f}, 우연보다 나은 비율 {out['frac_above_1']:.1%}")
    if out["median_lift"] < 1.3:
        print("  ⚠️ 모델이 병변을 특별히 보고 있지 않습니다. 배경 학습을 의심하세요.")
    return out
