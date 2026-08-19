"""추론 + 사용자에게 보여줄 문구 생성.

이 파일의 절반은 모델이 아니라 **말하는 방식**에 관한 코드입니다.
그게 이 프로젝트의 요구사항이기 때문입니다 —
"수의사를 대체하는 게 아니라, 의심된다까지만 알려주기".

지켜야 할 원칙 (docs/cautions/03_의료AI_안전설계_원칙.md):
  1. 진단명을 단정하지 않는다. 병변 "형태"의 소견까지만 말한다.
  2. 확신이 낮으면 답을 만들어내지 말고 "판단 어려움"으로 돌린다.
  3. 어떤 결과가 나오든 수의사 진료 안내를 함께 준다.
  4. "정상"이라는 판정도 단정하지 않는다 — 놓쳤을 수 있다.

    from src import infer
    engine = infer.Engine.load("checkpoints/convnextv2_base/best.pt")
    print(engine.explain("my_dog.jpg"))
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from src.config import CFG, CLASS_EN, CLASS_KO, CLASSES, URGENCY_HINT

DISCLAIMER = (
    "이 결과는 수의학적 진단이 아니며, 수의사의 진료를 대체하지 않습니다. "
    "참고용 스크리닝 정보로만 사용해 주세요."
)


# ──────────────────────────────────────────────────────────────
# 결과
# ──────────────────────────────────────────────────────────────
@dataclass
class Prediction:
    topk: list[tuple[str, float]] = field(default_factory=list)   # [(class, prob), ...]
    abstain: bool = False
    confidence_band: str = ""      # "높음" | "보통" | "낮음"
    image: str = ""

    @property
    def top1(self) -> tuple[str, float] | None:
        return self.topk[0] if self.topk else None

    def to_dict(self) -> dict:
        return {
            "image": self.image,
            "abstain": self.abstain,
            "confidence_band": self.confidence_band,
            "topk": [{"class": c, "name_ko": CLASS_KO.get(c, c),
                      "name_en": CLASS_EN.get(c, c), "prob": round(p, 4)}
                     for c, p in self.topk],
            "disclaimer": DISCLAIMER,
        }


def band(p: float) -> str:
    return "높음" if p >= 0.75 else ("보통" if p >= 0.5 else "낮음")


def compose_message(pred: Prediction, topk_show: int = 3) -> str:
    """보호자에게 보여줄 한국어 문구. 단정하지 않는 표현만 씁니다."""
    L: list[str] = []

    if pred.abstain or not pred.topk:
        L.append("📷 **판단이 어려운 사진입니다.**")
        L.append("")
        L.append("병변 부위가 화면 가운데에 오도록, 밝은 곳에서 초점을 맞춰 다시 찍어주세요.")
        L.append("털에 가려져 있다면 손으로 살짝 헤쳐 피부가 보이게 해주시면 좋습니다.")
        L.append("")
        L.append(f"_{DISCLAIMER}_")
        return "\n".join(L)

    c, p = pred.topk[0]
    ko = CLASS_KO.get(c, c)

    if c == "A0":
        L.append("🟢 **뚜렷한 피부 병변 소견은 보이지 않습니다.**")
        L.append("")
        L.append("다만 사진 한 장으로 확인할 수 있는 범위에는 한계가 있습니다.")
        L.append("가려워하거나, 냄새가 나거나, 계속 핥는 등 평소와 다른 행동이 있다면 "
                 "결과와 무관하게 병원에 가보시는 것을 권합니다.")
    else:
        L.append(f"🔎 **{ko}** 형태의 병변이 의심됩니다. (신뢰도 {p:.0%}, {pred.confidence_band})")
        L.append("")
        hint = URGENCY_HINT.get(c)
        if hint and hint != "관찰":
            L.append(f"⚠️ {hint}")
            L.append("")
        L.append("이건 **병변의 겉모습**에 대한 소견이지 병명이 아닙니다. "
                 "같은 모양이라도 원인 질환은 알레르기, 감염, 내분비 질환 등 여러 가지일 수 있고, "
                 "원인에 따라 치료가 완전히 달라집니다.")

    if len(pred.topk) > 1:
        others = ", ".join(f"{CLASS_KO.get(cc, cc)} {pp:.0%}" for cc, pp in pred.topk[1:topk_show])
        if others:
            L.append("")
            L.append(f"다른 가능성: {others}")

    if pred.confidence_band == "낮음":
        L.append("")
        L.append("_신뢰도가 낮습니다. 사진을 더 선명하게 다시 찍으면 결과가 달라질 수 있습니다._")

    L.append("")
    L.append(f"_{DISCLAIMER}_")
    return "\n".join(L)


# ──────────────────────────────────────────────────────────────
# 엔진
# ──────────────────────────────────────────────────────────────
class Engine:
    """체크포인트 하나(또는 앙상블)로 추론합니다."""

    def __init__(self, model, cfg: CFG, classes: list[str],
                 temperature: float = 1.0, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.cfg = cfg
        self.classes = classes
        self.T = temperature
        from src.data import transforms_for_model

        base = model.models[0] if hasattr(model, "models") else model
        self.tf = transforms_for_model(cfg, base, train=False)

    @classmethod
    def load(cls, ckpt_path: str | Path, spec=None, temperature: float | None = None,
             device: str | None = None) -> "Engine":
        from src.models import build

        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = CFG.from_dict(ck.get("cfg", {}))
        classes = ck.get("classes") or CLASSES
        model = build(spec or cfg.model_name, len(classes), pretrained=False, verbose=False)
        model.load_state_dict(ck.get("ema") or ck["model"], strict=False)

        T = temperature
        if T is None:
            tp = Path(ckpt_path).parent / "temperature.json"
            T = json.loads(tp.read_text())["temperature"] if tp.exists() else 1.0
        return cls(model, cfg, classes, T, device)

    # -------------------------------------------------------
    @torch.no_grad()
    def predict_batch(self, paths: list[str], tta: bool | None = None) -> list[Prediction]:
        from PIL import Image

        tta = self.cfg.tta_hflip if tta is None else tta
        tensors, valid = [], []
        for p in paths:
            try:
                with Image.open(p) as im:
                    tensors.append(self.tf(im.convert("RGB")))
                valid.append(p)
            except Exception:
                continue
        if not tensors:
            return [Prediction(image=p, abstain=True) for p in paths]

        x = torch.stack(tensors).to(self.device)
        logit = self.model(x).float()
        if tta:
            logit = (logit + self.model(torch.flip(x, dims=[3])).float()) / 2
        probs = torch.softmax(logit / self.T, dim=1).cpu().numpy()

        out = []
        for p, pr in zip(valid, probs):
            order = np.argsort(-pr)[: self.cfg.topk_report]
            topk = [(self.classes[i], float(pr[i])) for i in order]
            conf = topk[0][1]
            out.append(Prediction(
                topk=topk,
                abstain=conf < self.cfg.abstain_threshold,
                confidence_band=band(conf),
                image=p,
            ))
        # 열지 못한 이미지도 자리를 채워 돌려줍니다
        missing = [Prediction(image=p, abstain=True) for p in paths if p not in set(valid)]
        return out + missing

    def predict(self, path: str, tta: bool | None = None) -> Prediction:
        return self.predict_batch([path], tta)[0]

    def explain(self, path: str, tta: bool | None = None) -> str:
        return compose_message(self.predict(path, tta), self.cfg.topk_report)

    def predict_json(self, path: str) -> str:
        return json.dumps(self.predict(path).to_dict(), ensure_ascii=False, indent=2)

    # -------------------------------------------------------
    def show(self, path: str) -> None:
        """이미지 + 판정 + CAM 을 한 번에 보여줍니다 (노트북용)."""
        import matplotlib.pyplot as plt
        from PIL import Image

        pred = self.predict(path)
        fig, ax = plt.subplots(1, 2, figsize=(9, 4.4))
        with Image.open(path) as im:
            pil = im.convert("RGB").resize((self.cfg.img_size, self.cfg.img_size))
        ax[0].imshow(pil); ax[0].axis("off"); ax[0].set_title("입력")

        try:
            from src.explain import cam_for, overlay

            base = self.model.models[0] if hasattr(self.model, "models") else self.model
            ci = self.classes.index(pred.topk[0][0]) if pred.topk else 0
            heat = cam_for(base, self.tf(pil), ci, self.device)
            ax[1].imshow(overlay(np.array(pil) / 255.0, heat))
            ax[1].set_title("모델이 주목한 곳")
        except Exception as exc:
            ax[1].text(0.5, 0.5, f"CAM 실패\n{str(exc)[:50]}", ha="center", fontsize=8)
        ax[1].axis("off")
        plt.tight_layout(); plt.show()
        print(compose_message(pred, self.cfg.topk_report))


# ──────────────────────────────────────────────────────────────
# 2단계 파이프라인 (정상/이상 → 병변 6종)
# ──────────────────────────────────────────────────────────────
class TwoStageEngine:
    """1단계에서 '이상'으로 걸러진 것만 2단계 병변 분류로 넘깁니다.

    1단계 임계값은 재현율 우선으로 잡습니다 — 놓치는 것이 오탐보다 나쁘므로.
    """

    def __init__(self, stage1: Engine, stage2: Engine, threshold: float = 0.5):
        self.s1, self.s2, self.thr = stage1, stage2, threshold

    def predict(self, path: str) -> Prediction:
        p1 = self.s1.predict(path)
        if p1.abstain:
            return p1
        abnormal = next((p for c, p in p1.topk if c != "A0"), 0.0)
        if abnormal < self.thr:
            return Prediction(topk=[("A0", 1 - abnormal)], abstain=False,
                              confidence_band=band(1 - abnormal), image=path)
        p2 = self.s2.predict(path)
        # 2단계 확률에 1단계의 '이상' 확률을 곱해 전체 신뢰도를 보수적으로 유지
        p2.topk = [(c, p * abnormal) for c, p in p2.topk]
        p2.confidence_band = band(p2.topk[0][1]) if p2.topk else "낮음"
        p2.abstain = bool(p2.topk) and p2.topk[0][1] < self.s2.cfg.abstain_threshold
        return p2

    def explain(self, path: str) -> str:
        return compose_message(self.predict(path))
