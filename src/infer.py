"""추론 + 사용자에게 보여줄 문구 생성.

이 파일의 절반은 모델이 아니라 **말하는 방식**에 관한 코드입니다.
그게 이 프로젝트의 요구사항이기 때문입니다 —
"수의사를 대체하는 게 아니라, 의심된다까지만 알려주기".

지켜야 할 원칙 (docs/cautions/03_의료AI_안전설계_원칙.md):
  1. 진단명을 단정하지 않는다. 병변 "형태"의 소견까지만 말한다.
  2. 확신이 낮으면 답을 만들어내지 말고 "판단 어려움"으로 돌린다.
  3. 어떤 결과가 나오든 수의사 진료 안내를 함께 준다.
  4. "정상"이라는 판정도 단정하지 않는다 — 놓쳤을 수 있다.
  5. ★ 2단계 파이프라인은 **병변 형태 이름도 단정하지 않는다.**
     여섯 개 확률을 전부 보여주고 "판단 불가 → 진료 권함" 으로 끝낸다.
     (2026-08-26 멘토 피드백. 근거는 `compose_screening_message` 의 docstring)

문구를 만드는 함수가 둘입니다 — 섞어 쓰지 마세요:
  · `compose_message`            — 단일 모델(Engine) 용. 1등 이름을 말합니다
  · `compose_screening_message`  — ★ 2단계 파이프라인 용. 이름을 말하지 않습니다

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

from src.config import CFG, CLASS_EN, CLASS_KO, CLASSES, NORMAL_LABEL, URGENCY_HINT

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

    # ── 2단계 파이프라인에서만 채워집니다 ──
    # topk 는 1단계 이상확률을 곱해 **보수적으로 깎은** 값입니다 (거절 판정용).
    # 화면에 띄우는 분포는 깎기 전 원본이어야 합니다 — 합이 1 이 아니면
    # 사람은 "다 낮네, 별 거 아닌가" 로 읽습니다. 그래서 둘을 따로 들고 다닙니다.
    stage1_abnormal: float | None = None                          # 1단계 '이상' 확률
    stage2_probs: list[tuple[str, float]] | None = None           # 2단계 원본 (합 = 1)

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
            "stage1_abnormal": (None if self.stage1_abnormal is None
                                else round(self.stage1_abnormal, 4)),
            "stage2_probs": (None if self.stage2_probs is None else
                             [{"class": c, "name_ko": CLASS_KO.get(c, c),
                               "name_en": CLASS_EN.get(c, c), "prob": round(p, 4)}
                              for c, p in self.stage2_probs]),
            "disclaimer": DISCLAIMER,
        }


def _cells(text: str) -> int:
    """터미널에서 차지하는 칸 수. 한글은 두 칸입니다.

    `len("비듬")` 은 2 지만 화면에서는 4 칸을 먹습니다. 이걸 안 맞추면
    확률 막대가 클래스마다 다른 위치에서 시작해 표로 안 읽힙니다.
    """
    import unicodedata

    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _cells(text))


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

    if c == NORMAL_LABEL:
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


def compose_screening_message(pred: Prediction, abnormal_p: float | None = None) -> str:
    """★ 확정된 출력 형식 (2026-08-26, 멘토 피드백).

    **병변 이름을 단정하지 않습니다.** 1단계가 '이상' 이라고 하면 2단계 확률을
    **여섯 개 전부** 보여주고, 어느 쪽인지는 판단할 수 없다고 말한 뒤 진료를 권합니다.

    왜 단정하지 않나 — holdout 실측:
      · 화면에 뜬 병변 이름이 틀린 비율 **56.6%** (병원 보낸 4,478건 중 2,534건)
      · A6(결절·종괴)를 A2(비듬)로 부른 것 56장. 종괴를 비듬이라 하면 병원을 미룹니다
      · 서로 다른 아키텍처(convnextv2 0.5744 / swinv2 0.5718)가 0.0026 차이로 겹침
        → 모델이 아니라 데이터가 한계. 백본을 더 바꿔도 안 넘습니다

    왜 그래도 숫자를 보여주나 — 참고 정보로서의 값은 있고, **여섯 개를 다 보여주면**
    "확실하지 않다" 가 눈에 보입니다.

    ⚠️ 그래서 표시 규칙이 중요합니다:
      · 1등만 뽑아 굵게 쓰지 않습니다. 그러면 사람은 그걸 답으로 읽습니다
      · 클래스별 긴급도 문구(URGENCY_HINT)를 붙이지 않습니다 — 이름을 단정하는 셈입니다
      · "판단할 수 없습니다" 를 숫자보다 **위에** 씁니다
    """
    L: list[str] = []

    if pred.abstain or not pred.topk:
        return compose_message(pred)          # "다시 찍어주세요" 는 그대로

    c0 = pred.topk[0][0]
    if c0 == NORMAL_LABEL:
        p_norm = pred.topk[0][1]
        L.append(f"🟢 **뚜렷한 피부 병변 소견은 보이지 않습니다.** (정상 가능성 {p_norm:.0%})")
        L.append("")
        L.append("다만 사진 한 장으로 확인할 수 있는 범위에는 한계가 있습니다.")
        L.append("가려워하거나, 냄새가 나거나, 계속 핥는 등 평소와 다른 행동이 있다면 "
                 "결과와 무관하게 병원에 가보시는 것을 권합니다.")
        L.append("")
        L.append(f"_{DISCLAIMER}_")
        return "\n".join(L)

    if abnormal_p is None:
        abnormal_p = pred.stage1_abnormal

    head = "🔎 **피부에 이상 소견이 보입니다.**"
    if abnormal_p is not None:
        head += f" (이상 가능성 {abnormal_p:.0%})"
    L.append(head)
    L.append("")
    L.append("**어떤 병변인지는 이 사진만으로 판단할 수 없습니다.** "
             "아래는 모델이 비슷하다고 본 정도이며, 진단이 아닙니다.")
    L.append("")

    # 깎기 전 원본 분포를 씁니다 (합 = 1). pred.topk 는 1단계 확률을 곱해
    # 낮춰둔 값이라, 그걸 그대로 띄우면 여섯 개가 다 작아져 분포로 안 읽힙니다.
    dist = pred.stage2_probs if pred.stage2_probs else pred.topk

    # ★ 여섯 개를 **전부** 보여줍니다. 상위 몇 개만 자르면 그게 답처럼 읽힙니다.
    width = max((_cells(CLASS_KO.get(cc, cc)) for cc, _ in dist), default=10)
    for cc, pp in dist:
        name = _pad(CLASS_KO.get(cc, cc), width)
        bar = "█" * max(0, round(pp * 20))
        L.append(f"    {name}  {pp:>4.0%}  {bar}")

    L.append("")
    L.append("→ **수의사 진료를 받아보시기를 권합니다.**")
    if pred.confidence_band == "낮음":
        L.append("")
        L.append("_사진을 더 선명하게 다시 찍으면 결과가 달라질 수 있습니다._")
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
    def show(self, path: str, pred: Prediction | None = None) -> None:
        """이미지 + 판정 + CAM 을 한 번에 보여줍니다 (노트북용).

        pred 를 주면 그 판정을 그대로 씁니다 — 2단계 파이프라인처럼
        확률을 이미 조정해둔 경우에 필요합니다.
        """
        import matplotlib.pyplot as plt
        from PIL import Image

        pred = pred if pred is not None else self.predict(path)
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
        # 2단계 파이프라인이 넘겨준 판정이면(= 1단계 확률이 붙어 있으면)
        # 이름을 단정하지 않는 스크리닝 문구를 씁니다.
        print(compose_screening_message(pred) if pred.stage1_abnormal is not None
              else compose_message(pred, self.cfg.topk_report))


# ──────────────────────────────────────────────────────────────
# 2단계 파이프라인 (정상/이상 → 병변 6종)
# ──────────────────────────────────────────────────────────────
class TwoStageEngine:
    """1단계에서 '이상'으로 걸러진 것만 2단계 병변 분류로 넘깁니다.

        사진 → 1단계 ─ 이상확률 < threshold → "정상으로 보입니다" (2단계 안 봄)
                      └ threshold 이상 ────→ 2단계 → 병변 6종 **확률 분포를 통째로**

    ⚠️ 2단계의 1등을 병변 이름으로 **단정하지 않습니다.** holdout 에서 그 이름이
    틀린 비율이 56.6% 였습니다. 여섯 개를 다 보여주고 "판단 불가 → 진료 권함" 으로
    끝냅니다 (`compose_screening_message` 에 근거를 적어뒀습니다).

    threshold 는 재현율 우선으로 잡습니다 — 놓치는 것이 오탐보다 나쁘므로.
    노트북 03 이 `stage1_threshold.json` 에 저장한 값을 쓰세요.

        s1 = infer.Engine.load(".../stage1_.../best.pt")
        s2 = infer.Engine.load(".../stage2_.../best.pt")
        eng = infer.TwoStageEngine(s1, s2, threshold=0.31)
        print(eng.explain("my_dog.jpg"))
    """

    def __init__(self, stage1: Engine, stage2: Engine, threshold: float = 0.5):
        from src.stages import ABNORMAL_LABEL

        self.s1, self.s2, self.thr = stage1, stage2, threshold
        self._ab = ABNORMAL_LABEL
        if self._ab not in stage1.classes:
            raise ValueError(
                f"1단계 엔진의 클래스가 {stage1.classes} 입니다 — "
                f"'{self._ab}' 가 없습니다. 2단계용으로 학습한 체크포인트인지 확인하세요."
            )

    def predict(self, path: str) -> Prediction:
        p1 = self.s1.predict(path)
        if not p1.topk:                       # 이미지를 못 열었음
            return p1
        abnormal = dict(p1.topk).get(self._ab, 0.0)

        if abnormal < self.thr:
            # ⚠️ "정상" 도 단정하지 않습니다 — 놓쳤을 수 있으므로 문구가 그 한계를 말합니다
            return Prediction(topk=[(NORMAL_LABEL, 1 - abnormal)], abstain=False,
                              confidence_band=band(1 - abnormal), image=path,
                              stage1_abnormal=abnormal)

        p2 = self.s2.predict(path)
        # 화면에 띄울 분포는 **깎기 전** 원본입니다 (합 = 1).
        p2.stage2_probs = list(p2.topk)
        p2.stage1_abnormal = abnormal
        # 2단계 확률에 1단계의 '이상' 확률을 곱해 전체 신뢰도를 보수적으로 유지합니다.
        # (1단계가 애매하게 통과시킨 사진에 2단계가 90% 라고 말하면 과신입니다)
        # 이 값은 **거절 판정용**이고, 사람에게 보여주는 숫자가 아닙니다.
        p2.topk = [(c, p * abnormal) for c, p in p2.topk]
        p2.confidence_band = band(p2.topk[0][1]) if p2.topk else "낮음"
        p2.abstain = bool(p2.topk) and p2.topk[0][1] < self.s2.cfg.abstain_threshold
        return p2

    def explain(self, path: str) -> str:
        """★ 최종 출력. 병변 **이름을 단정하지 않고** 2단계 분포를 통째로 보여줍니다.

        (2026-08-26 멘토 피드백. 근거는 `compose_screening_message` 참고)
        """
        return compose_screening_message(self.predict(path))

    def show(self, path: str) -> None:
        """이미지 + CAM + 최종 문구. 1단계에서 걸러지면 CAM 은 생략합니다."""
        pred = self.predict(path)
        if pred.topk and pred.topk[0][0] == NORMAL_LABEL:
            import matplotlib.pyplot as plt
            from PIL import Image

            with Image.open(path) as im:
                plt.figure(figsize=(4.4, 4.4))
                plt.imshow(im.convert("RGB")); plt.axis("off")
                plt.title("1단계: 정상으로 판단 (2단계 미실행)")
                plt.tight_layout(); plt.show()
            print(compose_screening_message(pred))
            return
        self.s2.show(path, pred)          # CAM 은 2단계 모델, 문구는 파이프라인 확률로
