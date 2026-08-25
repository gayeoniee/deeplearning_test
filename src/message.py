"""보호자에게 보여줄 **문구**만 담은 모듈 — 모델도 torch 도 안 씁니다.

`infer.py` 에서 떼어냈습니다. 이유는 두 가지입니다:
  · 원래 그 파일 설명이 "절반은 모델이 아니라 말하는 방식에 관한 코드" 였습니다
  · 데모 서버를 **torch 없이** 띄우려면 (`serve.py --mock`) 문구가 torch 와
    같은 파일에 있으면 안 됩니다

`from src.infer import compose_message` 는 그대로 됩니다 — infer 가 재수출합니다.

지켜야 할 원칙은 `docs/cautions/03_의료AI_안전설계_원칙.md` 에 있고,
2단계 파이프라인 규격은 그 문서 §7-B 입니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.config import CLASS_EN, CLASS_KO, NORMAL_LABEL, URGENCY_HINT


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

    if abnormal_p is None:
        abnormal_p = pred.stage1_abnormal

    # ★ 1단계만 돌리는 구성 — 2단계 없이 "이상" 까지만 말합니다.
    #   분포가 없다고 재촬영으로 보내면 안 됩니다. 1단계는 판정을 냈으니까요.
    #   (멘토 피드백대로 어차피 이름은 안 말하므로 이것만으로도 제품이 됩니다)
    if not pred.abstain and not pred.topk and abnormal_p is not None:
        L.append("🔎 **피부에 이상 소견이 보입니다.**" + f" (이상 가능성 {abnormal_p:.0%})")
        L.append("")
        L.append("**어떤 병변인지는 판단하지 않습니다.** 이 사진만으로는 알 수 없습니다.")
        L.append("")
        L.append("→ **수의사 진료를 받아보시기를 권합니다.**")
        L.append("")
        L.append(f"_{DISCLAIMER}_")
        return "\n".join(L)

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
