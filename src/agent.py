"""★ 스크리닝 에이전트 — 사진 한 장을 받아 판정 JSON 을 내놓습니다.

안드로이드 앱(DAENGS_APP)이 붙을 자리입니다. HTTP 는 `serve.py` 가 담당하고
여기는 **파이프라인과 응답 계약**만 봅니다 (규칙 3: 바뀔 수 있는 로직은 src/ 에).

    사진 → [서빙 크롭] → 1단계(정상/이상) ─ 낮으면 → "정상으로 보입니다"
                                          └ 높으면 → 2단계(6종 분포) → 진료 권함

⚠️ **응답에 "1등 병변" 필드가 없습니다. 실수가 아닙니다.**
holdout 에서 그 이름이 56.6% 틀렸습니다. 앱이 고를 수 없게 계약에서 아예 뺐습니다.
`docs/cautions/03_의료AI_안전설계_원칙.md` §7-B 를 먼저 읽어주세요.

    from src.agent import ScreeningAgent
    agent = ScreeningAgent.load("ckpt/stage1_.../best.pt", "ckpt/stage2_.../best.pt",
                                threshold=0.1823)
    agent.screen("my_dog.jpg")            # → dict (JSON 직렬화 가능)

가중치 없이 화면만 보려면:

    from src.agent import MockAgent
    MockAgent().screen("any.jpg")
"""

from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path
from typing import Any

from src.config import CLASS_EN, CLASS_KO, CLASSES, NORMAL_LABEL

CONTRACT_VERSION = "1.0"

# ──────────────────────────────────────────────────────────────
# 서빙 크롭 — 학습이 쓰는 **그 함수**를 그대로 부릅니다
# ──────────────────────────────────────────────────────────────
#
# 예전에는 여기서 "중앙 몇 %" 를 직접 계산했습니다. 학습은 `crop.crop_window()`
# 를 쓰는데 서빙은 딴 식으로 자르면, 둘이 갈라지는 순간 아무도 모릅니다.
# 지금은 **같은 함수**를 부릅니다. 갈라질 수가 없습니다.
#
# 남는 문제는 하나였습니다 — 학습은 라벨 bbox 를 알고 있고 서빙은 모릅니다.
# 그래서 **촬영 가이드 프레임을 bbox 로 받습니다.** 앱이 카메라에 띄운 네모를
# 사용자가 병변에 맞추면, 그 네모가 곧 bbox 입니다.
#
# 두 단계가 그 네모를 **다르게** 씁니다 (이게 핵심입니다):
#
#   1단계 f320 : 네모의 **중심만** 씁니다. 창은 320px 고정.
#                → 네모 크기가 틀려도 결과가 같습니다. 중심만 맞으면 됩니다.
#   2단계 m2.5 : 네모의 **크기**를 씁니다 (긴 변 × 2.5).
#                → 네모를 병변에 맞게 **조절할 수 있어야** 학습과 같아집니다.
#                  고정 크기 프레임이면 2단계는 여전히 어긋납니다.
#
# 그래서 앱의 가이드 프레임은 **끌고 늘릴 수 있어야** 합니다.

TRAIN_SHORT_SIDE = 1080     # AI Hub 원본이 1920×1080 (docs/data/DATASET_CARD.md §1)
STAGE1_TAG = "f320"         # 1단계 학습 크롭 (STEP 9-A 에서 확정)
STAGE2_TAG = "m2.5"         # 2단계 학습 크롭 (STEP 4C 에서 확정)

# 촬영 가이드 밴드 — `robust.usable_range()` 가 STEP 16 에서 실측한 값입니다
# (STATUS.md "촬영 가이드", n=2,000). 화면 **가로** 대비 병변의 비율입니다.
#
# 어디서 나온 숫자인가 — 매번 다시 유도하지 않도록 적어 둡니다.
#   `usable_range()` 가 배율을 화면 점유율로 바꾸는 식이 occ(z) = z / crop_margin
#   (src/robust.py) 이고 2단계 크롭이 m2.5 이므로, **배율 밴드 ÷ 2.5** 입니다.
#     권장 배율 0.7 ~ 1.2  ÷ 2.5 → (0.28, 0.48)
#     허용 배율 0.6 ~ 1.4  ÷ 2.5 → (0.24, 0.56)
#
# ⚠️ STEP 10 은 (0.34, 0.56) / (0.28, 0.68) 이었습니다. **크게 잡는 쪽이
#    빡빡해졌습니다** — 2.0x 에서 macro-F1 이 0.449 까지 떨어집니다. 작게 잡는
#    쪽은 반대로 너그러워졌습니다.
# ⚠️ 여기를 고치면 `demo/index.html` 의 `BAND` 와 `tests/test_agent.py` 를
#    **같이** 고치세요. 하나만 고치면 화면과 서버가 갈라지고, 갈라져도
#    아무도 모릅니다.
GUIDE_RECOMMEND = (0.28, 0.48)     # 하락 5% 이내
GUIDE_ALLOW = (0.24, 0.56)         # 하락 10% 이내
GUIDE_CENTER_MAX = 0.10            # 화면 중앙에서 이만큼 이내


def to_train_space(im):
    """짧은 변을 1080 으로 맞춥니다. 그 뒤로는 학습과 **같은 픽셀 공간**입니다.

    이걸 안 하면 f320 의 320 이 뜻을 잃습니다 — 휴대폰 사진은 4032×3024 라
    320px 이 학습 때보다 훨씬 좁은 피부 조각이 됩니다. 원본이 전부 1920×1080
    이었으므로, 짧은 변을 1080 에 맞추면 320px 이 다시 같은 화각이 됩니다.
    """
    w, h = im.size
    short = min(w, h)
    if short == TRAIN_SHORT_SIDE:
        return im
    k = TRAIN_SHORT_SIDE / short
    return im.resize((max(1, round(w * k)), max(1, round(h * k))))


def box_to_px(box, w: int, h: int) -> list[float] | None:
    """정규화 [x, y, bw, bh] (0~1) → 픽셀 [x1, y1, x2, y2]. 못 읽으면 None."""
    try:
        x, y, bw, bh = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    if not (bw > 0 and bh > 0):
        return None
    x1, y1 = max(0.0, x * w), max(0.0, y * h)
    return [x1, y1, min(float(w), x1 + bw * w), min(float(h), y1 + bh * h)]


def check_guide(box) -> dict:
    """가이드 프레임이 촬영 가이드 밴드 안에 있는가. 추론 **전에** 봅니다.

    밴드 밖 사진은 모델에 넣지 말고 다시 찍게 하는 게 맞습니다 — 그 구간에서
    성능이 떨어지는 걸 이미 재 뒀는데(STEP 10), 굳이 넣고 나서 틀리는 것보다
    안 넣는 편이 낫습니다.

    Returns:
        {"ok", "reason", "width_frac", "center_off"} — reason 은 보호자에게
        그대로 보여줄 한국어입니다.
    """
    try:
        x, y, bw, bh = (float(v) for v in box)
    except (TypeError, ValueError):
        return {"ok": True, "reason": "", "width_frac": None, "center_off": None}

    off = max(abs(x + bw / 2 - 0.5), abs(y + bh / 2 - 0.5))
    r = {"ok": True, "reason": "", "width_frac": round(bw, 4), "center_off": round(off, 4)}

    if bw < GUIDE_ALLOW[0]:
        r.update(ok=False, reason="병변이 너무 작게 잡혔습니다. 조금 더 가까이에서 찍어주세요.")
    elif bw > GUIDE_ALLOW[1]:
        r.update(ok=False, reason="너무 가까워서 주변 피부가 안 보입니다. 조금 더 멀리서 찍어주세요.")
    elif off > GUIDE_CENTER_MAX:
        r.update(ok=False, reason="병변이 화면 가운데에서 벗어났습니다. 가운데에 오도록 다시 맞춰주세요.")
    return r


def crop_for(im, bbox, tag: str):
    """학습이 쓰는 `crop.crop_window()` 로 잘라냅니다 — 재구현하지 않습니다.

    bbox 가 None 이면 그 함수가 알아서 물러섭니다:
      f320 → 이미지 중앙에서 320px,  m2.5 → 중앙 정사각.
    """
    from src import crop as _crop

    w, h = im.size
    win = _crop.crop_window({"bbox": bbox, "img_w": w, "img_h": h}, tag=tag)
    return im.crop(tuple(win)) if win else im


# ──────────────────────────────────────────────────────────────
# 응답 계약
# ──────────────────────────────────────────────────────────────
# 가이드 프레임을 받았을 때 / 못 받았을 때 각각 무엇이 남는지.
# 계약에 실어 보내서 앱이 이 한계를 모른 척할 수 없게 합니다.
CROP_NOTE = {
    "user_box": ("가이드 프레임을 bbox 로 써서 학습과 **같은 함수**로 잘랐습니다. "
                 "다만 사용자가 맞춘 네모는 라벨러가 그린 네모와 분포가 다릅니다 — "
                 "2단계는 네모 크기로 배율이 정해지므로 영향을 받습니다."),
    "center": ("가이드 프레임 없이 화면 중앙을 잘랐습니다. 1단계는 중심만 쓰므로 "
               "큰 차이가 없지만, 2단계는 학습 크롭과 어긋납니다."),
}

def base_meta(*, mock: bool, tag1: str, tag2: str, temperature: float, box) -> dict:
    """응답 `meta` 의 **공통 부분 — 여기 한 곳에서만 만듭니다.**

    ⚠️ 예전엔 `ScreeningAgent` 와 `MockAgent` 가 각자 dict 를 썼습니다. 그래서
       진짜에만 있는 키가 **넷** 생겼고(`stage1_temperature` ·
       `stage2_low_confidence` · `stage2_top_prob` · `stage2_abstain_threshold`),
       앱이 mock 으로 만들어졌다가 진짜에서 처음 보는 키를 만날 뻔했습니다.
       `--mock` 의 존재 이유가 *"가중치 없이 **같은 모양의** 응답"* 인데
       그 약속이 깨져 있었습니다.
       → 같은 계약을 **두 곳에 적지 않습니다.** 갈라지면 아무도 모릅니다.
    """
    return {"mock": bool(mock), "stage1_crop": tag1, "stage2_crop": tag2,
            "stage1_temperature": float(temperature),
            "box_source": "user" if box is not None else "center",
            "crop_note": CROP_NOTE["user_box" if box is not None else "center"]}


def stage2_meta(pred, raw, abstain_threshold: float) -> dict:
    """2단계까지 갔을 때 붙는 `meta` — 이것도 **한 곳에서만**."""
    return {"stage2_low_confidence": bool(getattr(pred, "abstain", False)),
            "stage2_top_prob": round(float(raw[0][1]), 4) if raw else None,
            "stage2_abstain_threshold": float(abstain_threshold)}


def _dist(probs: list[tuple[str, float]]) -> list[dict]:
    """분포를 앱이 그대로 그릴 수 있는 모양으로. **정렬은 하되 자르지 않습니다.**"""
    return [{"code": c,
             "name_ko": CLASS_KO.get(c, c),
             "name_en": CLASS_EN.get(c, c),
             "prob": round(float(p), 4),
             "percent": round(float(p) * 100, 1)}
            for c, p in sorted(probs, key=lambda kv: -kv[1])]


def lesion_group_dist(probs: list[tuple[str, float]] | None) -> list[dict]:
    """★ **계열 네 묶음의 확률 분포** — 화면 막대가 쓰는 값.

    6종을 **자른 게 아니라 더한 것**입니다. 여섯 개가 전부 어딘가에 들어가
    있어 **아무것도 안 숨깁니다** — `cautions/03 §7-B` 의 *"상위 몇 개로
    자르지 마라"* 취지가 그대로 지켜집니다.

    ⚠️ `lesion_group()`(주장) 과 다릅니다. 이건 **분포**라 확신과 무관하게
       항상 나옵니다. 확신이 낮으면 `group` 이 `null` 이 되고 막대만 남습니다.

    ⚠️ 묶음표는 `MORPH_GROUP_KEEP_A6` **한 곳**에서만 읽습니다.
    """
    from src.config import MORPH_GROUP_KEEP_A6

    if not probs:
        return []
    tot: dict[str, float] = {}
    for code, p in probs:
        g = MORPH_GROUP_KEEP_A6.get(code)
        if g is None:
            return []                       # 모르는 코드가 섞이면 안 그립니다
        tot[g] = tot.get(g, 0.0) + float(p)
    return [{"name": k, "prob": round(v, 4), "percent": round(v * 100, 1)}
            for k, v in sorted(tot.items(), key=lambda kv: -kv[1])]


def a6_alert(probs: list[tuple[str, float]] | None,
             abnormal_p: float | None = None) -> dict | None:
    """★ **"덩어리가 의심됩니다"** — 유일하게 병변 이름을 말하는 자리입니다.

    점수는 **`p(이상) × p(A6)`** 이고 문턱은 `config.A6_ALERT_MIN`(0.40).
    ⚠️ `p(A6)` 단독으로 재면 STEP 34 의 표를 못 읽습니다 — 거기 문턱은
       `tools/naming_granularity.py` 의 `score = p1 * ens[:, ia6]` 기준입니다.

    실측(STEP 34): 문턱 0.40 에서 재현율 **val 61.1% / holdout 59.4%**.
    정밀도(81.4% / 71.3%)는 **유병률에 좌우되므로 기준으로 쓰지 않습니다.**

    왜 여기만 이름을 말하나 — 임상 해설이 *"A6 으로 오탐하는 건 상대적으로
    안전"* 이라 적었고(병원에 가서 확인하면 되니까), A6 은 **종양 감별**이
    필요한 유일한 클래스라 4묶음에서도 혼자 뒀습니다. 놓치는 쪽이 훨씬 나쁩니다.
    """
    from src.config import A6_ALERT_MIN
    from src.message import SHOW_A6_ALERT

    if not SHOW_A6_ALERT or not probs:
        return None
    p6 = dict(probs).get("A6")
    if p6 is None:
        return None
    score = float(p6) * (abnormal_p if abnormal_p is not None else 1.0)
    if score < A6_ALERT_MIN:
        return None
    return {"code": "A6",
            "score": round(score, 4),
            "threshold": A6_ALERT_MIN,
            # ⚠️ 앱·콘솔이 이 문장을 **그대로** 띄우게 합니다.
            "text": "덩어리가 의심됩니다.",
            "action": "빠른 진료를 권합니다.",
            "caveat": "진단이 아닙니다. 덩어리처럼 보이는 다른 것일 수 있습니다."}


def lesion_group(probs: list[tuple[str, float]] | None,
                 abnormal_p: float | None = None) -> dict | None:
    """★ **계열** 한 덩어리 — 이름이 아니라 묶음입니다.

    `None` 이면 화면에 아무것도 띄우지 않습니다. 두 경우입니다:
      · `message.SHOW_GROUP` 이 꺼져 있음 (2026-09-08 부터 기본은 **켜짐**.
        끄려면 `DOG_SKIN_SHOW_GROUP=0`)
      · 확신이 문턱 아래 — **확신 없으면 말하지 않습니다**

    ⚠️ **이건 "1등 병변" 이 아닙니다.** 6종 중 하나를 고르는 게 아니라
       네 묶음 중 하나이고, 그 묶음의 확률은 **안에 든 것을 더한 값**입니다.
       6종 이름(`구진·플라크` 등)은 여기 절대 안 들어갑니다 —
       `tests/test_agent.py` 가 감시합니다.

    근거: STEP 28~33. holdout 에서 6종 이름은 커버리지 41.1% 인데
    계열 4군은 **67.9%** 이고, 긴급도 하향 3.7% · A6 오명명 12.5% 로
    두 안전 관문 안입니다.
    """
    from src.config import (DOWNGRADE_BLOCK_MIN, GROUP_DETAIL, GROUP_FEATURE,
                            MORPH_GROUP_KEEP_A6, URGENT_GROUPS)
    from src.message import GROUP_CONF_MIN, SHOW_GROUP

    if not SHOW_GROUP or not probs:
        return None
    tot: dict[str, float] = {}
    for code, p in probs:
        g = MORPH_GROUP_KEEP_A6.get(code)
        if g is None:
            return None                     # 모르는 코드가 섞이면 말하지 않습니다
        tot[g] = tot.get(g, 0.0) + float(p)

    # ★ 하향 방지 (STEP 35) — 급한 쪽 합이 문턱을 넘으면 **덜 급한 묶음을
    #   후보에서 뺍니다.** 그러면 급한 쪽을 말하거나, 확신이 모자라 아무 말도
    #   안 합니다. 전체 하향(3.7%)이 관문을 통과하는 동안 말한 A5 의 43.8% 가
    #   하향이던 구멍을 막습니다 → 36.8%, 커버리지는 1.3%p 만 내줍니다.
    #   ⚠️ `distribution`(6종 분포)은 **안 건드립니다** — 여기서 고르는 것은
    #      "네 묶음 중 무엇을 말할까" 뿐입니다.
    urgent = sum(v for k, v in tot.items() if k in URGENT_GROUPS)
    pool = ({k: v for k, v in tot.items() if k in URGENT_GROUPS}
            if urgent >= DOWNGRADE_BLOCK_MIN else tot)
    name, p = max(pool.items(), key=lambda kv: kv[1])
    conf = p * (abnormal_p if abnormal_p is not None else 1.0)
    if conf < GROUP_CONF_MIN:
        return None
    return {"name": name,
            "prob": round(float(p), 4),
            "percent": round(float(p) * 100, 1),
            "confidence": round(float(conf), 4),
            # ⚠️ 앱·콘솔이 이 문장들을 **그대로** 띄우게 합니다. 각자 지어 쓰면
            #    표현이 갈리고, 갈리면 한쪽이 단정적으로 읽힙니다.
            # ⚠️ 새 이름은 "…변화 / …혹 / …상처" 로 끝나 **"계열" 을 붙이면 어색**합니다
            #    ("피부 표면·색·두께 변화 계열에"). 넷 다 받침과 무관하게 조사가
            #    "에" 라 그대로 이어 붙습니다.
            "text": f"모양만 보면 {name}에 가깝습니다.",
            # ★ 보호자가 사진에서 **직접 확인할 수 있는** 특징 (2026-09-10).
            #   이름만으로는 자기 개 사진과 대조가 안 됩니다.
            "feature": GROUP_FEATURE.get(name, ""),
            # ★ "자세히 보기" 전용 — 본문에 띄우지 마세요 (STEP 30 과잉 문제).
            "detail": GROUP_DETAIL.get(name, ""),
            "caveat": "진단이 아닙니다. 염증·감염·기생충·알레르기·면역질환 등 "
                      "여러 원인에서 나타날 수 있어 모양만으로는 원인을 알 수 없어요."}


def contract(verdict: str, *, abnormal_p: float | None = None,
             threshold: float | None = None, calibrated: bool = False,
             stage2: list[tuple[str, float]] | None = None,
             text: str = "", meta: dict | None = None) -> dict:
    """앱이 받는 JSON. **여기에 "1등" 필드를 추가하지 마세요.**

    추가하는 순간 앱은 그걸 화면에 크게 띄웁니다 — 그게 우리가 막으려던 것입니다.
    `tests/test_agent.py` 가 금지 키 목록을 들고 감시합니다.
    """
    from src.message import DISCLAIMER

    if verdict not in {"normal", "abnormal", "retake"}:
        raise ValueError(f"verdict 는 normal/abnormal/retake 중 하나입니다 — {verdict!r}")

    HEAD = {
        "normal": "뚜렷한 이상 소견은 보이지 않습니다.",
        "abnormal": "피부에 이상 소견이 보입니다.",
        "retake": "판단이 어려운 사진입니다.",
    }
    BODY = {
        "normal": ("다만 사진 한 장으로 확인할 수 있는 범위에는 한계가 있습니다. "
                   "가려워하거나, 냄새가 나거나, 계속 핥는 등 평소와 다른 행동이 있다면 "
                   "결과와 무관하게 병원에 가보시는 것을 권합니다."),
        "abnormal": ("무엇 때문인지까지는 이 사진만으로 알 수 없습니다. "
                     "아래는 모델이 비슷하다고 본 정도이며, 진단이 아닙니다."),
        "retake": ("이상한 부위가 화면 가운데에 오도록, 밝은 곳에서 초점을 맞춰 다시 찍어주세요. "
                   "털에 가려져 있다면 손으로 살짝 헤쳐 피부가 보이게 해주시면 좋습니다."),
    }
    ACTION = {
        "normal": "평소와 다른 점이 있으면 진료를 받아보세요.",
        "abnormal": "수의사 진료를 받아보시기를 권합니다.",
        "retake": "사진을 다시 찍어주세요.",
    }

    return {
        "contract_version": CONTRACT_VERSION,
        "verdict": verdict,
        "headline": HEAD[verdict],
        "body": BODY[verdict],
        "action": ACTION[verdict],
        "stage1": {
            "abnormal_prob": None if abnormal_p is None else round(float(abnormal_p), 4),
            "abnormal_percent": None if abnormal_p is None else round(float(abnormal_p) * 100, 1),
            "threshold": None if threshold is None else round(float(threshold), 4),
            # ⚠️ **하드코딩하지 마세요.** 엔진이 실제로 물고 있는 T 를 보고 정합니다.
            #    보정 안 된 확률을 "보정됨" 으로 내보내면 앱이 그걸 믿고 띄웁니다.
            "calibrated": bool(calibrated),
        },
        # ★ 2026-09-08 화면 규격 — **앱은 `groups`(계열 4개)만 그립니다.**
        #   `distribution`(6종)은 계약에 그대로 남습니다: 콘솔이 쓰고, 마음이
        #   바뀌어도 앱 한 줄이지 규격 변경이 아닙니다.
        #
        #   groups        계열 4묶음 **분포** — 확신과 무관하게 항상 나옵니다 (막대)
        #   group         계열 **주장** — 확신이 낮으면 `null` (한 줄)
        #   alert         "덩어리가 의심됩니다" — 안 뜨면 `null`
        #   distribution  병변 6종. 앱은 **안 그립니다**
        #
        # ⚠️ `null` 이면 통째로 안 그리면 됩니다 — 필드가 늘어도 안 깨집니다.
        "stage2": {"shown": bool(stage2), "distribution": _dist(stage2 or []),
                   "groups": lesion_group_dist(stage2),
                   "group": lesion_group(stage2, abnormal_p),
                   "alert": a6_alert(stage2, abnormal_p)},
        "text": text,
        "disclaimer": DISCLAIMER,
        "meta": {**(meta or {})},
    }


# ──────────────────────────────────────────────────────────────
# 진짜 에이전트
# ──────────────────────────────────────────────────────────────
class ScreeningAgent:
    """1단계 + 2단계 체크포인트를 물고 사진 한 장을 판정합니다."""

    def __init__(self, stage1, stage2, threshold: float,
                 stage1_tag: str = STAGE1_TAG, stage2_tag: str = STAGE2_TAG,
                 extra_arms: "list[tuple[Any, str]] | None" = None):
        """`extra_arms` 는 2단계를 **앙상블**로 돌릴 때의 추가 팔입니다.

        `[(engine, crop_tag), ...]` — 팔마다 **자기 크롭**으로 자릅니다.
        이득의 정체가 '비례 창 vs 고정 창' 이라 크롭이 달라야 합니다 (STEP 25).

        실측(STEP 33 holdout): 계열 4군 커버리지 58.4% → **67.9%**.
        값은 전체 파이프라인 **1.36배**(774 → 1052ms, CPU 1장) · 메모리 +0.16GB.
        """
        # stage2 가 None 이면 **1단계만** 돕니다 (정상/이상까지).
        self.s1, self.s2, self.thr = stage1, stage2, float(threshold)
        self.tag1, self.tag2 = stage1_tag, stage2_tag
        # 첫 팔이 릴리스입니다. `s2`·`tag2` 는 그대로 두어 기존 호출부가 안 깨집니다.
        self.arms2: list[tuple[Any, str]] = (
            [] if stage2 is None else [(stage2, stage2_tag), *(extra_arms or [])])
        from src.stages import ABNORMAL_LABEL

        self._ab = ABNORMAL_LABEL
        if self._ab not in stage1.classes:
            raise ValueError(
                f"1단계 엔진의 클래스가 {stage1.classes} 입니다 — '{self._ab}' 가 없습니다."
            )

    @classmethod
    def from_release(cls, release: str | Path, device: str | None = None,
                     stage1_only: bool = False) -> "ScreeningAgent":
        """노트북 06 이 만든 `release/` 폴더 하나만 주면 알아서 찾습니다.

            release/
              stage1_threshold.json
              checkpoints/stage1_effnetv2_s_f320_.../best.pt
              checkpoints/stage2_convnextv2_base_m2.5_.../best.pt

        어느 파일이 1단계인지 사람이 고를 필요가 없습니다 —
        **이름에 다 적혀 있습니다** (`train.infer_run_settings` 와 같은 규칙).
        """
        import json

        root = Path(release).expanduser()
        ck = root / "checkpoints"
        if not ck.is_dir():
            # 압축을 풀면 한 겹 더 감싸져 있는 일이 흔합니다
            # (Downloads/release/release/checkpoints/… 또는 archive/release/…).
            # 그래서 아래로 훑어서 stage1_…/best.pt 를 찾아 그 부모를 씁니다.
            hit = next((q for q in sorted(root.glob("**/stage1_*/best.pt"))), None)
            ck = hit.parent.parent if hit else root
        # ★ 2단계가 **여럿이면 앙상블**로 뭅니다 (STEP 25~33).
        #   릴리스에 `stage2_…` 폴더를 하나만 두면 예전과 똑같이 돕니다 —
        #   두 개 더 넣으면 자동으로 3팔이 됩니다. 켜고 끄는 스위치가 따로
        #   없는 게 의도입니다: **릴리스에 넣은 것이 곧 구성**입니다.
        #   ⚠️ 첫 팔(= 이름 순 첫 번째)이 기준이 되므로, 릴리스 모델이
        #      `stage2_convnextv2_…` 처럼 앞서도록 이름을 두세요.
        found: dict[str, Path] = {}
        stage2_all: list[Path] = []
        for d in sorted(ck.iterdir() if ck.is_dir() else []):
            if not (d / "best.pt").exists():
                continue
            if d.name.startswith("stage1_"):
                found["stage1"] = d / "best.pt"
            elif d.name.startswith("stage2_"):
                stage2_all.append(d / "best.pt")
                found.setdefault("stage2", d / "best.pt")
        if "stage1" not in found:
            # 못 찾았을 때 **뭐가 있는지 보여줍니다.** "못 찾았습니다" 만 던지면
            # 폴더를 잘못 준 건지 다운로드가 덜 된 건지 알 수가 없습니다.
            here = ([f"  {d.name}/" + ("  ← best.pt 있음" if (d / "best.pt").exists() else "")
                     for d in sorted(ck.iterdir())] if ck.is_dir()
                    else [f"  (폴더가 아닙니다: {ck})"])
            raise FileNotFoundError(
                f"{ck} 안에서 'stage1_…/best.pt' 를 못 찾았습니다.\n"
                "거기 있는 것:\n" + ("\n".join(here[:20]) or "  (비어 있음)") +
                "\n\n노트북 06 의 Output 에서 `release` 폴더를 통째로 받으셨나요? "
                "안에 checkpoints/stage1_…/best.pt 가 있어야 합니다.")
        if not stage1_only and "stage2" not in found:
            raise FileNotFoundError(
                f"{ck} 안에서 'stage2_…/best.pt' 를 못 찾았습니다. "
                "1단계만 돌리려면 stage1_only=True (CLI 는 --stage1-only).")

        thr = None
        for c in (root / "stage1_threshold.json", ck.parent / "stage1_threshold.json",
                  *sorted(root.glob("**/stage1_threshold.json"))):
            if c.exists():
                thr = json.loads(c.read_text(encoding="utf-8"))["threshold"]
                break
        if len(stage2_all) > 1:
            print(f"[agent] 2단계 앙상블 {len(stage2_all)}팔: "
                  + ", ".join(p.parent.name for p in stage2_all))
        ag = cls.load(found["stage1"], found.get("stage2"), thr, device,
                      stage1_only=stage1_only,
                      ckpt2_extra=stage2_all[1:])
        # ★ `describe()`(=/healthz) 가 **어느 폴더의 무엇**인지 말할 수 있게.
        #   로그가 아니라 값으로 남겨야 배포 뒤에도 확인됩니다.
        ag.release_dir = str(root)
        ag.arm_names = [p.parent.name for p in stage2_all]
        return ag

    def describe(self) -> dict:
        """★ **지금 무엇을 물고 있나** — 헬스체크가 쓰는, 추론 없는 요약.

        왜 있나 — 배포에서 앙상블이 **조용히 1팔로 줄어든 적**이 있습니다
        (2026-09-07). 응답은 200 이었고 에러도 경고도 없었습니다. 알아챌 단서가
        로그에 `[agent] … 3팔:` 이 **안 찍힌 것**, 즉 *성공 로그의 부재*뿐이라
        아무도 못 봤습니다.

        → **없는 줄을 찾게 하지 말고, 있는 값을 보게 합니다.** 이 dict 를
        `/healthz` 가 그대로 실어 보내면 배포 확인이 `curl` 한 번입니다.

        ⚠️ 무거운 일을 하지 마세요 — 헬스체크는 자주 불립니다.
        """
        return {
            "stage1_crop": self.tag1,
            "stage2_crop": self.tag2,
            "stage2_arms": len(self.arms2),
            "stage2_crops": [t for _, t in self.arms2],
            "stage2_experiments": list(getattr(self, "arm_names", [])),
            "threshold": float(self.thr),
            "release_dir": str(getattr(self, "release_dir", "") or "") or None,
        }

    @classmethod
    def load(cls, ckpt1: str | Path, ckpt2: str | Path | None = None,
             threshold: float | None = None, device: str | None = None,
             stage1_only: bool = False,
             ckpt2_extra: "list[str | Path] | None" = None) -> "ScreeningAgent":
        """체크포인트 두 개로 에이전트를 세웁니다.

        threshold 를 안 주면 1단계 체크포인트 옆의 `stage1_threshold.json` 을 찾습니다
        (노트북 03/06 이 저장합니다). 그것도 없으면 에러 — **기본값 0.5 로 조용히
        넘어가면 안 됩니다.** recall 0.95 를 사려고 0.1823 까지 내린 값입니다.

        크롭 태그는 **체크포인트 폴더 이름에서** 읽습니다. 백본을 이름에서 읽는
        것과 같은 이유입니다 — 하드코딩했다가 05 가 죽은 적이 있습니다
        (`train.model_key_from_exp` 주석).
        """
        import json

        from src.infer import Engine

        if threshold is None:
            for p in (Path(ckpt1).parent / "stage1_threshold.json",
                      Path(ckpt1).parent.parent / "stage1_threshold.json"):
                if p.exists():
                    threshold = json.loads(p.read_text())["threshold"]
                    break
        if threshold is None:
            raise FileNotFoundError(
                "1단계 임계값을 못 찾았습니다. `stage1_threshold.json` 을 체크포인트 옆에 두거나 "
                "threshold= 로 직접 주세요. 기본값을 쓰면 recall 이 조용히 무너집니다."
            )
        if stage1_only or ckpt2 is None:
            # 1단계만. "이상" 까지만 말하고 병변 분포는 안 냅니다 —
            # 멘토 피드백대로 이름은 어차피 안 말하므로 이것만으로도 제품이 됩니다.
            return cls(Engine.load(ckpt1, device=device), None, threshold,
                       crop_tag_from_exp(Path(ckpt1).parent.name) or STAGE1_TAG)
        # ★ 추가 팔도 **폴더 이름에서** 크롭 태그를 읽습니다 — ckpt2 와 같은 규칙.
        #   팔마다 크롭이 달라야 앙상블이 값을 합니다 (STEP 25: 비례 창 vs 고정 창).
        extra = [(Engine.load(c, device=device),
                  crop_tag_from_exp(Path(c).parent.name) or STAGE2_TAG)
                 for c in (ckpt2_extra or [])]
        return cls(Engine.load(ckpt1, device=device),
                   Engine.load(ckpt2, device=device), threshold,
                   crop_tag_from_exp(Path(ckpt1).parent.name) or STAGE1_TAG,
                   crop_tag_from_exp(Path(ckpt2).parent.name) or STAGE2_TAG,
                   extra_arms=extra)

    # -------------------------------------------------------
    def screen(self, image: "str | Path | Any", box=None) -> dict:
        """사진 한 장 → 판정 dict.

        Args:
            image: 경로 또는 PIL 이미지.
            box: 앱의 **가이드 프레임**. 정규화 `[x, y, w, h]` (0~1, 원본 기준).
                주면 학습과 같은 함수로 자릅니다. 없으면 화면 중앙으로 물러섭니다.
        """
        import tempfile

        from PIL import Image

        from src.message import Prediction, band, compose_screening_message

        t0 = time.perf_counter()
        try:
            im = image if hasattr(image, "size") else Image.open(image)
            im = im.convert("RGB")
        except Exception as exc:
            return contract("retake", meta={"error": f"이미지를 열 수 없습니다: {exc}"})

        cal1 = getattr(self.s1, "T", 1.0) not in (None, 1.0)
        meta = base_meta(mock=False, tag1=self.tag1, tag2=self.tag2,
                         temperature=getattr(self.s1, "T", 1.0), box=box)

        # ★ 밴드 밖 사진은 **모델에 넣기 전에** 돌려보냅니다.
        #   그 구간에서 성능이 떨어지는 걸 이미 재 뒀는데(STEP 10), 넣고 나서
        #   틀리는 것보다 안 넣는 편이 낫습니다.
        if box is not None:
            g = check_guide(box)
            meta["guide"] = g
            if not g["ok"]:
                meta["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                return contract("retake", text="", calibrated=cal1,
                                meta={**meta, "retake_reason": g["reason"]})

        im = to_train_space(im)                       # 짧은 변 1080 = 학습 픽셀 공간
        bbox = box_to_px(box, *im.size) if box is not None else None

        with tempfile.TemporaryDirectory() as td:
            p1 = Path(td) / "s1.jpg"
            crop_for(im, bbox, self.tag1).save(p1, quality=95)
            pr1 = self.s1.predict(str(p1))
            abnormal = dict(pr1.topk).get(self._ab, 0.0)

            if abnormal < self.thr:
                pred = Prediction(topk=[(NORMAL_LABEL, 1 - abnormal)],
                                  confidence_band=band(1 - abnormal),
                                  stage1_abnormal=abnormal)
                meta["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                return contract("normal", abnormal_p=abnormal, threshold=self.thr,
                                calibrated=cal1,
                                text=compose_screening_message(pred), meta=meta)

            if self.s2 is None:                       # 1단계만 돌리는 구성
                pred = Prediction(topk=[], stage1_abnormal=abnormal,
                                  confidence_band=band(abnormal))
                meta["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                meta["stage2_crop"] = None
                return contract("abnormal", abnormal_p=abnormal, threshold=self.thr,
                                calibrated=cal1,
                                text=compose_screening_message(pred, abnormal), meta=meta)

            # ★ 팔마다 **자기 크롭**으로 자르고 확률을 평균합니다.
            #   팔이 하나면 예전과 똑같이 돕니다 (평균할 게 없으니).
            sums: dict[str, float] = {}
            for k, (eng, tag) in enumerate(self.arms2):
                pk = Path(td) / f"s2_{k}.jpg"
                crop_for(im, bbox, tag).save(pk, quality=95)
                pr = eng.predict(str(pk))
                if k == 0:
                    pred = pr                        # 기권·문구는 첫 팔 기준
                for c, p in pr.topk:
                    sums[c] = sums.get(c, 0.0) + float(p)
            n_arms = max(len(self.arms2), 1)
            pred.topk = sorted(((c, s / n_arms) for c, s in sums.items()),
                               key=lambda kv: -kv[1])

        meta["stage2_arms"] = len(self.arms2)
        meta["stage2_crops"] = [t for _, t in self.arms2]
        raw = list(pred.topk)                       # 깎기 전 원본 (합 = 1)
        pred.stage2_probs = raw
        pred.stage1_abnormal = abnormal
        pred.topk = [(c, p * abnormal) for c, p in raw]
        pred.confidence_band = band(pred.topk[0][1]) if pred.topk else "낮음"

        # ★ 기권은 **깎기 전** 확률로 판정합니다.
        #
        # ⚠️ 예전에는 `pred.topk[0][1]`(= 원본 × 1단계 확률)과 비교했습니다.
        #    abstain_threshold 는 "2단계가 이만큼 확신하는가" 라는 뜻인데 1단계
        #    확률을 한 번 더 곱해 비교하면 **이중 감점**이 됩니다 — 1단계가 95%
        #    확신해도 2단계가 47% 넘게 확신해야 통과했습니다. 6종 분류에서는
        #    거의 없는 일이라 사실상 전부 기권했습니다.
        #    노트북 06 이 문턱을 `coverage_risk_curve(probs_after)` — 곱하지 않은
        #    확률 — 로 뽑는 것과도 척도가 어긋나 있었습니다. 이제 맞습니다.
        pred.abstain = bool(raw) and raw[0][1] < self.s2.cfg.abstain_threshold

        # ★ **기권해도 재촬영으로 보내지 않습니다.**
        #
        # 기권은 병변 *종류*를 말할지 말지를 정하는 장치입니다. 1단계는 이미
        # "이상" 이라고 판정했고, 종류를 모른다는 게 괜찮다는 뜻은 아닙니다.
        # 노트북 06 §5 에 그렇게 적혀 있는데 코드만 어기고 있었습니다:
        # 1단계가 95.6% 로 본 사진이 "다시 찍어주세요" 로 나갔습니다.
        #
        # 게다가 D-023 이후로 **병변 이름을 아예 말하지 않습니다.** 기권이
        # 감추려던 것(못 믿을 이름)을 애초에 안 보여주므로, 기권이 판정을
        # 뒤집을 이유가 남아 있지 않습니다. 분포는 그대로 보여주고 — 여섯 개가
        # 고만고만한 것이 곧 "확실하지 않다" 입니다 — 확신이 낮다는 사실만
        # meta 로 알립니다.
        #
        # `retake` 는 이제 **모델을 돌리기 전** 판단만 남습니다:
        # 이미지를 못 열었을 때, 가이드 프레임이 밴드 밖일 때.
        # ⚠️ `stage2_top_prob` 은 이름이 아니라 **숫자**입니다. 분포 1등의
        #    확률이라 distribution[0].prob 와 같은 값이고, 새 정보를 흘리지
        #    않습니다. 이름 필드는 만들지 마세요.
        meta.update(stage2_meta(pred, raw, self.s2.cfg.abstain_threshold))

        meta["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return contract("abnormal", abnormal_p=abnormal, threshold=self.thr,
                        calibrated=cal1, stage2=raw,
                        text=compose_screening_message(pred), meta=meta)


def crop_tag_from_exp(name: str) -> str | None:
    """체크포인트 폴더 이름에서 크롭 태그를 되찾습니다.

        stage1_effnetv2_s_f320_384_moderate_photometric → 'f320'
        stage2_convnextv2_base_m2.5_384_moderate        → 'm2.5'
    """
    from src import crop as _crop

    if not isinstance(name, str):
        return None
    for t in name.split("_"):
        if t == "full" or _crop.margin_of_tag(t) or _crop.fixed_of_tag(t):
            return t
    return None


# ──────────────────────────────────────────────────────────────
# 가중치 없이 화면만 보기
# ──────────────────────────────────────────────────────────────
class MockAgent:
    """모델 없이 **똑같은 모양의** 응답을 만듭니다 — 데모/앱 연동 확인용.

    torch 도 가중치도 필요 없습니다. 파일 내용의 해시로 값을 만들기 때문에
    같은 사진은 항상 같은 결과가 나옵니다 (시연 중에 숫자가 흔들리면 곤란합니다).

    가이드 프레임 검사(`check_guide`)는 **진짜로 돕니다** — 밴드 밖이면 mock
    에서도 재촬영이 나옵니다. 앱이 그 경로를 확인할 수 있어야 하니까요.

    ⚠️ 확률은 **모델이 낸 것이 아닙니다.** 응답의 `meta.mock` 이 true 입니다.
    """

    #: mock 전용 임계값. **릴리스 값이 아닙니다** — 릴리스는
    #: `stage1_threshold.json` 에 있고 `ScreeningAgent` 는 그게 없으면
    #: **아예 안 뜹니다**(기본값을 쓰면 recall 이 조용히 무너지므로).
    #: 여기 값은 화면만 볼 때 쓰는 자리표시자이고, 있으면 진짜 값을 읽습니다.
    MOCK_THRESHOLD = 0.1823

    def __init__(self, threshold: float | None = None):
        if threshold is None:
            threshold = self.MOCK_THRESHOLD
            try:                       # 릴리스가 옆에 있으면 그걸 씁니다
                from src import env

                f = env.work_root() / "stage1_threshold.json"
                if f.is_file():
                    threshold = float(json.loads(f.read_text(encoding="utf-8"))
                                      ["threshold"])
            except Exception:          # 데이터가 없는 환경(앱 개발용)이면 그냥 넘어갑니다
                pass
        self.thr = float(threshold)
        self.tag1, self.tag2 = STAGE1_TAG, STAGE2_TAG

    def describe(self) -> dict:
        """`ScreeningAgent.describe()` 와 **같은 키**를 냅니다.

        ⚠️ mock 과 real 이 다른 키를 낸 적이 여섯 번 있습니다 — 그래서 계약을
        두 곳에 적지 않고 `tests/test_serve_contract.py` 가 대조합니다.
        mock 은 팔이 하나뿐이라 `stage2_arms = 1` 입니다.
        """
        return {"stage1_crop": self.tag1, "stage2_crop": self.tag2,
                "stage2_arms": 1, "stage2_crops": [self.tag2],
                "stage2_experiments": [], "threshold": float(self.thr),
                "release_dir": None}

    def screen(self, image: "str | Path | Any", box=None) -> dict:
        from src.message import Prediction, band, compose_screening_message

        t0 = time.perf_counter()
        try:
            raw_bytes = (Path(image).read_bytes() if not hasattr(image, "size")
                         else image.tobytes())
        except Exception as exc:
            return contract("retake", meta={"mock": True, "error": str(exc)})

        # 진짜와 **같은 함수**로 만듭니다 — 계약을 두 곳에 적지 않습니다.
        meta = base_meta(mock=True, tag1=self.tag1, tag2=self.tag2,
                         temperature=1.0, box=box)   # mock 은 보정을 안 합니다
        if box is not None:
            g = check_guide(box)
            meta["guide"] = g
            if not g["ok"]:
                meta["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                return contract("retake", meta={**meta, "retake_reason": g["reason"]})

        h = hashlib.sha256(raw_bytes).digest()
        abnormal = 0.03 + (h[0] / 255) * 0.94
        meta["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        if abnormal < self.thr:
            pred = Prediction(topk=[(NORMAL_LABEL, 1 - abnormal)],
                              confidence_band=band(1 - abnormal), stage1_abnormal=abnormal)
            return contract("normal", abnormal_p=abnormal, threshold=self.thr,
                            text=compose_screening_message(pred), meta=meta)

        w = [1 + h[i + 1] / 255 * 3 for i in range(len(CLASSES))]
        raw = sorted(zip(CLASSES, [x / sum(w) for x in w]), key=lambda kv: -kv[1])
        pred = Prediction(topk=[(c, p * abnormal) for c, p in raw],
                          confidence_band=band(raw[0][1] * abnormal),
                          stage1_abnormal=abnormal)
        pred.stage2_probs = raw
        from src.config import CFG as _CFG

        # 진짜와 **같은 키**를 냅니다 (test_serve_contract [6] 이 감시).
        # mock 은 팔이 하나뿐이라 릴리스 구성과 같은 값을 냅니다.
        meta["stage2_arms"] = 1
        meta["stage2_crops"] = [self.tag2]
        meta.update(stage2_meta(pred, raw, _CFG().abstain_threshold))
        return contract("abnormal", abnormal_p=abnormal, threshold=self.thr,
                        stage2=raw, text=compose_screening_message(pred), meta=meta)
