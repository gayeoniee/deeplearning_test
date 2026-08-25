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

# 촬영 가이드 밴드 — `robust.usable_range()` 가 STEP 10 에서 실측한 값입니다
# (STATUS.md "촬영 가이드"). 화면 **가로** 대비 병변의 비율입니다.
GUIDE_RECOMMEND = (0.34, 0.56)     # 하락 5% 이내
GUIDE_ALLOW = (0.28, 0.68)         # 하락 10% 이내
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
def _dist(probs: list[tuple[str, float]]) -> list[dict]:
    """분포를 앱이 그대로 그릴 수 있는 모양으로. **정렬은 하되 자르지 않습니다.**"""
    return [{"code": c,
             "name_ko": CLASS_KO.get(c, c),
             "name_en": CLASS_EN.get(c, c),
             "prob": round(float(p), 4),
             "percent": round(float(p) * 100, 1)}
            for c, p in sorted(probs, key=lambda kv: -kv[1])]


def contract(verdict: str, *, abnormal_p: float | None = None,
             threshold: float | None = None,
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
        "normal": "뚜렷한 피부 병변 소견은 보이지 않습니다.",
        "abnormal": "피부에 이상 소견이 보입니다.",
        "retake": "판단이 어려운 사진입니다.",
    }
    BODY = {
        "normal": ("다만 사진 한 장으로 확인할 수 있는 범위에는 한계가 있습니다. "
                   "가려워하거나, 냄새가 나거나, 계속 핥는 등 평소와 다른 행동이 있다면 "
                   "결과와 무관하게 병원에 가보시는 것을 권합니다."),
        "abnormal": ("어떤 병변인지는 이 사진만으로 판단할 수 없습니다. "
                     "아래는 모델이 비슷하다고 본 정도이며, 진단이 아닙니다."),
        "retake": ("병변 부위가 화면 가운데에 오도록, 밝은 곳에서 초점을 맞춰 다시 찍어주세요. "
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
            # ⚠️ 1단계는 온도 보정을 한 적이 없습니다 (2단계만 T=1.1063).
            #    이 확률을 사람에게 보여주는 이상 보정이 필요합니다 — STATUS.md 열린 문제 6번.
            "calibrated": False,
        },
        # 병변 6종 분포. verdict != "abnormal" 이면 비어 있습니다.
        "stage2": {"shown": bool(stage2), "distribution": _dist(stage2 or [])},
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
                 stage1_tag: str = STAGE1_TAG, stage2_tag: str = STAGE2_TAG):
        # stage2 가 None 이면 **1단계만** 돕니다 (정상/이상까지).
        self.s1, self.s2, self.thr = stage1, stage2, float(threshold)
        self.tag1, self.tag2 = stage1_tag, stage2_tag
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

        root = Path(release)
        ck = root / "checkpoints"
        if not ck.is_dir():
            ck = root                       # checkpoints/ 를 직접 준 경우
        found: dict[str, Path] = {}
        for d in sorted(ck.iterdir() if ck.is_dir() else []):
            if not (d / "best.pt").exists():
                continue
            for st in ("stage1", "stage2"):
                if d.name.startswith(st + "_"):
                    found[st] = d / "best.pt"
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
        for c in (root / "stage1_threshold.json", ck.parent / "stage1_threshold.json"):
            if c.exists():
                thr = json.loads(c.read_text())["threshold"]
                break
        return cls.load(found["stage1"], found.get("stage2"), thr, device,
                        stage1_only=stage1_only)

    @classmethod
    def load(cls, ckpt1: str | Path, ckpt2: str | Path | None = None,
             threshold: float | None = None, device: str | None = None,
             stage1_only: bool = False) -> "ScreeningAgent":
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
        return cls(Engine.load(ckpt1, device=device),
                   Engine.load(ckpt2, device=device), threshold,
                   crop_tag_from_exp(Path(ckpt1).parent.name) or STAGE1_TAG,
                   crop_tag_from_exp(Path(ckpt2).parent.name) or STAGE2_TAG)

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

        meta = {"mock": False, "stage1_crop": self.tag1, "stage2_crop": self.tag2,
                "box_source": "user" if box is not None else "center",
                "crop_note": CROP_NOTE["user_box" if box is not None else "center"]}

        # ★ 밴드 밖 사진은 **모델에 넣기 전에** 돌려보냅니다.
        #   그 구간에서 성능이 떨어지는 걸 이미 재 뒀는데(STEP 10), 넣고 나서
        #   틀리는 것보다 안 넣는 편이 낫습니다.
        if box is not None:
            g = check_guide(box)
            meta["guide"] = g
            if not g["ok"]:
                meta["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                return contract("retake", text="", meta={**meta, "retake_reason": g["reason"]})

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
                                text=compose_screening_message(pred), meta=meta)

            if self.s2 is None:                       # 1단계만 돌리는 구성
                pred = Prediction(topk=[], stage1_abnormal=abnormal,
                                  confidence_band=band(abnormal))
                meta["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                meta["stage2_crop"] = None
                return contract("abnormal", abnormal_p=abnormal, threshold=self.thr,
                                text=compose_screening_message(pred, abnormal), meta=meta)

            p2 = Path(td) / "s2.jpg"
            crop_for(im, bbox, self.tag2).save(p2, quality=95)
            pred = self.s2.predict(str(p2))

        raw = list(pred.topk)                       # 깎기 전 원본 (합 = 1)
        pred.stage2_probs = raw
        pred.stage1_abnormal = abnormal
        pred.topk = [(c, p * abnormal) for c, p in raw]
        pred.confidence_band = band(pred.topk[0][1]) if pred.topk else "낮음"
        pred.abstain = bool(pred.topk) and pred.topk[0][1] < self.s2.cfg.abstain_threshold

        meta["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        if pred.abstain:
            return contract("retake", abnormal_p=abnormal, threshold=self.thr,
                            text=compose_screening_message(pred), meta=meta)
        return contract("abnormal", abnormal_p=abnormal, threshold=self.thr,
                        stage2=raw, text=compose_screening_message(pred), meta=meta)


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

    def __init__(self, threshold: float = 0.1823):
        self.thr = float(threshold)
        self.tag1, self.tag2 = STAGE1_TAG, STAGE2_TAG

    def screen(self, image: "str | Path | Any", box=None) -> dict:
        from src.message import Prediction, band, compose_screening_message

        t0 = time.perf_counter()
        try:
            raw_bytes = (Path(image).read_bytes() if not hasattr(image, "size")
                         else image.tobytes())
        except Exception as exc:
            return contract("retake", meta={"mock": True, "error": str(exc)})

        meta = {"mock": True, "stage1_crop": self.tag1, "stage2_crop": self.tag2,
                "box_source": "user" if box is not None else "center",
                "crop_note": CROP_NOTE["user_box" if box is not None else "center"]}
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
        return contract("abnormal", abnormal_p=abnormal, threshold=self.thr,
                        stage2=raw, text=compose_screening_message(pred), meta=meta)
