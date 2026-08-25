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
# 서빙 크롭 — 학습과 배포의 간극이 여기 있습니다
# ──────────────────────────────────────────────────────────────
#
# 학습은 라벨 bbox 를 중심으로 잘랐습니다 (1단계 f320 / 2단계 m2.5).
# **보호자 사진에는 bbox 가 없습니다.** 그래서 서빙에서는 촬영 가이드를 지켰다는
# 전제로 **화면 중앙**을 씁니다 — 가이드가 "병변을 가운데, 화면 가로의 34~56%" 를
# 요구하는 이유가 정확히 이것입니다.
#
# f320 의 크기는 원본 픽셀 320 이었고, 원본은 1920×1080 이었습니다
# (`docs/data/DATASET_CARD.md` §1). 즉 **짧은 변의 29.6%** 입니다.
# 휴대폰 사진은 해상도가 다르므로 픽셀 320 을 그대로 쓰면 안 됩니다 —
# 같은 **비율**로 잘라야 피부 1mm 가 학습 때와 비슷한 픽셀 수가 됩니다.
F320_PX = 320
TRAIN_SHORT_SIDE = 1080            # DATASET_CARD §1 — metaData.resolution "1920X1080"
STAGE1_FRAC = F320_PX / TRAIN_SHORT_SIDE          # ≈ 0.296
STAGE2_FRAC = 1.0                  # m2.5 는 배율 크롭이라 비율 고정이 불가 → 중앙 정사각

# ⚠️ 이 대응은 **실측된 적이 없습니다.**
# holdout 숫자(AUROC 0.9304 등)는 전부 bbox 중심 크롭에서 나왔습니다. 실제 앱
# 사진에서는 그보다 나쁠 수 있고, 얼마나 나쁜지는 앱으로 찍은 사진에 수의사
# 라벨을 붙여봐야 압니다. 응답 `meta.crop_untested` 가 이 사실을 실어 나릅니다.
CROP_UNTESTED_NOTE = (
    "서빙 크롭(중앙)은 학습 크롭(bbox 중심)과 다릅니다. "
    "보고된 성능은 bbox 크롭 기준이며 실사용 성능은 아직 실측하지 않았습니다."
)


def center_square(img, frac: float = 1.0):
    """중앙에서 정사각형을 잘라냅니다. frac 은 **짧은 변 대비 비율**입니다.

    frac=1.0 이면 짧은 변 크기의 정사각형(= 흔한 center crop),
    frac=0.296 이면 f320 이 1920×1080 에서 차지하던 만큼입니다.
    """
    w, h = img.size
    side = max(int(round(min(w, h) * float(frac))), 32)
    side = min(side, w, h)
    left, top = (w - side) // 2, (h - side) // 2
    return img.crop((left, top, left + side, top + side))


# ──────────────────────────────────────────────────────────────
# 응답 계약
# ──────────────────────────────────────────────────────────────
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
        "meta": {**{"crop_untested": CROP_UNTESTED_NOTE}, **(meta or {})},
    }


# ──────────────────────────────────────────────────────────────
# 진짜 에이전트
# ──────────────────────────────────────────────────────────────
class ScreeningAgent:
    """1단계 + 2단계 체크포인트를 물고 사진 한 장을 판정합니다."""

    def __init__(self, stage1, stage2, threshold: float,
                 stage1_frac: float = STAGE1_FRAC, stage2_frac: float = STAGE2_FRAC):
        self.s1, self.s2, self.thr = stage1, stage2, float(threshold)
        self.f1, self.f2 = float(stage1_frac), float(stage2_frac)
        from src.stages import ABNORMAL_LABEL

        self._ab = ABNORMAL_LABEL
        if self._ab not in stage1.classes:
            raise ValueError(
                f"1단계 엔진의 클래스가 {stage1.classes} 입니다 — '{self._ab}' 가 없습니다."
            )

    @classmethod
    def load(cls, ckpt1: str | Path, ckpt2: str | Path,
             threshold: float | None = None, device: str | None = None) -> "ScreeningAgent":
        """체크포인트 두 개로 에이전트를 세웁니다.

        threshold 를 안 주면 1단계 체크포인트 옆의 `stage1_threshold.json` 을 찾습니다
        (노트북 03/06 이 저장합니다). 그것도 없으면 에러 — **기본값 0.5 로 조용히
        넘어가면 안 됩니다.** recall 0.95 를 사려고 0.1823 까지 내린 값입니다.
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
        return cls(Engine.load(ckpt1, device=device),
                   Engine.load(ckpt2, device=device), threshold)

    # -------------------------------------------------------
    def screen(self, image: "str | Path | Any") -> dict:
        """사진 한 장 → 판정 dict. 경로도 되고 PIL 이미지도 됩니다."""
        import tempfile

        from PIL import Image

        from src.message import Prediction, band, compose_screening_message

        t0 = time.perf_counter()
        try:
            im = image if hasattr(image, "size") else Image.open(image)
            im = im.convert("RGB")
        except Exception as exc:
            return contract("retake", text="", meta={"error": f"이미지를 열 수 없습니다: {exc}"})

        with tempfile.TemporaryDirectory() as td:
            # 두 단계가 **다른 크롭**을 씁니다 — 학습 때와 같은 규칙입니다
            p1 = Path(td) / "s1.jpg"
            center_square(im, self.f1).save(p1, quality=95)
            pr1 = self.s1.predict(str(p1))
            abnormal = dict(pr1.topk).get(self._ab, 0.0)

            meta = {"elapsed_ms": None, "mock": False,
                    "stage1_crop": f"center {self.f1:.3f}× short side",
                    "stage2_crop": f"center {self.f2:.3f}× short side"}

            if abnormal < self.thr:
                pred = Prediction(topk=[(NORMAL_LABEL, 1 - abnormal)],
                                  confidence_band=band(1 - abnormal),
                                  stage1_abnormal=abnormal)
                meta["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                return contract("normal", abnormal_p=abnormal, threshold=self.thr,
                                text=compose_screening_message(pred), meta=meta)

            p2 = Path(td) / "s2.jpg"
            center_square(im, self.f2).save(p2, quality=95)
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


# ──────────────────────────────────────────────────────────────
# 가중치 없이 화면만 보기
# ──────────────────────────────────────────────────────────────
class MockAgent:
    """모델 없이 **똑같은 모양의** 응답을 만듭니다 — 데모/앱 연동 확인용.

    torch 도 가중치도 필요 없습니다. 파일 내용의 해시로 값을 만들기 때문에
    같은 사진은 항상 같은 결과가 나옵니다 (시연 중에 숫자가 흔들리면 곤란합니다).

    ⚠️ 여기서 나오는 숫자는 **모델이 낸 것이 아닙니다.** 응답의 `meta.mock` 이
    true 이고, 화면에도 그렇게 표시해야 합니다.
    """

    def __init__(self, threshold: float = 0.1823):
        self.thr = float(threshold)

    def screen(self, image: "str | Path | Any") -> dict:
        from src.message import Prediction, band, compose_screening_message

        t0 = time.perf_counter()
        try:
            raw_bytes = (Path(image).read_bytes() if not hasattr(image, "size")
                         else image.tobytes())
        except Exception as exc:
            return contract("retake", meta={"mock": True, "error": str(exc)})

        h = hashlib.sha256(raw_bytes).digest()
        abnormal = 0.03 + (h[0] / 255) * 0.94
        meta = {"mock": True, "stage1_crop": f"center {STAGE1_FRAC:.3f}× short side",
                "stage2_crop": "center 1.000× short side",
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}

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
