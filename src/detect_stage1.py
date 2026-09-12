"""검출기를 **1단계 엔진 자리**에 끼우는 어댑터 (STEP 52 배선 초안 — 관문 통과 전엔 배포에 안 붙입니다).

    from src.detect_stage1 import DetectorStage1
    s1 = DetectorStage1.load("data/work/dfine_step52/dfine_best", threshold=0.35)
    agent = ScreeningAgent(s1, stage2, threshold=s1.threshold, stage1_tag="whole", extra_arms=[...])

왜 어댑터인가 — `ScreeningAgent.screen()` 은 1단계에서 `s1.predict(path).topk` 와 `s1.classes` 만 씁니다.
그 둘만 맞추면 `screen()`·계약·2단계 앙상블·하향 방지·A6 경보는 **한 줄도 안 바뀝니다.**
검출기는 사진 **전체**를 보므로 `stage1_tag="whole"` 로 두면 `crop_for` 가 자르지 않고 넘깁니다 —
가이드 프레임 좌표가 1단계 입력에서 빠지는 것이 이 배선의 요점입니다 (STEP 43 의 −0.14 가 사라질 자리).

점수 = 최고 쿼리의 sigmoid (STEP 52 사전등록의 정의와 같음). `T` 는 없습니다(보정은 통과 뒤 따로) —
`getattr(s1, "T", 1.0)` 이 1.0 을 돌려주므로 계약의 `calibrated` 가 False 로 정직하게 나갑니다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.message import Prediction
from src.stages import ABNORMAL_LABEL

CLASSES_STAGE1 = ["A7", ABNORMAL_LABEL]


class DetectorStage1:
    classes = CLASSES_STAGE1

    def __init__(self, model: Any, processor: Any, device: str = "cpu", threshold: float | None = None):
        self.model, self.proc, self.device = model.to(device).eval(), processor, device
        self.threshold = threshold          # 사전등록 판정 뒤 val 에서 고른 값. None 이면 호출부가 정해야 합니다
        self.T = 1.0                        # 보정 없음 — 계약이 calibrated=False 로 내보냅니다

    @classmethod
    def load(cls, path: str | Path, device: str | None = None, threshold: float | None = None) -> "DetectorStage1":
        import torch
        from transformers import AutoImageProcessor, DFineForObjectDetection

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        return cls(DFineForObjectDetection.from_pretrained(str(path)), AutoImageProcessor.from_pretrained(str(path)), dev, threshold)

    def score_and_box(self, image) -> tuple[float, list[float] | None]:
        """(병변 점수 0~1, top-1 네모 0~1 xyxy 또는 None). 네모는 **표시용**입니다 — 크롭에 쓰지 않습니다 (STEP 50·51)."""
        import torch
        from PIL import Image

        im = image if hasattr(image, "size") else Image.open(image)
        im = im.convert("RGB")
        with torch.no_grad():
            out = self.model(**self.proc(images=im, return_tensors="pt").to(self.device))
        score = float(out.logits.float().sigmoid().amax())
        r = self.proc.post_process_object_detection(out, threshold=0.0, target_sizes=[(1, 1)])[0]
        box = r["boxes"][int(r["scores"].argmax())].float().tolist() if len(r["scores"]) else None
        return score, box

    def predict(self, path: str, tta: bool | None = None) -> Prediction:
        """`Engine.predict` 와 같은 모양 — `topk` 에 (ABNORMAL, p)·(A7, 1−p)."""
        p, box = self.score_and_box(path)
        pred = Prediction(topk=sorted([(ABNORMAL_LABEL, p), ("A7", 1.0 - p)], key=lambda kv: -kv[1]),
                          confidence_band="", image=str(path))
        pred.detector_box = box            # 화면 표시용. 계약에 실을지는 배선 때 결정
        return pred
