"""STEP 52 배선 초안 — 검출기 어댑터가 ScreeningAgent 의 1단계 인터페이스를 지키는가 (모델 없이).

    uv run --extra train python -m pytest -q tests/test_detect_stage1.py
"""
import inspect

import torch

from src import agent
from src.detect_stage1 import DetectorStage1
from src.stages import ABNORMAL_LABEL


class _FakeOut:
    def __init__(self, logits):
        self.logits = logits


class _FakeModel(torch.nn.Module):
    def __init__(self, peak):
        super().__init__(); self.peak = peak
    def forward(self, **kw):
        z = torch.full((1, 300, 1), -6.0); z[0, 7, 0] = torch.logit(torch.tensor(self.peak)); return _FakeOut(z)


class _FakeProc:
    def __call__(self, images, return_tensors):
        class _B(dict):
            def to(self, d): return self
        return _B(pixel_values=torch.zeros(1, 3, 8, 8))
    def post_process_object_detection(self, out, threshold, target_sizes):
        s = out.logits.sigmoid()[0, :, 0]
        return [{"scores": s, "boxes": torch.tensor([[0.2, 0.3, 0.6, 0.7]]).repeat(len(s), 1), "labels": torch.zeros(len(s))}]


def test_adapter_matches_engine_interface_and_scores(tmp_path):
    s1 = DetectorStage1(_FakeModel(0.83), _FakeProc(), device="cpu", threshold=0.4)
    assert s1.classes == ["A7", ABNORMAL_LABEL] and ABNORMAL_LABEL in s1.classes      # agent.__init__ 의 검사
    assert getattr(s1, "T", 1.0) == 1.0                                               # calibrated=False 로 나가야 함
    p = tmp_path / "x.jpg"
    from PIL import Image
    Image.new("RGB", (64, 64)).save(p)
    pred = s1.predict(str(p))
    d = dict(pred.topk)
    assert abs(d[ABNORMAL_LABEL] - 0.83) < 1e-5 and abs(d["A7"] - 0.17) < 1e-5 and pred.topk[0][0] == ABNORMAL_LABEL
    assert all(abs(a - b) < 1e-5 for a, b in zip(pred.detector_box, [0.2, 0.3, 0.6, 0.7]))
    # ScreeningAgent 가 1단계에서 쓰는 것은 predict().topk 와 classes 뿐이어야 어댑터가 성립합니다
    src = inspect.getsource(agent.ScreeningAgent.screen)
    assert "self.s1.predict(" in src and "self.s1.cfg" not in src and "self.s1.model" not in src


def test_whole_tag_passes_entire_image_to_stage1():
    from PIL import Image
    im = Image.new("RGB", (1920, 1080))
    assert agent.crop_for(im, None, "whole").size == im.size                            # 검출기는 사진 전체를 봅니다
    assert agent.crop_for(im, None, "full").size == (1080, 1080)                        # `full` 은 중앙 정사각 — 다른 것
    assert agent.crop_for(im, [100, 100, 300, 300], "whole").size == im.size            # 네모를 줘도 무시
