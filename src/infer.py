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
from pathlib import Path

import numpy as np
import torch

from src.config import CFG, CLASS_KO, CLASSES, NORMAL_LABEL

# 문구 생성은 src/message.py 로 옮겼습니다 (torch 없이 쓰려고).
# 여기서 재수출하므로 `from src.infer import compose_message` 는 그대로 됩니다.
from src.message import (                                          # noqa: F401,E402
    DISCLAIMER,
    Prediction,
    _cells,
    _pad,
    band,
    compose_message,
    compose_screening_message,
)


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
