"""병변 **검출** — 사람에게 안 묻고 모델이 네모를 찾습니다 (STEP 42).

## 왜 이게 남았나

네모를 받는 길이 **아홉 개** 닫혔습니다 (STEP 36~41). 전부 같은 곳에서
막힙니다 — **사람은 병변이 어디에 얼마나 크게 있는지 못 알려줍니다**
(크기 상관 −0.05 · 탭이 병변 안에 든 것 47.5%).

그런데 **`bbox` 라벨이 36만 장** 있습니다. 사람에게 묻지 말고 **모델이 그 일을
직접 배우게** 하는 정공법인데, 지금까지 시도조차 안 했습니다.

⚠️ STEP 37 의 *"모델이 제안"* 과 다릅니다. 거기서는 **분류기를 창 탐지기로
빌려 썼습니다** — 네모를 뽑도록 배운 적이 없는 모델이었고 6.7% 였습니다.
여기서는 **그 일을 직접 배웁니다.**

## 무엇을 배우나

사진 한 장 → 네모 하나 `(x1, y1, x2, y2)` 0~1 정규화.

⚠️ **한 장에 네모 하나**만 냅니다. 매니페스트의 `n_lesion` 은 라벨 항목 수라
병변 하나당 2입니다 — **78% 가 병변 1개**이므로 단일 회귀로 시작합니다.
여러 개인 22% 는 **가장 큰 것**을 정답으로 씁니다 (크롭을 걸 자리를 고르는
것이 목적이므로).

## 판정

`experiments.DETECT_MIN_USABLE` (밴드 안 비율) 과
`experiments.DETECT_MIN_COVERAGE_GAIN` (커버리지 이득) — **둘 다** 넘어야
합니다. 네모가 좋아 보여도 **크롭이 걸리면 중심 오차가 치명적이 될 수**
있습니다 (STEP 41).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class BoxHead(nn.Module):
    """백본 + 네모 회귀 헤드.

    ★ **중심·크기를 따로, 로그 공간에서** 냅니다:

        cx, cy   sigmoid  → 0~1
        log w, log h      → exp 해서 크기

    왜 로그인가 — 병변 크기가 화면의 **4.6% ~ 62.5%** 로 10배 넘게 흩어져
    있습니다. 선형으로 회귀하면 **큰 것에 손실이 쏠려** 작은 병변을 통째로
    놓칩니다 (STEP 17 에서 크기와 놓침의 관계를 이미 봤습니다).
    """

    def __init__(self, backbone: str = "effnetv2_s", pretrained: bool = True,
                 img_size: int = 384):
        super().__init__()
        from src import models

        self.body = models.build(backbone, n_classes=0, pretrained=pretrained,
                                 img_size=img_size, verbose=False)
        feat = getattr(self.body, "num_features", None)
        if feat is None:                      # timm 이 0-class 를 안 주는 경우
            with torch.no_grad():
                feat = self.body(torch.zeros(1, 3, img_size, img_size)).shape[-1]
        self.head = nn.Linear(int(feat), 4)
        # 초기값을 **화면 가운데 · 병변 중앙값 크기** 로 둡니다. 안 그러면 첫
        # 에폭이 그걸 찾는 데 쓰이고, 작은 데이터에서는 못 찾고 끝납니다.
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor(
                [0.0, 0.0, math.log(0.215), math.log(0.215)]))

    def forward(self, x):
        z = self.head(self.body(x))
        cx, cy = torch.sigmoid(z[:, 0]), torch.sigmoid(z[:, 1])
        w = torch.exp(z[:, 2].clamp(-4.0, 0.0))     # 1.8% ~ 100%
        h = torch.exp(z[:, 3].clamp(-4.0, 0.0))
        return torch.stack([cx, cy, w, h], 1)


def to_xyxy(cwh: torch.Tensor) -> torch.Tensor:
    """(cx, cy, w, h) → (x1, y1, x2, y2). 자르지 않습니다 — 손실은 원본에서."""
    cx, cy, w, h = cwh.unbind(1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)


def to_cwh(xyxy: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = xyxy.unbind(1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], 1)


def box_loss(pred_cwh: torch.Tensor, true_xyxy: torch.Tensor) -> torch.Tensor:
    """중심은 선형, 크기는 **로그**에서 잽니다.

    ⚠️ 크기를 선형으로 재면 큰 병변이 손실을 지배합니다. 우리가 정작 필요한
    건 **배율이 밴드 안에 드는 것**이고, 그건 **비율**의 문제입니다 —
    `usable_range()` 밴드가 0.71~1.67**배** 인 것과 같은 이유입니다.
    """
    t = to_cwh(true_xyxy)
    ctr = nn.functional.smooth_l1_loss(pred_cwh[:, :2], t[:, :2], beta=0.05)
    eps = 1e-6
    size = nn.functional.smooth_l1_loss(
        torch.log(pred_cwh[:, 2:].clamp_min(eps)),
        torch.log(t[:, 2:].clamp_min(eps)), beta=0.2)
    return ctr + size


def band_report(pred_xyxy, true_xyxy) -> dict:
    """제안 네모가 **촬영 밴드** 안에 드는 비율. 사람·창탐지기와 같은 잣대.

    밴드는 `config.ZOOM_ALLOW` / `ZOOM_CENTER_MAX` 에서 끌어 씁니다 — 여기
    베껴 적으면 출처가 바뀌어도 아무 일이 안 일어납니다 (촬영 밴드에서 당한 것).
    """
    import numpy as np

    from src.config import ZOOM_ALLOW, ZOOM_CENTER_MAX

    p = np.asarray(pred_xyxy, dtype=float)
    t = np.asarray(true_xyxy, dtype=float)
    p_long = np.maximum(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1])
    t_long = np.maximum(t[:, 2] - t[:, 0], t[:, 3] - t[:, 1])
    ok = t_long > 0
    ratio = np.where(ok, p_long / np.maximum(t_long, 1e-9), np.nan)
    off = np.maximum(
        np.abs((p[:, 0] + p[:, 2]) / 2 - (t[:, 0] + t[:, 2]) / 2),
        np.abs((p[:, 1] + p[:, 3]) / 2 - (t[:, 1] + t[:, 3]) / 2))
    # 네모를 r 배로 그리면 환산 줌은 1/r → 밴드를 뒤집습니다
    lo, hi = 1 / ZOOM_ALLOW[1], 1 / ZOOM_ALLOW[0]
    in_size = (ratio >= lo) & (ratio <= hi)
    in_pos = off <= ZOOM_CENTER_MAX
    return {"n": int(ok.sum()),
            "ratio_median": float(np.nanmedian(ratio)),
            "off_median": float(np.median(off)),
            "in_size": float(np.nanmean(in_size)),
            "in_pos": float(np.mean(in_pos)),
            "both": float(np.nanmean(in_size & in_pos)),
            "band": [lo, hi], "center_max": float(ZOOM_CENTER_MAX)}


def print_report(rep: dict, *, human_both: float = 0.05,
                 window_both: float = 0.067) -> bool:
    """판정을 찍고 통과 여부를 돌려줍니다. 기준은 `experiments` 에 있습니다."""
    from src.experiments import DETECT_MIN_USABLE

    print(f"\n■ 검출기 제안 네모 (n={rep['n']:,})")
    print(f"    크기비 중앙값 {rep['ratio_median']:.2f}배   "
          f"중심 어긋남 중앙값 {rep['off_median']:.3f}")
    print(f"\n■ 밴드 안 (허용 {rep['band'][0]:.2f}~{rep['band'][1]:.2f}배 · "
          f"중심 {rep['center_max']})")
    print(f"    배율     {rep['in_size']:.1%}")
    print(f"    위치     {rep['in_pos']:.1%}")
    print(f"    둘 다    {rep['both']:.1%}      "
          f"(사람 {human_both:.1%} · 창탐지기 {window_both:.1%})")
    ok = rep["both"] >= DETECT_MIN_USABLE
    print(f"\n■ 사전등록 문턱 {DETECT_MIN_USABLE:.0%}")
    print("  ⭕ **통과** — 다음은 이 네모로 자른 크롭의 **커버리지**입니다."
          if ok else "  ❌ 미달")
    if ok:
        print("  ⚠️ 밴드만 통과한 것입니다. **커버리지가 안 오르면 채택 안 합니다** "
              "(STEP 41 에서 배운 것).")
    return ok
