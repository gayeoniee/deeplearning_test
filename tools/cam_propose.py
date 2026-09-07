"""모델이 네모를 **제안**할 수 있나 — 사람 없이, 로컬에서 잽니다.

    uv run --extra train python tools/cam_propose.py            # 800장 (약 10분)
    uv run --extra train python tools/cam_propose.py --n 2000

왜 이걸 재나 (STEP 36 → 37) — 사람에게 병변의 기하를 물어보는 두 길이 **둘 다
닫혔습니다**:

    네모 크기   병변 크기와 **무관**하게 그림 (상관 −0.05, 정답의 2.80배)
    점 위치     탭이 병변 **안**에 든 것이 47.5% 뿐

번진 병변(비듬·각질 · 태선화)에는 사람이 답할 수 있는 점도 크기도 **없습니다.**

남은 길은 **모델이 제안하고 사람은 판단만** 하는 것입니다. 그런데 제안이 자주
틀리면 *"아니요"* 가 반복돼 아무도 안 씁니다. **짓기 전에** 그 비율을 잽니다.

## 어떻게 제안하나

1단계 모델은 `f320`(320px 고정 창)으로 학습됐습니다. 그래서 **분류기를 그대로
창 탐지기로** 씁니다 — 사진을 320px 창으로 훑어 창마다 '이상' 확률을 재고,
뜨거운 칸들의 외접사각형을 제안으로 냅니다.

⚠️ **Grad-CAM 이 아닙니다.** Grad-CAM 은 *분류 근거*라 "배경을 안 본다"(lift
4.55, STEP 16)는 말은 되지만 네모를 뽑는 도구가 아닙니다. 여기서는 모델이
**학습 때 본 것과 같은 입력**(320px 창)만 먹입니다 — 분포 밖으로 안 나갑니다.

판정 기준은 **돌리기 전에** 박았습니다: `experiments.CAM_BOX_MIN_USABLE`.

⚠️ 한계 — VL01 기준이고, 자극은 `box_error.py` 와 같은 '앱처럼 가까이 찍은'
창입니다. 실제 앱 사진은 또 다를 수 있습니다.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

OUT = ROOT / "reports" / "cam_propose.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--stride", type=int, default=6, help="한 변을 몇 칸으로 나누나")
    ap.add_argument("--batch", type=int, default=48)
    a = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image

    from src import crop, data, env, models, stages, train
    from src.config import CFG, ZOOM_ALLOW, ZOOM_CENTER_MAX
    from src.experiments import CAM_BOX_MIN_USABLE

    import box_error  # 자극을 **같은 함수**로 (두 곳에 안 적습니다)

    items = box_error.pick(a.n, a.seed)
    if not items:
        raise SystemExit("[X] 자극을 못 만들었습니다.")

    ck = env.ensure_dirs()["checkpoints"]
    s1 = next((d for d in sorted(ck.iterdir())
               if d.name.startswith("stage1_") and (d / "best.pt").exists()), None)
    if s1 is None:
        raise SystemExit("[X] stage1_*/best.pt 가 없습니다.")
    print(f"■ 1단계: {s1.name}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = CFG(img_size=384)
    key = train.model_key_from_exp(s1.name) or "effnetv2_s"
    net = models.build(key, n_classes=2, pretrained=False, verbose=False)
    net = train._load_into(net, s1 / "best.pt", verbose=False).to(dev).eval()
    tf = data.build_transforms(cfg, train=False)
    ab = stages.ABNORMAL_LABEL
    classes = ["A7", ab] if ab != "A7" else ["normal", "abnormal"]
    print(f"■ 장치 {dev} · 창 {a.stride}×{a.stride}")

    K = a.stride
    rows = []
    t0 = time.perf_counter()
    for i, it in enumerate(items):
        im = Image.open(it["path"]).convert("RGB")
        W, H = im.size
        # ── 320px 창으로 훑기 (학습이 본 것과 같은 크기) ──────────
        # ★ **320px 절대값** — 1단계가 학습 때 본 창과 같은 크기입니다.
        #   비율로 환산하면 분포 밖으로 나갑니다 (자극이 이미 원본 해상도).
        side = min(320.0, W * 0.9, H * 0.9)
        xs = np.linspace(0, W - side, K)
        ys = np.linspace(0, H - side, K)
        tiles, boxes = [], []
        for y in ys:
            for x in xs:
                boxes.append((x, y, x + side, y + side))
                tiles.append(tf(im.crop((int(x), int(y), int(x + side), int(y + side)))))
        with torch.no_grad():
            sc = []
            for b in range(0, len(tiles), a.batch):
                x = torch.stack(tiles[b:b + a.batch]).to(dev)
                sc.append(torch.softmax(net(x), 1)[:, 1].float().cpu())
            scores = torch.cat(sc).numpy().reshape(K, K)

        # ── 뜨거운 칸의 외접사각형을 **제안**으로 ────────────────
        # ★ **위치만** 냅니다. 320px 고정 창으로는 그보다 작은 네모를 만들 수
        #   없어 크기는 원리상 못 냅니다 — 그런데 `f320` 은 중심만 씁니다.
        #   점수 가중 무게중심을 제안 위치로 씁니다 (argmax 는 칸 격자에 묶임).
        lo, hi = float(scores.min()), float(scores.max())
        w = np.clip(scores - (hi - 0.5 * (hi - lo)), 0, None) if hi > lo else scores
        if w.sum() <= 0:
            w = scores - scores.min() + 1e-9
        cy_i = float((w.sum(1) * np.arange(K)).sum() / w.sum())
        cx_i = float((w.sum(0) * np.arange(K)).sum() / w.sum())
        pcx = (xs[0] + (xs[-1] - xs[0]) * cx_i / max(K - 1, 1) + side / 2) / W
        pcy = (ys[0] + (ys[-1] - ys[0]) * cy_i / max(K - 1, 1) + side / 2) / H
        prop = [pcx - side / (2 * W), pcy - side / (2 * H),
                pcx + side / (2 * W), pcy + side / (2 * H)]

        t = it["truth"]
        t_long = max(t[2] - t[0], t[3] - t[1])
        p_long = max(prop[2] - prop[0], prop[3] - prop[1])
        if t_long <= 0 or p_long <= 0:
            continue
        ratio = p_long / t_long
        off = max(abs((prop[0] + prop[2]) / 2 - (t[0] + t[2]) / 2),
                  abs((prop[1] + prop[3]) / 2 - (t[1] + t[3]) / 2))
        rows.append({"ratio": ratio, "off": off, "prop": prop, "truth": t,
                     "hot": float(hi)})
        if (i + 1) % 100 == 0:
            print(f"   {i+1}/{len(items)}  {(time.perf_counter()-t0)/60:.1f}분", flush=True)

    if not rows:
        raise SystemExit("[X] 잰 표본이 없습니다.")
    R = np.array([r["ratio"] for r in rows])
    O = np.array([r["off"] for r in rows])
    # 네모를 r 배로 그리면 환산 줌은 1/r → 밴드를 뒤집습니다 (box_error 와 같은 규칙)
    lo_b, hi_b = 1 / ZOOM_ALLOW[1], 1 / ZOOM_ALLOW[0]
    in_size = (R >= lo_b) & (R <= hi_b)
    in_pos = O <= ZOOM_CENTER_MAX
    both = float((in_size & in_pos).mean())

    print(f"\n■ 표본 {len(rows):,}장 · {(time.perf_counter()-t0)/60:.1f}분")
    print("\n■ 모델 제안 네모")
    print(f"    크기비 (제안÷정답)  중앙값 {np.median(R):.2f}배   "
          f"10~90% {np.percentile(R,10):.2f} ~ {np.percentile(R,90):.2f}")
    print(f"    중심 어긋남         중앙값 {np.median(O):.3f}   "
          f"10~90% {np.percentile(O,10):.3f} ~ {np.percentile(O,90):.3f}")
    print(f"\n■ 밴드 안 (허용 {lo_b:.2f}~{hi_b:.2f}배 · 중심 {ZOOM_CENTER_MAX})")
    print(f"    배율     {in_size.mean():.1%}      (사람 10.0%)")
    print(f"    위치     {in_pos.mean():.1%}      (사람 12.5%)")
    print(f"    둘 다    {both:.1%}      (사람 **5.0%**)")
    print(f"\n■ 사전등록 문턱 {CAM_BOX_MIN_USABLE:.0%}  (돌리기 전에 박았습니다)")
    if both >= CAM_BOX_MIN_USABLE:
        print("  ⭕ **통과** — '모델이 제안하고 사람은 판단' 설계가 설 수 있습니다.")
    else:
        print("  ❌ **미달** — 제안이 절반 넘게 틀립니다. 그건 제안이 아니라 방해입니다.")
    print("\n⚠️ VL01 기준 · 자극은 '앱처럼 가까이 찍은' 창입니다.")

    OUT.write_text(json.dumps(
        {"n": len(rows), "both": both, "in_size": float(in_size.mean()),
         "in_pos": float(in_pos.mean()), "ratio_median": float(np.median(R)),
         "off_median": float(np.median(O)), "human_both": 0.05},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"원본: {OUT}")


if __name__ == "__main__":
    main()
