"""사용자가 실제로 그리는 네모로 바꾸면 어느 크롭이 버티나 — 로컬 GPU 로 잽니다.

    uv run --extra train python tools/user_bbox_sim.py            # 3,000장 (약 3분)
    uv run --extra train python tools/user_bbox_sim.py --n 27000  # 전체 (약 25분)

무엇을 묻나 — `m2.5`(네모 **크기**로 배율이 정해짐) vs `f320`(**중심만** 씀)
비교는 지금까지 전부 **라벨 bbox** 로 했습니다 (STEP 22·23). 라벨 bbox 는
정답에 딱 맞는 네모라 크기 오차가 0 입니다.

그런데 실측(STEP 36, `tools/box_error.py`)하니 사람은 이렇게 그립니다:

    실제 병변 긴 변   중앙값 21.5%   범위  5.6% ~ 59.6%   (화면 대비)
    사람이 그린 것    중앙값 58.7%   범위 47.2% ~ 70.6%
    상관                             **−0.05**

**병변 크기와 무관하게 화면의 절반쯤**으로 그립니다. 그러면 크기를 쓰는
크롭만 그 오차를 그대로 받습니다. 그게 실제로 얼마나 해로운지를 잽니다.

판정 기준은 **돌리기 전에** `src/experiments.py` 에 박았습니다 (작업 규칙 2):
`USER_BBOX_DROP_GAP_MIN`.

⚠️ **한계 셋 — 결과에 반드시 같이 적으세요.**
  ① 원본 사진이 VL01 zip 에만 있어 **VL01 기준**입니다. 두 번 데인 청크입니다
  ② 사용자 네모 분포가 **n=40 · 한 사람 · 마우스**입니다
  ③ 이건 **크롭 비교**이지 "2단계를 바꾸자" 가 아닙니다 — STEP 23 의 기각 근거
     (라벨 bbox 기준 macro-F1 −0.0099)는 그대로 살아 있습니다
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── 사람이 그리는 네모 (STEP 36 실측, n=40) ────────────────────────
#   ⚠️ 여기 베껴 적지 않고 **측정 결과 파일에서 읽습니다** — 값을 복사한 곳끼리
#      비교하면 출처가 바뀌어도 아무 일이 안 일어납니다 (촬영 밴드에서 당한 것).
#      파일이 없을 때만 아래 값으로 물러섭니다.
USER_BOX_FRAC = (0.472, 0.706)     # 화면 가로 대비, 10~90분위
USER_CENTER_OFF = 0.10             # 중심 이탈 (화면 대비) — 실측 중앙값 근처


def user_box_stats() -> tuple[tuple[float, float], float]:
    """`box_error.json` 에서 사람 네모 분포를 읽습니다. 없으면 기본값."""
    f = ROOT / "reports" / "box_error.json"
    if not f.is_file():
        print(f"[!] {f.name} 이 없어 기본 분포를 씁니다 {USER_BOX_FRAC}")
        return USER_BOX_FRAC, USER_CENTER_OFF
    rows = [r for r in json.loads(f.read_text(encoding="utf-8")) if r.get("user")]
    if len(rows) < 10:
        print(f"[!] 표본이 {len(rows)}개뿐이라 기본 분포를 씁니다")
        return USER_BOX_FRAC, USER_CENTER_OFF
    import statistics as st
    longs = sorted(max(r["user"][2], r["user"][3]) for r in rows)
    offs = []
    for r in rows:
        u, t = r["user"], r["truth"]
        offs.append(max(abs((u[0] + u[2] / 2) - (t[0] + t[2]) / 2),
                        abs((u[1] + u[3] / 2) - (t[1] + t[3]) / 2)))
    lo = longs[max(0, int(len(longs) * 0.10))]
    hi = longs[min(len(longs) - 1, int(len(longs) * 0.90))]
    print(f"[측정] 사람 네모 긴 변 {lo:.1%}~{hi:.1%} · 중심 이탈 중앙값 "
          f"{st.median(offs):.3f}  (n={len(rows)})")
    return (lo, hi), float(st.median(offs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="장수 (기본: 사전등록 값)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=24)
    a = ap.parse_args()

    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image

    from src import crop, data, env, evaluate, models, train
    from src.config import CFG, CLASSES
    from src.experiments import USER_BBOX_DROP_GAP_MIN, USER_BBOX_SIM_N, user_bbox_report

    n = a.n or USER_BBOX_SIM_N
    (box_lo, box_hi), center_off = user_box_stats()

    mf = env.work_root() / "manifests" / "manifest_final.parquet"
    df = pd.read_parquet(mf)
    df = df[df.get("label_orig", df.get("label")) != "A7"]
    df = df[df["zip_path"].notna() & df["zip_member"].notna() & df["bbox"].notna()]
    if df.empty:
        raise SystemExit("[X] 원본 zip 경로가 있는 행이 없습니다.")
    df = df.sample(min(len(df), n), random_state=a.seed)
    print(f"■ {len(df):,}장 (VL01) · 클래스 {df['label'].nunique()}종")

    # ── 두 모델 (백본이 같아야 크롭만 갈립니다) ──────────────────
    ck = env.ensure_dirs()["checkpoints"]
    want = {"m2.5": "stage2_effnetv2_s_m2.5_384_moderate",
            "f320": "stage2_effnetv2_s_f320_384_moderate"}
    for tag, exp in want.items():
        if not (ck / exp / "best.pt").exists():
            raise SystemExit(f"[X] {exp}/best.pt 가 없습니다 — 두 크롭 모두 필요합니다.")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"■ 장치 {dev}")
    cfg = CFG(img_size=384)
    nets = {}
    for tag, exp in want.items():
        key = train.model_key_from_exp(exp) or "effnetv2_s"
        m = models.build(key, n_classes=len(CLASSES), pretrained=False)
        m = train._load_into(m, ck / exp / "best.pt", verbose=False)
        nets[tag] = m.to(dev).eval()
    tf = data.build_transforms(cfg, train=False)

    rng = random.Random(a.seed)
    zips: dict[str, zipfile.ZipFile] = {}
    # 두 조건 × 두 크롭 = 네 갈래. 같은 사진·같은 순서로 모읍니다.
    logits: dict[tuple[str, str], list] = {(c, t): [] for c in ("label", "user")
                                           for t in want}
    truth: list[int] = []
    t0 = time.perf_counter()
    try:
        rows = list(df.iterrows())
        buf: dict[tuple[str, str], list] = {k: [] for k in logits}

        def flush():
            for key, imgs in buf.items():
                if not imgs:
                    continue
                x = torch.stack(imgs).to(dev)
                with torch.no_grad():
                    logits[key].append(nets[key[1]](x).float().cpu())
                imgs.clear()

        for i, (_, r) in enumerate(rows):
            zp = str(r["zip_path"])
            if zp not in zips:
                zips[zp] = zipfile.ZipFile(zp)
            try:
                blob = zips[zp].read(str(r["zip_member"]))
            except KeyError:
                continue
            full = Image.open(io.BytesIO(blob)).convert("RGB")
            FW, FH = full.size
            b0 = crop._box4(json.loads(r["bbox"]) if isinstance(r["bbox"], str)
                            else r["bbox"])
            if b0 is None:
                continue

            # ★ **사용자가 찍은 사진**을 먼저 만듭니다 (`tools/box_error.py` 와 같은
            #   방식) — 보호자는 이상한 데를 알고 **가까이** 찍습니다. 측정한
            #   네모 비율이 이 사진 기준이므로 여기서 적용해야 맞습니다.
            #   ⚠️ 원본 1920px 기준으로 적용하면 오차를 2.4배 부풀립니다.
            long0 = max(b0[2] - b0[0], b0[3] - b0[1])
            if long0 <= 0:
                continue
            side0 = min(max(long0 * rng.uniform(3.5, 6.0), 480.0), FW, FH)
            if side0 < long0 * 1.6:
                continue
            lo_x, hi_x = max(0.0, b0[2] - side0), min(b0[0], FW - side0)
            lo_y, hi_y = max(0.0, b0[3] - side0), min(b0[1], FH - side0)
            if lo_x > hi_x or lo_y > hi_y:
                continue
            wx, wy = rng.uniform(lo_x, hi_x), rng.uniform(lo_y, hi_y)
            im = full.crop((int(wx), int(wy), int(wx + side0), int(wy + side0)))
            # `agent.to_train_space()` 와 같게 — 짧은 변 1080 으로 맞춥니다.
            if im.size[0] != 1080:
                im = im.resize((1080, 1080))
            W, H = im.size
            s = W / side0
            b = [(b0[0] - wx) * s, (b0[1] - wy) * s,
                 (b0[2] - wx) * s, (b0[3] - wy) * s]

            # ① 라벨 네모 ② 사용자 네모 — 크기는 병변과 **무관하게** 뽑습니다
            side = rng.uniform(box_lo, box_hi) * W
            cx = (b[0] + b[2]) / 2 + rng.uniform(-center_off, center_off) * W
            cy = (b[1] + b[3]) / 2 + rng.uniform(-center_off, center_off) * W
            ub = [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2]

            ok = True
            for cond, box in (("label", b), ("user", ub)):
                for tag in want:
                    row = {"bbox": list(box), "img_w": W, "img_h": H}
                    win = crop.crop_window(row, tag=tag, cfg=cfg)
                    if win is None:
                        ok = False
                        continue
                    buf[(cond, tag)].append(tf(im.crop(tuple(int(v) for v in win))))
            if not ok:
                for k in buf:
                    if buf[k]:
                        buf[k].pop()
                continue
            truth.append(CLASSES.index(str(r["label"])))

            if len(buf[("label", "m2.5")]) >= a.batch:
                flush()
            if (i + 1) % 500 == 0:
                el = time.perf_counter() - t0
                print(f"   {i+1:,}/{len(rows):,}  {el/60:.1f}분 경과", flush=True)
        flush()
    finally:
        for z in zips.values():
            z.close()

    y = np.asarray(truth)
    print(f"■ 모은 표본 {len(y):,}장 · {(time.perf_counter()-t0)/60:.1f}분")

    print("\n■ 라벨 네모 → 사용자 네모 (macro-F1)")
    res = {}
    for tag in want:
        f1 = {}
        for cond in ("label", "user"):
            p = torch.cat(logits[(cond, tag)]).argmax(1).numpy()[:len(y)]
            f1[cond] = evaluate.metrics(y, p, None, len(CLASSES))["macro_f1"]
        res[tag] = user_bbox_report(tag, f1["label"], f1["user"])

    gap = res["m2.5"]["drop"] - res["f320"]["drop"]
    print(f"\n■ 하락 차이 (m2.5 − f320) = {gap:+.4f}   문턱 {USER_BBOX_DROP_GAP_MIN}")
    if gap >= USER_BBOX_DROP_GAP_MIN:
        print("  ⭕ **크기 의존이 실사용에서 실제로 해롭습니다.**")
    else:
        print("  ❌ 문턱 미달 — 크기 의존의 해가 이 표본에서는 안 보입니다.")
    print("\n⚠️ VL01 기준 · 사용자 네모 분포는 n=40·한 사람 · 크롭 비교일 뿐입니다.")

    out = ROOT / "reports" / "user_bbox_sim.json"
    out.write_text(json.dumps({"n": int(len(y)), "gap": gap, "by_crop": res,
                               "box_frac": [box_lo, box_hi],
                               "center_off": center_off},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"원본: {out}")


def _macro_f1(y, p, k: int) -> float:
    import numpy as np
    fs = []
    for c in range(k):
        tp = int(((p == c) & (y == c)).sum())
        fp = int(((p == c) & (y != c)).sum())
        fn = int(((p != c) & (y == c)).sum())
        fs.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(fs))


if __name__ == "__main__":
    main()
