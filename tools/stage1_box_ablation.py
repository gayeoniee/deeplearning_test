"""틀(가이드 프레임)을 없애면 **1단계**가 얼마나 상하나 — 로컬 GPU (STEP 43).

    uv run --extra train python tools/stage1_box_ablation.py            # 3,000장
    uv run --extra train python tools/stage1_box_ablation.py --seed 99  # 재현

## 왜 이게 남았나

STEP 40 이 *"가이드 프레임은 **2단계** 결과를 하나도 안 바꾼다"* 를 다섯 번
확인했습니다 (사용자 네모 = 네모 없음, 소수점까지 같음). `m2.5` 창이 100%
포화라 2단계는 이미 사진 전체를 보고 있기 때문입니다.

그런데 **1단계 `f320` 은 중심을 씁니다.** 창은 320px 고정이지만 *어디를*
자를지는 네모에서 옵니다. 틀을 없애면 중심이 **화면 한가운데**로 갑니다.
그게 recall 을 얼마나 깎는지는 **한 번도 안 쟀습니다.**

1단계의 놓침은 *"괜찮아 보여요"* 로 나가 병원에 아예 안 가게 만듭니다.
이름이 틀리는 것보다 훨씬 나쁩니다 — 그래서 관문은 recall 하나입니다.

## 세 조건 (같은 사진 · 같은 순서)

    label   라벨 bbox 중심          상한 (지금 평가 방식)
    user    사용자 네모 중심        지금 배포 (라벨 중심 + 실측 이탈)
    none    화면 한가운데           틀 없음 (제안)

⚠️ `f320` 은 **크기를 안 씁니다.** 그래서 세 조건은 **중심만** 다릅니다 —
   이게 이 실험이 성립하는 이유입니다.

## 판정

`experiments.STAGE1_NOBOX_MAX_RECALL_DROP` — 돌리기 전에 박았습니다 (규칙 2).
AUROC·헛알림률은 **관문이 아니고 찍기만** 합니다.

## 자극 검사

시뮬레이션 사진에서 병변이 화면 중앙에서 얼마나 떨어져 있나. 0 에 가까우면
`none` 조건이 공짜로 이기므로 **멈춥니다**
(`experiments.STAGE1_NOBOX_MIN_CENTER_OFF`).

⚠️ 한계 — VL01 기준이고, 사용자 네모 분포는 n=40 · 한 사람 · 마우스입니다.
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


def auroc(y, s):
    """순위 기반 AUROC. 동점은 평균 순위."""
    import numpy as np

    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    su = s[order]
    i = 0
    while i < len(su):
        j = i
        while j + 1 < len(su) and su[j + 1] == su[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--batch", type=int, default=24)
    a = ap.parse_args()

    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image

    from src import crop, data, env, models, train
    from src.config import CFG, CLASSES_STAGE1, NORMAL_LABEL
    from src.experiments import (STAGE1_NOBOX_MAX_RECALL_DROP,
                                 STAGE1_NOBOX_MIN_CENTER_OFF, STAGE1_NOBOX_N)

    sys.path.insert(0, str(ROOT / "tools"))
    from user_bbox_sim import user_box_stats

    n = a.n or STAGE1_NOBOX_N
    (box_lo, box_hi), center_off = user_box_stats()

    # ── 데이터 ──────────────────────────────────────────────
    df = pd.read_parquet(env.work_root() / "manifests" / "manifest_final.parquet")
    df = df[df["zip_path"].notna() & df["zip_member"].notna() & df["bbox"].notna()]
    if "is_holdout" in df.columns:
        df = df[~df["is_holdout"].astype(bool)]      # holdout 은 안 엽니다
    if df.empty:
        raise SystemExit("[X] 원본 zip 경로가 있는 행이 없습니다.")
    df = df.sample(min(len(df), n), random_state=a.seed)
    n_no = int((df["label"] == NORMAL_LABEL).sum())
    print(f"■ {len(df):,}장 (VL01 val) · 정상 {n_no:,} / 이상 {len(df) - n_no:,}")

    # ── 모델 ────────────────────────────────────────────────
    thr_f = env.work_root() / "stage1_threshold.json"
    meta = json.loads(thr_f.read_text(encoding="utf-8"))
    tau, exp = float(meta["threshold"]), str(meta["stage1_exp"])
    ck = env.ensure_dirs()["checkpoints"] / exp
    if not (ck / "best.pt").exists():
        raise SystemExit(f"[X] {ck}/best.pt 가 없습니다.")
    T = 1.0
    tf_json = ck / "temperature.json"
    if tf_json.is_file():
        T = float(json.loads(tf_json.read_text(encoding="utf-8")).get("temperature", 1.0))
    print(f"■ {exp}")
    print(f"■ 임계값 {tau:.4f} · 온도 {T:.4f}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    key = train.model_key_from_exp(exp) or "effnetv2_s"
    net = models.build(key, n_classes=len(CLASSES_STAGE1), pretrained=False)
    net = train._load_into(net, ck / "best.pt", verbose=False)
    net = net.to(dev).eval()
    cfg = CFG(img_size=384)
    tf = data.build_transforms(cfg, train=False)
    print(f"■ 장치 {dev} · 백본 {key}")

    CONDS = ("label", "user", "none")
    rng = random.Random(a.seed)
    zips: dict[str, zipfile.ZipFile] = {}
    logits: dict[str, list] = {c: [] for c in CONDS}
    buf: dict[str, list] = {c: [] for c in CONDS}
    truth: list[int] = []
    stim_off: list[float] = []           # ★ 자극 검사
    t0 = time.perf_counter()

    def flush():
        for c, imgs in buf.items():
            if not imgs:
                continue
            with torch.no_grad():
                logits[c].append(net(torch.stack(imgs).to(dev)).float().cpu())
            imgs.clear()

    try:
        rows = list(df.iterrows())
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
            bb = r["bbox"]
            b0 = crop._box4(json.loads(bb) if isinstance(bb, str) else bb)
            if b0 is None:
                full.close()
                continue
            long0 = max(b0[2] - b0[0], b0[3] - b0[1])
            if long0 <= 0:
                full.close()
                continue
            # 앱처럼 가까이 · 위치는 무작위 (user_bbox_sim 과 같은 자극)
            side0 = min(max(long0 * rng.uniform(3.5, 6.0), 480.0), FW, FH)
            if side0 < long0 * 1.6:
                full.close()
                continue
            lo_x, hi_x = max(0.0, b0[2] - side0), min(b0[0], FW - side0)
            lo_y, hi_y = max(0.0, b0[3] - side0), min(b0[1], FH - side0)
            if lo_x > hi_x or lo_y > hi_y:
                full.close()
                continue
            wx, wy = rng.uniform(lo_x, hi_x), rng.uniform(lo_y, hi_y)
            im = full.crop((int(wx), int(wy), int(wx + side0), int(wy + side0)))
            full.close()                       # ⚠️ 안 닫으면 메모리가 터집니다
            if im.size[0] != 1080:
                im2 = im.resize((1080, 1080))
                im.close()
                im = im2
            W, H = im.size
            s = W / side0
            b = [(b0[0] - wx) * s, (b0[1] - wy) * s,
                 (b0[2] - wx) * s, (b0[3] - wy) * s]

            # ★ 자극 검사 — 병변 중심이 화면 중앙에서 얼마나 떨어졌나
            stim_off.append(max(abs((b[0] + b[2]) / 2 - W / 2),
                                abs((b[1] + b[3]) / 2 - H / 2)) / W)

            side = rng.uniform(box_lo, box_hi) * W
            cx = (b[0] + b[2]) / 2 + rng.uniform(-center_off, center_off) * W
            cy = (b[1] + b[3]) / 2 + rng.uniform(-center_off, center_off) * W
            ub = [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2]
            # 틀 없음 = 화면 한가운데. 크기는 앱 기본값 44% 지만 f320 은 안 씁니다.
            nb = [W * 0.28, H * 0.28, W * 0.72, H * 0.72]

            ok, tiles = True, {}
            for cond, box in (("label", b), ("user", ub), ("none", nb)):
                win = crop.crop_window({"bbox": list(box), "img_w": W, "img_h": H},
                                       tag="f320", cfg=cfg)
                if win is None:
                    ok = False
                    break
                tiles[cond] = tf(im.crop(tuple(int(v) for v in win)))
            im.close()
            if not ok:
                stim_off.pop()
                continue
            for cond in CONDS:
                buf[cond].append(tiles[cond])
            truth.append(0 if str(r["label"]) == NORMAL_LABEL else 1)

            if len(buf["label"]) >= a.batch:
                flush()
            if (i + 1) % 500 == 0:
                print(f"   {i+1:,}/{len(rows):,}  {(time.perf_counter()-t0)/60:.1f}분",
                      flush=True)
        flush()
    finally:
        for z in zips.values():
            z.close()

    y = np.asarray(truth)
    off_med = float(np.median(stim_off))
    print(f"\n■ 모은 표본 {len(y):,}장 · {(time.perf_counter()-t0)/60:.1f}분")

    # ── ★ 자극 검사부터 (표본 수보다 먼저) ─────────────────
    print(f"\n■ 자극 검사 — 병변 중심의 화면중앙 이탈 중앙값 {off_med:.3f}")
    if off_med < STAGE1_NOBOX_MIN_CENTER_OFF:
        raise SystemExit(
            f"  [X] {STAGE1_NOBOX_MIN_CENTER_OFF} 미만입니다 — 병변이 원래 가운데라\n"
            "      '틀 없음' 조건이 공짜로 이깁니다. 이 측정은 성립하지 않습니다.")
    print(f"  OK  ({STAGE1_NOBOX_MIN_CENTER_OFF} 이상)")

    ab = int(y.sum())
    i_ab = CLASSES_STAGE1.index("ABNORMAL")
    res = {}
    for cond in CONDS:
        lg = torch.cat(logits[cond]).numpy()[:len(y)] / T
        e = np.exp(lg - lg.max(1, keepdims=True))
        p = (e / e.sum(1, keepdims=True))[:, i_ab]
        passed = p >= tau
        res[cond] = {
            "auroc": auroc(y, p),
            "recall": float(passed[y == 1].mean()),
            "false_alarm_rate": float(passed[y == 0].mean()),
            "fa_share_of_passed": float((passed & (y == 0)).sum() / max(passed.sum(), 1)),
        }

    print(f"\n■ 1단계 — 세 조건 (같은 사진 {len(y):,}장 · 이상 {ab:,} · 임계값 {tau:.4f})\n")
    print("    조건               AUROC     recall    헛알림률   넘긴것중 헛알림")
    ko = {"label": "라벨 중심(상한)", "user": "사용자 네모 중심", "none": "화면 한가운데"}
    for c in CONDS:
        v = res[c]
        print(f"    {ko[c]:<16} {v['auroc']:.4f}    {v['recall']:.4f}    "
              f"{v['false_alarm_rate']:.4f}     {v['fa_share_of_passed']:.4f}")

    drop = res["user"]["recall"] - res["none"]["recall"]
    print(f"\n■ 판정 — 사용자 네모 → 틀 없음, recall {drop:+.4f}")
    print(f"    사전등록 허용 하락 {STAGE1_NOBOX_MAX_RECALL_DROP}")
    ok_gate = drop <= STAGE1_NOBOX_MAX_RECALL_DROP
    print("  ⭕ **통과** — 1단계는 틀 없이도 견딥니다." if ok_gate
          else "  ❌ 미달 — 틀을 없애면 1단계가 병변을 더 놓칩니다.")
    print("\n■ 찍기만 (관문 아님)")
    print(f"    AUROC    사용자 {res['user']['auroc']:.4f} → 틀없음 "
          f"{res['none']['auroc']:.4f}  ({res['none']['auroc']-res['user']['auroc']:+.4f})")
    print(f"    헛알림률  사용자 {res['user']['false_alarm_rate']:.4f} → 틀없음 "
          f"{res['none']['false_alarm_rate']:.4f}  "
          f"({res['none']['false_alarm_rate']-res['user']['false_alarm_rate']:+.4f})")
    print("\n⚠️ VL01 val 기준 · 사용자 네모 분포는 n=40·한 사람 · holdout 안 엶.")

    out = ROOT / "reports" / f"stage1_box_ablation_seed{a.seed}.json"
    out.write_text(json.dumps(
        {"n": int(len(y)), "n_abnormal": ab, "seed": a.seed, "tau": tau, "T": T,
         "exp": exp, "stimulus_center_off_median": off_med,
         "by_cond": res, "recall_drop_user_to_none": drop, "passed": bool(ok_gate),
         "gate": STAGE1_NOBOX_MAX_RECALL_DROP,
         "box_frac": [box_lo, box_hi], "center_off": center_off},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"원본: {out}")


if __name__ == "__main__":
    main()
