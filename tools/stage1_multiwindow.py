"""STEP 55 — 1단계에 320px 창을 **여러 개** 주면 네모 위치 민감도가 줄고 헛알림은 안 느나 (추론만, 재학습 0).

    uv run --extra train python tools/stage1_multiwindow.py --n 2500 --release <hf_release>

같은 자극(STEP 50~54 `detect_coverage.py` 와 같은 표본·같은 난수 순서)에서 1단계 점수만 여러 방식으로 냅니다:

    user            사용자 중심 창 하나 (지금 배포)                      ← 비교 기준
    label           라벨 중심 창 하나 (상한)
    center          화면 한가운데 창 하나 (하한선)
    user_g3s160     사용자 중심 3×3 창(간격 160px = 반 겹침) → max / mean
    user_g3s320     사용자 중심 3×3 창(간격 320px = 안 겹침) → max / mean
    grid5           사진 전체 5×5 창(간격 190px, 네모 안 씀) → max / mean   ★ 네모 의존이 아예 없음

관문(사전등록 STEP55): 현 1단계 recall 에 맞춘 문턱에서 **헛알림 −5%p 이상** · AUROC ≥ user. 결론은 같은 실행 안의 차이.
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
sys.path.insert(0, str(ROOT / "tools"))

MANIFEST = Path("/Users/gayeon/deeplearning_test/data/work/safe_crop_v1/manifest.parquet")
RELEASE = Path("/Users/gayeon/deeplearning_test/data/work/hf_release")
WIN = 320


def largest_box(boxes_json):
    boxes = {tuple(float(v) for v in b) for b in json.loads(boxes_json)}
    boxes = [b for b in boxes if b[2] > b[0] and b[3] > b[1]]
    return max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1])) if boxes else None


def grid_centers(cx, cy, n, stride, W):
    """(cx,cy) 중심 n×n 격자. 창이 사진 밖으로 못 나가게 중심을 [160, W-160] 로 죕니다."""
    off = [(i - (n-1)/2) * stride for i in range(n)]
    lo, hi = WIN/2, W - WIN/2
    return [(min(max(cx+dx, lo), hi), min(max(cy+dy, lo), hi)) for dy in off for dx in off]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--release", type=Path, default=RELEASE)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "stage1_multiwindow_step55.json")
    a = ap.parse_args()

    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image, ImageFile
    from sklearn.metrics import roc_auc_score

    from src import data, models, train
    from src.config import CFG
    from user_bbox_sim import user_box_stats

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    (box_lo, box_hi), center_off = user_box_stats()
    m = pd.read_parquet(a.manifest)
    m = m[(~m.is_holdout) & (m.fold == 0)]
    les = m[m.label != "A7"].sample(a.n, random_state=a.seed)
    nrm = m[m.label == "A7"].sample(int(a.n * 0.19 / 0.81), random_state=a.seed)
    print(f"■ fold 0 · 병변 {len(les):,} + 정상 {len(nrm):,} (STEP 50~54 와 같은 표본)")

    ck = a.release / "checkpoints"
    s1 = next(d for d in sorted(ck.iterdir()) if d.name.startswith("stage1_"))
    thr = json.loads((a.release / "stage1_threshold.json").read_text())
    t1 = float(thr.get("threshold_raw", thr.get("threshold", 0.5)))
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    tf = data.build_transforms(CFG(img_size=384), train=False)
    net1 = models.build(train.model_key_from_exp(s1.name) or "effnetv2_s", n_classes=2, pretrained=False, verbose=False)
    net1 = train._load_into(net1, s1 / "best.pt", verbose=False).to(dev).eval()
    print(f"■ 1단계 {s1.name} (raw 문턱 {t1:.4f}) · {dev}")

    zips = {}

    def zf(p):
        if p not in zips:
            zips[p] = zipfile.ZipFile(p)
        return zips[p]

    rng = random.Random(a.seed)
    rows = list(pd.concat([les, nrm]).sample(frac=1, random_state=a.seed).itertuples())

    def photo(r):   # detect_coverage.photo 와 같은 난수 순서 (같은 자극)
        blob = zf(r.zip_path).read(r.zip_member)
        with Image.open(io.BytesIO(blob)) as f:
            full = f.convert("RGB")
        FW, FH = full.size
        b0 = largest_box(r.boxes) if r.label != "A7" else None
        if b0 is None:
            side0 = min(FW, FH) * 0.6
            wx, wy = (FW - side0) / 2, (FH - side0) / 2
            b0 = [wx + side0 * .4, wy + side0 * .4, wx + side0 * .6, wy + side0 * .6]
        long0 = max(b0[2]-b0[0], b0[3]-b0[1])
        side0 = min(max(long0 * rng.uniform(3.5, 6.0), 480.0), FW, FH)
        lo_x, hi_x = max(0.0, b0[2]-side0), min(b0[0], FW-side0)
        lo_y, hi_y = max(0.0, b0[3]-side0), min(b0[1], FH-side0)
        if lo_x > hi_x or lo_y > hi_y:
            full.close()
            return None
        wx, wy = rng.uniform(lo_x, hi_x), rng.uniform(lo_y, hi_y)
        im = full.crop((int(wx), int(wy), int(wx+side0), int(wy+side0)))
        full.close()
        if im.size[0] != 1080:
            im = im.resize((1080, 1080))
        W = im.size[0]
        s = W / side0
        lb = [(b0[0]-wx)*s, (b0[1]-wy)*s, (b0[2]-wx)*s, (b0[3]-wy)*s]
        return im, lb, W

    # 조건 → 창 중심 목록 (검출기 없이 순수 1단계)
    SETS = {}
    scores = {}     # cond → list of per-image np arrays (창별 p)
    truth = []
    buf_x, buf_key = [], []
    per_img = {}
    t0 = time.perf_counter()

    def flush():
        if not buf_x:
            return
        x = torch.stack(buf_x).to(dev)
        with torch.no_grad():
            p = torch.softmax(net1(x), 1)[:, 1].float().cpu().numpy()
        for (idx, cond, j), v in zip(buf_key, p):
            per_img.setdefault(idx, {}).setdefault(cond, {})[j] = float(v)
        buf_x.clear(); buf_key.clear()

    n_ok = 0
    for i, r in enumerate(rows):
        got = photo(r)
        if got is None:
            continue
        im, lb, W = got
        ucx = (lb[0]+lb[2])/2 + rng.uniform(-center_off, center_off)*W
        ucy = (lb[1]+lb[3])/2 + rng.uniform(-center_off, center_off)*W
        rng.uniform(box_lo, box_hi)          # 같은 난수 순서 유지 (사용자 네모 크기 — 여기선 안 씀)
        lcx, lcy = (lb[0]+lb[2])/2, (lb[1]+lb[3])/2
        centers = {"user": grid_centers(ucx, ucy, 1, 0, W),
                   "label": grid_centers(lcx, lcy, 1, 0, W),
                   "center": grid_centers(W/2, W/2, 1, 0, W),
                   "user_g3s160": grid_centers(ucx, ucy, 3, 160, W),
                   "user_g3s320": grid_centers(ucx, ucy, 3, 320, W),
                   "label_g3s160": grid_centers(lcx, lcy, 3, 160, W),
                   "grid5": grid_centers(W/2, W/2, 5, 190, W)}
        for cond, cs in centers.items():
            for j, (cx, cy) in enumerate(cs):
                x0, y0 = int(cx - WIN/2), int(cy - WIN/2)
                buf_x.append(tf(im.crop((x0, y0, x0+WIN, y0+WIN)))); buf_key.append((n_ok, cond, j))
        truth.append(r.label); n_ok += 1
        if len(buf_x) >= a.batch:
            flush()
        if (i+1) % 400 == 0:
            print(f"   {i+1:,}/{len(rows):,}  {(time.perf_counter()-t0)/60:.1f}분", flush=True)
    flush()
    y = np.asarray(truth); lesion = y != "A7"
    print(f"\n■ 표본 {len(y):,}장 · {(time.perf_counter()-t0)/60:.1f}분")

    conds = ["user", "label", "center", "user_g3s160", "user_g3s320", "label_g3s160", "grid5"]
    S = {c: np.array([[per_img[i][c][j] for j in sorted(per_img[i][c])] for i in range(len(y))]) for c in conds}

    def metrics(v, ref_rec, ref_fa):
        order = np.argsort(-v); ys = lesion[order]
        tp, fp = np.cumsum(ys), np.cumsum(~ys); rec, fpr = tp/lesion.sum(), fp/(~lesion).sum()
        return {"auroc": float(roc_auc_score(lesion, v)),
                "recall_raw_thr": float((v[lesion] >= t1).mean()), "fa_raw_thr": float((v[~lesion] >= t1).mean()),
                "fa_at_user_recall": float(fpr[np.searchsorted(rec, ref_rec)]) if (rec >= ref_rec).any() else 1.0,
                "recall_at_user_fa": float(rec[np.searchsorted(fpr, ref_fa, side="right")-1]) if (fpr <= ref_fa).any() else 0.0}

    u = S["user"][:, 0]
    ref_rec, ref_fa = float((u[lesion] >= t1).mean()), float((u[~lesion] >= t1).mean())
    out = {}
    for c in conds:
        if S[c].shape[1] == 1:
            out[c] = metrics(S[c][:, 0], ref_rec, ref_fa)
        else:
            out[c + "_max"] = metrics(S[c].max(1), ref_rec, ref_fa)
            out[c + "_mean"] = metrics(S[c].mean(1), ref_rec, ref_fa)
            out[c + "_top3mean"] = metrics(np.sort(S[c], 1)[:, -3:].mean(1), ref_rec, ref_fa)
    print(f"\n■ ★ STEP 55 — 1단계 다중 창 (같은 표본 · 재학습 0). 기준 user: recall {ref_rec:.1%} / 헛알림 {ref_fa:.1%} (raw 문턱)")
    print(f"    {'점수':22}{'AUROC':>8}{'헛알림@user recall':>20}{'recall@user 헛알림':>20}{'raw문턱 recall':>16}{'raw문턱 헛알림':>16}")
    for k, v in out.items():
        print(f"    {k:22}{v['auroc']:>8.4f}{v['fa_at_user_recall']:>20.1%}{v['recall_at_user_fa']:>20.1%}{v['recall_raw_thr']:>16.1%}{v['fa_raw_thr']:>16.1%}")
    base_fa = out["user"]["fa_at_user_recall"]
    print(f"\n    관문: 헛알림@user recall ≤ {base_fa-0.05:.1%} 이고 AUROC ≥ {out['user']['auroc']:.4f}")
    for k, v in out.items():
        if k in ("user", "label", "center"):
            continue
        d = v["fa_at_user_recall"] - base_fa; da = v["auroc"] - out["user"]["auroc"]
        verdict = "후보" if (d <= -0.05 and da >= 0) else ("구분 불가" if (d < 0 and da >= 0) else "기각")
        print(f"    {k:22} 헛알림 {d:+.1%}p · AUROC {da:+.4f} → {verdict}")
    # 위치 민감도: 라벨 중심 대비 사용자 중심에서 잃는 recall (한 창 vs 3×3)
    sens1 = out["label"]["recall_raw_thr"] - out["user"]["recall_raw_thr"]
    sens3 = out["label_g3s160_max"]["recall_raw_thr"] - out["user_g3s160_max"]["recall_raw_thr"]
    print(f"\n    위치 민감도 (라벨 중심 − 사용자 중심 recall, raw 문턱): 창 하나 {sens1:+.1%}p → 3×3 max {sens3:+.1%}p")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"n": int(len(y)), "n_lesion": int(lesion.sum()), "threshold_raw": t1, "ref_recall": ref_rec, "ref_fa": ref_fa,
                                 "results": out, "position_sensitivity": {"single": sens1, "g3s160_max": sens3},
                                 "stage1": s1.name, "seed": a.seed}, ensure_ascii=False, indent=1))
    np.savez_compressed(a.out.with_suffix(".npz"), y=y, **{c: S[c] for c in conds})
    print(f"원본: {a.out}")


if __name__ == "__main__":
    main()
