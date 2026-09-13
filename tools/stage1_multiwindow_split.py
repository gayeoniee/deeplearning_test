"""STEP 56 — 1단계 다중 창(3×3 평균)을 **val 에서 문턱 재산정 · holdout 에서 확인** (추론만).

    uv run --extra train python tools/stage1_multiwindow_split.py --split val --n 5000
    uv run --extra train python tools/stage1_multiwindow_split.py --split holdout --n 5000 --thr reports/stage1_multiwindow_val.json

원본 사진(짧은 변 1080 = 학습 픽셀 공간)에서 라벨 중심(배포 holdout 조건)과 사용자 흉내 중심(±0.161)에 대해
    single  : 창 1개 (지금 배포 f320)
    mw_mean : 3×3 · 간격 160 · 평균          mw_top3 : 상위 3 평균
을 같은 사진에서 냅니다. val 은 target recall 0.95 로 문턱을 뽑고, holdout 은 그 문턱을 그대로 씁니다 (관문은 STEP56 문서).
⚠️ holdout 을 여는 도구입니다 — `tests/test_holdout_discipline.py` 목록에 이유와 함께 등록합니다.
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
WIN, STRIDE, N = 320, 160, 3


def largest_box(boxes_json):
    boxes = {tuple(float(v) for v in b) for b in json.loads(boxes_json)}
    boxes = [b for b in boxes if b[2] > b[0] and b[3] > b[1]]
    return max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1])) if boxes else None


def grid(cx, cy, W, H, n=N, stride=STRIDE):
    off = [(i - (n-1)/2) * stride for i in range(n)]
    return [(min(max(cx+dx, WIN/2), W-WIN/2), min(max(cy+dy, WIN/2), H-WIN/2)) for dy in off for dx in off]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["val", "holdout"], required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--release", type=Path, default=RELEASE)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--thr", type=Path, default=None, help="holdout: val 실행이 낸 json (문턱)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out_path = a.out or ROOT / "reports" / f"stage1_multiwindow_{a.split}.json"

    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image, ImageFile
    from sklearn.metrics import roc_auc_score

    from src import data, models, train
    from src.agent import to_train_space
    from src.config import CFG
    from user_bbox_sim import user_box_stats

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    _, center_off = user_box_stats()
    m = pd.read_parquet(a.manifest)
    m = m[m.is_holdout] if a.split == "holdout" else m[(~m.is_holdout) & (m.fold == 0)]
    smp = m.sample(min(a.n, len(m)), random_state=a.seed)
    print(f"■ {a.split} {len(m):,}행 중 {len(smp):,}장 · 정상 {(smp.label=='A7').mean():.1%}")

    ck = a.release / "checkpoints"
    s1 = next(d for d in sorted(ck.iterdir()) if d.name.startswith("stage1_"))
    thr_json = json.loads((a.release / "stage1_threshold.json").read_text())
    t_raw = float(thr_json.get("threshold_raw", thr_json.get("threshold", 0.5)))
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    tf = data.build_transforms(CFG(img_size=384), train=False)
    net1 = models.build(train.model_key_from_exp(s1.name) or "effnetv2_s", n_classes=2, pretrained=False, verbose=False)
    net1 = train._load_into(net1, s1 / "best.pt", verbose=False).to(dev).eval()
    print(f"■ 1단계 {s1.name} · 배포 raw 문턱 {t_raw:.4f} · {dev}")

    zips = {}

    def zf(p):
        if p not in zips:
            zips[p] = zipfile.ZipFile(p)
        return zips[p]

    rng = random.Random(a.seed)
    per = {}
    bx, bk = [], []
    truth = []
    t0 = time.perf_counter()

    def flush():
        if not bx:
            return
        x = torch.stack(bx).to(dev)
        with torch.no_grad():
            p = torch.softmax(net1(x), 1)[:, 1].float().cpu().numpy()
        for (i, c, j), v in zip(bk, p):
            per.setdefault(i, {}).setdefault(c, {})[j] = float(v)
        bx.clear(); bk.clear()

    n_ok = 0
    for i, r in enumerate(smp.itertuples()):
        try:
            blob = zf(r.zip_path).read(r.zip_member)
            with Image.open(io.BytesIO(blob)) as f:
                im = to_train_space(f.convert("RGB"))
        except Exception as e:  # noqa: BLE001
            print(f"   [skip] {r.zip_member}: {e}")
            continue
        W, H = im.size
        b = largest_box(r.boxes)
        if b is None:
            continue
        sc = min(W, H) / 1080.0 if False else 1.0
        lcx, lcy = (b[0]+b[2])/2*sc, (b[1]+b[3])/2*sc
        short = min(W, H)
        ucx = lcx + rng.uniform(-center_off, center_off)*short
        ucy = lcy + rng.uniform(-center_off, center_off)*short
        cents = {"label_single": grid(lcx, lcy, W, H, 1, 0), "label_mw": grid(lcx, lcy, W, H),
                 "user_single": grid(ucx, ucy, W, H, 1, 0), "user_mw": grid(ucx, ucy, W, H)}
        for c, cs in cents.items():
            for j, (cx, cy) in enumerate(cs):
                x0, y0 = int(cx-WIN/2), int(cy-WIN/2)
                bx.append(tf(im.crop((x0, y0, x0+WIN, y0+WIN)))); bk.append((n_ok, c, j))
        truth.append(r.label); n_ok += 1
        if len(bx) >= a.batch:
            flush()
        if (i+1) % 500 == 0:
            print(f"   {i+1:,}/{len(smp):,}  {(time.perf_counter()-t0)/60:.1f}분", flush=True)
    flush()
    y = np.asarray(truth); les = y != "A7"
    print(f"\n■ 표본 {len(y):,}장 (병변 {int(les.sum()):,}) · {(time.perf_counter()-t0)/60:.1f}분")
    S = {}
    for c in ("label_single", "label_mw", "user_single", "user_mw"):
        arr = np.array([[per[i][c][j] for j in sorted(per[i][c])] for i in range(len(y))])
        if c.endswith("_single"):
            S[c] = arr[:, 0]
        else:
            S[c + "_mean"], S[c + "_top3"] = arr.mean(1), np.sort(arr, 1)[:, -3:].mean(1)

    def thr_at_recall(v, target=0.95):
        pos = np.sort(v[les])[::-1]
        k = int(np.ceil(target * len(pos))) - 1
        return float(pos[max(k, 0)])

    def report(v, t):
        return {"auroc": float(roc_auc_score(les, v)), "threshold": t,
                "recall": float((v[les] >= t).mean()), "false_alarm": float((v[~les] >= t).mean())}

    if a.split == "val":
        thr = {k: thr_at_recall(v) for k, v in S.items() if k.startswith("label_")}
        thr["label_single_deployed_raw"] = t_raw
    else:
        thr = json.loads(a.thr.read_text())["thresholds"]
    res = {}
    for k, v in S.items():
        base = "label_single" if k == "label_single" else ("label_mw_mean" if k.endswith("_mean") else ("label_mw_top3" if k.endswith("_top3") else "label_single"))
        res[k] = report(v, thr[base])
        if k == "label_single":
            res["label_single@deployed_raw"] = report(v, t_raw)
    print(f"\n■ ★ STEP 56 · {a.split} · 문턱 {'val 에서 새로 (recall 0.95)' if a.split=='val' else 'val 에서 가져옴'}")
    print(f"    {'점수':26}{'AUROC':>8}{'문턱':>9}{'recall':>9}{'헛알림':>9}")
    for k, v in res.items():
        print(f"    {k:26}{v['auroc']:>8.4f}{v['threshold']:>9.4f}{v['recall']:>9.1%}{v['false_alarm']:>9.1%}")
    d_auroc = res["label_mw_mean"]["auroc"] - res["label_single"]["auroc"]
    d_sens = (res["label_single"]["recall"] - res["user_single"]["recall"]) - (res["label_mw_mean"]["recall"] - res["user_mw_mean"]["recall"])
    print(f"\n    라벨 중심 AUROC 차(mw_mean − single) {d_auroc:+.4f} · 위치 민감도 감소(single − mw_mean, recall 차이) {d_sens:+.1%}p")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"split": a.split, "n": int(len(y)), "n_lesion": int(les.sum()), "thresholds": thr, "results": res,
                                    "stage1": s1.name, "seed": a.seed}, ensure_ascii=False, indent=1))
    np.savez_compressed(out_path.with_suffix(".npz"), y=y, **S)
    print(f"원본: {out_path}")


if __name__ == "__main__":
    main()
