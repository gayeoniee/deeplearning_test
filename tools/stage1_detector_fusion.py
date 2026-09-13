"""STEP 58 — 1단계 점수 = 창 하나 점수와 검출기(정상 학습, STEP 52) 점수의 **평균** · 원본 사진 val/holdout (추론만).

    uv run --extra train --with 'transformers>=4.52' python tools/stage1_detector_fusion.py --split val --n 5000
    uv run --extra train --with 'transformers>=4.52' python tools/stage1_detector_fusion.py --split holdout --n 5000 --thr reports/stage1_detector_fusion_val.json

`tools/stage1_multiwindow_split.py` 와 **같은 표본·같은 난수 순서**(seed 7)로 사진을 열어, 사용자 흉내 중심(±0.161)과 라벨 중심 각각에 대해
그 중심 주변 **1080 정사각**(짧은 변 전체 = 앱이 네모 중심으로 만들 수 있는 창)을 검출기에 넣고 최고 쿼리 점수를 냅니다.
창 하나 점수는 이미 저장된 `reports/stage1_multiwindow_{split}.npz` 에서 읽습니다 (같은 행 순서인지 y 로 대조).

    single        창 하나 (지금 배포)
    det           검출기 점수만
    fuse_mean     (창 + 검출기) / 2          ★ 판정 대상
    fuse_rank     순위 평균                   참고
관문은 STEP58 문서. 문턱은 val 라벨 중심 recall 0.95.
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
DETECTOR = Path("/Users/gayeon/deeplearning_test/data/work/dfine_step52/dfine_best")


def largest_box(boxes_json):
    boxes = {tuple(float(v) for v in b) for b in json.loads(boxes_json)}
    boxes = [b for b in boxes if b[2] > b[0] and b[3] > b[1]]
    return max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1])) if boxes else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["val", "holdout"], required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--detector", type=Path, default=DETECTOR)
    ap.add_argument("--single", type=Path, default=None, help="창 하나 점수 npz (기본 reports/stage1_multiwindow_<split>.npz)")
    ap.add_argument("--thr", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out_path = a.out or ROOT / "reports" / f"stage1_detector_fusion_{a.split}.json"
    single_path = a.single or ROOT / "reports" / f"stage1_multiwindow_{a.split}.npz"

    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image, ImageFile
    from scipy.stats import rankdata
    from sklearn.metrics import roc_auc_score
    from transformers import AutoImageProcessor, DFineForObjectDetection

    from src.agent import to_train_space
    from user_bbox_sim import user_box_stats

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    _, center_off = user_box_stats()
    m = pd.read_parquet(a.manifest)
    m = m[m.is_holdout] if a.split == "holdout" else m[(~m.is_holdout) & (m.fold == 0)]
    smp = m.sample(min(a.n, len(m)), random_state=a.seed)
    print(f"■ {a.split} {len(smp):,}장 (multiwindow_split 과 같은 표본·난수 순서)")

    zs = np.load(single_path)
    y_single = zs["y"]; S_user, S_label = zs["user_single"], zs["label_single"]
    proc = AutoImageProcessor.from_pretrained(a.detector)
    det = DFineForObjectDetection.from_pretrained(a.detector).to("cpu").eval()
    print(f"■ 검출기 {a.detector.name} (CPU) · 창 하나 점수 {single_path.name}")

    def score(im):
        with torch.no_grad():
            out = det(**proc(images=im, return_tensors="pt"))
        return float(out.logits.float().sigmoid().amax())

    zips = {}

    def zf(p):
        if p not in zips:
            zips[p] = zipfile.ZipFile(p)
        return zips[p]

    def square1080(im, cx, cy):
        W, H = im.size; s = min(W, H)
        x0 = int(min(max(cx - s/2, 0), W - s)); y0 = int(min(max(cy - s/2, 0), H - s))
        return im.crop((x0, y0, x0 + s, y0 + s))

    rng = random.Random(a.seed)
    d_user, d_label, truth = [], [], []
    t0 = time.perf_counter()
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
        lcx, lcy = (b[0]+b[2])/2, (b[1]+b[3])/2
        short = min(W, H)
        ucx = lcx + rng.uniform(-center_off, center_off)*short
        ucy = lcy + rng.uniform(-center_off, center_off)*short
        d_user.append(score(square1080(im, ucx, ucy)))
        d_label.append(score(square1080(im, lcx, lcy)))
        truth.append(r.label)
        if (i+1) % 250 == 0:
            print(f"   {i+1:,}/{len(smp):,}  {(time.perf_counter()-t0)/60:.1f}분", flush=True)
    y = np.asarray(truth); les = y != "A7"
    assert len(y) == len(y_single) and (y == y_single).all(), "창 하나 npz 와 행이 안 맞습니다 — 같은 seed·표본이어야 합니다"
    D_user, D_label = np.asarray(d_user), np.asarray(d_label)
    print(f"\n■ 표본 {len(y):,}장 (병변 {int(les.sum()):,}) · {(time.perf_counter()-t0)/60:.1f}분")

    def rk(v):
        return rankdata(v) / len(v)
    S = {"label_single": S_label, "label_det": D_label, "label_fuse_mean": (S_label + D_label) / 2, "label_fuse_rank": (rk(S_label) + rk(D_label)) / 2,
         "user_single": S_user, "user_det": D_user, "user_fuse_mean": (S_user + D_user) / 2, "user_fuse_rank": (rk(S_user) + rk(D_user)) / 2}

    def thr_at_recall(v, target=0.95):
        pos = np.sort(v[les])[::-1]
        return float(pos[max(int(np.ceil(target * len(pos))) - 1, 0)])

    def report(v, t):
        return {"auroc": float(roc_auc_score(les, v)), "threshold": t,
                "recall": float((v[les] >= t).mean()), "false_alarm": float((v[~les] >= t).mean())}
    if a.split == "val":
        thr = {k: thr_at_recall(v) for k, v in S.items() if k.startswith("label_")}
    else:
        thr = json.loads(a.thr.read_text())["thresholds"]
    res = {k: report(v, thr["label_" + k.split("_", 1)[1]]) for k, v in S.items()}
    print(f"\n■ ★ STEP 58 · {a.split} · 창 하나 + 검출기 융합 (문턱 = val 라벨 중심 recall 0.95)")
    print(f"    {'점수':20}{'AUROC':>8}{'문턱':>9}{'recall':>9}{'헛알림':>9}")
    for k, v in res.items():
        if "auroc" in v:
            print(f"    {k:20}{v['auroc']:>8.4f}{v['threshold']:>9.4f}{v['recall']:>9.1%}{v['false_alarm']:>9.1%}")
    # STEP 59 관문 B: 사용자 흉내 recall 차이의 짝지은 부트스트랩 95% CI (같은 병변 사진을 다시 뽑음)
    rng_b = np.random.default_rng(0); idx = np.flatnonzero(les)
    t_s, t_f = thr["label_single"], thr["label_fuse_mean"]
    hit_s, hit_f = (S["user_single"][idx] >= t_s).astype(float), (S["user_fuse_mean"][idx] >= t_f).astype(float)
    diffs = [(hit_f[b] - hit_s[b]).mean() for b in (rng_b.integers(0, len(idx), len(idx)) for _ in range(1000))]
    ci = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))
    res["user_recall_gain_ci95"] = {"gain": float((hit_f - hit_s).mean()), "lo": ci[0], "hi": ci[1]}
    print(f"    사용자 recall 차(fuse_mean − single) {res['user_recall_gain_ci95']['gain']:+.1%}p · 95% CI [{ci[0]:+.1%}, {ci[1]:+.1%}]  (STEP 59 관문: ≥ +2%p 이고 하한 > 0)")
    for agg in ("fuse_mean", "fuse_rank"):
        dA = res[f"label_{agg}"]["auroc"] - res["label_single"]["auroc"]; dFA_l = res[f"label_{agg}"]["false_alarm"] - res["label_single"]["false_alarm"]
        dR = res[f"user_{agg}"]["recall"] - res["user_single"]["recall"]; dFA_u = res[f"user_{agg}"]["false_alarm"] - res["user_single"]["false_alarm"]
        dAu = res[f"user_{agg}"]["auroc"] - res["user_single"]["auroc"]
        print(f"    [{agg}] 관문 A: 라벨 AUROC {dA:+.4f} (≥ −0.01) · 라벨 헛알림 {dFA_l:+.1%}p (≤ +2) | 관문 B: 사용자 recall {dR:+.1%}p (≥ +5) · 헛알림 {dFA_u:+.1%}p (≤ +2) · AUROC {dAu:+.4f}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"split": a.split, "n": int(len(y)), "n_lesion": int(les.sum()), "thresholds": thr, "results": res,
                                    "detector": str(a.detector), "single_npz": str(single_path), "seed": a.seed}, ensure_ascii=False, indent=1))
    np.savez_compressed(out_path.with_suffix(".npz"), y=y, det_user=D_user, det_label=D_label)
    print(f"원본: {out_path}")


if __name__ == "__main__":
    main()
