"""검출기 네모로 자르면 **계열 커버리지**가 얼마나 돌아오나 — 끝단(end-to-end) 판정 (STEP 50).

    uv run --extra train python tools/detect_coverage.py --detector data/work/detect_step50/detect_best.pt

`tools/user_bbox_coverage.py`(STEP 39) 와 같은 파이프라인·같은 자극에 조건을 더했습니다:

    label         라벨 네모                       상한
    user          사용자 네모 시뮬레이션 (STEP 36)   비교 기준  ← 관문의 분모
    center        화면 한가운데 · 크기 0.215 고정     하한선 (아무것도 안 배운 네모)
    fixed         사용자 중심 + 크기 0.215           STEP 41
    detect        검출기 네모 그대로                 ★ 판정 대상
    detect_fixed  검출기 중심 + 크기 0.215           부지표

관문: `detect − user ≥ experiments.DETECT_MIN_COVERAGE_GAIN` (+0.05). 사전등록 STEP50.
부지표: 밴드 `둘 다` · 중심 오차 · **1단계 recall**(검출기 중심을 f320 에 줬을 때).

⚠️ 릴리스 팔이 로컬에 있는 것만 씁니다 (1단계 + 2단계 convnextv2_base). 3팔 값(STEP 39)과
절대값을 비교하지 마세요 — 이 도구의 결론은 **같은 실행 안의 상대 차이**입니다.
⚠️ holdout 은 안 엽니다 — 표본은 비 holdout **fold 0** (검출기가 학습에 안 쓴 fold).
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

from src.config import MORPH_GROUP_KEEP_A6 as _M4  # noqa: E402
G4 = list(dict.fromkeys(_M4[c] for c in ("A1", "A2", "A5", "A6")))
CONDS = ("label", "user", "center", "fixed", "detect", "detect_fixed")
RELEASE = Path("/Users/gayeon/deeplearning_test/data/work/release_reference/dogskin_06/release")
MANIFEST = Path("/Users/gayeon/deeplearning_test/data/work/safe_crop_v1/manifest.parquet")


def largest_box(boxes_json):
    boxes = {tuple(float(v) for v in b) for b in json.loads(boxes_json)}
    boxes = [b for b in boxes if b[2] > b[0] and b[3] > b[1]]
    return max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1])) if boxes else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detector", type=Path, required=True, help="코랩이 낸 detect_best.pt")
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--release", type=Path, default=RELEASE)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "detect_coverage.json")
    a = ap.parse_args()

    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image, ImageFile
    from torchvision import transforms as T

    from src import crop, data, models, train
    from src.agent import crop_tag_from_exp
    from src.config import CFG, CLASSES, MORPH_GROUP_KEEP_A6
    from src.detect import BoxHead, band_report, to_xyxy
    from src.experiments import (DETECT_MIN_COVERAGE_GAIN, FIXEDSCALE_LESION_FRAC,
                                 NAMING_TARGET_ERROR)
    from user_bbox_sim import user_box_stats

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    (box_lo, box_hi), center_off = user_box_stats()
    m = pd.read_parquet(a.manifest)
    m = m[(~m.is_holdout) & (m.fold == 0)]
    les = m[m.label != "A7"].sample(a.n, random_state=a.seed)
    nrm = m[m.label == "A7"].sample(int(a.n * 0.19 / 0.81), random_state=a.seed)
    print(f"■ fold 0 · 병변 {len(les):,} + 정상 {len(nrm):,} · 청크 {les.source_chunk.value_counts().to_dict()}")

    ck = a.release / "checkpoints"
    s1 = next(d for d in sorted(ck.iterdir()) if d.name.startswith("stage1_"))
    s2s = [d for d in sorted(ck.iterdir()) if d.name.startswith("stage2_")]
    thr = json.loads((a.release / "stage1_threshold.json").read_text())
    t1 = float(thr.get("threshold_raw", thr.get("threshold", 0.5)))
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = CFG(img_size=384)
    tf = data.build_transforms(cfg, train=False)
    tf_det = T.Compose([T.Resize((384, 384)), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def load(d, n_cls):
        net = models.build(train.model_key_from_exp(d.name) or "effnetv2_s", n_classes=n_cls, pretrained=False, verbose=False)
        return train._load_into(net, d / "best.pt", verbose=False).to(dev).eval()

    net1 = load(s1, 2)
    net2 = [(load(d, len(CLASSES)), crop_tag_from_exp(d.name) or "m2.5") for d in s2s]
    det = BoxHead(pretrained=False, img_size=384)
    det.load_state_dict(torch.load(a.detector, map_location="cpu", weights_only=False)["model"])
    det = det.to(dev).eval()
    print(f"■ 1단계 {s1.name} (raw 문턱 {t1:.4f}) · 2단계 {len(net2)}팔 · 검출기 {a.detector}")

    zips = {}

    def zf(p):
        if p not in zips:
            zips[p] = zipfile.ZipFile(p)
        return zips[p]

    rng = random.Random(a.seed)
    rows = list(pd.concat([les, nrm]).sample(frac=1, random_state=a.seed).itertuples())

    def photo(r):
        """앱처럼 가까이 찍은 1080 정사각 사진 + 라벨 네모 (STEP 39 와 같은 자극)."""
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

    def square(cx, cy, side):
        return [cx-side/2, cy-side/2, cx+side/2, cy+side/2]

    res = {c: {"p1": [], "p2": []} for c in CONDS}
    buf = {(c, k): [] for c in CONDS for k in range(1+len(net2))}
    truth, det_pred, det_true, det_pending = [], [], [], []
    t0 = time.perf_counter()

    def flush():
        for (c, k), imgs in buf.items():
            if not imgs:
                continue
            x = torch.stack(imgs).to(dev)
            with torch.no_grad():
                if k == 0:
                    res[c]["p1"].append(torch.softmax(net1(x), 1)[:, 1].float().cpu())
                else:
                    res[c]["p2"].append(torch.softmax(net2[k-1][0](x), 1).float().cpu())
            imgs.clear()

    for i, r in enumerate(rows):
        got = photo(r)
        if got is None:
            continue
        im, lb, W = got
        with torch.no_grad():
            d = to_xyxy(det(tf_det(im).unsqueeze(0).to(dev))).float().cpu()[0].numpy() * W
        det_pred.append(d / W)
        det_true.append(np.array(lb) / W)
        ucx = (lb[0]+lb[2])/2 + rng.uniform(-center_off, center_off)*W
        ucy = (lb[1]+lb[3])/2 + rng.uniform(-center_off, center_off)*W
        fs = FIXEDSCALE_LESION_FRAC * W
        boxes = {"label": lb,
                 "user": square(ucx, ucy, rng.uniform(box_lo, box_hi)*W),
                 "center": square(W/2, W/2, fs),
                 "fixed": square(ucx, ucy, fs),
                 "detect": list(d),
                 "detect_fixed": square((d[0]+d[2])/2, (d[1]+d[3])/2, fs)}
        pending = {}
        for c in CONDS:
            row = {"bbox": [float(v) for v in boxes[c]], "img_w": W, "img_h": W}
            windows = [crop.crop_window(row, tag="f320", cfg=cfg)] + \
                      [crop.crop_window(row, tag=tag, cfg=cfg) for _, tag in net2]
            if any(w is None for w in windows):
                pending = None
                break
            for k, w in enumerate(windows):
                pending[(c, k)] = tf(im.crop(tuple(int(v) for v in w)))
        if pending is None:               # 조건 하나라도 창을 못 만들면 그 사진은 전 조건에서 뺍니다
            det_pred.pop(); det_true.pop()
            continue
        for key, x in pending.items():
            buf[key].append(x)
        truth.append(r.label)
        if len(buf[("label", 0)]) >= a.batch:
            flush()
        if (i+1) % 400 == 0:
            print(f"   {i+1:,}/{len(rows):,}  {(time.perf_counter()-t0)/60:.1f}분", flush=True)
    flush()

    y = np.asarray(truth)
    true_g = np.asarray(["정상" if t == "A7" else MORPH_GROUP_KEEP_A6[t] for t in y])
    lesion = y != "A7"
    print(f"\n■ 표본 {len(y):,}장 · {(time.perf_counter()-t0)/60:.1f}분")

    def coverage(cond, cond2=None):
        """`cond2` 를 주면 1단계는 `cond`, 2단계는 `cond2` 의 크롭 — 배선 "검출기는 2단계에만" 을 잽니다."""
        p1 = torch.cat(res[cond]["p1"]).numpy()[:len(y)]
        n_arm = len(net2)
        P2 = np.mean([torch.cat(res[cond2 or cond]["p2"][k::n_arm]).numpy()[:len(y)] for k in range(n_arm)], axis=0)
        P = np.zeros((len(y), 4))
        for j, c in enumerate(CLASSES):
            P[:, G4.index(MORPH_GROUP_KEEP_A6[c])] += P2[:, j]
        said = np.asarray([G4[i] for i in P.argmax(1)])
        conf = p1 * P.max(1)
        order = np.argsort(-conf)
        err = np.cumsum((true_g != said)[order]) / np.arange(1, len(conf)+1)
        okk = np.flatnonzero(err <= NAMING_TARGET_ERROR)
        k = int(okk[-1]) + 1 if len(okk) else 0
        acc = float((said == true_g)[lesion].mean())
        recall1 = float((p1[lesion] >= t1).mean())
        fp1 = float((p1[~lesion] >= t1).mean())
        return {"coverage": k/len(y), "n_said": k, "group_acc_lesion": acc, "stage1_recall": recall1, "stage1_false_alarm": fp1}

    out = {c: coverage(c) for c in CONDS}
    # ★ 배선 후보 — 1단계는 지금처럼 사용자 중심, 검출기는 이상 판정 뒤 2단계 크롭에만. 1단계 지표는 user 와 같습니다.
    out["user1_detect2"] = coverage("user", "detect")
    out["user1_detect_fixed2"] = coverage("user", "detect_fixed")
    # 원 확률을 남깁니다 — 다음엔 다시 안 돌리고 조합만 바꿔 잴 수 있게.
    np.savez_compressed(a.out.with_suffix(".npz"), y=y,
                        **{f"p1_{c}": torch.cat(res[c]["p1"]).numpy()[:len(y)] for c in CONDS},
                        **{f"p2_{c}": np.stack([torch.cat(res[c]["p2"][k::len(net2)]).numpy()[:len(y)] for k in range(len(net2))]) for c in CONDS})
    band = band_report(np.asarray(det_pred), np.asarray(det_true))
    band_les = band_report(np.asarray(det_pred)[lesion], np.asarray(det_true)[lesion])
    print("\n■ 계열 4군 커버리지 (오답률 20% 목표, 헛알림 포함) · 1단계 recall (raw 문턱)")
    print(f"    {'조건':14}{'커버리지':>10}{'장수':>8}{'계열정확도(병변)':>16}{'1단계 recall':>14}{'헛알림':>8}")
    for c in list(CONDS) + ["user1_detect2", "user1_detect_fixed2"]:
        o = out[c]
        print(f"    {c:20}{o['coverage']:>10.1%}{o['n_said']:>8,}{o['group_acc_lesion']:>16.1%}{o['stage1_recall']:>14.1%}{o['stage1_false_alarm']:>8.1%}")
    gain = out["detect"]["coverage"] - out["user"]["coverage"]
    gain2 = out["user1_detect2"]["coverage"] - out["user"]["coverage"]
    print(f"\n■ ★ 배선 '검출기는 2단계에만' (1단계는 사용자 중심 그대로): user1_detect2 − user = {gain2:+.1%}p")
    span = out["label"]["coverage"] - out["user"]["coverage"]
    print(f"\n■ ★ 관문: detect − user = {gain:+.1%}p  (문턱 +{DETECT_MIN_COVERAGE_GAIN:.0%})  "
          f"· 메울 수 있던 폭 {span:.1%}p 중 {gain/span if span > 0 else 0:.0%} 회복")
    print(f"    하한선(center) 대비 {out['detect']['coverage'] - out['center']['coverage']:+.1%}p")
    verdict = "채택 후보" if gain >= DETECT_MIN_COVERAGE_GAIN else ("구분 불가" if gain > 0 else "기각")
    print(f"    → **{verdict}**  (사전등록 STEP50 의 세 문장 중 하나)")
    print(f"\n■ 밴드 (병변만 n={int(lesion.sum()):,}) 둘 다 {band_les['both']:.1%} · 배율 {band_les['in_size']:.1%} · 위치 {band_les['in_pos']:.1%} "
          f"· 중심 오차 중앙값 {band_les['off_median']:.3f} · 하한선 둘 다 {band_les['baseline']['both']:.1%}")
    print("\n⚠️ 릴리스 팔 로컬 구성 기준 — 3팔(STEP 39)과 절대값 비교 금지. 결론은 같은 실행 안의 차이입니다.")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"n": int(len(y)), "n_lesion": int(lesion.sum()), "conditions": out, "gain": gain, "span": span,
                                 "verdict": verdict, "gain_stage2_only": gain2, "band_lesion": band_les, "band_all": band,
                                 "arms": [d.name for d in s2s], "stage1": s1.name, "threshold_raw": t1,
                                 "detector": str(a.detector), "seed": a.seed}, ensure_ascii=False, indent=1))
    print(f"원본: {a.out}")


if __name__ == "__main__":
    main()
