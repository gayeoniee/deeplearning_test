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
CONDS = ("label", "user", "center", "fixed", "detect", "detect_fixed", "uwdetect")
USER_WINDOW = 640      # 사용자 중심 주변 이 크기 창 안에서만 검출 (1080 사진 기준) — 탐색 조건, 사전등록 밖
MULTI_WINDOWS = (1080, 800, 640)   # STEP 54: 사용자 중심 창 여럿을 2단계에 넣고 평균 (1080 = 사진 전체). 재학습 0
RELEASE = Path("/Users/gayeon/deeplearning_test/data/work/release_reference/dogskin_06/release")
MANIFEST = Path("/Users/gayeon/deeplearning_test/data/work/safe_crop_v1/manifest.parquet")


def largest_box(boxes_json):
    boxes = {tuple(float(v) for v in b) for b in json.loads(boxes_json)}
    boxes = [b for b in boxes if b[2] > b[0] and b[3] > b[1]]
    return max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1])) if boxes else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detector", type=Path, required=True, help="detect_best.pt (boxhead) 또는 save_pretrained 폴더 (dfine)")
    ap.add_argument("--detector-kind", choices=["boxhead", "dfine"], default="boxhead")
    ap.add_argument("--detector-stage1", action="store_true", help="검출기 최고 점수를 1단계 점수로도 잽니다 (STEP 52, dfine 만)")
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
    if a.detector_kind == "boxhead":
        det = BoxHead(pretrained=False, img_size=384)
        det.load_state_dict(torch.load(a.detector, map_location="cpu", weights_only=False)["model"])
        det = det.to(dev).eval()

        def detect(im):
            with torch.no_grad():
                return to_xyxy(det(tf_det(im).unsqueeze(0).to(dev))).float().cpu()[0].numpy(), float("nan")
    else:
        # STEP 51 — 진짜 object detection. 점수 최고 네모 하나를 씁니다 (같은 잣대: top-1 vs 가장 큰 정답).
        from transformers import AutoImageProcessor, DFineForObjectDetection
        det_dev = "cpu"                     # MPS 는 deformable attention 이 느리거나 미지원일 수 있어 CPU 로 (0.5s/장)
        proc = AutoImageProcessor.from_pretrained(a.detector)
        det = DFineForObjectDetection.from_pretrained(a.detector).to(det_dev).eval()

        def detect(im):
            """(top-1 네모 0~1, 최고 쿼리 점수). 점수는 STEP 52 의 1단계 후보 점수."""
            with torch.no_grad():
                out = det(**proc(images=im, return_tensors="pt").to(det_dev))
            score = float(out.logits.float().sigmoid().amax())
            r = proc.post_process_object_detection(out, threshold=0.0, target_sizes=[(1, 1)])[0]
            if not len(r["scores"]):
                return np.array([0.4, 0.4, 0.6, 0.6]), score
            return r["boxes"][int(r["scores"].argmax())].float().numpy(), score
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

    MW = [f"mw{w}" for w in MULTI_WINDOWS]
    res = {c: {"p1": [], "p2": []} for c in list(CONDS) + MW}
    buf = {(c, k): [] for c in list(CONDS) + MW for k in range(1+len(net2))}
    truth, det_pred, det_true, det_scores, uw_pred = [], [], [], [], []
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
        d, dscore = detect(im)
        d = d * W
        det_pred.append(d / W)
        det_true.append(np.array(lb) / W)
        det_scores.append(dscore)
        ucx = (lb[0]+lb[2])/2 + rng.uniform(-center_off, center_off)*W
        ucy = (lb[1]+lb[3])/2 + rng.uniform(-center_off, center_off)*W
        # ★ 사용자 창 안 검출: 사용자 중심 주변 USER_WINDOW 창만 검출기에 주고 좌표를 되돌립니다 (탐색 조건)
        uw = USER_WINDOW
        ux0 = int(min(max(ucx - uw/2, 0), W - uw)); uy0 = int(min(max(ucy - uw/2, 0), W - uw))
        dw, _ = detect(im.crop((ux0, uy0, ux0+uw, uy0+uw)))
        dw = np.array([ux0 + dw[0]*uw, uy0 + dw[1]*uw, ux0 + dw[2]*uw, uy0 + dw[3]*uw])
        uw_pred.append(dw / W)
        fs = FIXEDSCALE_LESION_FRAC * W
        boxes = {"label": lb,
                 "user": square(ucx, ucy, rng.uniform(box_lo, box_hi)*W),
                 "center": square(W/2, W/2, fs),
                 "fixed": square(ucx, ucy, fs),
                 "detect": list(d),
                 "detect_fixed": square((d[0]+d[2])/2, (d[1]+d[3])/2, fs),
                 "uwdetect": list(dw)}
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
        if pending is not None:
            # ★ 다중 창: 사용자 중심 주변 창을 잘라 **그 창을 사진으로 보고** 팔마다 자기 크롭(bbox 없음 → 중앙)을 적용
            for w in MULTI_WINDOWS:
                wx0 = int(min(max(ucx - w/2, 0), W - w)); wy0 = int(min(max(ucy - w/2, 0), W - w))
                sub = im.crop((wx0, wy0, wx0 + w, wy0 + w))
                pending[(f"mw{w}", 0)] = pending[("user", 0)]                     # 1단계는 사용자 조건 그대로 (2단계만 바꿈)
                for k, (_, tag) in enumerate(net2, start=1):
                    win = crop.crop_window({"bbox": None, "img_w": w, "img_h": w}, tag=tag, cfg=cfg)
                    pending[(f"mw{w}", k)] = tf(sub.crop(tuple(int(v) for v in win)))
        if pending is None:               # 조건 하나라도 창을 못 만들면 그 사진은 전 조건에서 뺍니다
            det_pred.pop(); det_true.pop(); det_scores.pop(); uw_pred.pop()
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

    def coverage(cond, cond2=None, p1_override=None):
        """`cond2` 를 주면 1단계는 `cond`, 2단계는 `cond2` 의 크롭 — 배선 "검출기는 2단계에만" 을 잽니다.
        `p1_override` 는 1단계 점수를 통째로 바꿉니다 (STEP 52: 검출기 최고 점수)."""
        p1 = torch.cat(res[cond]["p1"]).numpy()[:len(y)] if p1_override is None else np.asarray(p1_override)[:len(y)]
        n_arm = len(net2)
        conds2 = [cond2 or cond] if not isinstance(cond2, (list, tuple)) else list(cond2)
        P2 = np.mean([torch.cat(res[c2]["p2"][k::n_arm]).numpy()[:len(y)] for c2 in conds2 for k in range(n_arm)], axis=0)
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
    out["user1_uwdetect2"] = coverage("user", "uwdetect")           # 사용자 창 안 검출 → 2단계만
    for w in MULTI_WINDOWS:                                          # 창 하나씩
        out[f"user1_mw{w}"] = coverage("user", f"mw{w}")
    out["user1_multiwin"] = coverage("user", MW)                     # ★ STEP 54: 창 셋 평균 (3팔 × 3창 = 9 확률 평균)
    band_uw = band_report(np.asarray(uw_pred)[lesion], np.asarray(det_true)[lesion])
    stage1 = None
    if a.detector_stage1 and not np.isnan(det_scores).all():
        # ★ STEP 52 — 검출기 최고 점수를 1단계 점수로: 현 1단계(사용자 중심 f320)와 같은 표본에서 AUROC · 헛알림@recall · recall@헛알림
        from sklearn.metrics import roc_auc_score
        def s1(scores):
            S = np.asarray(scores); order = np.argsort(-S); ys = lesion[order]
            tp, fp = np.cumsum(ys), np.cumsum(~ys); rec, fpr = tp/lesion.sum(), fp/(~lesion).sum()
            ref_rec = out["user"]["stage1_recall"]; ref_fa = out["user"]["stage1_false_alarm"]
            return {"auroc": float(roc_auc_score(lesion, S)),
                    "fa_at_user_recall": float(fpr[np.searchsorted(rec, ref_rec)]) if (rec >= ref_rec).any() else 1.0,
                    "recall_at_user_fa": float(rec[np.searchsorted(fpr, ref_fa, side="right")-1]) if (fpr <= ref_fa).any() else 0.0}
        stage1 = {"user_f320": s1(torch.cat(res["user"]["p1"]).numpy()[:len(y)]),
                  "center_f320": s1(torch.cat(res["center"]["p1"]).numpy()[:len(y)]),
                  "label_f320": s1(torch.cat(res["label"]["p1"]).numpy()[:len(y)]),
                  "detector_score": s1(det_scores)}
        out["det1_user2"] = coverage("user", "user", p1_override=det_scores)          # 검출기 점수를 1단계로, 2단계는 지금대로
        out["det1_detect2"] = coverage("user", "detect", p1_override=det_scores)      # 검출기가 1단계 점수 + 2단계 크롭 둘 다
    # 원 확률을 남깁니다 — 다음엔 다시 안 돌리고 조합만 바꿔 잴 수 있게.
    np.savez_compressed(a.out.with_suffix(".npz"), y=y,
                        det_score=np.asarray(det_scores)[:len(y)],
                        **{f"p1_{c}": torch.cat(res[c]["p1"]).numpy()[:len(y)] for c in CONDS},
                        **{f"p2_{c}": np.stack([torch.cat(res[c]["p2"][k::len(net2)]).numpy()[:len(y)] for k in range(len(net2))]) for c in list(CONDS) + MW})
    band = band_report(np.asarray(det_pred), np.asarray(det_true))
    band_les = band_report(np.asarray(det_pred)[lesion], np.asarray(det_true)[lesion])
    print("\n■ 계열 4군 커버리지 (오답률 20% 목표, 헛알림 포함) · 1단계 recall (raw 문턱)")
    print(f"    {'조건':14}{'커버리지':>10}{'장수':>8}{'계열정확도(병변)':>16}{'1단계 recall':>14}{'헛알림':>8}")
    for c in list(CONDS) + [k for k in ["user1_detect2", "user1_detect_fixed2", "user1_uwdetect2", *[f"user1_mw{w}" for w in MULTI_WINDOWS], "user1_multiwin", "det1_user2", "det1_detect2"] if k in out]:
        o = out[c]
        print(f"    {c:20}{o['coverage']:>10.1%}{o['n_said']:>8,}{o['group_acc_lesion']:>16.1%}{o['stage1_recall']:>14.1%}{o['stage1_false_alarm']:>8.1%}")
    gain = out["detect"]["coverage"] - out["user"]["coverage"]
    gain2 = out["user1_detect2"]["coverage"] - out["user"]["coverage"]
    gain_mw = out["user1_multiwin"]["coverage"] - out["user"]["coverage"]
    print(f"\n■ ★ STEP 54 다중 창(사용자 중심 {MULTI_WINDOWS}, 2단계 확률 평균, 재학습 0): user1_multiwin − user = {gain_mw:+.1%}p  (관문 +5%p)")
    print(f"\n■ ★ 배선 '검출기는 2단계에만' (1단계는 사용자 중심 그대로): user1_detect2 − user = {gain2:+.1%}p")
    span = out["label"]["coverage"] - out["user"]["coverage"]
    print(f"\n■ ★ 관문: detect − user = {gain:+.1%}p  (문턱 +{DETECT_MIN_COVERAGE_GAIN:.0%})  "
          f"· 메울 수 있던 폭 {span:.1%}p 중 {gain/span if span > 0 else 0:.0%} 회복")
    print(f"    하한선(center) 대비 {out['detect']['coverage'] - out['center']['coverage']:+.1%}p")
    verdict = "채택 후보" if gain >= DETECT_MIN_COVERAGE_GAIN else ("구분 불가" if gain > 0 else "기각")
    print(f"    → **{verdict}**  (사전등록 STEP50 의 세 문장 중 하나)")
    print(f"\n■ 밴드 (병변만 n={int(lesion.sum()):,}) 둘 다 {band_les['both']:.1%} · 배율 {band_les['in_size']:.1%} · 위치 {band_les['in_pos']:.1%} "
          f"· 중심 오차 중앙값 {band_les['off_median']:.3f} · 하한선 둘 다 {band_les['baseline']['both']:.1%}")
    print(f"■ 사용자 창({USER_WINDOW}) 안 검출: 둘 다 {band_uw['both']:.1%} · 위치 {band_uw['in_pos']:.1%} · 중심 오차 중앙값 {band_uw['off_median']:.3f}")
    if stage1:
        print("\n■ ★ 1단계 후보 (STEP 52) — 같은 표본, 같은 가혹 조건")
        print(f"    {'점수':16}{'AUROC':>8}{'헛알림@사용자recall':>20}{'recall@사용자헛알림':>20}")
        for k, v in stage1.items():
            print(f"    {k:16}{v['auroc']:>8.4f}{v['fa_at_user_recall']:>20.1%}{v['recall_at_user_fa']:>20.1%}")
        d1 = stage1["detector_score"]["fa_at_user_recall"] - stage1["user_f320"]["fa_at_user_recall"]
        print(f"    → 검출기 점수의 헛알림 − 현 1단계 = {d1:+.1%}p  (관문 ≤ −5%p) · AUROC 차 {stage1['detector_score']['auroc']-stage1['user_f320']['auroc']:+.4f}")
    print("\n⚠️ 릴리스 팔 로컬 구성 기준 — 3팔(STEP 39)과 절대값 비교 금지. 결론은 같은 실행 안의 차이입니다.")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"n": int(len(y)), "n_lesion": int(lesion.sum()), "conditions": out, "gain": gain, "span": span,
                                 "verdict": verdict, "gain_stage2_only": gain2, "gain_multiwin": gain_mw, "multi_windows": list(MULTI_WINDOWS), "band_lesion": band_les, "band_user_window": band_uw, "stage1_candidates": stage1, "band_all": band,
                                 "arms": [d.name for d in s2s], "stage1": s1.name, "threshold_raw": t1,
                                 "detector": str(a.detector), "detector_kind": a.detector_kind, "seed": a.seed}, ensure_ascii=False, indent=1))
    print(f"원본: {a.out}")


if __name__ == "__main__":
    main()
