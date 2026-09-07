"""사용자 네모에서 **계열 커버리지**가 얼마나 떨어지나 — 전체 파이프라인으로.

    uv run --extra train python tools/user_bbox_coverage.py --n 2500

## 왜 이걸 재나

우리가 제일 많이 인용하는 숫자가 **계열 4군 커버리지 67.9%** 입니다. 그런데
그건 **라벨 네모**(라벨러가 병변에 딱 맞춰 그린 것)로 잰 값입니다.

STEP 36 에서 2단계 macro-F1 이 사용자 네모로 **상대 22~24%** 떨어지는 걸
쟀는데, **커버리지가 얼마나 떨어지는지는 안 쟀습니다.** 문서와 노션에
*"67.9%는 상한이고 실제는 더 낮습니다 — 얼마나인지는 모릅니다"* 라고
적어뒀습니다. **그 구멍을 메웁니다.**

⚠️ STEP 11 에서 *"실제는 더 나쁩니다"* 를 각주로만 적고 안 재서 **두 달을
버렸습니다.** 같은 자리를 또 만들지 않습니다.

## 어떻게

전체 파이프라인 그대로입니다 — 정상 사진까지 넣어 **헛알림도 셉니다.**

    1단계   f320 크롭(중심만) → p(이상)
    2단계   m2.5 크롭 × 3팔 확률 평균 → 6종 분포 → 계열 4군으로 접기
    확신    p1 × p(계열)          ← 서빙과 같은 규칙
    커버리지  확신 내림차순으로 누적 오답률이 20% 를 안 넘는 지점까지

`라벨 네모` 와 `사용자 네모` 두 조건을 **같은 사진**으로 돌려 비교합니다.

⚠️ 한계 — 원본이 VL01 zip 에만 있어 **VL01 기준**입니다. 두 번 데인 청크이고,
커버리지에서는 **비관적**인 쪽이었습니다 (STEP 29·31). 그러니 이 값은
"전체에서의 하락폭" 이 아니라 **"하락이 있다는 것과 그 크기의 눈금"** 입니다.
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

OUT = ROOT / "reports" / "user_bbox_coverage.json"
G4 = ["융기·발진", "표면 변화", "미란·궤양", "결절·종괴"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2500, help="병변 장수")
    ap.add_argument("--n-normal", type=int, default=None,
                    help="정상 장수. 안 주면 실측 헛알림 몫(19%%)에 맞춥니다")
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--batch", type=int, default=20)
    a = ap.parse_args()

    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image

    from src import crop, data, env, models, train
    from src.config import CFG, CLASSES, MORPH_GROUP_KEEP_A6
    from src.experiments import NAMING_TARGET_ERROR

    from user_bbox_sim import user_box_stats

    (box_lo, box_hi), center_off = user_box_stats()
    df = pd.read_parquet(env.work_root() / "manifests" / "manifest_final.parquet")
    df = df[df["zip_path"].notna() & df["zip_member"].notna()]
    lab = df.get("label_orig", df.get("label")).astype(str)
    les = df[(lab != "A7") & df["bbox"].notna()]
    nrm = df[lab == "A7"]
    n_norm = a.n_normal
    if n_norm is None:
        # 넘긴 사진 중 헛알림 몫 19.0% (STEP 31 실측) 를 맞춥니다
        n_norm = int(a.n * 0.19 / (1 - 0.19))
    les = les.sample(min(len(les), a.n), random_state=a.seed)
    nrm = nrm.sample(min(len(nrm), n_norm), random_state=a.seed)
    print(f"■ 병변 {len(les):,}장 + 정상 {len(nrm):,}장 (VL01)")

    ck = env.ensure_dirs()["checkpoints"]
    s1 = next((d for d in sorted(ck.iterdir())
               if d.name.startswith("stage1_") and (d / "best.pt").exists()), None)
    # ★ **배포된 3팔만** 씁니다. 체크포인트 폴더에는 안 쓰는 것도 있습니다
    #   (`f448` 은 STEP 20 에서 기각). 로컬에 있는 걸 다 넣으면 배포와 다른
    #   구성을 재게 되고, 그러면 숫자가 뭘 말하는지 알 수 없습니다.
    DEPLOYED = ("stage2_convnextv2_base_m2.5_384_n121k_moderate",
                "stage2_effnetv2_s_f320_384_moderate",
                "stage2_effnetv2_s_m2.5_384_moderate")
    s2s = [ck / n for n in DEPLOYED if (ck / n / "best.pt").exists()]
    missing = [n for n in DEPLOYED if not (ck / n / "best.pt").exists()]
    if missing:
        print(f"⚠️ 배포 팔이 빠졌습니다: {missing} — 숫자가 배포와 다릅니다")
    if s1 is None or not s2s:
        raise SystemExit("[X] 체크포인트가 없습니다.")
    print(f"■ 1단계 {s1.name}")
    print(f"■ 2단계 {len(s2s)}팔: " + ", ".join(d.name for d in s2s))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = CFG(img_size=384)
    tf = data.build_transforms(cfg, train=False)

    def load(d, n_cls):
        m = models.build(train.model_key_from_exp(d.name) or "effnetv2_s",
                         n_classes=n_cls, pretrained=False, verbose=False)
        return train._load_into(m, d / "best.pt", verbose=False).to(dev).eval()

    from src.agent import crop_tag_from_exp

    net1 = load(s1, 2)
    net2 = [(load(d, len(CLASSES)), crop_tag_from_exp(d.name) or "m2.5")
            for d in s2s]

    zips: dict[str, zipfile.ZipFile] = {}

    def zf(p):
        if p not in zips:
            zips[p] = zipfile.ZipFile(p)
        return zips[p]

    rng = random.Random(a.seed)
    rows = []
    for _, r in pd.concat([les, nrm]).sample(frac=1, random_state=a.seed).iterrows():
        rows.append(r)

    def photo_and_boxes(r):
        """앱처럼 가까이 찍은 사진 + (라벨 네모, 사용자 네모)."""
        try:
            blob = zf(str(r["zip_path"])).read(str(r["zip_member"]))
        except KeyError:
            return None
        full = Image.open(io.BytesIO(blob)).convert("RGB")
        FW, FH = full.size
        bb = r["bbox"]
        b0 = crop._box4(json.loads(bb) if isinstance(bb, str) else bb) \
            if bb is not None and not (isinstance(bb, float)) else None
        if b0 is None:                     # 정상 사진 — 화면 가운데를 씁니다
            side0 = min(FW, FH) * 0.6
            wx, wy = (FW - side0) / 2, (FH - side0) / 2
            b0 = [wx + side0 * .4, wy + side0 * .4, wx + side0 * .6, wy + side0 * .6]
        long0 = max(b0[2] - b0[0], b0[3] - b0[1])
        if long0 <= 0:
            return None
        side0 = min(max(long0 * rng.uniform(3.5, 6.0), 480.0), FW, FH)
        lo_x, hi_x = max(0.0, b0[2] - side0), min(b0[0], FW - side0)
        lo_y, hi_y = max(0.0, b0[3] - side0), min(b0[1], FH - side0)
        if lo_x > hi_x or lo_y > hi_y:
            return None
        wx, wy = rng.uniform(lo_x, hi_x), rng.uniform(lo_y, hi_y)
        im = full.crop((int(wx), int(wy), int(wx + side0), int(wy + side0)))
        if im.size[0] != 1080:
            im = im.resize((1080, 1080))
        W = im.size[0]
        s = W / side0
        lb = [(b0[0] - wx) * s, (b0[1] - wy) * s, (b0[2] - wx) * s, (b0[3] - wy) * s]
        side = rng.uniform(box_lo, box_hi) * W
        cx = (lb[0] + lb[2]) / 2 + rng.uniform(-center_off, center_off) * W
        cy = (lb[1] + lb[3]) / 2 + rng.uniform(-center_off, center_off) * W
        ub = [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2]
        return im, lb, ub, W

    res = {c: {"p1": [], "p2": []} for c in ("label", "user")}
    truth = []
    buf = {(c, k): [] for c in ("label", "user")
           for k in range(1 + len(net2))}
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
                    res[c]["p2"].append(
                        torch.softmax(net2[k - 1][0](x), 1).float().cpu())
            imgs.clear()

    for i, r in enumerate(rows):
        got = photo_and_boxes(r)
        if got is None:
            continue
        im, lb, ub, W = got
        ok = True
        for c, box in (("label", lb), ("user", ub)):
            row = {"bbox": list(box), "img_w": W, "img_h": W}
            w1 = crop.crop_window(row, tag="f320", cfg=cfg)
            if w1 is None:
                ok = False
                break
            buf[(c, 0)].append(tf(im.crop(tuple(int(v) for v in w1))))
            for k, (_, tag) in enumerate(net2, start=1):
                w2 = crop.crop_window(row, tag=tag, cfg=cfg)
                if w2 is None:
                    ok = False
                    break
                buf[(c, k)].append(tf(im.crop(tuple(int(v) for v in w2))))
        if not ok:
            for key in buf:
                if buf[key]:
                    buf[key].pop()
            continue
        truth.append(str(r.get("label_orig", r["label"])))
        if len(buf[("label", 0)]) >= a.batch:
            flush()
        if (i + 1) % 400 == 0:
            print(f"   {i+1:,}/{len(rows):,}  {(time.perf_counter()-t0)/60:.1f}분",
                  flush=True)
    flush()

    y = np.asarray(truth)
    true_g = np.asarray(["정상" if t == "A7" else MORPH_GROUP_KEEP_A6[t] for t in y])
    print(f"\n■ 표본 {len(y):,}장 · {(time.perf_counter()-t0)/60:.1f}분")

    def coverage(cond, pick=None):
        """`pick` 으로 팔을 골라 잽니다 — 크롭 종류별 비교용.

        ★ 왜 필요한가 — `m2.5` 는 네모 **크기**로 창이 정해져 사용자 네모에서
        **포화**됩니다(STEP 38). `f320` 은 크기를 안 쓰므로 면역입니다.
        STEP 36 은 이 둘을 **macro-F1** 로 비교해 기각했는데(+0.021),
        커버리지는 −55% 로 훨씬 크게 무너지므로 **답이 다를 수 있습니다.**

        ⚠️ 이건 **측정이지 판정이 아닙니다.** 채택하려면 문턱을 먼저 박고
        holdout 으로 확인해야 합니다.
        """
        p1 = torch.cat(res[cond]["p1"]).numpy()[:len(y)]
        n_arm = len(net2)
        idx = list(range(n_arm)) if pick is None else pick
        P2 = np.mean([torch.cat(res[cond]["p2"][k::n_arm]).numpy()[:len(y)]
                      for k in idx], axis=0)
        P = np.zeros((len(y), 4))
        for j, c in enumerate(CLASSES):
            P[:, G4.index(MORPH_GROUP_KEEP_A6[c])] += P2[:, j]
        said = np.asarray([G4[i] for i in P.argmax(1)])
        conf = p1 * P.max(1)
        order = np.argsort(-conf)
        err = np.cumsum((true_g != said)[order]) / np.arange(1, len(conf) + 1)
        ok = np.flatnonzero(err <= NAMING_TARGET_ERROR)
        k = int(ok[-1]) + 1 if len(ok) else 0
        # ★ **덜 정확해진 것**과 **틀렸는데 확신하는 것**을 갈라야 합니다.
        #   전부 말했을 때의 정확도(acc)는 전자, 확신도의 변별력(AUROC)은 후자.
        acc = float((said == true_g).mean())
        corr = (said == true_g).astype(float)
        o = np.argsort(conf)
        r = np.empty(len(conf))
        r[o] = np.arange(len(conf))
        n1, n0 = corr.sum(), len(corr) - corr.sum()
        auroc = float(((r * corr).sum() - n1 * (n1 - 1) / 2) / (n1 * n0)) \
            if n1 and n0 else float("nan")
        return k / len(y), k, acc, auroc

    cl, kl, al, ul = coverage("label")
    cu, ku, au, uu = coverage("user")
    print("\n■ 계열 4군 커버리지 (오답률 20% 목표, 헛알림 포함)")
    print(f"    라벨 네모    {cl:.1%}  ({kl:,}장)")
    print(f"    사용자 네모  {cu:.1%}  ({ku:,}장)")
    print(f"    차이         {cu - cl:+.1%}p   상대 {((cu - cl) / cl if cl else 0):+.1%}")
    print("\n■ ★ 왜 그만큼 무너지나 — 두 몫으로 가릅니다")
    print(f"    {'':14}{'전부 말했을 때 정확도':>22}{'확신도 변별력(AUROC)':>22}")
    print(f"    {'라벨 네모':14}{al:>22.1%}{ul:>22.4f}")
    print(f"    {'사용자 네모':14}{au:>22.1%}{uu:>22.4f}")
    print(f"    {'차이':14}{au - al:>+22.1%}{uu - ul:>+22.4f}")
    print("    → 정확도가 주로 빠지면 **덜 맞히는 것**,")
    print("      AUROC 가 주로 빠지면 **틀렸는데 확신하는 것**입니다.")

    # ── ★ 크롭 종류별 — 포화가 없는 f320 이 사용자 네모에서 버티나 ──────
    tags = [t for _, t in net2]
    grp = {"배포 3팔": None,
           "m2.5 팔만": [i for i, t in enumerate(tags) if t.startswith("m")],
           "f320 팔만": [i for i, t in enumerate(tags) if t.startswith("f")]}
    print("\n■ ★ 크롭 종류별 커버리지 — `m2.5` 는 포화, `f320` 은 면역")
    print(f"    {'':12}{'라벨 네모':>12}{'사용자 네모':>14}{'하락':>10}")
    cov_by = {}
    for nm, pick in grp.items():
        if pick is not None and not pick:
            continue
        a_, *_ = coverage("label", pick)
        b_, *_ = coverage("user", pick)
        cov_by[nm] = {"label": a_, "user": b_}
        print(f"    {nm:12}{a_:>12.1%}{b_:>14.1%}{(b_ - a_) / a_ if a_ else 0:>10.1%}")
    print("\n    ⚠️ **측정이지 판정이 아닙니다.** 채택하려면 문턱을 먼저 박고")
    print("       holdout 으로 확인해야 합니다. 여기 팔 수가 다른 것도 섞여 있습니다")
    print("       (3팔 vs 2팔 vs 1팔) — 앙상블 효과와 크롭 효과가 안 갈립니다.")
    print("\n⚠️ VL01 기준 — 커버리지에서는 **비관적**인 청크였습니다 (STEP 29·31).")
    print("   이 값은 '전체에서의 하락폭' 이 아니라 **하락의 눈금**입니다.")

    OUT.write_text(json.dumps(
        {"n": int(len(y)), "label": cl, "user": cu, "delta": cu - cl,
         "relative": (cu - cl) / cl if cl else 0, "arms": len(net2),
         "acc": {"label": al, "user": au},
         "conf_auroc": {"label": ul, "user": uu},
         "by_crop": cov_by},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"원본: {OUT}")


if __name__ == "__main__":
    main()
