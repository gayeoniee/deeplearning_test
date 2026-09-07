"""네모 잡음 증강 — 사람처럼 엉성한 네모로 학습시켜 2단계를 튼튼하게.

    uv run --extra train python tools/boxnoise_train.py --epochs 3

## 왜

STEP 36 에서 둘이 확인됐습니다:

    ① 사람은 병변의 기하를 못 줍니다
       크기는 병변과 **무관**(상관 −0.05, 정답의 2.80배),
       점을 찍게 해도 병변 **안**에 든 것이 47.5% 뿐
    ② 그 네모로 바꾸면 2단계가 **상대 22~24%** 무너집니다

UI 로 될 문제가 아닙니다 — 번진 병변(비듬·각질 · 태선화)에는 사람이 답할 수
있는 점도 크기도 **없습니다.**

**그러면 사람 말고 모델을 고칩니다.** 지금 2단계는 *"정답에 딱 맞는 네모"* 만
보고 배웠는데, 배포에서는 그런 네모가 안 옵니다. **학습 입력을 배포 입력에
맞추는 것** — 증강의 원래 목적 그대로입니다.

## 무엇을 흔드나

`src/config.py` 의 크롭 함수도, 모델도, 레시피도 **그대로**입니다.
학습이 보는 **네모만** 실측 분포에서 뽑아 흔듭니다:

    크기   화면 대비 U(box_lo, box_hi)  ← 병변 크기와 **무관하게** (실측 그대로)
    중심   ±center_off                   ← 실측 중앙값

분포는 `reports/box_error.json` 에서 **읽습니다** (베껴 적지 않습니다).

## 판정

`src/experiments.py` 에 **돌리기 전에** 박았습니다:

    BOXNOISE_MIN_GAIN        사용자 네모에서 +0.03 이상 → 채택
    BOXNOISE_LABEL_DROP_MAX  라벨 네모가 −0.03 넘게 깎이면 기각

⚠️ 한쪽만 보면 STEP 30 의 "관문을 한쪽만 세우면 반대로 도망간다" 를 반복합니다.

## ⚠️ 한계

원본 사진이 VL01 zip 에만 있어 **VL01 기준**입니다. 두 번 데인 청크입니다
(macro-F1·A4 에선 낙관적, 커버리지·A6 에선 비관적 — 방향을 짐작하지 마세요).
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

OUT = ROOT / "reports" / "boxnoise_train.json"


def build_rows(df, n: int, seed: int):
    """자극을 `box_error.py` 와 **같은 방식**으로 미리 정합니다 (앱처럼 가까이)."""
    rng = random.Random(seed)
    out = []
    for _, r in df.sample(min(len(df), n), random_state=seed).iterrows():
        out.append({"zip": str(r["zip_path"]), "member": str(r["zip_member"]),
                    "bbox": r["bbox"], "label": str(r["label"]),
                    "u": (rng.random(), rng.random(), rng.random(),
                          rng.random(), rng.random())})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=9000, help="학습 장수")
    ap.add_argument("--n-val", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--noise-p", type=float, nargs="*", default=[0.5, 1.0],
                    help="네모 잡음을 **장마다 이 확률로** 적용합니다. 증강을 100%% "
                         "적용하는 건 드문 일입니다 — 섞으면 라벨 네모 성능을 "
                         "지키면서 잡음에 튼튼해질 수 있습니다.")
    a = ap.parse_args()

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from PIL import Image

    from src import crop, data, env, evaluate, models, train
    from src.config import CFG, CLASSES
    from src.experiments import BOXNOISE_LABEL_DROP_MAX, BOXNOISE_MIN_GAIN

    from user_bbox_sim import user_box_stats

    (box_lo, box_hi), center_off = user_box_stats()

    df = pd.read_parquet(env.work_root() / "manifests" / "manifest_final.parquet")
    df = df[df.get("label_orig", df.get("label")) != "A7"]
    df = df[df["zip_path"].notna() & df["zip_member"].notna() & df["bbox"].notna()]
    # ⚠️ 개체 단위로 가릅니다 — 같은 개가 학습·평가에 갈라지면 평가가 거짓말합니다
    gcol = "group" if "group" in df.columns else "animal_id"
    gs = sorted(df[gcol].astype(str).unique())
    random.Random(a.seed).shuffle(gs)
    cut = int(len(gs) * 0.85)
    tr_g, va_g = set(gs[:cut]), set(gs[cut:])
    dtr = df[df[gcol].astype(str).isin(tr_g)]
    dva = df[df[gcol].astype(str).isin(va_g)]
    print(f"■ 개체 단위 분할 — 학습 그룹 {len(tr_g):,} / 평가 그룹 {len(va_g):,}")

    tr = build_rows(dtr, a.n, a.seed)
    va = build_rows(dva, a.n_val, a.seed + 1)
    print(f"■ 학습 {len(tr):,}장 · 평가 {len(va):,}장 (VL01)")

    ck = env.ensure_dirs()["checkpoints"]
    base = ck / "stage2_effnetv2_s_m2.5_384_moderate"
    if not (base / "best.pt").exists():
        raise SystemExit(f"[X] {base.name}/best.pt 가 없습니다.")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = CFG(img_size=384)
    tf = data.build_transforms(cfg, train=False)
    print(f"■ 시작점 {base.name} · 장치 {dev}")

    zips: dict[str, zipfile.ZipFile] = {}

    def zf(p):
        if p not in zips:
            zips[p] = zipfile.ZipFile(p)
        return zips[p]

    def make(row, noisy):
        """한 장 → (텐서, 라벨).

        `noisy` 가 True 면 **사용자처럼** 엉성한 네모로, `"none"` 이면
        **네모를 아예 안 쓰고** 사진 전체를 씁니다.

        ★ `"none"` 팔이 중요한 이유 — 입력 쪽 길이 다 막혔으니(STEP 37),
        *"네모를 물어보지 않는다"* 가 실제 선택지입니다. 그게 사용자 네모만큼만
        해줘도 **앱에서 가이드 프레임을 통째로 뺄 수 있습니다.**
        """
        try:
            blob = zf(row["zip"]).read(row["member"])
        except KeyError:
            return None
        full = Image.open(io.BytesIO(blob)).convert("RGB")
        FW, FH = full.size
        b0 = crop._box4(json.loads(row["bbox"]) if isinstance(row["bbox"], str)
                        else row["bbox"])
        if b0 is None:
            return None
        u = row["u"]
        long0 = max(b0[2] - b0[0], b0[3] - b0[1])
        if long0 <= 0:
            return None
        side0 = min(max(long0 * (3.5 + 2.5 * u[0]), 480.0), FW, FH)
        if side0 < long0 * 1.6:
            return None
        lo_x, hi_x = max(0.0, b0[2] - side0), min(b0[0], FW - side0)
        lo_y, hi_y = max(0.0, b0[3] - side0), min(b0[1], FH - side0)
        if lo_x > hi_x or lo_y > hi_y:
            return None
        wx = lo_x + (hi_x - lo_x) * u[1]
        wy = lo_y + (hi_y - lo_y) * u[2]
        im = full.crop((int(wx), int(wy), int(wx + side0), int(wy + side0)))
        # ⚠️ **원본을 닫습니다.** 1920×1080 을 장마다 열고 안 닫으면 쌓여서
        #    시스템 메모리가 터집니다 — 실제로 학습이 두 번 죽었습니다.
        full.close()
        if im.size[0] != 1080:
            im2 = im.resize((1080, 1080))
            im.close()
            im = im2
        W = im.size[0]
        s = W / side0
        b = [(b0[0] - wx) * s, (b0[1] - wy) * s, (b0[2] - wx) * s, (b0[3] - wy) * s]
        if noisy == "none":
            # ★ 네모를 **안 씁니다** — 사진 전체. 앱이 가이드를 뺄 수 있나?
            out = tf(im), CLASSES.index(row["label"])
            im.close()
            return out
        if noisy:
            # ★ 병변 크기와 **무관하게** — 실측 그대로입니다
            side = (box_lo + (box_hi - box_lo) * random.random()) * W
            cx = (b[0] + b[2]) / 2 + (random.random() * 2 - 1) * center_off * W
            cy = (b[1] + b[3]) / 2 + (random.random() * 2 - 1) * center_off * W
            b = [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2]
        win = crop.crop_window({"bbox": list(b), "img_w": W, "img_h": W},
                               tag="m2.5", cfg=cfg)
        if win is None:
            im.close()
            return None
        piece = im.crop(tuple(int(v) for v in win))
        im.close()
        out = tf(piece), CLASSES.index(row["label"])
        piece.close()
        return out

    def batches(rows, noisy, bs: int, shuffle: bool):
        """`noisy` 가 bool 이면 전부/전무, 실수면 **장마다 그 확률로** 잡음."""
        idx = list(range(len(rows)))
        if shuffle:
            random.shuffle(idx)
        xs, ys = [], []
        for k in idx:
            nz = (noisy if isinstance(noisy, str)
                  else (random.random() < noisy) if isinstance(noisy, float)
                  else noisy)
            got = make(rows[k], nz)
            if got is None:
                continue
            xs.append(got[0])
            ys.append(got[1])
            if len(xs) >= bs:
                yield torch.stack(xs), torch.tensor(ys)
                xs, ys = [], []
        if xs:
            yield torch.stack(xs), torch.tensor(ys)

    def score(net, noisy):
        net.eval()
        P, Y = [], []
        with torch.no_grad():
            for x, y in batches(va, noisy, 24, False):
                P.append(net(x.to(dev)).argmax(1).cpu())
                Y.append(y)
        p, y = torch.cat(P).numpy(), torch.cat(Y).numpy()
        return evaluate.metrics(y, p, None, len(CLASSES))["macro_f1"]

    def score3(net):
        """★ 세 조건을 **다** 잽니다 — 라벨 · 사용자 · 네모 없음.

        한 조건만 보면 STEP 30 의 *"관문을 한쪽만 세우면 반대로 도망간다"* 를
        반복합니다. 특히 '네모 없음' 은 앱에서 가이드를 뺄 수 있느냐의 답이라
        어느 팔에서든 찍혀야 합니다.
        """
        return score(net, False), score(net, True), score(net, "none")

    def fresh():
        m = models.build(train.model_key_from_exp(base.name) or "effnetv2_s",
                         n_classes=len(CLASSES), pretrained=False, verbose=False)
        return train._load_into(m, base / "best.pt", verbose=False).to(dev)

    t0 = time.perf_counter()
    print("\n■ 기준선 (지금 배포된 모델 — 파인튜닝 없음)")
    base_net = fresh()
    b_label, b_user, b_none = score3(base_net)
    print(f"    라벨 {b_label:.4f}   사용자 {b_user:.4f}   네모없음 {b_none:.4f}")
    del base_net
    if dev == "cuda":
        torch.cuda.empty_cache()

    def finetune(noisy: bool, tag: str):
        """★ **대조군을 반드시 같이 돌립니다.**

        기준 모델은 원본 1920px 크롭으로 배웠는데 여기 평가는 1080 근접
        크롭입니다 — 분포가 다릅니다. 그러면 **어떤** 파인튜닝이든 오릅니다.
        '잡음 덕분' 과 '분포 적응 덕분' 을 가르려면 **같은 예산으로 라벨
        네모 파인튜닝**을 나란히 돌려야 합니다.
        """
        net = fresh()
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss(label_smoothing=0.05)
        scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))
        print(f"\n■ {tag} 학습 {a.epochs}에폭 (lr {a.lr})")
        lb = us = 0.0
        for ep in range(a.epochs):
            net.train()
            tot = k = 0
            for x, y in batches(tr, noisy, a.batch, True):
                x, y = x.to(dev), y.to(dev)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
                    loss = crit(net(x), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                tot += float(loss.detach())
                k += 1
                if k % 150 == 0:
                    print(f"   {tag} ep{ep+1} {k}배치 loss {tot/k:.4f} "
                          f"{(time.perf_counter()-t0)/60:.1f}분", flush=True)
            lb, us, nn_ = score3(net)
            print(f"  {tag} ep{ep+1}  라벨 {lb:.4f}   사용자 {us:.4f}   "
                  f"네모없음 {nn_:.4f}", flush=True)
        del net
        if dev == "cuda":
            torch.cuda.empty_cache()
        return lb, us, nn_

    c_label, c_user, c_none = finetune(0.0, "대조군(잡음 0%)")
    arms = {}
    for p in a.noise_p:
        arms[p] = finetune(float(p), f"잡음 {p:.0%}")
    # ★ 네모를 아예 안 쓰는 팔 — 앱에서 가이드를 뺄 수 있느냐의 답입니다
    arms["none"] = finetune("none", "네모 없음(사진 전체)")

    print("\n" + "=" * 78)
    print(f"  {'':26}{'라벨 네모':>12}{'사용자 네모':>14}{'네모없음':>12}")
    print(f"  {'기준선 (파인튜닝 없음)':26}{b_label:>12.4f}{b_user:>14.4f}{b_none:>12.4f}")
    print(f"  {'대조군 (잡음 0%)':26}{c_label:>12.4f}{c_user:>14.4f}{c_none:>12.4f}")
    for p, (lb, us, nn_) in arms.items():
        nm = "네모 없음으로 학습" if p == "none" else f"잡음 {p:.0%}"
        print(f"  {nm:26}{lb:>12.4f}{us:>14.4f}{nn_:>12.4f}")

    # ★ 잡음의 **순수 몫** = 잡음 − 대조군. 분포 적응 몫은 대조군이 흡수합니다.
    print(f"\n■ 판정 — 잡음의 순수 몫 (대조군 대비, 문턱 +{BOXNOISE_MIN_GAIN} · "
          f"라벨 −{BOXNOISE_LABEL_DROP_MAX})")
    best, res = None, {}
    for p, (lb, us, nn_) in arms.items():
        gain, drop = us - c_user, c_label - lb
        ok = gain >= BOXNOISE_MIN_GAIN and drop <= BOXNOISE_LABEL_DROP_MAX
        nm = "네모 없음" if p == "none" else f"잡음 {p:.0%}"
        print(f"    {nm:12} 사용자 {gain:+.4f}   라벨 {lb - c_label:+.4f}"
              f"   {'⭕ 통과' if ok else '❌'}")
        res[str(p)] = {"label": lb, "user": us, "none": nn_,
                       "gain": gain, "drop": drop, "passed": bool(ok)}
        if ok and (best is None or gain > res[str(best)]["gain"]):
            best = p
    print(f"\n  → {('채택 후보: ' + str(best)) if best is not None else '통과한 설정 없음'}")

    # ★ 앱이 가이드를 뺄 수 있나 — '네모 없음' 팔을 사용자 네모와 견줍니다
    nb = arms["none"]
    print(f"\n■ ★ 앱에서 가이드 프레임을 뺄 수 있나")
    print(f"    네모 없음으로 학습 → 네모 없이 평가  {nb[2]:.4f}")
    print(f"    대조군            → 사용자 네모     {c_user:.4f}")
    d = nb[2] - c_user
    print(f"    차이 {d:+.4f} — "
          + ("**네모를 안 받아도 사용자 네모만큼은 합니다.**" if d >= -0.02
             else "네모를 받는 편이 낫습니다."))
    print(f"\n  참고 — 분포 적응만의 몫: 사용자 {c_user - b_user:+.4f} "
          f"(대조군 − 기준선). 이걸 잡음 덕분이라 하면 안 됩니다.")
    print("\n⚠️ VL01 기준 · 개체 단위 분할 · 두 번 데인 청크입니다.")
    print("⚠️ 여러 확률을 시도했습니다 — **판정 문턱은 안 옮겼습니다.**")

    OUT.write_text(json.dumps(
        {"base": {"label": b_label, "user": b_user},
         "control": {"label": c_label, "user": c_user},
         "arms": res, "best": best, "adaptation_only": c_user - b_user,
         "n_train": len(tr), "n_val": len(va), "epochs": a.epochs},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"원본: {OUT}")


if __name__ == "__main__":
    main()
