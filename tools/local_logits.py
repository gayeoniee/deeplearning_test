"""STEP 16 가중치로 로짓을 뽑습니다 — 로컬이든 캐글이든.

    uv run --extra train python tools/local_logits.py                # 로컬(VL01)
    uv run --extra train python tools/local_logits.py --chunk all    # 캐글(전체 val)

왜 이 도구가 있나
-----------------
STEP 16 산출물에는 `logits_*.npz` 가 없습니다 — 혼동행렬도 그림으로만 남았고
숫자가 안 남았습니다. 그런데 로짓이 있으면 **재학습 없이** 되는 게 많습니다:
로짓 보정(prior 교정) · 클래스별 dispersion · recall CI · 크기 효과.

전체 val(58,315행)의 크롭은 이 PC 에 없지만, **VL01 청크(39,508행)의 크롭은
100% 있습니다.** 같은 모델 · 같은 분할 · 같은 크롭이라 로짓은 진짜 STEP 16
출력입니다.

⚠️ **VL01 부분집합입니다 (val 의 10.9%).** 절대값을 STEP 16 전체 val 숫자와
   나란히 놓지 마세요 — 청크마다 클래스 분포가 다릅니다 (VL01 에는 A5 가
   70장뿐입니다). **이 안에서의 비교**만 유효합니다.

⚠️ holdout 로짓도 뽑을 수 있지만 **판정에 쓰지 않습니다** — 판정은 val,
   holdout 은 같은 모양인지 확인만 (작업 규칙).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402
import torch                                                    # noqa: E402

from src import crop, data, env, labels, models, split, stages, train   # noqa: E402
from src.config import CFG, CLASSES, CLASSES_STAGE1, MODEL_BY_KEY      # noqa: E402

# ⚠️ VRAM 에 맞춥니다. 로컬 3050(6GB)은 8, 캐글 T4(16GB)는 32 쯤 됩니다.
#    `--batch` 로 덮어쓸 수 있습니다.
def _default_batch() -> int:
    try:
        import torch as _t
        if not _t.cuda.is_available():
            return 8
        gb = _t.cuda.get_device_properties(0).total_memory / 2 ** 30
        return 32 if gb >= 14 else (16 if gb >= 10 else 8)
    except Exception:
        return 8


def _stage(df: pd.DataFrame, *, stage: int, tag: str, exp: str, key: str,
           batch: int, img_size: int, device: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """한 단계의 로짓을 뽑습니다. `ds.df` 를 같이 돌려줍니다 (행 순서가 정답)."""
    view = (stages.to_stage1 if stage == 1 else stages.to_stage2)(
        crop.switch_tag(df, tag, verbose=False))
    _, va = split.get_fold(view, 0)
    classes = CLASSES_STAGE1 if stage == 1 else CLASSES

    ck = train.ckpt_dir(exp) / "best.pt"
    if not ck.exists():
        raise SystemExit(f"[X] {ck} 가 없습니다 — release 의 best.pt 를 여기에 두세요.")
    mkey = train.model_key_from_exp(exp)       # ⚠️ 백본은 **폴더 이름에서** 읽습니다
    spec = MODEL_BY_KEY[mkey]
    print(f"\n[{stage}단계] {mkey} / 크롭 {tag} / val {len(va):,}행")

    cfg = CFG(model_name=spec.timm_name, img_size=img_size, batch_size=batch)
    model = models.load_checkpoint(str(ck), spec, len(classes), device=device).to(device).eval()
    dl, ds = data.eval_loader(va, cfg, model, classes=classes, batch_mult=1)

    lg, y = train.cached_logits(model, dl, key=key, exp=exp, n_cls=len(classes),
                                device=device, ckpt=ck, use_cache=True)
    del model
    torch.cuda.empty_cache()
    return lg.numpy(), y.numpy(), ds.df.reset_index(drop=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="manifest_final.parquet",
                    help="365,428행 매니페스트 (리포 루트)")
    # ⚠️ 로컬은 VL01 크롭만 있어서 기본이 chunk_VL01 입니다. **캐글에서는
    #    `--chunk all`** 로 전체 val 을 돌리세요 — 거기엔 TL01·TL02 크롭이
    #    이미 붙어 있고, 그러면 A4 를 제대로(전체의 88%가 TL02) 잴 수 있습니다.
    ap.add_argument("--chunk", default="chunk_VL01",
                    help="크롭이 있는 청크. 'all' 이면 전부 (캐글용)")
    ap.add_argument("--batch", type=int, default=None)
    # ⚠️ A4 분석에 꼭 필요한 건 **2단계(m2.5)** 뿐입니다. 캐글에 f320 크롭이
    #    안 붙어 있으면 `--stages 2` 로 2단계만 돌리세요.
    ap.add_argument("--stages", default="1,2", help="돌릴 단계 (예: '2')")
    ap.add_argument("--out", default="data/work/reports/step18_local")
    a = ap.parse_args(argv)

    if a.batch is None:
        a.batch = _default_batch()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("⚠️ GPU 가 없습니다 — CPU 로는 아주 오래 걸립니다.")
    else:
        n = torch.cuda.get_device_name(0)
        free, tot = torch.cuda.mem_get_info()
        print(f"GPU {n}  여유 {free/2**30:.1f}/{tot/2**30:.1f} GiB  배치 {a.batch}")

    # ── 캐글/코랩 준비 ────────────────────────────────────────────
    # ⚠️ 로컬에서는 아무 일도 안 합니다 (이미 제자리에 있으므로).
    #    캐글에서는 이 두 줄이 없으면 크롭도 가중치도 못 찾습니다.
    try:
        env.load_prepared()
    except Exception as exc:                     # 로컬은 붙일 게 없어 실패해도 정상
        print(f"[env] load_prepared 건너뜀 ({type(exc).__name__})")
    train.import_previous_run(verbose=True)      # 입력으로 붙인 release → 작업 폴더

    # 매니페스트: 준 경로가 없으면 작업 폴더에서 찾습니다 (캐글 경로).
    mpath = ROOT / a.manifest
    if not mpath.exists():
        alt = env.work_root() / "manifests" / "manifest_final.parquet"
        if not alt.exists():
            raise SystemExit(f"[X] 매니페스트가 없습니다.\n  찾아본 곳: {mpath}\n  그리고: {alt}")
        print(f"[manifest] {mpath} 가 없어 {alt} 를 씁니다")
        mpath = alt

    df = labels.load(mpath)
    if len(df) < 300_000:
        print(f"\n⚠️ {len(df):,}행 — 365,428 보다 훨씬 적습니다. **옛 데이터**일 수 "
              "있습니다.\n   STEP 16 과 비교가 안 됩니다. 붙인 데이터셋을 확인하세요.")
    if a.chunk.lower() not in ("all", "*", ""):
        have = sorted(set(df["chunk"]))
        if a.chunk not in have:
            raise SystemExit(f"[X] 청크 '{a.chunk}' 가 없습니다. 있는 것: {have}")
        df = df[df["chunk"] == a.chunk].reset_index(drop=True)
    else:
        # 캐글: TL01·TL02 크롭이 다 붙어 있으면 전체 val 을 돌립니다.
        df = df.reset_index(drop=True)
        print("[chunk] 전체 청크 — 크롭이 다 붙어 있어야 합니다")
    print(f"\n{a.chunk} {len(df):,}행 / 개체 {df['animal_id'].nunique():,}마리")
    print("클래스:", df["label"].value_counts().sort_index().to_dict())

    # ⚠️ 실험 이름을 하드코딩하지 않고 체크포인트 폴더에서 읽습니다.
    #    1단계 effnetv2_s / 2단계 convnextv2_base 로 **서로 다릅니다.**
    ckroot = ROOT / "data/work/checkpoints"
    exps = {int(p.name[5]): p.name for p in sorted(ckroot.glob("stage*"))
            if (p / "best.pt").exists()}
    want = [int(x) for x in a.stages.split(",") if x.strip()]
    miss = [k for k in want if k not in exps]
    if miss:
        raise SystemExit(
            f"[X] {miss}단계 체크포인트가 없습니다. 찾은 것: {sorted(exps)}\n"
            "   release 를 Add Input 했는지 확인하세요.")
    for k in want:
        print(f"  {k}단계 exp = {exps[k]}")

    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    key = f"{a.chunk}_val"

    for stage, tag in [(1, "f320"), (2, "m2.5")]:
        if stage not in want:
            print(f"\n[{stage}단계] 건너뜀 (--stages {a.stages})")
            continue
        lg, y, rows = _stage(df, stage=stage, tag=tag, exp=exps[stage], key=key,
                             batch=a.batch, img_size=384, device=dev)
        classes = CLASSES_STAGE1 if stage == 1 else CLASSES
        pred = lg.argmax(1)
        acc = float((pred == y).mean())
        print(f"  로짓 {lg.shape}  정확도 {acc:.4f}")

        np.savez_compressed(out / f"stage{stage}_logits.npz", logits=lg, y=y,
                            classes=np.array(classes))
        # 행 메타 — 분석에서 bbox·부위·개체로 쪼갤 수 있게 같이 저장합니다.
        # ⚠️ `label_orig` 를 꼭 넣습니다. `to_stage1` 이 `label` 을 A7/ABNORMAL 로
        #    덮어써서, 이게 없으면 **1단계 오답을 병변 종류별로 못 쪼갭니다.**
        keep = ["image_path", "label", "label_orig", "bbox", "area_ratio",
                "region", "breed", "animal_id", "img_w", "img_h"]
        meta = rows[[c for c in keep if c in rows.columns]].copy()
        meta["y"] = y
        meta["pred"] = pred
        assert len(meta) == len(y), f"행 수 불일치 {len(meta)} vs {len(y)}"
        meta.to_parquet(out / f"stage{stage}_rows.parquet", index=False)
        print(f"  저장 → {out.name}/stage{stage}_logits.npz · stage{stage}_rows.parquet")

    print(f"\n완료. {out}")
    print("⚠️ VL01 부분집합입니다 — STEP 16 전체 val 숫자와 직접 비교하지 마세요.")


if __name__ == "__main__":
    main()
