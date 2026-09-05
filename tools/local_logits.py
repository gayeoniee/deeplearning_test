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


def chunks_with_crops(df: pd.DataFrame, tag: str, sample: int = 300,
                      need: float = 0.95) -> list[str]:
    """크롭이 실제로 있는 청크만 골라냅니다 (청크마다 `sample` 장만 확인).

    ⚠️ 전수 확인은 느립니다 — 365,428번 파일 검사는 네트워크 볼륨(캐글 입력)에서
       몇 분씩 걸립니다. 청크는 통째로 있거나 통째로 없으므로 표본이면 충분합니다.

    2026-09-05 캐글 실측: `m2.5` 데이터셋에 376,074장이 붙어 있는데 **VL01
    39,508장이 통째로 빠져** 있었습니다 (TL01·TL02 만 올라감). 그대로 돌리면
    `switch_tag` 가 보유율 89.2% 로 멈춥니다 — 맞는 동작이지만, 어느 청크가
    빠졌는지는 안 알려줍니다.
    """
    out_dir = env.work_root() / "crops"
    keep = []
    for ch, g in df.groupby("chunk"):
        s = g["image_path"].sample(min(sample, len(g)), random_state=0)
        hit = sum(Path(crop._out_path(x, out_dir, tag)).exists() for x in s)
        rate = hit / len(s)
        mark = "O" if rate >= need else "X"
        print(f"  [{mark}] {ch:<12} 표본 {len(s):>3}장 중 {hit:>3}장 ({rate:.0%})")
        if rate >= need:
            keep.append(ch)
    return sorted(keep)


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
    # ⚠️ 쉼표로 **여러 개**를 줄 수 있습니다. 캐글의 m2.5 데이터셋에 VL01 크롭이
    #    빠져 있는 경우가 있어서(2026-09-05 실측: 376,074장인데 VL01 39,508장이
    #    없음), 그럴 땐 `--chunk chunk_TL01,chunk_TL02` 로 있는 것만 씁니다.
    #    A4 의 88.2% 가 TL02 에 있으므로 A4 분석에는 그걸로 충분합니다.
    ap.add_argument("--chunk", default="chunk_VL01",
                    help="쓸 청크. 쉼표로 여러 개 / 'all' 전부 / 'auto' 크롭 있는 것만")
    ap.add_argument("--batch", type=int, default=None)
    # ⚠️ A4 분석에 꼭 필요한 건 **2단계(m2.5)** 뿐입니다. 캐글에 f320 크롭이
    #    안 붙어 있으면 `--stages 2` 로 2단계만 돌리세요.
    ap.add_argument("--stages", default="1,2", help="돌릴 단계 (예: '2')")
    # 자동 탐색이 못 찾을 때의 탈출구. 캐글 데이터셋이 한 겹 더 싸여 있으면
    # (/kaggle/input/release/release/checkpoints) 여기에 그 경로를 주세요.
    ap.add_argument("--release", default=None,
                    help="release 폴더를 직접 지정 (자동 탐색 실패 시)")
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
    train.import_previous_run(a.release, verbose=True)   # 입력의 release → 작업 폴더

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
    have = sorted(set(df["chunk"]))
    if a.chunk.lower() == "auto":
        # ⚠️ 크롭이 있는 청크만 자동으로 고릅니다. 캐글에서 한 청크가 통째로
        #    빠져 있어도 왕복 없이 진행됩니다 (무엇을 뺐는지는 아래에 찍힙니다).
        print("[chunk] 크롭 보유율을 청크별로 확인합니다 (표본 300장씩)")
        want_chunks = chunks_with_crops(df, "m2.5" if 2 in
                                        [int(x) for x in a.stages.split(",") if x.strip()]
                                        else "f320")
        if not want_chunks:
            raise SystemExit("[X] 크롭이 있는 청크가 하나도 없습니다. "
                             "데이터셋을 Add Input 했는지 확인하세요.")
    elif a.chunk.lower() in ("all", "*", ""):
        want_chunks = have
        print("[chunk] 전체 청크 — 크롭이 다 붙어 있어야 합니다")
    else:
        want_chunks = [c.strip() for c in a.chunk.split(",") if c.strip()]
        bad = [c for c in want_chunks if c not in have]
        if bad:
            raise SystemExit(f"[X] 청크 {bad} 가 없습니다. 있는 것: {have}")
    n_before = len(df)
    df = df[df["chunk"].isin(want_chunks)].reset_index(drop=True)
    if len(df) != n_before:
        # ⚠️ 부분집합을 쓰는 건 **의도적으로 고른 것**일 때만 괜찮습니다.
        #    무엇을 뺐는지 화면에 남겨야 나중에 숫자를 잘못 비교하지 않습니다.
        print(f"[chunk] {want_chunks} 만 씁니다 — {len(df):,}/{n_before:,}행 "
              f"({len(df)/n_before:.1%}). 뺀 청크: {sorted(set(have) - set(want_chunks))}")
    print(f"\n{a.chunk} {len(df):,}행 / 개체 {df['animal_id'].nunique():,}마리")
    print("클래스:", df["label"].value_counts().sort_index().to_dict())

    # ⚠️ 실험 이름을 하드코딩하지 않고 체크포인트 폴더에서 읽습니다.
    #    1단계 effnetv2_s / 2단계 convnextv2_base 로 **서로 다릅니다.**
    # ⚠️ **`ROOT / "data/work"` 를 쓰면 안 됩니다.** 로컬에서는 리포 폴더와
    #    작업 폴더가 같아서 안 드러나지만, 캐글에서는 갈라집니다:
    #        리포     /kaggle/working/deeplearning_test/data/work   ← 비어 있음
    #        작업폴더 /kaggle/working/data/work                     ← 여기로 복사됨
    #    `import_previous_run()` 은 `env.work_root()` 로 복사하므로 여기도 그걸
    #    봐야 합니다. 안 그러면 release 를 제대로 붙여놓고도
    #    "체크포인트가 없습니다" 로 죽습니다 (2026-09-05 캐글에서 실제로).
    #    `train.ckpt_dir()` 도 같은 뿌리를 씁니다.
    ckroot = env.work_root() / "checkpoints"
    exps = {int(p.name[5]): p.name for p in sorted(ckroot.glob("stage*"))
            if (p / "best.pt").exists()}
    want = [int(x) for x in a.stages.split(",") if x.strip()]
    miss = [k for k in want if k not in exps]
    if miss:
        # ⚠️ "없습니다" 만 찍고 죽으면 30분 뒤에 원인을 모릅니다. **어디를 봤고
        #    입력에 뭐가 있는지**를 같이 찍습니다.
        lines = [f"[X] {miss}단계 체크포인트가 없습니다. 찾은 것: {sorted(exps)}",
                 f"    찾아본 곳: {ckroot}",
                 f"    (env.work_root() = {env.work_root()})"]
        found = train.find_checkpoint_sources()
        lines.append(f"    자동 탐색이 본 checkpoints 폴더: {found or '없음'}")
        inp = Path("/kaggle/input")
        if inp.is_dir():
            lines.append(f"    /kaggle/input 목록: {[p.name for p in inp.glob('*')]}")
            hits = list(inp.rglob("best.pt"))[:5]
            lines.append(f"    입력 안의 best.pt: {[str(h) for h in hits] or '없음'}")
            if hits and not found:
                lines.append("    → best.pt 는 있는데 자동 탐색이 못 찾았습니다."
                             " `--release <그 폴더>` 로 직접 주세요.")
        raise SystemExit("\n".join(lines))
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
    if set(want_chunks) != set(have):
        print(f"[!] {want_chunks} **부분집합**입니다 — STEP 16 전체 val 숫자와"
              " 직접 비교하지 마세요 (청크마다 클래스 분포가 다릅니다)."
              f" 쓴 청크: {want_chunks}")
    else:
        print("전체 val 입니다 — STEP 16 숫자와 같은 조건입니다.")


if __name__ == "__main__":
    main()
