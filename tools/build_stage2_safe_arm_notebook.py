"""`notebooks/22_2단계_병변보존창_팔교체_kaggle.ipynb` — 배포 2단계 팔 하나(`stage2_effnetv2_s_m2.5_384_moderate`)를
**같은 레시피 + 학습 크롭 안 병변 보존 창 흔들기**로 다시 배워 갈아 끼우는 확대 실험 (STEP 53, 캐글 무료 T4).

    uv run python tools/build_stage2_safe_arm_notebook.py

환경 셀은 노트북 09(그 팔을 실제로 배운 노트북)에서 베끼고 `NB_BRANCH` 만 main 으로 못 박습니다.
판정은 캐글이 아니라 **로컬** `tools/detect_coverage.py --release <safe 팔로 바꾼 3팔>` (보호자 조건, 작업 규칙 7).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.build_detect_notebook import _pin_branch, code, md  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "09_2단계_고정창_전체확인.ipynb"
OUT = ROOT / "notebooks" / "22_2단계_병변보존창_팔교체_kaggle.ipynb"
NB_BRANCH = "main"

INTRO = r"""
# 22 — 2단계 팔 하나를 **병변 보존 창**으로 다시 배워 갈아 끼우기 (STEP 53, 캐글 T4)

## 묻는 것 하나

STEP 49 파일럿(작은 모델 · 1만 장)에서 *"학습 때 창을 흔들면 위치 강건성 +25%p, clean −0.02"* 를 봤습니다.
그게 **배포 팔 하나를 실제로 갈아 끼웠을 때 보호자 조건 커버리지**(작업 규칙 7)로 이어지는지 봅니다.

- 대조군: 지금 배포 `stage2_effnetv2_s_m2.5_384_moderate` — **재학습 없이 배포 가중치 그대로** (예산 절약, 노트북 09 가 배운 팔)
- 처치: **같은 데이터 · 같은 레시피**(effnetv2_s · 384 · `moderate` · `default` 증강 · 10 epoch · fold 0) 에
  학습 크롭만 `m2.5` 크롭 안에서 **병변(중앙 40%)을 보존하는 창 (변 0.42~1.0, 위치 무작위)** 으로 흔듦 (`CFG.train_window_side`)
- 검증 변환은 그대로 (창 안 흔듦) → 이 노트북의 val macro-F1 은 **라벨 네모 조건** 값. 채택 판정은 로컬 프록시에서.

## 붙일 것 (Add Input)

| 입력 | 필수 |
|---|---|
| `dogskin-m25-step16` (m2.5 크롭) | ✅ |
| `dogskin-manifest-365k` | ✅ |
| `release` (STEP 16 릴리스 — 설정 참조) | ✅ |

**Accelerator: GPU T4** · Internet ON → **Save & Run All (Commit)**. holdout 은 안 엽니다.
사전등록: [`STEP53`](../docs/results/STEP53_2단계_병변보존창_팔교체_사전등록.md) — 관문은 여기서 안 바꿉니다.
"""

SETUP = r"""
import sys
sys.path.insert(0, DIR)
import torch
from src import crop, env, labels, split, stages, experiments
from src.config import CFG

env.load_prepared()
env.require_gpu()
DEV = "cuda"

# ── 여기만 바꾸면 됩니다 ───────────────────────────────────
TAG    = "m2.5"                # 배포 팔과 같은 크롭
MODEL  = "effnetv2_s"          # 배포 팔과 같은 백본 (노트북 09 · STEP 23)
EPOCHS = 10                    # 노트북 09 와 같음 (best epoch 4 였음, 조기 종료 patience 5)
SUBSET = 1.0
WINDOW = (0.42, 1.0)           # 창 변 / 크롭 변. 0.42 = 중앙 40% 병변 상자 + 여유. 상한 1.0 = 크롭 전체
# ──────────────────────────────────────────────────────────

mpath = env.work_root() / "manifests" / "manifest_final.parquet"
df = labels.load(mpath)
print(f"{len(df):,}행")
if len(df) < 300_000:
    print("[!] 365,428 보다 훨씬 적습니다 — 옛 데이터일 수 있습니다.")

have = crop.available_tags()
print("붙어 있는 태그:", have)
if TAG not in have:
    raise SystemExit(f"[X] 태그가 없습니다: {TAG}. dogskin-m25-step16 을 Add Input 하세요.")

keep = crop.chunks_with_crops(df, [TAG])
if not keep:
    raise SystemExit("[X] m2.5 크롭이 있는 청크가 없습니다.")
df = df[df["chunk"].isin(keep)].reset_index(drop=True)
print(f"\n쓸 청크 {keep} — {len(df):,}행")

view = stages.to_stage2(crop.switch_tag(df, TAG, verbose=False))
split.verify(view, fold=0, strict=True)
tr, va = split.get_fold(view, CFG().use_fold)
print(f"2단계 train {len(tr):,} / val {len(va):,}  (holdout 제외)")
print("클래스별 val:", va["label"].value_counts().sort_index().to_dict())

est = experiments.estimate_runtime([MODEL], img_size=384, n_train=int(len(tr) * SUBSET),
                                   epochs=EPOCHS, device=DEV)
print("\n[!] 추정치입니다 — 실측이 아닙니다. 노트북 09 실측: 이 팔 하나에 약 4.5h (T4).")
"""

TRAIN = r"""
import json, time
from pathlib import Path
import numpy as np
from src import train
from src.config import CLASSES

t0 = time.time()
r = experiments.train_and_measure(
    view, stage=2, img_size=384, crop_tag=TAG, device=DEV,
    model_name=MODEL, finetune="moderate", aug="default",
    epochs=EPOCHS, subset_frac=SUBSET,
    train_window=WINDOW,                       # ★ 처치 — 이것 하나만 배포 팔과 다릅니다
    measure_robust=True, measure_blur=False, n_robust=2000)

exp = r["exp_name"]
print(f"\n실험 이름 {exp}")
assert exp.endswith("_safe0.42"), f"이름에 처치가 안 붙었습니다: {exp}"
print(f"  macro-F1 {r['macro_f1']:.4f} · A6 recall {r['a6_recall']:.3f} · 배율 하락 {r.get('scale_drop', float('nan')):.1%}"
      f" · best epoch {r['best_epoch']} / {r['n_epochs']} · {(time.time()-t0)/60:.0f}분")
print("  ⚠️ 이 값은 라벨 네모(val) 조건입니다 — 배포 팔 val macro-F1 0.5894(STEP 23) 옆에 참고만. 채택 판정은 로컬 프록시.")
"""

EXPORT = r"""
import shutil, zipfile
OUTDIR = Path("/kaggle/working/step53_safe_arm")
shutil.rmtree(OUTDIR, ignore_errors=True)
src_dir = train.ckpt_dir(exp)
dst = OUTDIR / "checkpoints" / exp
dst.mkdir(parents=True)
for name in ["best.pt", "result.json", "history.csv", "config.json", "logits_val.npz"]:
    p = src_dir / name
    if p.exists():
        shutil.copy2(p, dst / name)
print("담은 파일:", sorted(q.name for q in dst.iterdir()))
assert (dst / "best.pt").exists(), "best.pt 가 없습니다"

summary = {"step": "STEP 53 — 2단계 병변 보존 창, 배포 팔 교체 확대 실험", "exp_name": exp, "window": list(WINDOW),
           "model": MODEL, "tag": TAG, "epochs": EPOCHS, "subset_frac": SUBSET, "chunks": keep,
           "n_train": r["n_train"], "run": {k: v for k, v in r.items() if k != "report"}}
(OUTDIR / "step53_kaggle.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
zpath = Path("/kaggle/working/step53_safe_arm.zip")
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as zf:
    for p in OUTDIR.rglob("*"):
        if p.is_file():
            zf.write(p, p.relative_to(OUTDIR))
print(f"저장: {zpath}  ({zpath.stat().st_size/1e6:.0f} MB) ← Output 탭에서 받아 로컬 판정에 씁니다")
"""

NEXT = r"""
## 다음 — 판정은 로컬에서 (보호자 조건 · 배포 3팔)

```
# 릴리스 사본을 만들고 effnetv2_s m2.5 팔만 safe 팔로 바꿉니다
cp -r data/work/hf_release data/work/hf_release_step53
rm -rf data/work/hf_release_step53/checkpoints/stage2_effnetv2_s_m2.5_384_moderate
unzip step53_safe_arm.zip -d data/work/step53 && cp -r data/work/step53/checkpoints/* data/work/hf_release_step53/checkpoints/
uv run --extra train python tools/detect_coverage.py --n 2500 --release data/work/hf_release_step53 \
    --detector data/work/detect_step50/detect_best.pt --out reports/detect_coverage_step53_safe_3arm.json
```

관문(사전등록): `user` 커버리지 − 31.6% ≥ +5%p · `label` 커버리지 ≥ 58.8% − 2%p. 어느 쪽이든 `docs/results/STEP53_*.md` 에 남깁니다.
"""


def main() -> None:
    src = json.loads(SRC.read_text(encoding="utf-8"))
    env_cell = next((c for c in src["cells"] if "0. 환경 준비" in "".join(c["source"])), None)
    if env_cell is None:
        raise SystemExit("[X] 09 번에서 환경 셀을 못 찾았습니다.")
    env_cell = _pin_branch(env_cell, NB_BRANCH)
    cells = [md(INTRO), env_cell,
             md("## 1. 데이터와 시간 — 노트북 09 와 같은 분할 (fold 0 · holdout 제외)"), code(SETUP),
             md("## 2. 처치 팔 하나만 배웁니다 (대조군 = 배포 가중치 그대로)"), code(TRAIN),
             md("## 3. 산출물 — best.pt 를 릴리스 폴더 규격(`checkpoints/<실험>/`)으로 담아 ZIP"), code(EXPORT),
             md(NEXT)]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"✅ {OUT.relative_to(ROOT)}  ({len(cells)} cells, NB_BRANCH={NB_BRANCH})")


if __name__ == "__main__":
    main()
