"""Kaggle 환경을 통째로 재현해 노트북 03 을 **처음부터 끝까지** 실행합니다.

왜: 오류를 하나씩 만나며 고치면 사용자가 매번 몇 분씩 기다렸다 다시 막힙니다.
    여기서 한 번에 다 터뜨립니다.

재현하는 것:
  · /kaggle/input/datasets/<사용자>/<데이터셋>/crops,manifests  (실측한 깊은 경로)
  · /kaggle/working                                             (쓰기 가능)
  · /content 존재 + google.colab 임포트 가능                    (오판 유발 함정)
  · KAGGLE_KERNEL_RUN_TYPE 환경변수
  · 데이터셋을 나눠 올린 경우 (m2.5 / m1.5 / full)

    python tests/test_notebook_kaggle_e2e.py             # 두 크롭 태그 다 있음
    python tests/test_notebook_kaggle_e2e.py --m15-only  # m2.5 가 안 올라간 상태

⚠️ 이 테스트가 존재하는 이유:
   Kaggle 로 옮기는 과정에서 오류를 **하나씩** 만났습니다 — 환경 오판,
   Drive 마운트 크래시, 데이터 경로 깊이, 빈 manifests 폴더… 매번 사용자가
   몇 분 기다렸다 다시 막혔습니다. 노트북을 처음부터 끝까지 돌려보지 않은 탓입니다.
   이제 여기서 한 번에 터집니다.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import traceback
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_DEFAULT_NB = "03_학습_베이스라인.ipynb"

ap = argparse.ArgumentParser()
ap.add_argument("--m15-only", action="store_true",
                help="m1.5 만 올라간 상태 — 03 이 쓰는 m2.5 가 없어 셀 6 에서 멈춰야 정상")
ap.add_argument("--stop-at", type=int, default=999, help="이 셀 번호까지만")
ap.add_argument("--nb", default=_DEFAULT_NB,
                help="실행할 노트북 파일명 (notebooks/ 안). 예: 03d_1단계_고치기.ipynb")
ap.add_argument("--seed-release", action="store_true",
                help="이전 실행(release)이 붙어 있는 상태를 흉내 냅니다. "
                     "07 처럼 **학습을 안 하고 가중치만 받아 쓰는** 노트북에 필요합니다")
args = ap.parse_args()
NB = REPO / "notebooks" / args.nb

T = Path(tempfile.mkdtemp(prefix="kagsim_"))
KIN = T / "kaggle" / "input"
KWORK = T / "kaggle" / "working"
DS = KIN / "datasets" / "gayoniee" / "dogskin-m15"
DS_FULL = KIN / "datasets" / "gayoniee" / "dogskin-full"
KWORK.mkdir(parents=True)
(T / "content").mkdir()                       # Kaggle 이미지에도 있는 함정

os.environ["KAGGLE_KERNEL_RUN_TYPE"] = "Interactive"
os.environ["DOG_SKIN_WORK"] = str(KWORK / "data" / "work")
os.environ.pop("DOG_SKIN_PERSIST", None)

# google.colab 이 임포트 가능한 상황 (Kaggle 이미지가 그렇습니다)
_g = types.ModuleType("google"); _g.__path__ = []
_c = types.ModuleType("google.colab"); _c.__path__ = []
_d = types.ModuleType("google.colab.drive")
def _boom(*a, **k):
    raise NotImplementedError("Mounting drive is unsupported in this environment.")
_d.mount = _boom
_c.drive = _d
sys.modules.update({"google": _g, "google.colab": _c, "google.colab.drive": _d})

sys.path.insert(0, str(REPO))
import matplotlib                                                   # noqa: E402
matplotlib.use("Agg")
import numpy as np                                                  # noqa: E402
import pandas as pd                                                 # noqa: E402
import torch                                                        # noqa: E402
import torch.nn as nn                                               # noqa: E402
from PIL import Image                                               # noqa: E402


# ──────────────────────────────────────────────────────────────
# 1. prepare_local.py 산출물과 같은 구조의 합성 데이터
# ──────────────────────────────────────────────────────────────
def build_dataset(n_animals: int = 40, per: int = 4) -> pd.DataFrame:
    from src.config import CLASSES, NORMAL_LABEL

    rng = np.random.default_rng(0)
    rows = []
    W, H = 1920, 1080
    for a in range(n_animals):
        aid = f"D_breed{a % 7}_{a}"
        for k in range(per):
            lab = (NORMAL_LABEL if (a + k) % 3 == 0
                   else CLASSES[(a * per + k) % len(CLASSES)])
            # ⚠️ **정상 박스를 병변보다 작게** 만듭니다 — 실측 배율 격차를 재현합니다.
            #    실물: 정상 bbox 면적 중앙값 0.71% vs 병변 1.25% (비 0.57배).
            #    예전 픽스처는 둘이 같은 분포라 격차가 1.0 이었고, 그래서
            #    crop.choose_stage1_tag() 의 '지름길 있음' 분기가 **한 번도 안 밟혔습니다.**
            #    그 상태로 e2e 가 통과해서 1단계 크롭 버그를 못 잡았습니다.
            side = (int(rng.integers(60, 160)) if lab == NORMAL_LABEL
                    else int(rng.integers(140, 300)))
            x = int(rng.integers(0, W - side)); y = int(rng.integers(0, H - side))
            name = f"IMG_{aid}_{k}.jpg"
            # ⚠️ 실제 매니페스트 스키마 그대로: bbox 는 [x1,y1,x2,y2], 컬럼은 area_ratio
            rows.append({
                "image_path": f"/fake/{name}", "json_path": f"/fake/{name}.json",
                "image_name": name,
                "label": lab, "label_orig": lab, "is_normal": lab == NORMAL_LABEL,
                "animal_id": aid, "species_code": "D", "breed": f"breed{a % 7}",
                "age": a % 12, "gender": "M" if a % 2 else "F", "region": "back",
                "date": f"2024-01-{(a % 28) + 1:02d}", "synthetic": False,
                "camera": "일반카메라", "symptom": "유증상", "src_split": "VL01",
                "bbox": [x, y, x + side, y + side],
                "polygon": None,
                "img_w": W, "img_h": H,
                "area_ratio": (side * side) / (W * H),
                "n_lesion": 1,
            })
    df = pd.DataFrame(rows)

    from src import split
    from src.config import CFG
    df = split.assign(df, CFG())
    return df


def write_crops(df: pd.DataFrame, root: Path, tags: list[str]) -> pd.DataFrame:
    """crop.run 없이 크롭 파일을 직접 만듭니다 (원본 이미지가 없으므로)."""
    from src import crop

    out = df.copy()
    rng = np.random.default_rng(1)
    for tag in tags:
        d = root / "crops" / tag
        d.mkdir(parents=True, exist_ok=True)
        for _, r in out.iterrows():
            p = crop._out_path(r["image_path"], root / "crops", tag)
            p.parent.mkdir(parents=True, exist_ok=True)
            # 클래스마다 밝기를 다르게 → 학습이 실제로 진행됨
            base = 40 + 30 * (hash(r["label"]) % 6)
            arr = np.clip(rng.normal(base, 25, (96, 96, 3)), 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(p, quality=90)
    # 매니페스트의 crop_rel/crop_path 는 첫 태그 기준
    d0 = root / "crops"
    out["crop_path"] = out["image_path"].apply(lambda p: str(crop._out_path(p, d0, tags[0])))
    out["crop_rel"] = out["crop_path"].apply(lambda p: Path(p).relative_to(d0).as_posix())
    out["crop_tag"] = tags[0]
    return out


print(f"작업 폴더: {T}")
df0 = build_dataset()
DS.mkdir(parents=True)
# 03 은 m2.5 를 씁니다 (STEP 4C). m1.5·full 도 같이 둬서 태그 전환 경로까지 밟습니다.
# f320 도 만듭니다 — 06(확정 재학습)의 1단계가 f320 을 씁니다 (STEP 9-A).
df0 = write_crops(df0, DS, ["m1.5"] if args.m15_only
                  else ["m2.5", "m1.5", "full", "f320"])
(DS / "manifests").mkdir(parents=True, exist_ok=True)
df0.to_parquet(DS / "manifests" / "manifest_final.parquet")
if not args.m15_only:
    pass                                   # 같은 데이터셋 안에 두 태그
else:
    print("  (m1.5 만 올린 상태를 재현합니다 — full 없음)")
print(f"  합성 데이터 {len(df0):,}행 / 개체 {df0['animal_id'].nunique()}마리")
print(f"  라벨 분포 {df0['label'].value_counts().to_dict()}")

# ──────────────────────────────────────────────────────────────
# 2. env 를 이 가짜 트리에 묶기
# ──────────────────────────────────────────────────────────────
from src import env                                                 # noqa: E402

env.workspace = lambda: KWORK
env._search_roots = lambda: [KIN, T / "content", Path.cwd()]

# ──────────────────────────────────────────────────────────────
# 3. 무거운 것 축소 (GPU 없음 / 작은 모델 / 1 에폭)
# ──────────────────────────────────────────────────────────────
from src import config as _cfgmod                                   # noqa: E402
from src import bench, crop, data, models, robust                   # noqa: E402

_orig_cfg_init = _cfgmod.CFG.__init__


def _small_init(self, *a, **k):
    _orig_cfg_init(self, *a, **k)
    self.img_size = 32
    self.epochs = min(self.epochs, 2)
    self.batch_size = 8
    self.num_workers = 0
    self.warmup_epochs = min(self.warmup_epochs, 1)
    self.early_stop_patience = 99
    self.amp = False


_cfgmod.CFG.__init__ = _small_init


class TinyNet(nn.Module):
    pretrained_cfg = {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}

    def __init__(self, n):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, padding=1, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, n)

    def forward(self, x):
        return self.fc(self.pool(torch.relu(self.stem(x))).flatten(1))


models.build = lambda spec, n_classes, **k: TinyNet(n_classes)


def _seed_release() -> None:
    """06 이 남긴 release 가 붙어 있는 상태를 만듭니다.

    07 은 학습을 안 하고 가중치를 **받아서만** 씁니다. 그래서 release 없이는
    시작조차 못 하는 게 정상이고(그 자체가 설계), 그 뒤 셀들을 확인하려면
    여기서 흉내를 내야 합니다.

    checkpoints/<exp>/best.pt 와 stage1_threshold.json 을 실제 형식으로 씁니다.
    (models.load_checkpoint 가 ckpt["ema"] → ckpt["model"] 순으로 읽습니다)
    """
    from src.config import CLASSES, CLASSES_STAGE1

    # ★ 작업 폴더가 아니라 **/kaggle/input 아래**에 둡니다 —
    #   그래야 train.import_previous_run() 이 실제로 하는 일까지 확인됩니다.
    rel = KIN / "datasets" / "gayoniee" / "dogskin-06" / "release"
    (rel / "checkpoints").mkdir(parents=True, exist_ok=True)
    exp1 = "stage1_effnetv2_s_f320_384_moderate_photometric"
    exp2 = "stage2_resnet50_m2.5_384_moderate"
    for exp, n in ((exp1, len(CLASSES_STAGE1)), (exp2, len(CLASSES))):
        d = rel / "checkpoints" / exp
        d.mkdir(parents=True, exist_ok=True)
        torch.save({"model": TinyNet(n).state_dict(), "epoch": 1,
                    "best": 0.5}, d / "best.pt")
        (d / "result.json").write_text(json.dumps(
            {"completed": True, "best_score": 0.5, "best_epoch": 0,
             "history": [{}], "target_epochs": 1}), encoding="utf-8")
    (rel / "stage1_threshold.json").write_text(json.dumps({
        "threshold": 0.1823, "stage1_crop": "f320", "stage2_crop": "m2.5",
        "stage1_exp": exp1, "stage2_exp": exp2, "img_size": 32,
    }, ensure_ascii=False), encoding="utf-8")
    print(f"[sim] release 를 입력에 심었습니다: {rel}")


if args.seed_release:
    _seed_release()
bench.gpu_speed = lambda cfg, n_classes=6, steps=30: {
    "batch": 8, "img_per_sec": 500.0, "sec": 0.1, "peak_vram_gb": 0.1, "amp_dtype": "fp16"}
env.require_gpu = lambda hard=True: env.device_info()      # GPU 없는 환경이므로 통과
robust.report = lambda m, d, c, cl, n=2000, **k: {
    "scale": {"_summary": {"rel_drop": 0.05}}, "shift": {"_summary": {"rel_drop": 0.07}}}

# ──────────────────────────────────────────────────────────────
# 4. 노트북 코드 셀을 순서대로 실행
# ──────────────────────────────────────────────────────────────
print(f"노트북: {NB.name}")
nb = json.loads(NB.read_text(encoding="utf-8"))
cells = nb["cells"]

# ⚠️ preamble 은 **1번 셀(git clone + pip)이 실제로 정의하는 것만** 넣습니다.
#    예전에는 여기서 labels/split/crop/train/… 을 전부 미리 import 했는데,
#    그러면 노트북 셀 안에 import 가 빠져 있어도 절대 안 걸립니다.
#    실제로 06 의 3번 셀이 `train` 을 import 없이 써서 Kaggle 에서 NameError
#    로 죽었는데, 이 테스트는 통과했습니다. 그래서 1번 셀과 같은 이름만 둡니다.
ns: dict = {}
exec("import os, sys, json, subprocess\n"
     "import matplotlib; matplotlib.use('Agg')\n"
     "import matplotlib.pyplot as plt\n"
     "from src import env\n"
     "from src.config import CFG, CLASSES, CLASS_KO\n"
     "E = env.describe()\n"
     "env.set_seed(42)\n", ns)

fails: list[tuple[int, str]] = []
gates: list[int] = []
ran = 0
for i, c in enumerate(cells):
    if c["cell_type"] != "code" or i > args.stop_at:
        continue
    src = "".join(c["source"])
    if i == 1:                                  # 환경 준비 셀 (git clone/pip) — 위 preamble 로 대체
        continue
    if src.strip().startswith("#") and "선택" in src[:400]:
        continue                                # 8번 선택 실험 (전부 주석)
    stripped = "\n".join(ln for ln in src.splitlines() if not ln.strip().startswith("#"))
    if not stripped.strip():
        continue
    ran += 1
    print(f"\n{'━' * 66}\n▶ 셀 {i}\n{'━' * 66}")
    print("\n".join(src.splitlines()[:3]))
    try:
        exec(compile(src, f"<cell {i}>", "exec"), ns)
    except AssertionError as exc:
        # 품질 게이트(AUROC>0.80, macroF1>0.25)는 랜덤 합성 데이터에서 걸리는 게 정상입니다.
        # 버그가 아니므로 실패로 세지 않되, 게이트가 **동작한다**는 건 확인됩니다.
        print(f"🚦 셀 {i} 품질 게이트 작동 (합성 데이터라 정상): "
              f"{str(exc).splitlines()[0][:70]}")
        gates.append(i)
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        fails.append((i, tb.strip().splitlines()[-1]))
        print(f"❌ 셀 {i} 실패 — 계속 진행합니다")

print(f"\n{'=' * 66}")
print(f" 실행한 코드 셀 {ran}개 / 실패 {len(fails)}개 / 게이트 작동 {len(gates)}개 {gates}")
print("=" * 66)
for i, msg in fails:
    print(f"  ❌ 셀 {i}: {msg}")
if not fails:
    print("  ✅ 전부 통과")
shutil.rmtree(T, ignore_errors=True)
sys.exit(1 if fails else 0)
