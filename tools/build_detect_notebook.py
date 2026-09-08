"""캐글용 검출 노트북(13)을 만듭니다 — 환경 셀은 기존 노트북에서 **그대로** 복사.

    uv run python tools/build_detect_notebook.py

⚠️ 환경 셀을 **베껴 적지 않습니다.** 12번 노트북의 것을 그대로 가져옵니다 —
그 셀에는 브랜치 못 박기·클론 검증·버전 대조가 들어 있고, 베끼면 그쪽이
고쳐져도 여기가 안 따라옵니다 (촬영 밴드에서 당한 것과 같은 함정).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "13_병변_검출기.ipynb"
SRC = ROOT / "notebooks" / "12_홀드아웃_확인.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


def main() -> None:
    src = json.loads(SRC.read_text(encoding="utf-8"))
    env_cell = next((c for c in src["cells"]
                     if "0. 환경 준비" in "".join(c["source"])), None)
    if env_cell is None:
        raise SystemExit("[X] 12번에서 환경 셀을 못 찾았습니다.")

    cells = [
        md(r"""
# 13 — **병변 검출기** (STEP 42)

## 왜 이게 남았나

네모를 받는 길이 **아홉 개** 닫혔습니다 (STEP 36~41):

| 길 | 결과 |
|---|---|
| 사용자가 네모 **크기** | ❌ 병변 크기와 무관 (상관 −0.05) |
| 앱 기본값 44% 조정 | ❌ 병변이 화면의 5.6~59.6% |
| `f320` 전환 (macro-F1 / 커버리지) | ❌ +0.021 / 라벨에서도 8.0% |
| 사용자가 **점 위치** (탭) | ❌ 0.111 > 0.10 · 병변 안 47.5% |
| 분류기를 **창 탐지기**로 | ❌ 둘 다 밴드 안 6.7% |
| 네모 잡음 증강 | ❌ 라벨 −0.084 |
| 네모 없이 학습 | ❌ 재현에서 뒤집힘 |
| 고정 배율 크롭 | ❌ −1.1%p (중심 오차와 상충) |

**전부 같은 곳에서 막힙니다** — 사람은 병변이 어디에 얼마나 크게 있는지
못 알려줍니다.

★ 그런데 **`bbox` 라벨이 36만 장** 있습니다. 사람에게 묻지 말고 **모델이 그
일을 직접 배우게** 하는 정공법인데, **시도조차 안 했습니다.**

⚠️ 위 "창 탐지기" 와 다릅니다 — 거기서는 **분류기를 빌려 썼습니다.**
네모를 뽑도록 배운 적이 없는 모델이었고 6.7% 였습니다. 여기서는 **직접
배웁니다.**

## 판정 (돌리기 전에 박아둔 것)

| 상수 | 뜻 |
|---|---|
| `DETECT_MIN_USABLE = 0.50` | 제안이 **배율·위치 밴드에 둘 다** 드는 비율 |
| `DETECT_MIN_COVERAGE_GAIN = 0.05` | 그 네모로 자른 크롭의 **계열 커버리지** 이득 |

⚠️ **둘 다** 넘어야 합니다. 네모가 좋아 보여도 크롭이 걸리면 중심 오차가
치명적이 될 수 있습니다 (STEP 41 에서 배운 것). **최종 판정은 항상 제품
지표(커버리지)로** 합니다.

## 🚨 이 노트북은 holdout 을 **안 엽니다**

`tests/test_holdout_discipline.py` 가 감시합니다. 판정은 val 로 합니다.

## 붙일 데이터

| Add input | 무엇 |
|---|---|
| `dogskin-detect` (Private) | `tools/make_detect_dataset.py` 가 만든 것 — **앱처럼 찍은 사진 + 정답 네모** |

⚠️ 크롭 데이터셋(`m2.5`·`f320`)은 **못 씁니다.** 병변을 가운데 놓고 자른
것이라 정답이 항상 한가운데입니다 — 모델이 "가운데" 만 외우면 됩니다.
"""),
        env_cell,
        md(r"""
## 1. 데이터 붙이기

`boxes.parquet` 과 `images/` 를 찾습니다. **못 찾으면 멈춥니다** — 조용히
물러서면 몇 시간을 버립니다.
"""),
        code(r"""
from pathlib import Path
import pandas as pd

# 캐글 입력 어디에 있든 찾습니다 (계정/데이터셋 이름이 사람마다 다릅니다)
cands = list(Path("/kaggle/input").rglob("boxes.parquet")) \
    if Path("/kaggle/input").exists() else []
cands += sorted(Path(".").glob("data/work/detect*/boxes.parquet"))
if not cands:
    raise SystemExit(
        "[X] boxes.parquet 을 못 찾았습니다.\n"
        "    Add input 으로 검출 데이터셋을 붙였는지 확인하세요.\n"
        f"    /kaggle/input 아래: {[p.name for p in Path('/kaggle/input').iterdir()] if Path('/kaggle/input').exists() else '(없음)'}")
BOXES = cands[0]
IMGS = BOXES.parent / "images"
if not IMGS.is_dir():
    raise SystemExit(f"[X] {IMGS} 가 없습니다 (캐글이 폴더를 한 겹 더 싸는 일이 있습니다)")
df = pd.read_parquet(BOXES)
print(f"■ {len(df):,}장 · 개체 {df['group'].nunique():,} · {BOXES.parent}")
print(f"■ 이미지 {sum(1 for _ in IMGS.iterdir()):,}개")
assert len(df) > 0
"""),
        md(r"""
## 2. 개체 단위로 가릅니다

같은 개가 학습·평가에 갈라지면 **평가가 거짓말합니다.** 이 프로젝트에서
제일 먼저 못 박은 규칙입니다.
"""),
        code(r"""
import random
SEED = 17
gs = sorted(df["group"].astype(str).unique())
random.Random(SEED).shuffle(gs)
cut = int(len(gs) * 0.85)
tr_g, va_g = set(gs[:cut]), set(gs[cut:])
dtr = df[df["group"].astype(str).isin(tr_g)].reset_index(drop=True)
dva = df[df["group"].astype(str).isin(va_g)].reset_index(drop=True)
print(f"■ 학습 {len(dtr):,}장 / 평가 {len(dva):,}장")
print(f"■ 개체 {len(tr_g):,} / {len(va_g):,}  (겹침 {len(tr_g & va_g)})")
assert not (tr_g & va_g), "개체가 겹칩니다 — 평가가 거짓말합니다"
"""),
        md("## 3. 학습"),
        code(r"""
import torch, time
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from src.detect import BoxHead, box_loss, to_xyxy, band_report, print_report
from src.config import CFG
from src import data as sdata

cfg = CFG(img_size=384)
tf = sdata.build_transforms(cfg, train=False)   # 검출은 기하 증강을 안 씁니다

class DS(Dataset):
    def __init__(self, d): self.d = d
    def __len__(self): return len(self.d)
    def __getitem__(self, i):
        r = self.d.iloc[i]
        with Image.open(IMGS / r["image"]) as im:
            x = tf(im.convert("RGB"))
        return x, torch.tensor([r.x1, r.y1, r.x2, r.y2], dtype=torch.float32)

dev = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, BS, LR = 6, 24, 2e-4
net = BoxHead(pretrained=True, img_size=384).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))
ltr = DataLoader(DS(dtr), batch_size=BS, shuffle=True, num_workers=2, pin_memory=True)
lva = DataLoader(DS(dva), batch_size=BS, shuffle=False, num_workers=2)

def evaluate():
    net.eval(); P, T = [], []
    with torch.no_grad():
        for x, y in lva:
            P.append(to_xyxy(net(x.to(dev))).float().cpu()); T.append(y)
    return band_report(torch.cat(P).numpy(), torch.cat(T).numpy())

t0 = time.perf_counter()
best = None
for ep in range(EPOCHS):
    net.train(); tot = k = 0
    for x, y in ltr:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
            loss = box_loss(net(x), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        tot += float(loss.detach()); k += 1
    sched.step()
    rep = evaluate()
    print(f"ep{ep+1}  loss {tot/max(k,1):.4f}  둘다 {rep['both']:.1%}  "
          f"크기비 {rep['ratio_median']:.2f}  중심 {rep['off_median']:.3f}  "
          f"{(time.perf_counter()-t0)/60:.1f}분", flush=True)
    if best is None or rep["both"] > best["both"]:
        best = rep
        torch.save({"model": net.state_dict(), "report": rep},
                   "detect_best.pt")
print()
ok = print_report(best)
"""),
        md(r"""
## 4. ⚠️ 여기서 멈춥니다

밴드 판정만 나왔습니다. **채택하려면 이 네모로 자른 크롭의 커버리지**가
`DETECT_MIN_COVERAGE_GAIN` 만큼 올라야 합니다 — 그건 2단계 모델이 필요하므로
**별도 노트북**에서 합니다.

⚠️ **여기서 설정을 바꿔 다시 돌리지 마세요.** 밴드가 미달이면 미달로 적습니다.
"""),
        code(r"""
import json
json.dump(best, open("detect_report.json", "w"), ensure_ascii=False, indent=1)
print("저장: detect_best.pt · detect_report.json")
print("\n⚠️ 이 노트북은 holdout 을 안 엽니다. 판정은 val 로 했습니다.")
print("⚠️ 밴드 통과는 절반입니다 — 커버리지가 안 오르면 채택 안 합니다.")
"""),
    ]

    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3",
                                      "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11"}},
          "nbformat": 4, "nbformat_minor": 5}
    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"만들었습니다: {NB.relative_to(ROOT)}  (셀 {len(cells)})")


if __name__ == "__main__":
    main()
