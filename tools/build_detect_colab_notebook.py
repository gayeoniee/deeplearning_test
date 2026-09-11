"""`notebooks/19_병변_검출기_전체데이터_colab.ipynb` 를 만듭니다 (STEP 50).

    uv run python tools/build_detect_colab_notebook.py

코랩(Pay-as-you-go) 용. 캐글 비공개 Dataset 조각을 `kagglehub` 로 받고, Drive 에
epoch 체크포인트를 써서 세션이 끊겨도 이어갑니다. `src/detect.py` 스냅샷을 셀에 담습니다.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "19_병변_검출기_전체데이터_colab.ipynb"
BUNDLED = ["src/__init__.py", "src/config.py", "src/models.py", "src/detect.py", "src/experiments.py", "src/env.py"]

INTRO = """# 19 · 병변 검출기 — 전체 원본 157k 장 (STEP 50, 코랩)

**런타임 → A100 (없으면 L4)** → 두 번째 셀에서 Drive 허용 → 세 번째 셀 로그인(캐글 사용자명 + 토큰) → Run All.
사전등록: [`STEP50`](../docs/results/STEP50_병변_검출기_전체데이터_사전등록.md) — 문턱은 여기서 안 바꿉니다.

STEP 42 와 **같은 모델·같은 손실**(`src/detect.py`)에 데이터만 7.2배·세 청크. 입력 변환은
`Resize(384)` 만 씁니다 (STEP 42 의 CenterCrop 이 가장자리를 잘라 정답과 어긋났던 것을 고침).

- 분할: `boxes.parquet` 의 `fold` — **0 = 검증**, 1~4 = 학습 (개체 단위)
- 매 epoch: fold 0 전체에 `band_report` (밴드 · 하한선) → Drive 에 `history.json`
- 끝: `detect_best.pt` + `detect_report.json` 을 Drive 와 내 PC 로. **커버리지 판정은 로컬에서**
  (`tools/detect_coverage.py`, 릴리스 팔이 로컬에 있음)

⚠️ holdout 은 안 엽니다 — 데이터에 애초에 없습니다."""

SETUP_HEAD = """from pathlib import Path
import json, os, sys, time, shutil, subprocess
EPOCHS = 6      # STEP 42 와 같게. 재개할 때 바꾸지 않기
BATCH_SIZE = 48
LR = 2e-4
IMG = 384
WORKERS = 8
CODE = Path('/content/detect_code')
from google.colab import drive
drive.mount('/content/drive')
CKPT = Path('/content/drive/MyDrive/dogskin_detect_step50'); CKPT.mkdir(parents=True, exist_ok=True)
"""

SETUP_TAIL = """
for name, source in FILES.items():
    path = CODE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
sys.path.insert(0, str(CODE))
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U', 'timm==1.0.29', 'kagglehub'], check=True)  # 새 캐글 토큰은 최신 kagglehub 필요 — 구버전은 validated 뒤 403
import torch
assert torch.cuda.is_available(), 'GPU 런타임을 고르세요'
print(torch.__version__, torch.cuda.get_device_name(0))
"""

FETCH = """DETECT_DATASETS = ['gayoniee/dogskin-detect-full-0', 'gayoniee/dogskin-detect-full-1', 'gayoniee/dogskin-detect-full-2']
import pandas as pd, kagglehub
for key in ['KAGGLE_USERNAME', 'KAGGLE_KEY', 'KAGGLE_API_TOKEN']:
    os.environ.pop(key, None)
kagglehub.login()
roots = [Path(kagglehub.dataset_download(slug)) for slug in DETECT_DATASETS]
frames = []
for root in roots:
    table = next(root.rglob('boxes.parquet'))
    frames.append((root, pd.read_parquet(table)))
df = frames[0][1]
index = {}
for root, _ in frames:
    for p in root.rglob('*.jpg'):
        index[p.name] = p
df = df[df.image.isin(index)].reset_index(drop=True)
print(f'표 {len(frames[0][1]):,}행 · 사진 {len(index):,}장 · 쓰는 행 {len(df):,}')
assert len(df) >= 0.99*len(frames[0][1]), '조각이 빠졌습니다 — 세 Dataset 을 다 연결하세요'
dtr, dva = df[df.fold != 0].reset_index(drop=True), df[df.fold == 0].reset_index(drop=True)
assert not (set(dtr.group) & set(dva.group)), '개체가 겹칩니다'
print(f'학습 {len(dtr):,} / 검증 {len(dva):,} · 개체 {dtr.group.nunique():,} / {dva.group.nunique():,}')
"""

TRAIN = """import torch, math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from src.detect import BoxHead, box_loss, to_xyxy, band_report, print_report

# STEP 42 는 build_transforms(train=False) 의 CenterCrop 이 가장자리 12% 를 잘랐습니다 — 여기서는 전체 프레임.
tf = T.Compose([T.Resize((IMG, IMG)), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

class DS(Dataset):
    def __init__(self, d): self.d = d
    def __len__(self): return len(self.d)
    def __getitem__(self, i):
        r = self.d.iloc[i]
        with Image.open(index[r.image]) as im:
            x = tf(im.convert('RGB'))
        return x, torch.tensor([r.x1, r.y1, r.x2, r.y2], dtype=torch.float32)

dev = 'cuda'
torch.manual_seed(17)
net = BoxHead(pretrained=True, img_size=IMG).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
scaler = torch.amp.GradScaler('cuda')
ltr = DataLoader(DS(dtr), batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=True, drop_last=True, persistent_workers=True)
lva = DataLoader(DS(dva), batch_size=BATCH_SIZE*2, shuffle=False, num_workers=WORKERS)

last, history, start = CKPT / 'last.pt', [], 0
if last.exists():
    s = torch.load(last, map_location=dev, weights_only=False)
    net.load_state_dict(s['model']); opt.load_state_dict(s['opt']); sched.load_state_dict(s['sched']); scaler.load_state_dict(s['scaler'])
    history, start = s['history'], s['epoch']
    assert s['epochs'] == EPOCHS and s['rows'] == len(dtr), '재개 설정이 다릅니다'
    print(f'재개: epoch {start} 부터 · 지금까지', [(h['epoch'], round(h['both'], 3)) for h in history])
else:
    print('⚠️ 재개 없음 — 처음부터 (Drive 에 last.pt 가 없음)')

def evaluate():
    net.eval(); P, Tt = [], []
    with torch.no_grad(), torch.autocast('cuda'):
        for x, y in lva:
            P.append(to_xyxy(net(x.to(dev))).float().cpu()); Tt.append(y)
    rep = band_report(torch.cat(P).numpy(), torch.cat(Tt).numpy())
    return rep, torch.cat(P).numpy()

t0 = time.perf_counter()
for ep in range(start, EPOCHS):
    net.train(); tot = k = 0
    for x, y in ltr:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            loss = box_loss(net(x).float(), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        tot += float(loss.detach()); k += 1
        if k % 500 == 0:
            print(f'  ep{ep+1} {k}/{len(ltr)} loss {tot/k:.4f} {(time.perf_counter()-t0)/60:.1f}분', flush=True)
    sched.step()
    rep, pred = evaluate()
    rec = {'epoch': ep+1, 'loss': tot/max(k, 1), **{kk: rep[kk] for kk in ['both', 'in_size', 'in_pos', 'ratio_median', 'off_median']},
           'baseline_both': rep['baseline']['both'], 'minutes': (time.perf_counter()-t0)/60}
    history.append(rec)
    print(json.dumps(rec), flush=True)
    if rep['both'] >= max(h['both'] for h in history):
        torch.save({'model': net.state_dict(), 'report': rep, 'epoch': ep+1}, CKPT / 'detect_best.pt')
        import numpy as np
        np.savez_compressed(CKPT / 'val_pred_best.npz', pred=pred, image=dva.image.to_numpy(dtype=str))
    torch.save({'model': net.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'scaler': scaler.state_dict(),
                'history': history, 'epoch': ep+1, 'epochs': EPOCHS, 'rows': len(dtr)}, CKPT / 'last.pt')
    (CKPT / 'history.json').write_text(json.dumps(history, indent=1))
best = torch.load(CKPT / 'detect_best.pt', map_location='cpu', weights_only=False)['report']
print(f"\\n■ best epoch {torch.load(CKPT / 'detect_best.pt', map_location='cpu', weights_only=False)['epoch']}")
print_report(best)
(CKPT / 'detect_report.json').write_text(json.dumps({'best': best, 'history': history, 'train_rows': len(dtr), 'val_rows': len(dva),
                                                     'img': IMG, 'epochs': EPOCHS, 'batch': BATCH_SIZE, 'lr': LR,
                                                     'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)}, indent=1))
"""

DOWNLOAD = """from google.colab import files
print('Drive:', CKPT)
files.download(str(CKPT / 'detect_best.pt'))
files.download(str(CKPT / 'detect_report.json'))
print('⚠️ 밴드는 참고입니다 — 판정은 로컬 tools/detect_coverage.py 의 커버리지로 합니다 (STEP 50 사전등록).')
"""


def cell(kind, text):
    src = text.splitlines(keepends=True)
    if kind == 'markdown':
        return {'cell_type': 'markdown', 'metadata': {}, 'source': src}
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': src}


def main():
    files = {n: (ROOT / n).read_text(encoding='utf-8') for n in BUNDLED}
    setup = SETUP_HEAD + 'FILES = ' + json.dumps(files, ensure_ascii=False) + '\n' + SETUP_TAIL
    nb = {'cells': [cell('markdown', INTRO), cell('code', setup), cell('code', FETCH), cell('code', TRAIN), cell('code', DOWNLOAD)],
          'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                       'language_info': {'name': 'python'}, 'accelerator': 'GPU'},
          'nbformat': 4, 'nbformat_minor': 5}
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'{OUT.relative_to(ROOT)} — 셀 {len(nb["cells"])}개 · {OUT.stat().st_size/1024:.0f}KB · 소스 {len(files)}개')


if __name__ == '__main__':
    main()
