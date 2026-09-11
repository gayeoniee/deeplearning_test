"""`notebooks/20_병변_검출_DFINE_colab.ipynb` — 진짜 object detection (STEP 51, 코랩).

    uv run python tools/build_dfine_colab_notebook.py

D-FINE-L (HF transformers) 을 1클래스 "병변" 으로 파인튜닝합니다. 데이터는 STEP 50 의 캐글 조각 3개 +
`dogskin-detect-full-labels`(창 안 모든 네모). Drive 에 epoch 체크포인트·재개. `src/detect.band_report` 로
STEP 42·50 과 같은 잣대의 밴드·중심 오차를 매 epoch 찍습니다. 판정은 로컬 `tools/detect_coverage.py --detector-kind dfine`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import build_detect_colab_notebook as base  # noqa: E402

ROOT = base.ROOT
OUT = ROOT / "notebooks" / "20_병변_검출_DFINE_colab.ipynb"

INTRO = """# 20 · 병변 object detection — D-FINE-L 1클래스 (STEP 51, 코랩)

**런타임 → A100** → 2번 셀 Drive 허용 → 3번 셀에 캐글 토큰 붙여넣기 → Run All.
사전등록: [`STEP51`](../docs/results/STEP51_object_detection_DFINE_사전등록.md) — 문턱은 여기서 안 바꿉니다.

STEP 42·50 의 "검출기" 는 좌표 4개 회귀였고 중심 오차가 0.107 에서 멈췄습니다. 이번은 **진짜 검출기** —
D-FINE (분포 정제 회귀, Apache-2.0, Objects365→COCO 사전학습) 를 병변 1클래스로 파인튜닝합니다.

- 데이터: STEP 50 과 같은 창 149,800장 + **창 안 모든 네모**(`boxes_multi.parquet`) · fold 0 = 검증
- 증강(분석으로 고른 것): 좌우반전 · 배율 지터 0.7~1.3 · 밝기/대비 0.2. 모자이크·상하반전·블러 없음
- `SWEEP = True` 로 바꾸면 서브셋 2만 장·2 epoch 로 증강 조합 3개를 먼저 비교합니다 (마구 실험)
- 매 epoch: fold 0 표본 6,000장에 top-1 네모 vs 가장 큰 정답 → 밴드·**중심 오차 중앙값** (best 기준) · 마지막에 fold 0 전체
- 끝: `dfine_best/`(save_pretrained) 를 Drive 와 내 PC 로 → 로컬에서 `tools/detect_coverage.py --detector-kind dfine`

⚠️ holdout 은 안 엽니다 — 데이터에 없습니다."""

SETUP_HEAD = base.SETUP_HEAD.replace("EPOCHS = 6      # STEP 42 와 같게. 재개할 때 바꾸지 않기", "EPOCHS = 8      # 재개할 때 바꾸지 않기") \
    .replace("BATCH_SIZE = 48", "BATCH_SIZE = 16") \
    .replace("LR = 2e-4", "LR = 1e-4          # 헤드/인코더·디코더. 백본은 ×0.1") \
    .replace("IMG = 384", "IMG = 640          # D-FINE 기본 입력") \
    .replace("CODE = Path('/content/detect_code')", "CODE = Path('/content/dfine_code')") \
    .replace("CKPT = Path('/content/drive/MyDrive/dogskin_detect_step50')", "CKPT = Path('/content/drive/MyDrive/dogskin_dfine_step51')")
SETUP_HEAD += "MODEL_ID = 'ustc-community/dfine-large-obj2coco-e25'\nSWEEP = False       # True 면 서브셋 증강 비교만 하고 끝냅니다\nAUG = 'base'        # base | color | scale_wide\n"
assert SETUP_HEAD != base.SETUP_HEAD
SETUP_TAIL = base.SETUP_TAIL.replace("'timm==1.0.29', 'kagglehub'", "'transformers>=4.52', 'kagglehub'")
assert SETUP_TAIL != base.SETUP_TAIL

FETCH = base.FETCH.replace(
    "DETECT_DATASETS = ['gayoniee/dogskin-detect-full-0', 'gayoniee/dogskin-detect-full-1', 'gayoniee/dogskin-detect-full-2']",
    "DETECT_DATASETS = ['gayoniee/dogskin-detect-full-0', 'gayoniee/dogskin-detect-full-1', 'gayoniee/dogskin-detect-full-2']\nLABELS_DATASET = 'gayoniee/dogskin-detect-full-labels'"
).replace(
    "df = frames[0][1]\n",
    "labels_root = Path(kagglehub.dataset_download(LABELS_DATASET))\ndf = pd.read_parquet(next(labels_root.rglob('boxes_multi.parquet')))\nimport json as _json\ndf['boxes'] = df.boxes.map(_json.loads)\n"
)
assert FETCH != base.FETCH

TRAIN = """import torch, random, math, json, time
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance
from transformers import AutoImageProcessor, DFineForObjectDetection
from src.detect import band_report, print_report

proc = AutoImageProcessor.from_pretrained(MODEL_ID)
proc.size = {'height': IMG, 'width': IMG}
MEAN = tuple(int(255*m) for m in proc.image_mean)

AUGS = {'base':       dict(zoom=(0.7, 1.3), color=0.2),
        'color':      dict(zoom=(0.7, 1.3), color=0.4),
        'scale_wide': dict(zoom=(0.5, 1.5), color=0.2)}

def augment(im, boxes, cfg, rng):
    \"\"\"좌우반전 · 배율 지터(잘라내거나 여백 채움) · 밝기/대비. 네모는 같이 움직이고 50% 미만 남으면 버립니다.\"\"\"
    W = im.size[0]
    boxes = [list(b) for b in boxes]
    if rng.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
        boxes = [[1-b[2], b[1], 1-b[0], b[3]] for b in boxes]
    z = rng.uniform(*cfg['zoom'])
    if z < 1.0:                                   # 확대: 창 안을 잘라냄
        s = z
        ox, oy = rng.uniform(0, 1-s), rng.uniform(0, 1-s)
        im = im.crop((int(ox*W), int(oy*W), int((ox+s)*W), int((oy+s)*W))).resize((W, W))
        kept = []
        for b in boxes:
            x1, y1, x2, y2 = (b[0]-ox)/s, (b[1]-oy)/s, (b[2]-ox)/s, (b[3]-oy)/s
            cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(1, x2), min(1, y2)
            if cx2 > cx1 and cy2 > cy1 and (cx2-cx1)*(cy2-cy1) >= 0.5*(x2-x1)*(y2-y1):
                kept.append([cx1, cy1, cx2, cy2])
        if not kept:
            return im, None
        boxes = kept
    elif z > 1.0:                                 # 축소: 여백을 평균색으로 채움
        canvas = Image.new('RGB', (int(W*z), int(W*z)), MEAN)
        ox, oy = rng.randint(0, canvas.size[0]-W), rng.randint(0, canvas.size[1]-W)
        canvas.paste(im, (ox, oy))
        im = canvas.resize((W, W))
        boxes = [[(b[0]*W+ox)/(W*z), (b[1]*W+oy)/(W*z), (b[2]*W+ox)/(W*z), (b[3]*W+oy)/(W*z)] for b in boxes]
    if cfg['color']:
        im = ImageEnhance.Brightness(im).enhance(rng.uniform(1-cfg['color'], 1+cfg['color']))
        im = ImageEnhance.Contrast(im).enhance(rng.uniform(1-cfg['color'], 1+cfg['color']))
    return im, boxes

class DS(Dataset):
    def __init__(self, d, train, cfg=None, seed=0):
        self.d, self.train, self.cfg, self.seed = d.reset_index(drop=True), train, cfg, seed
    def __len__(self): return len(self.d)
    def __getitem__(self, i):
        r = self.d.iloc[i]
        with Image.open(index[r.image]) as im:
            im = im.convert('RGB')
        boxes = r.boxes
        if self.train:
            out = augment(im, boxes, self.cfg, random.Random(self.seed*1000003 + i))
            if out[1] is None:
                out = (im, boxes)
            im, boxes = out
        px = proc(images=im, return_tensors='pt')['pixel_values'][0]
        b = torch.tensor(boxes, dtype=torch.float32)
        cxcywh = torch.stack([(b[:,0]+b[:,2])/2, (b[:,1]+b[:,3])/2, b[:,2]-b[:,0], b[:,3]-b[:,1]], 1)
        largest = max(boxes, key=lambda x: (x[2]-x[0])*(x[3]-x[1]))
        return px, {'class_labels': torch.zeros(len(boxes), dtype=torch.long), 'boxes': cxcywh}, torch.tensor(largest)

def collate(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch], torch.stack([b[2] for b in batch])

dev = 'cuda'
def make_model():
    return DFineForObjectDetection.from_pretrained(MODEL_ID, num_labels=1, ignore_mismatched_sizes=True).to(dev)

def make_opt(model):
    bb = [p for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if 'backbone' not in n and p.requires_grad]
    return torch.optim.AdamW([{'params': bb, 'lr': LR*0.1}, {'params': rest, 'lr': LR}], weight_decay=1e-4)

@torch.no_grad()
def evaluate(model, d, n=None):
    \"\"\"top-1 네모 vs 가장 큰 정답 — STEP 42·50 과 같은 잣대.\"\"\"
    model.eval()
    sub = d if n is None else d.sample(n, random_state=7)
    dl = DataLoader(DS(sub, False), batch_size=BATCH_SIZE*2, shuffle=False, num_workers=WORKERS, collate_fn=collate)
    P, T = [], []
    for px, _, largest in dl:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(pixel_values=px.to(dev))
        res = proc.post_process_object_detection(out, threshold=0.0, target_sizes=[(1, 1)]*len(px))
        for r in res:
            k = int(r['scores'].argmax()) if len(r['scores']) else None
            P.append(r['boxes'][k].float().cpu().numpy() if k is not None else np.array([0.4, 0.4, 0.6, 0.6]))
        T.append(largest.numpy())
    return band_report(np.stack(P), np.concatenate(T))

def train(model, d, epochs, cfg, tag, resume=True):
    opt = make_opt(model)
    steps = epochs * (len(d)//BATCH_SIZE)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.1 + 0.9*(1+math.cos(math.pi*min(s, steps)/steps))/2)
    last, history, start, best = CKPT / f'{tag}_last.pt', [], 0, None
    if resume and last.exists():
        s = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(s['model']); opt.load_state_dict(s['opt']); sched.load_state_dict(s['sched'])
        history, start, best = s['history'], s['epoch'], s['best']
        assert s['epochs'] == epochs and s['rows'] == len(d), '재개 설정이 다릅니다'
        print(f'재개: epoch {start} 부터', [(h['epoch'], round(h['off_median'], 4)) for h in history])
    else:
        print('⚠️ 재개 없음 — 처음부터')
    t0 = time.perf_counter()
    for ep in range(start, epochs):
        model.train(); tot = k = 0
        dl = DataLoader(DS(d, True, cfg, seed=ep), batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS,
                        pin_memory=True, drop_last=True, collate_fn=collate, persistent_workers=True)
        for px, labels, _ in dl:
            labels = [{kk: v.to(dev) for kk, v in l.items()} for l in labels]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(pixel_values=px.to(dev, non_blocking=True), labels=labels).loss
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            opt.step(); sched.step()
            tot += float(loss.detach()); k += 1
            if k % 500 == 0:
                print(f'  {tag} ep{ep+1} {k}/{len(dl)} loss {tot/k:.3f} {(time.perf_counter()-t0)/60:.1f}분', flush=True)
        rep = evaluate(model, dva, n=6000)
        rec = {'epoch': ep+1, 'loss': tot/max(k, 1), 'both': rep['both'], 'in_size': rep['in_size'], 'in_pos': rep['in_pos'],
               'ratio_median': rep['ratio_median'], 'off_median': rep['off_median'], 'baseline_both': rep['baseline']['both'],
               'minutes': (time.perf_counter()-t0)/60}
        history.append(rec); print(json.dumps(rec), flush=True)
        if best is None or rep['off_median'] < best:          # 사전등록 부관문이 중심 오차 — 그걸로 best
            best = rep['off_median']
            model.save_pretrained(CKPT / f'{tag}_best'); proc.save_pretrained(CKPT / f'{tag}_best')
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'history': history,
                    'epoch': ep+1, 'epochs': epochs, 'rows': len(d), 'best': best}, last)
        (CKPT / f'{tag}_history.json').write_text(json.dumps(history, indent=1))
    return history

if SWEEP:
    sub = dtr.sample(20000, random_state=3)
    for name, cfg in AUGS.items():
        h = train(make_model(), sub, 2, cfg, f'sweep_{name}', resume=False)
        print(f'■ 증강 {name}: 중심 오차 {h[-1]["off_median"]:.4f} · 둘 다 {h[-1]["both"]:.1%}')
    raise SystemExit('SWEEP 끝 — 이긴 증강을 AUG 에 넣고 SWEEP=False 로 다시 돌리세요')

model = make_model()
history = train(model, dtr, EPOCHS, AUGS[AUG], 'dfine')
best_dir = CKPT / 'dfine_best'
model = DFineForObjectDetection.from_pretrained(best_dir).to(dev)
final = evaluate(model, dva)                                   # fold 0 전체
print('\\n■ fold 0 전체 (best epoch)'); print_report(final)
(CKPT / 'dfine_report.json').write_text(json.dumps({'final': final, 'history': history, 'model': MODEL_ID, 'aug': AUG,
    'train_rows': len(dtr), 'val_rows': len(dva), 'img': IMG, 'epochs': EPOCHS, 'batch': BATCH_SIZE, 'lr': LR,
    'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)}, indent=1))
"""

DOWNLOAD = """import shutil
from google.colab import files
shutil.make_archive('/content/dfine_best', 'zip', CKPT / 'dfine_best')
shutil.copy(CKPT / 'dfine_report.json', '/content/dfine_report.json')
files.download('/content/dfine_best.zip'); files.download('/content/dfine_report.json')
print('⚠️ 판정은 로컬: uv run --extra train python tools/detect_coverage.py --detector-kind dfine --detector <풀어둔 폴더> --release <3팔>')
"""


def main():
    files = {n: (ROOT / n).read_text(encoding='utf-8') for n in base.BUNDLED}
    setup = SETUP_HEAD + 'FILES = ' + json.dumps(files, ensure_ascii=False) + '\n' + SETUP_TAIL
    cell = base.cell
    nb = {'cells': [cell('markdown', INTRO), cell('code', setup), cell('code', FETCH), cell('code', TRAIN), cell('code', DOWNLOAD)],
          'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                       'language_info': {'name': 'python'}, 'accelerator': 'GPU'},
          'nbformat': 4, 'nbformat_minor': 5}
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'{OUT.relative_to(ROOT)} — 셀 {len(nb["cells"])}개 · {OUT.stat().st_size/1024:.0f}KB')


if __name__ == '__main__':
    main()
