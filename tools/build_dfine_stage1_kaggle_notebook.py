"""`notebooks/21_검출기_1단계후보_kaggle.ipynb` — 검출기에 정상을 "네모 없음" 으로 가르쳐 1단계 후보로 (STEP 52, 캐글 무료 T4).

    uv run python tools/build_dfine_stage1_kaggle_notebook.py

캐글 네이티브: Dataset 6+1개를 Add Input 으로 붙이고(병변 조각 3 · 정상 조각 3 · 라벨), 재개는 이전 출력 ZIP(캐글이
풀어 올린 폴더도 됨)을 붙이면 됩니다. fp16+GradScaler(T4), EMA 워밍업, 시간 예산에서 epoch 경계 정지, 끝나면 ZIP.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import build_detect_colab_notebook as base  # noqa: E402
from tools import build_dfine_colab_notebook as dfine  # noqa: E402

ROOT = base.ROOT
OUT = ROOT / "notebooks" / "21_검출기_1단계후보_kaggle.ipynb"

INTRO = """# 21 · 검출기 + 정상 음성 = 1단계 후보 (STEP 52, 캐글 T4)

**Add Input**: `dogskin-detect-full-0/1/2` · `dogskin-detect-normals-0/1/2` · `dogskin-detect-full-labels` (+ 재개 시 이전 출력).
**Accelerator: GPU T4 ×2 도 되지만 하나만 씁니다** · Internet ON → **Save & Run All (Commit)**.
사전등록: [`STEP52`](../docs/results/STEP52_검출기_1단계후보_정상음성_사전등록.md) — 문턱은 여기서 안 바꿉니다.

- 병변 창 149,800 + **정상 창 ~150k(정답 네모 0개)** · fold 0 = 검증
- D-FINE-M · 512 · fp16 · 증강 `dfine` · EMA 워밍업 · `HOURS` 예산 안에서 epoch 경계 정지 → `/kaggle/working/dfine_stage1_resume.zip`
- 매 epoch: 검증 표본(병변 3,000 + 정상 3,000)에 **최고 쿼리 점수 = 1단계 점수** → AUROC · recall@헛알림34% · 헛알림@recall73% · 병변 밴드
- 재개: 출력 ZIP 을 Dataset 으로 붙이면 자동 (풀린 폴더도 찾음). **판정은 로컬** `tools/detect_coverage.py --detector-stage1`

⚠️ holdout 은 안 엽니다."""

SETUP_HEAD = """from pathlib import Path
import json, os, sys, time, shutil, subprocess, zipfile
HOURS = 11.5        # 캐글 세션 상한 12h. 재개할 때만 줄이기
EPOCHS = 3          # 재개할 때 바꾸지 않기
BATCH_SIZE = 16
LR = 1e-4
IMG = 512
WORKERS = 4
CODE = Path('/kaggle/working/dfine_code')
CKPT = Path('/kaggle/working/ckpt'); CKPT.mkdir(parents=True, exist_ok=True)
MODEL_ID = 'ustc-community/dfine-medium-obj2coco'
AUG = 'dfine'
"""

SETUP_TAIL = """
for name, source in FILES.items():
    path = CODE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
sys.path.insert(0, str(CODE))
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U', 'transformers>=4.52'], check=True)
import torch
assert torch.cuda.is_available(), 'GPU T4 를 고르세요'
print(torch.__version__, torch.cuda.get_device_name(0))
T0 = time.monotonic()
"""

FETCH = """import pandas as pd, json as _json
IN = Path('/kaggle/input')
labels = next(IN.rglob('boxes_multi.parquet'))
les = pd.read_parquet(labels); les['boxes'] = les.boxes.map(_json.loads)
normals = pd.concat([pd.read_parquet(p) for p in IN.rglob('boxes.parquet') if 'normals' in str(p)]).drop_duplicates('image')
normals = normals[normals.label == 'A7'].copy(); normals['boxes'] = [[] for _ in range(len(normals))]
index = {p.name: p for p in IN.rglob('*.jpg')}
df = pd.concat([les[['image', 'boxes', 'label', 'group', 'fold']], normals[['image', 'boxes', 'label', 'group', 'fold']]], ignore_index=True)
df = df[df.image.isin(index)].reset_index(drop=True)
print(f'병변 {int((df.label != "A7").sum()):,} · 정상 {int((df.label == "A7").sum()):,} · 사진 파일 {len(index):,}')
assert (df.label != 'A7').sum() > 140000 and (df.label == 'A7').sum() > 140000, '조각이 빠졌습니다 — Dataset 6개를 다 붙이세요'
dtr, dva = df[df.fold != 0].reset_index(drop=True), df[df.fold == 0].reset_index(drop=True)
assert not (set(dtr.group) & set(dva.group)), '개체가 겹칩니다'
# 검증 표본: 병변 3,000 + 정상 3,000 고정 (seed 7) — 매 epoch 같은 사진
dva_s = pd.concat([dva[dva.label != 'A7'].sample(3000, random_state=7), dva[dva.label == 'A7'].sample(3000, random_state=7)]).reset_index(drop=True)
print(f'학습 {len(dtr):,} / 검증 {len(dva):,} (표본 {len(dva_s):,})')
# 재개: 이전 출력 ZIP 또는 캐글이 풀어 올린 폴더
resumed = None
for z in IN.rglob('dfine_stage1_resume*.zip'):
    with zipfile.ZipFile(z) as zf:
        for n in zf.namelist():
            assert not Path(n).is_absolute() and '..' not in Path(n).parts
        zf.extractall(CKPT); resumed = z; break
if resumed is None:
    for p in IN.rglob('dfine_last.pt'):
        shutil.copytree(p.parent, CKPT, dirs_exist_ok=True); resumed = p.parent; break
print('재개 입력:', resumed, '→', sorted(x.name for x in CKPT.iterdir()) if resumed else '⚠️ 재개 없음 — 처음부터')
"""

# 학습 셀: 코랩판(20)의 증강·DS·collate·multiscale 을 그대로 쓰고, 정상(빈 정답)·fp16·시간 예산·1단계 평가를 더합니다.
_src = dfine.TRAIN
_head = _src[: _src.index("class DS(Dataset):")]
_head = _head.replace("from src.detect import band_report, print_report", "from src.detect import band_report\nfrom sklearn.metrics import roc_auc_score")
_head = _head.replace("SCALES = [512, 544, 576, 608, 640, 672, 704, 736, 768]", "SCALES = [416, 448, 480, 512, 544, 576, 608]")
TRAIN = _head + """class DS(Dataset):
    def __init__(self, d, train, cfg=None, seed=0):
        self.d, self.train, self.cfg, self.seed = d.reset_index(drop=True), train, cfg, seed
    def __len__(self): return len(self.d)
    def __getitem__(self, i):
        r = self.d.iloc[i]
        with Image.open(index[r.image]) as im:
            im = im.convert('RGB')
        boxes = list(r.boxes)
        if self.train and self.cfg is not None and boxes:
            im, boxes = augment(im, boxes, self.cfg, random.Random(self.seed*1000003 + i))
        elif self.train:
            rr = random.Random(self.seed*1000003 + i)
            if rr.random() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT); boxes = [[1-b[2], b[1], 1-b[0], b[3]] for b in boxes]
            if self.cfg is not None and rr.random() < 0.5:                 # 정상 사진: 색·줌만 (네모가 없어 잘라낼 제약이 없음)
                im, _ = augment(im, [[0.45, 0.45, 0.55, 0.55]], self.cfg, rr)
        px = proc(images=im, return_tensors='pt')['pixel_values'][0]
        if boxes:
            b = torch.tensor(boxes, dtype=torch.float32)
            cxcywh = torch.stack([(b[:,0]+b[:,2])/2, (b[:,1]+b[:,3])/2, b[:,2]-b[:,0], b[:,3]-b[:,1]], 1)
            largest = max(boxes, key=lambda x: (x[2]-x[0])*(x[3]-x[1]))
        else:
            cxcywh, largest = torch.zeros(0, 4), [-1.0, -1.0, -1.0, -1.0]
        return px, {'class_labels': torch.zeros(len(boxes), dtype=torch.long), 'boxes': cxcywh}, torch.tensor(largest), int(r.label != 'A7')

def collate(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch], torch.stack([b[2] for b in batch]), torch.tensor([b[3] for b in batch])

def multiscale(px, rng):
    size = rng.choice(SCALES)
    return px if size == IMG else torch.nn.functional.interpolate(px, size=(size, size), mode='bilinear', align_corners=False)

dev = 'cuda'
def make_model():
    return DFineForObjectDetection.from_pretrained(MODEL_ID, num_labels=1, ignore_mismatched_sizes=True).to(dev)

def make_opt(model):
    bb = [p for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if 'backbone' not in n and p.requires_grad]
    return torch.optim.AdamW([{'params': bb, 'lr': LR*0.05}, {'params': rest, 'lr': LR}], weight_decay=1.25e-4)

@torch.no_grad()
def evaluate(model, d):
    '''1단계 후보로서: 최고 쿼리 점수 → AUROC · recall@헛알림 34.3% · 헛알림@recall 72.9% (STEP 52 관문 자리) + 병변 밴드.'''
    model.eval()
    dl = DataLoader(DS(d, False), batch_size=BATCH_SIZE*2, shuffle=False, num_workers=WORKERS, collate_fn=collate)
    S, Y, P, T = [], [], [], []
    for px, _, largest, y in dl:
        with torch.autocast('cuda', dtype=torch.float16):
            out = model(pixel_values=px.to(dev))
        S.extend(out.logits.float().sigmoid().amax(dim=(1, 2)).cpu().tolist()); Y.extend(y.tolist())
        res = proc.post_process_object_detection(out, threshold=0.0, target_sizes=[(1, 1)]*len(px))
        for r, t in zip(res, largest.numpy()):
            if t[0] < 0:
                continue
            k = int(r['scores'].argmax()) if len(r['scores']) else None
            P.append(r['boxes'][k].float().cpu().numpy() if k is not None else np.array([0.4, 0.4, 0.6, 0.6])); T.append(t)
    S, Y = np.asarray(S), np.asarray(Y)
    order = np.argsort(-S); s_sorted, y_sorted = S[order], Y[order]
    tp = np.cumsum(y_sorted); fp = np.cumsum(1-y_sorted)
    recall, fpr = tp/max(Y.sum(), 1), fp/max((1-Y).sum(), 1)
    rep = band_report(np.stack(P), np.stack(T))
    return {'auroc': float(roc_auc_score(Y, S)),
            'recall_at_fa343': float(recall[np.searchsorted(fpr, 0.343, side='right')-1]) if (fpr <= 0.343).any() else 0.0,
            'fa_at_recall729': float(fpr[np.searchsorted(recall, 0.729)]) if (recall >= 0.729).any() else 1.0,
            'both': rep['both'], 'in_pos': rep['in_pos'], 'off_median': rep['off_median'], 'ratio_median': rep['ratio_median']}

def export():
    z = Path('/kaggle/working/dfine_stage1_resume.zip'); tmp = z.with_suffix('.part')
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_STORED) as zf:
        for p in sorted(CKPT.rglob('*')):
            if p.is_file():
                zf.write(p, p.relative_to(CKPT))
    tmp.replace(z); print('Resume bundle:', z, flush=True)

from torch.optim.swa_utils import AveragedModel
model = make_model(); opt = make_opt(model)
steps = EPOCHS * (len(dtr)//BATCH_SIZE)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.1 + 0.9*(1+math.cos(math.pi*min(s, steps)/steps))/2)
scaler = torch.amp.GradScaler('cuda')
ema_step = [0]
def ema_avg(avg, new, num_averaged):
    ema_step[0] += 1
    d = 0.9999 * (1 - math.exp(-ema_step[0] / 2000))
    return avg * d + new * (1 - d)
ema = AveragedModel(model, avg_fn=ema_avg, use_buffers=True)
last, history, start, best = CKPT / 'dfine_last.pt', [], 0, None
if last.exists():
    s = torch.load(last, map_location=dev, weights_only=False)
    model.load_state_dict(s['model']); opt.load_state_dict(s['opt']); sched.load_state_dict(s['sched']); scaler.load_state_dict(s['scaler'])
    ema.load_state_dict(s['ema']); ema_step[0] = s['ema_step']; history, start, best = s['history'], s['epoch'], s['best']
    assert s['epochs'] == EPOCHS and s['rows'] == len(dtr), '재개 설정이 다릅니다'
    print(f'재개: epoch {start} 부터', [(h['epoch'], round(h['auroc'], 4)) for h in history], flush=True)
else:
    print('⚠️ 재개 없음 — 처음부터', flush=True)

deadline = T0 + HOURS*3600 - 900          # ZIP 내보내기용 15분 예비
reason = 'completed'
try:
    for ep in range(start, EPOCHS):
        done = [h['minutes'] for h in history]
        per_epoch = max(done[-2:]) if done else 0
        if history and time.monotonic() + per_epoch*60*1.15 > deadline:
            reason = 'time_budget_before_epoch'; print('⏱ 예산 부족 — 다음 세션에서 재개', flush=True); break
        model.train(); tot = k = 0; t_ep = time.monotonic()
        last_epoch = ep == EPOCHS-1; rng = random.Random(1000+ep)
        dl = DataLoader(DS(dtr, True, None if last_epoch else AUGS[AUG], seed=ep), batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=WORKERS, pin_memory=True, drop_last=True, collate_fn=collate, persistent_workers=True)
        for px, labels, _, _ in dl:
            if time.monotonic() > deadline:
                reason = 'time_budget_during_epoch'; break
            if not last_epoch:
                px = multiscale(px, rng)
            labels = [{kk: v.to(dev) for kk, v in l.items()} for l in labels]
            with torch.autocast('cuda', dtype=torch.float16):
                loss = model(pixel_values=px.to(dev, non_blocking=True), labels=labels).loss
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True); continue            # fp16 오버플로 배치는 건너뜀
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            scaler.step(opt); scaler.update(); sched.step(); ema.update_parameters(model)
            tot += float(loss.detach()); k += 1
            if k % 500 == 0:
                print(f'  ep{ep+1} {k}/{len(dl)} loss {tot/k:.3f} {(time.monotonic()-T0)/60:.1f}분', flush=True)
        if reason == 'time_budget_during_epoch':
            print('⏱ epoch 중간에 예산 소진 — 이 epoch 은 다음 세션에서 다시 돕니다', flush=True); break
        rep = evaluate(ema.module, dva_s)
        rec = {'epoch': ep+1, 'loss': tot/max(k, 1), **rep, 'minutes': (time.monotonic()-t_ep)/60}
        history.append(rec); print(json.dumps(rec), flush=True)
        if best is None or rep['auroc'] > best:                        # 1단계 후보 — best 는 AUROC
            best = rep['auroc']; ema.module.save_pretrained(CKPT / 'dfine_best'); proc.save_pretrained(CKPT / 'dfine_best')
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'scaler': scaler.state_dict(),
                    'ema': ema.state_dict(), 'ema_step': ema_step[0], 'history': history, 'epoch': ep+1, 'epochs': EPOCHS,
                    'rows': len(dtr), 'best': best}, last)
        (CKPT / 'dfine_history.json').write_text(json.dumps(history, indent=1))
finally:
    (CKPT / 'session.json').write_text(json.dumps({'stop_reason': reason, 'elapsed_min': (time.monotonic()-T0)/60, 'model': MODEL_ID,
                                                   'img': IMG, 'epochs': EPOCHS, 'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__}, indent=1))
    export()
print('끝:', reason, '· epoch 별', [(h['epoch'], round(h['auroc'], 4), round(h['fa_at_recall729'], 3)) for h in history])
"""

REPORT = """print((CKPT / 'dfine_history.json').read_text() if (CKPT / 'dfine_history.json').exists() else '(epoch 이 하나도 안 끝남)')
print('다운로드: /kaggle/working/dfine_stage1_resume.zip — 다음 세션 재개용이자 로컬 판정용 (dfine_best 폴더 포함)')
print('로컬 판정: uv run --extra train python tools/detect_coverage.py --detector-kind dfine --detector <dfine_best> --detector-stage1 --release <3팔>')
"""


def main():
    files = {n: (ROOT / n).read_text(encoding='utf-8') for n in base.BUNDLED}
    setup = SETUP_HEAD + 'FILES = ' + json.dumps(files, ensure_ascii=False) + '\n' + SETUP_TAIL
    cell = base.cell
    nb = {'cells': [cell('markdown', INTRO), cell('code', setup), cell('code', FETCH), cell('code', TRAIN), cell('code', REPORT)],
          'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                       'language_info': {'name': 'python'}},
          'nbformat': 4, 'nbformat_minor': 5}
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'{OUT.relative_to(ROOT)} — 셀 {len(nb["cells"])}개 · {OUT.stat().st_size/1024:.0f}KB')


if __name__ == '__main__':
    main()
