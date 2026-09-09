"""`notebooks/16_photometric_복원_비교.ipynb` 를 **현재 소스에서** 만듭니다.

## 왜 생성기인가

이 노트북은 캐글에 8GB 데이터를 다시 안 올리려고 `src/` 스냅샷을 셀 안에
문자열로 담습니다. 그러면 **소스를 고쳐도 노트북은 안 따라옵니다** —
`git pull` 로 안 바뀌는 노트북 셀과 같은 함정입니다 (CLAUDE.md 참고).

그래서 손으로 고치지 말고 이걸 돌리세요:

    uv run python tools/build_photo_notebook.py

⚠️ **이미 돌린 노트북은 다시 만들지 마세요.** 담긴 스냅샷이 그때 실제로
돈 코드라서, 바꾸면 그 실행을 재현할 수 없게 됩니다
(`protocol.json` 의 `runtime_code_sha256` 와도 어긋납니다).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "16_photometric_복원_비교.ipynb"

# 캐글로 실어 보낼 스냅샷. photo_data.py 가 roi_data → safe_crop 을 타고 들어갑니다.
BUNDLED = [
    "src/__init__.py",
    "src/config.py",
    "src/safe_crop.py",
    "src/original_data.py",
    "src/roi_data.py",
    "src/photo_data.py",
    "tools/kaggle_safe_crop.py",
]

INTRO = """# 16 · `photometric` 복원 확인 — 고정 ROI 를 고정해 두고 증강만 바꿉니다

기존 `safe-crop-pilot` Dataset 연결 → **GPU T4** 선택, Internet ON → Run All.
데이터 재업로드 불필요. **HOURS 는 남은 GPU 할당량에 맞춰** 줄이세요.

## 왜 이걸 먼저 하나

STEP 44·45 파일럿에는 채택 레시피의 **`photometric` 이 빠져 있었습니다**
(backbone LR 배율·warmup·EMA 도). 그래서 *"보존 crop 이 졌다"* 가 crop 탓인지
레시피 탓인지 **안 갈립니다**. 새 기법을 찾기 전에 **이미 효과가 확인된 것이
지금 조건에서도 사는지**부터 봅니다 → [`docs/experiment_history_audit.md`](../docs/experiment_history_audit.md)

## 바꾸는 것은 하나뿐입니다

- **두 팔 다 고정 ROI** (`fixed` / `photo`). crop 은 완전히 같습니다
- `photo` 팔만 letterbox 뒤에 과거 `photometric` 연산자를 겁니다 —
  CLAHE .2 / blur .3 / noise .25 / JPEG .3 (`src.data` 프리셋과 **같은 값**)
- 같은 EfficientNetV2-S 사전학습 초기값 · 384px · train 20,000 / val 4,000 · 최대 5 epoch

⚠️ **LR·EMA·기간은 안 건드립니다.** 같이 바꾸면 무엇의 효과인지 못 가릅니다.

## 판정에 쓸 것 — 돌리기 전에 정합니다

`photometric` 의 원래 효과는 **화질 지름길 차단**이었습니다 (STEP 6:
흐림 교란 하락 −38.5%p, AUROC 는 그대로). 그러니 clean 점수만 보면 안 됩니다.

매 epoch **검증 4,000장 전부**에 대해 clean 과 **흐림 교란**(가우시안 kernel 13,
sigma 1·2)을 같이 재서 `history.csv` 와 `comparison.json` 에 남깁니다.

    기대: clean 은 비슷하거나 조금 낮고, blur1/blur2 에서 photo 가 앞선다

⚠️ **holdout 은 안 엽니다.** 같은 epoch 끼리만 비교합니다.
⚠️ 단일 seed 예비 실험입니다 — 작은 차이를 확정 개선으로 읽지 않습니다.

결과: `/kaggle/working/photo_crop_pilot_resume.zip` (ROI·safe 실험 출력과 별도).
재개: 그 ZIP 을 **비공개** Dataset 으로 연결하고 같은 노트북을 실행하세요.
`EPOCHS`·`BATCH_SIZE` 는 유지합니다."""

SETUP_HEAD = """from pathlib import Path
import json, subprocess, sys, os, zipfile
HOURS = 3.0  # 이번 세션 예산. 실제 남은 GPU 시간보다 여유 있게 작게 설정
EPOCHS = 5  # 재개할 때 변경하지 않기
BATCH_SIZE = 16
WORKERS = 2
CODE = Path('/kaggle/working/photo_code')
OUT = Path('/kaggle/working/photo_crop_pilot')
"""

SETUP_TAIL = """
for name, source in FILES.items():
    path = CODE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
# 이전 실행과 동일한 Pillow 복구 설정: subprocess 와 worker 모두 적용.
(CODE / 'sitecustomize.py').write_text('from PIL import ImageFile\\nImageFile.LOAD_TRUNCATED_IMAGES = True\\n')
ENV = dict(os.environ)
ENV['PYTHONPATH'] = str(CODE)
for package in ['timm==1.0.29', 'albumentations']:   # photo 팔이 albumentations 를 씁니다
    module = package.split('==')[0]
    try:
        __import__(module)
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', package], check=True)
# 실제 사용할 모델의 GPU forward/backward 를 먼저 확인.
probe = \"\"\"import torch
from tools.kaggle_safe_crop import make_model
assert torch.cuda.is_available(), 'GPU T4를 선택하세요'
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), flush=True)
m = make_model(False, profile='photo').cuda().train()
with torch.autocast('cuda'):
    loss = m(torch.randn(2, 3, 64, 64, device='cuda')).float().square().mean()
loss.backward()
torch.cuda.synchronize()
print('GPU probe passed', flush=True)
\"\"\"
subprocess.run([sys.executable, '-c', probe], env=ENV, cwd=CODE, check=True)
"""

FIND_DATA = """candidates = list(Path('/kaggle/input').rglob('pilot_manifest.parquet'))
assert len(candidates) == 1, f'기존 safe-crop-pilot Dataset 하나를 연결하세요: {candidates}'
DATA = candidates[0].parent
# photometric 실험 결과만 재개. roi/safe 쪽 ZIP 은 쓰지 않습니다.
resumes = list(Path('/kaggle/input').rglob('photo_crop_pilot_resume.zip'))
if not OUT.exists() and resumes:
    assert len(resumes) == 1, '재개 ZIP 하나만 연결하세요'
    with zipfile.ZipFile(resumes[0]) as z:
        for name in z.namelist():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
        assert json.loads(z.read('protocol.json')).get('profile') == 'photo'
        z.extractall(OUT)
print('Dataset:', DATA, 'Output:', OUT)
"""

RUN = """command = [sys.executable, '-u', str(CODE / 'tools/kaggle_safe_crop.py'),
           '--profile', 'photo', '--data', str(DATA), '--out', str(OUT),
           '--hours', str(HOURS), '--epochs', str(EPOCHS),
           '--batch-size', str(BATCH_SIZE), '--workers', str(WORKERS)]
subprocess.run(command, check=True, cwd=CODE, env=ENV)
"""

REPORT = """import pandas as pd
print((OUT / 'comparison.json').read_text())
history = pd.read_csv(OUT / 'history.csv')
# clean 만 보면 photometric 의 값어치를 못 봅니다 — 흐림 열을 같이 봅니다.
columns = [c for c in ['mode', 'epoch', 'macro_f1', 'auroc',
                       'blur1_macro_f1', 'blur1_auroc',
                       'blur2_macro_f1', 'blur2_auroc'] if c in history.columns]
display(history[columns])
print('다운로드: /kaggle/working/photo_crop_pilot_resume.zip')
"""


def cell(kind: str, text: str) -> dict:
    source = text.splitlines(keepends=True)
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": source}
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}


def main() -> None:
    missing = [n for n in BUNDLED if not (ROOT / n).exists()]
    if missing:                       # 조용히 반쪽짜리 번들을 만들지 않습니다
        raise SystemExit(f"소스가 없습니다: {missing}")
    files = {name: (ROOT / name).read_text(encoding="utf-8") for name in BUNDLED}
    setup = SETUP_HEAD + "FILES = " + json.dumps(files, ensure_ascii=False) + "\n" + SETUP_TAIL

    notebook = {
        "cells": [cell("markdown", INTRO), cell("code", setup),
                  cell("code", FIND_DATA), cell("code", RUN), cell("code", REPORT)],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    size = OUT.stat().st_size / 1024
    print(f"{OUT.relative_to(ROOT)} — 셀 {len(notebook['cells'])}개 · {size:.0f}KB "
          f"· 담은 소스 {len(files)}개")


if __name__ == "__main__":
    main()
