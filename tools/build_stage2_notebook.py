"""`notebooks/18_2단계_병변보존_crop_파일럿.ipynb` 를 **현재 소스에서** 만듭니다 (STEP 49).

## 왜 생성기인가

이 노트북은 캐글에 8GB 데이터를 다시 안 올리려고 `src/` 스냅샷을 셀 안에
문자열로 담습니다. 그러면 **소스를 고쳐도 노트북은 안 따라옵니다** —
`git pull` 로 안 바뀌는 노트북 셀과 같은 함정입니다 (CLAUDE.md 참고).

그래서 손으로 고치지 말고 이걸 돌리세요:

    uv run python tools/build_stage2_notebook.py

⚠️ **이미 돌린 노트북은 다시 만들지 마세요.** 담긴 스냅샷이 그때 실제로
돈 코드라서, 바꾸면 그 실행을 재현할 수 없게 됩니다
(`protocol.json` 의 `runtime_code_sha256` 와도 어긋납니다).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "18_2단계_병변보존_crop_파일럿.ipynb"

# 캐글로 실어 보낼 스냅샷. 실행기가 roi_data → safe_crop 을 타고 들어갑니다.
# photo_data.py 는 이 프로파일이 안 쓰지만 실행기가 `photo` 분기에서 import 하므로 같이 담습니다.
BUNDLED = [
    "src/__init__.py",
    "src/config.py",
    "src/safe_crop.py",
    "src/original_data.py",
    "src/roi_data.py",
    "src/photo_data.py",
    "tools/kaggle_safe_crop.py",
]

INTRO = """# 18 · 2단계 병변 보존 crop 파일럿 — 정상을 빼고, 병변 6종에서 `fixed` vs `safe`

기존 `safe-crop-pilot` Dataset 연결 → **GPU T4** 선택, Internet ON → Run All.
데이터 재업로드 불필요. **HOURS 는 남은 GPU 할당량에 맞춰** 줄이세요
(지금 값 0.9 는 이번 주 남은 1시간 7분 기준. 다 못 돌면 epoch 경계에서 이어받습니다).

## 왜 2단계인가

1단계에서 `safe`(병변 보존 random ROI)는 **두 번 졌습니다** (STEP 45 · 48). 남은 가설은
*"정상 사진엔 보존할 병변이 없어 두 클래스의 크롭 분포가 갈린다"* 였습니다.
**2단계에는 그 비대칭이 없습니다** — 넘어온 사진은 전부 병변이 있습니다.
그리고 2단계의 알려진 약점이 **위치 교란 하락 32.2%** (STEP 16 holdout) 입니다.
그래서 같은 패키지에서 **A7 을 빼고** 병변 6종만으로 같은 비교를 합니다.
사전등록: [`docs/results/STEP49_2단계_병변보존_crop_사전등록.md`](../docs/results/STEP49_2단계_병변보존_crop_사전등록.md)

## 두 팔 — 다른 건 crop 하나뿐

- **`fixed`**(대조군): 병변 bbox 를 5% 여유로 감싼 정사각 ROI(최소 320px), 가운데
- **`safe`**: 같은 ROI 를 ×1.0~1.25 키우고 **병변을 다 담는 범위 안에서** 위치를 무작위로
- 검증은 **둘 다 고정 ROI**. 같은 EfficientNetV2-S 초기값 · 384px · 6클래스 head ·
  train 10,177 / val 2,040 · 최대 5 epoch · class-weighted CE

⚠️ 배포 백본(convnextv2_base)이 아닙니다 — *"확대 실험을 할 가치가 있나"* 까지만 답합니다.

## 판정에 쓸 것 — 돌리기 전에 정했습니다

매 epoch **검증 2,040장 전부**를 두 번 봅니다: **clean**(고정 ROI) 과
**shift**(같은 ROI 를 변 길이의 20% 만큼 오른쪽·아래로, STEP 47 과 같은 정의).

    주 지표: shift 하락률 = (clean − shift) / clean macro-F1
    하한선: fixed 팔 자신의 하락률 (같은 학습·같은 val)
    후보:   safe 의 하락률이 fixed 보다 5%p 이상 작다  (교란 잡음 ±5%p)
    관문:   clean macro-F1 −0.02 이내 · 계열 4군 정확도 −0.01 이내 · A6 recall −0.03 이내

⚠️ **holdout 은 안 엽니다.** 같은 epoch 끼리만 비교합니다. 단일 seed 예비 실험입니다.
⚠️ 커버리지는 1단계 확률이 있어야 재는 값이라 **여기서는 못 잽니다** — 계열 4군 정확도로 대신합니다.

결과: `/kaggle/working/stage2_crop_pilot_resume.zip` (1단계 실험 출력과 별도).
재개: 그 ZIP 을 **비공개** Dataset 으로 연결하고 같은 노트북을 실행하세요.
`EPOCHS`·`BATCH_SIZE` 는 유지합니다."""

SETUP_HEAD = """from pathlib import Path
import json, subprocess, sys, os, zipfile
HOURS = 0.9  # 이번 세션 예산. 실제 남은 GPU 시간(1h 07m)보다 여유 있게 작게 설정
EPOCHS = 5  # 재개할 때 변경하지 않기
BATCH_SIZE = 16
WORKERS = 2
CODE = Path('/kaggle/working/stage2_code')
OUT = Path('/kaggle/working/stage2_crop_pilot')
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
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
ENV['NO_ALBUMENTATIONS_UPDATE'] = '1'
try:
    import timm
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'timm==1.0.29'], check=True)
# 검증한 버전으로 고정. 이미 설치된 구버전도 교체하고 학습은 새 subprocess에서 시작.
subprocess.run([sys.executable, '-m', 'pip', 'install', 'albumentations==2.0.8'], check=True)
# 실제 사용할 모델의 GPU forward/backward 를 먼저 확인.
probe = \"\"\"import torch
from tools.kaggle_safe_crop import make_model
assert torch.cuda.is_available(), 'GPU T4를 선택하세요'
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), flush=True)
m = make_model(False, profile='stage2').cuda().train()
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
# 2단계 실험 결과만 재개. 1단계(roi/photo) ZIP 은 쓰지 않습니다.
resumes = list(Path('/kaggle/input').rglob('stage2_crop_pilot_resume.zip'))
if not OUT.exists() and resumes:
    assert len(resumes) == 1, '재개 ZIP 하나만 연결하세요'
    with zipfile.ZipFile(resumes[0]) as z:
        for name in z.namelist():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
        assert json.loads(z.read('protocol.json')).get('profile') == 'stage2'
        z.extractall(OUT)
print('Dataset:', DATA, 'Output:', OUT)
"""

RUN = """command = [sys.executable, '-u', str(CODE / 'tools/kaggle_safe_crop.py'),
           '--profile', 'stage2', '--data', str(DATA), '--out', str(OUT),
           '--hours', str(HOURS), '--epochs', str(EPOCHS),
           '--batch-size', str(BATCH_SIZE), '--workers', str(WORKERS)]
subprocess.run(command, check=True, cwd=CODE, env=ENV)
"""

REPORT = """import pandas as pd
print((OUT / 'comparison.json').read_text())
history = pd.read_csv(OUT / 'history.csv')
# clean 만 보면 crop 의 값어치를 못 봅니다 — 위치 교란(shift) 열을 같이 봅니다.
columns = [c for c in ['mode', 'epoch', 'macro_f1', 'group4_accuracy', 'A6_recall', 'A4_recall',
                       'shift_macro_f1', 'shift_drop_rel', 'shift_group4_accuracy',
                       'shift_A6_recall', 'elapsed_sec'] if c in history.columns]
display(history[columns])
print('다운로드: /kaggle/working/stage2_crop_pilot_resume.zip')
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
