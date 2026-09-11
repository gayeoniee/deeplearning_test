"""`notebooks/18b_2단계_병변보존_crop_파일럿_colab.ipynb` — 노트북 18 의 **코랩판** (STEP 49).

캐글 주간 할당량이 바닥났을 때 씁니다. 데이터를 다시 올리지 않습니다 —
캐글에 이미 있는 두 Dataset(파일럿 8.8GB · 재개 폴더)을 **캐글 API 로 코랩에 받아옵니다.**
소스 스냅샷·실행기·판정은 노트북 18 과 완전히 같습니다 (같은 생성기 상수를 씁니다).

    uv run python tools/build_stage2_colab_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import build_stage2_notebook as base  # noqa: E402

ROOT = base.ROOT
OUT = ROOT / "notebooks" / "18b_2단계_병변보존_crop_파일럿_colab.ipynb"

INTRO = """# 18b · 노트북 18 의 코랩판 — 캐글 할당량이 없을 때

**런타임 → T4 GPU** 선택 → 아래 두 슬러그 확인 → Run All.
소스·실행기·판정은 [노트북 18](18_2단계_병변보존_crop_파일럿.ipynb) 과 같습니다 — 사전등록은
[`STEP49`](../docs/results/STEP49_2단계_병변보존_crop_사전등록.md).

## 준비물 — 캐글 토큰 (파일 아님)

캐글 → Settings → API → **Create New Token** 에서 보이는 **문자열**을 복사해 두세요.
세 번째 셀이 **항상** 로그인 창을 띄웁니다 — 캐글 사용자명과 그 문자열을 붙여 넣습니다
(코랩 Secrets 의 `KAGGLE_*` 는 무시합니다 — 낡은 값이 조용히 쓰여 403 이 난 적이 있습니다).
토큰은 노트북에 **적지 않습니다.**

## 무엇을 받나

- `PILOT_DATASET` — 파일럿 데이터 (`pilot_manifest.parquet` 가 든 것, 8.8GB). 코랩 디스크로 바로 받습니다
- `RESUME_DATASET` — 이어 돌릴 재개 폴더(캐글이 ZIP 을 풀어 올린 그 Dataset). **비워 두면 처음부터** 돕니다

⚠️ 코랩 무료는 세션이 예고 없이 끊길 수 있습니다. 끝나면 마지막 셀이 재개 ZIP 을 **내 PC 로 내려받게** 합니다 —
그걸 다시 캐글 Dataset 으로 올리거나 다음 코랩 세션에 `files.upload()` 로 넣으면 이어집니다."""

SETUP_HEAD = base.SETUP_HEAD.replace(
    "CODE = Path('/kaggle/working/stage2_code')", "CODE = Path('/content/stage2_code')"
).replace(
    "OUT = Path('/kaggle/working/stage2_crop_pilot')", "OUT = Path('/content/stage2_crop_pilot')"
).replace(
    "HOURS = 0.9  # 이번 세션 예산. 실제 남은 GPU 시간(1h 07m)보다 여유 있게 작게 설정",
    "HOURS = 1.0  # 코랩 무료 세션이 끊기기 전에 끝나도록. 남은 2 epoch 는 약 25분",
)
assert SETUP_HEAD != base.SETUP_HEAD

FETCH = """PILOT_DATASET = 'gayoniee/safe-crop-pilot'   # 캐글 Dataset 슬러그 (계정/이름)
RESUME_DATASET = ''                              # 예: 'gayoniee/stage2-crop-pilot-resume'. 비우면 처음부터
import os, shutil, json, zipfile, subprocess, sys
from pathlib import Path
# 캐글 새 API 토큰은 kagglehub >= 0.4.1 부터 — 코랩 기본 버전이 낡으면 "validated" 뒤에 403 이 납니다.
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U', 'kagglehub>=0.4.1'], check=True)
import kagglehub
print('kagglehub', kagglehub.__version__)
# 로그인 창이 뜨면 캐글 사용자명과 토큰(캐글 → Settings → API → Create New Token 에서 보이는 문자열)을 붙여 넣으세요.
# 코랩 Secrets 의 낡은 KAGGLE_* 값이 조용히 쓰여 403 이 났던 적이 있어, 환경변수를 지우고 **항상** 창을 띄웁니다.
for key in ['KAGGLE_USERNAME', 'KAGGLE_KEY', 'KAGGLE_API_TOKEN']:
    os.environ.pop(key, None)
kagglehub.login()
DATA_ROOT = Path(kagglehub.dataset_download(PILOT_DATASET))
candidates = list(DATA_ROOT.rglob('pilot_manifest.parquet'))
assert len(candidates) == 1, candidates
DATA = candidates[0].parent
if RESUME_DATASET and not OUT.exists():
    RES = Path(kagglehub.dataset_download(RESUME_DATASET))
    zips = list(RES.rglob('stage2_crop_pilot_resume*.zip'))
    if zips:
        with zipfile.ZipFile(zips[0]) as z:
            for name in z.namelist():
                assert not Path(name).is_absolute() and '..' not in Path(name).parts
            z.extractall(OUT)
    else:
        found = [p.parent for p in RES.rglob('protocol.json') if json.loads(p.read_text()).get('profile') == 'stage2']
        assert found, '재개 폴더를 못 찾았습니다'
        # 출력 폴더와 풀린 ZIP 이 같이 올라와 둘일 수 있습니다 — 끝난 epoch 가 가장 많은 쪽.
        shutil.copytree(max(found, key=lambda d: len(list(d.glob('*_epoch*_val.npz')))), OUT)
if OUT.exists():
    done = sorted(p.name for p in OUT.glob('*_epoch*_val.npz'))
    print('재개:', OUT, '— 끝난 epoch 파일', done)
    assert done, '재개 폴더에 epoch 결과가 없습니다 — 잘못된 입력'
else:
    print('⚠️ 재개 없음 — 처음부터 돕니다. 이어서 돌리려던 거면 지금 멈추고 RESUME_DATASET 을 확인하세요')
print('Dataset:', DATA, 'Output:', OUT)
"""

RUN = base.RUN
REPORT = base.REPORT.replace(
    "print('다운로드: /kaggle/working/stage2_crop_pilot_resume.zip')",
    "from google.colab import files\nfiles.download('/content/stage2_crop_pilot_resume.zip')  # 내 PC 로 — 세션이 끊기면 사라집니다",
)
assert REPORT != base.REPORT


def main() -> None:
    files = {name: (ROOT / name).read_text(encoding="utf-8") for name in base.BUNDLED}
    setup = SETUP_HEAD + "FILES = " + json.dumps(files, ensure_ascii=False) + "\n" + base.SETUP_TAIL
    cell = base.cell
    notebook = {
        "cells": [cell("markdown", INTRO), cell("code", setup),
                  cell("code", FETCH), cell("code", RUN), cell("code", REPORT)],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}, "accelerator": "GPU"},
        "nbformat": 4, "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{OUT.relative_to(ROOT)} — 셀 {len(notebook['cells'])}개 · {OUT.stat().st_size/1024:.0f}KB")


if __name__ == "__main__":
    main()
