# 런팟(임대 GPU)에서 돌리기

캐글 주간 할당량(30h)이 떨어졌을 때, 또는 T4 16GB 가 답답할 때 씁니다.
**켜져 있는 매 순간 과금**되므로 순서가 중요합니다.

> 이 문서의 전제: 크롭·매니페스트는 이미 만들어져 있고 (`prepare_local.py`),
> 캐글에 Private 데이터셋으로 올라가 있습니다. 원본 다운로드는 한국 PC 에서만
> 됩니다 (AI Hub 가 해외 IP 를 막습니다).

## 0. 켜기 전에 — 뭘 돌릴지 정하고 켜세요

팟이 도는 동안 노트북을 고치면 그 시간도 돈입니다.

| | GPU | 무엇 |
|---|---|---|
| `03g` | ~40분 | 털 가중 샘플러 (기준선 + alpha 2개) |
| `03f` 판 B | ~1h | `convnextv2_base` 1단계 백본 |
| `06` | ~1.5h | 풀 학습 + **holdout** + 보정 + 릴리스 |

⚠️ **`06` 은 반드시 마지막**입니다. holdout 을 여는 유일한 노트북이라,
설정이 다 확정된 뒤에 한 번만 열어야 합니다.

## ⚠️ 먼저 — 런팟은 캐글이 **아닙니다**

| | 캐글 | 런팟 |
|---|---|---|
| 뭘 빌리나 | 노트북 서비스 | **빈 리눅스 서버 한 대** |
| 노트북 | Import 해야 함 | **`git clone` 하면 같이 옵니다** |
| 데이터 | [Add Input] 으로 붙임 | 직접 넣어야 함 |
| 셀 갱신 | ⚠️ `git pull` 해도 **안 바뀜** (다시 Import) | ✅ **파일이 곧 리포라 그냥 바뀝니다** |
| 출력 | 자동 보존 | ❌ **직접 빼내야 함** |

**자동으로 연결되는 건 없습니다.** 팟은 빈 서버라 아래를 순서대로 다 해야
합니다. 대신 노트북을 따로 Import 할 필요는 **없습니다.**

💡 그리고 이 리포의 오래된 함정 하나가 런팟에서는 사라집니다 —
*"노트북 셀은 `git pull` 로 안 바뀝니다"*. 여기선 노트북 파일이 리포 파일
그 자체라 `git pull` 이 진짜로 먹습니다.

## 1. 팟 고르기

| 항목 | 권장 | 왜 |
|---|---|---|
| GPU | RTX 4090 24GB | T4(16GB)에서 배치가 눌렸습니다. 24GB 면 여유 |
| 템플릿 | PyTorch | CUDA 가 맞춰져 나옵니다 |
| 디스크 | 40GB+ | torch 5GB + 크롭 3GB + 체크포인트·릴리스 |
| vCPU | 많을수록 | 학습이 **데이터 로딩에 묶입니다** (실측 1.45배) |

💡 **Community Cloud** 토글이 있으면 확인하세요 — 같은 GPU 가 더 쌀 때가 있습니다.

⚠️ **네트워크 볼륨은 팟을 지워도 계속 과금됩니다.** 출력이 보존돼서 좋지만,
다 끝나면 **볼륨도 따로 지우세요.**

## 2. 셋업 (복붙)

⚠️ **가상환경(`uv sync`)을 만들지 마세요.** 노트북 첫 셀이 `--system` 으로
설치하고 주피터도 시스템 파이썬으로 돕니다. venv 를 따로 만들면 **패키지가
두 곳으로 갈라져서** "터미널에선 되는데 노트북에선 안 되는" 상태가 됩니다.
런팟 PyTorch 템플릿에는 torch 가 이미 있습니다.

```bash
cd /workspace
git clone -b claude/dog-disease-diagnosis-model-1s6jtf \
  https://github.com/gayeoniee/deeplearning_test.git
cd deeplearning_test

# 노트북 첫 셀과 **똑같은 방식**으로 시스템 파이썬에 설치합니다
# ⚠️ --break-system-packages 가 필요합니다. 런팟 이미지의 파이썬은
#    "externally managed"(PEP 668) 라 그냥은 시스템 설치를 거부합니다.
pip install -q uv
python -m uv pip install -q --system --break-system-packages \
  numpy pandas pyarrow Pillow scikit-learn opencv-python-headless tqdm matplotlib \
  timm imagehash grad-cam albumentations kaggle

# torch 는 템플릿에 이미 있습니다. 어느 파이썬인지 확인:
python -c "import sys, torch; print(sys.executable, torch.__version__, torch.cuda.is_available())"

cat >> ~/.bashrc <<'RC'
export DOG_SKIN_WORK=/workspace/data/work
export DOG_SKIN_PERSIST=/workspace/data/work
export NO_ALBUMENTATIONS_UPDATE=1
RC
source ~/.bashrc
mkdir -p "$DOG_SKIN_WORK"
```

`~/.bashrc` 에 넣는 이유: `export` 는 터미널을 새로 열면 사라집니다.
⚠️ **주피터 커널은 `~/.bashrc` 를 안 읽을 수 있습니다.** 노트북에서 경로가
이상하면 첫 셀 **위에** 이 두 줄을 넣으세요:

```python
import os
os.environ["DOG_SKIN_WORK"] = "/workspace/data/work"
os.environ["DOG_SKIN_PERSIST"] = "/workspace/data/work"
```

### 브랜치 — 노트북이 덮어쓰지 않습니다

첫 셀은 **지금 있는 브랜치를 그대로 유지**합니다. (예전엔 `main` 으로
`git reset --hard` 해서 작업 브랜치를 통째로 덮어썼습니다 — 방금 만든 코드가
사라진 채 몇 시간을 돌 뻔했습니다.)
`main` 으로 강제하려면 `export DOG_SKIN_BRANCH=main`.

## 3. 크롭 가져오기 — 집에서 올리지 말고 캐글에서 받으세요

2.6GB 를 가정 회선으로 올리는 것보다 데이터센터에서 받는 게 훨씬 빠릅니다.

### ⚠️ 먼저 읽으세요 — 캐글 CLI 는 **믿지 마세요** (2026-08-26 실측)

런팟에서 캐글 다운로드를 시도하다 **30분 넘게 태웠고 결국 실패했습니다.**
원인은 **캐글이 API 토큰 형식을 바꿨기 때문**입니다:

| | 옛 방식 | 새 방식 |
|---|---|---|
| 받는 법 | `kaggle.json` **파일 다운로드** | 파일 안 줌 (화면에 값만) |
| 키 모양 | 소문자+숫자 **32자** hex | **대문자 4자 + `_` + 소문자·숫자 = 37자** |
| CLI 인식 | `auth_method: LEGACY_API_KEY` | ❓ **확인 못 했습니다** |

새 방식 키를 이 CLI 에 어떻게 물리는지 **우리가 확인하지 못했습니다.**
`kaggle config view` 가 `/root/.config/kaggle` 을 본다고 말해도, 거기에
파일을 만들어도 `Authentication required` 가 계속 났습니다.

> **큰 파일은 `runpodctl` 로 옮기세요.** 아래 3-b 가 그 방법입니다.
> 캐글 CLI 는 토큰이 옛 방식(32자 hex)일 때만 시도할 가치가 있습니다.

### 3-a. (옛 방식 토큰이 있을 때만) 캐글에서 받기

<details><summary>32자 hex 토큰을 갖고 있다면 펼치세요</summary>

1. <https://www.kaggle.com/settings> → **API** → **[Create New Token]**
2. `kaggle.json` 이 다운로드되면 **옛 방식**입니다 (안 받아지면 새 방식 — 위 참고)
3. 팟에서:

```bash
mkdir -p /root/.config/kaggle
printf '{"username":"%s","key":"%s"}\n' 'USER' 'KEY' > /root/.config/kaggle/kaggle.json
chmod 600 /root/.config/kaggle/kaggle.json
kaggle datasets list --mine          # 목록이 나오면 성공
```

⚠️ 환경변수(`KAGGLE_USERNAME`/`KAGGLE_KEY`)보다 **파일이 우선**입니다.
낡은 파일이 남아 있으면 환경변수를 무시합니다 — 먼저 지우세요.
⚠️ 토큰을 새로 만들면 **옛 토큰은 그 자리에서 무효**가 됩니다.

</details>

### 3-b. ★ 권장 — PC 에서 직접 보내기 (`runpodctl`)

브라우저(JupyterLab Upload)는 런팟 프록시를 거쳐서 **느립니다.**
`runpodctl` 은 직접 전송이라 훨씬 빠르고, 인증 싸움이 없습니다.

**① PC 에서 zip 만들기** — 크롭은 4만 개 넘는 낱개 파일이라 그대로는 못 올립니다:

```
cd C:\Users\403\deeplearning_test
uv run python prepare_local.py --package --tags f320 --out dogskin_f320.zip
```

`--tags f320` 이면 1GB, `--tags f320,m2.5` 면 2.6GB 입니다.
`03g` 는 f320 만, `06` 은 둘 다 필요합니다.

**② PC 에서 보내기** — <https://github.com/runpod/runpodctl/releases> 에서
`runpodctl-windows-amd64.exe` 를 받아 zip 폴더에 두고:

```
runpodctl-windows-amd64.exe send dogskin_f320.zip
```

일회용 코드(`8338-galileo-...` 모양)가 뜹니다.

**③ 팟에서 받기**:

```bash
cd /workspace/data/work
runpodctl receive 그코드
unzip -q dogskin_f320.zip && ls crops manifests
```

## 4. ★ preflight — 학습 전에 반드시

```bash
cd /workspace/deeplearning_test
python tools/preflight.py
```

GPU·데이터·분할·영속저장소·디스크·코드 배선을 몇 초에 확인합니다.
`❌` 가 하나라도 있으면 **학습을 시작하지 마세요.** 두 시간짜리가 5분 뒤에
죽는 것보다 10초 만에 죽는 게 훨씬 쌉니다.

## 5. 노트북 돌리기

주피터가 열려 있으면 `notebooks/` 에서 바로 엽니다. 첫 셀(환경 준비)은
런팟에서도 그대로 동작합니다 — `env.detect()` 가 `local` 로 잡고,
`load_prepared()` 는 크롭이 이미 제자리에 있으면 **아무것도 안 합니다.**

각 노트북의 시간 게이트 셀에서 `QUOTA_H` 를 **예산 ÷ 시간당요금**으로 바꾸세요.
(예: $5 예산 · $0.75/hr → `QUOTA_H = 6.6`)

터미널에서 돌리려면:

```bash
cd /workspace/deeplearning_test
python -m jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 notebooks/03g_털가중_샘플러.ipynb \
  --output /workspace/out_03g.ipynb
```

⚠️ 이렇게 돌리면 **중간에 못 멈추고** 로그도 다 끝나야 보입니다. 처음이면
Jupyter Lab 에서 셀을 하나씩 돌리세요 — 6번 셀의 시간 게이트를 보고 판단할
수 있어야 합니다.

## 6. ★ 끄기 전에 — 결과를 빼내세요

캐글은 출력이 자동 보존되지만 **런팟은 아닙니다.**

```bash
cd /workspace
tar czf release.tgz data/work/release data/work/reports
# 주피터 파일 브라우저에서 release.tgz 를 내려받으세요
```

빼낼 것:

* `data/work/release/` — 체크포인트 + `stage1_threshold.json` + `temperature.json`
* `data/work/reports/*.json` — 실측 기록 (`docs/results/` 에 옮겨 적습니다)
* `data/work/reports/hair_index.parquet` — 다시 재지 않으려면

그 다음 **팟 정지 → 팟 삭제 → 네트워크 볼륨 삭제** 순으로 지우세요.
볼륨을 남기면 계속 과금됩니다.

## 7. 받아온 릴리스로 서빙

```bash
python -m uv pip install -q --system fastapi "uvicorn[standard]" python-multipart
python serve.py --release <풀어놓은 폴더>
```

`tests/test_calibration_wiring.py` 가 `temperature.json` 이 실렸는지 감시합니다 —
빠지면 보정 안 된 확률이 보호자에게 갑니다 (에러도 안 나고 조용히 T=1.0 으로
물러섭니다).

## 겪을 수 있는 것

| 증상 | 원인 | 고치기 |
|---|---|---|
| `load_prepared` 가 FileNotFoundError | 크롭이 `work_root` 밖에 있음 | `DOG_SKIN_WORK` 확인, 또는 `DOG_SKIN_PREPARED` 로 위치 지정 |
| preflight 에서 "영속 저장소가 없습니다" | `DOG_SKIN_PERSIST` 미설정 | 위 2단계 `export` 다시 |
| 터미널에선 되는데 노트북에서 ImportError | venv 와 시스템 파이썬이 갈라짐 | `uv sync` 로 만든 `.venv` 를 지우고 `--system` 으로 다시 |
| 노트북이 옛 코드를 씀 | 브랜치가 `main` 으로 리셋됨 | `git branch --show-current` 확인, `DOG_SKIN_BRANCH` 로 지정 |
| `externally managed` 로 설치 거부 | PEP 668 (런팟 이미지) | `--break-system-packages` 추가 |
| `No module named 'pandas'` | 임대 GPU 이미지엔 **torch 만** 있습니다 | 위 설치 목록에 numpy·pandas·sklearn·cv2 가 다 들어 있는지 확인 |
| `import torch` 가 안 됨 | `python` 이 torch 있는 인터프리터가 아님 | `python -c "import sys;print(sys.executable)"` 로 확인 |
| 학습이 추정보다 느림 | 데이터 로딩 병목 | 정상입니다 (실측 1.27~1.55배). vCPU 많은 팟이 유리 |
| 새 터미널에서 경로가 다름 | `export` 가 안 넘어감 | `~/.bashrc` 에 넣기 |
