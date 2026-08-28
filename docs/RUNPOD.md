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

## 3. 크롭 가져오기

세 갈래입니다. **`runpodctl` 이 제일 단순합니다** — 계정 연동도, 토큰도 없습니다.

### 3-A. runpodctl — 내 PC 에서 팟으로 직접 (권장)

주피터 파일 업로드보다 훨씬 빠릅니다 (peer-to-peer).

**내 PC 에 받기** (한 번만) — <https://github.com/runpod/runpodctl/releases> 에서
`runpodctl-windows-amd64.exe` (맥은 `runpodctl-darwin-arm64`) 를 받아
zip 이 있는 폴더에 두고 이름을 `runpodctl.exe` 로 바꿉니다.

⚠️ 윈도우에서 그 exe 를 **더블클릭하면 창이 떴다 사라집니다.** CLI 라서요.
   반드시 cmd/PowerShell 에서 실행하세요.

```
runpodctl.exe send dogskin_prepared.zip
```

이런 코드가 뜹니다:

```
Code is: 8471-purple-tiger-mango
On the other computer run
runpodctl receive 8471-purple-tiger-mango
```

**팟 터미널에서** (runpodctl 은 런팟 이미지에 이미 깔려 있습니다):

```bash
cd /workspace
runpodctl receive 8471-purple-tiger-mango
```

⚠️ 파일은 **`receive` 를 실행한 폴더**에 떨어집니다. `cd` 를 먼저 하세요.
⚠️ 전송이 **끝난 뒤에** `mv` 하세요. 아직 도착 안 한 파일을 옮기려다
   "그런 파일 없음" 으로 헤맸습니다.
⚠️ 코드는 일회용입니다. 끊기면 `send` 부터 다시.

### 3-B. 구글 드라이브 → 팟

zip 이 이미 드라이브에 있으면 팟이 **데이터센터 속도로** 받습니다
(내 회선 업로드를 안 씁니다). 파일을 "링크가 있는 모든 사용자" 로 공유하고
링크의 `/d/<아이디>/` 부분을 씁니다:

```bash
pip install gdown --break-system-packages
cd /workspace
gdown <아이디>
```

⚠️ 대용량은 드라이브 쪽 다운로드 쿼터에 걸릴 수 있습니다. 그러면 3-A 로.

### 3-C. 캐글에서 받기

이미 캐글에 데이터셋을 올려둔 경우에만. 토큰 형식이 바뀌어서 막힐 수 있습니다.


#### 토큰 받기 (내 PC 에서, 한 번만)

1. <https://www.kaggle.com/settings> 접속 (우측 상단 프로필 → **Settings**)
2. **API** 항목 → **[Create New Token]**
3. `kaggle.json` 이 다운로드됩니다. 열어보면 이렇게 생겼습니다:

```json
{"username":"내캐글아이디","key":"0123456789abcdef0123456789abcdef"}
```

⚠️ **계정 이름은 이 파일의 `username`** 입니다. 깃허브 아이디와 다를 수 있어요.
⚠️ 이전에 만든 토큰이 있으면 새로 만드는 순간 **옛 토큰이 무효**가 됩니다.

#### 팟에 넣기 — 환경변수가 제일 쉽습니다

`nano` 로 JSON 을 붙여넣다 따옴표가 깨지는 일이 흔해서, **파일 없이** 갑니다:

```bash
export KAGGLE_USERNAME=내캐글아이디
export KAGGLE_KEY=0123456789abcdef0123456789abcdef
```

캐글 CLI 가 이 두 변수를 그대로 읽습니다. `chmod` 도, 파일도 필요 없습니다.

<details><summary>파일로 넣고 싶다면</summary>

```bash
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json <<'JSON'
{"username":"내캐글아이디","key":"0123456789abcdef0123456789abcdef"}
JSON
chmod 600 ~/.kaggle/kaggle.json
```

`<<'JSON'` 의 **따옴표가 중요합니다** — 없으면 셸이 내용을 건드립니다.
</details>

⚠️ 토큰을 **노트북 셀에 붙여넣지 마세요.** 셀은 출력과 함께 저장돼서 그대로
남습니다. `.gitignore` 가 `*apikey*` / `.env` 를 막고 있지만, 애초에 리포
안에 두지 않는 게 확실합니다.

#### 데이터셋 이름 확인 — **추측하지 말고 물어보세요**

```bash
kaggle datasets list --mine
```

내가 올린 데이터셋이 `ref` 열에 `아이디/이름` 형태로 그대로 나옵니다.
거기 나온 값을 그대로 복사해서 씁니다:

```bash
cd "$DOG_SKIN_WORK"
kaggle datasets download -d 아이디/dogskin-f320 --unzip
# 06 까지 갈 거면 2단계 크롭도:
kaggle datasets download -d 아이디/dogskin-m25 --unzip
```

받고 나서 확인:

```bash
ls "$DOG_SKIN_WORK/crops"        # f320 (m2.5) 가 보여야 합니다
ls "$DOG_SKIN_WORK/manifests"    # manifest_final.parquet
```

받고 나면 이런 모양이어야 합니다:

```
/workspace/data/work/crops/f320/...
/workspace/data/work/manifests/manifest_final.parquet
```

⚠️ **`kaggle.json` 은 절대 커밋하지 마세요.** `.gitignore` 가 막고 있지만
리포 바깥(`~/.kaggle/`)에 두는 게 확실합니다.
⚠️ AI Hub 데이터는 **재배포 금지**입니다. 볼륨을 공개로 두지 마세요.

### 3-D. ★ zip 풀기 — `unzip` 말고 `fetch_crops.py`

zip 이 팟에 도착했으면 **`unzip` 을 쓰지 마세요.** 세 가지가 조용히 틀립니다.

| `unzip -o/-n` | `tools/fetch_crops.py` |
|---|---|
| zip 안의 **423,080장 전부** 풉니다 | 매니페스트가 쓰는 **365,428장만** |
| 디스크가 차서 죽으면 **쓰다 만 파일**이 남고 `-n` 은 그걸 "있음" 으로 건너뜁니다 | zip 이 적어둔 **크기와 대조**해 다르면 다시 꺼냅니다 |
| 한 줄로 돌리면 ~70장/초 | 스레드 32개 = **1,000장/초 이상** |

```bash
cd /workspace/deeplearning_test
export DOG_SKIN_WORK=/workspace/deeplearning_test/data/work

# 매니페스트가 먼저 있어야 합니다 (있어야 무엇이 필요한지 압니다)
unzip -o /root/dogskin_prepared.zip "manifests/*" -d "$DOG_SKIN_WORK"

python tools/fetch_crops.py /root/dogskin_prepared.zip --tags f320,m2.5
```

미리보기가 "꺼낼 것 / 안 쓰는 것 / 순수 필요 공간" 을 GB 로 알려줍니다. 그다음:

```bash
python tools/fetch_crops.py /root/dogskin_prepared.zip --tags f320,m2.5 \
  --apply --prune --workers 32
```

`--prune` 은 매니페스트가 **안 쓰는** 크롭을 지웁니다 (태그당 57,652장).
공간이 빠듯하면 이걸로 먼저 자리를 만들고 꺼냅니다.

끝나면 스스로 다시 세어 `f320 365,428/365,428 (100.00%)` 를 찍습니다.
100% 가 아니면 **멈추고** 볼륨을 의심하라고 말합니다 — 여기서 통과해야 06 을 엽니다.

⚠️ **`Exit 50` 은 디스크가 찼다는 뜻입니다.** 네트워크 볼륨 할당량은 `df` 에
   **안 보입니다** (MooseFS 라 클러스터 전체 용량인 2.0P 가 찍힙니다). 런팟 콘솔
   Storage 탭에서 크기를 보고, 모자라면 볼륨을 키우세요.

⚠️ zip 을 `/root` 에 두면 컨테이너 디스크(30GB)를 먹습니다. 팟이 꺼지면 사라지니
   **다 풀고 나면 지우세요** — 볼륨(`/workspace`)에 두면 그만큼 크롭 자리가 줍니다.

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
