# ⚠️ Colab / Kaggle 실전 트러블슈팅

> **읽는 시점**: 학습이 안 돌거나, 죽거나, 느릴 때.

---

## Colab vs Kaggle — 언제 뭘 쓰나

| | Colab 무료 | Colab Pro | Kaggle 무료 |
|---|---|---|---|
| GPU | T4 16GB (가변) | T4/L4/A100 | **T4 ×2** 또는 P100 |
| 세션 한도 | ~4시간, 자주 끊김 | ~24시간 | **12시간** (안정적) |
| 주간 한도 | 불규칙 | 컴퓨팅 단위 | **주 30시간** (명확) |
| 디스크 | ~100GB (휘발) | ~200GB | ~70GB |
| 데이터 영속 | Drive 연동 | Drive 연동 | Dataset 업로드 |
| 인터넷 | 항상 | 항상 | **켜야 함** (설정에서) |

**권장**: 탐색·전처리는 Colab, 긴 학습은 Kaggle.
Kaggle 은 12시간 연속 + 세션 끊김이 적어서 학습에 유리합니다.

`src/env.py` 가 환경을 자동 감지하니 같은 코드가 양쪽에서 돕니다.

---

## 자주 만나는 문제들

### 1. `CUDA out of memory`

가장 흔합니다. 순서대로 시도하세요:

```python
# ① 배치 줄이고 누적으로 보상 (실효 배치는 유지)
cfg = CFG(batch_size=8, grad_accum=8)   # 실효 64

# ② 해상도 낮추기 — 메모리는 해상도의 제곱에 비례
cfg = CFG(img_size=224)   # 384 → 224 면 메모리 1/3

# ③ 더 작은 모델
"convnextv2_tiny" 대신 "convnextv2_base"

# ④ AMP 확인 (기본 켜져 있음)
cfg = CFG(amp=True)
```

**셀을 다시 돌리기 전에 메모리를 비우세요:**
```python
import torch, gc
del model, dl_tr, dl_va
gc.collect(); torch.cuda.empty_cache()
```

> 💡 주피터는 셀 실행 후에도 변수를 붙들고 있습니다.
> OOM 후 배치만 줄이고 재실행하면 이전 모델이 아직 VRAM에 있어서 또 터집니다.

### 2. 세션이 끊겨서 학습이 날아감

**대비책 (이 프로젝트에 이미 들어 있음):**
- 매 에폭 best 체크포인트 저장 → `work/checkpoints/{exp}/best.pt`
- `history.csv` 에 에폭별 기록 append

**추가로 하면 좋은 것:**
```python
# Colab: 체크포인트를 Drive 로
DRIVE = env.mount_drive()
import shutil
shutil.copy(res.best_ckpt, DRIVE/"dogskin/best.pt")
```

Colab 무료는 유휴 상태로 두면 끊깁니다. 브라우저 탭을 열어두세요.

### 3. `DataLoader worker killed` / RAM 부족

```python
cfg = CFG(num_workers=2)     # Colab 은 2가 안전, Kaggle 은 4까지
```

`num_workers` 를 크게 잡으면 각 워커가 메모리를 복제해서 RAM 이 터집니다.
Colab 무료는 RAM 12GB 라 4 이상은 위험합니다.

### 4. 학습이 너무 느림

**체크 순서:**

```python
env.describe()      # GPU 가 잡혔는지 먼저 확인
```

`GPU: 없음` 이면 → 런타임 → 런타임 유형 변경 → **T4 GPU**

GPU 가 있는데도 느리면:

| 원인 | 확인 | 해결 |
|---|---|---|
| Drive 에서 직접 읽기 | 경로가 `/content/drive/...` | 로컬 디스크로 복사 후 학습 |
| 작은 파일 수십만 개 | 이미지 개수 | 크롭본으로 학습 (이미 512px 로 줄임) |
| AMP 꺼짐 | `cfg.amp` | `True` 로 |
| num_workers=0 | | 2로 |

> ⚠️ **Google Drive 에서 직접 이미지를 읽으면 10배 이상 느립니다.**
> Drive 는 네트워크 마운트라 작은 파일 수십만 개에 최악입니다.
> 반드시 `/content` 로 복사(또는 zip 째 복사 후 압축 해제)하고 학습하세요.

### 5. `timm` 모델 이름이 없다는 에러

timm 버전마다 이름이 바뀝니다. 이 프로젝트는 자동 fallback 이 있지만:

```python
from src import models
models.available()      # 이 환경에서 실제로 되는 것만 표시

import timm
timm.list_models("*convnext*", pretrained=True)   # 직접 찾기
```

### 6. 사전학습 가중치 다운로드 실패

```
HTTPError / ConnectionError while downloading
```

Kaggle 은 **인터넷이 기본 꺼져 있습니다.** 우측 설정 패널 → Internet → On.
(전화번호 인증이 필요할 수 있습니다)

계속 실패하면 캐시를 지우고 재시도:
```python
!rm -rf ~/.cache/huggingface ~/.cache/torch
```

### 7. 한글이 네모(□)로 나옴

matplotlib 기본 폰트에 한글이 없습니다.

```python
!apt-get install -y fonts-nanum -qq
import matplotlib.font_manager as fm, matplotlib.pyplot as plt
fm.fontManager.addfont("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False
```

### 8. 디스크 가득 참

```python
print(env.free_disk_gb())
```

- 원본 압축 파일 삭제 (`aihub.unpack_all(remove_archives=True)` 가 자동 처리)
- 안 쓰는 크롭 태그 삭제: `shutil.rmtree(env.work_root()/"crops"/"m2.5")`
- `aihub.download(..., max_gb=...)` 로 나눠 받기

---

## 시간 예산 짜기

Colab 무료 T4 기준 대략적인 감:

| 작업 | 데이터 5만 장 기준 |
|---|---|
| 다운로드 40GB | 30~60분 |
| 스캔 | 5~15분 |
| phash 중복 제거 | 20~40분 |
| 크롭 생성 (1종) | 20~40분 |
| ResNet50 학습 10에폭 | 40~80분 |
| ConvNeXt-V2 base 15에폭 | 2~4시간 |
| EVA-02 base 15에폭 | 4~8시간 (T4 에선 버거움) |

**전략**: 전처리는 한 번만 하고 결과를 Drive 에 백업.
그 다음부터는 학습만 반복하면 됩니다.

무거운 모델은 Kaggle 12시간 세션에서 돌리세요.

---

## 체크리스트

세션 시작할 때마다:

- [ ] `env.describe()` 로 GPU 확인
- [ ] `env.free_disk_gb()` 로 디스크 확인
- [ ] 크롭본이 로컬 디스크에 있는지 (Drive 직독 금지)
- [ ] 이전 실험 변수를 지웠는지 (`gc.collect()`)
- [ ] 긴 학습이면 체크포인트를 Drive 로 백업하는 셀을 넣었는지

---

**다음**: [`05_실험기록과_재현성.md`](05_실험기록과_재현성.md)
