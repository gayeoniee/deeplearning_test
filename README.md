# 🐕 반려견 피부질환 스크리닝 보조 모델

AI Hub 「반려동물 피부 질환 데이터」(dataSetSn=561)로 **반려견 피부 병변 6종**을
분류하는 딥러닝 파이프라인.

> ⚠️ **이 프로젝트는 수의학적 진단 도구가 아닙니다.**
> 보호자에게 "이런 소견이 의심되니 병원에 가보세요" 수준의 안내만 제공하는
> 스크리닝 보조 기능입니다. 수의사의 진료를 대체하지 않습니다.

---

## 빠른 시작

### 1. AI Hub 데이터 신청 (먼저 하세요 — 승인에 ~1영업일)

1. [AI Hub](https://aihub.or.kr) 회원가입
2. [561번 데이터셋](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=561) → **활용신청**
3. 승인 후 마이페이지 → **API Key 발급**

### 2. ⚠️ 데이터는 한국 PC 에서 받아야 합니다

**AI Hub 는 해외 IP 다운로드를 차단합니다.** Colab/Kaggle VM 은 한국 밖이라
다운로드가 502 로 실패합니다 (목록 조회는 되는데 다운로드만 막혀서 헷갈립니다).

그래서 역할을 나눕니다 — **한국 PC 에서 다운로드+전처리, 클라우드에서 학습**:

```bash
# 내 컴퓨터(한국)에서. GPU 불필요, CPU 만 있으면 됩니다
git clone https://github.com/gayeoniee/deeplearning_test.git
cd deeplearning_test
pip install -r requirements.txt
export AIHUB_API_KEY="발급받은키"
python prepare_local.py --all
```

<details><summary><b>Windows 는 명령이 조금 다릅니다</b> (클릭)</summary>

```cmd
py -m pip install -r requirements.txt
set AIHUB_API_KEY=발급받은키
py prepare_local.py --all
```

`aihubshell` 은 bash 스크립트라 **Git for Windows(Git Bash)** 가 필요합니다.
설치돼 있으면 자동으로 찾아 씁니다. → [`docs/cautions/07`](docs/cautions/07_Windows_로컬_환경_설정.md)

</details>

데이터를 더 쓰고 싶으면 청크를 이어붙일 수 있습니다 (받고→정제→원본삭제 반복):

```bash
python prepare_local.py --chunk VL01     # 21GB
python prepare_local.py --chunk TL01     # 90GB 추가
python prepare_local.py --finalize       # ★ 마지막에 한 번 (교차 누수 방지)
python prepare_local.py --package
```

원본 21GB → ROI 크롭 후 **2~5GB** 로 줄어들어 업로드가 현실적입니다.
생성된 `dogskin_prepared.zip` 을 **Kaggle 에 비공개로** 올린 뒤 학습만 클라우드에서 하세요.

📖 [`docs/cautions/06_해외IP_다운로드_차단_우회.md`](docs/cautions/06_해외IP_다운로드_차단_우회.md)

### 3. Colab / Kaggle 에서 학습

> 💡 리포를 다운로드하거나 드라이브에 올릴 필요 없습니다.
> 아래 버튼으로 열면 노트북 첫 셀이 알아서 `git clone` 합니다.

[![Colab 에서 열기](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gayeoniee/deeplearning_test/blob/main/notebooks/03_학습_베이스라인.ipynb)

또는 Colab → **파일 → 노트북 열기 → GitHub 탭** → `gayeoniee/deeplearning_test` 검색.

| 노트북 | 내용 |
|---|---|
| [`03_학습_베이스라인`](notebooks/03_학습_베이스라인.ipynb) | 크롭 눈으로 확인 → **1단계+2단계 베이스라인** → 파이프라인 성능 → 크롭 비교 |
| [`04_학습_최신모델_비교`](notebooks/04_학습_최신모델_비교.ipynb) | timm 6종 + 앙상블 (2단계 기준) |
| [`05_평가_보정_GradCAM`](notebooks/05_평가_보정_GradCAM.ipynb) | 온도 보정, 거절 임계값, CAM 게이트, holdout 최종 |

> 번호가 03부터인 이유: **데이터 확보·확인·전처리(00~02)는 노트북이 아니라
> `prepare_local.py` 가 합니다.** AI Hub 가 해외 IP 다운로드를 막기 때문에
> 그 단계는 한국 PC 에서만 돌아가고, 클라우드에서는 학습만 하면 됩니다.
> 번호는 [로드맵](docs/00_로드맵.md)의 STEP 과 맞춰 두었습니다.

각 노트북의 첫 셀이 리포 clone + 패키지 설치 + 환경 감지를 자동으로 합니다.
**Colab / Kaggle 양쪽에서 같은 코드가 돕니다.**

> 🔑 API 키는 학습 노트북에서는 필요하지 않습니다 (다운로드가 로컬에서 끝났으므로).
> 로컬에서는 환경변수 `AIHUB_API_KEY` 로 주고, **코드나 노트북 셀에 직접 붙여넣지 마세요.**

---

## 이 프로젝트가 신경 쓴 것

이 데이터를 먼저 써 본 공개 프로젝트들이 공통으로 실패한 지점이 있습니다.
그 실패를 설계에 반영했습니다.

### 1. 데이터 누수 — 개체 단위 분할

강아지 1만 마리에서 50만 장. **개체당 평균 50장**입니다.
이미지 단위로 train/val 을 나누면 같은 강아지가 양쪽에 들어가
**정확도가 가짜로 부풀려집니다.**

→ `src/split.py` 가 `개체ID ∪ 중복클러스터`를 union-find 로 묶어 그룹 분할합니다.
`split.verify()` 가 누수를 발견하면 **에러를 냅니다.**

### 2. 중복 오염 — phash 기반 제거

같은 이미지가 **여러 클래스 폴더에 중복 존재**한다는 보고가 있습니다.
라벨이 모순되니 학습이 망가집니다.

→ `src/dedup.py` 가 perceptual hash + LSH 밴딩으로 near-duplicate 를 찾고,
**라벨이 충돌하는 그룹은 전부 제거**합니다.

### 3. 병변이 너무 작음 — ROI 크롭

실측한 병변 면적은 이미지의 **0.3%** 수준입니다.
전체 이미지로 학습하면 모델이 **배경을 학습**합니다.

→ `src/crop.py` 가 bbox/polygon 주변만 잘라냅니다. margin 3종을 만들어 실험 비교.

> ⚠️ 단, `m1.5`/`m2.5` 크롭은 **정답 박스를 안다는 전제**입니다. 실제 보호자 사진에는
> 박스가 없으므로 `full` 점수가 배포에 정직한 숫자입니다. 노트북 03 이 격차를 잽니다.

### 4. 스키마를 추측하면 틀립니다 — 실물로 확정

코드를 처음 만들 때는 AI Hub 원본에 접근할 수 없어 `src/scan.py` 로 키를 추론했습니다.
그 뒤 실물을 열어 보니 **세 군데가 틀렸고, 셋 다 에러 없이 그럴듯한 값을 냈습니다**:

| 틀린 곳 | 증상 | 실제 |
|---|---|---|
| 라벨을 폴더명에서 읽음 | 무증상 23,669장이 병변 라벨을 받음 | `metaData.lesions` |
| 이미지 크기를 박스에서 읽음 | 병변 면적이 **항상 100%** | `metaData.resolution` |
| polygon 을 점 목록으로 가정 | 추출률 **0%** (bbox 로 조용히 대체됨) | 평평한 `{x1,y1,…}` dict |

→ `diagnose.py` 로 실제 JSON 을 열어 확정했습니다.
전체 내용: [`docs/data/DATASET_CARD.md`](docs/data/DATASET_CARD.md)

### 5. 두 문제를 섞지 않기 — 2단계 구조

"병원에 가봐야 하나?"(이진)와 "무슨 병변인가?"(6종)는 임상적 무게가 다른 질문입니다.
7클래스 손실함수는 A7→A2 오류와 A2→A3 오류를 똑같이 취급하고,
"놓치지 않는 게 우선"인 임계값을 1단계에만 걸 수도 없습니다.

→ `src/stages.py` 가 **하나의 분할을 공유하는** 두 뷰를 만들고,
`stages.pipeline_report()` 가 두 단계를 **이어붙인 실제 성능**을 잽니다.
단계별 점수를 곱한 추정치는 낙관적이라 보고에 쓰지 않습니다.
→ [`docs/cautions/08`](docs/cautions/08_2단계_파이프라인_설계_주의점.md)

### 6. "의심된다"를 말할 자격 — 확률 보정

신경망은 과신합니다. "95% 확신"이 실제로는 70%인 게 흔합니다.
그 숫자를 보호자에게 보여주면 거짓말입니다.

→ `src/calibrate.py` 가 온도 스케일링 + ECE 측정 + coverage-risk 곡선을 제공하고,
`src/infer.py` 가 저신뢰 시 **"판단이 어려운 사진입니다"** 로 물러섭니다.

### 7. 배경을 보는지 검사 — Grad-CAM 게이트

정확도가 좋아도 병변이 아니라 진료대·조명을 보고 맞히면 실사용에서 무너집니다.

→ `src/explain.py` 가 CAM–병변 정렬도를 **수치로** 계산합니다.
`median_lift < 1.3` 이면 정확도와 무관하게 재작업입니다.

### 8. Colab 세션은 끊깁니다 — 이어받기

`/content` 는 세션이 끝나면 통째로 사라집니다. 90분짜리 학습이 80분에 끊기면
체크포인트까지 다 날아갑니다.

→ `train.fit` 이 매 에폭 **옵티마이저·스케줄러·EMA·난수 상태까지** `last.pt` 로
저장하고 Drive 로 복사합니다. 끊기면 **노트북을 그냥 다시 돌리세요** — 끝난 학습은
`⏭️` 건너뛰고, 끊긴 학습은 `▶️` 그 에폭부터 이어갑니다.
→ [`docs/cautions/09`](docs/cautions/09_세션이_끊겼을_때.md)

---

## 문서

### 📘 딥러닝 기초 (머신러닝 경험자용)

[`docs/basics/`](docs/basics/) — 10편. 각 STEP 직전에 읽도록 매칭되어 있습니다.

`01` ML과 딥러닝 차이 · `02` 이미지→텐서 · `03` CNN · **`04` 전이학습** ·
`05` 학습루프 · `06` 과적합·증강 · `07` 평가지표 · `08` 확률보정 ·
`09` ViT와 최신 백본 · `10` 경량화·배포

### ⚠️ 주의사항

[`docs/cautions/`](docs/cautions/) — 9편.

`01` 라이선스·재배포 · **`02` 데이터 누수** · `03` 의료AI 안전설계 ·
`04` Colab/Kaggle 트러블슈팅 · `05` 실험기록·재현성 ·
**`06` 해외IP 다운로드 차단** · `07` Windows 로컬 환경 설정 ·
**`08` 2단계 파이프라인 설계** · `09` 세션이 끊겼을 때

### 🗺️ 전체 진행표

[`docs/00_로드맵.md`](docs/00_로드맵.md)

---

## 분류 대상

**A1~A6은 병명이 아니라 병변의 형태입니다.**
([`docs/data/병변_6종_임상_해설.md`](docs/data/병변_6종_임상_해설.md))

| 코드 | 병변 형태 | 긴급도 |
|---|---|---|
| A1 | 구진·플라크 | 관찰 |
| A2 | 비듬·각질·상피성잔고리 | 관찰 |
| A3 | 태선화·과다색소침착 | 진료 권장 |
| A4 | 농포·여드름 | 진료 권장 |
| A5 | 미란·궤양 | 조기 진료 권장 |
| A6 | 결절·종괴 | **종양 감별 필요** |

**A7 = 무증상(정상)** 이 데이터에 존재합니다.
따라서 **2단계** 구조로 갑니다: ① 정상/이상 → ② 병변 6종.

VL01 청크 실측 (반려견 + 일반카메라, 중복 제거 후 45,885장):

| 단계 | 구성 | 불균형 |
|---|---|---|
| 1단계 정상/이상 | A7 22,815 : 나머지 23,070 | 거의 5:5 |
| 2단계 병변 6종 | 23,070장 (A2 7,693 ↔ A5 1,464) | **5.3배** |

> ⚠️ 무증상 이미지가 `무증상/A1_구진_플라크/` 처럼 **병변 폴더 안에** 들어 있습니다.
> 라벨은 폴더명이 아니라 `metaData.lesions` 를 봐야 합니다.
> → [`docs/data/DATASET_CARD.md`](docs/data/DATASET_CARD.md) 에 확정 스키마와 당한 함정 정리

---

## 목표 성능

**파이프라인 기준**으로 봅니다. 2단계만 좋아도 사용자에게 도달하지 않습니다.

| 지표 | 실측 (VL01/ResNet50) | 목표 |
|---|---|---|
| **스크리닝 recall** (병변을 놓치지 않은 비율) | — | **≥ 0.95** ← 가장 중요 |
| 1단계 AUROC (`full` 크롭) | 0.8031 | ≥ 0.85 |
| 2단계 6종 macro-F1 (개체 단위 분할) | 0.4865 | 0.55 ~ 0.65 |
| **A6 recall** (종양 감별) | 0.328 | **≥ 0.50** |
| 보정 후 ECE | — | < 0.10 |
| CAM–병변 정렬도 `median_lift` | — | ≥ 1.3 |
| 배율·위치 교란 시 점수 하락 | — | < 15% |

📊 측정 근거: [`docs/results/STEP4A_베이스라인_실측.md`](docs/results/STEP4A_베이스라인_실측.md)

> 개체 단위 분할에서 **0.95+ macro-F1 이 나오면 누수를 의심**하세요.
> 축하할 숫자가 아니라 검증해야 할 숫자입니다.

---

## 구조

```
src/
├── env.py         Colab/Kaggle/로컬 자동 감지, 경로·시크릿 통합
├── config.py      모든 하이퍼파라미터 + 클래스 정의 + 모델 라인업
├── aihub.py       aihubshell 래퍼 (부분 다운로드, 중단 재개)
├── scan.py        ★ 스키마 자동 추론 + 데이터셋 카드 생성
├── labels.py      JSON → 매니페스트(parquet)
├── dedup.py       phash 중복 제거
├── split.py       ★ 개체 단위 그룹 분할
├── stages.py      ★ 2단계 구조 (정상/이상 → 병변 6종) + 파이프라인 평가
├── crop.py        ROI 크롭
├── data.py        Dataset / DataLoader / 증강
├── models.py      timm 팩토리 (이름 자동 fallback) + EMA + 앙상블
├── train.py       AMP + EMA + cosine warmup 학습 루프 + ★ 중단/재개
├── evaluate.py    macro-F1, 클래스별 recall, 부트스트랩 CI
├── calibrate.py   온도 스케일링, ECE, coverage-risk
├── explain.py     Grad-CAM + 병변 정렬도 수치화
├── robust.py      ★ 실사용 견고성 (배율·위치 교란 검사)
├── bench.py       처리량 진단 (입력 파이프라인 vs GPU 병목 판정)
└── infer.py       추론 + 안전한 안내 문구 생성
```

---

## 라이선스 / 데이터 취급

- **AI Hub 데이터는 재배포 금지**입니다. `.gitignore` 가 이미지·매니페스트·가중치를 막고 있습니다.
- API 키는 Colab/Kaggle Secrets 에만 저장하세요.
- 자세한 내용: [`docs/cautions/01`](docs/cautions/01_데이터_라이선스와_재배포_금지.md)

출처 명시: `AI Hub 반려동물 피부 질환 데이터, 한국지능정보사회진흥원`
