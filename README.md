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

[![Colab 에서 열기](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gayeoniee/deeplearning_test/blob/main/notebooks/00_데이터_다운로드.ipynb)

또는 Colab → **파일 → 노트북 열기 → GitHub 탭** → `gayeoniee/deeplearning_test` 검색.

| 노트북 | 내용 |
|---|---|
| [`00_데이터_다운로드`](notebooks/00_데이터_다운로드.ipynb) | aihubshell 로 반려견+일반카메라만 부분 다운로드 |
| [`01_데이터_스캔_EDA`](notebooks/01_데이터_스캔_EDA.ipynb) | 스키마 자동 추론, 무증상 데이터 존재 여부 판정 |
| [`02_전처리_매니페스트`](notebooks/02_전처리_매니페스트.ipynb) | 중복 제거 → 개체 단위 분할 → ROI 크롭 |
| [`03_학습_베이스라인`](notebooks/03_학습_베이스라인.ipynb) | ResNet50 검증 + 크롭 방식 비교 |
| [`04_학습_최신모델_비교`](notebooks/04_학습_최신모델_비교.ipynb) | timm 6종 + 앙상블 |
| [`05_평가_보정_GradCAM`](notebooks/05_평가_보정_GradCAM.ipynb) | 온도 보정, 거절 임계값, CAM 검증 |

각 노트북의 첫 셀이 리포 clone + 패키지 설치 + 환경 감지를 자동으로 합니다.
**Colab / Kaggle 양쪽에서 같은 코드가 돕니다.**

> 🔑 API 키는 **Colab 🔑 보안 비밀** 또는 **Kaggle Secrets** 에 `AIHUB_API_KEY` 로 등록하세요.
> 코드나 노트북 셀에 직접 붙여넣지 마세요.

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

병변이 이미지의 5% 미만인 경우가 93% 라는 보고가 있습니다.
전체 이미지로 학습하면 모델이 **배경을 학습**합니다.

→ `src/crop.py` 가 bbox/polygon 주변만 잘라냅니다. margin 3종을 만들어 실험 비교.

### 4. 스키마를 모름 — 자동 추론

이 코드를 만들 때 AI Hub 원본에 접근할 수 없었습니다.
JSON 키를 추측해 박아두면 거의 확실히 틀립니다.

→ `src/scan.py` 가 키 경로를 평탄화해 `label`/`polygon`/`bbox`/`animal_id`
후보를 **자동 지목**합니다. 스캔 결과가 항상 이깁니다.

### 5. "의심된다"를 말할 자격 — 확률 보정

신경망은 과신합니다. "95% 확신"이 실제로는 70%인 게 흔합니다.
그 숫자를 보호자에게 보여주면 거짓말입니다.

→ `src/calibrate.py` 가 온도 스케일링 + ECE 측정 + coverage-risk 곡선을 제공하고,
`src/infer.py` 가 저신뢰 시 **"판단이 어려운 사진입니다"** 로 물러섭니다.

### 6. 배경을 보는지 검사 — Grad-CAM 게이트

정확도가 좋아도 병변이 아니라 진료대·조명을 보고 맞히면 실사용에서 무너집니다.

→ `src/explain.py` 가 CAM–병변 정렬도를 **수치로** 계산합니다.
`median_lift < 1.3` 이면 정확도와 무관하게 재작업입니다.

---

## 문서

### 📘 딥러닝 기초 (머신러닝 경험자용)

[`docs/basics/`](docs/basics/) — 10편. 각 STEP 직전에 읽도록 매칭되어 있습니다.

`01` ML과 딥러닝 차이 · `02` 이미지→텐서 · `03` CNN · **`04` 전이학습** ·
`05` 학습루프 · `06` 과적합·증강 · `07` 평가지표 · `08` 확률보정 ·
`09` ViT와 최신 백본 · `10` 경량화·배포

### ⚠️ 주의사항

[`docs/cautions/`](docs/cautions/) — 7편.

`01` 라이선스·재배포 · **`02` 데이터 누수** · `03` 의료AI 안전설계 ·
`04` Colab/Kaggle 트러블슈팅 · `05` 실험기록·재현성 ·
**`06` 해외IP 다운로드 차단** · `07` Windows 로컬 환경 설정

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

무증상(정상) 데이터가 존재하면 **2단계** 구조(정상/이상 → 병변 6종)로 갑니다.
존재 여부는 STEP 2 스캔으로 확정합니다.

---

## 목표 성능

| 지표 | 목표 |
|---|---|
| 6종 macro-F1 (개체 단위 분할) | 0.70 ~ 0.80 |
| 모든 클래스 recall | ≥ 0.5 |
| 1단계 정상/이상 recall | ≥ 0.95 |
| 보정 후 ECE | < 0.10 |

> 개체 단위 분할에서 **0.95+ 가 나오면 누수를 의심**하세요.

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
├── crop.py        ROI 크롭
├── data.py        Dataset / DataLoader / 증강
├── models.py      timm 팩토리 (이름 자동 fallback) + EMA + 앙상블
├── train.py       AMP + EMA + cosine warmup 학습 루프
├── evaluate.py    macro-F1, 클래스별 recall, 부트스트랩 CI
├── calibrate.py   온도 스케일링, ECE, coverage-risk
├── explain.py     Grad-CAM + 병변 정렬도 수치화
└── infer.py       추론 + 안전한 안내 문구 생성
```

---

## 라이선스 / 데이터 취급

- **AI Hub 데이터는 재배포 금지**입니다. `.gitignore` 가 이미지·매니페스트·가중치를 막고 있습니다.
- API 키는 Colab/Kaggle Secrets 에만 저장하세요.
- 자세한 내용: [`docs/cautions/01`](docs/cautions/01_데이터_라이선스와_재배포_금지.md)

출처 명시: `AI Hub 반려동물 피부 질환 데이터, 한국지능정보사회진흥원`
