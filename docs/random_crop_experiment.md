# 원본 기반 병변 보존 random crop — 2026-09-09

## 완료한 작업

TL01(90.07 GiB), TL02(80.11 GiB), VL01(21.10 GiB)의 ZIP을 직접 읽었다. 전체 JSON 496,062건을 검사했고, 반려견·일반카메라 범위에서 채택한 사진 419,752건은 ZIP CRC, JPEG 헤더, JSON 대비 해상도, 이미지 ID, 전체 ROI 좌표를 확인하고 JPEG SHA256을 계산했다. 범위 밖 이미지와 격리된 이미지까지 모든 JPG를 디코딩한 검사는 아니다.

기존 `data/work/crops`와 `data/work/manifests`는 수정하지 않았다. 새 데이터와 보고서는 `data/work/safe_crop_v1/`에 저장했다. 원본 ZIP도 수정하거나 풀지 않았다.

| 묶음 | 채택한 사진·JSON 쌍 | 범위 밖 | 격리 |
|---|---:|---:|---:|
| TL01 | 263,330 | 0 | 10 |
| TL02 | 109,860 | 64,823 | 2,922 |
| VL01 | 46,562 | 8,104 | 451 |

격리 3,383건: JSON 내부 이미지 ID와 파일명 불일치 3,362건, 이미지 범위를 벗어나거나 유효하지 않은 박스 17건, 같은 폴더의 JPG 누락 3건, 잘못된 polygon 1건. ID 불일치는 이름 체계 변경일 가능성도 있어 라벨이 틀렸다고 단정하지 않는다. 자동으로 짝을 바꾸지 않고 각 청크의 `*_audit.json`에 기록했다.

채택한 419,752건 중 동일 JPEG 바이트에 다른 라벨이 붙은 10,419건은 `conflicting_labels.parquet`으로 격리했다. 같은 라벨의 완전 중복 38,662건을 제거하여 **370,671장**을 새 실험 대상으로 고정했다.

## 새 실험 분할

`manifest.parquet`의 seed 42, 5-fold, holdout 분할을 모든 비교군이 공유한다. holdout은 61,281장이다. 대용 개체 ID와 동일 JPEG SHA256의 연결 관계를 먼저 합친 뒤 중복을 제거했다. fold 간 그룹·대용 ID·완전 동일 사진 중복을 검사했다.

이 분할은 과거 실험 분할이 아니다. 과거 최종 분할은 로컬에서 찾지 못했다. 과거 f320/m2.5 성능과 숫자를 직접 비교하면 안 된다. 대용 개체 ID는 실제 개체를 보장하지 않으며, **인코딩이 다른 유사 이미지의 pHash 중복 검사는 아직 하지 않았다.** 실제 성능 해석에는 이 제한이 남아 있다.

## crop 범위 선정

fold 0의 **학습 데이터에서만** 클래스별 200장, 사진별 후보 범위당 5회씩 검사했다. 각 범위·방법마다 7,000회이고 전체 42,000회다. 비율 범위는 0.85–1.18, 모든 ROI의 폭·높이 각각 5% 여유를 보존한다. 정상 A7의 주석 ROI도 동일하게 처리한다.

| 원본 면적 범위 | 일반 random crop의 ROI 잘림 | 보존 crop의 ROI 잘림 | 보존 crop의 전체 사진 fallback |
|---|---:|---:|---:|
| 35–100% | 36.83% | 0% | 1.16% |
| 50–100% | 22.09% | 0% | 1.19% |
| 70–100% | 10.14% | 0% | 100% |

1920×1080 원본에서 거의 정사각형으로 면적 70% 이상을 자르려 하면 조건을 만족하기 어렵다. **35–100%를 첫 실험값**으로 선정했다. 이는 원본 면적 비율이며 병변 보존율 35%라는 뜻이 아니다. 모든 주석 ROI를 100% 포함하는 창만 채택하고, 불가능하면 원본 전체를 쓴다. 주석 밖의 미표기 병변까지 보장하는 것은 아니다.

정량 결과: `crop_summary.csv`, `crop_trials.parquet`. 육안 검토 이미지: `preview.jpg` (학습 사진, 원본과 무작위 crop 두 개, 빨간 주석 경계). 이 수치는 **증강의 좌표 검사**이며 분류 성능 개선 수치가 아니다.

## 구현과 실행

- `src/safe_crop.py`: JSON의 모든 box/polygon/location 파싱, 원본 좌표에서 병변 보존 crop.
- `src/original_data.py`: 원본 ZIP을 직접 읽는 학습 로더. 프로세스별 ZIP 핸들, 읽기 오류 시 명시적 실패. crop 이후에는 뒤집기와 약한 색 변화만 적용하고 비율 유지 resize/padding. 검증은 모든 방법에서 원본 전체를 동일하게 처리한다.
- `tools/prepare_safe_crop.py`: 독립 매니페스트와 오류 보고서 생성. 기존 결과가 있으면 덮어쓰기를 거부한다.
- `tools/finalize_safe_crop.py`: 충돌 격리·완전 중복 처리·분할 고정 및 학습 표본 crop 검사.
- `tools/train_safe_crop.py`: 기존 `src.train.fit`에 연결된 비교 실험 실행기. stage1, stage2, seven 지원. 모델/seed/분할이 같고 crop 방식만 다른 비교군을 실행한다.

로컬 동작 확인:

```bash
.venv/bin/python -m tools.train_safe_crop --smoke --mode safe --workers 0 --batch-size 4
```

GPU 환경에서 프로젝트와 원본 ZIP, `data/work/safe_crop_v1/manifest.parquet`을 준비한 뒤:

```bash
python -m tools.train_safe_crop --mode full --task stage1 --raw /path/to/raw
python -m tools.train_safe_crop --mode random --task stage1 --raw /path/to/raw
python -m tools.train_safe_crop --mode safe --task stage1 --raw /path/to/raw
```

stage2는 `--task stage2`로 별도 비교한다. 본 학습 기본값은 사전학습 ResNet50, 288px, 15 epoch, seed 42다. 확정 최적 설정이 아니며 첫 비교 기준이다. 기본값에서 후속 회전/affine/erasing/mixup/CutMix는 적용되지 않는다. split 파일 해시와 crop 설정을 실행별로 기록하고 설정이 다른 기존 실행에 이어 학습하지 않는다.

## 검증과 남은 일

좌표 보존, 다중 주석, 오류 처리, 비율 유지와 가장자리 보존, 검증 입력의 방법 간 동일성, 다중 프로세스 ZIP 로딩, 역전파를 검사했다. 기존 학습기의 1 epoch 학습과 best/last 체크포인트 저장·복원도 통합 테스트를 통과했다. 실제 원본으로 작은 네트워크의 학습·검증 동작 테스트는 full/random/safe 세 방법 모두 통과했다. `smoke_*.json`은 연결 확인용이며 모델 평가 결과가 아니다. 채택한 419,752건 전체에 대해 엄격한 polygon 파서로 JSON을 다시 읽고 저장 좌표와 일치함을 확인했다(`strict_annotation_audit.json`).

현재 본 학습과 분류 성능 비교는 실행하지 않았다. 로컬에서 CUDA GPU를 사용할 수 없어 본 학습은 GPU 환경에서 진행해야 한다. 세 비교군의 macro F1·클래스별 recall과 위치/배율 변화 평가를 같은 조건으로 비교해야 성능 향상을 판단할 수 있다. 기존 실패 유형에 맞춘 평가와 유사 이미지 중복 점검도 필요하다.
