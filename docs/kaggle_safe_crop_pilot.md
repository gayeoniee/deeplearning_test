# 캐글 9시간용 원본 random crop 예비 실험

## 준비물

- `data/packages/dogskin_safe_crop_pilot.zip`: 원본 JPG 24,000장, 원본 JSON, 전체 ROI 좌표, 고정 표본, 실행 코드 스냅샷. 원본 사진을 리사이즈하거나 재압축하지 않는다.
- `notebooks/14_원본_병변보존_crop_파일럿.ipynb`: 이 패키지를 찾아 실행하는 노트북.

학습 20,000장·검증 4,000장을 기존 fold 0의 각 분할에서 A1–A7 비율에 맞춰 추출했다. 새로 train/val을 나누지 않는다. holdout은 포함하지 않는다. 두 방법이 완전히 같은 표본을 사용한다. 대용 개체 ID와 완전 동일 JPG의 분할 간 중복은 없다. 유사 사진 중복과 실제 개체 ID 부재의 제한은 원래 실험과 같다.

## 실행 순서

1. Kaggle Datasets → New Dataset에서 `dogskin_safe_crop_pilot.zip`을 **Private**로 업로드한다. 업로드와 Dataset 처리 완료까지 GPU를 켜지 않는다.
2. 새 Notebook에 `14_원본_병변보존_crop_파일럿.ipynb`를 Import한다. Add Input으로 위 Dataset만 붙인다. GitHub 코드나 191 GiB 원본 ZIP 전체는 필요 없다.
3. **GPU T4 x2**, Internet On으로 설정한다. 이 파일럿은 한 장만 사용한다. 첫 셀에서 별도 프로세스로 ResNet50 FP32/AMP 순전파·역전파·optimizer 실행을 검사한다. 통과한 뒤 첫 학습에서 torchvision 사전학습 ResNet50 가중치를 다운로드한다.
4. 남은 GPU 시간을 확인하고 노트북의 `HOURS`를 설정한다. 9시간이 남았을 때 기본값은 **7.0시간**이다. 데이터 준비·가중치 다운로드·출력 저장 여유를 남긴다.
5. **Save Version → Save & Run All**로 한 번 실행한다. 같은 전체 학습을 편집 세션에서 먼저 실행하고 다시 Save & Run All로 중복 실행하지 않는다.
6. 완료 후 Output의 `safe_crop_pilot_resume.zip`과 `safe_crop_pilot/comparison.json`, `history.csv`를 받는다.

Kaggle 공식 안내: https://www.kaggle.com/docs/notebooks , https://www.kaggle.com/docs/efficient-gpu-usage

### `CUDA error: no kernel image is available` 수정

일부 최신 PyTorch CUDA 빌드는 P100(Pascal)을 지원하지 않는다. GPU가 표시되거나 `torch.cuda.is_available()`이 True인 것만으로 실제 연산 호환성을 확인할 수 없다. 기존 P100 추천은 정정한다. T4 x2를 먼저 선택하고 수정 노트북을 새 세션에서 실행한다. 8.24 GiB Dataset은 다시 업로드하지 않는다. Dataset에 들어 있는 README는 최초 버전이므로 이 문서와 수정 노트북 안내를 우선한다.

첫 학습 전에 실패해 완료 epoch가 없는 재개 ZIP은 붙이지 않고 새 실행한다. 완료 epoch가 있는 경우에는 파일을 보존하고 환경 호환성을 확인한 후 이어간다. T4에서도 실패하면 첫 셀의 GPU 이름, compute capability, PyTorch/torchvision, CUDA build, compiled architectures와 연산 오류를 확인한다. 로컬 CPU 테스트는 실제 Kaggle CUDA 테스트를 대신하지 않는다.

공식 지원 변경: https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128

## 비교 내용

1단계 정상(A7) / 이상(A1–A6) 이진 분류 예비 실험이다. 두 조건은 일반 random crop과 병변 보존 random crop이다. 사전학습 ResNet50의 동일 초기 state_dict를 두 조건에 사용한다. 최대 5 epoch씩, 288px, batch 32, seed 42, 면적 35–100%, 병변 여유 5%다. AdamW·클래스 가중치·약한 색 변화·좌우 뒤집기 등 나머지 설정은 동일하다. 검증은 항상 전체 사진을 비율 유지 resize/padding하며 ROI 좌표를 사용하지 않는다.

두 조건을 한 epoch씩 번갈아 학습하고 epoch 쌍마다 순서를 바꾼다. 첫 epoch 시간을 출력하고, 관측 시간에 여유를 더해 다음 epoch 쌍이 예산에 들어오는지 확인한다. 시간 내 5쌍 완료를 보장하지 않는다. 새로운 설정을 고르기 위한 작은 실험이며 전체 데이터 성능이나 2단계 병변 종류 분류 결과가 아니다.

macro F1, 이상 recall, 정상 specificity, AUROC와 원본 A1–A7별 맞힘 비율을 저장한다. `comparison.json`은 **두 방법 모두 완료한 마지막 공통 epoch**끼리 비교한다. 단일 seed 예비 실험이므로 작은 차이를 확정 개선으로 해석하지 않는다.

## 시간이 부족하거나 중단됐을 때

- 시간 예산은 코드 실행 시점부터 계산하고 출력용 여유를 내부에서 추가로 남긴다. epoch 도중 예산에 도달하면 그 미완료 epoch는 폐기하고 마지막 완료 epoch를 보존한다.
- `random_last.pt` / `safe_last.pt`에는 모델·optimizer·AMP scaler·이력이 들어 있다. `initial.pt`도 함께 보관한다. 재개하면 미완료 epoch부터 같은 epoch seed로 다시 실행한다.
- 다음 세션에서 원본 파일럿 Dataset과 이전 `safe_crop_pilot_resume.zip`을 함께 Add Input하고 같은 노트북을 실행한다. 노트북이 재개 파일을 찾아 작업 폴더에 복원한다. 여러 재개 파일이 있으면 사용할 경로를 명시한다.
  ⚠️ **캐글은 Dataset 생성 때 ZIP 을 자동으로 풀어 올린다.** 노트북이 ZIP 파일만 찾으면 재개를 못 보고 **조용히 처음부터** 돈다 (2026-09-11 실제로 그랬음). 노트북 18 부터는 풀린 폴더도 찾고 재개 여부를 찍는다 — 14·15·16 노트북은 안 고쳤으니 그쪽으로 재개하려면 같은 수정이 필요하다.
- 재개할 때 `EPOCHS`, batch size, workers 등 설정과 코드 스냅샷은 그대로 둔다. `HOURS`만 그 세션의 남은 시간에 맞춰 조정할 수 있다.
- 플랫폼이 갑자기 세션을 종료하면 최신 Output의 보존은 보장되지 않는다. 코드의 체크포인트 작성만으로 Kaggle의 출력 저장을 대신할 수 없으므로, 정상 종료 후 Output/재개 ZIP을 확보한다.

## 로컬 검증

원본 JPG 바이트의 SHA256과 패키지 전체 CRC, 표본 비율·분할 격리, 일반 파일/ZIP 이미지 로더의 동일 입력, 동일 초기 가중치, 두 조건 학습·체크포인트 재개·공통 epoch 비교를 검사한다. CPU의 작은 네트워크를 사용하는 `--smoke`는 코드 연결 확인용이며 성능 검증이 아니다.

```bash
python tools/kaggle_safe_crop.py --data /path/to/unpacked/pilot --out /path/to/output --smoke --epochs 1 --workers 0 --batch-size 4
```
