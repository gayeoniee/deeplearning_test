# ROI crop 후속 비교

실행: `notebooks/03i_ROI_crop_비교.ipynb`를 Kaggle에 Import하고 기존
`safe-crop-pilot` Dataset을 연결한다. GPU T4 / Internet ON. 코드는 노트북에
포함되므로 8GB 데이터 재업로드는 필요 없다. 기본 HOURS=3, batch=16,
EPOCHS=5이며 남은 GPU 시간에 맞춰 HOURS만 조절한다. 완료 시간 보장은 없다.

두 방법 모두 동일한 20,000 train / 4,000 val, EfficientNetV2-S ImageNet
사전학습 초기값, 384px letterbox, AdamW 3e-4, class-weight CE, smoothing .1,
epoch cosine 5회, flip/color를 사용한다. backbone 별도 학습률·EMA·과거 photometric
프리셋은 사용하지 않는다. 따라서 역사적 최고 성능의 재현이나 직접 비교는 아니다.

- fixed: 모든 병변 주석에 각각 5% 여백을 더한 영역을 포함하는 중앙 crop.
  최소 변 길이 320px이며 병변이 크거나 여러 개면 확장, 영상 경계에서 제한한다.
- safe: 같은 최소 변 길이의 1–1.25배를 샘플링하고, 모든 병변을 포함할 수 있는
  위치 내에서 무작위 이동한다. 회전·erasing 등 추가적인 병변 절단은 없다.
- 검증: 두 방법 모두 fixed. 주석 없는 사진은 두 방법 모두 전체 이미지.

고정 ROI도 병변을 보존하므로, 여기서는 주변 배경과 위치 변화의 추가 효과를 본다.
실제 사용자 사진에는 정답 병변 주석이 없으므로 이 검증만으로 사용자 성능을
입증할 수 없다. ROI 선택 단계 및 외부 사진 평가는 별도 과제다.

기존 실험과 동일한 Pillow truncated 허용을 subprocess/worker에 적용한다.
이는 손상된 원본을 복원하지 않는다. 설정과 실행 소스 SHA를 protocol에 기록한다.
기존 Dataset 소스 검증은 유지하며, 실행 코드는 별도 working 디렉터리에 둔다.

출력은 `roi_crop_pilot/` 및 `roi_crop_pilot_resume.zip`. 이전 ResNet 실험의
resume은 사용하지 않는다. 재개 ZIP을 비공개 Dataset으로 연결하고 동일 노트북을
실행한다. 모델·코드·epoch horizon·batch 등 protocol이 다르면 재개를 거부한다.
시간 초과 시 마지막 완료 epoch부터 재개하며, 중간 epoch는 다시 수행한다.

판단: 같은 완료 epoch의 F1·AUROC·이상 recall·정상 specificity를 함께 확인한다.
한 번의 소규모 결과로 성능 상한이나 멘토 제안 전체의 유효성을 단정하지 않는다.
holdout은 열지 않는다. 로컬 검증은 crop 경계·모든 병변 포함, 작은 모델의
쌍별 실행/재개 및 내장 소스 실행이며 실제 GPU 학습 성능은 아직 측정하지 않았다.
