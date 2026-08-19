# 데이터셋 카드 — (아직 생성 전)

이 파일은 **`src/scan.py` 가 실물 데이터를 훑어 자동으로 덮어씁니다.**
손으로 채우지 마세요.

## 생성 방법

```python
from src import scan
rep = scan.run()              # STEP 1(다운로드) 이후 실행
scan.write_dataset_card(rep)  # 이 파일을 덮어씀
```

`notebooks/01_데이터_스캔_EDA.ipynb` 를 그냥 돌리면 위 두 줄이 실행됩니다.

## 여기에 채워질 내용

- 이미지/라벨 수, 용량
- 클래스별 분포와 불균형 비
- 폴더 축 (반려견/반려묘, 일반카메라/더모스코프, 유증상/**무증상 존재 여부**)
- JSON 키 경로 전체와 역할별 추정 (`label`, `polygon`, `bbox`, `animal_id` …)
- 병변 면적 비율 분포 — ROI 크롭이 필요한지 판단하는 근거
- 중복률과 **클래스 간 중복 그룹 수**
- 개체ID 후보 — 데이터 누수를 막는 그룹 분할의 기준

## 아직 확인되지 않은 것들

계획 수립 시점에는 `aihub.or.kr` 접근이 불가능해서, 아래는 이 데이터를 사용한
공개 프로젝트들에서 교차 확인한 **추정치**입니다. 스캔 결과로 반드시 검증하세요.

| 항목 | 추정 | 검증 방법 |
|---|---|---|
| 반려견 병변 클래스 | A1~A6 6종 | `rep.class_counts` |
| 무증상(정상) 데이터 | 초기엔 없었고 이후 추가되었다는 보고가 있음 | `rep.has_normal` |
| 전체 규모 | 반려동물 1만 마리 / 50만 장 이상 | `rep.n_images` |
| 병변 면적 | 93% 가 이미지의 5% 미만 | `rep.lesion_area["under_5pct"]` |
| 중복 오염 | 같은 이미지가 여러 클래스에 존재 | `rep.dup_estimate["cross_class_groups"]` |

추정이 틀리면 계획을 바꾸면 됩니다. **스캔 결과가 항상 이깁니다.**

---

## 참고: 파이프라인 검증에 쓴 합성 데이터

실물 데이터가 없는 상태에서 코드를 검증하기 위해, AI Hub 구조를 흉내낸
합성 데이터로 전 구간을 돌려봤습니다. 그때 스캐너가 추론해낸 스키마 예시입니다
(**실물이 아니라 합성 데이터 기준**이니 참고만 하세요):

| 역할 | 추론된 키 |
|---|---|
| label | `labelingInfo[].label.label_disease_lv_3` |
| polygon | `labelingInfo[].polygon.location[]` |
| bbox | `labelingInfo[].box.location[]` (x/y/width/height 형식) |
| image_name | `images.file_name` |
| animal_id | `metadata.pet_id` |
| width / height | `images.width` / `images.height` |

실물 AI Hub JSON 이 이와 다르더라도 `scan.py` 가 알아서 찾아냅니다.
못 찾으면 `rep.json_keys` 를 직접 보고 `labels.py` 의 추출 함수를 조정하면 됩니다.
