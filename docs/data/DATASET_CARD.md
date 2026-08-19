# 데이터셋 카드 — AI Hub 반려동물 피부 질환 (561)

> ⚠️ 이 파일은 `src/scan.py` 가 실물 데이터를 훑어 자동 생성합니다. 손으로 고치지 마세요.
> 생성 시각: `2026-08-19T03:52:33+00:00`  |  스캔 경로: `/tmp/claude-0/-home-user-deeplearning-test/a4adefca-fa0a-5067-890b-cec6b2351c8d/scratchpad/fakedata`

## 규모

| 항목 | 값 |
|---|---|
| 이미지 | 419 장 |
| 라벨 JSON | 419 개 |
| 용량 | 0.0 GB |
| 무증상(정상) 데이터 | 없음 |

## 클래스 분포

| 클래스 | 이미지 수 | 비율 |
|---|---:|---:|
| A2 | 96 | 22.9% |
| A5 | 67 | 16.0% |
| A6 | 64 | 15.3% |
| A4 | 64 | 15.3% |
| A1 | 64 | 15.3% |
| A3 | 64 | 15.3% |

불균형 비(최다/최소): **1.5배**

## 폴더 축

- **species**: {'반려견': 774, '반려묘': 64}
- **camera**: {'일반카메라': 614, '더모스코프': 224}
- **symptom**: {'유증상': 838}
- **split**: {'Training': 422, 'Validation': 416, 'TS': 211, 'TL': 211, 'VS': 208, 'VL': 208}
- **class**: {'A2': 192, 'A5': 134, 'A6': 128, 'A4': 128, 'A1': 128, 'A3': 128}

## JSON 스키마 추정

| 역할 | 추정 키 |
|---|---|
| label | `labelingInfo[].label.label_disease_lv_3` |
| polygon | `labelingInfo[].polygon.location[].x` |
| bbox | `labelingInfo[].box.location[].x` |
| image_name | `images.file_name` |
| animal_id | `metadata.pet_id` |
| breed | `metadata.breed` |
| age | `metadata.age` |
| camera | `metadata.camera` |
| width | `images.width` |
| height | `images.height` |

<details><summary>키 경로 전체 (상위 60)</summary>

| 출현율 | 키 | 타입 | 예시 |
|---:|---|---|---|
| 100% | `images.file_name` | ['str'] | ['IMG_견_DA2001_A2_02.jpg'] |
| 100% | `images.width` | ['int'] | ['640'] |
| 100% | `images.height` | ['int'] | ['480'] |
| 100% | `metadata.pet_id` | ['str'] | ['DA2001'] |
| 100% | `metadata.breed` | ['str'] | ['말티즈'] |
| 100% | `metadata.age` | ['int'] | ['3'] |
| 100% | `metadata.camera` | ['str'] | ['일반카메라'] |
| 100% | `labelingInfo[].label.label_disease_lv_3` | ['str'] | ['A2'] |
| 100% | `labelingInfo[].label.label_disease_nm` | ['str'] | ['테스트'] |
| 100% | `labelingInfo[].polygon.location[].x` | ['int'] | ['99'] |
| 100% | `labelingInfo[].polygon.location[].y` | ['int'] | ['35'] |
| 100% | `labelingInfo[].box.location[].x` | ['int'] | ['99'] |
| 100% | `labelingInfo[].box.location[].y` | ['int'] | ['35'] |
| 100% | `labelingInfo[].box.location[].width` | ['int'] | ['99'] |
| 100% | `labelingInfo[].box.location[].height` | ['int'] | ['99'] |

</details>

## 병변 면적 비율

- 중앙값 **2.08%**, p90 3.59%
- 이미지의 5% 미만인 비율: **100.0%**
- 1% 미만: 11.9%

> **→ ROI 크롭 필수.** 전체 이미지를 그대로 넣으면 모델이 배경을 학습합니다.

## 중복

- 샘플 419장 기준 중복률 **16.23%**
- 서로 다른 클래스에 걸친 중복 그룹: **45건**

## 해상도 (샘플)

- 640x480: 400

## 파일명 패턴 (숫자→`#`, 영문→`L`)

- `L_견_L#_L#_#` × 387
- `L_묘_L#_L#_#` × 32

## 개체ID 후보

데이터 누수를 막으려면 **개체 단위**로 train/val 을 나눠야 합니다.

| 토큰 위치 | 고유값 | 그룹당 평균 장수 | 예시 |
|---:|---:|---:|---|
| #2 | 38 | 11.0 | ['DA1000', 'DA5001', 'DA5000'] |

JSON 필드 후보: `metadata.pet_id`

## ⚠️ 경고

- ⚠️ 서로 다른 클래스에 동일 이미지가 45건 발견됨 (샘플 419장 기준). 선행 프로젝트가 실패한 원인입니다 — dedup.py 로 반드시 제거하세요.

## 샘플 JSON

```json
{
  "images": {
    "file_name": "IMG_견_DA2001_A2_02.jpg",
    "width": 640,
    "height": 480
  },
  "metadata": {
    "pet_id": "DA2001",
    "breed": "말티즈",
    "age": 3,
    "camera": "일반카메라"
  },
  "labelingInfo": [
    {
      "label": {
        "label_disease_lv_3": "A2",
        "label_disease_nm": "테스트"
      },
      "polygon": {
        "location": [
          {
            "x": 99,
            "y": 35
          },
          {
            "x": 198,
            "y": 35
          },
          {
            "x": 198,
            "y": 134
          },
          {
            "x": 99,
            "y": 134
          }
        ]
      },
      "box": {
        "location": [
          {
            "x": 99,
            "y": 35,
            "width": 99,
            "height": 99
          }
        ]
      }
    }
  ]
}
```