# 앱 연동 — 스크리닝 API

안드로이드 앱(**DAENGS_APP**)이 붙을 자리입니다.
아직 합치지 않았고, 합칠 때 알아야 할 것을 여기 모았습니다.

```
사진 → [서빙 크롭] → 1단계(정상/이상) ─ 낮으면 → "정상으로 보입니다"
                                       └ 높으면 → 2단계(6종 분포) → 진료 권함
```

| 파일 | 역할 |
|---|---|
| [`src/agent.py`](../src/agent.py) | 파이프라인 + **응답 계약** (HTTP 모름) |
| [`src/message.py`](../src/message.py) | 문구 생성. torch 안 씁니다 |
| [`serve.py`](../serve.py) | FastAPI 서버 + 데모 화면 |
| [`demo/index.html`](../demo/index.html) | 데모 UI — **DAENGS 챗봇 화면** 재현 |
| [`tests/test_agent.py`](../tests/test_agent.py) | 계약 감시 43개 |

---

## 띄우기

```bash
# 가중치 없이 화면만 (torch 불필요, ~10초)
uv sync --extra serve
uv run --extra serve python serve.py --mock

# 진짜 모델로
uv sync --extra train --extra serve
uv run --extra train --extra serve python serve.py \
    --ckpt1 runs/stage1_effnetv2_s_f320_.../best.pt \
    --ckpt2 runs/stage2_convnextv2_base_m25_.../best.pt
```

* `http://127.0.0.1:8000/` — 데모 화면
* `http://127.0.0.1:8000/docs` — FastAPI 자동 생성 API 문서

`--threshold` 를 생략하면 1단계 체크포인트 옆의 `stage1_threshold.json` 을 찾습니다
(노트북 03/06 이 저장합니다). **못 찾으면 에러로 멈춥니다** — 기본값 0.5 로 조용히
넘어가면 recall 이 0.95 에서 0.82 로 떨어지는데 아무도 모릅니다.

---

## 응답 계약 (v1.0)

```jsonc
{
  "contract_version": "1.0",
  "verdict": "abnormal",              // "normal" | "abnormal" | "retake"
  "headline": "피부에 이상 소견이 보입니다.",
  "body":     "어떤 병변인지는 이 사진만으로 판단할 수 없습니다. …",
  "action":   "수의사 진료를 받아보시기를 권합니다.",

  "stage1": {
    "abnormal_prob": 0.6231, "abnormal_percent": 62.3,
    "threshold": 0.1823,
    "calibrated": false               // ⚠️ 아래 "아직 안 된 것" 참고
  },

  "stage2": {
    "shown": true,
    "distribution": [                 // ★ 항상 여섯 개. 확률 내림차순
      {"code":"A2","name_ko":"비듬·각질·상피성잔고리",
       "name_en":"Scale / Crust / Epidermal collarette","prob":0.31,"percent":31.0},
      … 다섯 개 더
    ]
  },

  "text": "…",                        // 터미널/디버그용 전문. 앱은 안 씁니다
  "disclaimer": "이 결과는 수의학적 진단이 아니며…",
  "meta": {"elapsed_ms": 412, "mock": false, "crop_untested": "…"}
}
```

### ★ "1등 병변" 필드가 없습니다 — 실수가 아닙니다

holdout 에서 그 이름이 **56.6% 틀렸습니다.** 필드로 주면 앱은 그걸 화면 제일 크게
띄웁니다. 그래서 계약에서 아예 뺐고, `tests/test_agent.py` 가 `top1` `predicted`
`diagnosis` 같은 키가 생기는지 감시합니다.

**앱 쪽 화면 규칙** (`docs/cautions/03_의료AI_안전설계_원칙.md` §7-B):

1. `distribution[0]` 을 뽑아 크게/굵게 쓰지 않습니다 — 여섯 줄을 **같은 무게**로
2. 여섯 개를 **전부** 보여줍니다. 상위 3개로 자르면 그게 답처럼 읽힙니다
3. `body`("판단할 수 없습니다")를 숫자보다 **위에** 놓습니다
4. `disclaimer` 는 접거나 회색으로 숨기지 않습니다
5. 병변별 긴급도 문구를 앱이 자체적으로 붙이지 않습니다 — 이름을 단정하는 셈입니다

`verdict` 별로 `stage2.shown` 이 false 면 분포 영역을 통째로 그리지 않으면 됩니다.

### 오류 응답

| 코드 | 뜻 |
|---|---|
| 400 | 빈 파일 |
| 413 | 12MB 초과 |
| 415 | 이미지로 열리지 않음 |

FastAPI 기본 형식(`{"detail": "..."}`)이고, 메시지는 한국어라 그대로 띄워도 됩니다.

---

## ⚠️ 아직 안 된 것 — 합치기 전에 읽어주세요

### 1. 서빙 크롭이 학습 크롭과 다릅니다 (실측 안 됨)

학습은 라벨 bbox 를 중심으로 잘랐습니다. **보호자 사진에는 bbox 가 없습니다.**
그래서 서빙에서는 **화면 중앙**을 씁니다:

| 단계 | 학습 | 서빙 |
|---|---|---|
| 1단계 | `f320` — bbox 중심 320px | 중앙, **짧은 변의 29.6%** |
| 2단계 | `m2.5` — bbox 긴 변 ×2.5 | 중앙 정사각 |

29.6% 는 320 ÷ 1080 입니다 (원본이 1920×1080 — `data/DATASET_CARD.md` §1).
픽셀 320 을 그대로 쓰면 안 됩니다 — 휴대폰 사진은 4032×3024 라 320px 이면
피부의 훨씬 좁은 부분만 보게 됩니다. **비율**로 잘라야 화각이 맞습니다.

이 대응이 성립하려면 보호자가 **촬영 가이드를 지켜야** 합니다:

```
권장 : 병변이 화면 가로의 34% ~ 56%
허용 : 28% ~ 68%
위치 : 병변이 화면 중앙에서 10% 이내
```

→ 앱 카메라에 **가이드 프레임**을 띄우고, 벗어나면 셔터를 막거나 "조금 더 가까이"
를 띄워주세요. 이 구간 밖 사진은 추론하지 말고 다시 찍게 하는 게 맞습니다.

⚠️ **그래도 이 대응은 실측된 적이 없습니다.** `STATUS.md` 의 숫자(1단계 holdout
AUROC 0.9304 등)는 전부 bbox 크롭에서 나온 것입니다. 앱으로 찍은 사진에서
얼마나 떨어지는지는 **그 사진에 수의사 라벨을 붙여봐야** 압니다.
응답의 `meta.crop_untested` 가 이 사실을 실어 나릅니다.

### 2. 1단계 확률이 보정(calibration)되지 않았습니다

화면에 뜨는 "이상 가능성 62%" 는 **보정 안 된 값**입니다. 온도 보정은 2단계에만
걸려 있습니다 (T=1.1063). 06 재실행 때 1단계도 같이 잽니다.
ECE ≥ 0.10 으로 나오면 **숫자를 빼고 문구만** 띄우기로 합니다.
그때까지 `stage1.calibrated` 가 `false` 로 나갑니다 — 앱은 이 값을 보고
숫자를 띄울지 정하면 됩니다.

### 3. 데모 서버는 배포용이 아닙니다

인증·레이트 리밋·HTTPS 가 없습니다. 공개 주소에 그대로 띄우지 마세요.
사진은 **디스크에 저장하지 않고** 메모리에서 처리한 뒤 버립니다
(보호자 사진을 동의 없이 모으지 않기 — `cautions/03` §8).

### 4. 모델이 아직 확정 전입니다

2단계 백본은 서브셋 비교에서 `convnextv2_base` 가 앞섰지만 **풀 학습 확인이
남았습니다** (STATUS.md 열린 문제 2번). 계약은 안 바뀌니 앱 작업은 지금
시작해도 되고, 가중치만 나중에 갈아 끼우면 됩니다.

---

## 데모 화면 = 앱의 챗봇 카드

`demo/index.html` 은 **DAENGS 앱의 챗봇 화면을 그대로 옮긴 것**입니다
([`gayeoniee/isometric_test`](https://github.com/gayeoniee/isometric_test) —
`app/src/main/java/com/daengs/app/ui/home/ChatbotCard.kt`).

### 동선

```
챗봇 카드 ─ [📷 이미지 진단] 눌러 사진 선택
              ↓
          내 말풍선에 사진이 뜸 (오른쪽, 핑크)
              ↓
          🐶 발자국 세 개가 통통 (타이핑 표시)
              ↓
          POST /v1/screen
              ↓
          봇 말풍선에 결과 카드 — 이상 가능성 게이지 →
          "판단할 수 없어요" → 여섯 줄 분포 → 진료 권함 → 면책
```

칩을 누르면 입력창에 채워지는 동작은 `ChatbotCard.kt` 와 같습니다.
**텍스트 대화는 아직 없습니다** — 보내면 "지금은 피부 사진만 볼 수 있어" 로
정직하게 답하고 이미지 진단으로 안내합니다.

### 색

`:root` 블록의 값이 전부 `ui/theme/Color.kt` 에서 그대로 온 것입니다:

| 데모 토큰 | Color.kt |
|---|---|
| `--cream-bg` `#FDF1EC` | `CreamBg` |
| `--card` `#FFFFFF` | `CardWhite` |
| `--pink` `#F0A0A0` | `DaengPink` |
| `--pink-deep` `#E08585` | `DaengPinkDeep` |
| `--pink-soft` `#FBE4E0` | `PinkSoft` |
| `--pink-faint` `#F7ECE8` | `PinkFaint` |
| `--ink` `#4A3B36` | `TextDark` |
| `--muted` `#A79089` | `TextMuted` |
| `--safe` `#7FC98F` / `--alert` `#E87F7F` | `RoomPalette.GhostValid` / `GhostInvalid` |

**다크 모드를 넣지 않았습니다** — 앱이 라이트 전용이기 때문입니다
(`Theme.kt`: "dynamicColor 를 지원하지 않는다 … 다크 모드도 동일 스킴").
대신 모든 색을 명시해서 브라우저 테마와 무관하게 같게 보입니다.

### Compose 로 옮길 때

결과 카드를 `ChatbotCard.kt` 옆에 붙일 때 그대로 쓸 수 있는 값들:

* 말풍선 모서리 — 봇 `19/19/19/5.dp`, 나 `19/19/5/19.dp` (앱 카드가 22.dp)
* 분포 막대 — 트랙 `PinkFaint`, 채움 `DaengPink`, 높이 6.dp, 완전 둥글게
* 분포 이름은 `FontWeight.Normal` **고정** — 1등만 Bold 로 바꾸지 마세요
* 판정 점 색 — 이상 `GhostInvalid`, 정상 `GhostValid`, 재촬영 `TextMuted`

⚠️ 한국어 줄바꿈은 어절 단위로 끊어야 합니다. 웹에서는 `word-break:keep-all`,
Compose 에서는 `LineBreak.Paragraph` 를 쓰세요 — 기본값이면
"권합니 / 다." 처럼 낱말 가운데가 잘립니다 (실제로 그렇게 나왔습니다).
