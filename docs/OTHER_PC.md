# 다른 PC 에서 청크 하나 처리해서 가져오기

이 PC 는 디스크가 모자라 TL01/TL02 를 못 받습니다
(다 치워도 92.9GB, 필요한 건 90.2~170.2GB — `STATUS.md` 참고).
**여유가 넉넉한 다른 PC 에서 크롭까지 하고, 크롭만 가져오면** 됩니다.

크롭은 10GB 정도라 옮기기 쉽습니다. zip 80GB 는 그 PC 에서 자동으로 지워집니다.

---

## 그 PC 가 갖춰야 할 것

| | |
|---|---|
| **위치** | **한국** ⚠️ AI Hub 가 해외 IP 를 막습니다. VPN 우회 금지 |
| 디스크 여유 | **최소 90GB**, 안전하게 **170GB** (`--plan` 이 판정해 줍니다) |
| Python | 3.10 이상 + [uv](https://docs.astral.sh/uv/) |
| API 키 | AI Hub 활용신청 승인된 계정의 키 |

---

## 그 PC 에서 (순서대로)

```bash
git clone https://github.com/gayeoniee/deeplearning_test
cd deeplearning_test
git checkout claude/dog-disease-diagnosis-model-1s6jtf
uv sync

# 1) 용량이 되는지 먼저 봅니다 (크롭이 없어도 계산해 줍니다)
uv run python tools/crops.py --plan TL02

# 2) 되면 받아서 크롭까지 — 쓰는 크롭 2종만 만듭니다
set AIHUB_API_KEY=키
uv run python prepare_local.py --chunk TL02 --margins 2.5,-320
```

`--margins 2.5,-320` 이 중요합니다. 기본값은 크롭을 4종 만드는데
지금 파이프라인은 `f320`·`m2.5` 둘만 씁니다 — 절반이 낭비입니다.

⚠️ **`--finalize` 는 여기서 돌리지 마세요.** 그건 모든 청크를 합쳐 분할하는
단계라, VL01 이 없는 PC 에서 돌리면 TL02 만으로 분할이 잡힙니다.

중간에 멈춰도 괜찮습니다. `--chunk` 는 이미 받은 건 건너뜁니다.

---

## 가져올 것 (두 가지)

```
data/work/crops/f320/         ← 새로 생긴 것
data/work/crops/m2.5/         ← 새로 생긴 것
data/work/manifests/chunk_TL02.parquet   ★ 이거 빠뜨리기 쉽습니다
```

**매니페스트를 꼭 챙기세요.** 크롭 파일만 있으면 어떤 사진이 어떤 라벨인지
알 수가 없습니다.

zip 으로 묶어서 옮기면 편합니다:

```bash
uv run python prepare_local.py --package --tags f320,m2.5 --out tl02_crops.zip
```

옮기는 수단은 **USB/외장하드가 제일 빠릅니다** (10GB). 구글 드라이브도 되지만
업로드 + 다운로드로 두 번 기다려야 합니다.

---

## 이 PC 로 돌아와서

```bash
# 1) 받아온 것을 제자리에 풉니다
#    crops/f320, crops/m2.5 는 기존 폴더에 **덮어쓰지 말고 합치기**
#    manifests/chunk_TL02.parquet 도 같은 자리에

# 2) 이제 합칩니다 — VL01 + TL02
uv run python prepare_local.py --finalize

# 3) 학습용 zip
uv run python prepare_local.py --package --tags f320,m2.5
```

`--finalize` 가 두 청크를 합쳐 중복 제거하고 개체 단위로 다시 나눕니다.

---

## ⚠️ 합치고 나면 달라지는 것

**holdout 이 일부 바뀝니다.**

holdout 배정은 그룹 해시라 결정론적이고, `animal_id`(견종·나이·성별·날짜 대용)도
안정적입니다. 그런데 그룹은 `dup_cluster`(phash)와의 **합집합**이라,
TL02 사진이 VL01 의 두 클러스터를 이으면 그 그룹의 배정이 바뀝니다.

→ **지금까지 잰 숫자 중 일부는 비교가 깨집니다.** `--finalize` 직후
`split.verify()` 출력과 holdout 크기를 `STATUS.md` 에 새로 적고,
기준선을 다시 잡으세요. 옛 숫자와 새 숫자를 섞어 쓰면 안 됩니다.

---

## 안 되면

디스크가 90GB 도 안 나오면 이 길은 접고, 다음 주 06 재실행
(2단계 백본 교체 + 1단계 보정)만 하는 게 맞습니다. 그것만으로도
열린 문제 세 개가 닫힙니다 — `STATUS.md` 의 "다음 할 일" 참고.
