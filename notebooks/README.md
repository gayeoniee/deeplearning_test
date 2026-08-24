# 노트북 — 무엇을 언제 돌리나

## 🚨 holdout 을 여는 노트북은 **06 하나뿐입니다**

holdout 은 학습에도 모델 선택에도 안 쓴 마지막 시험지입니다. 열어보고 설정을
바꾸면 그 순간 holdout 도 val 이 됩니다.

| | 노트북 |
|---|---|
| 설정을 **고르는** 실험 | 03b · 03c · 03d · 03e · 04 — holdout 안 엶 |
| 고른 설정을 **확정 측정** | **06** — holdout 엶 |

⚠️ 05 는 06 이 대체합니다. 05 는 인계 문제로 별도 세션이 필요했는데,
06 은 학습과 평가가 한 세션이라 인계가 아예 없습니다.

---

⚠️ **Kaggle 노트북은 하나에 하나씩.** 한 노트북에 03 을 지우고 05 를 붙여넣으면
05 에 붙일 03 이 **없어집니다** (자기 자신은 입력이 안 됩니다). 실제로 막혔습니다.

| 파일 | Kaggle 노트북 이름(권장) | 학습 | 시간 | 상태 |
|---|---|---|---|---|
| `03_학습_베이스라인` | `dogskin-03` | 1·2단계 | ~2시간 20분 | ✅ 끝 (m2.5) |
| `03b_증강_빠른스윕` | `dogskin-03b` | 서브셋 7종 | ~3시간 | ✅ 끝 — 다시 안 돌림 |
| `03c_크롭비교` | `dogskin-03c` | 2단계 ×2 | ~2시간 40분 | ✅ 끝 — 다시 안 돌림 |
| `05_평가_보정_GradCAM` | `dogskin-05` | **없음** | ~25분 | ✅ 끝 (촬영 가이드 확정) |
| `03d_1단계_고치기` | `dogskin-03d` | 1단계 2×2 | ~2시간 | ✅ 끝 (effnetv2_s+photometric) |
| `04_학습_최신모델_비교` | `dogskin-04` | 2단계 백본 3종 | ~3시간 20분 | ◐ 끝 — 기준선 미수렴, 판정 보류 |
| `03e_1단계_f320비교` | `dogskin-03e` | 1단계 입력 2종 | ~1시간 53분 | ✅ 끝 — **f320 채택** |
| **`06_확정재학습_홀드아웃`** | `dogskin-06` | 1·2단계 풀 + **holdout** | ~4시간 | ← 지금 (03+05 합친 것, 인계 없음) |

`00`~`02` 가 없는 이유: 데이터 확보·전처리는 노트북 말고 **한국 PC 의
`prepare_local.py`** 가 합니다 (AI Hub 가 해외 IP 를 막습니다).

---

## Kaggle 실행 절차

### 1. 노트북 만들기
`New Notebook` → 해당 `.ipynb` **Import** → Settings 에서 `Accelerator: GPU T4 x2`

⚠️ 첫 셀이 `MY_NOTEBOOK_VERSION` 과 `config.NOTEBOOK_VERSION` 을 비교합니다.
**노트북 셀은 `git pull` 로 안 바뀝니다** — 경고가 뜨면 `.ipynb` 를 다시 Import 하세요.
(`src/` 는 첫 셀의 `git clone` 으로 매번 최신입니다)

### 2. 입력 붙이기 (Add Input)

| 노트북 | 붙일 것 |
|---|---|
| 03 | **`dogskin-f320`**, `dogskin-m25`, (선택) **이전 release** |
| **06** | **`dogskin-f320`**, `dogskin-m25` ← **release 안 붙입니다** (인계 없음) |
| | ↳ 1단계 f320 확정(STEP 9-A). 크롭 두 개는 **학습만이 아니라 평가·holdout 추론에도** 씁니다 — 빼면 3시간 학습 뒤에 죽습니다 |
| | ↳ ⚠️ 08-23 이전 release 는 `result.json` 이 없어 **붙여도 건너뛰지 않습니다** (`b588664` 이후 것부터 됩니다) |
| 03b / 03c | `dogskin-m15`, `dogskin-m25` |
| 03d | `dogskin-full` (1단계는 full 크롭을 씁니다) |
| 04 | `dogskin-m25` (2단계 백본 비교 — 1단계는 안 건드림) |
| 03e | `dogskin-full`, `dogskin-f320` (2단계는 안 돌립니다) ✅ 끝 — f320 채택 |
| 05 | **`dogskin-f320`**, `dogskin-m25`, **03 의 release 데이터셋** |

⚠️ 데이터셋 업로드는 **반드시 Private**. AI Hub 데이터는 재배포 금지입니다.

### 3. 돌리기 — 반드시 Commit

```
[Save Version] → "Save & Run All (Commit)"
```

그냥 `Run` 은 세션이 끝나면 출력이 사라집니다. Commit 이어야 브라우저를 닫아도
끝까지 돌고 **출력이 보존**됩니다.

### 4. 다음 노트북에 넘기기 — `release/` 폴더

학습 노트북은 끝에 `train.export_release()` 로 **넘길 것만 한 폴더에** 모읍니다:

```
/kaggle/working/release/
  READ_ME_FIRST.txt          ← 크롭·점수가 적혀 있음. 열어서 확인하세요
  stage1_threshold.json
  reports/step4a_summary.json
  checkpoints/<실험>/best.pt
```

> **왜 이 폴더가 따로 있나** — Kaggle 출력을 데이터셋으로 만들면 안쪽 폴더가
> 빠지는 일이 있습니다. 실제로 `checkpoints/` 는 왔는데
> `data/work/stage1_threshold.json` 은 안 왔습니다. 무거운 쪽이 오고 가벼운
> 쪽이 빠졌습니다. 그래서 넘길 것만 골라 **최상위**에 둡니다.

⚠️ **Kaggle 은 Output 에서 폴더 하나만 못 빼냅니다.** 그래서 zip 도 같이 만듭니다.

넘기는 법 — 둘 중 하나:

**① release.zip 만 받아서 올리기** (권장, 가볍습니다)
```
03 노트북 → [Output] 탭 → release.zip 다운로드
        → [New Dataset] 으로 그 zip 업로드 → Private → Create
   (Kaggle 이 업로드할 때 알아서 풉니다. 폴더 구조가 그대로 살아납니다)
05 노트북 → [Add Input] → 그 데이터셋
```

**② 출력 전체를 데이터셋으로** (간단하지만 1GB 안팎)
```
03 노트북 → [Output] 탭 → [New Dataset] → Private → Create
```
`import_previous_run()` 이 입력 전체를 깊이 4까지 뒤지므로 이것도 동작합니다.
다만 `last.pt`(옵티마이저 상태)까지 딸려와 무겁고 붙이는 데 오래 걸립니다.

⚠️ 데이터셋 크기가 **10GB 를 넘으면** 크롭 심볼릭 링크가 실제 파일로 풀린 것입니다.
그때는 ①번(zip)으로 가세요.

`train.import_previous_run()` 이 경로를 알아서 찾습니다. 못 찾으면 05 첫 셀의
`PREV_RUN` 에 경로를 직접 적으면 됩니다.

---

## 자주 막히는 곳

| 증상 | 원인 | 해결 |
|---|---|---|
| `stage1_threshold.json 이 없습니다` | 03 출력을 안 붙임 | 위 4번 |
| `2단계 크롭이 'm1.5' 입니다` | **예전 버전**의 출력을 붙임 | Version 목록에서 올바른 버전. 낡은 입력은 제거 |
| `크롭 'full' 가 0.0%` | `dogskin-full` 안 붙임 | Add Input |
| 1단계가 조용히 `full` 로 감 | `dogskin-f320` 안 붙임 | Add Input — 멈추지 않으니 로그를 보세요 |
| `임계값 0.5000` | 인계 실패인데 그냥 진행 | 이제 멈춥니다 (조용히 틀리는 게 최악) |
| 8분째 아무 출력 없음 | 크롭 세는 중 (네트워크 마운트) | 태그당 ~2분. 정상 |
| 몇 시간 걸릴지 모르겠음 | — | 학습 노트북이 시작 전에 `estimate_runtime()` 으로 예상 시간을 찍습니다 |
| 노트북 버전 경고 | 셀이 낡음 | `.ipynb` 다시 Import |

**올바른 버전인지 확인하는 법**: release 의 `READ_ME_FIRST.txt` 를 열어보세요.
크롭과 점수가 적혀 있습니다. 이름만 보고 고르면 틀립니다 — 실제로 틀렸습니다.
