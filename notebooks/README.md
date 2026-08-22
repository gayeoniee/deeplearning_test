# 노트북 — 무엇을 언제 돌리나

⚠️ **Kaggle 노트북은 하나에 하나씩.** 한 노트북에 03 을 지우고 05 를 붙여넣으면
05 에 붙일 03 이 **없어집니다** (자기 자신은 입력이 안 됩니다). 실제로 막혔습니다.

| 파일 | Kaggle 노트북 이름(권장) | 학습 | 시간 | 상태 |
|---|---|---|---|---|
| `03_학습_베이스라인` | `dogskin-03` | 1·2단계 | ~2시간 20분 | ✅ 끝 (m2.5) |
| `03b_증강_빠른스윕` | `dogskin-03b` | 서브셋 7종 | ~3시간 | ✅ 끝 — 다시 안 돌림 |
| `03c_크롭비교` | `dogskin-03c` | 2단계 ×2 | ~2시간 40분 | ✅ 끝 — 다시 안 돌림 |
| `05_평가_보정_GradCAM` | `dogskin-05` | **없음** | ~20분 | ← 지금 |
| `04_학습_최신모델_비교` | `dogskin-04` | 백본 6종 | 길다 | 05 다음 |

`00`~`02` 가 없는 이유: 데이터 확보·전처리는 노트북이 아니라 **한국 PC 의
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
| 03 | `dogskin-full`, `dogskin-m25` |
| 03b / 03c | `dogskin-m15`, `dogskin-m25` |
| 04 / 05 | `dogskin-full`, `dogskin-m25`, **03 의 release 데이터셋** |

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

넘기는 법:
```
03 노트북 → [Output] 탭 → release 폴더 확인 → [New Dataset] → Private → Create
05 노트북 → [Add Input] → 그 데이터셋
```

`train.import_previous_run()` 이 경로를 알아서 찾습니다. 못 찾으면 05 첫 셀의
`PREV_RUN` 에 경로를 직접 적으면 됩니다.

---

## 자주 막히는 곳

| 증상 | 원인 | 해결 |
|---|---|---|
| `stage1_threshold.json 이 없습니다` | 03 출력을 안 붙임 | 위 4번 |
| `2단계 크롭이 'm1.5' 입니다` | **예전 버전**의 출력을 붙임 | Version 목록에서 올바른 버전. 낡은 입력은 제거 |
| `크롭 'full' 가 0.0%` | `dogskin-full` 안 붙임 | Add Input |
| `임계값 0.5000` | 인계 실패인데 그냥 진행 | 이제 멈춥니다 (조용히 틀리는 게 최악) |
| 8분째 아무 출력 없음 | 크롭 세는 중 (네트워크 마운트) | 태그당 ~2분. 정상 |
| 노트북 버전 경고 | 셀이 낡음 | `.ipynb` 다시 Import |

**올바른 버전인지 확인하는 법**: release 의 `READ_ME_FIRST.txt` 를 열어보세요.
크롭과 점수가 적혀 있습니다. 이름만 보고 고르면 틀립니다 — 실제로 틀렸습니다.
