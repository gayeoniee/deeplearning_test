# ⚠️ Windows 로컬 환경 설정

> **읽는 시점**: 로컬 PC(Windows)에서 `prepare_local.py` 를 돌리기 전

---

## 필요한 것 두 가지

| | 왜 |
|---|---|
| **Python 3.10+** | 스크립트 실행 |
| **Git for Windows** | 리포 clone + **Git Bash** (aihubshell 실행에 필요) |

---

## 1. Python 설치

### 이미 있는지 확인

```cmd
python --version
py --version
```

- 버전이 나오면 → 설치돼 있습니다. 아래 "pip 이 없다고 나올 때" 로
- **Microsoft Store 가 열리면** → 진짜 Python 이 아니라 스토어 안내입니다. 아래대로 설치하세요
- `'python'은(는) 내부 또는 외부 명령...` → 설치 필요

### 설치

https://www.python.org/downloads/ 에서 최신 3.x 다운로드

> 🚨 설치 첫 화면에서 **"Add python.exe to PATH"** 체크박스를 **반드시 켜세요.**
> 이걸 놓치면 `python`, `pip` 명령이 안 잡힙니다. (가장 흔한 실수)

설치 후 **명령 프롬프트를 새로 열고** 확인:

```cmd
python --version
pip --version
```

### `pip` 이 없다고 나올 때

Python 은 있는데 `pip` 만 안 잡히는 경우입니다. `py -m pip` 로 우회하세요:

```cmd
py -m pip install -r requirements.txt
```

`py` 는 Python 공식 설치 프로그램이 함께 넣어주는 실행기라, PATH 설정이 꼬여 있어도
대부분 동작합니다. **이 프로젝트의 모든 명령에 `py -m` 을 붙여 쓰면 됩니다:**

```cmd
py prepare_local.py --chunk VL01
```

그래도 안 되면 PATH 를 고치는 것보다 **Python 을 "Add to PATH" 체크하고 재설치**하는 게 빠릅니다.

---

## 2. Git for Windows — Git Bash 가 필요합니다

### 왜

`aihubshell` 은 AI Hub 가 제공하는 **bash 스크립트**입니다.
Windows 의 cmd / PowerShell 에서는 실행되지 않습니다.

이 프로젝트는 **Git Bash 를 자동으로 찾아서** 그걸로 실행합니다.
리포를 `git clone` 했다면 이미 설치돼 있을 가능성이 높습니다.

### 확인

```cmd
git --version
```

없으면 https://git-scm.com/download/win 에서 설치 (기본 옵션 그대로 두면 됩니다).

### 자동 탐색 순서

`src/aihub.py` 의 `find_bash()` 가 아래를 차례로 봅니다:

```
1. PATH 의 bash
2. C:\Program Files\Git\bin\bash.exe
3. C:\Program Files (x86)\Git\bin\bash.exe
4. %LOCALAPPDATA%\Programs\Git\bin\bash.exe
5. wsl.exe
```

하나라도 찾으면 자동으로 씁니다. 못 찾으면 설치 안내를 띄우고 멈춥니다.

### WSL 을 쓰고 싶다면

```powershell
wsl --install      # PowerShell 관리자 권한
```

WSL 안에서 아예 전부 작업해도 됩니다. 그 경우 리눅스 절차를 그대로 따르면 됩니다.

---

## 3. API 키 설정 (Windows)

셸마다 문법이 다릅니다.

```cmd
:: 명령 프롬프트 (cmd) — 이 창에서만 유효
set AIHUB_API_KEY=발급받은키
```

```powershell
# PowerShell — 이 창에서만 유효
$env:AIHUB_API_KEY="발급받은키"
```

```cmd
:: 영구 설정 (새 창부터 적용)
setx AIHUB_API_KEY "발급받은키"
```

> ⚠️ `setx` 로 넣으면 **현재 창에는 적용되지 않습니다.** 새 창을 여세요.
> ⚠️ 키를 코드 파일에 쓰지 마세요.

확인:
```cmd
echo %AIHUB_API_KEY%
```

---

## 4. 전체 순서 (Windows 기준)

```cmd
git clone https://github.com/gayeoniee/deeplearning_test.git
cd deeplearning_test

py -m pip install -r requirements.txt

set AIHUB_API_KEY=발급받은키

py prepare_local.py --chunk VL01
py prepare_local.py --finalize
py prepare_local.py --package
```

---

## 자주 막히는 곳

**`'pip'은(는) 내부 또는 외부 명령...`**
→ `py -m pip` 를 쓰세요. 또는 Python 을 "Add to PATH" 켜고 재설치.

**`'python'을 입력했더니 Microsoft Store 가 열림`**
→ Windows 기본 앱 실행 별칭 때문입니다.
설정 → 앱 → 고급 앱 설정 → 앱 실행 별칭 → `python.exe`, `python3.exe` **끄기**.
또는 그냥 `py` 를 쓰세요.

**`aihubshell 은 bash 스크립트라...` 오류**
→ Git for Windows 설치. 설치 후 **명령 프롬프트를 새로 열어야** PATH 가 반영됩니다.

**`pip install -r requirements.txt` 에서 `UnicodeDecodeError: 'cp949' codec can't decode byte 0xe2`**

Windows pip 은 `requirements.txt` 를 **시스템 로케일(한국어 Windows = cp949)** 로 읽습니다.
파일에 한글이나 `─`, `→` 같은 문자가 하나라도 있으면 이 오류로 죽습니다.

→ 이미 고쳤습니다. `git pull` 후 다시 실행하세요.
   (`requirements.txt` 를 순수 ASCII 로 유지합니다. 편집할 때 한글을 넣지 마세요)

**`UnicodeEncodeError: 'cp949' codec can't encode character '\u2705'`**

반대 방향 문제입니다. cmd 가 `✅`, `⚠️`, `★` 같은 문자를 **출력**하지 못해서 죽습니다.

→ 이것도 고쳤습니다. `prepare_local.py` 와 `src/env.py` 가 시작할 때
   콘솔을 UTF-8 로 바꾸고, 그래도 못 찍는 글자는 대체 문자로 넘깁니다.
   로그 한 줄 때문에 몇십 분짜리 전처리가 죽으면 안 되니까요.

수동으로 하려면:
```cmd
chcp 65001
set PYTHONUTF8=1
```

**torch 설치가 너무 오래 걸림**
→ 전처리만 할 거면 torch 는 필요 없습니다. 최소 설치로 충분합니다:
```cmd
py -m pip install numpy pandas pillow scikit-learn imagehash pyarrow tqdm matplotlib
```

**디스크 부족**
→ `--mode zip` 이 기본이라 VL01 은 약 25GB 면 됩니다.
`py prepare_local.py --chunk VL01` 실행 시 필요 용량을 먼저 알려줍니다.

---

**관련 문서**: [`06_해외IP_다운로드_차단_우회.md`](06_해외IP_다운로드_차단_우회.md)
