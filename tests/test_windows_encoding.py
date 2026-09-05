"""Windows(cp949) 인코딩 회귀 테스트.

이 프로젝트는 같은 종류의 인코딩 버그로 세 번 막혔습니다:

  1. requirements.txt 에 한글 → Windows pip 이 cp949 로 읽다 UnicodeDecodeError
     (→ pyproject.toml + uv 로 옮겨 해결. TOML 은 항상 UTF-8 입니다)
  2. 콘솔에 ✅ ⚠️ 출력 → cp949 로 인코딩 못 해 UnicodeEncodeError
  3. aihubshell(UTF-8) 출력을 subprocess 로 읽음 → cp949 로 디코딩하다 UnicodeDecodeError

전부 "인코딩을 명시하지 않으면 시스템 로케일이 쓰인다" 는 같은 원인입니다.
리눅스/맥에서는 UTF-8 이 기본이라 절대 재현되지 않아서, 테스트로 못 박아둡니다.

실행:
    uv run python tests/test_windows_encoding.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        if detail:
            print(f"        {detail}")
        FAILS.append(name)


# ──────────────────────────────────────────────────────────────
def test_pyproject_is_utf8() -> None:
    """의존성 선언은 pyproject.toml 입니다 (requirements.txt 를 대체했습니다).

    TOML 규격은 파일을 **항상 UTF-8** 로 읽도록 못 박고 있고 uv 도 그렇게 읽으므로,
    requirements.txt 때와 달리 한글 주석을 넣어도 안전합니다. 여기서는 그 전제가
    실제로 성립하는지(=UTF-8 로 파싱되는지)만 확인합니다.
    """
    try:
        import tomllib
    except ModuleNotFoundError:                      # Python 3.10
        check("pyproject.toml parses as UTF-8 TOML", True, "tomllib 없음 — 건너뜀")
        return

    data = (ROOT / "pyproject.toml").read_bytes()
    try:
        cfg = tomllib.loads(data.decode("utf-8"))
        ok, detail = True, ""
    except (UnicodeDecodeError, Exception) as e:     # noqa: B014
        ok, detail = False, str(e)
    check("pyproject.toml parses as UTF-8 TOML", ok, detail)
    if not ok:
        return

    # uv 가 빌드를 시도하지 않아야 합니다 (src/ 는 설치 패키지가 아니라 경로 임포트)
    check("uv is told not to build this repo as a package",
          cfg.get("tool", {}).get("uv", {}).get("package") is False,
          "[tool.uv] package = false 가 없습니다")

    # 로컬 전처리에는 torch 가 필요 없습니다 — 기본 의존성에 들어가면 안 됩니다
    base = " ".join(cfg["project"]["dependencies"]).lower()
    check("torch stays out of the default dependencies",
          "torch" not in base,
          "기본 의존성에 torch 가 있으면 로컬 전처리 설치가 2GB 로 불어납니다")

    check("uv.lock is committed", (ROOT / "uv.lock").exists(),
          "uv.lock 이 없으면 팀원마다 다른 버전이 깔립니다")


def test_subprocess_calls_declare_encoding() -> None:
    """text=True 로 subprocess 를 쓰면 encoding 을 반드시 명시해야 합니다."""
    # ⚠️ 우리 코드만 봅니다. .venv/ 안의 서드파티 .py 에는 UTF-8 이 아닌 파일이
    #    섞여 있어서(테스트 픽스처 등) read_text 가 그냥 죽습니다.
    skip = {".git", ".venv", "venv", "__pycache__", "site-packages", "node_modules"}
    for py in sorted(ROOT.rglob("*.py")):
        if skip & set(py.parts) or py.parent.name == "tests":
            continue
        src = py.read_text(encoding="utf-8")
        for m in re.finditer(r"subprocess\.(run|Popen)\((.*?)\)\n", src, re.S):
            call = m.group(2)
            if "text=True" not in call:
                continue
            has_enc = "encoding=" in call or "**dec" in call or "dec)" in call
            line = src[: m.start()].count("\n") + 1
            check(f"{py.relative_to(ROOT)}:{line} declares encoding", has_enc,
                  "text=True without encoding= falls back to the system locale")


def test_utf8_child_output_decodes() -> None:
    """UTF-8 로 한글을 뱉는 자식 프로세스를 우리 방식대로 읽으면 성공해야 합니다."""
    child = (
        "import sys;"
        "sys.stdout.buffer.write("
        "'AI 허브는 해외에서의 데이터 다운로드를 제한하고 있습니다\\n'.encode('utf-8'))"
    )
    p = subprocess.Popen([sys.executable, "-c", child], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace")
    out = "".join(p.stdout)
    p.wait()
    check("UTF-8 child output decodes", "해외에서의" in out, repr(out))


def test_invalid_bytes_do_not_crash() -> None:
    """깨진 바이트가 섞여도 예외 없이 넘어가야 합니다 (errors='replace')."""
    child = r"import sys; sys.stdout.buffer.write(b'\xff\xfe partial\n')"
    try:
        p = subprocess.Popen([sys.executable, "-c", child], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace")
        "".join(p.stdout)
        p.wait()
        ok = True
    except UnicodeDecodeError as e:
        ok, detail = False, str(e)
    check("invalid bytes survive errors='replace'", ok, locals().get("detail", ""))


def test_console_fix_is_installed() -> None:
    """콘솔 인코딩 보정이 import 시점에 걸려 있어야 합니다."""
    env_src = (ROOT / "src" / "env.py").read_text(encoding="utf-8")
    check("src/env.py fixes console encoding",
          "_fix_console_encoding()" in env_src and "SetConsoleOutputCP" in env_src)
    pl_src = (ROOT / "prepare_local.py").read_text(encoding="utf-8")
    check("prepare_local.py fixes console encoding before importing src",
          "SetConsoleOutputCP" in pl_src
          and pl_src.index("SetConsoleOutputCP") < pl_src.index("from src import"))


def test_cp949_console_would_not_crash() -> None:
    """cp949 스트림에 이모지를 써도 errors='replace' 면 죽지 않아야 합니다."""
    import io

    raw = io.TextIOWrapper(io.BytesIO(), encoding="cp949", errors="replace")
    try:
        raw.write("완료 [OK] ✅ ⭐ ─")
        ok = True
    except UnicodeEncodeError as e:
        ok, detail = False, str(e)
    check("emoji on cp949 stream survives", ok, locals().get("detail", ""))



def test_dataloader_workers_are_few_on_windows():
    """윈도우에서 DataLoader 워커를 적게 씁니다 — 안 그러면 RAM 이 터집니다.

    2026-09-05 실측 (RTX 3050 / RAM 15.9GB, 여유 8.4GB):
    `suggest_workers()` 가 8 을 돌려줘서 train 8 + val 8 = **워커 16개**가
    각 ~690MB 씩 상주했습니다 (윈도우는 spawn 이라 워커마다 torch·timm·src 를
    통째로 다시 올립니다). 합계 **약 11GB** 로 여유를 넘겨 **교착**했습니다 —
    GPU 97% → 1%, 메모리는 3.6GB 를 잡은 채 CPU 시간이 584.2초에서 멈췄습니다.
    (`persistent_workers` 는 이미 켜져 있어서 재생성 문제는 아니었습니다.)

    ⚠️ 리눅스(Colab/Kaggle)는 `fork` 라 사정이 다릅니다 — 거기 값은 건드리면
       안 됩니다. 그래서 **플랫폼별로 갈라** 두었고 이 검사가 그걸 지킵니다.
    """
    import sys as _sys

    sys.path.insert(0, str(ROOT))
    from src import env as _env

    n = _env.suggest_workers()
    if _sys.platform == "win32":
        check("windows: DataLoader workers <= 2", n <= 2, f"suggest_workers()={n}")
    else:
        check("non-windows: workers unchanged (>=2)", n >= 2, f"suggest_workers()={n}")

    # 환경변수로 덮어쓸 수 있어야 합니다 (0 = 메인 프로세스에서 로딩)
    import importlib
    import os

    old = os.environ.get("DOG_SKIN_WORKERS")
    try:
        os.environ["DOG_SKIN_WORKERS"] = "0"
        importlib.reload(_env)
        check("DOG_SKIN_WORKERS=0 honoured", _env.suggest_workers() == 0,
              str(_env.suggest_workers()))
    finally:
        if old is None:
            os.environ.pop("DOG_SKIN_WORKERS", None)
        else:
            os.environ["DOG_SKIN_WORKERS"] = old
        importlib.reload(_env)

if __name__ == "__main__":
    print("Windows(cp949) encoding regression tests\n")
    for fn in (test_pyproject_is_utf8,
               test_subprocess_calls_declare_encoding,
               test_utf8_child_output_decodes,
               test_invalid_bytes_do_not_crash,
               test_console_fix_is_installed,
               test_cp949_console_would_not_crash,
               test_dataloader_workers_are_few_on_windows):
        fn()
    print()
    if FAILS:
        print(f"{len(FAILS)} check(s) failed:")
        for f in FAILS:
            print(f"  - {f}")
        sys.exit(1)
    print("all checks passed")
