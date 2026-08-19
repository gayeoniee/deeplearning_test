"""Windows(cp949) 인코딩 회귀 테스트.

이 프로젝트는 같은 종류의 인코딩 버그로 세 번 막혔습니다:

  1. requirements.txt 에 한글 → Windows pip 이 cp949 로 읽다 UnicodeDecodeError
  2. 콘솔에 ✅ ⚠️ 출력 → cp949 로 인코딩 못 해 UnicodeEncodeError
  3. aihubshell(UTF-8) 출력을 subprocess 로 읽음 → cp949 로 디코딩하다 UnicodeDecodeError

전부 "인코딩을 명시하지 않으면 시스템 로케일이 쓰인다" 는 같은 원인입니다.
리눅스/맥에서는 UTF-8 이 기본이라 절대 재현되지 않아서, 테스트로 못 박아둡니다.

실행:
    python tests/test_windows_encoding.py
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
def test_requirements_is_ascii() -> None:
    """pip 이 로케일 인코딩으로 읽으므로 순수 ASCII 여야 합니다."""
    data = (ROOT / "requirements.txt").read_bytes()
    bad = [(i, b) for i, b in enumerate(data) if b > 127]
    check("requirements.txt is pure ASCII", not bad,
          f"non-ASCII at {bad[:3]}")
    try:
        data.decode("cp949")
        ok = True
    except UnicodeDecodeError as e:
        ok, detail = False, str(e)
    check("requirements.txt decodes as cp949", ok, locals().get("detail", ""))


def test_subprocess_calls_declare_encoding() -> None:
    """text=True 로 subprocess 를 쓰면 encoding 을 반드시 명시해야 합니다."""
    for py in sorted(ROOT.rglob("*.py")):
        if ".git" in py.parts or py.parent.name == "tests":
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


if __name__ == "__main__":
    print("Windows(cp949) encoding regression tests\n")
    for fn in (test_requirements_is_ascii,
               test_subprocess_calls_declare_encoding,
               test_utf8_child_output_decodes,
               test_invalid_bytes_do_not_crash,
               test_console_fix_is_installed,
               test_cp949_console_would_not_crash):
        fn()
    print()
    if FAILS:
        print(f"{len(FAILS)} check(s) failed:")
        for f in FAILS:
            print(f"  - {f}")
        sys.exit(1)
    print("all checks passed")
