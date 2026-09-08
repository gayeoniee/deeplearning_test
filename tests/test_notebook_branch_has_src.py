"""노트북이 못 박은 브랜치에 **그 노트북이 import 하는 모듈이 실제로 있는가.**

    uv run python tests/test_notebook_branch_has_src.py

## 왜 이 검사가 생겼나

노트북 13 을 만들 때 환경 셀을 12번에서 **그대로 복사**했습니다 — 클론·검증
로직을 베끼면 그쪽이 고쳐져도 안 따라오는 함정을 피하려고요. 그런데 그 셀에는
`NB_BRANCH` 도 같이 들어 있었고, 12번은
`claude/dog-disease-diagnosis-model-1s6jtf`(main 보다 44 커밋 뒤)를 가리킵니다.
거기엔 `src/detect.py` 가 **없습니다.**

캐글은 데이터를 붙이고 개체를 갈라 놓고 **56초 뒤에** 죽었습니다:

    ModuleNotFoundError: No module named 'src.detect'

⚠️ 그 앞에서 첫 셀은 **`[nb] 노트북 최신 (2026-09-04.5)`** 를 태연히 찍었습니다.
버전 대조는 *노트북 셀 vs 리포* 를 보지 *브랜치가 맞는지* 는 안 봅니다 —
이미 적어둔 함정("첫 셀이 코드 버전을 찍는다고 그게 최신이란 뜻은 아니다")의
**다른 얼굴**입니다.

## 무엇을 보나

각 노트북에서 `NB_BRANCH` 와 `from src.<모듈>` 을 뽑아, 그 브랜치의 git 트리에
`src/<모듈>.py` 가 있는지 봅니다. 그 ref 가 로컬에 없으면 **건너뛰되 말합니다**
— 조용히 통과하면 이 검사도 아무 일을 안 하게 됩니다.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NBDIR = ROOT / "notebooks"

ok = fail = skip = 0


def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {name}")
    else:
        fail += 1
        print(f"  FAIL  {name}" + (f"\n        {detail}" if detail else ""))


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args],
                          capture_output=True, text=True)


def _resolve(branch: str) -> str | None:
    """브랜치 이름을 로컬에서 읽을 수 있는 ref 로. 없으면 None."""
    for ref in (branch, f"origin/{branch}", f"refs/remotes/origin/{branch}"):
        if _git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}").returncode == 0:
            return ref
    return None


def _imports(text: str) -> set[str]:
    mods = set(re.findall(r"from\s+src\.(\w+)\s+import", text))
    mods |= set(re.findall(r"from\s+src\s+import\s+([\w,\s]+)", text)) and set()
    for grp in re.findall(r"from\s+src\s+import\s+([^\n\\]+)", text):
        for piece in grp.split(","):
            piece = piece.strip().split(" as ")[0].strip()
            if re.fullmatch(r"\w+", piece):
                mods.add(piece)
    return mods


print("[1] 노트북마다 못 박은 브랜치에 import 하는 src 모듈이 있는가")
nbs = sorted(NBDIR.glob("*.ipynb"))
check("노트북을 찾았다", bool(nbs), str(NBDIR))

for nb in nbs:
    cells = json.loads(nb.read_text(encoding="utf-8"))["cells"]
    text = "\n".join("".join(c["source"]) for c in cells)
    m = re.search(r'^NB_BRANCH\s*=\s*"([^"]+)"', text, re.M)
    if not m:
        continue
    branch = m.group(1)
    mods = {x for x in _imports(text) if (ROOT / "src" / f"{x}.py").exists()
            or not (ROOT / "src" / x).is_dir()}
    if not mods:
        continue
    ref = _resolve(branch)
    if ref is None:
        skip += 1
        print(f"  SKIP  {nb.name} — 브랜치 '{branch}' 가 로컬에 없습니다 "
              f"(`git fetch origin {branch}` 뒤 다시 도세요)")
        continue
    missing = sorted(
        mod for mod in mods
        if _git("cat-file", "-e", f"{ref}:src/{mod}.py").returncode != 0)
    check(f"{nb.name} → {branch}", not missing,
          f"그 브랜치에 없는 모듈: {', '.join(missing)}\n"
          f"        → 브랜치를 고치거나 그 모듈을 그 브랜치에 올리세요")

print("\n[2] 검출 노트북(13)은 src/detect.py 가 있는 브랜치를 봐야 한다")
nb13 = NBDIR / "13_병변_검출기.ipynb"
if nb13.is_file():
    t = "\n".join("".join(c["source"])
                  for c in json.loads(nb13.read_text(encoding="utf-8"))["cells"])
    m = re.search(r'^NB_BRANCH\s*=\s*"([^"]+)"', t, re.M)
    check("NB_BRANCH 가 있다", m is not None)
    if m:
        ref = _resolve(m.group(1))
        if ref is None:
            skip += 1
            print(f"  SKIP  브랜치 '{m.group(1)}' 가 로컬에 없습니다")
        else:
            check(f"'{m.group(1)}' 에 src/detect.py 가 있다",
                  _git("cat-file", "-e", f"{ref}:src/detect.py").returncode == 0)
    check("★ 브랜치가 어디인지 사람 눈에도 보이게 적혀 있다",
          "ModuleNotFoundError" in t)

print("\n" + "=" * 60)
print(f" 통과 {ok} / {ok + fail}" + (f"   (건너뜀 {skip})" if skip else ""))
sys.exit(1 if fail else 0)
