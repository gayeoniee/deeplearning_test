"""노트북 첫 셀이 **정말 최신 코드를 받는가**.

    uv run python tests/test_notebook_git_sync.py

실제로 당한 것 (2026-09-04): 캐글 클론이 **원격에 더 이상 없는 커밋**
(75445c0) 에 붙박인 채 며칠을 돌았습니다. src/ 를 고쳐 푸시해도 하나도
안 실렸고, 첫 셀은 "코드 버전 …" 을 태연히 찍었습니다.

범인은 두 줄이었습니다:

    subprocess.run([... "fetch" ...], check=False)
    subprocess.run([... "reset", "--hard", f"origin/{BRANCH}"], check=False)

`check=False` 라 실패해도 조용히 넘어갔고, shallow clone 에서 히스토리가
갈리면 `origin/<브랜치>` 가 옛 커밋을 가리킨 채 남습니다.
⚠️ 그 상태에서도 **에러가 하나도 안 납니다.** 그냥 옛 코드가 돕니다.

여기서는 노트북의 **진짜 셀 내용을 꺼내** 임시 git 저장소에 대고 돌립니다.
셀을 흉내 낸 사본을 검사하면 정작 노트북이 낡아도 통과합니다.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
nb = json.loads((ROOT / "notebooks" / "03h_2단계_학습률.ipynb").read_text(encoding="utf-8"))
cell = "".join(nb["cells"][1]["source"])
# 동기화 블록만 떼어냅니다 (BRANCH 정의 ~ _fresh_clone 재시도까지)
i = cell.index("BRANCH = BRANCH or NB_BRANCH")
j = cell.index("os.chdir(DIR)")
block = cell[i:j]

tmp = Path(tempfile.mkdtemp())
origin = tmp / "origin.git"; work = tmp / "seed"

def git(*a, cwd=None): return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True)

git("init", "--bare", "-b", "main", str(origin))
git("init", "-b", "main", str(work))
(work / "f.txt").write_text("v1")
git("-C", str(work), "add", "-A"); git("-C", str(work), "-c", "user.email=a@b", "-c", "user.name=t", "commit", "-m", "v1")
git("-C", str(work), "remote", "add", "origin", str(origin)); git("-C", str(work), "push", "-u", "origin", "main")
old_sha = git("-C", str(work), "rev-parse", "HEAD").stdout.strip()

DIR = str(tmp / "clone")
env = {"os": os, "subprocess": subprocess, "REPO": str(origin),
       "DIR": DIR, "BRANCH": "main", "NB_BRANCH": "main"}

print("── ① 클론이 없을 때")
exec(block, env)
print("   HEAD:", git("-C", DIR, "rev-parse", "HEAD").stdout.strip()[:8], "(기대", old_sha[:8] + ")")
assert git("-C", DIR, "rev-parse", "HEAD").stdout.strip() == old_sha

print("── ② 원격에 새 커밋 → 따라오는가")
(work / "f.txt").write_text("v2")
git("-C", str(work), "add", "-A"); git("-C", str(work), "-c", "user.email=a@b", "-c", "user.name=t", "commit", "-m", "v2")
git("-C", str(work), "push")
new_sha = git("-C", str(work), "rev-parse", "HEAD").stdout.strip()
exec(block, env)
got = git("-C", DIR, "rev-parse", "HEAD").stdout.strip()
print("   HEAD:", got[:8], "(기대", new_sha[:8] + ")")
assert got == new_sha, "새 커밋을 안 따라옴"

print("── ③ ★ 히스토리가 갈림 (지워진 커밋에 붙박인 클론) — 실제로 당한 상황")
git("-C", str(work), "reset", "--hard", old_sha)
(work / "f.txt").write_text("v3-rewritten")
git("-C", str(work), "add", "-A"); git("-C", str(work), "-c", "user.email=a@b", "-c", "user.name=t", "commit", "-m", "v3")
git("-C", str(work), "push", "--force")
forced = git("-C", str(work), "rev-parse", "HEAD").stdout.strip()
exec(block, env)
got = git("-C", DIR, "rev-parse", "HEAD").stdout.strip()
print("   HEAD:", got[:8], "(기대", forced[:8] + ")")
assert got == forced, f"강제 푸시를 못 따라옴: {got[:8]}"
assert (Path(DIR) / "f.txt").read_text() == "v3-rewritten"

print("── ④ 클론이 깨졌을 때 (.git 만 있고 내용 없음)")
shutil.rmtree(Path(DIR) / ".git"); (Path(DIR) / ".git").mkdir()
exec(block, env)
got = git("-C", DIR, "rev-parse", "HEAD").stdout.strip()
print("   HEAD:", got[:8])
assert got == forced

print("── ⑤ ★ 브랜치를 못 박았는가 — 캐글은 리포 없이 시작합니다")
_bad = []
for f in sorted((ROOT / "notebooks").glob("*.ipynb")):
    c1 = "".join(json.loads(f.read_text(encoding="utf-8"))["cells"][1]["source"])
    if 'BRANCH = BRANCH or "main"' in c1:
        _bad.append(f.name + ": 기본값이 main")
    elif "NB_BRANCH" not in c1:
        _bad.append(f.name + ": NB_BRANCH 없음")
if _bad:
    print("   " + "\n   ".join(_bad))
    raise AssertionError(
        "노트북이 main 을 받게 돼 있습니다. 캐글/콜랩은 리포가 없는 상태로\n"
        "시작해서 _ROOT 탐색이 실패하고 기본값을 그대로 씁니다. main 이\n"
        "뒤처져 있으면 셀은 최신인데 src/ 만 옛것인 채로 돕니다.")
print(f"   {len(list((ROOT / 'notebooks').glob('*.ipynb')))}개 전부 NB_BRANCH 로 고정됨")

shutil.rmtree(tmp)
print("\n✅ 네 경우 다 통과 — 없을 때 / 새 커밋 / 강제푸시 / 깨진 클론")
print("all checks passed")
