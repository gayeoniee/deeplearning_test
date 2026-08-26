"""tools/crops.py — 쓰는 크롭을 지우지 않는가.

지우면 되돌리는 데 재다운로드(21GB)가 필요합니다. 그래서 안전장치를
테스트로 못 박습니다.

    uv run python tests/test_crops_tool.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from src import env                                              # noqa: E402
from src.agent import STAGE1_TAG, STAGE2_TAG                     # noqa: E402

ok = fail = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {name}")
    else:
        fail += 1
        print(f"  FAIL  {name}  {detail}")


def make(root: Path, spec: dict[str, int]) -> None:
    for tag, n in spec.items():
        d = root / "crops" / tag
        d.mkdir(parents=True)
        for i in range(n):
            (d / f"{i:03d}.jpg").write_bytes(b"x" * 1024)


with tempfile.TemporaryDirectory() as td:
    work = Path(td)
    make(work, {STAGE1_TAG: 5, STAGE2_TAG: 5, "full": 5, "m1.5": 5})
    env.work_root = lambda: work                                 # noqa: E731
    import crops                                                 # noqa: E402

    s = crops.survey()

    print("\n[1] 쓰는 태그를 안 지운다")
    check("파이프라인 태그를 코드에서 읽음",
          crops.STAGE1_TAG == STAGE1_TAG and crops.STAGE2_TAG == STAGE2_TAG)
    check(f"{STAGE1_TAG} 는 '쓰는 것'", s["tags"][STAGE1_TAG]["used"])
    check(f"{STAGE2_TAG} 는 '쓰는 것'", s["tags"][STAGE2_TAG]["used"])
    check("full 은 '안 쓰는 것'", not s["tags"]["full"]["used"])
    check("m1.5 는 '안 쓰는 것'", not s["tags"]["m1.5"]["used"])

    print("\n[2] 확인 입력이 틀리면 아무것도 안 지운다")
    import io                                                    # noqa: E402

    real_in = sys.stdin
    for typed, why in [("yes", "yes"), ("", "빈 입력"), ("full", "일부만"),
                       ("m1.5,full", "순서 다름")]:
        sys.stdin = io.StringIO(typed + "\n")
        try:
            crops.prune(crops.survey())
        finally:
            sys.stdin = real_in
        left = sorted(p.name for p in (work / "crops").iterdir())
        check(f"'{why}' 입력 → 그대로 4개", len(left) == 4, str(left))

    print("\n[3] 정확히 입력해야 지운다")
    sys.stdin = io.StringIO("full,m1.5\n")
    try:
        crops.prune(crops.survey())
    finally:
        sys.stdin = real_in
    left = sorted(p.name for p in (work / "crops").iterdir())
    check("안 쓰는 것만 지워짐", left == sorted([STAGE1_TAG, STAGE2_TAG]), str(left))
    check("쓰는 것의 파일은 남아 있음",
          len(list((work / "crops" / STAGE1_TAG).iterdir())) == 5)

    print("\n[4] 지울 게 없으면 조용히 끝난다")
    crops.prune(crops.survey())          # 확인 입력 없이도 안전해야 합니다
    check("쓰는 것만 남은 상태에서 아무 일 없음",
          sorted(p.name for p in (work / "crops").iterdir())
          == sorted([STAGE1_TAG, STAGE2_TAG]))

print("\n" + "=" * 60)
print(f" 통과 {ok} / {ok + fail}")
sys.exit(1 if fail else 0)
