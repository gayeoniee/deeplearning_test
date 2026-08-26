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


# ── 원본(raw) 정리 ────────────────────────────────────────────────────
# 여기서 실수하면 21GB 재다운로드입니다. zip 과 `.downloaded_keys` 는
# **어떤 경로로도** 사라지면 안 됩니다.
with tempfile.TemporaryDirectory() as td:
    raw = Path(td) / "raw"
    (raw / "TL01" / "라벨링데이터").mkdir(parents=True)      # 압축 해제본
    (raw / "TL01" / "라벨링데이터" / "a.json").write_text("{}")
    (raw / "TL01" / "라벨링데이터" / "b.json").write_text("{}")
    (raw / "섞임").mkdir()                                    # 안에 zip 이 있는 폴더
    (raw / "섞임" / "keep.zip").write_bytes(b"z" * 100)
    (raw / "섞임" / "junk.json").write_text("{}")
    (raw / "VL01.zip").write_bytes(b"z" * 4096)               # 원본
    (raw / ".downloaded_keys").write_text("561/1234")         # 다운로드 기록
    (raw / "scan_report.txt").write_text("x" * 50)            # 지워도 되는 파일

    env.data_root = lambda: raw                               # noqa: E731

    def alive() -> set[str]:
        return {p.name for p in raw.iterdir()}

    print("\n[5] raw — 무엇을 지키고 무엇을 지우나")
    v, g = crops.raw_victims(raw)
    vn = sorted(x["path"].name for x in v)
    check("zip 은 지울 목록에 없다", "VL01.zip" not in vn, str(vn))
    check(".downloaded_keys 는 지울 목록에 없다",
          ".downloaded_keys" not in vn, str(vn))
    check("압축 해제본 폴더는 지울 목록에 있다", "TL01" in vn, str(vn))
    check("zip 아닌 파일도 지울 목록에 있다", "scan_report.txt" in vn, str(vn))
    check("zip 을 품은 폴더는 지키는 쪽으로",
          [x["path"].name for x in g] == ["섞임"], str(g))

    print("\n[6] 확인 입력이 틀리면 아무것도 안 지운다")
    before = alive()
    for typed, why in [("yes", "yes"), ("", "빈 입력"), ("TL01", "일부만"),
                       ("scan_report.txt,TL01", "순서 다름"),
                       ("TL01,scan_report.txt,VL01.zip", "zip 을 끼워넣음")]:
        sys.stdin = io.StringIO(typed + "\n")
        try:
            crops.prune_raw(crops.survey())
        finally:
            sys.stdin = real_in
        check(f"'{why}' 입력 → 그대로", alive() == before, str(alive()))

    print("\n[7] 정확히 입력해야 지운다")
    names = ",".join(x["path"].name for x in crops.raw_victims(raw)[0])
    sys.stdin = io.StringIO(names + "\n")
    try:
        crops.prune_raw(crops.survey())
    finally:
        sys.stdin = real_in
    check("zip 은 남아 있다", (raw / "VL01.zip").is_file())
    check("다운로드 기록은 남아 있다", (raw / ".downloaded_keys").is_file())
    check("zip 을 품은 폴더는 남아 있다", (raw / "섞임" / "keep.zip").is_file())
    check("압축 해제본은 지워졌다", not (raw / "TL01").exists())
    check("zip 아닌 파일도 지워졌다", not (raw / "scan_report.txt").exists())

    print("\n[8] 두 번 돌려도 안전하다")
    crops.prune_raw(crops.survey())      # 확인 입력 없이도 지울 게 없어야 합니다
    check("zip 과 기록이 그대로", alive() == {"VL01.zip", ".downloaded_keys", "섞임"},
          str(alive()))

print("\n" + "=" * 60)
print(f" 통과 {ok} / {ok + fail}")
sys.exit(1 if fail else 0)
