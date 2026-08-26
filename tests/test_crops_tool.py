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
#
# ⚠️ 실제 배치는 zip 이 **하위 폴더 안**에 있습니다 (prepare_local.py:115 —
#    raw = env.data_root() / name). 최상위만 보면 아무것도 못 지웁니다.
with tempfile.TemporaryDirectory() as td:
    raw = Path(td) / "raw"
    real = raw / "VL01" / "152.반려동물_피부질환_데이터" / "01.데이터" / "2.Validation"
    (real / "2_라벨링데이터_240422_add" / "반려견").mkdir(parents=True)
    for i in range(3):
        (real / "2_라벨링데이터_240422_add" / "반려견" / f"{i}.json").write_text("{}")
    (raw / "VL01" / "VL01.zip").write_bytes(b"z" * 4096)          # ← 폴더 **안**
    (raw / "VL01" / ".downloaded_keys").write_text("517022")      # ← 폴더 **안**
    (raw / "scan_report.txt").write_text("x" * 50)                # 최상위 잡파일

    env.data_root = lambda: raw                                   # noqa: E731

    def alive() -> set[str]:
        return {str(p.relative_to(raw)) for p in raw.rglob("*")}

    print("\n[5] raw — zip 이 하위 폴더에 있어도 안쪽을 지운다")
    units, keep = crops.raw_units(raw)
    un = sorted(str(u["path"].relative_to(raw)) for u in units)
    kn = sorted(str(k["path"].relative_to(raw)) for k in keep)
    check("zip 은 지킬 목록", "VL01/VL01.zip".replace("/", os.sep) in kn, str(kn))
    check(".downloaded_keys 는 지킬 목록",
          "VL01/.downloaded_keys".replace("/", os.sep) in kn, str(kn))
    check("압축 해제본은 **가장 얕은** 폴더 하나로 잡힌다",
          un == [str(Path("VL01") / "152.반려동물_피부질환_데이터"),
                 "scan_report.txt"], str(un))
    check("그 폴더 안 3개를 다 센다",
          any(u["n"] == 3 for u in units), str([u["n"] for u in units]))
    check("VL01 자체는 지울 목록에 없다",
          "VL01" not in un, str(un))

    print("\n[6] 확인 입력이 틀리면 아무것도 안 지운다")
    before = alive()
    for typed, why in [("yes", "yes"), ("", "빈 입력"), ("VL01", "일부만"),
                       ("scan_report.txt,VL01", "순서 다름"),
                       ("VL01,scan_report.txt,VL01.zip", "zip 을 끼워넣음")]:
        sys.stdin = io.StringIO(typed + "\n")
        try:
            crops.prune_raw(crops.survey())
        finally:
            sys.stdin = real_in
        check(f"'{why}' 입력 → 그대로", alive() == before, str(alive() ^ before))

    print("\n[7] 정확히 입력해야 지운다")
    sys.stdin = io.StringIO("VL01,scan_report.txt\n")
    try:
        crops.prune_raw(crops.survey())
    finally:
        sys.stdin = real_in
    check("zip 은 남아 있다", (raw / "VL01" / "VL01.zip").is_file())
    check("다운로드 기록은 남아 있다", (raw / "VL01" / ".downloaded_keys").is_file())
    check("압축 해제본은 지워졌다",
          not (raw / "VL01" / "152.반려동물_피부질환_데이터").exists())
    check("최상위 잡파일도 지워졌다", not (raw / "scan_report.txt").exists())
    check("has_usable_data 가 아직 True (zip 을 셈) → 재다운로드 안 함",
          __import__("src.aihub", fromlist=["x"]).has_usable_data(raw))

    print("\n[8] 두 번 돌려도 안전하다")
    crops.prune_raw(crops.survey())      # 확인 입력 없이도 지울 게 없어야 합니다
    check("zip 과 기록만 남았다",
          alive() == {"VL01", str(Path("VL01") / "VL01.zip"),
                      str(Path("VL01") / ".downloaded_keys")}, str(alive()))

print("\n" + "=" * 60)
print(f" 통과 {ok} / {ok + fail}")
sys.exit(1 if fail else 0)
