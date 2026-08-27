"""조각 순서가 뒤섞인 zip 을 고치는 도구 검증.

    uv run python tests/test_fix_split_zip.py

실제로 당한 것 (2026-08-27): TL02.zip 80GB 를 맥북에서 받았는데 파이썬이
`BadZipFile` 로 죽었습니다. 잘린 게 아니라 `aihubshell` 이 조각 81개를
**문자열 순서**로 붙였습니다 — `sort -V` 가 맥에서 안 먹은 것입니다:

    part1, part10, part11 … part19, part2, part20 … part8, part80, part81, part9

⚠️ 여기서 제일 중요한 검사는 **틀린 가설을 거절하는가** 입니다.
   처음엔 "마지막 두 조각만 뒤바뀌었다" 로 봤고, 목차 위치 검산(①)은
   그 가설도 통과시켰습니다. 항목을 실제로 찍어보는 검산(②)만이 1/24 로
   걸러냈습니다. ① 만 믿었으면 80GB 를 잘못 덮어썼습니다.
"""

from __future__ import annotations

import hashlib
import io
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import fix_split_zip as fx                              # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {extra}" if extra else ""))
    if not cond:
        FAILS.append(f"{name} {extra}".strip())


def _capture(fn, *a, **kw):
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            out = fn(*a, **kw)
        except SystemExit as exc:
            out = exc
    return out, buf.getvalue()


# ──────────────────────────────────────────────────────────────
def make_zip(path: Path, n_files: int, size_each: int) -> dict[str, bytes]:
    """압축 없이(STORED) 담습니다 — 실제 AI Hub zip 과 같은 방식입니다."""
    want = {}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for i in range(n_files):
            name = f"152.데이터/01.데이터/1.Training/반려견/피부/img_{i:05d}.jpg"
            body = (f"{i}-".encode() * size_each)[:size_each]
            want[name] = body
            z.writestr(name, body)
    return want


def scramble_lex(path: Path, part: int, start: int = 1) -> int:
    """조각으로 자른 뒤 **문자열 순서**로 다시 붙입니다 (맥에서 난 그 모양)."""
    data = path.read_bytes()
    parts = [data[i:i + part] for i in range(0, len(data), part)]
    nums = [start + i for i in range(len(parts))]
    order = sorted(range(len(parts)), key=lambda i: str(nums[i]))
    path.write_bytes(b"".join(parts[i] for i in order))
    return len(parts)


def scramble_tail(path: Path, part: int) -> None:
    """마지막 온전한 조각 하나만 맨 뒤로 보냅니다 (제가 처음 세웠던 가설)."""
    data = path.read_bytes()
    size = len(data)
    last = size % part
    mid = size - last - part
    path.write_bytes(data[:mid] + data[mid + part:] + data[mid:mid + part])


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


PART = 131072            # 검사용 조각 크기 (실제 값은 1 GiB 였습니다)


def _diag(p: Path):
    with open(p, "rb") as f:
        size = p.stat().st_size
        e = fx.find_eocd(f, size)
        return e, fx.solve(f, size, e, verbose=False, part_sizes=[PART])


# ──────────────────────────────────────────────────────────────
def test_repairs_lexicographic_scramble():
    print("\n[문자열 순서] 실제로 난 그 모양을 고치는가")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "TL02.zip"
        want = make_zip(p, 900, 4096)
        good_sha, size = sha(p), p.stat().st_size

        n = scramble_lex(p, PART)
        check(f"조각 {n}개로 섞었다", n > 20, f"n={n}")
        check("크기는 그대로", p.stat().st_size == size)
        check("내용은 달라졌다", sha(p) != good_sha)

        broke = True
        try:
            zipfile.ZipFile(p).namelist()
            broke = False
        except zipfile.BadZipFile:
            pass
        check("파이썬이 못 연다", broke)

        _, lay = _diag(p)
        check("순서를 찾아낸다", lay is not None)
        if lay:
            check("문자열 순서라고 말한다", "문자열 순서" in lay.name, lay.name)
            check("조각 수를 맞힌다", len(lay.segs) == n, f"{len(lay.segs)} vs {n}")

        _capture(fx.main, [str(p), "--apply", "--part-size", str(PART)])
        check("고친 뒤 원본과 **바이트가 같다**", sha(p) == good_sha)
        with zipfile.ZipFile(p) as z:
            got = {nm: z.read(nm) for nm in z.namelist()}
        check("내용이 전부 같다", got == want, f"{len(got)}/{len(want)}개")
        check("망가진 원본은 지웠다",
              not p.with_suffix(p.suffix + ".broken").exists())


def test_rejects_the_wrong_hypothesis():
    print("\n[거절] 그럴듯한 틀린 가설을 거르는가")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "TL02.zip"
        make_zip(p, 900, 4096)
        scramble_lex(p, PART)
        size = p.stat().st_size

        with open(p, "rb") as f:
            e = fx.find_eocd(f, size)
            n = -(-size // PART)
            sizes = [PART] * (n - 1) + [size - PART * (n - 1)]
            # 제가 실제로 세웠던 틀린 가설 — "뒤 1조각만 밀렸다"
            wrong = fx._from_order("뒤 1조각", sizes,
                                   list(range(n - 2)) + [n - 1] + [n - 2])

            # ⚠️ 실제 80GB 파일에서 **이 계산은 틀린 가설을 통과시켰습니다.**
            #    "목차는 EOCD 바로 앞에 통째로 붙어 있다" 고 가정했기 때문입니다.
            #    그래서 지금은 이 계산을 안 씁니다 — 대신 가설대로 **읽어봅니다**.
            naive = wrong.to_phys(e["cd_offset"]) == e["rec_at"] - e["cd_size"]
            print(f"     (자리 계산만 하는 옛 검사는 {'통과' if naive else '거절'}시켰습니다)")

            passed_head = fx.cd_head_matches(f, wrong, e)
            entries = fx.cd_parses(f, wrong, e) if passed_head else None
            ok, total = (0, 0)
            if entries:
                ok, total, _ = fx.verify(f, wrong, e, entries)
            where = ("① 목차 시작" if not passed_head else
                     "① 목차 끝까지 읽기" if entries is None else f"② 항목 {ok}/{total}")
            check("틀린 가설을 거절한다", not (entries and ok == total and total), where)
            print(f"     걸린 곳: {where}")

        # 그리고 도구 전체를 돌리면 **맞는** 가설을 찾아냅니다
        _, lay = _diag(p)
        check("맞는 가설은 찾아낸다", lay is not None and "문자열 순서" in lay.name,
              lay.name if lay else "못 찾음")


def test_matches_the_real_TL02_numbers():
    """실제 TL02.zip 에서 관측한 숫자를 그대로 재현하는가 (회귀 검사).

    2026-08-27 맥북에서 나온 값입니다. 이 숫자들이 다시 맞는지 봐두면,
    나중에 가설 생성 코드를 건드렸을 때 조용히 어긋나는 걸 잡습니다.
    """
    print("\n[실측] 진짜 TL02.zip 의 숫자를 재현하는가")
    SIZE = 86_016_937_297
    PS = 1 << 30                      # 1 GiB
    CD_OFFSET = 85_964_535_312        # ZIP64 EOCD 가 적어둔 목차 위치
    CD_PHYS = 84_890_793_488          # 목차가 실제로 있던 물리 위치

    n = -(-SIZE // PS)
    sizes = [PS] * (n - 1) + [SIZE - PS * (n - 1)]
    check("조각 81개 (1 GiB × 80 + 117MB)", n == 81 and sizes[-1] == 117_591_377,
          f"n={n} last={sizes[-1]:,}")

    nums = [1 + i for i in range(n)]
    lex = sorted(range(n), key=lambda i: str(nums[i]))
    lay = fx._from_order("문자열 순서", sizes, lex)
    check("문자열 순서가 관측된 목차 위치를 맞힌다",
          lay.to_phys(CD_OFFSET) == CD_PHYS,
          f"{lay.to_phys(CD_OFFSET):,} vs {CD_PHYS:,}")

    # 제가 처음 세웠던 가설도 **같은 값**을 냅니다 — 그래서 속았습니다
    wrong = fx._from_order("뒤 1조각", sizes,
                           list(range(n - 2)) + [n - 1] + [n - 2])
    check("틀린 가설도 같은 값을 낸다 (그래서 자리 계산만으론 못 가린다)",
          wrong.to_phys(CD_OFFSET) == CD_PHYS)

    # 두 가설은 앞쪽 항목에서 갈립니다 — 실제로 ② 가 여기서 걸렀습니다
    lfh = 2_909_712_736               # 목차에 적혀 있던 어느 항목
    check("두 가설이 앞쪽 항목에서 갈린다",
          lay.to_phys(lfh) != wrong.to_phys(lfh),
          f"{lay.to_phys(lfh):,} vs {wrong.to_phys(lfh):,}")


def test_repairs_tail_shift():
    print("\n[뒤로 밀림] 조각 하나만 뒤로 간 경우도 고치는가")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "a.zip"
        make_zip(p, 900, 4096)
        good = sha(p)
        scramble_tail(p, PART)
        check("망가졌다", sha(p) != good)
        _capture(fx.main, [str(p), "--apply", "--part-size", str(PART)])
        check("원본과 바이트가 같다", sha(p) == good)


def test_leaves_a_healthy_zip_alone():
    print("\n[안전] 멀쩡한 zip 은 안 건드리는가")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "ok.zip"
        make_zip(p, 50, 2048)
        before = sha(p)
        _, log = _capture(fx.main, [str(p), "--apply", "--part-size", str(PART)])
        check("고칠 게 없다고 말한다", "고칠 게 없습니다" in log)
        check("파일을 안 건드렸다", sha(p) == before)


def test_preview_writes_nothing():
    print("\n[미리보기] --apply 없이는 안 쓰는가")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "TL02.zip"
        make_zip(p, 900, 4096)
        scramble_lex(p, PART)
        before = sha(p)
        _, log = _capture(fx.main, [str(p), "--part-size", str(PART)])
        check("미리보기라고 말한다", "미리보기" in log)
        check("디스크가 얼마나 필요한지 말한다", "디스크 여유가 필요" in log)
        check("파일이 그대로다", sha(p) == before)
        check("새 파일을 안 만들었다",
              not p.with_suffix(p.suffix + ".fixed").exists())


def test_refuses_unexplainable():
    print("\n[거절] 설명 못 하는 파일은 안 고치는가")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "junk.zip"
        make_zip(p, 50, 2048)
        with open(p, "ab") as f:
            f.write(b"\x00" * 5000)                 # 조각 순서 문제가 아님
        before = sha(p)
        _, log = _capture(fx.main, [str(p), "--apply", "--part-size", str(PART)])
        check("못 찾았다고 말한다", "못 찾았습니다" in log, log.strip()[-70:])
        check("안 건드린다", sha(p) == before)


if __name__ == "__main__":
    print("조각 순서 뒤섞인 zip 고치기 검증")
    for fn in (test_repairs_lexicographic_scramble,
               test_rejects_the_wrong_hypothesis,
               test_matches_the_real_TL02_numbers,
               test_repairs_tail_shift,
               test_leaves_a_healthy_zip_alone,
               test_preview_writes_nothing,
               test_refuses_unexplainable):
        fn()
    print()
    if FAILS:
        print(f"{len(FAILS)} check(s) failed:")
        for f in FAILS:
            print(f"  - {f}")
        sys.exit(1)
    print("all checks passed")
