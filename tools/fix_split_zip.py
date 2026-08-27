"""조각을 잘못된 순서로 붙인 zip 을 **다시 받지 않고** 고칩니다.

    uv run python tools/fix_split_zip.py <zip경로>            진단만 (기본)
    uv run python tools/fix_split_zip.py <zip경로> --apply    고칩니다

무슨 일이 있었나
----------------
aihubshell 은 큰 파일을 조각으로 받아 이렇게 합칩니다 (232줄짜리 스크립트의 139줄):

    find … -print0 | sort -zt'.' -k2V | xargs -0 cat > "${prefix}"

`-V` 는 "part2 < part10" 으로 읽는 **버전 정렬**입니다. 맥의 BSD `sort` 에서
이게 안 먹으면 **문자열 순서**로 떨어집니다:

    part1, part10, part11 … part19, part2, part20 … part8, part80, part81, part9

크기는 정확히 맞고 파일 앞머리도 멀쩡해서 `file` 은 "Zip archive data" 라고
합니다.

증상이 **두 가지**입니다 — 목차 든 조각이 어디로 가느냐에 달렸습니다:

  · 밀린 경우  → `BadZipFile: File is not a zip file` 로 아예 안 열립니다
                 (TL02, 조각 81개 = 번호 0~80. 문자열 순서의 마지막은 "9")
  · 남은 경우  → **열립니다.** 목록도 다 나옵니다. 그런데 읽으면 대부분
                 BadZipFile 로 깨집니다 (TL01, 조각 91개 = 번호 0~90.
                 "90" 이 문자열 순서에서도 마지막이라 제자리에 남습니다)

⚠️ 두 번째가 훨씬 위험합니다. **아무도 안 죽습니다.** TL01 에서는 읽히는
   2.4% 만 가지고 크롭까지 다 돌고 "✅ 완료" 가 찍혔습니다.
   그래서 이 도구는 "목차가 밀렸나" 가 아니라 **항목이 실제로 읽히나**로
   고장을 판정합니다.

⚠️ 이건 손상이 아닙니다. **바이트는 다 있고 순서만 틀렸습니다.**

무엇을 근거로 고치나
--------------------
추측한 순서를 그냥 믿지 않습니다. 순서 가설마다 **두 번 검산**합니다.

  1. zip 이 스스로 적어둔 목차 위치(`cd_offset`)를 그 가설로 옮긴 값이
     **실제 목차가 있는 물리 위치**와 같은가
  2. 목차를 읽어 실제 항목 수십 개의 로컬 헤더가 그 자리에 있고
     **파일 이름까지 일치**하는가

둘 다 통과한 가설만 씁니다. 하나라도 어긋나면 아무것도 안 씁니다.
(처음에 "마지막 두 조각만 뒤바뀌었다" 는 가설을 세웠다가 ② 에서 1/24 로
걸렸습니다. ① 만 봤으면 80GB 를 잘못 덮어썼을 겁니다.)
"""

from __future__ import annotations

import argparse
import bisect
import shutil
import struct
from pathlib import Path

EOCD_SIG = b"PK\x05\x06"
Z64_EOCD_SIG = b"PK\x06\x06"
Z64_LOC_SIG = b"PK\x06\x07"
CD_SIG = b"PK\x01\x02"
LFH_SIG = b"PK\x03\x04"

SCAN_BACK = 16 << 30         # 뒤에서 이만큼까지 목차를 찾습니다
WINDOW = 1 << 26             # 64MB
COPY = 1 << 24               # 16MB


def human(n: int) -> str:
    return f"{n:,} bytes ({n / 1024**3:.2f} GiB)"


# ──────────────────────────────────────────────────────────────
# 1. 목차 찾기
# ──────────────────────────────────────────────────────────────
def find_eocd(f, size: int) -> dict:
    """진짜 EOCD 를 찾습니다.

    `PK\\x05\\x06` 네 바이트는 압축 안 된 사진 데이터 안에서도 우연히 나옵니다
    (실제 파일에서 72번 나왔습니다). 그래서 **바로 앞 20바이트가 ZIP64 locator,
    그 앞 56바이트가 ZIP64 EOCD** 인지까지 봅니다.
    """
    pos, tail = size, b""
    while pos > 0 and size - pos < SCAN_BACK:
        start = max(0, pos - WINDOW)
        f.seek(start)
        buf = f.read(pos - start) + tail
        i = buf.rfind(EOCD_SIG)
        while i >= 0:
            got = _try_eocd(f, start + i, size)
            if got:
                return got
            i = buf.rfind(EOCD_SIG, 0, i)
        tail, pos = buf[:3], start
    raise SystemExit("❌ 목차(EOCD)를 못 찾았습니다. 순서 문제가 아니라 "
                     "정말로 잘렸을 수 있습니다.")


def _try_eocd(f, at: int, size: int) -> dict | None:
    f.seek(at)
    rec = f.read(22)
    if len(rec) < 22:
        return None
    comment_len = struct.unpack("<H", rec[20:22])[0]
    zip_end = at + 22 + comment_len
    if zip_end > size:
        return None

    if at >= 76:                       # 4GB 넘는 zip — ZIP64 기록이 앞에 붙습니다
        f.seek(at - 20)
        if f.read(4) == Z64_LOC_SIG:
            f.seek(at - 76)
            z64 = f.read(56)
            if z64[:4] == Z64_EOCD_SIG:
                return {"kind": "zip64", "eocd_at": at, "zip_end": zip_end,
                        "rec_at": at - 76,
                        "entries": struct.unpack("<Q", z64[32:40])[0],
                        "cd_size": struct.unpack("<Q", z64[40:48])[0],
                        "cd_offset": struct.unpack("<Q", z64[48:56])[0]}

    entries, cd_size, cd_offset = struct.unpack("<HII", rec[10:20])
    if 0xFFFFFFFF in (cd_size, cd_offset) or entries == 0xFFFF:
        return None
    if cd_size == 0 or cd_size > at:
        return None
    return {"kind": "plain", "eocd_at": at, "zip_end": zip_end, "rec_at": at,
            "entries": entries, "cd_size": cd_size, "cd_offset": cd_offset}


# ──────────────────────────────────────────────────────────────
# 2. 순서 가설
# ──────────────────────────────────────────────────────────────
class Layout:
    """조각들이 어떤 순서로 붙어 있는가 — 올바른 위치 → 지금 위치.

    `segs` 는 **올바른 파일 기준 오름차순**인 (올바른시작, 물리시작, 길이).
    """

    def __init__(self, name: str, segs: list[tuple[int, int, int]]):
        self.name = name
        self.segs = segs
        self.starts = [s[0] for s in segs]
        self.size = segs[-1][0] + segs[-1][2]

    def _seg(self, off: int) -> tuple[int, int, int]:
        i = bisect.bisect_right(self.starts, off) - 1
        if i < 0:
            raise ValueError(f"구간 밖: {off}")
        c, p, n = self.segs[i]
        if not (c <= off < c + n):
            raise ValueError(f"구간 밖: {off}")
        return c, p, n

    def to_phys(self, off: int) -> int:
        c, p, _ = self._seg(off)
        return p + (off - c)

    def read(self, f, off: int, n: int) -> bytes:
        """올바른 파일 기준 [off, off+n) 을 지금 파일에서 이어붙여 읽습니다.

        조각 경계를 넘으면 물리 위치가 튀므로 **경계에서 잘라** 여러 번 읽습니다.
        """
        out = bytearray()
        while n > 0 and off < self.size:
            try:
                c, p, ln = self._seg(off)
            except ValueError:
                break
            step = min(n, c + ln - off)
            f.seek(p + (off - c))
            blk = f.read(step)
            if not blk:
                break
            out += blk
            off += len(blk)
            n -= len(blk)
        return bytes(out)

    def is_identity(self) -> bool:
        return all(c == p for c, p, _ in self.segs)

    def moved(self) -> int:
        return sum(1 for c, p, _ in self.segs if c != p)


def _from_order(name: str, sizes: list[int], order: list[int]) -> Layout:
    """`sizes` 는 올바른 순서의 조각 크기, `order` 는 물리적으로 붙은 순서."""
    corr, o = [], 0
    for n in sizes:
        corr.append(o)
        o += n
    phys = [0] * len(sizes)
    o = 0
    for i in order:
        phys[i] = o
        o += sizes[i]
    return Layout(name, [(corr[i], phys[i], sizes[i]) for i in range(len(sizes))])


def identity(size: int) -> Layout:
    """지금 파일을 **그대로** 보는 배치. 고장 판정의 기준입니다."""
    return Layout("지금 그대로", [(0, 0, size)])


# 다운로드 도구가 실제로 쓰는 조각 크기만 봅니다. 아무 값이나 허용하면
# 검산 ① 을 우연히 통과하는 가설이 생깁니다 (② 가 다시 거르긴 합니다).
PART_SIZES = sorted({1 << k for k in range(26, 34)}          # 64MiB ~ 8GiB
                    | {n * 1000 ** 3 for n in (1, 2, 4, 5, 10)})


def candidates(size: int, part_sizes: list[int] | None = None):
    """그럴듯한 조각 나눔 × 붙은 순서를 훑습니다."""
    for ps in sorted(set(part_sizes or PART_SIZES)):
        if ps >= size:
            continue
        n = -(-size // ps)                                # 올림
        if n < 2 or n > 4096:
            continue
        sizes = [ps] * (n - 1) + [size - ps * (n - 1)]
        for start in (0, 1):
            nums = [start + i for i in range(n)]
            # ★ `sort -V` 가 안 먹었을 때 떨어지는 자리 — 이번에 실제로 이거였습니다
            lex = sorted(range(n), key=lambda i: str(nums[i]))
            yield (f"문자열 순서 · 조각 {human(ps)} · {start}번부터", sizes, lex)
        for k in range(1, min(n, 9)):
            # 뒤쪽 k 조각이 통째로 맨 뒤로 밀린 경우
            order = list(range(n - 1 - k)) + [n - 1] + list(range(n - 1 - k, n - 1))
            yield (f"뒤 {k}조각이 맨 뒤로 · 조각 {human(ps)}", sizes, order)


# ──────────────────────────────────────────────────────────────
# 3. 검산
# ──────────────────────────────────────────────────────────────
def cd_head_matches(f, lay: Layout, e: dict) -> bool:
    """싸구려 1차 거름 — 목차가 시작해야 할 자리에 목차 표시가 있는가.

    ⚠️ "목차가 EOCD 바로 앞에 통째로 붙어 있다" 고 가정하면 안 됩니다.
       목차 자체가 조각 경계를 넘어갈 수 있습니다. 그래서 물리 위치를 빼서
       비교하지 않고, **가설대로 읽어서** 첫 네 바이트를 봅니다.
    """
    try:
        return lay.read(f, e["cd_offset"], 4) == CD_SIG
    except ValueError:
        return False


def cd_parses(f, lay: Layout, e: dict) -> list[tuple[int, bytes]] | None:
    """목차를 가설대로 이어붙여 읽고, 적힌 항목 수만큼 **끝까지** 읽히는가."""
    cd = lay.read(f, e["cd_offset"], e["cd_size"])
    if len(cd) != e["cd_size"]:
        return None
    entries = parse_cd(cd)
    if len(entries) != e["entries"]:
        return None
    return entries


def parse_cd(cd: bytes) -> list[tuple[int, bytes]]:
    out, pos = [], 0
    while pos + 46 <= len(cd) and cd[pos:pos + 4] == CD_SIG:
        n_len, x_len, c_len = struct.unpack("<HHH", cd[pos + 28:pos + 34])
        name = cd[pos + 46:pos + 46 + n_len]
        lfh = struct.unpack("<I", cd[pos + 42:pos + 46])[0]
        if lfh == 0xFFFFFFFF:                        # ZIP64 — 추가 필드에 있습니다
            ex = cd[pos + 46 + n_len:pos + 46 + n_len + x_len]
            j = 0
            while j + 4 <= len(ex):
                tag, ln = struct.unpack("<HH", ex[j:j + 4])
                if tag == 0x0001:
                    body = ex[j + 4:j + 4 + ln]
                    sizes = struct.unpack("<II", cd[pos + 20:pos + 28])
                    skip = 8 * sum(1 for v in sizes if v == 0xFFFFFFFF)
                    if len(body) >= skip + 8:
                        lfh = struct.unpack("<Q", body[skip:skip + 8])[0]
                    break
                j += 4 + ln
        out.append((lfh, name))
        pos += 46 + n_len + x_len + c_len
    return out


def verify(f, lay: Layout, e: dict, entries: list[tuple[int, bytes]],
           sample: int = 40) -> tuple[int, int, list[str]]:
    """목차 항목을 골고루 뽑아 로컬 헤더와 **파일 이름**이 맞는지 봅니다.

    ★ 이 검사가 이 도구의 핵심입니다. 목차 위치만 맞춰보는 1차 거름은
      **틀린 가설도 통과시킵니다** (실제로 그랬습니다). 흩어진 항목을
      실제로 찍어봐야 순서가 진짜 맞는지 알 수 있습니다.
    """
    if not entries:
        return 0, 0, ["목차를 한 항목도 못 읽었습니다"]
    step = max(1, len(entries) // sample)
    picks = entries[::step][:sample]
    ok, bad = 0, []
    for lfh, name in picks:
        head = lay.read(f, lfh, 30)
        if len(head) < 30 or head[:4] != LFH_SIG:
            bad.append(f"@ {lfh:,} — 로컬 헤더가 없습니다")
            continue
        n_len = struct.unpack("<H", head[26:28])[0]
        if lay.read(f, lfh + 30, n_len) != name:
            bad.append(f"@ {lfh:,} — 이름이 다릅니다")
            continue
        ok += 1
    return ok, len(picks), bad


def solve(f, size: int, e: dict, verbose: bool = True,
          part_sizes: list[int] | None = None) -> Layout | None:
    """가설을 하나씩 세워 세 관문을 다 통과하는 것을 찾습니다."""
    tried = 0
    for name, sizes, order in candidates(size, part_sizes):
        lay = _from_order(name, sizes, order)
        if lay.is_identity():
            continue
        tried += 1
        if not cd_head_matches(f, lay, e):
            continue
        entries = cd_parses(f, lay, e)
        if entries is None:
            if verbose:
                print(f"\n  후보: {name}\n    ① 목차 시작은 맞는데 끝까지 안 읽힙니다 — 버립니다")
            continue
        if verbose:
            print(f"\n  후보: {name}")
            print(f"    ① 목차 {len(entries):,}항목을 끝까지 읽었습니다")
            print("    ② 항목을 골고루 찍어봅니다 …")
        ok, total, bad = verify(f, lay, e, entries)
        if verbose:
            print(f"       {ok}/{total}")
            for b in bad[:3]:
                print(f"       ❌ {b}")
        if ok == total and total > 0:
            return lay
    if verbose:
        print(f"\n  가설 {tried}개를 봤지만 통과한 게 없습니다.")
    return None


# ──────────────────────────────────────────────────────────────
# 4. 다시 쓰기
# ──────────────────────────────────────────────────────────────
def rebuild(src: Path, lay: Layout, out: Path, progress: bool = True) -> None:
    done = 0
    with open(src, "rb") as f, open(out, "wb") as g:
        for _corr, phys, n in lay.segs:
            f.seek(phys)
            left = n
            while left:
                blk = f.read(min(left, COPY))
                if not blk:
                    raise SystemExit("❌ 원본이 짧게 끝났습니다.")
                g.write(blk)
                left -= len(blk)
                done += len(blk)
                if progress:
                    print(f"\r    {done / lay.size:6.1%}  "
                          f"({done / 1024**3:.1f} GiB)", end="", flush=True)
        g.flush()
        import os
        os.fsync(g.fileno())
    if progress:
        print()


# ──────────────────────────────────────────────────────────────
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("zip", help="고칠 zip 경로")
    ap.add_argument("--apply", action="store_true", help="실제로 고칩니다")
    ap.add_argument("--part-size", type=int, default=None,
                    help="조각 크기를 직접 지정 (기본은 흔한 값들을 훑습니다)")
    ap.add_argument("--keep-broken", action="store_true",
                    help="고친 뒤에도 원본을 .broken 으로 남깁니다 (용량 두 배)")
    a = ap.parse_args(argv)

    p = Path(a.zip).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"❌ 없는 파일: {p}")
    size = p.stat().st_size

    print("=" * 68)
    print(f" {p.name}   {human(size)}")
    print("=" * 68)

    with open(p, "rb") as f:
        print("\n[목차 찾는 중] 뒤에서부터 훑습니다 …")
        e = find_eocd(f, size)
        cd_phys = e["rec_at"] - e["cd_size"]
        print(f"  {e['kind']:<7} 기록          {e['rec_at']:,}")
        print(f"  항목 수              {e['entries']:,}개")
        print(f"  목차 크기            {human(e['cd_size'])}")
        print(f"  목차가 있어야 할 곳  {e['cd_offset']:,}")
        print(f"  목차가 실제 있는 곳  {cd_phys:,}")

        # ⚠️ "목차가 밀렸나" 로만 고장을 판정하면 안 됩니다.
        #    TL01 이 그 함정이었습니다 — 조각이 섞였는데 **목차 든 조각이 우연히
        #    제자리에 남아서** zip 이 멀쩡히 열렸습니다. 목록도 다 나오고요.
        #    그런데 실제로 읽으면 97.6% 가 BadZipFile 로 깨졌습니다.
        #    그래서 **지금 파일 그대로 항목을 읽어보는 것**으로 판정합니다.
        print("\n[고장 판정] 지금 파일 그대로 항목이 읽히는지 봅니다 …")
        ident = identity(size)
        ent = cd_parses(f, ident, e)
        if ent:
            ok, total, _ = verify(f, ident, e, ent)
            print(f"  항목 {ok}/{total}")
            if ok == total:
                print("\n✅ 멀쩡한 zip 입니다. 고칠 게 없습니다.")
                return
        else:
            print("  목차를 끝까지 못 읽었습니다")

        print("\n[순서 맞추기] 가설을 세우고 두 번씩 검산합니다 …")
        lay = solve(f, size, e,
                    part_sizes=[a.part_size] if a.part_size else None)
        if lay is None:
            print("\n❌ 설명할 수 있는 순서를 못 찾았습니다. **아무것도 안 씁니다.**")
            print("   이 출력을 그대로 공유해주세요.")
            return
        print(f"\n  ✅ 찾았습니다 — {lay.name}")
        print(f"     조각 {len(lay.segs)}개 중 자리를 옮겨야 하는 것 {lay.moved()}개")

    if not a.apply:
        print("\n미리보기입니다. 고치려면 --apply 를 붙이세요.")
        print(f"  {human(size)} 를 올바른 순서로 **새로 씁니다** — "
              "그만큼 디스크 여유가 필요합니다.")
        print("  다 쓰고 열리는 것까지 확인한 뒤에 망가진 원본을 지웁니다.")
        return

    free = shutil.disk_usage(p.parent).free
    if free < size + (1 << 30):
        raise SystemExit(f"❌ 디스크가 모자랍니다 — 여유 {human(free)} / 필요 {human(size)}")

    out = p.with_suffix(p.suffix + ".fixed")
    print(f"\n[다시 쓰는 중] → {out.name}")
    rebuild(p, lay, out)

    print("\n[확인] 파이썬으로 열어봅니다 …")
    import zipfile

    with zipfile.ZipFile(out) as z:
        names = z.namelist()
    if len(names) != e["entries"]:
        raise SystemExit(f"❌ 항목 수가 다릅니다 — {len(names):,} vs {e['entries']:,}.\n"
                         f"   고친 파일은 {out.name} 로 남겨둡니다. 원본은 그대로입니다.")
    print(f"  ✅ 열립니다 — 항목 {len(names):,}개")
    print(f"     예: {names[0]}")

    broken = p.with_suffix(p.suffix + ".broken")
    p.rename(broken)
    out.rename(p)
    if a.keep_broken:
        print(f"\n원본은 {broken.name} 으로 남겼습니다.")
    else:
        broken.unlink()
        print(f"\n망가진 원본은 지웠습니다 ({human(size)} 확보).")
    print("\n다음: uv run python prepare_local.py --chunk TL02 --margins 2.5,-320")


if __name__ == "__main__":
    main()
