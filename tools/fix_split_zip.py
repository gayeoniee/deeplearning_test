"""조각을 잘못된 순서로 붙인 zip 을 **다시 받지 않고** 고칩니다.

    uv run python tools/fix_split_zip.py <zip경로>            진단만 (기본)
    uv run python tools/fix_split_zip.py <zip경로> --apply    고칩니다
    uv run python tools/fix_split_zip.py <zip경로> --restore  중간에 끊겼을 때 되돌리기

무슨 일이 있었나
----------------
aihubshell 은 큰 파일을 조각으로 받아 이렇게 합칩니다 (232줄짜리 스크립트의 139줄):

    find … -print0 | sort -zt'.' -k2V | xargs -0 cat > "${prefix}"

정렬이 밀리면 조각이 **엉뚱한 순서로** 붙습니다. 크기는 정확히 맞고
파일 앞머리도 멀쩡해서 `file` 은 "Zip archive data" 라고 합니다 —
파이썬만 `BadZipFile: File is not a zip file` 로 죽습니다. zip 의 목차는
파일 **맨 뒤**에 있어야 하는데 중간에 가 있기 때문입니다.

⚠️ 이건 손상이 아닙니다. **바이트는 다 있고 순서만 틀렸습니다.**
   80GB 를 다시 받을 이유가 없습니다.

무엇을 근거로 고치나
--------------------
추측하지 않습니다. zip 자신이 답을 들고 있습니다:

  · ZIP64 EOCD 가 목차의 **원래 위치**(`cd_offset`)를 적어둡니다
  · 그 목차가 지금 **물리적으로** 어디 있는지는 우리가 잽니다
  · 두 값의 차이 = 밀려난 양

그리고 이 값이 "파일 끝 − zip 끝" 과 **일치하는지** 따로 확인합니다.
두 경로로 같은 숫자가 나와야만 고칩니다. 하나라도 안 맞으면 멈춥니다.

고치기 전에 목차를 읽어 **실제 항목 몇 개의 로컬 헤더가 제자리에 있는지**
확인합니다. 이게 통과해야 씁니다.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

EOCD_SIG = b"PK\x05\x06"
Z64_EOCD_SIG = b"PK\x06\x06"
Z64_LOC_SIG = b"PK\x06\x07"
CD_SIG = b"PK\x01\x02"
LFH_SIG = b"PK\x03\x04"

SCAN_BACK = 8 << 30          # 뒤에서 이만큼까지 목차를 찾습니다
WINDOW = 1 << 26             # 64MB


def human(n: int) -> str:
    return f"{n:,} bytes ({n / 1024**3:.2f} GiB)"


# ──────────────────────────────────────────────────────────────
# 1. 목차 찾기
# ──────────────────────────────────────────────────────────────
def find_eocd(f, size: int) -> dict:
    """진짜 EOCD 를 찾습니다.

    `PK\\x05\\x06` 네 바이트는 압축 안 된 사진 데이터 안에서도 우연히 나옵니다
    (이 파일에서 72번 나왔습니다). 그래서 **바로 앞 20바이트가 ZIP64 locator,
    그 앞 56바이트가 ZIP64 EOCD** 인지까지 봅니다. 우연히 셋이 줄맞춰 나오지는
    않습니다.
    """
    pos = size
    tail = b""
    while pos > 0 and size - pos < SCAN_BACK:
        start = max(0, pos - WINDOW)
        f.seek(start)
        buf = f.read(pos - start) + tail
        i = buf.rfind(EOCD_SIG)
        while i >= 0:
            at = start + i
            got = _try_eocd(f, at, size)
            if got:
                return got
            i = buf.rfind(EOCD_SIG, 0, i)
        tail = buf[:3]
        pos = start
    raise SystemExit("❌ 목차(EOCD)를 못 찾았습니다. 순서가 틀린 게 아니라 "
                     "정말로 잘렸을 수 있습니다.")


def _try_eocd(f, at: int, size: int) -> dict | None:
    """이 위치의 `PK\\x05\\x06` 이 진짜 EOCD 인지 확인하고 값을 읽습니다.

    큰 zip(4GB 초과)은 ZIP64 기록을 앞에 달고 있어서 그걸로 확인하면 확실합니다.
    작은 zip 은 그게 없으므로 EOCD 자체의 값이 앞뒤가 맞는지로 봅니다.
    """
    f.seek(at)
    rec = f.read(22)
    if len(rec) < 22:
        return None
    comment_len = struct.unpack("<H", rec[20:22])[0]
    zip_end = at + 22 + comment_len
    if zip_end > size:
        return None

    # ── ZIP64 가 붙어 있는 경우 (우리가 다루는 80GB 짜리는 전부 여기)
    if at >= 76:
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

    # ── 작은 zip — EOCD 값만으로 확인합니다
    entries, cd_size, cd_offset = struct.unpack("<HII", rec[10:20])
    if 0xFFFFFFFF in (cd_size, cd_offset) or entries == 0xFFFF:
        return None                       # ZIP64 를 써야 하는데 못 찾은 경우
    if cd_size == 0 or cd_size > at:
        return None
    return {"kind": "plain", "eocd_at": at, "zip_end": zip_end, "rec_at": at,
            "entries": entries, "cd_size": cd_size, "cd_offset": cd_offset}


# ──────────────────────────────────────────────────────────────
# 2. 얼마나 밀렸나 — 두 경로로 따로 구해서 대조합니다
# ──────────────────────────────────────────────────────────────
def diagnose(f, size: int) -> dict:
    e = find_eocd(f, size)

    # (가) 파일 끝에 남은 양
    extra_by_tail = size - e["zip_end"]
    # (나) 목차가 "있어야 할 곳" 과 "실제 있는 곳" 의 차이
    cd_phys = e["rec_at"] - e["cd_size"]
    extra_by_cd = e["cd_offset"] - cd_phys

    e.update({"extra_by_tail": extra_by_tail, "extra_by_cd": extra_by_cd,
              "cd_phys": cd_phys, "size": size})

    if extra_by_tail == 0 and extra_by_cd == 0:
        e["verdict"] = "ok"
        return e
    if extra_by_tail != extra_by_cd:
        e["verdict"] = "unknown"
        return e

    extra = extra_by_tail
    # 조각 크기 S 를 고릅니다. 밀려난 덩어리가 조각 k 개라고 보고,
    # **앞부분이 조각 경계에서 딱 끝나는** S 만 받아들입니다.
    for k in range(1, 65):
        if extra % k:
            continue
        s = extra // k
        if s <= 0:
            continue
        last = size % s                      # 진짜 마지막(작은) 조각
        mid = size - extra - last            # 밀려난 덩어리가 들어갈 자리
        if last and mid > 0 and mid % s == 0:
            e.update({"verdict": "shifted", "part_size": s, "parts_moved": k,
                      "last_size": last, "mid": mid, "extra": extra})
            return e
    e["verdict"] = "shifted_unknown_parts"
    e["extra"] = extra
    return e


# ──────────────────────────────────────────────────────────────
# 3. 고친 결과가 진짜 맞는지 — 목차를 읽어 항목을 찍어봅니다
# ──────────────────────────────────────────────────────────────
def mapper(d: dict):
    """올바른 파일에서의 위치 → 지금 파일에서의 위치."""
    mid, extra, size = d["mid"], d["extra"], d["size"]

    def to_phys(off: int) -> int:
        if off < mid:
            return off
        if off < mid + extra:
            return off - mid + d["zip_end"]
        return off - extra

    return to_phys


def read_correct(f, d: dict, off: int, n: int) -> bytes:
    """올바른 파일 기준 [off, off+n) 을 지금 파일에서 이어붙여 읽습니다.

    구간 경계를 넘으면 물리 위치가 튀므로, **경계에서 잘라** 여러 번 읽습니다.
    (바이트마다 확인하면 맞긴 한데 파이썬에선 견딜 수 없이 느립니다)
    """
    to_phys = mapper(d)
    edges = (d["mid"], d["mid"] + d["extra"], d["size"])
    out = bytearray()
    while n > 0 and off < d["size"]:
        stop = next((e for e in edges if e > off), d["size"])
        step = min(n, stop - off)
        f.seek(to_phys(off))
        blk = f.read(step)
        if not blk:
            break
        out += blk
        off += len(blk)
        n -= len(blk)
    return bytes(out)


def verify(f, d: dict, sample: int = 24) -> tuple[int, int, list[str]]:
    """목차 항목 몇 개를 골라 로컬 헤더가 제자리에 있는지 봅니다."""
    cd = read_correct(f, d, d["cd_offset"], d["cd_size"])

    entries, pos = [], 0
    while pos + 46 <= len(cd) and cd[pos:pos + 4] == CD_SIG:
        n_len, x_len, c_len = struct.unpack("<HHH", cd[pos + 28:pos + 34])
        name = cd[pos + 46:pos + 46 + n_len]
        lfh = struct.unpack("<I", cd[pos + 42:pos + 46])[0]
        if lfh == 0xFFFFFFFF:                       # ZIP64 — 추가 필드에 있습니다
            ex = cd[pos + 46 + n_len:pos + 46 + n_len + x_len]
            j = 0
            while j + 4 <= len(ex):
                tag, ln = struct.unpack("<HH", ex[j:j + 4])
                if tag == 0x0001:
                    body = ex[j + 4:j + 4 + ln]
                    # 압축·원본 크기가 0xFFFFFFFF 였던 만큼만 앞에 붙습니다.
                    sizes = struct.unpack("<II", cd[pos + 20:pos + 28])
                    skip = 8 * sum(1 for v in sizes if v == 0xFFFFFFFF)
                    if len(body) >= skip + 8:
                        lfh = struct.unpack("<Q", body[skip:skip + 8])[0]
                    break
                j += 4 + ln
        entries.append((lfh, name))
        pos += 46 + n_len + x_len + c_len

    if not entries:
        return 0, 0, ["목차를 한 항목도 못 읽었습니다"]

    step = max(1, len(entries) // sample)
    picks = entries[::step][:sample]
    ok, bad = 0, []
    for lfh, name in picks:
        head = read_correct(f, d, lfh, 30 + len(name))
        if len(head) < 30 or head[:4] != LFH_SIG:
            bad.append(f"{name[:60]!r} @ {lfh:,} — 로컬 헤더 없음")
            continue
        n_len = struct.unpack("<H", head[26:28])[0]
        got = read_correct(f, d, lfh + 30, n_len)
        if got != name:
            bad.append(f"@ {lfh:,} — 이름 불일치 {got[:40]!r} != {name[:40]!r}")
            continue
        ok += 1
    return ok, len(picks), bad


# ──────────────────────────────────────────────────────────────
# 4. 고치기 — 뒤쪽 (밀려난 덩어리 + 마지막 조각) 만 다시 씁니다
# ──────────────────────────────────────────────────────────────
def bak_path(p: Path) -> Path:
    return p.with_suffix(p.suffix + ".tailbak")


def apply(p: Path, d: dict) -> None:
    mid, extra, last, size = d["mid"], d["extra"], d["last_size"], d["size"]
    bak = bak_path(p)
    region = extra + last
    assert mid + region == size, "구간 계산이 안 맞습니다"

    # ① 손대는 구간만 통째로 백업합니다. 중간에 끊겨도 --restore 로 되돌아갑니다.
    if bak.exists() and bak.stat().st_size == region:
        print(f"  백업이 이미 있습니다: {bak.name} ({human(region)})")
    else:
        print(f"  ① 백업 {human(region)} → {bak.name}")
        with open(p, "rb") as src, open(bak, "wb") as dst:
            src.seek(mid)
            left = region
            while left:
                b = src.read(min(left, 1 << 24))
                if not b:
                    raise SystemExit("❌ 백업 중 파일이 짧게 끝났습니다.")
                dst.write(b)
                left -= len(b)
            dst.flush()
            import os
            os.fsync(dst.fileno())

    # ② 백업에서 **올바른 순서로** 되씁니다.
    #    지금:  [마지막조각 last][밀려난덩어리 extra]
    #    되어야: [밀려난덩어리 extra][마지막조각 last]
    print("  ② 순서 바꿔 되쓰는 중…")
    with open(bak, "rb") as b, open(p, "r+b") as out:
        out.seek(mid)
        for start, ln in ((last, extra), (0, last)):
            b.seek(start)
            left = ln
            while left:
                blk = b.read(min(left, 1 << 24))
                if not blk:
                    raise SystemExit("❌ 백업이 짧습니다. --restore 로 되돌리세요.")
                out.write(blk)
                left -= len(blk)
        out.flush()
        import os
        os.fsync(out.fileno())
    print("  ✅ 다 썼습니다.")


def restore(p: Path) -> None:
    bak = bak_path(p)
    if not bak.exists():
        raise SystemExit(f"❌ 백업이 없습니다: {bak}")
    size = p.stat().st_size
    mid = size - bak.stat().st_size
    print(f"  되돌리는 중 — {bak.name} → {p.name} 의 {mid:,} 위치부터")
    with open(bak, "rb") as b, open(p, "r+b") as out:
        out.seek(mid)
        while True:
            blk = b.read(1 << 24)
            if not blk:
                break
            out.write(blk)
    print("  ✅ 되돌렸습니다.")


# ──────────────────────────────────────────────────────────────
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("zip", help="고칠 zip 경로")
    ap.add_argument("--apply", action="store_true", help="실제로 고칩니다")
    ap.add_argument("--restore", action="store_true", help="백업으로 되돌립니다")
    a = ap.parse_args(argv)

    p = Path(a.zip).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"❌ 없는 파일: {p}")

    if a.restore:
        restore(p)
        return

    size = p.stat().st_size
    print("=" * 68)
    print(f" {p.name}   {human(size)}")
    print("=" * 68)

    with open(p, "rb") as f:
        print("\n[목차 찾는 중] 뒤에서부터 훑습니다 …")
        d = diagnose(f, size)

        print(f"  EOCD          {d['eocd_at']:,}")
        print(f"  {d['kind']:<13} {d['rec_at']:,}")
        print(f"  항목 수       {d['entries']:,}개")
        print(f"  목차 크기     {human(d['cd_size'])}")
        print(f"  목차가 있어야 할 곳 {d['cd_offset']:,}")
        print(f"  목차가 실제 있는 곳 {d['cd_phys']:,}")
        print()
        print(f"  밀린 양 (파일 끝 기준)  {d['extra_by_tail']:,}")
        print(f"  밀린 양 (목차 기준)     {d['extra_by_cd']:,}")

        if d["verdict"] == "ok":
            print("\n✅ 멀쩡한 zip 입니다. 고칠 게 없습니다.")
            return
        if d["verdict"] != "shifted":
            print(f"\n❌ 판정 불가 ({d['verdict']}). 두 숫자가 다르면 단순한 순서 문제가 "
                  "아닙니다.\n   이 출력을 그대로 공유해주세요. 함부로 고치지 않습니다.")
            return

        print(f"\n  ✅ 두 숫자가 같습니다 — 조각 순서 문제입니다.")
        print(f"     조각 크기      {human(d['part_size'])}")
        print(f"     밀려난 조각    {d['parts_moved']}개")
        print(f"     마지막 조각    {human(d['last_size'])}")
        print(f"     되돌릴 자리    {d['mid']:,}")

        print("\n[검증] 목차를 읽어 항목이 제자리에 있는지 봅니다 …")
        ok, total, bad = verify(f, d)
        print(f"  {ok}/{total} 통과")
        for b in bad[:5]:
            print(f"    ❌ {b}")
        if ok < total:
            print("\n❌ 검증을 통과 못 했습니다. **고치지 않습니다.**")
            return
        print("  ✅ 고른 항목이 전부 제자리입니다.")

    if not a.apply:
        print(f"\n미리보기입니다. 고치려면 --apply 를 붙이세요.")
        print(f"  손대는 구간은 뒤쪽 {human(d['extra'] + d['last_size'])} 뿐입니다 "
              "(앞 84GB 는 안 건드립니다).")
        return

    print("\n[고치는 중]")
    apply(p, d)

    # ── 진짜로 열리는지 확인 (이게 최종 판정입니다)
    print("\n[확인] 파이썬으로 열어봅니다 …")
    import zipfile

    with zipfile.ZipFile(p) as z:
        names = z.namelist()
    print(f"  ✅ 열립니다 — 항목 {len(names):,}개")
    print(f"     예: {names[0]}")
    print(f"\n백업 {bak_path(p).name} 은 지워도 됩니다 "
          f"({human(bak_path(p).stat().st_size)}).")
    print("다음: uv run python prepare_local.py --chunk TL02 --margins 2.5,-320")


if __name__ == "__main__":
    main()
