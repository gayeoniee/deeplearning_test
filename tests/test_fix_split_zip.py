"""조각 순서가 뒤바뀐 zip 을 고치는 도구 검증.

    uv run python tests/test_fix_split_zip.py

실제로 당한 것: TL02.zip 80GB 를 받았는데 `aihubshell` 이 조각을 붙일 때
마지막 두 조각이 뒤바뀌었습니다. 크기는 정확했고 `file` 도 "Zip archive data"
라고 했는데 파이썬만 `BadZipFile` 로 죽었습니다 — 목차(EOCD)가 파일 맨 뒤가
아니라 1 GiB 앞에 있었기 때문입니다.

여기서는 같은 상황을 작게 만들어 놓고
  ① 진단이 "밀린 양" 을 두 경로로 따로 구해 일치하는지 보는가
  ② 멀쩡한 zip 을 건드리지 않는가
  ③ 고친 뒤 진짜로 열리고 **내용이 원본과 같은가**
  ④ 미리보기가 파일을 안 건드리는가
  ⑤ 중간에 끊겨도 되돌릴 수 있는가
를 봅니다.
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
    contents = {}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for i in range(n_files):
            name = f"152.데이터/01.데이터/1.Training/img_{i:05d}.jpg"
            body = (f"{i}".encode() * size_each)[:size_each]
            contents[name] = body
            z.writestr(name, body)
    return contents


def scramble(path: Path, part_size: int) -> None:
    """마지막 두 조각을 뒤바꿔 놓습니다 (aihubshell 이 낸 것과 같은 모양).

    [… 앞 조각들][마지막(작은) 조각]  →  [… 앞 조각들][마지막 조각][한 조각]
    이 되도록, 마지막 온전한 조각을 맨 뒤로 보냅니다.
    """
    data = path.read_bytes()
    size = len(data)
    last = size % part_size                    # 진짜 마지막 조각 (작음)
    assert last, "마지막 조각이 딱 떨어지면 이 상황이 안 만들어집니다"
    mid = size - last - part_size              # 뒤로 보낼 조각의 시작
    assert mid > 0 and mid % part_size == 0
    path.write_bytes(data[:mid] + data[mid + part_size:] + data[mid:mid + part_size])


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ──────────────────────────────────────────────────────────────
def test_diagnoses_and_repairs():
    print("\n[고치기] 뒤바뀐 조각을 되돌리는가")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "TL02.zip"
        want = make_zip(p, 300, 4096)
        good_sha, good_size = sha(p), p.stat().st_size

        part = 131072                                  # 128KB 짜리 조각
        assert good_size % part, "표본을 조각 크기의 배수가 아니게 잡아야 합니다"
        scramble(p, part)

        check("망가뜨린 뒤 크기는 같다", p.stat().st_size == good_size)
        check("망가뜨린 뒤 내용은 다르다", sha(p) != good_sha)
        opened = True
        try:
            zipfile.ZipFile(p).namelist()
        except zipfile.BadZipFile:
            opened = False
        check("망가뜨린 zip 은 파이썬이 못 연다", not opened)

        with open(p, "rb") as f:
            d = fx.diagnose(f, good_size)
            check("판정: 조각 순서 문제", d["verdict"] == "shifted", d["verdict"])
            check("밀린 양을 두 경로로 구해 일치한다",
                  d["extra_by_tail"] == d["extra_by_cd"],
                  f"{d['extra_by_tail']} vs {d['extra_by_cd']}")
            check("조각 크기를 맞힌다", d.get("part_size") == part,
                  f"got {d.get('part_size')} want {part}")
            check("되돌릴 자리를 맞힌다",
                  d.get("mid") == good_size - (good_size % part) - part,
                  f"got {d.get('mid')}")
            ok, total, bad = fx.verify(f, d)
            check("고치기 전 검증이 통과한다", ok == total and total > 0,
                  f"{ok}/{total} {bad[:1]}")

        _capture(fx.main, [str(p), "--apply"])
        check("고친 뒤 원본과 **바이트가 같다**", sha(p) == good_sha)
        with zipfile.ZipFile(p) as z:
            got = {n: z.read(n) for n in z.namelist()}
        check("내용이 전부 같다", got == want, f"{len(got)}/{len(want)}개")


def test_leaves_a_healthy_zip_alone():
    print("\n[안전] 멀쩡한 zip 은 안 건드리는가")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "ok.zip"
        make_zip(p, 50, 2048)
        before = sha(p)
        _, log = _capture(fx.main, [str(p), "--apply"])
        check("고칠 게 없다고 말한다", "고칠 게 없습니다" in log, log.strip()[-60:])
        check("파일을 안 건드렸다", sha(p) == before)
        check("백업도 안 만들었다", not fx.bak_path(p).exists())


def test_preview_writes_nothing():
    print("\n[미리보기] --apply 없이는 안 쓰는가")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "TL02.zip"
        make_zip(p, 300, 4096)
        part = 131072
        scramble(p, part)
        before = sha(p)
        _, log = _capture(fx.main, [str(p)])
        check("미리보기라고 말한다", "미리보기" in log)
        check("손대는 구간을 알려준다", "앞" in log and "안 건드립니다" in log)
        check("파일이 그대로다", sha(p) == before)
        check("백업이 없다", not fx.bak_path(p).exists())


def test_restore_after_interrupt():
    print("\n[되돌리기] 중간에 끊겨도 복구되는가")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "TL02.zip"
        make_zip(p, 300, 4096)
        part = 131072
        scramble(p, part)
        broken = sha(p)

        with open(p, "rb") as f:
            d = fx.diagnose(f, p.stat().st_size)
        _capture(fx.apply, p, d)
        check("백업이 남아 있다", fx.bak_path(p).exists())

        # 되돌리면 **고치기 직전 상태**(뒤바뀐 채)로 돌아가야 합니다
        _capture(fx.restore, p)
        check("고치기 전 상태로 돌아간다", sha(p) == broken)


def test_refuses_when_numbers_disagree():
    print("\n[거절] 두 숫자가 안 맞으면 안 고치는가")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "junk.zip"
        make_zip(p, 50, 2048)
        with open(p, "ab") as f:                    # 뒤에 쓰레기를 붙입니다
            f.write(b"\x00" * 5000)                 # (조각 순서 문제가 아님)
        before = sha(p)
        _, log = _capture(fx.main, [str(p), "--apply"])
        check("판정 불가라고 말한다", "판정 불가" in log, log.strip()[-70:])
        check("안 건드린다", sha(p) == before)


if __name__ == "__main__":
    print("조각 순서 뒤바뀐 zip 고치기 검증")
    for fn in (test_diagnoses_and_repairs,
               test_leaves_a_healthy_zip_alone,
               test_preview_writes_nothing,
               test_restore_after_interrupt,
               test_refuses_when_numbers_disagree):
        fn()
    print()
    if FAILS:
        print(f"{len(FAILS)} check(s) failed:")
        for f in FAILS:
            print(f"  - {f}")
        sys.exit(1)
    print("all checks passed")
