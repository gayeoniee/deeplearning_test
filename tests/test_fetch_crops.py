"""zip 에서 크롭을 꺼낼 때 **조용히 반쪽짜리가 되지 않는가**.

    uv run python tests/test_fetch_crops.py

실제로 당한 것 (2026-08-28, RunPod): 네트워크 볼륨이 차서 `unzip` 16개가 전부
`Exit 50`(disk full) 로 죽었는데, `pgrep -c unzip` 이 0 이라 "전부 끝남" 으로 보였고
파일 개수만 세면 331,630 이라는 그럴듯한 숫자가 나왔습니다 (있어야 할 건 365,428).

여기서 못 박는 것 두 가지:

1. **`unzip -n` 은 잘린 파일을 고치지 못합니다.** "있으니까 건너뜀" 으로 넘어가고,
   학습 때 열다가 깨진 JPEG 로 죽습니다 → zip 이 적어둔 **크기와 대조**해야 합니다.
2. **zip 에는 `--finalize` 이전 크롭이 들어 있습니다.** 매니페스트가 안 쓰는 건
   풀어봐야 디스크만 먹습니다 → 필요한 것만 꺼내고 나머지는 지울 수 있어야 합니다.
"""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAILS: list[str] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {extra}" if extra else ""))
    if not cond:
        FAILS.append(f"{name} {extra}".strip())


def _capture(fn, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*a, **kw)
    return out, buf.getvalue()


def _stage(tmp: Path):
    """디스크를 일부러 망가진 상태로 만들어 둡니다 (정상/잘림/없음/안 쓰는 것)."""
    import pandas as pd

    from src.crop import _out_path

    work = tmp / "work"
    (work / "manifests").mkdir(parents=True)
    paths = [f"/raw/img_{i:04d}.jpg" for i in range(60)]
    pd.DataFrame({"image_path": paths[:50]}).to_parquet(
        work / "manifests" / "manifest_final.parquet")

    zp = tmp / "crops.zip"
    with zipfile.ZipFile(zp, "w") as z:
        for tag in ("f320", "m2.5"):
            for p in paths:                                 # zip 은 60장 (finalize 이전)
                z.writestr(_out_path(p, Path("crops"), tag).as_posix(), b"J" * 100)

    for tag in ("f320", "m2.5"):
        for i, p in enumerate(paths):
            dest = work / _out_path(p, Path("crops"), tag)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if i < 30:
                dest.write_bytes(b"J" * 100)                # 정상
            elif i < 33:
                dest.write_bytes(b"J" * 7)                  # ★ 디스크가 차서 잘림
            elif i < 50:
                pass                                        # 아직 안 나옴
            else:
                dest.write_bytes(b"J" * 100)                # 매니페스트가 안 쓰는 것
    return work, zp


def test_preview_counts_without_touching_disk():
    print("\n[미리보기] 세기만 하고 아무것도 안 쓰는가")
    import importlib

    fc = importlib.import_module("tools.fetch_crops")
    with tempfile.TemporaryDirectory() as tmp:
        work, zp = _stage(Path(tmp))
        _, log = _capture(fc.main, [str(zp), "--work", str(work)])

        check("필요한 장수를 매니페스트에서 읽는다", "50행" in log and "필요" in log)
        check("잘린 것을 따로 센다", "잘린 것 3장" in log, log)
        check("없는 것을 센다", "꺼낼 것         17장" in log or "17장" in log)
        check("안 쓰는 것을 센다", "안 쓰는 것       10장" in log or "10장" in log)
        check("미리보기라고 말한다", "미리보기" in log)
        # 정상 30 + 잘림 3 + 안 쓰는 것 10 = 43 (없는 17장은 아직 안 나왔습니다)
        check("미리보기는 파일을 안 건드린다",
              len(list((work / "crops" / "f320").rglob("*.jpg"))) == 43)


def test_apply_fixes_truncated_and_prunes():
    print("\n[적용] 잘린 것을 고치고 안 쓰는 것을 지우는가")
    import importlib

    fc = importlib.import_module("tools.fetch_crops")
    with tempfile.TemporaryDirectory() as tmp:
        work, zp = _stage(Path(tmp))
        _, log = _capture(fc.main, [str(zp), "--work", str(work),
                                    "--apply", "--prune", "--workers", "4"])

        for tag in ("f320", "m2.5"):
            got = sorted((work / "crops" / tag).rglob("*.jpg"))
            check(f"{tag}: 필요한 50장만 남는다", len(got) == 50, str(len(got)))
            check(f"{tag}: 잘린 파일이 없다",
                  all(g.stat().st_size == 100 for g in got),
                  str(sorted({g.stat().st_size for g in got})))
            check(f"{tag}: 쓰다 만 .part 가 안 남는다",
                  not list((work / "crops" / tag).rglob("*.part")))
        check("끝나고 다시 세어 확인한다", "100.00%" in log)


def test_stops_when_still_short():
    """디스크가 또 차서 못 채우면 **성공이라고 말하면 안 됩니다.**"""
    print("\n[정직] 못 채웠으면 멈추는가")
    import importlib

    fc = importlib.import_module("tools.fetch_crops")
    with tempfile.TemporaryDirectory() as tmp:
        work, zp = _stage(Path(tmp))
        real = fc.fetch
        fc.fetch = lambda *a, **kw: 0            # 한 장도 못 꺼낸 척
        try:
            died = ""
            try:
                _capture(fc.main, [str(zp), "--work", str(work), "--apply"])
            except SystemExit as exc:
                died = str(exc)
            check("모자라면 멈춘다", "아직 모자랍니다" in died,
                  died.splitlines()[1] if died else "안 멈춤")
            check("디스크를 의심하라고 말한다", "볼륨 크기" in died)
        finally:
            fc.fetch = real


if __name__ == "__main__":
    print("필요한 크롭만 온전하게 꺼내는가")
    for fn in (test_preview_counts_without_touching_disk,
               test_apply_fixes_truncated_and_prunes,
               test_stops_when_still_short):
        fn()
    print()
    if FAILS:
        print(f"{len(FAILS)} check(s) failed:")
        for f in FAILS:
            print(f"  - {f}")
        sys.exit(1)
    print("all checks passed")
