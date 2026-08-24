# -*- coding: utf-8 -*-
"""--recrop 이 분할을 건드리지 않고 크롭 태그만 추가하는지.

왜 이 검사가 필요한가
---------------------
f320 이 나중에 필요해졌을 때 `--chunk` 를 다시 돌리면 중복제거·분할이 다시 돌아
**어떤 개체가 holdout 에 가는지가 바뀝니다.** 그러면 지금까지 잰 숫자와 비교가
불가능해지고, holdout 에 있던 개체가 학습셋으로 넘어가 시험지가 오염됩니다.

`--recrop` 은 manifest_final.parquet 을 읽기만 합니다. 그 성질을 여기서 못 박습니다.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd                                              # noqa: E402
from PIL import Image                                            # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, msg: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"{'✅' if cond else '❌'} {name}" + (f"  — {msg}" if msg and not cond else ""))


def _build(tmp: Path, n: int = 6) -> tuple[Path, Path]:
    """zip 안에 이미지가 든, 실제와 같은 구조의 작은 데이터셋을 만듭니다."""
    work = tmp / "work"
    (work / "manifests").mkdir(parents=True)
    raw = tmp / "raw"
    raw.mkdir()

    zp = raw / "VL01.zip"
    members, rows = [], []
    with zipfile.ZipFile(zp, "w") as zf:
        for i in range(n):
            m = f"반려견/피부/일반카메라/유증상/A1/IMG_{i}.jpg"
            b = tmp / f"_t{i}.jpg"
            Image.new("RGB", (1920, 1080), (30 + i, 90, 140)).save(b, "JPEG")
            zf.write(b, m)
            b.unlink()
            members.append(m)
    for i, m in enumerate(members):
        rows.append({
            "image_path": f"{zp}!{m}",
            "zip_path": str(zp),
            "zip_member": m,
            "animal_id": f"dog{i // 2}",
            "class": "A1",
            "bbox": [800.0, 400.0, 1000.0, 600.0],
            "width": 1920, "height": 1080,
            "fold": i % 2, "split": "train" if i % 2 else "holdout",
        })
    df = pd.DataFrame(rows)
    df.to_parquet(work / "manifests" / "manifest_final.parquet")
    return work, raw


def _run(work: Path, *args: str) -> subprocess.CompletedProcess:
    e = dict(os.environ, DOG_SKIN_WORK=str(work), DOG_SKIN_DATA=str(work.parent / "raw"),
             PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, str(ROOT / "prepare_local.py"), *args],
                          capture_output=True, text=True, env=e, cwd=str(ROOT))


def test_recrop_adds_a_tag_without_touching_the_split():
    with tempfile.TemporaryDirectory() as d:
        work, _ = _build(Path(d))
        mf = work / "manifests" / "manifest_final.parquet"
        before_bytes = mf.read_bytes()
        before_mtime = mf.stat().st_mtime_ns

        r = _run(work, "--recrop", "f320")
        check("--recrop 이 성공한다", r.returncode == 0, r.stdout[-1500:] + r.stderr[-800:])
        check("f320 폴더가 생긴다", (work / "crops" / "f320").exists())
        made = list((work / "crops" / "f320").rglob("*.jpg"))
        check("전 행이 크롭된다", len(made) == 6, f"{len(made)}장")
        check("매니페스트 내용이 그대로다", mf.read_bytes() == before_bytes)
        check("매니페스트를 열어 쓰지도 않았다", mf.stat().st_mtime_ns == before_mtime)


def test_filename_comes_from_image_path_not_the_zip_location():
    """zip 이 다른 폴더로 옮겨져도 크롭 파일 이름이 같아야 합니다.

    이게 깨지면 새로 만든 f320 이 기존 full/m2.5 와 짝이 안 맞아
    Kaggle 에서 '크롭 0%' 가 됩니다.
    """
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        work, raw = _build(tmp)
        r1 = _run(work, "--recrop", "f320")
        assert r1.returncode == 0, r1.stdout[-1500:] + r1.stderr[-800:]
        names_a = sorted(p.name for p in (work / "crops" / "f320").rglob("*.jpg"))

        moved = tmp / "다른곳"
        moved.mkdir()
        (raw / "VL01.zip").rename(moved / "VL01.zip")
        import shutil
        shutil.rmtree(work / "crops")

        r2 = _run(work, "--recrop", "f320", "--raw", str(moved))
        check("옮긴 위치를 --raw 로 알려주면 된다", r2.returncode == 0,
              r2.stdout[-1500:] + r2.stderr[-800:])
        names_b = sorted(p.name for p in (work / "crops" / "f320").rglob("*.jpg"))
        check("파일 이름이 위치와 무관하게 같다", names_a == names_b and len(names_a) == 6,
              f"{len(names_a)} vs {len(names_b)}")


def test_missing_source_stops_before_cropping():
    """원본을 못 열면 크롭을 시작조차 하지 않아야 합니다."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        work, raw = _build(tmp)
        (raw / "VL01.zip").unlink()
        r = _run(work, "--recrop", "f320")
        check("원본이 없으면 실패로 끝난다", r.returncode != 0)
        out = r.stdout + r.stderr
        check("크롭을 시작하지 않았다고 알린다", "크롭을 시작하지 않았습니다" in out)
        check("--raw 를 안내한다", "--raw" in out)
        check("크롭 폴더를 만들지 않았다", not (work / "crops" / "f320").exists())


def test_recrop_and_finalize_together_is_refused():
    """분할을 다시 계산하는 --finalize 와 같이 쓰면 목적이 정반대입니다."""
    with tempfile.TemporaryDirectory() as d:
        work, _ = _build(Path(d))
        r = _run(work, "--recrop", "f320", "--finalize")
        check("같이 쓰면 거부한다", r.returncode != 0)
        check("왜 안 되는지 설명한다", "정반대" in (r.stdout + r.stderr))


def test_existing_tags_survive():
    """새 태그를 만들어도 기존 태그 파일이 줄지 않아야 합니다."""
    with tempfile.TemporaryDirectory() as d:
        work, _ = _build(Path(d))
        assert _run(work, "--recrop", "m2.5").returncode == 0
        n_before = len(list((work / "crops" / "m2.5").rglob("*.jpg")))
        r = _run(work, "--recrop", "f320")
        n_after = len(list((work / "crops" / "m2.5").rglob("*.jpg")))
        check("기존 태그가 그대로 남는다", n_before == n_after == 6, f"{n_before} → {n_after}")
        check("대조표를 찍어준다", "[대조]" in r.stdout)


if __name__ == "__main__":
    for fn in [test_recrop_adds_a_tag_without_touching_the_split,
               test_filename_comes_from_image_path_not_the_zip_location,
               test_missing_source_stops_before_cropping,
               test_recrop_and_finalize_together_is_refused,
               test_existing_tags_survive]:
        print(f"\n── {fn.__name__} ──")
        try:
            fn()
        except Exception as exc:                                  # noqa: BLE001
            check(fn.__name__, False, f"{type(exc).__name__}: {exc}")
    print(f"\n{'=' * 60}\n통과 {len(PASS)} / {len(PASS) + len(FAIL)}")
    if FAIL:
        print("실패:", ", ".join(FAIL))
        sys.exit(1)
