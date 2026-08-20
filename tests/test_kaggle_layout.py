"""Kaggle 데이터 배치 테스트.

⚠️ Kaggle 은 **업로드한 zip 을 데이터셋에 넣을 때 자동으로 풀어버립니다.**
   그래서 `/kaggle/input/<데이터셋>/crops/...` 만 있고 zip 은 없습니다.
   zip 만 찾는 코드는 여기서 FileNotFoundError 로 죽습니다.

   또 `/kaggle/input` 은 읽기 전용이고 `/kaggle/working` 은 20GB 제한이라,
   크롭 45,885장을 복사하면 시간도 용량도 낭비입니다 — 링크를 걸어야 합니다.

    python tests/test_kaggle_layout.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_TMP = Path(tempfile.mkdtemp(prefix="dogskin_kaggle_"))

import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402
from PIL import Image                                           # noqa: E402

PASS, FAIL = [], []


def check(name: str, cond: bool, msg: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"{'✅' if cond else '❌'} {name}" + (f"\n     {msg}" if msg and not cond else ""))


def make_prepared(root: Path, n: int = 6) -> Path:
    """prepare_local.py 산출물과 같은 구조를 만듭니다."""
    (root / "crops" / "m1.5").mkdir(parents=True, exist_ok=True)
    (root / "crops" / "full").mkdir(parents=True, exist_ok=True)
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    rows = []
    rng = np.random.default_rng(0)
    for i in range(n):
        for tag in ("m1.5", "full"):
            p = root / "crops" / tag / f"img{i}.jpg"
            Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(p)
        rows.append({"image_name": f"img{i}.jpg", "label": "A1",
                     "crop_rel": f"m1.5/img{i}.jpg"})
    pd.DataFrame(rows).to_parquet(root / "manifests" / "manifest_final.parquet")
    return root


def fresh_env(tag: str) -> Path:
    """작업 폴더를 비우고 env 를 다시 읽게 합니다."""
    w = _TMP / f"work_{tag}"
    shutil.rmtree(w, ignore_errors=True)
    w.mkdir(parents=True)
    os.environ["DOG_SKIN_WORK"] = str(w)
    return w


# ──────────────────────────────────────────────────────────────
def test_finds_extracted_dir():
    """Kaggle 처럼 이미 풀려 있는 폴더를 찾아야 합니다."""
    from src import env

    src = make_prepared(_TMP / "kaggle_input" / "dogskin-prepared")
    w = fresh_env("dir")
    # 자동 탐색은 test_autodetect_... 에서 따로 봅니다. 여기서는 경로를 직접 줍니다.
    env.load_prepared(src, dest=w)
    check("풀려 있는 폴더를 받아들인다", (w / "crops").exists() and (w / "manifests").exists(),
          f"{list(w.iterdir())}")
    check("매니페스트가 읽힌다", (w / "manifests" / "manifest_final.parquet").exists())
    # ⚠️ rglob 은 심볼릭 링크 하위로 안 들어갑니다 — 태그별로 세야 합니다
    n = sum(sum(1 for _ in (w / "crops" / t).rglob("*.jpg"))
            for t in (p.name for p in (w / "crops").iterdir() if p.is_dir()))
    check("크롭이 다 보인다 (12장)", n == 12, f"{n}장")


def test_readonly_source_is_linked_not_copied():
    """읽기 전용 원본을 복사하면 /kaggle/working 20GB 를 잡아먹습니다."""
    from src import env

    src = make_prepared(_TMP / "ro_input" / "ds")
    w = fresh_env("link")
    env.load_prepared(src, dest=w)
    crops = w / "crops"
    # 태그 단위로 링크합니다 (여러 데이터셋의 태그를 합칠 수 있어야 하므로)
    tags = sorted(p.name for p in crops.iterdir() if p.is_dir())
    check("크롭은 태그별 링크다 (복사 아님)", all((crops / t).is_symlink() for t in tags),
          f"복사되었습니다 — Kaggle 20GB 제한에 걸립니다: {tags}")
    check("링크가 원본을 가리킨다",
          all((crops / t).resolve() == (src / "crops" / t).resolve() for t in tags))
    # 매니페스트는 복사여야 합니다 (원본이 읽기 전용일 수 있으므로)
    man = w / "manifests"
    check("매니페스트는 복사한다", man.exists() and not man.is_symlink())


def test_link_is_idempotent():
    from src import env

    src = make_prepared(_TMP / "idem_input" / "ds")
    w = fresh_env("idem")
    env.load_prepared(src, dest=w)
    try:
        env.load_prepared(src, dest=w)      # 두 번 불러도 죽으면 안 됩니다
        check("두 번 불러도 안 죽는다", (w / "crops").exists())
    except Exception as exc:                                    # noqa: BLE001
        check("두 번 불러도 안 죽는다", False, repr(exc))


def test_zip_path_still_works():
    """기존 Colab 경로(zip)가 깨지지 않았는지."""
    from src import env

    src = make_prepared(_TMP / "zipsrc")
    z = _TMP / "dogskin_prepared.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for p in src.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(src))
    w = fresh_env("zip")
    env.load_prepared(z, dest=w)
    check("zip 도 그대로 풀린다", (w / "crops" / "m1.5").is_dir())
    check("zip 경로는 링크가 아니다", not (w / "crops").is_symlink())
    n = sum(1 for _ in (w / "crops").rglob("*.jpg"))
    check("zip 에서 크롭이 다 나온다 (12장)", n == 12, f"{n}장")


def test_autodetect_prefers_extracted_when_no_zip():
    """검색 경로에 zip 이 없고 풀린 폴더만 있을 때 자동으로 찾아야 합니다."""
    from src import env

    base = _TMP / "autoroot"
    make_prepared(base / "some-dataset")
    w = fresh_env("auto")
    orig = env._search_roots
    env._search_roots = lambda: [base]              # /kaggle/input 을 흉내
    try:
        found, kind = env.find_prepared(dest=w)
        check("zip 없이 풀린 폴더를 자동 탐색한다", kind == "dir", f"{kind} {found}")
        check("찾은 경로가 맞다", found.resolve() == (base / "some-dataset").resolve(),
              f"{found}")
    finally:
        env._search_roots = orig


def test_missing_gives_useful_error():
    from src import env

    w = fresh_env("missing")
    orig = env._search_roots
    env._search_roots = lambda: [_TMP / "아무것도없음"]
    try:
        env.find_prepared(dest=w)
        check("없으면 예외를 던진다", False, "예외가 안 났습니다")
    except FileNotFoundError as exc:
        m = str(exc)
        check("없으면 예외를 던진다", True)
        check("오류 메시지가 Kaggle 자동 해제를 알려준다", "자동으로 풀" in m, m[:200])
    finally:
        env._search_roots = orig


def test_split_upload_is_merged():
    """태그를 나눠 올린 두 데이터셋이 하나로 합쳐져야 합니다.

    5.7GB zip 은 업로드가 자주 끊깁니다. crops/m1.5 와 crops/full 을
    따로 올릴 수 있어야 하고, 노트북에서는 하나로 보여야 합니다.
    """
    from src import env

    base = _TMP / "split_input"
    a = make_prepared(base / "ds-m15")          # crops(m1.5, full) + manifests
    b = base / "ds-extra"                       # crops 만 있는 부분 업로드
    (b / "crops" / "f320").mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((32, 32, 3), np.uint8)).save(b / "crops" / "f320" / "x.jpg")

    w = fresh_env("split")
    orig = env._search_roots
    env._search_roots = lambda: [base]
    try:
        found = env.find_prepared_all(dest=w)
        check("두 입력을 모두 찾는다", len(found) == 2, f"{[str(p) for p, _ in found]}")
        env.load_prepared(dest=w)
    finally:
        env._search_roots = orig

    tags = sorted(p.name for p in (w / "crops").iterdir())
    check("두 데이터셋의 태그가 합쳐진다", tags == ["f320", "full", "m1.5"], f"{tags}")
    check("매니페스트는 가진 쪽에서 온다", (w / "manifests" / "manifest_final.parquet").exists())
    check("태그마다 개별 링크다", all((w / "crops" / t).is_symlink() for t in tags),
          f"{[(t, (w / 'crops' / t).is_symlink()) for t in tags]}")


def test_partial_dir_without_manifests_is_accepted():
    """crops 만 있는 폴더도 입력으로 받아야 합니다 (나눠 올린 쪽)."""
    from src import env

    d = _TMP / "partial" / "only-crops"
    (d / "crops" / "m1.5").mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((32, 32, 3), np.uint8)).save(d / "crops" / "m1.5" / "a.jpg")
    check("crops 만 있어도 인식한다", env._looks_partial(d))


def test_manifest_rebase_works_through_link():
    """링크를 통해서도 crop_path 가 실제 파일을 가리켜야 합니다."""
    from src import env, labels

    src = make_prepared(_TMP / "rebase_input" / "ds")
    w = fresh_env("rebase")
    env.load_prepared(src, dest=w)
    # rebase_paths 는 env.work_root()/crops 를 씁니다 (fresh_env 가 이미 맞춰둠)
    df = labels.rebase_paths(labels.load(w / "manifests" / "manifest_final.parquet"))
    ok = df["crop_path"].map(lambda p: Path(p).exists()).all()
    check("링크 너머의 크롭 파일이 실제로 열린다", bool(ok),
          f"{df['crop_path'].head(2).tolist()}")


if __name__ == "__main__":
    print(f"작업 폴더: {_TMP}\n")
    for fn in [test_finds_extracted_dir, test_readonly_source_is_linked_not_copied,
               test_link_is_idempotent, test_zip_path_still_works,
               test_autodetect_prefers_extracted_when_no_zip,
               test_missing_gives_useful_error, test_split_upload_is_merged,
               test_partial_dir_without_manifests_is_accepted,
               test_manifest_rebase_works_through_link]:
        print(f"\n── {fn.__name__} ──")
        try:
            fn()
        except Exception as exc:                                # noqa: BLE001
            check(fn.__name__, False, f"{type(exc).__name__}: {exc}")

    print(f"\n{'=' * 60}\n통과 {len(PASS)} / {len(PASS) + len(FAIL)}")
    if FAIL:
        print("실패:", ", ".join(FAIL))
    shutil.rmtree(_TMP, ignore_errors=True)
    sys.exit(1 if FAIL else 0)
