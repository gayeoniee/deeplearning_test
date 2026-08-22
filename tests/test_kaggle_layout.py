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


def test_finds_nested_dataset():
    """Kaggle 데이터셋 안에 폴더가 한 겹 더 있어도 찾아야 합니다.

    zip 을 만들 때 폴더째 담으면 /kaggle/input/<ds>/dogskin_min/crops/ 처럼
    한 단계 깊어집니다. 한 겹만 보면 못 찾습니다.
    """
    from src import env

    base = _TMP / "nested_input"
    make_prepared(base / "my-dataset" / "dogskin_min")
    w = fresh_env("nested")
    orig = env._search_roots
    env._search_roots = lambda: [base]
    try:
        found = env.find_prepared_all(dest=w)
        check("한 겹 더 깊어도 찾는다", len(found) == 1, f"{[str(p) for p, _ in found]}")
    finally:
        env._search_roots = orig


def test_finds_kaggle_deep_layout():
    """★ 실측한 Kaggle 경로: /kaggle/input/datasets/<사용자>/<데이터셋>/crops

    `/kaggle/input` 기준 4단계입니다. 얕게 훑으면 못 찾습니다.
    """
    from src import env

    base = _TMP / "deep_input"
    make_prepared(base / "datasets" / "gayoniee" / "dogskin-m15")
    w = fresh_env("deep")
    orig = env._search_roots
    env._search_roots = lambda: [base]
    try:
        found = env.find_prepared_all(dest=w)
        check("Kaggle 의 깊은 경로를 찾는다", len(found) == 1,
              f"{[str(p) for p, _ in found]}")
        env.load_prepared(dest=w)
    finally:
        env._search_roots = orig
    tags = sorted(p.name for p in (w / "crops").iterdir() if p.is_dir())
    check("깊은 경로에서도 크롭이 연결된다", tags == ["full", "m1.5"], f"{tags}")


def test_walk_does_not_descend_into_crops():
    """크롭 폴더(4만 장) 안으로 내려가면 탐색이 하염없이 느려집니다."""
    from src import env

    base = _TMP / "walkperf"
    d = base / "ds"
    make_prepared(d)
    for i in range(30):                       # 크롭 안의 하위 폴더를 흉내
        (d / "crops" / "m1.5" / f"sub{i}").mkdir(parents=True, exist_ok=True)
    visited = env._walk(base)
    inside = [p for p in visited if "sub" in p.name]
    check("크롭 내부로는 안 내려간다", not inside, f"{len(inside)}개 들어감")
    check("전처리 폴더 자체는 방문한다", any(p.name == "ds" for p in visited))


def test_finds_zip_inside_dataset_folder():
    """Kaggle 이 zip 을 안 풀었을 때 — 데이터셋 폴더 안의 zip 도 찾아야 합니다."""
    from src import env

    base = _TMP / "zip_in_ds"
    d = base / "my-dataset"
    d.mkdir(parents=True)
    src = make_prepared(_TMP / "zipsrc2")
    with zipfile.ZipFile(d / "dogskin_m15.zip", "w") as zf:
        for p in src.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(src))
    w = fresh_env("zipds")
    orig = env._search_roots
    env._search_roots = lambda: [base]
    try:
        found = env.find_prepared_all(dest=w)
        check("데이터셋 폴더 안의 zip 을 찾는다",
              len(found) == 1 and found[0][1] == "zip", f"{found}")
    finally:
        env._search_roots = orig


def test_error_lists_actual_contents():
    """★ '못 찾았다' 만으로는 다음에 뭘 볼지 알 수 없습니다."""
    from src import env

    base = _TMP / "empty_input"
    (base / "관계없는-데이터셋").mkdir(parents=True)
    (base / "관계없는-데이터셋" / "readme.txt").write_text("x", encoding="utf-8")
    w = fresh_env("err")
    orig = env._search_roots
    env._search_roots = lambda: [base]
    try:
        env.find_prepared_all(dest=w)
        check("없으면 예외", False, "예외가 안 났습니다")
    except FileNotFoundError as exc:
        m = str(exc)
        check("오류가 실제 내용물을 보여준다", "관계없는-데이터셋" in m, m[:300])
        check("오류가 무엇을 찾는지 알려준다", "crops/" in m and "dogskin*.zip" in m)
    finally:
        env._search_roots = orig


def test_kaggle_wins_over_colab_signals():
    """★ Kaggle 이미지에도 google.colab 과 /content 가 있습니다.

    실제로 겪은 것: Kaggle 세션이 colab 으로 판정되어 노트북이 drive.mount() 를
    부르고 NotImplementedError 로 죽었습니다. Colab 신호가 다 있어도
    **Kaggle 신호가 있으면 kaggle** 이어야 합니다.
    """
    import types

    from src import env

    saved_env = {k: os.environ.get(k) for k in
                 ("KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE", "COLAB_RELEASE_TAG")}
    saved_mod = sys.modules.get("google.colab")
    orig_isdir = env.Path.is_dir
    try:
        os.environ["KAGGLE_KERNEL_RUN_TYPE"] = "Interactive"
        sys.modules["google.colab"] = types.ModuleType("google.colab")   # Colab 패키지 존재
        check("Kaggle 신호가 Colab 신호를 이긴다", env.detect() == "kaggle",
              f"{env.detect()}")
        check("Kaggle 에서는 마운트를 시도하지 않는다", env.can_mount_drive() is False)

        # 반대: Kaggle 신호가 하나도 없고 진짜 Colab 표식만 있으면 colab
        del os.environ["KAGGLE_KERNEL_RUN_TYPE"]
        os.environ["COLAB_RELEASE_TAG"] = "release-colab-20260101"
        if not Path("/kaggle/working").is_dir() and not Path("/kaggle/input").is_dir():
            check("진짜 Colab 표식이면 colab", env.detect() == "colab", f"{env.detect()}")
        else:
            check("진짜 Colab 표식이면 colab", True, "(이 머신에 /kaggle 이 있어 생략)")
    finally:
        env.Path.is_dir = orig_isdir
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if saved_mod is None:
            sys.modules.pop("google.colab", None)
        else:
            sys.modules["google.colab"] = saved_mod


def test_diagnose_reports_signals():
    from src import env

    sig = env.diagnose()
    for k in ("detect()", "/content 있음", "/var/colab/hostname 있음", "can_mount_drive()"):
        if k not in sig:
            check("diagnose 가 판정 근거를 다 찍는다", False, f"'{k}' 없음")
            return
    check("diagnose 가 판정 근거를 다 찍는다", True)


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


# ──────────────────────────────────────────────────────────────
# 노트북 사이의 체크포인트 인계
#
# Kaggle 은 **노트북마다 세션이 따로**입니다. 03 이 3시간 걸려 만든 가중치는
# 그 세션의 /kaggle/working 에 있고, 05 를 열면 이미 없습니다.
# 03 을 'Save & Run All (Commit)' 로 돌린 뒤 그 출력을 05 에 붙이는 게 정답인데,
# 경로를 사람이 외우게 하면 결국 3시간을 다시 씁니다. 그래서 자동으로 찾습니다.
# ──────────────────────────────────────────────────────────────
def _fake_ckpt(root: Path, *exps: str) -> Path:
    for e in exps:
        d = root / "checkpoints" / e
        d.mkdir(parents=True, exist_ok=True)
        (d / "best.pt").write_bytes(b"weights")
        (d / "state.json").write_text('{"epochs_done": 25, "completed": true}')
    return root


def test_finds_checkpoints_in_notebook_output():
    """05 에 붙인 03 의 노트북 출력에서 체크포인트를 찾아야 합니다."""
    from src import env, train

    base = _TMP / "ckpt_input"
    _fake_ckpt(base / "dogskin-03-output",
               "stage1_resnet50_m2.5_moderate", "stage2_resnet50_m2.5_moderate")
    fresh_env("ckpt1")
    orig = env._search_roots
    env._search_roots = lambda: [base]
    try:
        found = train.find_checkpoint_sources()
        check("노트북 출력 안의 checkpoints/ 를 찾는다", len(found) == 1,
              f"{[str(p) for p in found]}")
        got = sorted(train.import_checkpoints(verbose=False))
    finally:
        env._search_roots = orig
    check("두 실험을 모두 가져온다",
          got == ["stage1_resnet50_m2.5_moderate", "stage2_resnet50_m2.5_moderate"], f"{got}")
    check("가중치가 실제로 놓인다",
          (train.ckpt_dir("stage1_resnet50_m2.5_moderate") / "best.pt").exists())


def test_checkpoint_import_is_quiet_when_nothing_attached():
    """같은 세션에서 방금 학습했으면 가져올 게 없는 게 정상 — 죽으면 안 됩니다."""
    from src import env, train

    empty = _TMP / "ckpt_empty"
    empty.mkdir(exist_ok=True)
    fresh_env("ckpt2")
    orig = env._search_roots
    env._search_roots = lambda: [empty]
    try:
        check("빈 입력이면 조용히 빈 목록", train.import_checkpoints(verbose=False) == [])
    finally:
        env._search_roots = orig


def test_checkpoint_search_ignores_half_written_dirs():
    """best.pt 가 없는 폴더는 후보가 아닙니다 (학습이 중간에 끊긴 출력)."""
    from src import env, train

    base = _TMP / "ckpt_partial"
    d = base / "half" / "checkpoints" / "stage1_resnet50_m2.5_moderate"
    d.mkdir(parents=True, exist_ok=True)
    (d / "last.pt").write_bytes(b"partial")        # best.pt 는 없음
    fresh_env("ckpt3")
    orig = env._search_roots
    env._search_roots = lambda: [base]
    try:
        check("best.pt 없는 폴더는 무시한다", train.find_checkpoint_sources() == [])
    finally:
        env._search_roots = orig


def test_notebook05_actually_calls_the_import():
    """코드만 있고 노트북이 안 부르면 소용없습니다."""
    import json as _json

    nb = _json.loads((ROOT / "notebooks" / "05_평가_보정_GradCAM.ipynb")
                     .read_text(encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    check("05 가 import_checkpoints() 를 부른다", "train.import_checkpoints()" in src)



if __name__ == "__main__":
    print(f"작업 폴더: {_TMP}\n")
    for fn in [test_finds_extracted_dir, test_readonly_source_is_linked_not_copied,
               test_link_is_idempotent, test_zip_path_still_works,
               test_autodetect_prefers_extracted_when_no_zip,
               test_missing_gives_useful_error, test_split_upload_is_merged,
               test_partial_dir_without_manifests_is_accepted,
               test_finds_nested_dataset, test_finds_kaggle_deep_layout,
               test_walk_does_not_descend_into_crops, test_finds_zip_inside_dataset_folder,
               test_error_lists_actual_contents,
               test_kaggle_wins_over_colab_signals, test_diagnose_reports_signals,
               test_manifest_rebase_works_through_link,
               test_finds_checkpoints_in_notebook_output,
               test_checkpoint_import_is_quiet_when_nothing_attached,
               test_checkpoint_search_ignores_half_written_dirs,
               test_notebook05_actually_calls_the_import]:
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
