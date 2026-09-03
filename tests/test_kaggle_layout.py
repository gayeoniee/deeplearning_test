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


def make_prepared_flat(root: Path, n: int = 6) -> Path:
    """`crops/` 층이 **없는** 캐글 데이터셋 — `<데이터셋>/m2.5/...` 입니다.

    `data/work/crops/m2.5` 를 통째로 올리면 이 모양이 됩니다.
    """
    src = make_prepared(root / "_staging", n=n)
    root.mkdir(parents=True, exist_ok=True)
    for tag in ("m1.5", "full"):
        (src / "crops" / tag).rename(root / tag)
    (src / "manifests").rename(root / "manifests")
    shutil.rmtree(src, ignore_errors=True)
    return root


def test_dataset_without_a_crops_folder_is_found():
    """캐글 업로드에서 `crops/` 층이 빠져도 찾아야 합니다.

    실제로 여기서 막혔습니다 — 데이터셋을 붙여놓고도
    "전처리 결과(dogskin_prepared)를 찾지 못했습니다" 로 죽었습니다.
    """
    from src import env

    base = _TMP / "flat_input"
    make_prepared_flat(base / "dogskin-m25-step16")
    w = fresh_env("flat")
    orig = env._search_roots
    env._search_roots = lambda: [base]
    try:
        env.load_prepared(dest=w)
    finally:
        env._search_roots = orig

    tags = sorted(p.name for p in (w / "crops").iterdir() if p.is_dir())
    check("crops/ 층 없는 데이터셋도 붙는다", tags == ["full", "m1.5"], f"{tags}")
    check("매니페스트도 온다", (w / "manifests" / "manifest_final.parquet").exists())
    n = sum(sum(1 for _ in (w / "crops" / t).rglob("*.jpg")) for t in tags)
    check("크롭이 다 보인다 (12장)", n == 12, f"{n}장")


def test_manifests_is_not_linked_as_a_crop_tag():
    """`crops/` 층이 없으면 `manifests/` 가 형제로 나란히 있습니다.

    이름으로 걸러내지 않으면 매니페스트가 **크롭 태그**로 링크되어
    `crops/manifests` 가 생기고, 태그 수 검사가 조용히 틀립니다.
    """
    from src import env

    base = _TMP / "flat_input2"
    src = make_prepared_flat(base / "ds")
    w = fresh_env("flat2")
    env.load_prepared(src, dest=w)
    tags = sorted(p.name for p in (w / "crops").iterdir() if p.is_dir())
    check("manifests 가 크롭 태그로 안 들어온다", "manifests" not in tags, f"{tags}")


def test_a_plain_folder_is_not_mistaken_for_crops():
    """아무 폴더나 태그로 받으면 남의 데이터셋이 크롭으로 잡힙니다."""
    from src import env

    base = _TMP / "not_crops"
    (base / "ds" / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(
        base / "ds" / "images" / "x.jpg")
    check("모르는 이름의 폴더는 크롭이 아니다", env._crops_dir(base / "ds") is None)


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


def test_empty_crop_dir_does_not_shadow_the_real_one():
    """노트북 출력을 데이터셋으로 만들면 crops/ 가 빈 폴더로 남을 수 있습니다.

    이름 순서상 그게 먼저 걸리면 진짜 크롭 데이터셋을 가로막고
    "크롭이 0% 밖에 없습니다" 로 죽습니다 — 실제로 겪은 함정의 이웃 사례입니다.
    """
    from src import env

    base = _TMP / "shadow"
    # 03 출력 데이터셋: 이름이 앞서고, crops/m2.5 가 비어 있음
    (base / "03notebook" / "crops" / "m1.5").mkdir(parents=True, exist_ok=True)
    (base / "03notebook" / "manifests").mkdir(parents=True, exist_ok=True)
    # 진짜 크롭 데이터셋
    make_prepared(base / "dogskin-m25", n=4)
    w = fresh_env("shadow")
    orig = env._search_roots
    env._search_roots = lambda: [base]
    try:
        env.load_prepared(dest=w)
    finally:
        env._search_roots = orig
    link = w / "crops" / "m1.5"
    n = sum(1 for _ in link.rglob("*.jpg")) if link.exists() else 0
    check("빈 폴더가 진짜 크롭을 가로막지 않는다", n > 0,
          f"m1.5 에서 {n}장 — 빈 03notebook/crops/m1.5 가 이겼습니다")


# ──────────────────────────────────────────────────────────────
# 백본은 체크포인트 **이름**에서 읽는다
#
# ⚠️ 실제로 당한 버그입니다. 05 가 "03 의 체크포인트는 항상 resnet50" 으로
#    하드코딩돼 있었는데 STEP 6 에서 1단계를 effnetv2_s 로 바꾸면서 깨졌습니다.
#    effnetv2_s 가중치를 resnet50 껍데기에 부으려다 shape mismatch 로 죽었고,
#    그때는 이미 11분(크롭 세기)을 쓴 뒤였습니다.
# ──────────────────────────────────────────────────────────────
def test_backbone_key_is_read_from_the_exp_name():
    from src import train

    cases = {
        # ★ 이 이름이 우리를 죽였습니다
        "stage1_effnetv2_s_full_384_moderate_photometric": "effnetv2_s",
        "stage2_resnet50_m2.5_384_moderate": "resnet50",
        "stage1_resnet50_full_384_moderate": "resnet50",
        "stage2_convnextv2_base_m2.5_384_moderate": "convnextv2_base",
        "s1_convnextv2_base": "convnextv2_base",      # 04 형식
        "s2_effnetv2_s": "effnetv2_s",
        "f320 같은 고정 픽셀": None,
    }
    for name, want in cases.items():
        got = train.model_key_from_exp(name)
        check(f"{name[:44]:<44} → {want}", got == want, f"got {got}")

    # 고정 픽셀 크롭도 크롭 토큰으로 인식해야 합니다
    check("f320 크롭도 잘라낸다",
          train.model_key_from_exp("stage1_effnetv2_s_f320_384_moderate") == "effnetv2_s")


def test_every_parsed_key_exists_in_the_model_zoo():
    """이름에서 읽은 키가 MODEL_ZOO 에 없으면 05 가 KeyError 로 죽습니다."""
    from src.config import MODEL_BY_KEY
    from src import train

    for key in MODEL_BY_KEY:
        for stage, crop in ((1, "full"), (2, "m2.5")):
            name = f"stage{stage}_{key}_{crop}_384_moderate"
            got = train.model_key_from_exp(name)
            check(f"{key} 왕복", got == key, f"{name} → {got}")


def test_backbone_mismatch_says_what_is_wrong():
    """모양이 안 맞으면 원본 RuntimeError 대신 원인을 짚어야 합니다."""
    import torch

    from src import models

    ck = _TMP / "mismatch.pt"
    # resnet 은 bn1 이 64채널입니다. 24채널(effnetv2 stem)을 넣어 같은 상황을 만듭니다.
    torch.save({"model": {"bn1.weight": torch.zeros(24)}}, ck)
    try:
        models.load_checkpoint(str(ck), "resnet18", 2, device="cpu")
    except RuntimeError as exc:
        msg = str(exc)
        check("백본이 다르다고 말한다", "백본이 다릅니다" in msg, msg[:200])
        check("어느 파일인지 알려준다", "mismatch.pt" in msg, msg[:200])
        check("이름에서 읽는 법을 알려준다", "model_key_from_exp" in msg, msg[:200])
    else:
        raise AssertionError("모양이 안 맞는데 그냥 통과했습니다")


def test_notebook05_does_not_hardcode_resnet50():
    import json as _json

    nb = _json.loads((ROOT / "notebooks" / "05_평가_보정_GradCAM.ipynb")
                     .read_text(encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    check("resnet50 하드코딩이 없다", 'MODEL_BY_KEY["resnet50"]' not in src)
    check("이름에서 백본을 읽는다", "model_key_from_exp(name1)" in src)
    check("두 단계를 따로 읽는다", "model_key_from_exp(name2)" in src)


def test_release_zip_keeps_the_folder_structure():
    """Kaggle 은 Output 에서 **폴더 하나만** 못 빼냅니다.

    [New Dataset] 은 출력 전체를 가져가고, 개별 파일 다운로드는 구조가 깨져
    checkpoints/<실험>/best.pt 배치가 사라집니다 → import 가 못 찾습니다.
    zip 하나면 그것만 받아 올릴 수 있고 Kaggle 이 풀 때 구조가 살아납니다.
    """
    import zipfile

    from src import env, train

    fresh_env("relzip")
    d = train.ckpt_dir("stage1_effnetv2_s_full_384_moderate")
    (d / "best.pt").write_bytes(b"w" * 2048)
    rel = train.export_release(
        ["stage1_effnetv2_s_full_384_moderate"],
        meta={"1단계 크롭": "full"},
        files={"stage1_threshold.json": {"threshold": 0.2706}}, verbose=False)
    z = rel.parent / "release.zip"
    check("release.zip 이 만들어진다", z.is_file())
    names = zipfile.ZipFile(z).namelist() if z.is_file() else []
    check("가중치가 실험 폴더 안에 그대로",
          "checkpoints/stage1_effnetv2_s_full_384_moderate/best.pt" in names, f"{names}")
    check("임계값 JSON 도 들어간다", "stage1_threshold.json" in names, f"{names}")
    check("사람이 볼 확인표도", "READ_ME_FIRST.txt" in names, f"{names}")

    # ★ 풀어서 붙이면 인계가 되는지 — Kaggle 이 업로드 때 하는 일 그대로
    base = _TMP / "relzip_input" / "ds"
    base.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(z).extractall(base)
    w2 = fresh_env("relzip_in")
    orig = env._search_roots
    env._search_roots = lambda: [base.parent]
    try:
        got = train.import_previous_run(verbose=False)
    finally:
        env._search_roots = orig
    check("zip 을 푼 것만으로 가중치가 온다",
          got["checkpoints"] == ["stage1_effnetv2_s_full_384_moderate"], f"{got}")
    check("JSON 도 온다", (w2 / "stage1_threshold.json").is_file())


def test_release_bundle_is_flat_and_self_describing():
    """넘길 것만 최상위 한 폴더에. 안쪽에 두면 데이터셋으로 만들 때 빠집니다."""
    from src import env, train

    w = fresh_env("release")
    d = train.ckpt_dir("stage2_resnet50_m2.5_384_moderate")
    (d / "best.pt").write_bytes(b"weights")
    rel = train.export_release(
        ["stage2_resnet50_m2.5_384_moderate"],
        meta={"2단계 크롭": "m2.5", "2단계 macro-F1": "0.5313"},
        files={"stage1_threshold.json": {"threshold": 0.2127},
               "reports/step4a_summary.json": {"img_size": 384}},
        verbose=False)
    check("가중치가 들어간다",
          (rel / "checkpoints" / "stage2_resnet50_m2.5_384_moderate" / "best.pt").is_file())
    check("임계값 JSON 이 들어간다", (rel / "stage1_threshold.json").is_file())
    check("reports/ 도 들어간다", (rel / "reports" / "step4a_summary.json").is_file())
    txt = (rel / "READ_ME_FIRST.txt").read_text(encoding="utf-8")
    check("사람이 크롭을 눈으로 확인할 수 있다", "m2.5" in txt, txt)
    check("점수도 적혀 있다", "0.5313" in txt, txt)

    # ★ 핵심 — 이 폴더만 붙여도 인계가 된다
    base = _TMP / "release_input"
    base.mkdir(exist_ok=True)
    shutil.copytree(rel, base / "ds" / "release", dirs_exist_ok=True)
    w2 = fresh_env("release_in")
    orig = env._search_roots
    env._search_roots = lambda: [base]
    try:
        got = train.import_previous_run(verbose=False)
    finally:
        env._search_roots = orig
    check("release 폴더만으로 가중치가 온다",
          got["checkpoints"] == ["stage2_resnet50_m2.5_384_moderate"], f"{got}")
    check("release 폴더만으로 JSON 도 온다", (w2 / "stage1_threshold.json").is_file())


def test_release_carries_the_completion_record():
    """안 바꾼 단계를 다시 학습하지 않도록 result.json 도 넘어가야 합니다.

    이게 빠지면 training_state() 가 "완료" 를 못 읽어서, 1단계만 바꾸는 실행에서
    2단계 25에폭(~1.5시간)이 통째로 다시 돕니다. 시간보다 더 나쁜 건 비교입니다 —
    안 바꾼 단계를 재학습하면 가중치가 미묘하게 달라져서, 파이프라인 숫자 변화에
    "바꾼 단계의 효과" 와 "안 바꾼 단계의 잡음" 이 섞입니다.
    """
    import json as _json

    from src import env, train

    exp = "stage2_resnet50_m2.5_384_moderate"
    fresh_env("release_done")
    d = train.ckpt_dir(exp)
    (d / "best.pt").write_bytes(b"weights")
    (d / "result.json").write_text(_json.dumps({
        "completed": True, "best_score": 0.5457, "best_epoch": 20,
        "target_epochs": 25, "early_stopped": True,
        "history": [{"epoch": i} for i in range(25)],
    }), encoding="utf-8")

    rel = train.export_release([exp], meta={"2단계 크롭": "m2.5"}, verbose=False)
    check("완료 기록이 꾸러미에 들어간다",
          (rel / "checkpoints" / exp / "result.json").is_file())

    # ★ 핵심 — 붙여 넣은 쪽에서 "이미 끝났다" 로 읽혀야 합니다
    base = _TMP / "release_done_input"
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True)
    shutil.copytree(rel, base / "ds" / "release", dirs_exist_ok=True)
    fresh_env("release_done_in")
    orig = env._search_roots
    env._search_roots = lambda: [base]
    try:
        train.import_previous_run(verbose=False)
    finally:
        env._search_roots = orig

    st = train.training_state(exp, check_persist=False)
    check("다음 실행이 '이미 끝남' 으로 읽는다", st["completed"] and st["has_best"], f"{st}")
    check("목표 에폭도 살아 있다", st["target_epochs"] == 25, f"{st}")
    check("조기 종료 사실도 살아 있다", st["early_stopped"], f"{st}")


def test_missing_completion_record_is_announced():
    """result.json 이 없으면 조용히 넘어가지 말고 알려야 합니다."""
    import contextlib
    import io

    from src import train

    exp = "stage2_resnet50_m2.5_384_moderate"
    fresh_env("release_norec")
    (train.ckpt_dir(exp) / "best.pt").write_bytes(b"weights")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        train.export_release([exp], verbose=False)
    out = buf.getvalue()
    check("완료 기록이 없으면 경고한다", "result.json" in out and "다시 학습" in out, out)


def test_notebook03_exports_a_release():
    import json as _json

    nb = _json.loads((ROOT / "notebooks" / "03_학습_베이스라인.ipynb")
                     .read_text(encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    check("03 이 인계 꾸러미를 만든다", "train.export_release(" in src)


def test_notebook05_validates_before_the_slow_part():
    """크롭 세는 데 4분 걸립니다. 잘못된 입력이면 그 전에 멈춰야 합니다."""
    import json as _json

    nb = _json.loads((ROOT / "notebooks" / "05_평가_보정_GradCAM.ipynb")
                     .read_text(encoding="utf-8"))
    for c in nb["cells"]:
        s = "".join(c["source"])
        if "load_prepared()" in s and "import_previous_run" in s:
            check("인계 검증이 load_prepared 보다 먼저",
                  s.index("import_previous_run") < s.index("load_prepared()"), s[:200])
            check("크롭 불일치를 여기서 잡는다", "ADOPTED_STAGE2_CROP" in s)
            return
    check("검증 셀을 찾았다", False, "load_prepared 와 import 가 같은 셀에 없습니다")


def test_settings_recovered_from_checkpoint_names():
    """JSON 이 안 넘어와도 크롭·실험 이름은 폴더 이름에서 살릴 수 있습니다."""
    from src import train

    w = fresh_env("infer")
    for n in ("stage1_resnet50_full_moderate", "stage2_resnet50_m2.5_moderate"):
        d = w / "checkpoints" / n
        d.mkdir(parents=True, exist_ok=True)
        (d / "best.pt").write_bytes(b"w")
    got = train.infer_run_settings()
    check("1단계 크롭 full", got.get("stage1_crop") == "full", f"{got}")
    check("2단계 크롭 m2.5", got.get("stage2_crop") == "m2.5", f"{got}")
    check("실험 이름도 살아난다",
          got.get("stage1_exp") == "stage1_resnet50_full_moderate", f"{got}")
    check("모델 이름도 살아난다", got.get("stage2_model") == "resnet50", f"{got}")


def test_settings_ignore_incomplete_checkpoints():
    """best.pt 가 없는 폴더는 설정 근거가 못 됩니다."""
    from src import train

    w = fresh_env("infer2")
    (w / "checkpoints" / "stage1_resnet50_m1.5_moderate").mkdir(parents=True, exist_ok=True)
    check("미완성 체크포인트는 무시", train.infer_run_settings() == {})


def test_notebook05_recovers_threshold_instead_of_defaulting():
    """임계값을 0.5 로 때우면 파이프라인 평가가 조용히 틀립니다."""
    import json as _json

    nb = _json.loads((ROOT / "notebooks" / "05_평가_보정_GradCAM.ipynb")
                     .read_text(encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    check("체크포인트 이름에서 설정을 되살린다", "infer_run_settings()" in src)
    check("임계값을 다시 계산한다", "if THR1 is None:" in src and "binary_report" in src)
    check("0.5 로 때우지 않는다", "THR1 = 0.5" not in src)


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


def test_run_files_come_over_too():
    """가중치만 가져오면 05 는 여전히 죽습니다 — 임계값 JSON 도 넘어와야 합니다."""
    from src import env, train

    base = _TMP / "run_out"
    src = base / "dogskin-03-output"
    _fake_ckpt(src, "stage1_resnet50_full_moderate")
    (src / "stage1_threshold.json").write_text('{"threshold": 0.2127}')
    (src / "reports").mkdir(parents=True, exist_ok=True)
    (src / "reports" / "step4a_summary.json").write_text('{"img_size": 384}')
    w = fresh_env("runfiles")
    orig = env._search_roots
    env._search_roots = lambda: [base]
    try:
        got = train.import_previous_run(verbose=False)
    finally:
        env._search_roots = orig
    check("임계값 JSON 이 넘어온다", (w / "stage1_threshold.json").is_file(),
          f"{sorted(p.name for p in w.iterdir())}")
    check("reports/ 도 넘어온다", (w / "reports" / "step4a_summary.json").is_file())
    check("가중치도 같이 온다", got["checkpoints"] == ["stage1_resnet50_full_moderate"],
          f"{got}")


def test_this_sessions_files_are_not_overwritten():
    """방금 만든 결과를 예전 입력이 덮으면 낡은 임계값으로 평가하고도 모릅니다."""
    from src import env, train

    base = _TMP / "run_stale"
    src = base / "old-output"
    src.mkdir(parents=True, exist_ok=True)
    (src / "stage1_threshold.json").write_text('{"threshold": 0.9}')
    w = fresh_env("stale")
    (w / "stage1_threshold.json").write_text('{"threshold": 0.2127}')   # 이번 세션 것
    orig = env._search_roots
    env._search_roots = lambda: [base]
    try:
        train.import_previous_run(verbose=False)
    finally:
        env._search_roots = orig
    check("이번 세션 파일이 살아남는다",
          "0.2127" in (w / "stage1_threshold.json").read_text())


def test_deep_notebook_output_layout_is_found():
    """Kaggle 노트북 출력은 한두 단계 더 들어가 있을 수 있습니다."""
    from src import env, train

    base = _TMP / "run_deep"
    _fake_ckpt(base / "nb" / "data" / "work", "stage2_resnet50_m2.5_moderate")
    fresh_env("deepnb")
    orig = env._search_roots
    env._search_roots = lambda: [base]
    try:
        found = train.find_checkpoint_sources()
    finally:
        env._search_roots = orig
    check("깊이 3 의 checkpoints/ 도 찾는다", len(found) == 1, f"{found}")


def test_notebook05_actually_calls_the_import():
    """코드만 있고 노트북이 안 부르면 소용없습니다."""
    import json as _json

    nb = _json.loads((ROOT / "notebooks" / "05_평가_보정_GradCAM.ipynb")
                     .read_text(encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    check("05 가 import_previous_run 을 부른다", "train.import_previous_run(" in src)
    check("자동 탐색 실패용 수동 경로 자리가 있다", "PREV_RUN" in src)
    check("실패하면 진단을 찍는다", "explain_handoff()" in src)



# ─────────────────────────────────────────────────────────────────────────────
# rebase_paths 의 경고 — 정상 흐름에서 겁주지 않기
#
# 05 로그에서 "⚠️ 크롭 파일 45,885/45,885개를 찾을 수 없습니다" 가 뜬 3초 뒤에
# 같은 파일들이 switch_tag 로 100.0% 존재 확인됐습니다. 매니페스트가 로컬에서 쓴
# 태그를 가리키는데 클라우드에는 다른 태그가 붙어 있어서 생긴 **정상 상황**입니다.
# ─────────────────────────────────────────────────────────────────────────────

def _rebase_output(tmp, crop_rel_tag, linked_tags, n=3):
    """rebase_paths 를 돌리고 찍힌 문구를 돌려줍니다."""
    import io, contextlib
    import pandas as pd
    from src import env, labels

    work = tmp / "work"
    (work / "crops").mkdir(parents=True, exist_ok=True)
    for t in linked_tags:
        (work / "crops" / t).mkdir(exist_ok=True)
    old = os.environ.get("DOG_SKIN_WORK")
    os.environ["DOG_SKIN_WORK"] = str(work)
    try:
        env.work_root.cache_clear() if hasattr(env.work_root, "cache_clear") else None
        df = pd.DataFrame({"crop_rel": [f"{crop_rel_tag}/ab/f{i}.jpg" for i in range(n)]})
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            labels.rebase_paths(df)
        return buf.getvalue()
    finally:
        if old is None:
            os.environ.pop("DOG_SKIN_WORK", None)
        else:
            os.environ["DOG_SKIN_WORK"] = old
        env.work_root.cache_clear() if hasattr(env.work_root, "cache_clear") else None


def test_tag_mismatch_is_informational_not_a_warning():
    """매니페스트 태그가 안 붙어 있으면 안내만 — ⚠️ 를 찍지 않습니다."""
    with tempfile.TemporaryDirectory() as d:
        out = _rebase_output(Path(d), crop_rel_tag="m1.5", linked_tags=["m2.5", "full"])
    check("태그 불일치는 경고가 아니라 안내", "⚠️" not in out and "[labels]" in out)
    check("어떤 태그를 찾는지 알려준다", "m1.5" in out)
    check("붙어 있는 태그를 알려준다", "m2.5" in out and "full" in out)
    check("전환하면 된다고 알려준다", "switch_tag" in out)


def test_real_missing_files_still_warn():
    """태그는 맞는데 파일이 없으면 진짜 경고입니다."""
    with tempfile.TemporaryDirectory() as d:
        out = _rebase_output(Path(d), crop_rel_tag="m2.5", linked_tags=["m2.5"])
    check("태그가 맞는데 파일이 없으면 ⚠️", "⚠️" in out)
    check("몇 장 중 몇 장인지 헷갈리지 않게", "3장 중 3장이 없습니다" in out)


def test_warning_never_reads_as_all_found():
    """'45,885/45,885개를 찾을 수 없습니다' 같은 애매한 문구를 쓰지 않습니다."""
    src = (ROOT / "src" / "labels.py").read_text(encoding="utf-8")
    check("애매한 X/Y 표기를 안 쓴다", "개를 찾을 수 없습니다" not in src)


# ──────────────────────────────────────────────────────────────
# 빈 폴더를 "이미 준비됨" 으로 보지 않는가
#
# ⚠️ 런팟에서 당했습니다. 노트북 첫 셀의 env.describe() 가 ensure_dirs() 를
#    불러 crops/ 와 manifests/ 를 **빈 폴더로** 만들어 둡니다. 그다음
#    load_prepared() 가 폴더 존재만 보고 "이미 준비돼 있습니다" 를 찍고
#    zip 을 안 풀었고, 몇 셀 뒤에 manifest_final.parquet 을 못 찾아 죽었습니다.
# ──────────────────────────────────────────────────────────────
def test_empty_dirs_are_not_prepared():
    import tempfile

    from src import env

    print("\n[준비 판정] 빈 폴더를 준비됐다고 하지 않는가")
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        (d / "crops").mkdir()
        (d / "manifests").mkdir()
        check("빈 crops+manifests 는 준비 안 됨", not env._looks_prepared(d))
        check("빈 폴더는 '일부만' 도 아님", not env._looks_partial(d))

        (d / "crops" / "f320" / "ab").mkdir(parents=True)
        (d / "crops" / "f320" / "ab" / "x.jpg").write_bytes(b"\xff\xd8")
        check("크롭만 있으면 '일부만'", env._looks_partial(d))
        check("크롭만으로는 준비 안 됨", not env._looks_prepared(d))

        (d / "manifests" / "manifest_final.parquet").write_bytes(b"PAR1")
        check("둘 다 내용이 있으면 준비됨", env._looks_prepared(d))


def test_data_already_in_work_root_is_used_as_is():
    """★ 런팟처럼 데이터를 work_root 에 **직접** 넣은 경우.

    실제로 당한 것 (2026-08-28): 크롭도 매니페스트도 제자리에 있는데
    `load_prepared()` 가 "전처리 결과를 찾지 못했습니다" 로 죽었습니다.
    `find_prepared_all()` 이 dest **자신을 후보에서 빼기** 때문인데,
    "이미 준비돼 있으면 그냥 쓴다" 검사가 그 **뒤에** 있어서 도달조차
    못 했습니다. 순서가 곧 버그였습니다.
    """
    from src import env

    print("\n[제자리] work_root 에 이미 있으면 그대로 쓰는가")
    w = fresh_env("inplace")
    for tag in ("f320", "m2.5"):
        (w / "crops" / tag / "ab").mkdir(parents=True)
        (w / "crops" / tag / "ab" / "x.jpg").write_bytes(b"\xff\xd8")
    (w / "manifests").mkdir(parents=True, exist_ok=True)
    (w / "manifests" / "manifest_final.parquet").write_bytes(b"PAR1")

    orig = env._search_roots
    env._search_roots = lambda: [w.parent]      # 다른 후보는 하나도 없는 상황
    try:
        got = env.load_prepared(dest=w)
        check("죽지 않고 work_root 를 돌려준다", got == w, str(got))
    except FileNotFoundError as exc:
        check("죽지 않고 work_root 를 돌려준다", False, str(exc).splitlines()[0])
    finally:
        env._search_roots = orig

    check("크롭을 건드리지 않는다",
          sorted(q.name for q in (w / "crops").iterdir()) == ["f320", "m2.5"])


if __name__ == "__main__":
    print(f"작업 폴더: {_TMP}\n")
    for fn in [test_empty_dirs_are_not_prepared,
               test_data_already_in_work_root_is_used_as_is, test_finds_extracted_dir, test_readonly_source_is_linked_not_copied,
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
               test_empty_crop_dir_does_not_shadow_the_real_one,
               test_backbone_key_is_read_from_the_exp_name,
               test_every_parsed_key_exists_in_the_model_zoo,
               test_backbone_mismatch_says_what_is_wrong,
               test_notebook05_does_not_hardcode_resnet50,
               test_release_bundle_is_flat_and_self_describing,
               test_release_zip_keeps_the_folder_structure,
               test_notebook03_exports_a_release,
               test_notebook05_validates_before_the_slow_part,
               test_settings_recovered_from_checkpoint_names,
               test_settings_ignore_incomplete_checkpoints,
               test_notebook05_recovers_threshold_instead_of_defaulting,
               test_run_files_come_over_too,
               test_this_sessions_files_are_not_overwritten,
               test_deep_notebook_output_layout_is_found,
               test_notebook05_actually_calls_the_import,
               test_tag_mismatch_is_informational_not_a_warning,
               test_real_missing_files_still_warn,
               test_warning_never_reads_as_all_found,
               test_release_carries_the_completion_record,
               test_missing_completion_record_is_announced,
               test_dataset_without_a_crops_folder_is_found,
               test_manifests_is_not_linked_as_a_crop_tag,
               test_a_plain_folder_is_not_mistaken_for_crops]:
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
