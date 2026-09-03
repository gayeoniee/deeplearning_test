"""다른 PC 에서 온 청크를 합치는 길 — 조용히 틀리는 두 곳을 못 박습니다.

    uv run python tests/test_merge_incoming.py

여기서 잡으려는 것은 **에러 없이 잘못되는** 두 가지입니다.

1. `dedup.run` 이 원본을 못 읽는 청크를 만나도 그냥 넘어감
   → 그 청크는 '중복 없음' 이 되고 청크 경계를 넘는 누수를 못 막습니다.
   `compute_hashes` 는 **한 장도** 못 읽었을 때만 에러를 내므로,
   VL01 이 읽히는 한 아무 말도 안 나옵니다.

2. 크롭만 옮기고 매니페스트를 빠뜨림 / 크롭 설정이 다른 파일을 덮어씀
"""

from __future__ import annotations

import io
import os
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
    """표준출력을 삼키고 (결과, 출력) 을 돌려줍니다."""
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*a, **kw)
    return out, buf.getvalue()


# ──────────────────────────────────────────────────────────────
# 1. dedup — 읽을 수 없는 청크를 조용히 넘기지 않는가
# ──────────────────────────────────────────────────────────────
def test_dedup_reuses_phash_and_refuses_silent_misses():
    print("\n[dedup] 못 읽는 청크를 조용히 넘기지 않는가")
    import pandas as pd

    from src import dedup
    from src.config import CFG

    # VL01 은 phash 가 없어서 읽어야 하고, TL02 는 phash 를 들고 왔습니다.
    rows = []
    for i in range(50):
        rows.append({"image_path": f"/vl01/{i}.jpg", "chunk": "chunk_VL01",
                     "label": "A1", "animal_id": f"v{i}", "phash": None,
                     "area_ratio": 0.05})
    for i in range(50):
        rows.append({"image_path": f"/tl02/{i}.jpg", "chunk": "chunk_TL02",
                     "label": "A1", "animal_id": f"t{i}",
                     "phash": f"{i:016x}", "area_ratio": 0.05})
    df = pd.DataFrame(rows)

    # 실제로 열리는 건 VL01 뿐인 상황을 만듭니다.
    calls: list[int] = []

    def fake_hashes(rows_, hash_size=8, workers=4, path_col="image_path"):
        paths = list(rows_[path_col]) if hasattr(rows_, "columns") else list(rows_)
        calls.append(len(set(paths)))
        return {p: f"{9000 + i:016x}" for i, p in enumerate(sorted(set(paths)))
                if p.startswith("/vl01/")}

    real = dedup.compute_hashes
    dedup.compute_hashes = fake_hashes
    try:
        cfg = CFG()
        out, _ = _capture(dedup.run, df, cfg, verbose=True)
        check("가져온 phash 는 다시 안 잰다", calls and calls[0] == 50,
              f"다시 잰 장수 {calls[0] if calls else '없음'} (기대 50 = VL01 만)")
        check("전부 읽혔으면 통과한다", len(out[0]) > 0)

        # 이제 TL02 도 phash 가 없는 경우 — 예전 상태입니다.
        df2 = df.copy()
        df2["phash"] = None
        calls.clear()
        raised = ""
        try:
            _capture(dedup.run, df2, cfg, verbose=True)
        except RuntimeError as exc:
            raised = str(exc)
        check("phash 없이 못 읽으면 **멈춘다**", "phash 를 못 읽은" in raised,
              raised.splitlines()[0] if raised else "안 멈췄습니다")
        check("어느 청크인지 말해준다", "chunk_TL02" in raised)

        # 아주 조금(문턱 이하) 빠진 건 경고만 하고 지나갑니다.
        df3 = df.copy()
        df3.loc[df3.index[-1], "phash"] = None      # 100장 중 1장 = 1%
        _, log = _capture(dedup.run, df3, cfg, verbose=True)
        check("문턱 이하는 경고만 하고 진행", "⚠️" in log and "못 읽은" in log)
    finally:
        dedup.compute_hashes = real


# ──────────────────────────────────────────────────────────────
# 2. merge_incoming — 미리보기가 기본이고, 빠뜨린 걸 말하는가
# ──────────────────────────────────────────────────────────────
def _make_zip(path: Path, crops: dict[str, bytes], manifest_df=None,
              manifest_name: str = "chunk_TL02.parquet") -> None:
    with zipfile.ZipFile(path, "w") as z:
        for rel, data in crops.items():
            z.writestr(rel, data)
        if manifest_df is not None:
            buf = io.BytesIO()
            manifest_df.to_parquet(buf, index=False)
            z.writestr(f"manifests/{manifest_name}", buf.getvalue())


def _manifest(with_phash: bool, n: int = 30):
    import pandas as pd

    d = {"image_path": [f"/tl02/{i}.jpg" for i in range(n)],
         "label": ["A1"] * n,
         "animal_id": [f"t{i}" for i in range(n)]}
    if with_phash:
        d["phash"] = [f"{i:016x}" for i in range(n)]
    return pd.DataFrame(d)


def test_merge_preview_and_apply():
    print("\n[merge_incoming] 미리보기 → 적용")
    import importlib

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        work = tmp / "work"
        (work / "crops" / "f320" / "ab").mkdir(parents=True)
        (work / "manifests").mkdir(parents=True)
        # 이 PC 에 이미 있는 크롭 한 장
        (work / "crops" / "f320" / "ab" / "old_1111.jpg").write_bytes(b"AAAA")

        os.environ["DOG_SKIN_WORK"] = str(work)
        import src.env
        importlib.reload(src.env)
        mi = importlib.import_module("tools.merge_incoming")
        importlib.reload(mi)

        zpath = tmp / "tl02.zip"
        _make_zip(zpath, {
            "crops/f320/ab/old_1111.jpg": b"AAAA",           # 이미 있음 (같음)
            "crops/f320/cd/new_2222.jpg": b"BBBBBB",         # 새 것
            "crops/m2.5/cd/new_2222.jpg": b"CCCCCCC",        # 새 것
        }, _manifest(with_phash=True))

        # ── 미리보기는 아무것도 안 씁니다
        _, log = _capture(mi.main, [str(zpath)])
        check("미리보기라고 말한다", "미리보기" in log)
        check("새 크롭 장수를 센다", "새로       2" in log or "새로" in log)
        check("이미 있는 것을 구분한다", "이미 있음" in log)
        check("phash 를 확인해 준다", "phash 30/30" in log or "phash" in log)
        check("미리보기는 파일을 안 만든다",
              not (work / "crops" / "f320" / "cd" / "new_2222.jpg").exists())

        # ── --apply
        _, log = _capture(mi.main, [str(zpath), "--apply"])
        check("새 크롭이 들어왔다",
              (work / "crops" / "f320" / "cd" / "new_2222.jpg").exists())
        check("새 태그 폴더가 생겼다", (work / "crops" / "m2.5").exists())
        check("매니페스트가 같이 들어왔다",
              (work / "manifests" / "chunk_TL02.parquet").exists())
        check("이미 있던 크롭은 그대로",
              (work / "crops" / "f320" / "ab" / "old_1111.jpg").read_bytes() == b"AAAA")
        check("다음 명령을 알려준다", "--finalize" in log)

        del os.environ["DOG_SKIN_WORK"]


def test_merge_stops_on_clash_and_missing_manifest():
    print("\n[merge_incoming] 부딪힘 · 빠진 매니페스트")
    import importlib

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        work = tmp / "work"
        (work / "crops" / "f320" / "ab").mkdir(parents=True)
        (work / "manifests").mkdir(parents=True)
        (work / "crops" / "f320" / "ab" / "x_1111.jpg").write_bytes(b"AAAA")

        os.environ["DOG_SKIN_WORK"] = str(work)
        import src.env
        importlib.reload(src.env)
        mi = importlib.import_module("tools.merge_incoming")
        importlib.reload(mi)

        # 이름은 같은데 내용이 다름 = 크롭 설정이 달랐다는 뜻
        zc = tmp / "clash.zip"
        _make_zip(zc, {"crops/f320/ab/x_1111.jpg": b"DIFFERENT"}, _manifest(True))
        _, log = _capture(mi.main, [str(zc)])
        check("부딪힘을 센다", "부딪힘" in log)
        check("왜 위험한지 말한다", "크롭 설정" in log)

        died = ""
        try:
            _capture(mi.main, [str(zc), "--apply"])
        except SystemExit as exc:
            died = str(exc)
        check("--apply 여도 부딪히면 멈춘다", "멈췄습니다" in died, died[:60])
        check("멈췄으면 안 덮었다",
              (work / "crops" / "f320" / "ab" / "x_1111.jpg").read_bytes() == b"AAAA")

        # 매니페스트 없이 크롭만
        zn = tmp / "nomani.zip"
        _make_zip(zn, {"crops/f320/cd/y_2222.jpg": b"EEEE"}, None)
        _, log = _capture(mi.main, [str(zn)])
        check("매니페스트가 없으면 크게 말한다", "❌ 하나도 없습니다" in log)

        # phash 없는 매니페스트
        zp = tmp / "nophash.zip"
        _make_zip(zp, {"crops/f320/cd/y_2222.jpg": b"EEEE"}, _manifest(False))
        _, log = _capture(mi.main, [str(zp)])
        check("phash 가 없으면 크게 말한다", "phash 컬럼이 없습니다" in log)

        # manifest_final 은 절대 덮지 않습니다
        zf = tmp / "final.zip"
        _make_zip(zf, {"crops/f320/cd/z_3333.jpg": b"FFFF"}, _manifest(True),
                  manifest_name="manifest_final.parquet")
        _capture(mi.main, [str(zf), "--apply"])
        check("manifest_final 은 안 가져온다",
              not (work / "manifests" / "manifest_final.parquet").exists())

        del os.environ["DOG_SKIN_WORK"]


# ──────────────────────────────────────────────────────────────
# 3. prepare_local — 원본을 지우기 **전에** phash 를 재는가
# ──────────────────────────────────────────────────────────────
def test_chunk_hashes_before_deleting_raw():
    print("\n[prepare_local] phash 를 원본 삭제 전에 재는가")
    src = (ROOT / "prepare_local.py").read_text(encoding="utf-8")
    body = src.split("def step_chunk")[1].split("\ndef ")[0]

    i_hash = body.find("compute_hashes")
    i_save = body.find('labels.save(df, f"chunk_')
    i_del = body.find("shutil.rmtree(raw")
    check("phash 를 잰다", i_hash > 0)
    check("매니페스트 저장 **전에** 잰다", 0 < i_hash < i_save, f"{i_hash} < {i_save}")
    check("원본 삭제 **전에** 잰다", 0 < i_hash < i_del, f"{i_hash} < {i_del}")


if __name__ == "__main__":
    print("다른 PC 청크 합치기 검증")
    for fn in (test_dedup_reuses_phash_and_refuses_silent_misses,
               test_merge_preview_and_apply,
               test_merge_stops_on_clash_and_missing_manifest,
               test_chunk_hashes_before_deleting_raw):
        fn()
    print()
    if FAILS:
        print(f"{len(FAILS)} check(s) failed:")
        for f in FAILS:
            print(f"  - {f}")
        sys.exit(1)
    print("all checks passed")
