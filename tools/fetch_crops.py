"""zip 에서 **매니페스트가 실제로 쓰는 크롭만** 꺼냅니다 (그리고 안 쓰는 건 지웁니다).

    python tools/fetch_crops.py dogskin_prepared.zip --tags f320,m2.5
    python tools/fetch_crops.py dogskin_prepared.zip --tags f320,m2.5 --apply --prune

왜 `unzip` 을 안 쓰나
---------------------
1. **zip 에는 `--finalize` 이전 크롭이 들어 있습니다.** 태그당 423,080장이 있는데
   매니페스트가 쓰는 건 365,428장입니다 — 나머지 57,652장은 중복제거로 빠진
   사진이라 풀어봐야 디스크만 먹습니다.
2. **`unzip -n` 은 잘린 파일을 고치지 못합니다.** 디스크가 차서 죽으면 쓰다 만
   파일이 남는데, `-n` 은 "있으니까 건너뜀" 으로 처리합니다. 학습 때 열다가
   깨진 JPEG 로 죽습니다. 여기서는 **zip 이 적어둔 크기와 대조**해서 다르면
   다시 꺼냅니다.
3. 네트워크 볼륨은 지연이 병목이라 스레드를 여러 개 쓰면 10배 이상 빠릅니다.

기본은 **미리보기**입니다. `--apply` 를 붙여야 실제로 씁니다.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_local = threading.local()


def _zip_for(path: Path) -> zipfile.ZipFile:
    """ZipFile 은 스레드 안전하지 않습니다 — 스레드마다 따로 엽니다."""
    z = getattr(_local, "z", None)
    if z is None:
        z = _local.z = zipfile.ZipFile(path)
    return z


def _gb(n: int) -> str:
    return f"{n / 1024 ** 3:.1f}GB"


def needed_rels(manifest: Path, tags: list[str]) -> dict[str, set[str]]:
    """매니페스트 → 태그별 {crops/<태그>/<hh>/<이름>.jpg} 집합."""
    import pandas as pd

    from src.crop import _out_path

    df = pd.read_parquet(manifest, columns=["image_path"])
    out: dict[str, set[str]] = {}
    for tag in tags:
        out[tag] = {
            _out_path(p, Path("crops"), tag).as_posix() for p in df["image_path"]
        }
    print(f"[매니페스트] {len(df):,}행 → 태그당 크롭 {len(out[tags[0]]):,}장")
    return out


def on_disk(work: Path, tag: str) -> dict[str, int]:
    """지금 디스크에 있는 것 {상대경로: 바이트}. 한 번만 훑습니다."""
    base = work / "crops" / tag
    if not base.is_dir():
        return {}
    got: dict[str, int] = {}
    for sub in base.iterdir():
        if not sub.is_dir():
            continue
        with os.scandir(sub) as it:
            for e in it:
                if e.name.endswith(".jpg"):
                    got[f"crops/{tag}/{sub.name}/{e.name}"] = e.stat().st_size
    return got


def survey(zp: Path, work: Path, tags: list[str], manifest: Path) -> dict:
    want = needed_rels(manifest, tags)

    t0 = time.time()
    with zipfile.ZipFile(zp) as z:
        in_zip: dict[str, dict[str, int]] = {t: {} for t in tags}
        for i in z.infolist():
            if i.is_dir():
                continue
            parts = i.filename.split("/")
            if len(parts) >= 3 and parts[0] == "crops" and parts[1] in in_zip:
                in_zip[parts[1]][i.filename] = i.file_size
    print(f"[zip] 목차 읽음 — {time.time() - t0:.0f}초")

    plan: dict[str, dict] = {}
    for tag in tags:
        disk = on_disk(work, tag)
        zt, wt = in_zip[tag], want[tag]

        missing = [r for r in wt if r not in disk and r in zt]
        broken = [r for r in wt if r in disk and r in zt and disk[r] != zt[r]]
        absent = [r for r in wt if r not in zt]                  # zip 에도 없음
        extra = [r for r in disk if r not in wt]                 # 안 쓰는 것

        plan[tag] = {
            "want": len(wt), "disk": len(disk), "zip": len(zt),
            "missing": missing, "broken": broken,
            "absent": absent, "extra": extra,
            "get_bytes": sum(zt[r] for r in missing) + sum(zt[r] for r in broken),
            "free_bytes": sum(disk[r] for r in extra),
        }
    return plan


def report(plan: dict) -> None:
    get_b = free_b = 0
    for tag, p in plan.items():
        have_ok = p["want"] - len(p["missing"]) - len(p["broken"]) - len(p["absent"])
        print(f"\n[{tag}]")
        print(f"  필요       {p['want']:>8,}장   (zip 안에는 {p['zip']:,}장)")
        print(f"  이미 정상  {have_ok:>8,}장   ({have_ok / max(p['want'], 1):.1%})")
        print(f"  꺼낼 것    {len(p['missing']):>8,}장   + 잘린 것 {len(p['broken']):,}장"
              f"   = {_gb(p['get_bytes'])}")
        print(f"  안 쓰는 것 {len(p['extra']):>8,}장   지우면 {_gb(p['free_bytes'])} 확보")
        if p["absent"]:
            print(f"  ❌ zip 에도 없음 {len(p['absent']):,}장 — 이건 여기서 못 고칩니다")
            for r in p["absent"][:3]:
                print(f"       {r}")
        get_b += p["get_bytes"]
        free_b += p["free_bytes"]
    print(f"\n합계: 꺼낼 것 {_gb(get_b)} / 지워서 버는 것 {_gb(free_b)}"
          f" → 순수 필요 공간 {_gb(max(get_b - free_b, 0))}")


def prune(work: Path, plan: dict) -> int:
    n = 0
    for tag, p in plan.items():
        if not p["extra"]:
            continue
        print(f"  crops/{tag}: 안 쓰는 {len(p['extra']):,}장 삭제 중 …", flush=True)
        for rel in p["extra"]:
            try:
                (work / rel).unlink()
                n += 1
            except FileNotFoundError:
                pass
    return n


def fetch(zp: Path, work: Path, plan: dict, workers: int) -> int:
    todo: list[str] = []
    for p in plan.values():
        todo += p["broken"] + p["missing"]      # 잘린 것부터 (개수가 적습니다)
    if not todo:
        return 0

    # 잘린 파일은 지우고 새로 씁니다 (이어쓰면 안 됩니다)
    for p in plan.values():
        for rel in p["broken"]:
            (work / rel).unlink(missing_ok=True)

    done = 0
    lock = threading.Lock()
    t0 = time.time()

    def one(rel: str) -> None:
        nonlocal done
        dest = work / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        z = _zip_for(zp)
        with z.open(rel) as fsrc, open(tmp, "wb") as fdst:
            fdst.write(fsrc.read())
        os.replace(tmp, dest)                   # ★ 중간에 죽어도 반쪽 파일이 안 남습니다
        with lock:
            done += 1
            if done % 5000 == 0 or done == len(todo):
                el = time.time() - t0
                print(f"    {done:,}/{len(todo):,}장  {done / max(el, 1):.0f}장/초  "
                      f"{el / 60:.1f}분", flush=True)

    print(f"  {len(todo):,}장 꺼내는 중 (스레드 {workers}개) …", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, todo))
    return done


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("zip", help="크롭 zip (예: dogskin_prepared.zip)")
    ap.add_argument("--tags", default="f320,m2.5")
    ap.add_argument("--work", default=None, help="data/work 경로 (기본: 자동 감지)")
    ap.add_argument("--manifest", default=None,
                    help="기본: <work>/manifests/manifest_final.parquet")
    ap.add_argument("--apply", action="store_true", help="실제로 꺼냅니다")
    ap.add_argument("--prune", action="store_true",
                    help="매니페스트가 안 쓰는 크롭을 지웁니다 (공간 확보)")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args(argv)

    if a.work:
        work = Path(a.work).expanduser().resolve()
    else:
        from src import env
        work = env.work_root()
    manifest = Path(a.manifest) if a.manifest else work / "manifests" / "manifest_final.parquet"
    zp = Path(a.zip).expanduser().resolve()

    if not manifest.exists():
        raise SystemExit(f"❌ 매니페스트가 없습니다: {manifest}\n"
                         f"   zip 의 manifests/ 를 먼저 풀어야 합니다.")

    tags = [t.strip() for t in a.tags.split(",") if t.strip()]
    print("=" * 68)
    print(f" zip      : {zp}")
    print(f" 풀 곳    : {work}")
    print(f" 매니페스트: {manifest}")
    print("=" * 68)

    plan = survey(zp, work, tags, manifest)
    report(plan)

    if not a.apply:
        print("\n미리보기입니다. 실제로 하려면 --apply (공간이 빠듯하면 --prune 도) 붙이세요.")
        return

    print("\n[적용]")
    if a.prune:
        print(f"  지운 파일 {prune(work, plan):,}개")
    n = fetch(zp, work, plan, a.workers)
    print(f"\n✅ {n:,}장 꺼냈습니다.")

    # 다시 세어서 확인합니다 — "다 됐겠지" 로 넘어가지 않습니다.
    print("\n[확인]")
    after = survey(zp, work, tags, manifest)
    bad = {t: len(p["missing"]) + len(p["broken"]) for t, p in after.items()}
    for t, p in after.items():
        ok = p["want"] - len(p["missing"]) - len(p["broken"]) - len(p["absent"])
        print(f"  {t:<6} {ok:,}/{p['want']:,} ({ok / max(p['want'], 1):.2%})")
    if any(bad.values()):
        raise SystemExit(f"\n❌ 아직 모자랍니다: {bad}\n"
                         f"   디스크가 찼을 가능성이 큽니다 — 볼륨 크기를 확인하세요.")
    print("\n✅ 두 태그 모두 100%. 06 을 돌려도 됩니다.")


if __name__ == "__main__":
    main()
