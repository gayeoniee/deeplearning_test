"""크롭 폴더 용량 보고 · 안 쓰는 태그 정리 · 새 청크 용량 예상.

    uv run python tools/crops.py                 # 보고만 (아무것도 안 지웁니다)
    uv run python tools/crops.py --plan TL01     # 새 청크를 받으면 얼마나 드나
    uv run python tools/crops.py --prune         # 안 쓰는 태그 삭제 (확인 입력 필요)
    uv run python tools/crops.py --prune-raw     # 원본에서 zip 만 남기고 정리

왜 필요한가 — 전처리는 크롭을 **네 종류**(m1.5 / m2.5 / full / f320) 만드는데
지금 파이프라인은 **둘만** 씁니다. 나머지는 실험이 끝나서 놀고 있는 용량입니다.

되돌리기 비용은 `data/raw` 가 살아 있느냐에 달렸습니다:
  · 원본 zip 이 있으면 → `--recrop` 으로 공짜로 되살립니다
  · 없으면            → 그 청크를 다시 받아야 합니다 (VL01 = 21GB)
이 도구가 그때그때 어느 쪽인지 알려줍니다.

그래도 **기본은 보고**이고, 삭제는 태그 이름을 직접 입력해야 합니다.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 지금 파이프라인이 쓰는 태그. **하드코딩하지 않고** 서빙 코드에서 읽습니다 —
# 두 곳에 적어두면 갈라지고, 갈라지면 쓰는 크롭을 지우게 됩니다.
from src.agent import STAGE1_TAG, STAGE2_TAG                     # noqa: E402

# 다 만들어진 청크 크기 (aihub.KNOWN_FILES_561 과 같은 값)
CHUNK_GB = {"VL01": 21, "VS01": 21, "TL01": 90, "TL02": 80, "TS01": 90, "TS02": 80}

# 다운로드 중 한순간 동시에 살아 있는 배수. **추정이 아니라 aihubshell 소스를
# 읽은 값입니다** (2026-08-26, `--shell-peek`). docs/results/AIHUBSHELL_피크_실측.md
#   ① curl -o download.tar                     1배
#   ② tar -xvf download.tar (뒤에 rm 없음)      + 풀린 내용   = 2배
#   ③ find|xargs cat > 합친파일 (조각 삭제는 그 뒤)  + 합친 것 = 3배
# 조각(.part*)으로 안 쪼개진 청크면 ②에서 끝나 2배입니다 — 받아보기 전엔 모릅니다.
from src.aihub import DL_PEAK        # ★ 한 곳에만 (src/aihub.py)
BASE_CHUNK = "VL01"          # 지금 갖고 있는 청크 — 이걸 기준으로 비례 계산합니다


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return ""


def scan_tag(d: Path) -> tuple[int, int]:
    """(파일 수, 총 바이트). 큰 폴더라 한 번만 훑습니다."""
    n = tot = 0
    for f in d.rglob("*"):
        if f.is_file():
            n += 1
            tot += f.stat().st_size
    return n, tot


def survey() -> dict:
    from src import env

    croot = env.work_root() / "crops"
    used = {STAGE1_TAG, STAGE2_TAG}
    tags = {}
    for d in (sorted(p for p in croot.iterdir() if p.is_dir())
              if croot.is_dir() else []):
        n, b = scan_tag(d)
        tags[d.name] = {"path": d, "n": n, "bytes": b, "used": d.name in used}

    # 원본(zip). 살아 있으면 지운 크롭을 `--recrop` 으로 공짜로 되살릴 수 있고,
    # 지우면 그만큼 공간이 나지만 되살리기는 재다운로드가 됩니다.
    from src import env as _env

    raw = _env.data_root()
    zn = zb = on = ob = 0
    if raw.is_dir():
        for f in raw.rglob("*"):
            if not f.is_file():
                continue
            if f.suffix.lower() == ".zip":
                zn += 1; zb += f.stat().st_size
            else:
                on += 1; ob += f.stat().st_size
    return {"root": croot, "tags": tags, "used": used,
            "raw": {"path": raw, "n": zn + on, "bytes": zb + ob, "alive": zn + on > 0,
                    "zip_n": zn, "zip_b": zb, "other_n": on, "other_b": ob}}


def report(s: dict) -> None:
    tags = s["tags"]
    if not tags:
        print(f"\n크롭 폴더  {s['root']}  — 아직 없습니다 (새 PC 인가요?)")
        try:
            du = shutil.disk_usage(s["root"].parent.parent)
            print(f"  디스크    여유 {human(du.free)} / 전체 {human(du.total)}")
        except OSError:
            pass
        return
    print(f"\n크롭 폴더  {s['root']}\n")
    print(f"  {'태그':<8}{'파일 수':>10}{'용량':>13}   쓰임")
    print("  " + "─" * 52)
    keep = drop = 0
    for name, t in tags.items():
        mark = ("★ 1단계" if name == STAGE1_TAG else
                "★ 2단계" if name == STAGE2_TAG else "안 씀")
        print(f"  {name:<8}{t['n']:>10,}{human(t['bytes']):>13}   {mark}")
        if t["used"]:
            keep += t["bytes"]
        else:
            drop += t["bytes"]
    print("  " + "─" * 52)
    print(f"  {'합계':<8}{sum(t['n'] for t in tags.values()):>10,}"
          f"{human(keep + drop):>13}")
    print(f"\n  쓰는 것   {human(keep)}")
    print(f"  안 쓰는 것 {human(drop)}   ← --prune 으로 지울 수 있습니다")

    r = s["raw"]
    if r["alive"]:
        print(f"\n  원본  {r['path']}")
        print(f"    zip        {human(r['zip_b']):>12}  ({r['zip_n']:,}개)   "
              f"{'← --recrop 이 쓰는 것' if r['zip_n'] else ''}")
        if r["other_n"]:
            print(f"    zip 아닌 것 {human(r['other_b']):>12}  ({r['other_n']:,}개)   "
                  f"← **안 씁니다** (압축 해제본으로 보입니다)")
            print("       크롭은 zip 에서 직접 읽습니다. 이건 지워도 --recrop 이 됩니다.")
            print("       → uv run python tools/crops.py --prune-raw")
    else:
        print("\n  원본  없음 — 크롭을 지우면 되살리는 데 재다운로드가 필요합니다.")

    try:
        du = shutil.disk_usage(s["root"])
        print(f"\n  디스크    여유 {human(du.free)} / 전체 {human(du.total)}")
    except OSError:
        pass


def plan(s: dict, chunk: str) -> None:
    """새 청크를 받으면 용량이 얼마나 드나 — **지금 폴더를 재서** 비례 계산."""
    if chunk not in CHUNK_GB:
        raise SystemExit(f"모르는 청크 '{chunk}'. 아는 것: {', '.join(CHUNK_GB)}")

    tags = s["tags"]
    have = sum(t["n"] for t in tags.values() if t["used"])

    if have:
        # 태그 하나당 이미지 한 장의 평균 크기 — **지금 폴더를 재서**
        per_img = {n: (t["bytes"] / t["n"] if t["n"] else 0) for n, t in tags.items()}
        used_per_img = sum(per_img[n] for n in s["used"] if n in per_img)
        n_base = max(t["n"] for t in tags.values())
        measured_here = True
    else:
        # 크롭이 아직 없는 PC (새로 클론한 작업용 PC)에서도 계획을 세울 수 있게,
        # 2026-08-26 에 VL01 에서 **실측한** 값을 기본으로 씁니다.
        #   47,605장 · f320 1.0GB + m2.5 1.6GB = 2.6GB  →  장당 약 57KB
        per_img = {STAGE1_TAG: 22 * 1024, STAGE2_TAG: 35 * 1024}
        used_per_img = sum(per_img.values())
        n_base = 47_605
        measured_here = False
        print("\n  ℹ️ 이 PC 엔 크롭이 없어서 VL01 실측값(장당 57KB)으로 계산합니다.")

    ratio = CHUNK_GB[chunk] / CHUNK_GB[BASE_CHUNK]
    n_new = int(round(n_base * ratio))

    zip_b = CHUNK_GB[chunk] * 1024 ** 3
    crop_b = int(n_new * used_per_img)
    # ⚠️ 전에 여기서 "보수 = zip × 2" 를 필요량으로 내걸었는데 **틀렸습니다.**
    #    그 2배는 src/aihub.py:498 의 경고인데, 그 경고는 "zip 을 받은 뒤
    #    **압축을 풀므로**" 라는 이유로 붙은 것입니다. 우리 파이프라인은
    #    `prepare_local.py --mode zip` (기본값) 으로 **압축을 안 풉니다** —
    #    zip 안에서 바로 읽습니다. 그래서 그 절반은 우리에게 해당이 없습니다.
    #
    #    남는 진짜 미지수는 하나뿐입니다: aihubshell 이 조각을 **병합할 때**
    #    조각과 합친 파일이 잠깐 같이 있는가. 이건 **안 재봤습니다.**
    #    추정치를 필요량에 못 박지 않고(규칙 1), 아래에 미지수로 따로 적습니다.
    #    settle 하는 법: PC 의 aihubshell 스크립트를 직접 읽으면 됩니다
    #    → uv run python tools/crops.py --shell-peek

    print(f"\n{chunk} 을(를) 받으면  ({BASE_CHUNK} {CHUNK_GB[BASE_CHUNK]}GB · "
          f"{n_base:,}장 을 기준으로 비례)\n")
    print(f"  예상 장수          {n_new:,}장   ({ratio:.1f}배)")
    print(f"  zip                {human(zip_b)}   (크롭 뒤 자동 삭제)")
    print(f"  크롭 {STAGE1_TAG}+{STAGE2_TAG} 만  {human(crop_b)}")

    if measured_here and len(tags) > len(s["used"]):
        all_per_img = sum(per_img.values())
        print(f"  (기본값처럼 {len(tags)}종을 다 만들면 크롭이 "
              f"{human(int(n_new * all_per_img))})")

    # ── 단계별 계획: 어디서 얼마가 필요한지 ──
    try:
        probe = s["root"] if s["root"].is_dir() else ROOT
        free = shutil.disk_usage(probe).free
    except OSError:
        print("\n  (디스크 여유를 못 읽어서 계획은 생략합니다)")
        return

    drop_b = sum(t["bytes"] for t in tags.values() if not t["used"])
    r = s["raw"]
    need = zip_b + crop_b

    # ⚠️ zip 이 떨어질 곳과 크롭이 쌓일 곳이 **다른 드라이브**일 수 있습니다
    #    (DOG_SKIN_DATA 로 외장을 지정한 경우). 아래 계획은 크롭 쪽 디스크를
    #    기준으로 세는데, 받는 도중 최대치는 zip 쪽에서 납니다. 갈라져 있으면
    #    그 사실과 그쪽 여유를 따로 말해줍니다 — 안 그러면 통과처럼 보입니다.
    try:
        from src import env

        dl_root = env.data_root()
        dl_root.mkdir(parents=True, exist_ok=True)
        if os.stat(dl_root).st_dev != os.stat(probe).st_dev:
            dl_free = shutil.disk_usage(dl_root).free
            print(f"\n  ⓘ 원본 zip 은 다른 드라이브로 갑니다: {dl_root}")
            print(f"     그쪽 여유 {human(dl_free)} — **받는 도중 최대치는 이쪽**에서 납니다.")
            print(f"     크롭은 {probe} 에 쌓입니다.")
    except Exception:                                             # noqa: BLE001
        pass

    print(f"\n  ── 단계별 계획 (지금 여유 {human(free)}) ──\n")
    steps = [("지금 그대로 시작", free, "")]
    run = free
    if drop_b:
        run += drop_b
        steps.append(("① --prune (안 쓰는 크롭)", run, f"+{human(drop_b)}"))
    if r["other_b"]:
        run += r["other_b"]
        steps.append(("② 압축 해제본 삭제 (안 씀)", run, f"+{human(r['other_b'])}"))
    if r["zip_b"]:
        run += r["zip_b"]
        steps.append(("③ 원본 zip 도 삭제", run, f"+{human(r['zip_b'])}"))

    peak_hi = int(zip_b * DL_PEAK[1]) + crop_b
    peak_lo = int(zip_b * DL_PEAK[0]) + crop_b
    fits = False
    for label, avail, note in steps:
        okmark = avail >= peak_hi
        fits = fits or okmark
        m = "✅" if okmark else ("◐" if avail >= peak_lo else "❌")
        print(f"  {m} {label:<28} 여유 {human(avail):>10}   {note}")

    lo, hi = (int(zip_b * m) + crop_b for m in DL_PEAK)
    print(f"\n     크롭이 끝난 뒤    {human(need):>10}   (zip {human(zip_b)} + 크롭 {human(crop_b)})")
    print("       └ 압축은 **안 풉니다** (--mode zip 이 기본). 해제본 몫은 안 듭니다.")
    print(f"     받는 도중 최대치  {human(lo)} ~ {human(hi)}"
          f"   ({DL_PEAK[0]:g}~{DL_PEAK[1]:g}배)")
    print("       └ aihubshell 이 download.tar → 조각 → 합친파일 을 겹쳐 놓습니다.")
    print("         **추정이 아니라 소스를 읽은 값입니다** → --shell-peek")
    if crop_b > 2 * 1024 ** 3:
        print(f"\n     💡 크롭 {human(crop_b)} 은 줄일 수 있습니다 — 필요한 라벨만:")
        print(f"        prepare_local.py --chunk {chunk} --margins 2.5,-320 --only A5,A6")
        print("        (다운로드는 안 줄어듭니다. zip 은 통짜로만 줍니다)")

    if not fits:
        short = peak_hi - run
        print(f"\n  ❌ 다 치워도 최대치({human(peak_hi)})에 {human(short)} 모자랍니다.")
        if run >= peak_lo:
            print(f"     2배({human(peak_lo)})면 되긴 합니다 — 조각으로 안 쪼개져 있을 때만.")
        try:
            tot = shutil.disk_usage(probe).total
            if peak_hi > tot:
                print(f"     ⚠️ 애초에 디스크 전체({human(tot)})보다 큽니다. "
                      "다 비워도 안 됩니다.")
        except OSError:
            pass
        # ⚠️ 환경변수 거는 법이 OS 마다 다릅니다. 맥/리눅스에서 `set X=Y` 는
        #    조용히 아무것도 안 하고, 그대로 받다가 원래 드라이브가 찹니다.
        win = os.name == "nt"
        setter = "set {k}={v}" if win else "export {k}={v}"
        other = "D:\\daengs_raw" if win else "/Volumes/USB/daengs_raw"
        work = "D:\\daengs_work" if win else "/Volumes/USB/daengs_work"
        here = "C: 드라이브" if win else "내장 디스크"
        print("\n  ── 그래도 하려면 ──")
        print("  1) 다른 드라이브(외장/USB)가 있으면 zip 만 거기로 보냅니다 (제일 깔끔):")
        print("       " + setter.format(k="DOG_SKIN_DATA", v=other))
        print(f"       uv run python prepare_local.py --chunk {chunk} --margins 2.5,-320")
        print(f"     → {here}에는 크롭 {human(crop_b)} 만 있으면 됩니다.")
        print("  2) 크롭도 옮기려면  " + setter.format(k="DOG_SKIN_WORK", v=work))
        print("  3) 안 되면 데이터 늘리기는 접고 06 재실행만 하는 게 맞습니다.")
        return

    print("\n  ── 명령 ──")
    if drop_b:
        print("  uv run python tools/crops.py --prune")
    if r["other_b"]:
        print("  uv run python tools/crops.py --prune-raw   ← zip 은 남깁니다")
    if run - r["zip_b"] < peak_hi <= run:
        # zip 까지 지워야 되는 경우엔 그 명령도 적어줍니다 (빠져 있었습니다)
        print(f"  del {r['path']}\\*.zip"
              f"        ⚠️ 되돌리려면 {BASE_CHUNK} 재다운로드 ({CHUNK_GB[BASE_CHUNK]}GB)")
    print(f"  uv run python prepare_local.py --chunk {chunk} --margins 2.5,-320")
    print("  uv run python prepare_local.py --finalize")


def prune(s: dict, yes: bool = False) -> None:
    tags = s["tags"]
    victims = {n: t for n, t in tags.items() if not t["used"]}
    if not victims:
        print("\n안 쓰는 태그가 없습니다. 지울 게 없어요.")
        return

    total = sum(t["bytes"] for t in victims.values())
    print("\n지울 것:")
    for n, t in victims.items():
        print(f"  {n:<8}{t['n']:>10,}{human(t['bytes']):>13}")
    print(f"  ────────────────────────────────  {human(total)} 확보")

    if s["raw"]["alive"]:
        print(f"\n✅ 원본 zip 이 살아 있어서({human(s['raw']['bytes'])}) 되돌리기 쉽습니다:")
        print("     uv run python prepare_local.py --recrop m1.5   (재다운로드 없이)")
    else:
        print("\n⚠️ 원본 zip 이 없어서 되돌리려면 **다시 받아야** 합니다 "
              f"({BASE_CHUNK} = {CHUNK_GB[BASE_CHUNK]}GB).")
        print("   `--recrop` 은 매니페스트만 읽고 픽셀은 zip 에서 가져오기 때문입니다.")
    print("\n   잃는 것 — 이 태그를 쓰던 실험은 다시 못 돌립니다:")
    if "m1.5" in victims:
        print("     m1.5   크롭 배율 비교 (STEP 4C — m2.5 채택으로 이미 결론)")
    if "full" in victims:
        print("     full   1단계 입력 비교 (STEP 9-A — f320 채택으로 이미 결론)")
        print("            crop.full_crop_loss() / shortcut_baseline() 도 못 씁니다")
    print("   둘 다 결론이 난 실험이라 다시 돌릴 일은 없어 보입니다.")

    names = ",".join(victims)
    if not yes:
        print(f"\n정말 지우려면 지울 태그를 그대로 입력하세요: {names}")
        try:
            got = input("> ").strip()
        except EOFError:
            got = ""
        if got.replace(" ", "") != names:
            print("입력이 달라서 **아무것도 안 지웠습니다.**")
            return

    for n, t in victims.items():
        shutil.rmtree(t["path"], ignore_errors=True)
        print(f"  지움  {n}")
    print(f"\n{human(total)} 확보했습니다.")


def raw_units(raw: Path) -> tuple[list[dict], list[dict]]:
    """`data/raw` 를 끝까지 걸어 (지울 것, 지킬 것) 으로 가릅니다.

    ⚠️ **최상위만 보면 안 됩니다.** 실제 배치는 이렇게 생겼습니다:

        data/raw/VL01/VL01.zip                        ← 지킴
        data/raw/VL01/.downloaded_keys                ← 지킴
        data/raw/VL01/152.반려동물_피부질환_데이터/…/반려견/…  ← 압축 해제본, 지움

    zip 이 `VL01/` **안**에 있어서, 최상위 `VL01` 을 통째로 판단하면
    "안에 zip 이 있네" 로 아무것도 못 지웁니다. 그래서 파일 단위로 가른 뒤,
    **지킬 게 하나도 없는 가장 얕은 폴더**를 삭제 단위로 잡습니다.

    지키는 규칙 두 줄 — 헷갈리면 지키는 쪽입니다:
      1. `.zip` — `--recrop` 이 픽셀을 여기서 읽습니다 (`zip_path` + `zip_member`)
      2. 이름이 `.` 로 시작 — `.downloaded_keys`. 지우면 `aihub.download()` 가
         또 받습니다. (`has_usable_data()` 도 zip 을 세므로 zip 만 남아도 안전)
    """
    keep: list[dict] = []

    def walk(d: Path) -> tuple[bool, list[dict]]:
        """(이 폴더 아래에 지킬 게 있나, 지울 단위 목록)"""
        has_keep = False
        units: list[dict] = []
        try:
            entries = sorted(d.iterdir())
        except OSError:
            return True, []                       # 못 읽으면 안 건드립니다
        for p in entries:
            if p.name.startswith("."):            # ② 기록 파일
                has_keep = True
                keep.append({"path": p, "why": "기록"})
                continue
            if p.is_symlink() or p.is_file():
                if p.is_file() and p.suffix.lower() == ".zip":   # ① 원본
                    has_keep = True
                    keep.append({"path": p, "why": "zip",
                                 "bytes": p.stat().st_size})
                    continue
                units.append({"path": p, "n": 1, "kind": "파일",
                              "bytes": p.stat().st_size if p.is_file() else 0})
                continue
            sub_keep, sub_units = walk(p)
            if sub_keep:
                has_keep = True
                units.extend(sub_units)           # 폴더는 남기고 안쪽만
            else:
                n, b = scan_tag(p)
                units.append({"path": p, "n": n, "kind": "폴더", "bytes": b})
        return has_keep, units

    if not raw.is_dir():
        return [], []
    return walk(raw)[1], keep


def prune_raw(s: dict, yes: bool = False) -> None:
    """원본 폴더에서 **zip 과 다운로드 기록만 남기고** 나머지를 지웁니다.

    크롭은 zip 안에서 직접 읽습니다 (`prepare_local.py --mode zip` 이 기본).
    그래서 압축 해제본은 크롭이 끝나면 아무도 안 읽습니다.
    """
    raw = s["raw"]["path"]
    if not raw.is_dir():
        print(f"\n원본 폴더가 없습니다  ({raw})")
        return

    units, keep = raw_units(raw)
    zips = [k for k in keep if k["why"] == "zip"]
    marks = [k for k in keep if k["why"] != "zip"]

    print(f"\n원본 폴더  {raw}\n")
    print("  남길 것:")
    for k in zips:
        print(f"    {human(k['bytes']):>10}  {k['path'].relative_to(raw)}")
    for k in marks:
        print(f"    {'(기록)':>8}  {k['path'].relative_to(raw)}")
    if not keep:
        print("    (없음)")

    if not units:
        print("\n지울 게 없습니다. 이미 zip 만 남아 있어요.")
        return

    total = sum(u["bytes"] for u in units)
    print("\n  지울 것:")
    for u in units:
        print(f"    {u['kind']}{u['n']:>9,}개{human(u['bytes']):>12}  "
              f"{u['path'].relative_to(raw)}")
    print(f"    {'─' * 56}  {human(total)} 확보")

    if zips:
        print("\n✅ zip 이 남으므로 되돌리기는 재다운로드가 아니라 재압축해제입니다.")
        print("   크롭은 zip 안에서 직접 읽어서(--mode zip), 이걸 지워도 --recrop 이 됩니다.")
    else:
        print("\n⚠️ 이 폴더엔 zip 이 **없습니다.** 지우면 그 청크는 재다운로드입니다 "
              f"({BASE_CHUNK} = {CHUNK_GB[BASE_CHUNK]}GB).")

    # 확인 입력은 **최상위 이름**으로 받습니다. 위 목록의 전체 경로는 한글이 길어
    # 타이핑이 사실상 불가능하고, 무엇이 지워지는지는 목록이 이미 보여줬습니다.
    tops: list[str] = []
    for u in units:
        t = u["path"].relative_to(raw).parts[0]
        if t not in tops:
            tops.append(t)
    names = ",".join(tops)
    if not yes:
        print(f"\n정말 지우려면 이 이름을 그대로 입력하세요: {names}")
        try:
            got = input("> ").strip()
        except EOFError:
            got = ""
        if got.replace(" ", "") != names:
            print("입력이 달라서 **아무것도 안 지웠습니다.**")
            return

    for u in units:
        if u["path"].is_dir() and not u["path"].is_symlink():
            shutil.rmtree(u["path"], ignore_errors=True)
        else:
            u["path"].unlink(missing_ok=True)
        print(f"  지움  {u['path'].relative_to(raw)}")
    print(f"\n{human(total)} 확보했습니다. zip {len(zips)}개는 그대로 있습니다.")


# 병합 여유를 **재는 대신 읽습니다.** aihubshell 은 셸 스크립트라 소스가 그대로 있습니다.
#
# ⚠️ 처음엔 정규식으로 몇 줄만 뽑았는데, 정작 중요한 줄이 잘려서 판단을 못 했습니다.
#    (`cat part* > 파일` 이 **주석**이었고, 진짜 병합은 가려진 줄에 있었습니다)
#    그래서 지금은 **덩어리째** 보여줍니다 — 셸 스크립트는 앞뒤 맥락이 답입니다.
PEEK_ANCHORS = [
    ("병합", r"^\s*merge_parts\s*\(\)|^\s*function\s+merge_parts"),
    ("압축 해제", r"\btar\b\s+-?[a-z]*x[a-z]*\s|\bunzip\b"),
    ("조각 다운로드", r"\.part|Range:|http_status"),
]


def _region(lines: list[str], start: int) -> tuple[int, int]:
    """셸 함수 한 덩어리. `}` 로 끝나면 거기까지, 아니면 앞뒤 몇 줄."""
    if "()" in lines[start]:
        depth = 0
        for k in range(start, min(start + 80, len(lines))):
            depth += lines[k].count("{") - lines[k].count("}")
            if depth <= 0 and k > start and "}" in lines[k]:
                return start, k
    return max(0, start - 4), min(len(lines) - 1, start + 10)


def shell_peek() -> None:
    """`aihubshell` 을 읽어 **디스크가 언제 2배가 되는지** 확인합니다.

    왜 이게 필요한가 — `--plan` 의 마지막 미지수가 "받는 도중에 큰 파일이
    둘 있는 순간이 있나" 입니다. 80GB 짜리를 도박으로 받아보며 재는 것보다,
    **스크립트를 읽는 게 공짜이고 확실합니다.**

    볼 것은 두 군데입니다:
      1. 조각 → download.tar   (`cat part* > tar` 면 2배, `>>` + `rm` 이면 조각 하나)
      2. download.tar → 내용물 (`tar -xvf` 뒤에 tar 를 **지우는 줄이 있나**)
         없으면 tar 와 풀린 내용이 같이 있어서 이쪽이 2배입니다.
    """
    import re

    from src import aihub

    p = aihub.shell_path()
    if not p.exists():
        print(f"\naihubshell 이 없습니다: {p}")
        print("  먼저 받으세요:  uv run python -c \"from src import aihub; aihub.install()\"")
        print("  ⚠️ AI Hub 는 해외 IP 를 막습니다 — **한국 PC 에서** 돌리세요.")
        return

    try:
        src = p.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"\n못 읽었습니다: {exc}")
        return

    lines = src.split("\n")
    print(f"\naihubshell  {p}  ({p.stat().st_size:,} bytes, {len(lines):,}줄)")

    shown: set[int] = set()
    for name, pat in PEEK_ANCHORS:
        rx = re.compile(pat)
        starts = [i for i, ln in enumerate(lines) if rx.search(ln)]
        if not starts:
            print(f"\n  [{name}]  못 찾았습니다 — 스크립트가 바뀐 것 같습니다.")
            continue
        print(f"\n  ── {name} ──")
        for st in starts[:3]:
            a, b = _region(lines, st)
            span = range(a, b + 1)
            if sum(k in shown for k in span) > 0.6 * len(span):
                continue          # 앞 항목에서 이미 보여준 덩어리
            for k in range(a, b + 1):
                shown.add(k)
                mark = "#" if lines[k].lstrip().startswith("#") else " "
                print(f"    {k + 1:>4}{mark} {lines[k][:110]}")
            print()

    # ── 자동 판정 ──────────────────────────────────────────────
    # ⚠️ 직전 버전은 `rm *.tar` 이 **어디에** 있는지를 안 봤습니다. 다운로드
    #    **앞**에 있는 청소 줄을 "푼 뒤 지운다" 로 읽어서 반대로 판정했습니다.
    #    이제 줄 번호 순서를 비교합니다.
    print("  ── 판정 ──")
    live = [(i + 1, ln.strip()) for i, ln in enumerate(lines)
            if not ln.lstrip().startswith("#")]

    def _find(pat: str) -> list[tuple[int, str]]:
        rx = re.compile(pat)
        return [(i, ln) for i, ln in live if rx.search(ln)]

    cat_over = _find(r"\bcat\b[^>]*>[^>]")           # cat … >  (덮어쓰기)
    cat_app = _find(r"\bcat\b[^>]*>>")               # cat … >> (이어붙이기)
    untar = _find(r"\btar\b\s+-?[a-z]*x")
    rm_tar = _find(r"\brm\b[^\n]*\.tar")
    rm_part = _find(r"\brm\b[^\n]*\.part")

    # 몇 배가 동시에 살아 있나 — 청크 내용물 크기를 G 라고 할 때
    peak = 1.0                                        # ① download.tar 그 자체
    why = ["download.tar 를 통째로 받습니다 (curl -o download.tar)"]

    if untar:
        after = [ln for ln, _ in rm_tar if ln > untar[0][0]]
        if after:
            peak = max(peak, 2.0)
            why.append(f"tar 를 풀고 → 줄 {after[0]} 에서 지웁니다 "
                       "(푸는 **동안**은 tar + 풀린 내용 = 2배)")
        else:
            peak = max(peak, 2.0)
            why.append(f"tar 를 풀지만(줄 {untar[0][0]}) **그 뒤에 지우는 줄이 없습니다** "
                       "→ tar 가 계속 남습니다 = 2배")
            if rm_tar:
                why.append(f"   (줄 {rm_tar[0][0]} 의 rm 은 다운로드 **앞**이라 "
                           "이전 찌꺼기 청소입니다)")

    if cat_over:
        peak = max(peak, 3.0 if untar and not [l for l, _ in rm_tar if l > untar[0][0]]
                   else 2.0)
        why.append(f"조각을 `cat … > 합친파일` 로 병합합니다 (줄 {cat_over[0][0]}) "
                   "→ 조각과 합친 파일이 **같이** 있습니다")
        if rm_part:
            why.append(f"   조각 삭제는 줄 {rm_part[0][0]} — 합친 **다음**이라 "
                       "최대치는 안 줄어듭니다")
    elif cat_app:
        why.append(f"조각은 `cat … >>` 로 이어붙입니다 (줄 {cat_app[0][0]}) "
                   "→ 이 구간은 조각 하나 여유면 됩니다")
    else:
        why.append("조각 병합 방식을 못 읽었습니다 — 위 [병합] 덩어리를 직접 보세요")

    for w in why:
        print(f"    · {w}")

    print(f"\n  ⇒ 최대치 약 **{peak:g}배** (청크 내용물 크기 기준)")
    if cat_over and untar:
        print("     tar + 조각 + 합친파일 이 한순간 같이 삽니다.")
        print("     단, 그 청크가 조각으로 안 쪼개져 있으면 2배에서 끝납니다 —")
        print("     쪼개졌는지는 받아보기 전엔 모릅니다. 그래서 **2~3배** 로 잡습니다.")
    print("\n  이건 추정이 아니라 소스를 읽은 값입니다. docs/results/ 에 남겨주세요.")
    return peak


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plan", metavar="청크", help="새 청크를 받으면 얼마나 드나 (예: TL01)")
    ap.add_argument("--prune", action="store_true", help="안 쓰는 크롭 태그 삭제")
    ap.add_argument("--prune-raw", action="store_true",
                    dest="prune_raw", help="원본에서 zip·기록만 남기고 삭제")
    ap.add_argument("--shell-peek", action="store_true", dest="shell_peek",
                    help="aihubshell 을 읽어 병합 시 조각을 지우며 합치는지 확인")
    ap.add_argument("--yes", action="store_true", help="확인 입력 건너뛰기")
    a = ap.parse_args(argv)

    if a.shell_peek:
        shell_peek()
        if not (a.plan or a.prune or a.prune_raw):
            return

    s = survey()
    report(s)
    if a.plan:
        plan(s, a.plan.upper())
    # 원본을 먼저 지웁니다 — 크롭 --prune 은 (zip 이 있으면) 되돌리기 쉬운 쪽이라
    # 순서가 바뀌어도 상관없지만, 화면에서 "남길 zip" 을 먼저 보는 게 안전합니다.
    if a.prune_raw:
        prune_raw(s, a.yes)
    if a.prune:
        prune(survey() if a.prune_raw else s, a.yes)
    if not (a.plan or a.prune or a.prune_raw or a.shell_peek):
        print("\n  --plan TL01   새 청크 용량 예상")
        print("  --shell-peek  aihubshell 이 병합할 때 2배를 쓰는지 확인")
        print("  --prune       안 쓰는 크롭 태그 삭제 (확인 입력 필요)")
        print("  --prune-raw   원본에서 zip 만 남기고 정리 (확인 입력 필요)")


if __name__ == "__main__":
    main()
