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
    # ⚠️ aihubshell 은 **조각으로 받아 병합**합니다. 병합하는 순간 조각과 합친
    #    파일이 같이 있을 수 있어서, 이 리포가 직접 경고하는 값이 zip 의 2배입니다
    #    (src/aihub.py: "압축 해제까지 순간 최대 약 {need*2}GB 필요").
    #    실제로 얼마인지는 안 재봤습니다 — 그래서 두 값을 다 보여줍니다.
    dl_peak_b = zip_b * 2

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

    safe = dl_peak_b + crop_b            # 리포가 경고하는 보수적 값
    fits = False
    for label, avail, note in steps:
        okmark = avail >= safe
        fits = fits or okmark
        m = "✅" if okmark else ("◐" if avail >= need else "❌")
        print(f"  {m} {label:<28} 여유 {human(avail):>10}   {note}")

    print(f"\n     필요 — 낙관 {human(need):>10}   (zip {human(zip_b)} + 크롭 {human(crop_b)})")
    print(f"     필요 — 보수 {human(safe):>10}   (다운로드 병합 중 2배 · src/aihub.py 경고)")
    print("     ⚠️ 실제 최대치는 **안 재봤습니다.** aihubshell 이 조각을 병합할 때")
    print("        조각과 합친 파일이 같이 있으면 보수 쪽에 가깝습니다.")
    print("        ◐ 는 낙관값은 넘지만 보수값은 못 넘는 구간입니다 — 도박입니다.")

    if not fits:
        print(f"\n  ❌ 다 치워도 보수값({human(safe)})에는 {human(safe - run)} 모자랍니다.")
        if run >= need:
            print(f"     낙관값({human(need)})은 넘지만, 실패하면 몇 시간을 날립니다.")
        print("\n  ── 그래도 하려면 ──")
        print("  1) 다른 드라이브가 있으면 zip 만 거기로 보냅니다 (제일 깔끔):")
        print(f"       set DOG_SKIN_DATA=D:\\daengs_raw")
        print(f"       uv run python prepare_local.py --chunk {chunk} --margins 2.5,-320")
        print(f"     → C: 에는 크롭 {human(crop_b)} 만 있으면 됩니다.")
        print("  2) 크롭도 옮기려면  set DOG_SKIN_WORK=D:\\daengs_work")
        print("  3) 안 되면 데이터 늘리기는 접고 06 재실행만 하는 게 맞습니다.")
        return

    print("\n  ── 명령 ──")
    if drop_b:
        print("  uv run python tools/crops.py --prune")
    if r["other_b"]:
        print("  uv run python tools/crops.py --prune-raw   ← zip 은 남깁니다")
    if run - r["zip_b"] < safe <= run:
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


def raw_victims(raw: Path) -> tuple[list[dict], list[dict]]:
    """`data/raw` 의 최상위 항목을 (지워도 되는 것, 지키는 것) 으로 가릅니다.

    지키는 규칙은 **세 줄**이고, 헷갈리면 지키는 쪽으로 넘깁니다:
      1. 이름이 `.` 로 시작하면 지킵니다 — `.downloaded_keys` 가 여기 있습니다.
         이걸 지우면 `aihub.download()` 가 이미 받은 걸 또 받습니다 (21GB).
      2. `.zip` 파일은 지킵니다 — `--recrop` 이 픽셀을 여기서 읽습니다.
      3. 폴더는 **그 안에 zip 이 하나라도 있으면** 통째로 지킵니다.
         압축 해제본과 zip 이 같은 폴더에 섞여 있을 수 있어서, 그럴 땐
         자동으로 안 건드리고 사람에게 넘깁니다.
    """
    victims: list[dict] = []
    guarded: list[dict] = []
    if not raw.is_dir():
        return victims, guarded

    for p in sorted(raw.iterdir()):
        if p.name.startswith("."):                    # ① 다운로드 기록 등
            continue
        if p.is_file():
            if p.suffix.lower() == ".zip":            # ② 원본
                continue
            victims.append({"path": p, "n": 1, "bytes": p.stat().st_size,
                            "kind": "파일"})
            continue
        n, b = scan_tag(p)                            # 폴더
        holds_zip = any(f.suffix.lower() == ".zip"
                        for f in p.rglob("*") if f.is_file())
        (guarded if holds_zip else victims).append(
            {"path": p, "n": n, "bytes": b, "kind": "폴더"})
    return victims, guarded


def prune_raw(s: dict, yes: bool = False) -> None:
    """원본 폴더에서 **zip 과 다운로드 기록만 남기고** 나머지를 지웁니다.

    크롭은 zip 에서 직접 읽기 때문에 압축 해제본은 크롭이 끝나면 쓰이지 않습니다.
    그래도 `--prune` 과 같은 규칙으로, 지울 이름을 그대로 입력해야 지웁니다.
    """
    raw = s["raw"]["path"]
    if not raw.is_dir():
        print(f"\n원본 폴더가 없습니다  ({raw})")
        return

    victims, guarded = raw_victims(raw)
    zb, zn = s["raw"]["zip_b"], s["raw"]["zip_n"]

    print(f"\n원본 폴더  {raw}")
    print(f"  남길 것   zip {zn:,}개 {human(zb)}  +  `.` 로 시작하는 기록 파일")
    if guarded:
        print("  ⚠️ 아래 폴더는 **안에 zip 이 있어서** 안 건드립니다:")
        for g in guarded:
            print(f"       {g['path'].name}  ({g['n']:,}개 {human(g['bytes'])})")
        print("     안의 zip 을 밖으로 옮긴 뒤 다시 돌리세요.")

    if not victims:
        print("\n지울 게 없습니다. 이미 zip 만 남아 있어요.")
        return

    total = sum(v["bytes"] for v in victims)
    print("\n지울 것:")
    for v in victims:
        print(f"  {v['path'].name:<28}{v['kind']:>4}{v['n']:>9,}개{human(v['bytes']):>13}")
    print(f"  {'─' * 54}  {human(total)} 확보")

    if zn:
        print("\n✅ zip 이 남으므로 되돌리기는 재다운로드가 아니라 재압축해제입니다.")
        print("   크롭 자체는 zip 에서 직접 읽어서, 이걸 지워도 --recrop 이 됩니다.")
    else:
        print("\n⚠️ 이 폴더엔 zip 이 **없습니다.** 지우면 그 청크는 재다운로드입니다 "
              f"({BASE_CHUNK} = {CHUNK_GB[BASE_CHUNK]}GB).")

    names = ",".join(v["path"].name for v in victims)
    if not yes:
        print(f"\n정말 지우려면 지울 이름을 그대로 입력하세요: {names}")
        try:
            got = input("> ").strip()
        except EOFError:
            got = ""
        if got.replace(" ", "") != names:
            print("입력이 달라서 **아무것도 안 지웠습니다.**")
            return

    for v in victims:
        if v["path"].is_dir():
            shutil.rmtree(v["path"], ignore_errors=True)
        else:
            v["path"].unlink(missing_ok=True)
        print(f"  지움  {v['path'].name}")
    print(f"\n{human(total)} 확보했습니다. zip {zn:,}개는 그대로 있습니다.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plan", metavar="청크", help="새 청크를 받으면 얼마나 드나 (예: TL01)")
    ap.add_argument("--prune", action="store_true", help="안 쓰는 크롭 태그 삭제")
    ap.add_argument("--prune-raw", action="store_true",
                    dest="prune_raw", help="원본에서 zip·기록만 남기고 삭제")
    ap.add_argument("--yes", action="store_true", help="확인 입력 건너뛰기")
    a = ap.parse_args(argv)

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
    if not (a.plan or a.prune or a.prune_raw):
        print("\n  --plan TL01   새 청크 용량 예상")
        print("  --prune       안 쓰는 크롭 태그 삭제 (확인 입력 필요)")
        print("  --prune-raw   원본에서 zip 만 남기고 정리 (확인 입력 필요)")


if __name__ == "__main__":
    main()
