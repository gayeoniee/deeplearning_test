"""크롭 폴더 용량 보고 · 안 쓰는 태그 정리 · 새 청크 용량 예상.

    uv run python tools/crops.py                 # 보고만 (아무것도 안 지웁니다)
    uv run python tools/crops.py --plan TL01     # 새 청크를 받으면 얼마나 드나
    uv run python tools/crops.py --prune         # 안 쓰는 태그 삭제 (확인 입력 필요)

왜 필요한가 — 전처리는 크롭을 **네 종류**(m1.5 / m2.5 / full / f320) 만드는데
지금 파이프라인은 **둘만** 씁니다. 나머지는 실험이 끝나서 놀고 있는 용량입니다.

⚠️ **지우면 되돌리는 데 재다운로드가 필요합니다.**
`--recrop` 은 매니페스트만 읽고 크롭을 다시 만드는 명령인데, 실제 픽셀은
zip 에서 읽습니다. `--chunk` 가 크롭 후 원본을 지우므로, 지운 태그를 되살리려면
그 청크 zip 을 다시 받아야 합니다 (VL01 = 21GB).

그래서 이 도구는 **기본이 보고**이고, 삭제는 태그 이름을 직접 입력해야 합니다.
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
    if not croot.is_dir():
        raise SystemExit(f"크롭 폴더가 없습니다: {croot}\n"
                         "한국 PC 에서 `prepare_local.py --chunk …` 를 먼저 도세요.")

    used = {STAGE1_TAG, STAGE2_TAG}
    tags = {}
    for d in sorted(p for p in croot.iterdir() if p.is_dir()):
        n, b = scan_tag(d)
        tags[d.name] = {"path": d, "n": n, "bytes": b, "used": d.name in used}
    return {"root": croot, "tags": tags, "used": used}


def report(s: dict) -> None:
    tags = s["tags"]
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
    if have == 0:
        raise SystemExit("쓰는 태그의 크롭이 없어서 비례 계산을 못 합니다.")

    # 태그 하나당 이미지 한 장의 평균 크기
    per_img = {n: (t["bytes"] / t["n"] if t["n"] else 0) for n, t in tags.items()}
    used_per_img = sum(per_img[n] for n in s["used"] if n in per_img)

    ratio = CHUNK_GB[chunk] / CHUNK_GB[BASE_CHUNK]
    n_base = max(t["n"] for t in tags.values())
    n_new = int(round(n_base * ratio))

    zip_b = CHUNK_GB[chunk] * 1024 ** 3
    crop_b = int(n_new * used_per_img)

    print(f"\n{chunk} 을(를) 받으면  ({BASE_CHUNK} {CHUNK_GB[BASE_CHUNK]}GB · "
          f"{n_base:,}장 을 기준으로 비례)\n")
    print(f"  예상 장수          {n_new:,}장   ({ratio:.1f}배)")
    print(f"  zip                {human(zip_b)}   (크롭 뒤 자동 삭제)")
    print(f"  크롭 {STAGE1_TAG}+{STAGE2_TAG} 만  {human(crop_b)}")
    print(f"  ── 처리 중 최대     {human(zip_b + crop_b)}")

    all_per_img = sum(per_img.values())
    print(f"\n  (참고) 기본값처럼 {len(tags)}종을 다 만들면 크롭이 "
          f"{human(int(n_new * all_per_img))} 입니다.")
    print(f"  → 쓰는 것만 만들려면:")
    print(f"     uv run python prepare_local.py --chunk {chunk} --margins 2.5,-320")

    try:
        free = shutil.disk_usage(s["root"]).free
        need = zip_b + crop_b
        print(f"\n  디스크 여유 {human(free)} vs 필요 {human(need)}   "
              f"{'✅ 됩니다' if free > need * 1.1 else '❌ 모자랍니다'}")
        if free <= need * 1.1:
            drop = sum(t["bytes"] for t in tags.values() if not t["used"])
            if drop:
                print(f"     --prune 으로 {human(drop)} 를 먼저 확보할 수 있습니다.")
    except OSError:
        pass


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

    print("\n⚠️ 되돌리려면 그 청크 zip 을 **다시 받아야** 합니다 "
          f"({BASE_CHUNK} = {CHUNK_GB[BASE_CHUNK]}GB).")
    print("   `--recrop` 은 매니페스트만 읽고 픽셀은 zip 에서 가져오는데,")
    print("   `--chunk` 가 크롭 뒤 원본을 지웠기 때문입니다.")
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plan", metavar="청크", help="새 청크를 받으면 얼마나 드나 (예: TL01)")
    ap.add_argument("--prune", action="store_true", help="안 쓰는 크롭 태그 삭제")
    ap.add_argument("--yes", action="store_true", help="--prune 확인 입력 건너뛰기")
    a = ap.parse_args(argv)

    s = survey()
    report(s)
    if a.plan:
        plan(s, a.plan.upper())
    if a.prune:
        prune(s, a.yes)
    if not (a.plan or a.prune):
        print("\n  --plan TL01   새 청크 용량 예상")
        print("  --prune       안 쓰는 태그 삭제 (확인 입력 필요)")


if __name__ == "__main__":
    main()
