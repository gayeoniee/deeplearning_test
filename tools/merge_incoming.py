"""다른 PC 에서 처리해 온 청크(크롭 + 매니페스트)를 이 PC 에 합칩니다.

    uv run python tools/merge_incoming.py tl02_crops.zip           # 미리보기만
    uv run python tools/merge_incoming.py tl02_crops.zip --apply   # 실제로 합침

zip 이든 이미 풀어놓은 폴더든 받습니다. 안에 `crops/<태그>/…` 와
`manifests/chunk_*.parquet` 이 있으면 됩니다 (`--package` 가 만드는 모양).

왜 손으로 안 하나
-----------------
`unzip -o` 로 기존 폴더에 덮어쓰면 **무엇이 새로 들어왔고 무엇이 덮였는지**
아무도 모릅니다. 크롭 파일 이름은 `md5(image_path)` 라 원본이 같으면 같은
이름인데, 크롭 **설정**이 다르면 내용은 다릅니다. 그런 걸 조용히 덮으면
학습 데이터가 반쯤 섞인 채로 돌아가고 숫자만 이상해집니다.

그리고 **매니페스트를 빠뜨리기 쉽습니다.** 크롭만 있으면 어떤 사진이 어떤
라벨인지 알 수가 없고, `--finalize` 는 있는 매니페스트만 보고 조용히 넘어갑니다.

★ 이 도구는 기본이 **미리보기**입니다. `--apply` 를 붙여야 실제로 씁니다.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 이건 `--finalize` 가 다시 만듭니다. 가져온 걸 덮으면 이 PC 의 분할이 날아갑니다.
NEVER_COPY = {"manifest_final.parquet"}


# ──────────────────────────────────────────────────────────────
# 들어온 것 읽기 — zip 과 폴더를 같은 모양으로 다룹니다
# ──────────────────────────────────────────────────────────────
class Incoming:
    """`{상대경로: 바이트수}` 와 "그 파일 꺼내기" 만 제공하면 충분합니다."""

    def __init__(self, src: Path):
        self.src = src
        self.zip: zipfile.ZipFile | None = None
        if src.is_dir():
            self.files = {
                f.relative_to(src).as_posix(): f.stat().st_size
                for f in sorted(src.rglob("*")) if f.is_file()
            }
        elif zipfile.is_zipfile(src):
            self.zip = zipfile.ZipFile(src)
            self.files = {i.filename: i.file_size
                          for i in self.zip.infolist() if not i.is_dir()}
        else:
            raise SystemExit(f"❌ zip 도 폴더도 아닙니다: {src}")

    def read(self, rel: str) -> bytes:
        return self.zip.read(rel) if self.zip else (self.src / rel).read_bytes()

    def extract_to(self, rel: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if self.zip:
            with self.zip.open(rel) as fsrc, open(dest, "wb") as fdst:
                shutil.copyfileobj(fsrc, fdst)
        else:
            shutil.copy2(self.src / rel, dest)

    def close(self) -> None:
        if self.zip:
            self.zip.close()


def classify(rel: str) -> tuple[str, str] | None:
    """상대경로 → (종류, 태그/파일명). 모르는 건 None (조용히 무시하지 않고 셉니다)."""
    parts = rel.split("/")
    if parts[0] == "crops" and len(parts) >= 3:
        return "crop", parts[1]
    if parts[0] == "manifests" and rel.endswith(".parquet"):
        return "manifest", parts[-1]
    if parts[0] == "reports":
        return "report", parts[-1]
    return None


# ──────────────────────────────────────────────────────────────
def plan(inc: Incoming, work: Path) -> dict:
    """무엇이 새로 들어오고, 무엇이 이미 있고, 무엇이 부딪히는지."""
    new: dict[str, list[str]] = {}
    same: dict[str, int] = {}
    clash: dict[str, list[str]] = {}
    manifests: list[str] = []
    reports: list[str] = []
    unknown: list[str] = []

    for rel, size in inc.files.items():
        kind = classify(rel)
        if kind is None:
            unknown.append(rel)
            continue
        what, tag = kind
        if what == "manifest":
            manifests.append(tag)
            continue
        if what == "report":
            reports.append(tag)
            continue
        dest = work / rel
        if not dest.exists():
            new.setdefault(tag, []).append(rel)
        elif dest.stat().st_size == size:
            same[tag] = same.get(tag, 0) + 1
        else:
            # 이름은 같은데 크기가 다름 = 크롭 설정이 다릅니다. 섞이면 안 됩니다.
            clash.setdefault(tag, []).append(rel)

    return {"new": new, "same": same, "clash": clash,
            "manifests": sorted(manifests), "reports": sorted(reports),
            "unknown": unknown}


def check_manifest(inc: Incoming, name: str) -> list[str]:
    """가져온 매니페스트가 --finalize 를 통과할 수 있는 모양인지."""
    import io

    import pandas as pd

    notes: list[str] = []
    try:
        df = pd.read_parquet(io.BytesIO(inc.read(f"manifests/{name}")))
    except Exception as exc:                                        # noqa: BLE001
        return [f"❌ 못 읽습니다: {type(exc).__name__}: {exc}"]

    notes.append(f"{len(df):,}행 / 개체 {df['animal_id'].nunique():,}마리"
                 if "animal_id" in df.columns else f"{len(df):,}행")

    # ★ 이 검사가 이 도구의 존재 이유입니다.
    #   phash 가 없으면 --finalize 가 원본 zip 을 다시 읽으려 하는데, 그 zip 은
    #   저쪽 PC 에서 이미 지워졌습니다. 그러면 이 청크는 '중복 없음' 으로 처리돼
    #   청크 경계를 넘는 누수를 못 막습니다.
    if "phash" not in df.columns:
        notes.append("❌ phash 컬럼이 없습니다 — 원본이 있던 PC 에서 재 왔어야 합니다.")
        notes.append("   그쪽에서 `--chunk` 를 다시 돌리거나(원본 필요), 원본 zip 을 같이 가져오세요.")
    else:
        got = int(df["phash"].notna().sum())
        mark = "✅" if got >= len(df) * 0.98 else "❌"
        notes.append(f"{mark} phash {got:,}/{len(df):,} ({got / max(len(df), 1):.1%})")

    if "label" in df.columns:
        top = df["label"].value_counts().to_dict()
        notes.append("라벨: " + ", ".join(f"{k} {v:,}" for k, v in sorted(top.items())))
    return notes


# ──────────────────────────────────────────────────────────────
def apply(inc: Incoming, work: Path, p: dict, force: bool) -> int:
    n = 0
    for tag, rels in sorted(p["new"].items()):
        print(f"  crops/{tag}: {len(rels):,}장 복사 중 …")
        for rel in rels:
            inc.extract_to(rel, work / rel)
            n += 1
    if force:
        for tag, rels in sorted(p["clash"].items()):
            print(f"  ⚠️ crops/{tag}: 부딪힌 {len(rels):,}장을 덮어씁니다 (--force)")
            for rel in rels:
                inc.extract_to(rel, work / rel)
                n += 1
    for name in p["manifests"]:
        if name in NEVER_COPY:
            continue
        inc.extract_to(f"manifests/{name}", work / "manifests" / name)
        print(f"  manifests/{name} ✓")
        n += 1
    for name in p["reports"]:
        dest = work / "reports" / name
        if dest.exists():
            continue                      # 이 PC 의 측정 기록을 덮지 않습니다
        inc.extract_to(f"reports/{name}", dest)
        n += 1
    return n


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", help="가져온 zip 또는 풀어놓은 폴더")
    ap.add_argument("--apply", action="store_true", help="실제로 합칩니다 (기본은 미리보기)")
    ap.add_argument("--force", action="store_true",
                    help="이름이 같고 내용이 다른 크롭까지 덮어씁니다 (권하지 않습니다)")
    a = ap.parse_args(argv)

    from src import env

    work = env.work_root()
    inc = Incoming(Path(a.src).expanduser().resolve())

    print("=" * 68)
    print(f" 가져올 것 : {inc.src}")
    print(f" 합칠 곳   : {work}")
    print("=" * 68)

    p = plan(inc, work)

    print("\n[크롭]")
    tags = sorted(set(p["new"]) | set(p["same"]) | set(p["clash"]))
    if not tags:
        print("  없음")
    for t in tags:
        print(f"  {t:<8} 새로 {len(p['new'].get(t, [])):>7,}장   "
              f"이미 있음 {p['same'].get(t, 0):>7,}장   "
              f"부딪힘 {len(p['clash'].get(t, [])):>5,}장")

    print("\n[매니페스트]")
    if not p["manifests"]:
        print("  ❌ 하나도 없습니다 — 크롭만 가져오면 라벨을 알 수 없습니다.")
    for name in p["manifests"]:
        if name in NEVER_COPY:
            print(f"  {name}: 건너뜁니다 (이 PC 의 분할은 --finalize 가 다시 만듭니다)")
            continue
        print(f"  {name}")
        for line in check_manifest(inc, name):
            print(f"      {line}")

    if p["unknown"]:
        print(f"\n[모르는 파일] {len(p['unknown']):,}개 — 건너뜁니다")
        for rel in p["unknown"][:5]:
            print(f"      {rel}")

    blocked = p["clash"] and not a.force
    if p["clash"]:
        print("\n⚠️ 이름이 같은데 내용이 다른 크롭이 있습니다.")
        print("   크롭 파일 이름은 md5(원본경로) 라, 이름이 같으면 원본이 같습니다.")
        print("   내용이 다르다 = **크롭 설정이 달랐다** 는 뜻입니다. 섞으면 안 됩니다.")
        print("   저쪽 PC 가 `--margins 2.5,-320` 로 돌렸는지 확인하세요.")

    if not a.apply:
        total = sum(len(v) for v in p["new"].values())
        print(f"\n미리보기입니다. 실제로 합치려면 --apply 를 붙이세요 "
              f"(크롭 {total:,}장 + 매니페스트 {len(p['manifests'])}개)")
        inc.close()
        return

    if blocked:
        inc.close()
        raise SystemExit("\n❌ 부딪힌 파일이 있어 멈췄습니다. 확인 후 --force 로만 진행하세요.")

    print("\n[합치는 중]")
    n = apply(inc, work, p, a.force)
    inc.close()

    counts = {}
    crops = work / "crops"
    if crops.exists():
        counts = {d.name: sum(1 for _ in d.rglob("*.jpg"))
                  for d in sorted(crops.iterdir()) if d.is_dir()}
    print(f"\n✅ {n:,}개 반영. 지금 이 PC 의 크롭:")
    for k, v in counts.items():
        print(f"     {k:<8} {v:>8,}장")
    have = sorted(q.name for q in (work / "manifests").glob("chunk_*.parquet"))
    print(f"   청크 매니페스트: {have}")
    print("\n다음: uv run python prepare_local.py --finalize")
    print("      (전 청크를 합쳐 중복제거 + 개체 단위로 다시 나눕니다)")


if __name__ == "__main__":
    main()
