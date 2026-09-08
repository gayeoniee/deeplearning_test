"""폴더 하나를 **캐글 웹 업로드용 zip** 으로 묶습니다 — CLI 키가 없어도 됩니다.

    uv run python tools/zip_for_kaggle.py
    uv run python tools/zip_for_kaggle.py --src data/work/detect --out data/work/dogskin-detect.zip

## 왜 무압축인가

담는 것이 거의 다 JPEG 이라 **이미 압축돼 있습니다.** 다시 압축하면 용량은
1~2% 줄고 시간은 몇 배 듭니다 → `ZIP_STORED`.

## 캐글 쪽 동작

웹 업로더는 올린 zip 을 **자동으로 풉니다.** 그래서 zip 뿌리에 `boxes.parquet`
과 `images/` 를 그대로 두면 데이터셋에도 그 모양으로 놓입니다. 캐글이 폴더를
한 겹 더 싸는 일이 있는데(알려진 함정), 노트북 13 의 1번 셀이 `rglob` 으로
찾으므로 둘 다 괜찮습니다.

⚠️ **공개 범위는 Private** 입니다 — AI Hub 데이터는 재배포 금지입니다.

## 담고 나서 다시 셉니다

크롭이 30%만 올라간 채 조용히 학습된 적이 있습니다. 여기서도 **쓴 개수와
zip 안 항목 수를 대조**하고, 어긋나면 멈춥니다.
"""

from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# dataset-metadata.json 은 CLI 전용입니다 — 웹 업로드에서는 제목을 화면에서
# 입력하므로 담아봐야 데이터셋에 쓰레기 파일 하나가 남을 뿐입니다.
SKIP = {"dataset-metadata.json"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "data" / "work" / "detect")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    src = a.src if a.src.is_absolute() else (ROOT / a.src)
    if not src.is_dir():
        raise SystemExit(f"[X] {src} 가 없습니다.")
    out = a.out or (src.parent / f"dogskin-{src.name}.zip")
    out = out if out.is_absolute() else (ROOT / out)

    files = [p for p in sorted(src.rglob("*")) if p.is_file() and p.name not in SKIP]
    if not files:
        raise SystemExit(f"[X] {src} 에 담을 파일이 없습니다.")
    src_mb = sum(p.stat().st_size for p in files) / 1e6
    print(f"■ {len(files):,}개 · {src_mb/1000:.2f}GB  →  {out}", flush=True)

    free = _free_gb(out.parent)
    if free is not None and free * 1000 < src_mb * 1.05:
        raise SystemExit(f"[X] 빈 공간이 {free:.1f}GB 뿐입니다 "
                         f"({src_mb/1000:.2f}GB 가 필요합니다).")

    t0 = time.perf_counter()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for i, p in enumerate(files, 1):
            z.write(p, p.relative_to(src).as_posix())
            if i % 4000 == 0:
                print(f"   {i:,}/{len(files):,}  "
                      f"{(time.perf_counter()-t0)/60:.1f}분", flush=True)

    with zipfile.ZipFile(out) as z:
        names = z.namelist()
    gb = out.stat().st_size / 1e9
    print(f"\n■ {out.name}  {gb:.2f}GB  항목 {len(names):,}개  "
          f"{(time.perf_counter()-t0)/60:.1f}분")
    if len(names) != len(files):
        raise SystemExit(f"[X] {len(files):,}개를 담으려 했는데 {len(names):,}개입니다.")
    print("■ 개수 일치")

    print("\n다음 — https://www.kaggle.com/datasets → **New Dataset** → 이 zip")
    print("⚠️ 공개 범위를 Private 에서 바꾸지 마세요 — AI Hub 데이터는 재배포 금지입니다.")
    print("⚠️ 캐글이 zip 을 자동으로 풉니다. 그 데이터셋을 노트북 13 에 Add input 하세요.")


def _free_gb(p: Path) -> float | None:
    try:
        import shutil
        return shutil.disk_usage(p).free / 1e9
    except OSError:
        return None


if __name__ == "__main__":
    main()
