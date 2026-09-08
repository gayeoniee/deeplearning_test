"""검출 학습용 데이터셋 만들기 — **앱처럼 찍은 사진 + 정답 네모**.

    uv run python tools/make_detect_dataset.py --n 23000 --px 512
    uv run python tools/make_detect_dataset.py --n 200 --px 512   # 먼저 작게

## 왜 따로 만드나

캐글에 올라간 데이터셋에는 **크롭만** 있습니다 (`m2.5` · `f320`). 그런데
크롭은 전부 **병변을 가운데 놓고 자른 것**이라 검출 학습에 못 씁니다 —
정답이 항상 한가운데라 **모델이 "가운데" 만 외우면 됩니다** (STEP 36 의 1차
측정에서 사람에게 같은 함정을 놨었습니다).

그래서 원본 zip 에서 **앱처럼 가까이 찍은 사진**을 만들고 그 좌표계의 정답
네모를 같이 저장합니다. `tools/box_error.py` 와 **같은 방식**입니다:

    창 = bbox 긴 변 × U(3.5, 6.0), 최소 480px, 사진 밖으로 안 나감
    창 위치는 **무작위** — 병변이 가운데 고정이면 잴 것이 없어집니다

## 무엇이 나오나

    data/work/detect/images/000000.jpg ...      (px × px)
    data/work/detect/boxes.parquet              path · x1 y1 x2 y2 (0~1) · label · group

⚠️ **AI Hub 재배포 금지** — 캐글 업로드는 **반드시 Private** 입니다
(`kaggle datasets create` 기본값이 private, `--public` 을 붙이지 마세요).

⚠️ `group` 을 같이 저장합니다. 캐글에서 **개체 단위로 갈라야** 평가가
거짓말을 안 합니다.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=23000)
    ap.add_argument("--px", type=int, default=512, help="저장 해상도 (정사각)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--quality", type=int, default=88)
    a = ap.parse_args()

    import pandas as pd
    from PIL import Image

    from src import crop, env

    out = a.out or (env.work_root() / "detect")
    imgs = out / "images"
    imgs.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(env.work_root() / "manifests" / "manifest_final.parquet")
    df = df[df.get("label_orig", df.get("label")) != "A7"]
    df = df[df["zip_path"].notna() & df["zip_member"].notna() & df["bbox"].notna()]
    if df.empty:
        raise SystemExit("[X] 원본 zip 경로가 있는 행이 없습니다. "
                         "`data/raw/<청크>/**.zip` 이 남아 있어야 합니다.")
    gcol = "group" if "group" in df.columns else "animal_id"
    df = df.sample(min(len(df), a.n), random_state=a.seed)
    print(f"■ 후보 {len(df):,}장 · 저장 {a.px}px · → {out}")

    rng = random.Random(a.seed)
    zips: dict[str, zipfile.ZipFile] = {}
    rows, t0 = [], time.perf_counter()
    try:
        for i, (_, r) in enumerate(df.iterrows()):
            zp = str(r["zip_path"])
            if zp not in zips:
                if not Path(zp).is_file():
                    continue
                zips[zp] = zipfile.ZipFile(zp)
            try:
                blob = zips[zp].read(str(r["zip_member"]))
            except KeyError:
                continue
            full = Image.open(io.BytesIO(blob)).convert("RGB")
            FW, FH = full.size
            bb = r["bbox"]
            b0 = crop._box4(json.loads(bb) if isinstance(bb, str) else bb)
            if b0 is None:
                full.close()
                continue
            long0 = max(b0[2] - b0[0], b0[3] - b0[1])
            if long0 <= 0:
                full.close()
                continue
            # ★ `box_error.py` 와 **같은 자극** — 앱처럼 가까이, 위치는 무작위
            side = min(max(long0 * rng.uniform(3.5, 6.0), 480.0), FW, FH)
            if side < long0 * 1.6:
                full.close()
                continue
            lo_x, hi_x = max(0.0, b0[2] - side), min(b0[0], FW - side)
            lo_y, hi_y = max(0.0, b0[3] - side), min(b0[1], FH - side)
            if lo_x > hi_x or lo_y > hi_y:
                full.close()
                continue
            wx, wy = rng.uniform(lo_x, hi_x), rng.uniform(lo_y, hi_y)
            im = full.crop((int(wx), int(wy), int(wx + side), int(wy + side)))
            full.close()                      # ⚠️ 안 닫으면 메모리가 터집니다
            im2 = im.resize((a.px, a.px))
            im.close()
            name = f"{len(rows):06d}.jpg"
            im2.save(imgs / name, quality=a.quality)
            im2.close()
            # 정답 네모를 **그 창 기준 0~1** 로 (x1,y1,x2,y2)
            t = [(b0[0] - wx) / side, (b0[1] - wy) / side,
                 (b0[2] - wx) / side, (b0[3] - wy) / side]
            if not all(0.0 <= v <= 1.0 for v in t) or t[2] <= t[0] or t[3] <= t[1]:
                (imgs / name).unlink(missing_ok=True)
                continue
            rows.append({"image": name, "x1": t[0], "y1": t[1], "x2": t[2], "y2": t[3],
                         "label": str(r["label"]), "group": str(r[gcol])})
            if (i + 1) % 2000 == 0:
                print(f"   {i+1:,}/{len(df):,}  담은 것 {len(rows):,}  "
                      f"{(time.perf_counter()-t0)/60:.1f}분", flush=True)
    finally:
        for z in zips.values():
            z.close()

    if not rows:
        raise SystemExit("[X] 담긴 것이 없습니다.")
    d = pd.DataFrame(rows)
    d.to_parquet(out / "boxes.parquet", index=False)
    mb = sum(p.stat().st_size for p in imgs.iterdir()) / 1e6
    print(f"\n■ {len(d):,}장 · {mb/1000:.2f}GB · {(time.perf_counter()-t0)/60:.1f}분")
    print(f"■ 개체 {d['group'].nunique():,} · 클래스 {d['label'].nunique()}종")
    print(f"\n다음 — 캐글에 **Private** 으로 올립니다:")
    print(f"    cd {out}")
    print( "    kaggle datasets init -p .")
    print( "    # dataset-metadata.json 의 title/id 를 채우고")
    print( "    kaggle datasets create -p . --dir-mode zip")
    print( "\n⚠️ `--public` 을 붙이지 마세요 — AI Hub 데이터는 재배포 금지입니다.")


if __name__ == "__main__":
    main()
