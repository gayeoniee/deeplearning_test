"""병변 ROI 크롭.

선행 프로젝트 보고: **병변이 이미지의 5% 미만인 경우가 93%**.
이 상태로 전체 이미지를 넣으면 모델은 병변이 아니라 바닥재·목줄·조명을 학습합니다.
(그리고 그게 클래스와 우연히 상관이 있으면 검증 점수까지 잘 나와서 속게 됩니다.)

그래서 bbox/polygon 주변만 잘라 512px 로 저장하고, 학습은 이 크롭본으로 합니다.
부수 효과로 용량도 크게 줄어 Colab/Drive 에 얹기 쉬워집니다.

    from src import crop
    df2 = crop.run(df, margin=1.5)          # 크롭 생성 + crop_path 컬럼 추가
    crop.preview(df2, n=8)                  # 눈으로 확인 (필수!)
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from src import env
from src.config import CFG


def expand_box(
    bbox: list[float], img_w: int, img_h: int, margin: float = 1.5,
    square: bool = True, min_px: int = 64,
) -> tuple[int, int, int, int]:
    """박스를 margin 배로 넓히고 정사각형으로 만든 뒤 이미지 안으로 잘라냅니다.

    정사각형으로 만드는 이유: 나중에 리사이즈할 때 종횡비가 안 찌그러집니다.
    병변의 모양(둥근 구진 vs 길쭉한 미란)이 라벨의 단서라 왜곡이 치명적입니다.
    """
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)

    if square:
        side = max(w, h) * margin
        w = h = side
    else:
        w, h = w * margin, h * margin

    w = max(w, min_px)
    h = max(h, min_px)

    nx1, ny1 = cx - w / 2, cy - h / 2
    nx2, ny2 = cx + w / 2, cy + h / 2

    # 경계를 벗어나면 중심을 밀어 넣습니다 (크기를 줄이는 대신 이동 → 왜곡 방지)
    if nx1 < 0:
        nx2 -= nx1; nx1 = 0
    if ny1 < 0:
        ny2 -= ny1; ny1 = 0
    if nx2 > img_w:
        nx1 -= nx2 - img_w; nx2 = img_w
    if ny2 > img_h:
        ny1 -= ny2 - img_h; ny2 = img_h

    return (max(int(nx1), 0), max(int(ny1), 0),
            min(int(round(nx2)), img_w), min(int(round(ny2)), img_h))


def _out_path(src: str, out_dir: Path, tag: str) -> Path:
    """원본 경로를 해시해 충돌 없는 출력 파일명을 만듭니다."""
    import hashlib

    h = hashlib.md5(src.encode()).hexdigest()[:12]
    return out_dir / tag / h[:2] / f"{Path(src).stem}_{h}.jpg"


_ZIP_CACHE: dict = {}


def _open_source(row):
    """이미지를 엽니다. zip 안에 있으면 풀지 않고 바로 읽습니다.

    zip 핸들은 스레드마다 따로 열어 캐시합니다 (ZipFile 은 스레드 안전하지 않음).
    """
    import io
    import threading
    import zipfile

    from PIL import Image

    member = row.get("zip_member")
    if not isinstance(member, str) or not member:
        return Image.open(row["image_path"])

    key = (threading.get_ident(), row["zip_path"])
    zf = _ZIP_CACHE.get(key)
    if zf is None:
        zf = zipfile.ZipFile(row["zip_path"])
        _ZIP_CACHE[key] = zf
    return Image.open(io.BytesIO(zf.read(member)))


def _crop_one(args) -> tuple[int, str | None]:
    i, row, out_dir, tag, cfg = args
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    dst = _out_path(row["image_path"], out_dir, tag)
    if dst.exists():
        return i, str(dst)

    try:
        with _open_source(row) as im:
            im = im.convert("RGB")
            W, H = im.size
            bbox = row.get("bbox")
            if isinstance(bbox, str):
                bbox = json.loads(bbox)

            if bbox and cfg.crop_margin > 0:
                x1, y1, x2, y2 = expand_box(
                    bbox, W, H, margin=cfg.crop_margin, min_px=cfg.crop_min_px
                )
                if x2 - x1 < 8 or y2 - y1 < 8:
                    return i, None
                im = im.crop((x1, y1, x2, y2))
            else:
                # 박스가 없으면 중앙 정사각 크롭 (전체이미지 실험군)
                s = min(W, H)
                im = im.crop(((W - s) // 2, (H - s) // 2, (W + s) // 2, (H + s) // 2))

            s = cfg.save_crop_size
            if max(im.size) > s:
                im.thumbnail((s, s), Image.LANCZOS)
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, "JPEG", quality=cfg.save_crop_quality, optimize=True)
        return i, str(dst)
    except Exception:
        return i, None


def run(
    df: pd.DataFrame,
    cfg: CFG | None = None,
    margin: float | None = None,
    tag: str | None = None,
    workers: int = 8,
    verbose: bool = True,
) -> pd.DataFrame:
    """크롭을 만들고 `crop_path` 컬럼을 붙입니다.

    margin 을 바꿔가며 여러 번 부르면 tag 별로 따로 저장되어
    STEP 4 에서 "어느 크롭이 제일 좋은가"를 실험할 수 있습니다.
    """
    cfg = cfg or CFG()
    if margin is not None:
        cfg = CFG(**{**cfg.to_dict(), "crop_margin": margin})
    tag = tag or (f"m{cfg.crop_margin:g}" if cfg.crop_margin > 0 else "full")

    out_dir = env.ensure_dirs()["crops"]
    if verbose:
        print(f"[crop] margin={cfg.crop_margin} tag={tag} → {out_dir / tag}")
        n_box = df["bbox"].notna().sum()
        print(f"[crop] bbox 있는 행 {n_box:,}/{len(df):,} "
              f"({n_box / max(len(df), 1):.1%}) — 없는 행은 중앙 크롭")

    tasks = [(i, r, out_dir, tag, cfg) for i, r in df.iterrows()]
    results: dict[int, str | None] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, p in tqdm(ex.map(_crop_one, tasks), total=len(tasks), desc=f"crop:{tag}"):
            results[i] = p

    out = df.copy()
    out["crop_path"] = out.index.map(results)
    out["crop_tag"] = tag
    # ⚠️ 절대경로는 다른 기기(로컬 → Drive → Colab)로 옮기면 전부 깨집니다.
    #    크롭 루트 기준 상대경로를 함께 저장하고, 로드할 때 현재 경로로 다시 붙입니다.
    out["crop_rel"] = out["crop_path"].apply(
        lambda p: str(Path(p).relative_to(out_dir)) if isinstance(p, str) else None
    )

    failed = out["crop_path"].isna().sum()
    if verbose:
        size = sum(f.stat().st_size for f in (out_dir / tag).rglob("*.jpg")) / 1024**3
        print(f"[crop] 완료 — 실패 {failed:,}건, 저장 용량 {size:.2f} GB")
    if failed:
        out = out[out["crop_path"].notna()].reset_index(drop=True)
        if verbose:
            print(f"[crop] 실패분 제외 후 {len(out):,}행")
    return out


def margin_of_tag(tag: str) -> float:
    """크롭 태그에서 margin 을 되돌립니다. 'm1.5' → 1.5, 'full' → 0."""
    if not isinstance(tag, str) or not tag.startswith("m"):
        return 0.0
    try:
        return float(tag[1:])
    except ValueError:
        return 0.0


def bbox_in_crop(row, tag: str | None = None, cfg: CFG | None = None) -> list[float] | None:
    """병변 bbox 를 **크롭 이미지 기준 정규화 좌표(0~1)** 로 옮깁니다.

    크롭은 원본을 열지 않고도 재현할 수 있습니다 — `expand_box()` 가
    (bbox, img_w, img_h, margin) 만으로 결정되기 때문입니다.

    이게 필요한 이유: 학습은 크롭만 올린 Colab 에서 하는데, Grad-CAM 게이트는
    "CAM 이 병변 위에 있는가"를 재야 합니다. 원본이 없으니 크롭 좌표계로
    병변 위치를 다시 계산해야 합니다. 정규화 좌표라 저장 해상도와 무관합니다.

    돌려주는 값: [x1, y1, x2, y2] (0~1). 계산 불가면 None.
    """
    cfg = cfg or CFG()
    bbox = row.get("bbox")
    if isinstance(bbox, str):
        try:
            bbox = json.loads(bbox)
        except Exception:
            return None
    if not bbox or len(bbox) != 4:
        return None

    try:
        W, H = int(row["img_w"]), int(row["img_h"])
    except (KeyError, TypeError, ValueError):
        return None
    if W <= 0 or H <= 0:
        return None

    tag = tag or row.get("crop_tag") or ""
    margin = margin_of_tag(tag)

    if margin > 0:
        wx1, wy1, wx2, wy2 = expand_box(bbox, W, H, margin=margin, min_px=cfg.crop_min_px)
    else:
        # 'full' 태그는 중앙 정사각 크롭이었습니다 (_crop_one 의 else 분기)
        s = min(W, H)
        wx1, wy1 = (W - s) // 2, (H - s) // 2
        wx2, wy2 = wx1 + s, wy1 + s

    ww, wh = wx2 - wx1, wy2 - wy1
    if ww <= 0 or wh <= 0:
        return None

    rel = [(bbox[0] - wx1) / ww, (bbox[1] - wy1) / wh,
           (bbox[2] - wx1) / ww, (bbox[3] - wy1) / wh]
    rel = [min(max(v, 0.0), 1.0) for v in rel]
    if rel[2] - rel[0] <= 0 or rel[3] - rel[1] <= 0:
        return None            # 크롭 밖으로 완전히 밀려난 경우 (full 태그에서 발생 가능)
    return rel


def available_tags() -> list[str]:
    """만들어져 있는 크롭 태그 목록 (예: ['full', 'm1.5', 'm2.5'])."""
    d = env.work_root() / "crops"
    return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.exists() else []


def switch_tag(df: pd.DataFrame, tag: str, verbose: bool = True) -> pd.DataFrame:
    """같은 매니페스트를 다른 크롭 버전으로 갈아 끼웁니다.

    크롭 파일명은 원본 경로의 해시로 정해지므로, 태그만 바꿔 경로를 다시 계산하면
    됩니다. 크롭 방식(margin 1.5 vs 2.5 vs full)을 비교 실험할 때 씁니다.
    매니페스트를 태그별로 여러 벌 들고 다닐 필요가 없습니다.
    """
    out_dir = env.work_root() / "crops"
    out = df.copy()
    out["crop_path"] = out["image_path"].apply(lambda p: str(_out_path(p, out_dir, tag)))
    out["crop_rel"] = out["crop_path"].apply(lambda p: str(Path(p).relative_to(out_dir)))
    out["crop_tag"] = tag

    exists = out["crop_path"].apply(lambda p: Path(p).exists())
    if verbose:
        print(f"[crop] 태그 '{tag}' 로 전환 — {exists.sum():,}/{len(out):,}장 존재")
    if not exists.all():
        miss = int((~exists).sum())
        print(f"⚠️ {miss:,}장이 없습니다. 이 태그의 크롭이 안 만들어졌을 수 있습니다.")
        print(f"   사용 가능한 태그: {available_tags()}")
        out = out[exists].reset_index(drop=True)
    return out


def preview(df: pd.DataFrame, n: int = 8, by_class: bool = True, seed: int = 0) -> None:
    """크롭 결과를 눈으로 확인합니다.

    ⚠️ 이 단계를 건너뛰지 마세요. 좌표계가 뒤집혔거나(x/y 교환),
       스케일이 다르거나(정규화 좌표를 픽셀로 착각), 박스가 엉뚱한 곳을 가리키는
       실수는 **그림을 봐야만** 발견됩니다. 숫자로는 절대 안 보입니다.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    if by_class and "label" in df.columns:
        picks = (df.dropna(subset=["crop_path"])
                   .groupby("label", group_keys=False)
                   .apply(lambda g: g.sample(min(len(g), max(n // max(df['label'].nunique(), 1), 1)),
                                             random_state=seed)))
        picks = picks.head(n * 2)
    else:
        picks = df.dropna(subset=["crop_path"]).sample(min(n, len(df)), random_state=seed)

    k = len(picks)
    if k == 0:
        print("보여줄 크롭이 없습니다.")
        return
    cols = min(4, k)
    rows = (k + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows))
    axes = [axes] if k == 1 else list(axes.flat)

    for ax, (_, r) in zip(axes, picks.iterrows()):
        try:
            ax.imshow(Image.open(r["crop_path"]))
        except Exception as exc:
            ax.text(0.5, 0.5, str(exc)[:40], ha="center")
        ar = r.get("area_ratio")
        ax.set_title(f"{r.get('label')}  area={ar:.1%}" if pd.notna(ar) else str(r.get("label")),
                     fontsize=9)
        ax.axis("off")
    for ax in axes[k:]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def preview_with_box(df: pd.DataFrame, n: int = 4, seed: int = 0) -> None:
    """원본 이미지 위에 bbox/polygon 을 그려 좌표계가 맞는지 확인합니다."""
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from PIL import Image

    picks = df[df["bbox"].notna()].sample(min(n, int(df["bbox"].notna().sum())),
                                          random_state=seed)
    fig, axes = plt.subplots(1, len(picks), figsize=(4.2 * len(picks), 4.4))
    axes = [axes] if len(picks) == 1 else list(axes)

    for ax, (_, r) in zip(axes, picks.iterrows()):
        with Image.open(r["image_path"]) as im:
            ax.imshow(im.convert("RGB"))
        b = r["bbox"]
        if isinstance(b, str):
            b = json.loads(b)
        ax.add_patch(patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                       fill=False, lw=2, edgecolor="red"))
        poly = r.get("polygon")
        if isinstance(poly, str):
            poly = json.loads(poly)
        if poly:
            ax.add_patch(patches.Polygon(poly, fill=False, lw=1.5, edgecolor="yellow"))
        ax.set_title(f"{r.get('label')}  {r.get('img_w')}x{r.get('img_h')}", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    print("빨강=bbox, 노랑=polygon. 병변 위에 정확히 얹혀 있어야 합니다.")
    print("어긋나 있다면 labels.extract_geometry 의 좌표 해석이 틀린 것입니다.")
