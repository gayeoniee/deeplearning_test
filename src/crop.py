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
    # ⚠️ as_posix() 로 저장합니다. Windows 의 str() 은 역슬래시를 넣는데,
    #    그 매니페스트를 Colab(리눅스)에서 읽으면 경로가 통째로 깨집니다.
    out["crop_rel"] = out["crop_path"].apply(
        lambda p: Path(p).relative_to(out_dir).as_posix() if isinstance(p, str) else None
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


def _as_list(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return None
    return v


def crop_window(row, tag: str | None = None, cfg: CFG | None = None):
    """크롭이 원본에서 어느 사각형을 잘라냈는지 되돌립니다.

    원본을 열지 않고 계산할 수 있습니다 — `expand_box()` 가
    (bbox, img_w, img_h, margin) 만으로 결정되기 때문입니다.
    돌려주는 값: (x1, y1, x2, y2) 원본 픽셀 좌표. 계산 불가면 None.
    """
    cfg = cfg or CFG()
    bbox = _as_list(row.get("bbox"))
    try:
        W, H = int(row["img_w"]), int(row["img_h"])
    except (KeyError, TypeError, ValueError):
        return None
    if W <= 0 or H <= 0:
        return None

    margin = margin_of_tag(tag or row.get("crop_tag") or "")
    if margin > 0 and bbox and len(bbox) == 4:
        return expand_box(bbox, W, H, margin=margin, min_px=cfg.crop_min_px)

    # margin 이 없거나 bbox 가 없으면 중앙 정사각 크롭이었습니다 (_crop_one 의 else 분기)
    s = min(W, H)
    x1, y1 = (W - s) // 2, (H - s) // 2
    return (x1, y1, x1 + s, y1 + s)


def geometry_in_crop(row, tag: str | None = None, cfg: CFG | None = None) -> dict:
    """병변 bbox/polygon 을 **크롭 이미지 기준 정규화 좌표(0~1)** 로 옮깁니다.

    이게 필요한 이유: 학습은 크롭만 올린 Colab 에서 하는데,
    "크롭이 병변을 담고 있는가"(눈으로 확인)와 "CAM 이 병변 위에 있는가"(게이트)를
    둘 다 재야 합니다. 원본이 없으니 크롭 좌표계로 병변 위치를 다시 계산합니다.
    정규화 좌표라 저장 해상도와 무관합니다.

    돌려주는 값: {"bbox": [x1,y1,x2,y2] | None, "polygon": [[x,y],...] | None}
    """
    win = crop_window(row, tag, cfg)
    if win is None:
        return {"bbox": None, "polygon": None}
    wx1, wy1, wx2, wy2 = win
    ww, wh = wx2 - wx1, wy2 - wy1
    if ww <= 0 or wh <= 0:
        return {"bbox": None, "polygon": None}

    def clip01(v):
        return min(max(v, 0.0), 1.0)

    out: dict = {"bbox": None, "polygon": None}

    bbox = _as_list(row.get("bbox"))
    if bbox and len(bbox) == 4:
        rel = [clip01((bbox[0] - wx1) / ww), clip01((bbox[1] - wy1) / wh),
               clip01((bbox[2] - wx1) / ww), clip01((bbox[3] - wy1) / wh)]
        # 크롭 밖으로 완전히 밀려난 경우 (full 태그에서 발생 가능)
        if rel[2] - rel[0] > 0 and rel[3] - rel[1] > 0:
            out["bbox"] = rel

    poly = _as_list(row.get("polygon"))
    if poly:
        try:
            pts = [[clip01((p[0] - wx1) / ww), clip01((p[1] - wy1) / wh)] for p in poly]
            if len(pts) >= 3:
                out["polygon"] = pts
        except (TypeError, IndexError):
            pass
    return out


def bbox_in_crop(row, tag: str | None = None, cfg: CFG | None = None) -> list[float] | None:
    """`geometry_in_crop()` 의 bbox 만. (0~1 정규화, 계산 불가면 None)"""
    return geometry_in_crop(row, tag, cfg)["bbox"]


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
    out["crop_rel"] = out["crop_path"].apply(
        lambda p: Path(p).relative_to(out_dir).as_posix())
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


def _blur_score(pil) -> float:
    """초점이 맞았는지 숫자로. 라플라시안 분산 — 클수록 선명합니다.

    "이 사진이 병변인가"는 수의사만 알지만, "이 사진이 흐린가"는 숫자가 압니다.
    사람에게 의학 판단을 요구하지 않고 데이터 품질을 재는 방법입니다.
    """
    import numpy as np

    g = np.asarray(pil.convert("L"), dtype=np.float32)
    if g.size < 9:
        return 0.0
    # 3x3 라플라시안을 직접 적용 (cv2 없이)
    lap = (4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])
    return float(lap.var())


def audit(df: pd.DataFrame, n_sample: int = 400, seed: int = 0,
          cfg: CFG | None = None, blur_threshold: float = 100.0) -> dict:
    """크롭 품질을 **숫자로** 점검합니다. 의학 지식이 필요 없습니다.

    "이게 병변인지 눈으로 보세요" 는 비전문가에게 할 수 없는 요구입니다.
    대신 코드가 확인할 수 있는 것들을 확인합니다:

      1. 정상(A7)과 병변의 **크롭 배율이 다른가**  ← 가장 위험
      2. 크롭 원본 창이 너무 작아 확대되고 있는가
      3. 초점이 안 맞은 사진이 얼마나 되는가
      4. 같은 사진이 여러 라벨을 갖고 있는가
      5. 박스가 이미지 밖을 가리키는가
    """
    import numpy as np
    from PIL import Image

    from src.config import CLASS_KO, NORMAL_LABEL

    cfg = cfg or CFG()
    out: dict = {}
    print("=" * 68)
    print(" 크롭 감사 — 숫자로 확인 (의학 지식 불필요)")
    print("=" * 68)

    # ── 1. 라벨별 병변 면적 비율: 크롭 배율이 라벨을 흘리는지 ────────
    print("\n[1] 라벨별 bbox 면적 비율 — ★ 가장 중요한 점검")
    print("    정상과 병변의 배율이 다르면 모델이 피부가 아니라 '확대 정도'를 학습합니다.")
    has_box = df["bbox"].notna()
    out["bbox_coverage"] = {}
    rows = []
    for lab, g in df.groupby("label"):
        ar = g["area_ratio"].dropna()
        frac_box = float(g["bbox"].notna().mean())
        out["bbox_coverage"][lab] = frac_box
        rows.append({
            "label": f"{lab} {CLASS_KO.get(lab, '')}"[:22],
            "n": len(g),
            "bbox있음": f"{frac_box:.1%}",
            "면적_중앙값": f"{ar.median():.2%}" if len(ar) else "-",
            "면적_25%": f"{ar.quantile(.25):.2%}" if len(ar) else "-",
            "면적_75%": f"{ar.quantile(.75):.2%}" if len(ar) else "-",
        })
    print(pd.DataFrame(rows).to_string(index=False))

    med = df.groupby("label")["area_ratio"].median()
    if NORMAL_LABEL in med.index and len(med) > 1:
        m_norm = med[NORMAL_LABEL]
        m_les = med.drop(NORMAL_LABEL).median()
        out["area_median_normal"] = float(m_norm)
        out["area_median_lesion"] = float(m_les)
        ratio = m_norm / max(m_les, 1e-9)
        out["area_ratio_normal_over_lesion"] = float(ratio)
        print(f"\n    정상 중앙값 {m_norm:.2%}  vs  병변 중앙값 {m_les:.2%}  →  {ratio:.2f}배")
        if ratio > 1.5 or ratio < 0.67:
            print("    🚨 배율이 크게 다릅니다. 크롭만 봐도 정상/병변이 티가 납니다.")
            print("       → 1단계가 피부가 아니라 '확대 정도'로 맞힐 수 있습니다.")
            print("       → 대응: 1단계는 'full' 크롭으로 학습하세요 (배율 정보를 없앰).")
        else:
            print("    ✅ 배율이 비슷합니다. 이 경로의 지름길은 없어 보입니다.")

    # ── 2. 크롭 원본 창 크기 ──────────────────────────────────────
    print(f"\n[2] 크롭 창 크기 — 학습 입력은 {cfg.img_size}px 입니다")
    sub = df[has_box].sample(min(n_sample * 4, int(has_box.sum())), random_state=seed)
    sides = []
    for _, r in sub.iterrows():
        w = crop_window(r, cfg=cfg)
        if w:
            sides.append(min(w[2] - w[0], w[3] - w[1]))
    if sides:
        s = np.array(sides)
        upscaled = float((s < cfg.img_size).mean())
        out["crop_side_median"] = float(np.median(s))
        out["frac_upscaled"] = upscaled
        print(f"    짧은 변 중앙값 {np.median(s):.0f}px  "
              f"(25% {np.quantile(s, .25):.0f} / 75% {np.quantile(s, .75):.0f})")
        print(f"    {cfg.img_size}px 미만 = 확대해서 씀: {upscaled:.1%}")
        if upscaled > 0.5:
            print("    ⚠️ 절반 이상이 확대됩니다 — 없는 디테일을 만들어내는 셈입니다.")
            print("       흐릿하게 보이는 게 정상입니다. margin 을 키우거나 img_size 를 낮추세요.")

    # ── 3. 초점 ─────────────────────────────────────────────────
    print("\n[3] 초점 (라플라시안 분산, 클수록 선명)")
    picks = df.dropna(subset=["crop_path"]).sample(min(n_sample, len(df)), random_state=seed)
    scores: dict[str, list[float]] = {}
    for _, r in picks.iterrows():
        try:
            with Image.open(r["crop_path"]) as im:
                scores.setdefault(str(r.get("label")), []).append(_blur_score(im))
        except Exception:
            continue
    allv = np.array([v for vs in scores.values() for v in vs])
    if len(allv):
        out["blur_median"] = float(np.median(allv))
        out["frac_blurry"] = float((allv < blur_threshold).mean())
        print(f"    전체 중앙값 {np.median(allv):.0f}  |  "
              f"{blur_threshold:.0f} 미만(흐림) {out['frac_blurry']:.1%}  (n={len(allv)})")
        bl = pd.DataFrame([{"label": k, "n": len(v), "중앙값": f"{np.median(v):.0f}",
                            "흐림비율": f"{(np.array(v) < blur_threshold).mean():.1%}"}
                           for k, v in sorted(scores.items())])
        print(bl.to_string(index=False))
        bym = {k: float(np.median(v)) for k, v in scores.items()}
        if NORMAL_LABEL in bym and len(bym) > 1:
            others = np.median([v for k, v in bym.items() if k != NORMAL_LABEL])
            r_blur = bym[NORMAL_LABEL] / max(others, 1e-9)
            out["blur_ratio_normal_over_lesion"] = float(r_blur)
            if r_blur > 1.6 or r_blur < 0.62:
                print(f"    🚨 선명도가 계통적으로 다릅니다 — "
                      f"정상 {bym[NORMAL_LABEL]:.0f} vs 병변 {others:.0f}")
                print("       모델이 피부가 아니라 '사진 화질'로 맞힐 수 있습니다.")
        if out["frac_blurry"] > 0.3:
            print("    ⚠️ 흐린 사진이 많습니다. 실사용에서는 흐린 사진을 거절하는 게 맞습니다")
            print("       (노트북 05 의 거절 임계값 + '다시 찍어주세요' 문구가 그 역할).")

    # ── 4. 라벨 충돌 ────────────────────────────────────────────
    print("\n[4] 같은 사진에 여러 라벨")
    if "image_name" in df.columns:
        per = df.groupby("image_name")["label"].nunique()
        n_conf = int((per > 1).sum())
        out["conflicting_images"] = n_conf
        print(f"    라벨이 2개 이상인 파일명: {n_conf:,}건 " + ("✅" if n_conf == 0 else "🚨"))
        if n_conf:
            print("    → 파일명이 Training/Validation 에서 재사용된 것일 수 있습니다.")
            print(f"       예: {list(per[per > 1].index[:3])}")

    # ── 5. 박스가 이미지 밖 ─────────────────────────────────────
    print("\n[5] 박스가 이미지 경계를 벗어남")
    bad = 0
    for _, r in df[has_box].iterrows():
        b = _as_list(r.get("bbox"))
        w, h = r.get("img_w"), r.get("img_h")
        if not b or not w or not h:
            continue
        if b[0] < -1 or b[1] < -1 or b[2] > w + 1 or b[3] > h + 1 or b[2] <= b[0] or b[3] <= b[1]:
            bad += 1
    out["boxes_out_of_bounds"] = bad
    print(f"    경계 이탈/역전 박스: {bad:,}건 " + ("✅" if bad == 0 else "🚨 좌표 해석 오류 의심"))

    print("\n" + "=" * 68)
    return out


def contact_sheet(df: pd.DataFrame, per_class: int = 6, seed: int = 0,
                  path_col: str = "crop_path") -> None:
    """클래스별로 한 줄씩 나란히 놓습니다.

    여기서 판단할 것은 **"이게 병변인가"가 아닙니다.** 그건 수의사의 일입니다.
    비전문가가 할 수 있고, 해야 하는 판단은 이것뿐입니다:

      · 줄마다 **서로 달라 보이는가** — 다 똑같아 보이면 모델도 구분 못 합니다
      · 한 줄 안에서는 **비슷해 보이는가** — 제각각이면 라벨이 섞였을 수 있습니다
      · 개 피부/털이 맞는가 — 사람 손, 바닥, 진료대만 보이면 크롭이 어긋난 것
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    from src.config import CLASS_KO

    labs = sorted(df["label"].dropna().unique())
    if not labs:
        print("라벨이 없습니다.")
        return

    fig, axes = plt.subplots(len(labs), per_class,
                             figsize=(2.1 * per_class, 2.3 * len(labs)),
                             squeeze=False)
    axes = np.asarray(axes)

    for i, lab in enumerate(labs):
        g = df[(df["label"] == lab) & df[path_col].notna()]
        picks = g.sample(min(per_class, len(g)), random_state=seed)
        for j in range(per_class):
            ax = axes[i][j]
            ax.axis("off")
            if j >= len(picks):
                continue
            try:
                with Image.open(picks.iloc[j][path_col]) as im:
                    ax.imshow(im.convert("RGB"))
            except Exception:
                ax.text(0.5, 0.5, "열기 실패", ha="center", fontsize=7)
        axes[i][0].set_ylabel(lab)
        axes[i][0].axis("on")
        axes[i][0].set_xticks([]); axes[i][0].set_yticks([])
        axes[i][0].set_ylabel(f"{lab}\n{CLASS_KO.get(lab, '')}", fontsize=7, rotation=0,
                              ha="right", va="center", labelpad=34)

    plt.tight_layout()
    plt.show()
    print("여기서 볼 것 (의학 지식 불필요):")
    print("  1. 줄마다 서로 달라 보입니까?  전부 똑같아 보이면 → 모델도 구분 못 합니다")
    print("  2. 한 줄 안에서는 비슷합니까?  제각각이면 → 라벨이 섞였을 수 있습니다")
    print("  3. 개 피부/털이 맞습니까?      사람 손·바닥만 보이면 → 크롭이 어긋난 것")


def preview_with_box(df: pd.DataFrame, n: int = 4, seed: int = 0,
                     frame: str = "auto", cfg: CFG | None = None) -> None:
    """bbox/polygon 을 이미지 위에 그려 좌표계가 맞는지 확인합니다.

    frame:
      "original" — 원본 이미지 위에. 원본이 있는 환경(로컬)에서 가장 직관적입니다.
      "crop"     — **크롭 이미지 위에.** 원본을 안 올린 Colab 용입니다.
                   크롭 창을 되돌려(`geometry_in_crop`) 박스를 다시 그립니다.
      "auto"     — 원본을 열어보고 안 되면 crop 으로 내려갑니다 (기본).
    """
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from PIL import Image

    cfg = cfg or CFG()
    sub = df[df["bbox"].notna()]
    if not len(sub):
        print("bbox 가 있는 행이 없습니다 — 보여줄 게 없습니다.")
        return

    if frame == "auto":
        frame = "original"
        probe = sub.head(20)
        if not any(isinstance(p, str) and Path(p).exists() for p in probe["image_path"]):
            frame = "crop"
            print("[crop] 원본 이미지가 없어 **크롭 이미지 위에** 박스를 그립니다.")
            print("       (로컬에서 만든 크롭만 업로드한 환경이면 정상입니다)")

    picks = sub.sample(min(n, len(sub)), random_state=seed)
    fig, axes = plt.subplots(1, len(picks), figsize=(4.2 * len(picks), 4.6))
    axes = [axes] if len(picks) == 1 else list(axes)

    for ax, (_, r) in zip(axes, picks.iterrows()):
        try:
            if frame == "crop":
                with Image.open(r["crop_path"]) as im:
                    im = im.convert("RGB")
                    W, H = im.size
                    ax.imshow(im)
                g = geometry_in_crop(r, cfg=cfg)
                # 정규화 좌표 → 이 크롭 이미지의 픽셀 좌표
                b = [g["bbox"][0] * W, g["bbox"][1] * H,
                     g["bbox"][2] * W, g["bbox"][3] * H] if g["bbox"] else None
                poly = [[p[0] * W, p[1] * H] for p in g["polygon"]] if g["polygon"] else None
                title = f"{r.get('label')}  크롭 {W}x{H} ({r.get('crop_tag')})"
            else:
                with Image.open(r["image_path"]) as im:
                    ax.imshow(im.convert("RGB"))
                b = _as_list(r.get("bbox"))
                poly = _as_list(r.get("polygon"))
                title = f"{r.get('label')}  원본 {r.get('img_w')}x{r.get('img_h')}"
        except Exception as exc:
            ax.text(0.5, 0.5, f"열기 실패\n{str(exc)[:60]}", ha="center", fontsize=8)
            ax.axis("off")
            continue

        if b:
            ax.add_patch(patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                           fill=False, lw=2, edgecolor="red"))
        if poly:
            ax.add_patch(patches.Polygon(poly, fill=False, lw=1.5, edgecolor="yellow"))
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    print("빨강=bbox, 노랑=polygon. 병변 위에 정확히 얹혀 있어야 합니다.")
    if frame == "crop":
        print("💡 크롭 프레임이라 박스가 화면 가운데를 크게 차지하는 게 정상입니다")
        print("   (margin 1.5 정사각이면 박스가 폭의 약 2/3). 박스가 병변을 감싸는지 보세요.")
    else:
        print("어긋나 있다면 labels 의 좌표 해석이 틀린 것입니다.")
