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
import time
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


def _box4(bbox) -> list[float] | None:
    """bbox 를 길이 4 실수 리스트로 정규화합니다. 아니면 None.

    ⚠️ **`if bbox:` 를 직접 쓰면 안 됩니다.** 매니페스트를 parquet 으로 저장했다가
    다시 읽으면 bbox 가 `numpy.ndarray` 로 돌아옵니다. 배열에 `if` 를 걸면
    `ValueError: truth value of an array is ambiguous` 가 납니다.
    `_crop_one` 이 예외를 삼키기 때문에 **전량 실패인데 조용히 넘어갑니다** —
    실제로 `--recrop` 을 만들다 6/6 실패를 성공으로 보고하는 걸 잡았습니다.
    """
    if bbox is None:
        return None
    try:
        vals = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    if len(vals) != 4 or any(v != v for v in vals):    # NaN 포함
        return None
    return vals


def fixed_box(bbox, img_w: int, img_h: int, side: int) -> tuple[int, int, int, int]:
    """병변 중심에서 **항상 같은 픽셀 크기**의 정사각형을 잘라냅니다.

    왜 필요한가 — margin 배율 크롭의 치명적 결함:
      margin 1.5 크롭은 병변이 클수록 넓게, 작을수록 좁게 자릅니다.
      그걸 다 같은 224px 로 리사이즈하면 **작은 병변은 크게 확대**됩니다.
      실측: A1 박스 중앙값 0.47% vs A6 3.08% → 6.5배 차이.
      그러면 모델은 피부 대신 "확대 배율"을 세서 클래스를 맞힐 수 있습니다.

      고정 픽셀 창은 그 경로를 막습니다. 피부 1mm 가 항상 같은 픽셀 수입니다.
      대신 큰 병변은 창을 넘어 잘립니다 — 그건 감수하는 대가입니다.
    """
    box = _box4(bbox)
    if box:
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    else:
        cx, cy = img_w / 2, img_h / 2

    side = min(side, img_w, img_h)          # 이미지보다 클 수는 없습니다
    half = side / 2
    x1, y1, x2, y2 = cx - half, cy - half, cx + half, cy + half

    # 경계를 벗어나면 크기를 줄이는 대신 중심을 밀어 넣습니다 (배율 유지가 목적)
    if x1 < 0:
        x2 -= x1; x1 = 0
    if y1 < 0:
        y2 -= y1; y1 = 0
    if x2 > img_w:
        x1 -= x2 - img_w; x2 = img_w
    if y2 > img_h:
        y1 -= y2 - img_h; y2 = img_h

    return (max(int(x1), 0), max(int(y1), 0),
            min(int(round(x2)), img_w), min(int(round(y2)), img_h))


def _window(bbox, img_w: int, img_h: int, margin: float = 0.0,
            fixed_px: int = 0, min_px: int = 64) -> tuple[int, int, int, int]:
    """크롭 창을 정하는 **단 하나의** 함수.

    `_crop_one`(크롭 생성)과 `crop_window`(사후 좌표 복원)가 모두 이걸 씁니다.
    두 곳에 같은 로직을 두면 갈라지고, 갈라지면 Grad-CAM 게이트가 거짓말합니다.
    """
    if fixed_px > 0:
        return fixed_box(bbox, img_w, img_h, fixed_px)
    box = _box4(bbox)
    if box and margin > 0:
        return expand_box(box, img_w, img_h, margin=margin, min_px=min_px)
    # 박스가 없거나 margin 이 없으면 중앙 정사각 (전체이미지 실험군)
    s = min(img_w, img_h)
    x1, y1 = (img_w - s) // 2, (img_h - s) // 2
    return (x1, y1, x1 + s, y1 + s)


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

            x1, y1, x2, y2 = _window(bbox, W, H, margin=cfg.crop_margin,
                                     fixed_px=cfg.crop_fixed_px, min_px=cfg.crop_min_px)
            if x2 - x1 < 8 or y2 - y1 < 8:
                return i, None
            im = im.crop((x1, y1, x2, y2))

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
    fixed_px: int | None = None,
    tag: str | None = None,
    workers: int = 8,
    verbose: bool = True,
    allow_partial: bool = False,
) -> pd.DataFrame:
    """크롭을 만들고 `crop_path` 컬럼을 붙입니다.

    두 가지 방식이 있습니다:
      margin=1.5    병변 박스의 1.5배를 잘라냄. 병변 크기에 따라 **배율이 달라짐**
      fixed_px=320  병변 중심에서 항상 320px. **배율이 일정** (지름길 차단)
      margin=0      중앙 정사각 (박스를 안 씀. 'full')

    여러 번 불러 tag 별로 저장해 두면 STEP 4 에서 비교할 수 있습니다.
    """
    cfg = cfg or CFG()
    over: dict = {}
    if margin is not None:
        over["crop_margin"] = margin
    if fixed_px is not None:
        over["crop_fixed_px"] = fixed_px
    if over:
        cfg = CFG(**{**cfg.to_dict(), **over})

    if tag is None:
        if cfg.crop_fixed_px > 0:
            tag = f"f{cfg.crop_fixed_px}"
        elif cfg.crop_margin > 0:
            tag = f"m{cfg.crop_margin:g}"
        else:
            tag = "full"

    out_dir = env.ensure_dirs()["crops"]
    if verbose:
        how = (f"고정 {cfg.crop_fixed_px}px" if cfg.crop_fixed_px > 0
               else (f"margin {cfg.crop_margin}" if cfg.crop_margin > 0 else "중앙 정사각"))
        print(f"[crop] {how}  tag={tag} → {out_dir / tag}")
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

    failed = int(out["crop_path"].isna().sum())
    made = len(out) - failed
    if verbose:
        size = sum(f.stat().st_size for f in (out_dir / tag).rglob("*.jpg")) / 1024**3
        print(f"[crop] 완료 — 실패 {failed:,}건, 저장 용량 {size:.2f} GB")

    # ⚠️ 실패를 세어 찍기만 하고 넘어가면 **전량 실패도 성공처럼 보입니다.**
    #    `_crop_one` 이 모든 예외를 삼키기 때문에, 원인이 무엇이든 결과는 "0장 저장"
    #    인데 로그에는 한 줄만 지나갑니다. 실제로 6/6 실패를 "✅ 완료" 로 보고하는 걸
    #    잡았습니다 (parquet 에서 읽은 bbox 가 ndarray 라 판정문에서 터진 경우).
    #    부분 업로드 게이트(MIN_CROP_COVERAGE)와 같은 이유로 여기서 멈춥니다.
    cover = made / max(len(out), 1)
    if not allow_partial and cover < MIN_CROP_COVERAGE:
        raise RuntimeError(
            f"\n크롭이 {made:,}/{len(out):,}장만 만들어졌습니다 ({cover:.1%}).\n"
            f"  태그: {tag}   저장 위치: {out_dir / tag}\n\n"
            "  흔한 원인:\n"
            "   · 원본을 못 엽니다 (경로가 바뀌었거나 zip 이 없음)\n"
            "   · bbox 형식이 예상과 다릅니다 (parquet 에서 읽으면 ndarray 입니다)\n"
            "   · 디스크가 찼습니다\n\n"
            "  일부만 만들어도 괜찮은 상황이면 allow_partial=True 로 부르세요."
        )

    if failed:
        out = out[out["crop_path"].notna()].reset_index(drop=True)
        if verbose:
            print(f"[crop] 실패분 제외 후 {len(out):,}행")
    return out


def margin_of_tag(tag: str) -> float:
    """크롭 태그에서 margin 을 되돌립니다. 'm1.5' → 1.5, 'full'/'f320' → 0."""
    if not isinstance(tag, str) or not tag.startswith("m"):
        return 0.0
    try:
        return float(tag[1:])
    except ValueError:
        return 0.0


def choose_stage1_tag(best_crop: str, scale_gap: float, full_ceiling: float,
                      target_recall: float, tags: list[str]) -> dict:
    """1단계(정상/이상)가 쓸 크롭을 고릅니다.

    ⚠️ **1단계는 2단계 크롭을 따라오지 않습니다.** 한 번 헷갈렸던 부분이라 적어둡니다.

    2단계는 병변 **형태**를 보므로 ROI 크롭(m1.5/m2.5)이 필요합니다. 그런데 1단계는
    정상/이상을 가르는 문제라, ROI 크롭을 쓰면 **크롭 창 크기 자체가 정답을 흘립니다** —
    정상의 bbox 면적 중앙값이 0.71%, 병변이 1.25% 라 창 크기만 봐도 반쯤 맞힙니다.
    실측 지름길 하한선(`shortcut_baseline`)에서 1단계 AUROC 0.6042 가 나온 게 그것입니다.
    게다가 그 신호는 **배포에는 존재하지 않습니다** (보호자 사진에는 bbox 가 없습니다).
    검증 점수만 올리고 실사용에서는 사라지는, 가장 나쁜 종류의 지름길입니다.

    그래서 우선순위는:
      1. 고정 픽셀 크롭(f320 …)  — 배율이 일정하면서 병변도 화면에 남습니다
      2. `full` (박스 미사용)     — 배율 정보가 아예 없지만 병변 일부를 잃습니다
      3. 어쩔 수 없으면 ROI 크롭  — 점수를 낙관적으로 취급해야 합니다

    Args:
        scale_gap: 정상 bbox 면적 / 병변 bbox 면적 (`audit` 의
            `area_ratio_normal_over_lesion`). 1 에서 멀수록 지름길이 큽니다.
        full_ceiling: `full` 로 갈 때 1단계 recall 의 상한 (`full_crop_loss`).
        tags: 실제로 **붙어 있는** 크롭 태그. 없는 걸 고르면 학습 직전에 죽습니다.

    Returns:
        {"tag", "why", "leaky", "warnings"} — `warnings` 는 그냥 출력하면 됩니다.
    """
    leaky = scale_gap > 1.5 or scale_gap < 0.67
    fixed = next((t for t in tags if fixed_of_tag(t)), None)
    warnings: list[str] = []

    if not leaky:
        tag = best_crop
        why = f"배율 격차 {scale_gap:.2f}배 — 지름길이 약해 ROI 크롭을 그대로 씀"
    elif fixed:
        tag = fixed
        why = f"배율 격차 {scale_gap:.2f}배 → 고정 픽셀 크롭 '{fixed}' (배율 일정 + 병변 보존)"
    elif "full" in tags and full_ceiling >= target_recall + 0.01:
        tag = "full"
        why = (f"배율 격차 {scale_gap:.2f}배 → full (박스 미사용). "
               f"천장 {full_ceiling:.3f} 이 목표 {target_recall:.2f} 보다 높아 사용 가능")
        warnings.append(
            f"⚠️ 천장 {full_ceiling:.3f} — 여유가 {full_ceiling - target_recall:.3f} 뿐입니다.\n"
            "   1단계 recall 이 목표에 못 미치면 모델 문제가 아니라 이 천장 때문일 수 있습니다.\n"
            "   f320 을 만들면 배율도 잡고 병변도 안 잃습니다 (--margins -320).")
    elif "full" not in tags:
        # ★ 여기서 멈추는 게 맞습니다. 예전에는 이 경우가 18번 셀까지 안 잡히고
        #   FileNotFoundError 로 죽었습니다 — 원인이 "업로드 누락" 이라는 게 안 보였습니다.
        raise FileNotFoundError(
            f"\n배율 격차가 {scale_gap:.2f}배라 1단계는 ROI 크롭을 쓸 수 없는데,\n"
            f"'full' 도 고정 픽셀 크롭도 붙어 있지 않습니다.\n"
            f"  붙어 있는 태그: {tags}\n\n"
            "해결:\n"
            "  · Kaggle 이면 [Add Input] 으로 'full' 크롭 데이터셋을 붙이세요\n"
            "  · 또는 로컬에서 고정 픽셀 크롭을 만드세요:\n"
            "      py prepare_local.py --chunk VL01 --margins -320\n"
            f"  · 지름길을 감수하고 진행하려면 STAGE1_CROP = '{best_crop}' 로 직접 지정하세요\n"
            "    (그 경우 1단계 점수는 낙관적입니다 — 배포에는 없는 신호를 씁니다)")
    else:
        tag = best_crop
        why = (f"배율 격차 {scale_gap:.2f}배지만 full 천장 {full_ceiling:.3f} 이 "
               f"목표 {target_recall:.2f} 에 못 미치고, 고정 픽셀 크롭도 없음")
        warnings.append(
            "🚨 지름길을 막지 못한 상태로 진행합니다. 점수를 낙관적으로 취급하세요.\n"
            "   · 로컬에서: py prepare_local.py --chunk VL01 --margins -320\n"
            "   · 6번 배율 교란 검사로 실제 의존도를 확인하세요")

    if tag not in tags:
        raise FileNotFoundError(
            f"1단계 크롭으로 '{tag}' 을 골랐는데 붙어 있지 않습니다. 붙어 있는 것: {tags}")
    return {"tag": tag, "why": why, "leaky": leaky, "warnings": warnings}


def fixed_of_tag(tag: str) -> int:
    """고정 픽셀 태그에서 창 크기를 되돌립니다. 'f320' → 320, 그 외 0."""
    if not isinstance(tag, str) or len(tag) < 2 or tag[0] != "f" or not tag[1:].isdigit():
        return 0
    return int(tag[1:])


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

    t = tag or row.get("crop_tag") or ""
    return _window(bbox, W, H, margin=margin_of_tag(t),
                   fixed_px=fixed_of_tag(t), min_px=cfg.crop_min_px)


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


# ──────────────────────────────────────────────────────────────────────
# 폴리곤 — **bbox 는 이것의 외접사각형입니다** (VL01 4,000행에서 오차 0px,
# 표준편차 0). 즉 라벨러가 그린 원본은 폴리곤이고 네모는 우리가 뽑은 값입니다.
# 그래서 "네모가 대충 그려졌나" 가 아니라 "네모로 줄이면서 무엇을 잃었나" 를
# 물어야 합니다 — 아래 함수들이 그걸 잽니다. 좌표는 bbox 와 같은 원본 픽셀.
# ──────────────────────────────────────────────────────────────────────


def _poly(v) -> list[list[float]] | None:
    """폴리곤을 [[x, y], ...] 실수 리스트로. 점이 3개 미만이거나 NaN 이면 None.

    ⚠️ parquet 왕복 뒤에는 `numpy.ndarray` 로 돌아옵니다 — `if poly:` 를 쓰면
       `ValueError: truth value of an array is ambiguous` 입니다 (`_box4` 와 같은 함정).
    """
    v = _as_list(v)
    if v is None:
        return None
    try:
        pts = [[float(a), float(b)] for a, b in v]
    except (TypeError, ValueError):
        return None
    if len(pts) < 3 or any(c != c for pt in pts for c in pt):     # NaN 포함
        return None
    return pts


def polygon_area(poly) -> float:
    """신발끈 공식. 넓이(항상 0 이상), 계산 불가면 0.0.

    점 순서(시계/반시계)를 모르므로 절댓값을 씁니다.
    """
    pts = _poly(poly)
    if pts is None:
        return 0.0
    a = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def polygon_centroid(poly) -> tuple[float, float] | None:
    """넓이 가중 무게중심. 넓이가 0(직선으로 눌린 폴리곤)이면 꼭짓점 평균.

    **네모 중심과 다를 수 있습니다** — 길쭉하거나 굽은 병변에서 벌어집니다.
    크롭 창은 네모 중심에 놓이므로 그 차이가 곧 "창이 병변을 벗어난 정도" 입니다.
    """
    pts = _poly(poly)
    if pts is None:
        return None
    a = cx = cy = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        cr = x1 * y2 - x2 * y1
        a += cr
        cx += (x1 + x2) * cr
        cy += (y1 + y2) * cr
    if abs(a) < 1e-9:
        n = len(pts)
        return (sum(q[0] for q in pts) / n, sum(q[1] for q in pts) / n)
    return (cx / (3.0 * a), cy / (3.0 * a))


def point_in_polygon(x: float, y: float, poly) -> bool:
    """광선 투사(ray casting). 경계 위는 구현 정의 — 판정에 쓰지 마세요."""
    pts = _poly(poly)
    if pts is None:
        return False
    inside = False
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i - 1) % n]
        if (y1 > y) != (y2 > y):
            if x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
                inside = not inside
    return inside


def _clip_to_rect(pts: list[list[float]], rect) -> list[list[float]]:
    """Sutherland-Hodgman — 볼록한 사각형으로 자릅니다 (근사 아님, 정확).

    자르는 쪽이 볼록하기만 하면 되고, 잘리는 폴리곤은 오목해도 됩니다.
    (오목한 폴리곤이 두 조각으로 갈리는 경우 잇는 선분이 생기는데, 그 선분이
     감싸는 넓이는 0 이라 **넓이 계산에는 영향이 없습니다.**)
    """
    x1, y1, x2, y2 = rect
    edges = (("x>", x1), ("y>", y1), ("x<", x2), ("y<", y2))

    def inside(pt, e):
        kind, v = e
        return (pt[0] >= v if kind == "x>" else pt[0] <= v if kind == "x<"
                else pt[1] >= v if kind == "y>" else pt[1] <= v)

    def cross(a, b, e):
        kind, v = e
        if kind in ("x>", "x<"):
            t = (v - a[0]) / (b[0] - a[0]) if b[0] != a[0] else 0.0
            return [v, a[1] + t * (b[1] - a[1])]
        t = (v - a[1]) / (b[1] - a[1]) if b[1] != a[1] else 0.0
        return [a[0] + t * (b[0] - a[0]), v]

    out = [list(q) for q in pts]
    for e in edges:
        if not out:
            return []
        cur, out = out, []
        for i in range(len(cur)):
            a, b = cur[i - 1], cur[i]
            ia, ib = inside(a, e), inside(b, e)
            if ib:
                if not ia:
                    out.append(cross(a, b, e))
                out.append(b)
            elif ia:
                out.append(cross(a, b, e))
    return out


def polygon_in_window(row, tag: str | None = None, cfg: CFG | None = None) -> dict:
    """크롭 창이 병변을 실제로 얼마나 담았나 — **모델 입력의 진짜 내용물**.

    지금까지는 bbox 점유율로 재 왔는데(STEP 17), bbox 는 폴리곤의 외접사각형이라
    길쭉하거나 굽은 병변에서 **실제보다 훨씬 크게** 잡힙니다. 이 함수가 그 자리를
    대신합니다. 돌려주는 값:

    ``occ``          창 넓이 대비 병변 넓이 (0~1) — 창이 병변으로 얼마나 찼나
    ``captured``     병변 넓이 대비 창에 들어온 병변 (0~1) — 병변을 얼마나 담았나
    ``slack``        1 - 폴리곤/네모 넓이 — 네모로 줄이며 생긴 빈 곳
    ``center_off``   네모 중심 ↔ 폴리곤 무게중심 거리 / 네모 긴 변
    ``center_on``    네모 중심이 병변 **안**인가 (창이 병변 위에 놓였는가)

    셋 다 계산 불가면 값은 ``nan`` / ``None`` 입니다 — 0 으로 채우지 않습니다.
    없는 것과 0 은 다르고, 섞으면 분석이 조용히 틀립니다.
    """
    nan = float("nan")
    bad = {"occ": nan, "captured": nan, "slack": nan,
           "center_off": nan, "center_on": None}
    pts = _poly(row.get("polygon"))
    win = crop_window(row, tag, cfg)
    if pts is None or win is None:
        return bad
    wx1, wy1, wx2, wy2 = win
    warea = (wx2 - wx1) * (wy2 - wy1)
    parea = polygon_area(pts)
    if warea <= 0 or parea <= 0:
        return bad

    inter = polygon_area(_clip_to_rect(pts, win)) if len(_clip_to_rect(pts, win)) >= 3 else 0.0

    box = _box4(row.get("bbox"))
    slack, off, on = nan, nan, None
    cen = polygon_centroid(pts)
    if box and cen is not None:
        barea = (box[2] - box[0]) * (box[3] - box[1])
        if barea > 0:
            slack = 1.0 - parea / barea
        bcx, bcy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        long = max(box[2] - box[0], box[3] - box[1])
        if long > 0:
            off = ((bcx - cen[0]) ** 2 + (bcy - cen[1]) ** 2) ** 0.5 / long
        on = point_in_polygon(bcx, bcy, pts)

    return {"occ": inter / warea, "captured": inter / parea,
            "slack": slack, "center_off": off, "center_on": on}


def available_tags() -> list[str]:
    """만들어져 있는 크롭 태그 목록 (예: ['full', 'm1.5', 'm2.5'])."""
    d = env.work_root() / "crops"
    return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.exists() else []


MIN_CROP_COVERAGE = 0.95   # 이 아래면 부분 데이터 학습을 막습니다


def chunks_with_crops(df, tags, sample: int = 300, need: float = 0.95,
                      verbose: bool = True) -> list[str]:
    """크롭이 실제로 있는 청크만 골라냅니다 — `tags` **전부** 있어야 통과.

        crop.chunks_with_crops(df, ["m2.5", "f320"])

    ⚠️ 전수 확인은 느립니다. 365,428번 파일 검사는 네트워크 볼륨(캐글 입력)에서
       몇 분씩 걸리고 그동안 화면이 조용합니다. 청크는 **통째로 있거나 통째로
       없으므로** 청크마다 `sample` 장만 봅니다.

    2026-09-05 캐글 실측: `m2.5` 데이터셋에 376,074장이 붙어 있는데 **VL01
    39,508장이 통째로 빠져** 있었습니다 (TL01·TL02 만 올라감). 그대로 돌리면
    `switch_tag` 가 보유율 89.2% 로 멈추는데, **어느 청크가 빠졌는지는 안
    알려줍니다.** 이 함수가 그걸 대신합니다.

    ⚠️ 크롭 **비교** 실험에서는 태그를 여러 개 넘기세요. 한 태그만 있는 청크를
       쓰면 판이 서로 다른 사진을 보게 됩니다.
    """
    tags = [tags] if isinstance(tags, str) else list(tags)
    out_dir = env.work_root() / "crops"
    keep = []
    if verbose:
        print(f"[crop] 청크별 보유율 확인 — 태그 {tags}, 표본 {sample}장씩")
    for ch, g in df.groupby("chunk"):
        rates = {}
        for t in tags:
            q = g["image_path"].sample(min(sample, len(g)), random_state=0)
            hit = sum(Path(_out_path(x, out_dir, t)).exists() for x in q)
            rates[t] = hit / len(q)
        ok = all(v >= need for v in rates.values())
        if verbose:
            detail = "  ".join(f"{t} {v:.0%}" for t, v in rates.items())
            print(f"  [{'O' if ok else 'X'}] {ch:<12} {detail}")
        if ok:
            keep.append(ch)
    return sorted(keep)


def switch_tag(df: pd.DataFrame, tag: str, verbose: bool = True,
               allow_missing: bool = False) -> pd.DataFrame:
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

    # ⚠️ 45,885번의 파일 확인입니다. Kaggle 입력은 네트워크 마운트라 캐시가
    #    차가우면 몇 분씩 걸리는데, 그동안 출력이 없으면 멈춘 줄 알고 세션을
    #    껐다 켜게 됩니다 — 그게 더 큰 낭비라 진행 상황을 찍습니다.
    if verbose:
        print(f"[crop] 태그 '{tag}' 로 전환 — {len(out):,}장 확인 중 …", flush=True)
    t0 = time.time()
    exists = out["crop_path"].apply(lambda p: Path(p).exists())
    cover = float(exists.mean())
    if verbose:
        print(f"[crop] 태그 '{tag}' 로 전환 — {exists.sum():,}/{len(out):,}장 존재 "
              f"({cover:.1%}, {time.time() - t0:.0f}초)")
    if not exists.all():
        miss = int((~exists).sum())
        print(f"⚠️ {miss:,}장이 없습니다. 이 태그의 크롭이 안 만들어졌거나 "
              f"업로드가 덜 끝났을 수 있습니다.")
        print(f"   사용 가능한 태그: {available_tags()}")
        # ⚠️ 경고만 하고 넘어가면 **부분 데이터로 몇 시간을 학습**하게 됩니다.
        #    실제로 겪은 일: full 크롭이 30% 만 업로드됐는데 경고 한 줄만 찍히고
        #    1단계가 13,783장(전체의 30%)으로 학습됐습니다. 로그에 묻혀 안 보였고,
        #    그 숫자를 이전 실행과 비교할 뻔했습니다.
        if cover < MIN_CROP_COVERAGE and not allow_missing:
            raise FileNotFoundError(
                f"\n크롭 '{tag}' 가 {cover:.1%} 밖에 없습니다 "
                f"(기준 {MIN_CROP_COVERAGE:.0%}).\n"
                f"  이대로 학습하면 전체의 {cover:.0%} 만 쓰게 되고, 나온 숫자는\n"
                f"  다른 실행과 비교할 수 없습니다.\n\n"
                f"확인할 것:\n"
                f"  · 업로드한 데이터셋의 파일 개수 (있어야 할 수: {len(out):,})\n"
                f"  · 로컬 zip 이 온전한지:\n"
                f"      py -c \"import zipfile;print(len(zipfile.ZipFile('dogskin_{tag}.zip').namelist()))\"\n\n"
                f"그래도 진행하려면: crop.switch_tag(df, '{tag}', allow_missing=True)"
            )
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
        print(f"\n    (a) 정상 vs 병변:  {m_norm:.2%}  vs  {m_les:.2%}  →  {ratio:.2f}배")
        if ratio > 1.5 or ratio < 0.67:
            print("        🚨 크롭만 봐도 정상/병변이 티가 납니다.")
            print("           → 1단계가 피부가 아니라 '확대 정도'로 맞힐 수 있습니다.")
            print("           → 대응: 1단계를 'full' 크롭으로 (노트북 03 이 자동 전환)")
        else:
            print("        ✅ 배율이 비슷합니다.")

    # 병변 6종 **사이**의 배율 차이 — 2단계의 지름길
    les = med.drop(NORMAL_LABEL, errors="ignore").dropna()
    if len(les) > 1:
        spread = float(les.max() / max(les.min(), 1e-9))
        out["area_spread_within_lesions"] = spread
        out["area_largest_class"] = str(les.idxmax())
        out["area_smallest_class"] = str(les.idxmin())
        print(f"\n    (b) 병변 6종 사이:  {les.idxmin()} {les.min():.2%}  ~  "
              f"{les.idxmax()} {les.max():.2%}  →  {spread:.1f}배")
        if spread > 2.0:
            print("        🚨 병변 종류끼리도 배율이 크게 다릅니다.")
            print(f"           → 2단계가 '박스 크면 {les.idxmax()}' 로 맞힐 수 있습니다.")
            print("           → 대응: 고정 픽셀 크롭(f320)을 만들어 배율을 통일하세요.")
            print("              로컬에서: py prepare_local.py --chunk VL01 --margins -320")
            print("           → 먼저 crop.shortcut_baseline() 으로 실제 피해량을 재세요.")
        else:
            print("        ✅ 병변 종류 간 배율은 비슷합니다.")

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
    bad_rows = []
    for idx, r in df[has_box].iterrows():
        b = _as_list(r.get("bbox"))
        w, h = r.get("img_w"), r.get("img_h")
        if not b or not w or not h:
            continue
        over = max(-b[0], -b[1], b[2] - w, b[3] - h)          # 얼마나 벗어났나 (px)
        if over > 1 or b[2] <= b[0] or b[3] <= b[1]:
            bad_rows.append((idx, r.get("label"), b, int(w), int(h), float(over)))
    out["boxes_out_of_bounds"] = len(bad_rows)
    print(f"    경계 이탈/역전 박스: {len(bad_rows):,}건 " + ("✅" if not bad_rows else "⚠️"))
    if bad_rows:
        worst = max(x[5] for x in bad_rows)
        out["box_overflow_max_px"] = worst
        print(f"    최대 이탈량: {worst:.0f}px")
        for idx, lab, b, w, h, over in bad_rows[:5]:
            bb = [round(v, 1) for v in b]
            print(f"      {lab}  bbox={bb}  이미지={w}x{h}  이탈 {over:.0f}px")
        # 판정은 절대 px 이 아니라 **건수 + 상대 비율**로 합니다.
        # 좌표 해석이 틀렸으면 수천 건이, 이미지 크기에 맞먹는 규모로 어긋납니다.
        frac_rows = len(bad_rows) / max(int(has_box.sum()), 1)
        rel = worst / max(float(df["img_h"].median() or 1), 1.0)
        out["boxes_out_of_bounds_frac"] = frac_rows
        out["box_overflow_max_rel"] = rel
        print(f"    전체 대비 {frac_rows:.3%},  최대 이탈량은 이미지 높이의 {rel:.1%}")
        # 라벨러 오차: 드물고(1% 미만) 이탈량도 이미지 크기에 비해 작을 때
        # 좌표 해석 오류: 계통적으로 많거나(1% 이상) 이미지 규모로 어긋날 때
        if frac_rows < 0.01 and rel < 0.3:
            print("    → **라벨러의 오차**로 보입니다. 좌표 해석 오류가 아닙니다.")
            print("       (해석이 틀렸다면 수천 건이, 이미지 크기에 맞먹게 어긋납니다)")
            print("       crop 이 어차피 이미지 안으로 잘라 넣으므로 학습에 영향 없습니다.")
        else:
            print("    🚨 좌표 해석 오류를 의심하세요 (x/y 교환, 스케일 불일치).")
            print("       crop.preview_with_box() 로 그 행들을 직접 보세요.")
        out["boxes_out_of_bounds_examples"] = [
            {"label": lab, "bbox": b, "img": [w, h], "overflow_px": over}
            for _, lab, b, w, h, over in bad_rows[:20]
        ]

    print("\n" + "=" * 68)
    return out


def full_crop_loss(df: pd.DataFrame, tag: str = "full", cfg: CFG | None = None,
                   verbose: bool = True) -> dict:
    """`full` 크롭으로 바꿀 때 **병변이 화면에서 잘려 나가는 비율**.

    `full` 은 이미지 중앙 정사각형만 씁니다 (1920×1080 → 가운데 1080×1080).
    배율 지름길을 없애는 대신, 병변이 좌우 끝에 있으면 아예 안 보입니다.
    그럼 그 사진은 "이상" 인데 정상처럼 보입니다 — **1단계 recall 의 천장이
    데이터 때문에 낮아집니다.**

    그 천장을 미리 알아야 "1단계를 full 로 간다" 는 결정을 할 수 있습니다.
    이미지를 열지 않고 좌표만으로 계산하므로 즉시 끝납니다.
    """
    import numpy as np

    from src.config import CLASS_KO, NORMAL_LABEL

    cfg = cfg or CFG()
    sub = df[df["bbox"].notna()]
    if not len(sub):
        print("bbox 가 있는 행이 없습니다.")
        return {}

    recs = []
    for _, r in sub.iterrows():
        b = _as_list(r.get("bbox"))
        win = crop_window(r, tag=tag, cfg=cfg)
        if not b or win is None:
            continue
        ix1, iy1 = max(b[0], win[0]), max(b[1], win[1])
        ix2, iy2 = min(b[2], win[2]), min(b[3], win[3])
        inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
        area = max((b[2] - b[0]) * (b[3] - b[1]), 1e-9)
        recs.append((str(r.get("label")), inter / area))

    if not recs:
        return {}
    labels = np.array([x[0] for x in recs])
    vis = np.array([x[1] for x in recs])

    out = {
        "tag": tag,
        "n": len(vis),
        "fully_visible": float((vis >= 0.9).mean()),
        "partial": float(((vis > 0.1) & (vis < 0.9)).mean()),
        "mostly_gone": float((vis <= 0.1).mean()),
        "median_visible": float(np.median(vis)),
    }
    les = labels != NORMAL_LABEL
    out["lesion_mostly_gone"] = float((vis[les] <= 0.1).mean()) if les.any() else 0.0
    out["stage1_recall_ceiling"] = 1.0 - out["lesion_mostly_gone"]

    if verbose:
        print("=" * 68)
        print(f" '{tag}' 크롭으로 갈 때의 손실 — 병변이 화면 밖으로 나가는 비율")
        print("=" * 68)
        print(f"  전부 보임(≥90%)   {out['fully_visible']:.1%}")
        print(f"  일부 잘림         {out['partial']:.1%}")
        print(f"  거의 사라짐(≤10%) {out['mostly_gone']:.1%}")
        print(f"\n  병변 중 사라진 비율      {out['lesion_mostly_gone']:.1%}")
        print(f"  ★ 1단계 recall 천장     {out['stage1_recall_ceiling']:.3f}   (목표 0.95)")

        rows = []
        for lab in sorted(set(labels)):
            m = labels == lab
            rows.append({"label": f"{lab} {CLASS_KO.get(lab, '')}"[:22],
                         "n": int(m.sum()),
                         "사라짐": f"{(vis[m] <= 0.1).mean():.1%}",
                         "중앙값보임": f"{np.median(vis[m]):.0%}"})
        print()
        print(pd.DataFrame(rows).to_string(index=False))

        ceil = out["stage1_recall_ceiling"]
        if ceil < 0.95:
            print(f"\n  🚨 천장 {ceil:.3f} < 0.95 — '{tag}' 로는 목표 recall 을 못 채웁니다.")
            print("     배율 지름길은 막지만 병변을 잃습니다. 대안:")
            print("       · 고정 픽셀 창을 넉넉하게 (예: --margins -512 → f512)")
            print("       · 1단계도 f320 을 쓰기 (배율은 일정하고 병변은 항상 포함)")
        elif ceil < 0.98:
            print(f"\n  ⚠️ 천장 {ceil:.3f} — 여유가 거의 없습니다. 다른 손실이 겹치면 위험합니다.")
        else:
            print(f"\n  ✅ 천장 {ceil:.3f} — '{tag}' 를 써도 병변 손실은 무시할 수준입니다.")
        print("=" * 68 + "\n")
    return out


# 어떤 메타데이터가 **크롭 이미지에 실제로 보이는가**.
#
# 이 구분이 중요합니다. 크롭에 안 보이는 특징까지 넣으면 하한선이 부풀려지고,
# CNN 이 넘어야 할 선을 실제보다 높게 잡게 됩니다.
#
#   보임    박스 면적, 크롭 창 크기 → 확대 배율로 나타남 (질감의 굵기)
#   안 보임 종횡비   → expand_box(square=True) 라 크롭은 항상 정사각형
#           병변 개수 → 박스 하나만 잘라내므로 나머지는 화면 밖
#           원본 해상도 → 크롭 후 리사이즈되어 흔적이 거의 없음
FEATURE_SETS: dict[str, list[str]] = {
    # 크롭 배율이 흘리는 양만. **f320 도입 여부는 이걸로 판단하세요.**
    "scale_only": ["area_ratio", "log_area", "win_side", "win_over_input"],
    # 메타데이터 전체. 데이터셋에 상관이 얼마나 있는지 보는 참고용 (상한).
    "all": ["area_ratio", "log_area", "aspect", "win_side", "win_over_input",
            "img_w", "img_h", "n_lesion", "synthetic"],
}


def shortcut_baseline(df: pd.DataFrame, cfg: CFG | None = None, fold: int = 0,
                      features: str = "scale_only", verbose: bool = True) -> dict:
    """★ 사진을 **안 보고** 라벨을 맞혀봅니다.

    픽셀은 한 장도 안 봅니다. 그런데도 점수가 잘 나오면, 그 점수는
    **크롭 방식이 정답을 흘린 양**입니다. CNN 이 넘어야 하는 **하한선**이죠.

        CNN macro-F1 0.45  vs  하한선 0.40   →  피부에서 얻은 건 0.05 뿐
        CNN macro-F1 0.45  vs  하한선 0.18   →  대부분 피부에서 얻음 ✅

    features:
      "scale_only" (기본) — 크롭 배율로 **이미지에 실제로 보이는** 것만.
                            f320 재크롭 여부는 이 숫자로 판단하세요.
      "all"               — 메타데이터 전체. 종횡비·병변 개수·해상도까지 포함하는데
                            그건 크롭에 안 보이므로 하한선이 부풀려집니다. 참고용.

    분할은 `fold` 컬럼을 그대로 씁니다 — 그래야 CNN 점수와 같은 조건입니다.
    """
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score

    from src.config import CLASS_KO, CLASSES, NORMAL_LABEL

    cfg = cfg or CFG()
    if features not in FEATURE_SETS:
        raise KeyError(f"모르는 특징 집합: {features}. 가능: {sorted(FEATURE_SETS)}")
    cols = FEATURE_SETS[features]

    need = {"fold", "is_holdout", "label", "bbox", "img_w", "img_h"}
    miss = need - set(df.columns)
    if miss:
        print(f"필요한 컬럼이 없습니다: {sorted(miss)}")
        return {}

    def feats(sub: pd.DataFrame) -> np.ndarray:
        rows = []
        for _, r in sub.iterrows():
            b = _as_list(r.get("bbox")) or [0, 0, 0, 0]
            w, h = float(r.get("img_w") or 1), float(r.get("img_h") or 1)
            bw, bh = max(b[2] - b[0], 0.0), max(b[3] - b[1], 0.0)
            win = crop_window(r, cfg=cfg)
            ws = float(min(win[2] - win[0], win[3] - win[1])) if win else 0.0
            f = {
                "area_ratio": bw * bh / max(w * h, 1),
                "log_area": np.log1p(bw * bh),
                "aspect": bw / max(bh, 1e-6),
                "win_side": ws,                            # 크롭 창 = 확대 배율
                "win_over_input": ws / max(cfg.img_size, 1),
                "img_w": w, "img_h": h,
                "n_lesion": float(r.get("n_lesion") or 0),
                "synthetic": float(bool(r.get("synthetic"))),
            }
            rows.append([f[c] for c in cols])
        return np.asarray(rows, dtype=np.float32)

    dev = df[~df["is_holdout"]]
    tr, va = dev[dev["fold"] != fold], dev[dev["fold"] == fold]
    Xtr, Xva = feats(tr), feats(va)
    out: dict = {}

    if verbose:
        print("=" * 68)
        print(" 지름길 하한선 — 사진을 보지 않고 메타데이터만으로 분류")
        print("=" * 68)
        note = ("크롭 배율로 이미지에 실제로 보이는 것만"
                if features == "scale_only" else "메타데이터 전체 (크롭에 안 보이는 것 포함)")
        print(f"  특징 집합: {features} — {note}")
        print(f"  사용 컬럼: {', '.join(cols)}   (픽셀 미사용)")
        print(f"  train {len(tr):,} / val {len(va):,}  (CNN 과 같은 fold {fold})")

    # ── 1단계: 정상 vs 이상 ──────────────────────────────────────
    ytr = (tr["label"] != NORMAL_LABEL).astype(int).to_numpy()
    yva = (va["label"] != NORMAL_LABEL).astype(int).to_numpy()
    if len(np.unique(ytr)) > 1 and len(np.unique(yva)) > 1:
        clf = HistGradientBoostingClassifier(max_iter=150, random_state=0).fit(Xtr, ytr)
        sc = clf.predict_proba(Xva)[:, 1]
        auroc = float(roc_auc_score(yva, sc))
        out["stage1_auroc_metadata_only"] = auroc
        if verbose:
            print(f"\n[1단계] 정상/이상 AUROC = {auroc:.4f}   (0.5 = 동전 던지기)")
            if auroc > 0.75:
                print("  🚨 사진을 안 봐도 이 정도 맞힙니다. 크롭 배율이 정답을 크게 흘립니다.")
                print("     → 1단계는 반드시 'full' 크롭으로 가세요.")
            elif auroc > 0.6:
                print("  ⚠️ 약한 지름길이 있습니다. CNN 점수에서 이만큼은 할인해서 보세요.")
            else:
                print("  ✅ 메타데이터로는 거의 못 맞힙니다. 배율 지름길이 약합니다.")

    # ── 2단계: 병변 6종 ─────────────────────────────────────────
    m_tr = tr["label"].isin(CLASSES).to_numpy()
    m_va = va["label"].isin(CLASSES).to_numpy()
    if m_tr.sum() > 50 and m_va.sum() > 20:
        c2i = {c: i for i, c in enumerate(CLASSES)}
        y2tr = tr.loc[m_tr, "label"].map(c2i).to_numpy()
        y2va = va.loc[m_va, "label"].map(c2i).to_numpy()
        # ⚠️ 클래스 가중치를 안 주면 큰 클래스(A2/A3)만 찍고 작은 클래스는 전멸합니다.
        #    그럼 macro-F1 이 '배율 지름길' 이 아니라 '클래스 빈도' 를 재게 됩니다.
        #    CNN 쪽은 class_weight 를 쓰므로 조건을 맞춰야 비교가 성립합니다.
        clf2 = HistGradientBoostingClassifier(
            max_iter=200, random_state=0, class_weight="balanced").fit(Xtr[m_tr], y2tr)
        p2 = clf2.predict(Xva[m_va])
        f1 = float(f1_score(y2va, p2, average="macro", zero_division=0))
        out["stage2_macro_f1_metadata_only"] = f1

        # ★ 클래스별로 저장합니다. 지름길은 평균에 고르게 퍼지지 않고
        #    박스 크기가 극단인 클래스(가장 작은 것, 가장 큰 것)에 몰립니다.
        #    macro 평균만 보면 그게 안 보입니다.
        _, _rec, _, _ = precision_recall_fscore_support(
            y2va, p2, labels=list(range(len(CLASSES))), zero_division=0)
        out["stage2_recall_metadata_only"] = {c: float(_rec[i])
                                              for i, c in enumerate(CLASSES)}
        random_f1 = 1.0 / len(CLASSES)
        if verbose:
            print(f"\n[2단계] 병변 6종 macro-F1 = {f1:.4f}   "
                  f"(무작위 ≈ {random_f1:.3f})")
            _, rec, _, sup = precision_recall_fscore_support(
                y2va, p2, labels=list(range(len(CLASSES))), zero_division=0)
            from src.evaluate import pad_ko
            print(f"  {pad_ko('클래스', 30)}{'recall':>9}{'n':>8}")
            for i, c in enumerate(CLASSES):
                mark = "  ← 배율로 맞힘" if rec[i] > 0.5 else ""
                name = pad_ko(f"{c} {CLASS_KO.get(c, '')}", 30)
                print(f"  {name}{rec[i]:>9.3f}{sup[i]:>8,}{mark}")
            print("\n  (클래스 가중치를 준 상태입니다 — 안 주면 큰 클래스만 찍어서"
                  " 배율 지름길과 클래스 빈도를 구분할 수 없습니다)")
            if f1 > random_f1 * 2:
                print(f"\n  🚨 무작위의 {f1 / random_f1:.1f}배입니다. 크롭 배율이 병변 종류까지 흘립니다.")
                print("     → CNN 이 이 숫자를 크게 넘지 못하면, 피부를 안 보고 있는 겁니다.")
            else:
                print("\n  ✅ 메타데이터만으로는 종류를 잘 못 맞힙니다.")

    if verbose:
        print("\n" + "=" * 68)
        print(" 이 숫자를 적어두세요. 4번에서 CNN 점수와 비교합니다.")
        print("=" * 68)
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
