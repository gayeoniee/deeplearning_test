"""라벨 JSON → 하나의 매니페스트(parquet).

이후 모든 단계(중복제거·분할·크롭·학습)는 이미지 폴더를 다시 뒤지지 않고
이 매니페스트 한 장만 봅니다. 그래야 재현성이 생깁니다.

매니페스트 컬럼
    image_path       이미지 절대경로
    json_path        출처 라벨 파일
    label            A1~A6 (없으면 None)
    animal_id        개체 식별자 (데이터 누수 방지의 핵심)
    bbox             [x1,y1,x2,y2] 픽셀
    polygon          [[x,y],...] (있으면)
    img_w, img_h     원본 크기
    area_ratio       병변 면적 / 이미지 면적
    species, camera, symptom, src_split   폴더 축에서 추출

    from src import labels
    df = labels.build()
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src import env, scan
from src.config import CLASSES, EXCLUDE_CAMERA, EXCLUDE_SPECIES, INCLUDE_CAMERA, INCLUDE_SPECIES

IMG_EXT = scan.IMG_EXT


# ──────────────────────────────────────────────────────────────
# 값 추출 — 키 이름을 모르는 상태에서 최대한 버티기
# ──────────────────────────────────────────────────────────────
def _find_first(rec: Any, hints: list[str], want: str = "any",
                exclude: list[str] | None = None) -> Any:
    """중첩 구조에서 hints 단어가 들어간 키의 첫 유효값을 찾습니다.

    want: "any" | "num" | "str" | "seq"
    """
    exclude = exclude or []
    stack: list[Any] = [rec]
    best = None
    while stack:
        cur = stack.pop(0)
        if isinstance(cur, dict):
            for k, v in cur.items():
                kl = str(k).lower()
                if any(x in kl for x in exclude):
                    continue
                if any(h in kl for h in hints):
                    if want == "num" and isinstance(v, (int, float)) and not isinstance(v, bool):
                        return v
                    if want == "str" and isinstance(v, str) and v.strip():
                        return v.strip()
                    if want == "seq" and isinstance(v, (list, dict)) and v:
                        return v
                    if want == "any" and v not in (None, "", [], {}):
                        return v
                if isinstance(v, (dict, list)):
                    stack.append(v)
        elif isinstance(cur, list):
            stack.extend(x for x in cur if isinstance(x, (dict, list)))
    return best


def extract_image_name(rec: dict) -> str | None:
    v = _find_first(rec, ["file", "image", "img", "이미지", "파일"], want="str")
    if isinstance(v, str) and re.search(r"\.(jpg|jpeg|png|bmp|webp)$", v, re.I):
        return Path(v.replace("\\", "/")).name
    # 확장자가 없는 이름일 수도 있음
    if isinstance(v, str) and 3 < len(v) < 120 and "/" not in v:
        return v
    return None


def extract_label(rec: dict, path_hint: str = "") -> str | None:
    """A1~A6 코드를 찾습니다. JSON → 경로 순으로 시도."""
    for hints in (["label", "class", "질환", "증상", "라벨", "disease", "lesion", "code"],):
        v = _find_first(rec, hints, want="any")
        for cand in (v, json.dumps(v, ensure_ascii=False) if v is not None else ""):
            m = re.search(r"\b(A[1-6])\b", str(cand))
            if m:
                return m.group(1)
    m = re.search(r"(?:^|[/_\\-])(A[1-6])(?:[_/\\-]|$)", path_hint)
    return m.group(1) if m else None


def extract_animal_id(rec: dict, image_name: str = "", token_index: int | None = None) -> str | None:
    """개체 식별자. 없으면 파일명 토큰으로 대체합니다.

    ⚠️ 이 값이 틀리면 train/val 에 같은 강아지가 섞여서
       정확도가 가짜로 부풀려집니다. docs/cautions/02 참고.
    """
    v = _find_first(
        rec,
        ["pet_id", "petid", "animal_id", "animalid", "개체", "환자", "chart", "subject"],
        want="any",
    )
    if v not in (None, "", [], {}):
        return str(v)
    # 넓은 그물: id 로 끝나는 키
    v = _find_first(rec, ["_id", "id_"], want="any", exclude=["image", "file", "label", "ann"])
    if v not in (None, "", [], {}):
        return str(v)
    if image_name and token_index is not None:
        toks = re.split(r"[_\-.]", Path(image_name).stem)
        if token_index < len(toks):
            return toks[token_index]
    return None


def extract_wh(rec: dict) -> tuple[int | None, int | None]:
    w = _find_first(rec, ["width", "가로", "너비"], want="num")
    h = _find_first(rec, ["height", "세로", "높이"], want="num")
    try:
        return (int(w) if w and w > 1 else None, int(h) if h and h > 1 else None)
    except (TypeError, ValueError):
        return (None, None)


def extract_geometry(rec: dict) -> tuple[list[float] | None, list[list[float]] | None]:
    """(bbox, polygon) 을 반환. bbox 는 [x1,y1,x2,y2]."""
    poly_raw = _find_first(rec, ["polygon", "segmentation", "contour"], want="seq")
    polygon = None
    if poly_raw is not None:
        pts = scan._as_xy(poly_raw)
        if len(pts) >= 3:
            polygon = [[float(x), float(y)] for x, y in pts]

    bbox = None
    box_raw = _find_first(rec, ["bbox", "box", "rect", "roi"], want="seq")
    if box_raw is None:
        box_raw = _find_first(rec, ["location", "region"], want="seq")
    if box_raw is not None:
        pts = scan._as_xy(box_raw)
        if len(pts) >= 2:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            bbox = [min(xs), min(ys), max(xs), max(ys)]

    if bbox is None and polygon:
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        bbox = [min(xs), min(ys), max(xs), max(ys)]
    return bbox, polygon


# ──────────────────────────────────────────────────────────────
# 경로 축 추출
# ──────────────────────────────────────────────────────────────
def _axis(path_str: str, tokens: list[str]) -> str | None:
    for t in tokens:
        if t in path_str:
            return t
    return None


def path_axes(p: Path) -> dict[str, str | None]:
    s = str(p)
    split = None
    if "Training" in s or re.search(r"[/\\](TS|TL)\d", s):
        split = "train"
    elif "Validation" in s or re.search(r"[/\\](VS|VL)\d", s):
        split = "val"
    return {
        "species": _axis(s, ["반려견", "반려묘"]),
        "camera": _axis(s, ["일반카메라", "더모스코프"]),
        "symptom": _axis(s, ["유증상", "무증상"]),
        "src_split": split,
    }


# ──────────────────────────────────────────────────────────────
# 이미지 인덱스
# ──────────────────────────────────────────────────────────────
def index_images(root: Path) -> dict[str, list[Path]]:
    """파일명(확장자 포함/제외 둘 다) → 경로 목록."""
    idx: dict[str, list[Path]] = defaultdict(list)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            idx[p.name].append(p)
            idx[p.stem].append(p)
    return idx


def _resolve_image(name: str | None, json_path: Path, idx: dict[str, list[Path]]) -> Path | None:
    if not name:
        # 라벨 JSON 과 같은 이름의 이미지가 있는지
        for key in (json_path.stem, json_path.stem + ".jpg"):
            if key in idx:
                return _closest(idx[key], json_path)
        return None
    for key in (name, Path(name).stem):
        if key in idx:
            return _closest(idx[key], json_path)
    return None


def _closest(cands: list[Path], json_path: Path) -> Path:
    """동명이인 이미지가 여러 개면 JSON 과 경로가 가장 비슷한 것을 고릅니다."""
    if len(cands) == 1:
        return cands[0]
    jparts = set(json_path.parts)
    return max(cands, key=lambda c: len(jparts & set(c.parts)))


# ──────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────
def build(
    root: Path | None = None,
    report: scan.ScanReport | None = None,
    animal_token_index: int | None = None,
    dogs_only: bool = True,
    normal_camera_only: bool = True,
    save: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """모든 라벨 JSON 을 읽어 매니페스트를 만듭니다."""
    root = Path(root) if root else env.data_root()
    if report is None:
        try:
            report = scan.ScanReport.load()
        except Exception:
            report = None

    # 개체ID 토큰 위치: 인자 > 스캔 리포트 추천 > None
    if animal_token_index is None and report is not None:
        cands = report.animal_id_stats.get("filename_token_candidates") or []
        if cands:
            animal_token_index = cands[0]["token_index"]

    if verbose:
        print(f"[labels] 이미지 인덱싱: {root}")
    idx = index_images(root)
    jsons = sorted(root.rglob("*.json"))
    if verbose:
        print(f"[labels] JSON {len(jsons):,}개 처리 중…")

    rows: list[dict] = []
    unmatched = 0

    for jp in jsons:
        try:
            data = json.loads(jp.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        axes = path_axes(jp)

        for rec in (data if isinstance(data, list) else [data]):
            if not isinstance(rec, dict):
                continue
            iname = extract_image_name(rec)
            ipath = _resolve_image(iname, jp, idx)
            if ipath is None:
                unmatched += 1
                continue

            iaxes = path_axes(ipath)
            merged = {k: (iaxes[k] or axes[k]) for k in axes}

            label = extract_label(rec, str(ipath))
            if label is None and merged["symptom"] == "무증상":
                label = "A0"
            w, h = extract_wh(rec)
            bbox, polygon = extract_geometry(rec)

            area = None
            if bbox and w and h:
                area = max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / (w * h)
                area = min(area, 1.0)

            rows.append({
                "image_path": str(ipath),
                "json_path": str(jp),
                "image_name": ipath.name,
                "label": label,
                "animal_id": extract_animal_id(rec, ipath.name, animal_token_index),
                "bbox": bbox,
                "polygon": polygon,
                "img_w": w,
                "img_h": h,
                "area_ratio": area,
                **merged,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            "매니페스트가 비었습니다.\n"
            f"  JSON {len(jsons)}개, 이미지 인덱스 {len(idx)}건, 매칭 실패 {unmatched}건\n"
            "  → scan.run() 의 field_guess 를 보고 extract_* 함수를 맞춰주세요."
        )

    if verbose:
        print(f"[labels] 원시 행 {len(df):,}개 (이미지 매칭 실패 {unmatched:,}건)")

    df = _filter_scope(df, dogs_only, normal_camera_only, verbose)
    df = _fill_missing_wh(df, verbose)
    df = _finalize_animal_id(df, verbose)

    if save:
        out = env.ensure_dirs()["manifests"] / "manifest_raw.parquet"
        _save(df, out)
    if verbose:
        report_manifest(df)
    return df


def _filter_scope(df: pd.DataFrame, dogs_only: bool, normal_camera_only: bool,
                  verbose: bool) -> pd.DataFrame:
    before = len(df)
    if dogs_only and df["species"].notna().any():
        df = df[~df["species"].isin(EXCLUDE_SPECIES)]
        if df["species"].isin(INCLUDE_SPECIES).any():
            df = df[df["species"].isin(INCLUDE_SPECIES) | df["species"].isna()]
    if normal_camera_only and df["camera"].notna().any():
        df = df[~df["camera"].isin(EXCLUDE_CAMERA)]
        if df["camera"].isin(INCLUDE_CAMERA).any():
            df = df[df["camera"].isin(INCLUDE_CAMERA) | df["camera"].isna()]
    if verbose and len(df) != before:
        print(f"[labels] 범위 필터(반려견/일반카메라): {before:,} → {len(df):,}")
    return df.reset_index(drop=True)


def _fill_missing_wh(df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    """JSON 에 크기가 없으면 이미지를 열어 채웁니다 (필요한 것만)."""
    miss = df["img_w"].isna() | df["img_h"].isna()
    if not miss.any():
        return df
    try:
        from PIL import Image
    except ImportError:
        return df
    if verbose:
        print(f"[labels] 크기 정보 없는 {miss.sum():,}건 — 이미지에서 읽는 중…")
    for i in df.index[miss]:
        try:
            with Image.open(df.at[i, "image_path"]) as im:
                df.at[i, "img_w"], df.at[i, "img_h"] = im.width, im.height
        except Exception:
            continue
    # area_ratio 재계산
    m = df["area_ratio"].isna() & df["bbox"].notna() & df["img_w"].notna()
    for i in df.index[m]:
        b = df.at[i, "bbox"]
        w, h = df.at[i, "img_w"], df.at[i, "img_h"]
        if b and w and h:
            df.at[i, "area_ratio"] = min(max(0.0, (b[2] - b[0]) * (b[3] - b[1])) / (w * h), 1.0)
    return df


def _finalize_animal_id(df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    """개체ID가 비었거나 거의 모두 고유하면 경고합니다."""
    n_missing = df["animal_id"].isna().sum()
    if n_missing:
        # 최후 수단: 이미지 파일명 자체를 그룹으로 (= 그룹 분할 효과 없음)
        df.loc[df["animal_id"].isna(), "animal_id"] = (
            "UNK_" + df.loc[df["animal_id"].isna(), "image_name"].astype(str)
        )
    uniq = df["animal_id"].nunique()
    per = len(df) / max(uniq, 1)
    df.attrs["animal_id_per_group"] = per
    if verbose:
        print(f"[labels] 개체ID: 고유 {uniq:,}개, 개체당 평균 {per:.1f}장")
    if per < 1.5:
        msg = (
            "⚠️ 개체당 평균 장수가 1.5 미만입니다. 개체ID 추출이 실패했을 가능성이 큽니다.\n"
            "   이대로 분할하면 데이터 누수를 못 막습니다.\n"
            "   → dedup.py 의 phash 클러스터를 그룹 대용으로 쓰세요 (split.build 가 자동 처리)."
        )
        print(msg)
        df.attrs["animal_id_warning"] = msg
    return df


def report_manifest(df: pd.DataFrame) -> None:
    print("\n" + "─" * 58)
    print(f" 매니페스트 {len(df):,} 행")
    print("─" * 58)
    print("\n[클래스 분포]")
    vc = df["label"].value_counts(dropna=False)
    for k, v in vc.items():
        name = str(k)
        print(f"  {name:>6}: {v:>8,}  ({v / len(df):5.1%})")
    if len(vc) > 1:
        print(f"  불균형 비: {vc.max() / max(vc.min(), 1):.1f}배")

    for col in ("species", "camera", "symptom", "src_split"):
        if df[col].notna().any():
            print(f"\n[{col}] {df[col].value_counts(dropna=False).to_dict()}")

    ar = df["area_ratio"].dropna()
    if len(ar):
        print(f"\n[병변 면적] 중앙값 {ar.median():.2%}, 5% 미만 {(ar < 0.05).mean():.1%}")
    print(f"\n[기하 정보] bbox 있음 {df['bbox'].notna().mean():.1%}, "
          f"polygon 있음 {df['polygon'].notna().mean():.1%}")
    print("─" * 58 + "\n")


def _save(df: pd.DataFrame, path: Path) -> Path:
    """list 컬럼이 있어 parquet 이 까다로우므로 JSON 문자열로 직렬화해 저장."""
    out = df.copy()
    for c in ("bbox", "polygon"):
        if c in out.columns:
            out[c] = out[c].apply(lambda v: json.dumps(v) if isinstance(v, (list, tuple)) else None)
    try:
        out.to_parquet(path, index=False)
    except Exception:
        path = path.with_suffix(".csv")
        out.to_csv(path, index=False)
    print(f"[labels] 저장: {path}")
    return path


def load(path: Path | None = None) -> pd.DataFrame:
    """저장된 매니페스트를 되읽습니다 (list 컬럼 복원 포함)."""
    p = Path(path) if path else env.work_root() / "manifests" / "manifest_raw.parquet"
    if not p.exists() and p.with_suffix(".csv").exists():
        p = p.with_suffix(".csv")
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    for c in ("bbox", "polygon"):
        if c in df.columns:
            df[c] = df[c].apply(
                lambda v: json.loads(v) if isinstance(v, str) and v.startswith("[") else None
            )
    return df


def save(df: pd.DataFrame, name: str) -> Path:
    return _save(df, env.ensure_dirs()["manifests"] / name)
