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


# ══════════════════════════════════════════════════════════════
# AI Hub 561 실제 스키마 전용 파서
#
# 실물 JSON 을 확인해서 확정한 구조입니다 (diagnose.py 출력 기준).
# 이전에는 키를 추론했는데, 그 방식이 세 군데에서 조용히 틀렸습니다:
#   · polygon 이 점 목록이 아니라 {"x1":..,"y1":..,"x2":..} 평평한 dict → 추출률 0%
#   · 이미지 크기가 box 의 width/height 로 잡혀 병변 면적이 항상 100%
#   · 라벨을 폴더명에서 읽어 무증상 사진에 병변 라벨이 붙음
#
# 실제 구조:
#   metaData.lesions      A1~A6 = 병변, **A7 = 무증상**  ← 진짜 정답
#   metaData.Path         유증상 / 무증상
#   metaData.resolution   "1920X1080"  ← 이미지 크기는 여기 (문자열)
#   metaData.species      D = 개, C = 고양이  ← 폴더가 반려견이어도 고양이가 섞여 있음
#   metaData.합성유무      Y = 합성 이미지
#   labelingInfo[].polygon.location[0]  {"x1","y1",...,"xN","yN"}
#   labelingInfo[].box.location[0]      {"x","y","width","height"}
#
# ⚠️ 개체ID 필드가 없습니다. (breed, age, gender, date) 조합으로 대용합니다 —
#    아래 surrogate_animal_id() 주석 참고.
# ══════════════════════════════════════════════════════════════
NORMAL_LESION_CODE = "A7"      # AI Hub 561 에서 무증상을 뜻하는 코드


def _parse_resolution(s) -> tuple[int | None, int | None]:
    """'1920X1080' → (1920, 1080)."""
    if not isinstance(s, str):
        return (None, None)
    m = re.match(r"\s*(\d+)\s*[xX*×]\s*(\d+)\s*$", s)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def _numbered_points(loc) -> list[list[float]]:
    """{"x1":..,"y1":..,...} → [[x,y], ...]."""
    if isinstance(loc, list):
        loc = loc[0] if loc else {}
    if not isinstance(loc, dict):
        return []
    pts: list[list[float]] = []
    i = 1
    while f"x{i}" in loc and f"y{i}" in loc:
        try:
            pts.append([float(loc[f"x{i}"]), float(loc[f"y{i}"])])
        except (TypeError, ValueError):
            pass
        i += 1
    return pts


def _xywh_box(loc) -> list[float] | None:
    """{"x":..,"y":..,"width":..,"height":..} → [x1,y1,x2,y2]."""
    if isinstance(loc, list):
        loc = loc[0] if loc else {}
    if not isinstance(loc, dict):
        return None
    try:
        x, y = float(loc["x"]), float(loc["y"])
        w, h = float(loc["width"]), float(loc["height"])
    except (KeyError, TypeError, ValueError):
        return None
    return [x, y, x + w, y + h]


def surrogate_animal_id(meta: dict) -> str:
    """개체ID 대용 키.

    ⚠️ 이 데이터셋에는 개체(강아지) 식별자가 **없습니다.**
       'Raw data ID' 는 이미지마다 고유하고, 파일명 번호도 전역 일련번호입니다.
       그대로 두면 개체당 1장이 되어 그룹 분할이 무의미해집니다.

    그래서 (종, 견종, 나이, 성별, 촬영일) 조합을 개체 대용으로 씁니다.
    같은 날 촬영된 같은 견종·나이·성별 개체는 같은 강아지일 가능성이 매우 높습니다.

    이 방식은 **과하게 묶는 쪽으로 틀립니다** — 서로 다른 두 마리가 한 그룹이 될 수는
    있어도, 같은 강아지가 train/val 로 쪼개지지는 않습니다. 누수 방지 관점에서
    안전한 방향의 오차입니다. (부위 region 은 같은 개체라도 사진마다 달라 제외)
    """
    parts = [str(meta.get(k, "")).strip() for k in
             ("species", "breed", "age", "gender", "date")]
    return "G_" + "|".join(parts)


def parse_record_561(rec: dict, path_hint: str = "") -> dict | None:
    """AI Hub 561 레코드 하나를 표준 필드로 변환합니다. 스키마가 다르면 None."""
    meta = rec.get("metaData")
    if not isinstance(meta, dict):
        return None

    lesion = str(meta.get("lesions", "")).strip().upper()
    if not re.fullmatch(r"A[1-7]", lesion):
        # 폴더명에서라도 건져봅니다 (metaData 가 비어 있는 소수 케이스)
        m = re.search(r"\b(A[1-7])\b", path_hint)
        lesion = m.group(1) if m else ""

    w, h = _parse_resolution(meta.get("resolution"))

    polygon: list[list[float]] | None = None
    bbox: list[float] | None = None
    n_lesion = 0
    for item in rec.get("labelingInfo") or []:
        if not isinstance(item, dict):
            continue
        if "polygon" in item and polygon is None:
            pts = _numbered_points(item["polygon"].get("location"))
            if len(pts) >= 3:
                polygon = pts
        if "box" in item and bbox is None:
            bbox = _xywh_box(item["box"].get("location"))
        n_lesion += 1

    if bbox is None and polygon:
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        bbox = [min(xs), min(ys), max(xs), max(ys)]

    area = None
    if bbox and w and h:
        area = min(max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / (w * h), 1.0)

    name = str(meta.get("Raw data ID") or "").strip()
    return {
        "image_name": Path(name).name if name else None,
        "label": lesion or None,
        "is_normal": lesion == NORMAL_LESION_CODE,
        "animal_id": surrogate_animal_id(meta),
        "species_code": str(meta.get("species", "")).strip().upper(),
        "breed": meta.get("breed"),
        "age": meta.get("age"),
        "gender": meta.get("gender"),
        "region": meta.get("region"),
        "date": meta.get("date"),
        "synthetic": str(meta.get("합성유무", "")).strip().upper() == "Y",
        "symptom_meta": meta.get("Path"),
        "bbox": bbox,
        "polygon": polygon,
        "img_w": w,
        "img_h": h,
        "area_ratio": area,
        "n_lesion": n_lesion,
    }


# ──────────────────────────────────────────────────────────────
# 값 추출 — 키 이름을 모르는 상태에서 최대한 버티기 (561 파서 실패 시 대비)
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
def build_from_zip(
    zip_path: Path | str,
    animal_token_index: int | None = None,
    dogs_only: bool = True,
    normal_camera_only: bool = True,
    save: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """zip 을 **풀지 않고** 그 안의 JSON 을 직접 읽어 매니페스트를 만듭니다.

    왜: 90GB zip 을 풀면 추가로 50GB 가 필요해서 디스크가 부족한 경우가 많습니다.
        zip 은 중앙 디렉터리가 있어 멤버 단위 임의 접근이 싸므로,
        압축을 풀지 않고 바로 읽는 편이 디스크를 아끼면서도 충분히 빠릅니다.

    image_path 대신 `zip_member` 컬럼에 zip 내부 경로를 담습니다.
    crop.run() 이 이 컬럼을 보고 zip 에서 직접 이미지를 읽습니다.
    """
    import zipfile

    zip_path = Path(zip_path)
    rows: list[dict] = []

    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        jsons = [n for n in names if n.lower().endswith(".json")]
        # ⚠️ AI Hub 는 Training 과 Validation 에 같은 파일명을 재사용합니다.
        #    파일명 하나에 이미지 하나만 기억하면 Validation 라벨이 Training 이미지에
        #    잘못 붙어서, split 이 전부 train 으로 뭉개집니다.
        #    → 후보를 전부 모아두고 경로가 가장 비슷한 것을 고릅니다.
        img_by_stem: dict[str, list[str]] = defaultdict(list)
        for n in names:
            if Path(n).suffix.lower() in IMG_EXT:
                img_by_stem[Path(n).name].append(n)
                img_by_stem[Path(n).stem].append(n)

        if verbose:
            print(f"[labels] {zip_path.name}: JSON {len(jsons):,}개 / "
                  f"이미지 {sum(1 for n in names if Path(n).suffix.lower() in IMG_EXT):,}장")

        unmatched = 0
        n_generic = 0
        for jn in tqdm_maybe(jsons, verbose):
            try:
                data = json.loads(z.read(jn).decode("utf-8-sig"))
            except Exception:
                continue
            axes = path_axes(Path(jn))

            for rec in (data if isinstance(data, list) else [data]):
                if not isinstance(rec, dict):
                    continue
                iname = extract_image_name(rec) or Path(jn).stem
                cands = img_by_stem.get(iname) or img_by_stem.get(Path(iname).stem) or []
                if not cands:
                    unmatched += 1
                    continue
                # 라벨 JSON 과 경로 구성요소가 가장 많이 겹치는 이미지를 고릅니다
                # (.../2.라벨링데이터/TL01/... ↔ .../1.원천데이터/TS01/... 를 맞춰줌)
                jparts = set(Path(jn).parts)
                member = max(cands, key=lambda c: len(jparts & set(Path(c).parts)))

                iaxes = path_axes(Path(member))
                merged = {k: (iaxes[k] or axes[k]) for k in axes}

                # 실제 스키마 파서를 먼저 시도하고, 안 맞으면 추론 방식으로 넘어갑니다.
                parsed = parse_record_561(rec, member)
                if parsed is None:
                    label = extract_label(rec, member)
                    w, h = extract_wh(rec)
                    bbox, polygon = extract_geometry(rec)
                    area = (min(max(0.0, (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])) / (w*h), 1.0)
                            if bbox and w and h else None)
                    parsed = {
                        "image_name": Path(member).name, "label": label,
                        "is_normal": merged["symptom"] == "무증상",
                        "animal_id": extract_animal_id(rec, Path(member).name,
                                                       animal_token_index),
                        "bbox": bbox, "polygon": polygon,
                        "img_w": w, "img_h": h, "area_ratio": area,
                    }
                    n_generic += 1

                parsed["image_name"] = Path(member).name    # zip 내부 실제 파일명으로 확정
                rows.append({
                    "image_path": f"{zip_path}!{member}",   # 사람이 읽기 위한 표기
                    "zip_path": str(zip_path),
                    "zip_member": member,
                    "json_path": f"{zip_path}!{jn}",
                    **merged,
                    **parsed,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"{zip_path} 에서 매니페스트를 만들지 못했습니다.")
    if verbose:
        print(f"[labels] 원시 행 {len(df):,}개 (이미지 매칭 실패 {unmatched:,}건)")
        if n_generic:
            print(f"[labels] ⚠️ {n_generic:,}건은 561 스키마가 아니라 추론 방식으로 처리")
        if "synthetic" in df.columns:
            print(f"[labels] 합성 이미지: {int(df['synthetic'].sum()):,}건")

    df = _filter_scope(df, dogs_only, normal_camera_only, verbose)
    df = drop_unlabeled(df, verbose)
    df = _finalize_animal_id(df, verbose)
    if save:
        _save(df, env.ensure_dirs()["manifests"] / f"raw_{zip_path.stem}.parquet")
    if verbose:
        report_manifest(df)
    return df


def tqdm_maybe(seq, verbose: bool = True):
    if not verbose:
        return seq
    try:
        from tqdm.auto import tqdm

        return tqdm(seq, desc="json")
    except ImportError:
        return seq


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
    df = drop_unlabeled(df, verbose)
    df = _fill_missing_wh(df, verbose)
    df = _finalize_animal_id(df, verbose)

    if save:
        out = env.ensure_dirs()["manifests"] / "manifest_raw.parquet"
        _save(df, out)
    if verbose:
        report_manifest(df)
    return df


def drop_unlabeled(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """라벨이 없는 행을 버립니다.

    ⚠️ 이걸 안 하면 label=NaN 행이 fold 에 'NA' 라는 가짜 클래스로 들어가
       학습·평가 지표를 오염시킵니다 (실측: 598행이 fold 마다 99행씩 배분됨).
       라벨이 없으면 학습에 쓸 수 없으니 조용히 섞이게 두면 안 됩니다.
    """
    bad = df["label"].isna() | (df["label"].astype(str).str.strip() == "")
    if not bad.any():
        return df
    if verbose:
        print(f"[labels] 라벨 없는 {int(bad.sum()):,}행 제외")
        cols = [c for c in ("camera", "symptom", "species_code", "zip_member")
                if c in df.columns]
        if cols:
            ex = df[bad][cols].head(3)
            for _, r in ex.iterrows():
                vals = ", ".join(f"{c}={r[c]}" for c in cols if c != "zip_member")
                mem = str(r.get("zip_member", ""))[:70]
                print(f"         {vals}  {mem}")
    return df[~bad].reset_index(drop=True)


def _filter_scope(df: pd.DataFrame, dogs_only: bool, normal_camera_only: bool,
                  verbose: bool) -> pd.DataFrame:
    before = len(df)

    # ⚠️ 폴더가 '반려견' 이어도 고양이 이미지가 섞여 있습니다 (IMG_C_A7_*.json 확인).
    #    폴더명이 아니라 metaData.species 로 걸러야 합니다.
    if dogs_only and "species_code" in df.columns and df["species_code"].notna().any():
        n_cat = int((df["species_code"] == "C").sum())
        df = df[df["species_code"] == "D"]
        if verbose and n_cat:
            print(f"[labels] 반려견 폴더 안의 고양이 {n_cat:,}건 제외 "
                  "(metaData.species 기준)")

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


def load(path: Path | None = None, rebase: bool = True) -> pd.DataFrame:
    """저장된 매니페스트를 되읽습니다 (list 컬럼 복원 + 경로 재조정 포함)."""
    p = Path(path) if path else env.work_root() / "manifests" / "manifest_raw.parquet"
    if not p.exists() and p.with_suffix(".csv").exists():
        p = p.with_suffix(".csv")
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    for c in ("bbox", "polygon"):
        if c in df.columns:
            df[c] = df[c].apply(
                lambda v: json.loads(v) if isinstance(v, str) and v.startswith("[") else None
            )
    return rebase_paths(df) if rebase else df


def rebase_paths(df: pd.DataFrame) -> pd.DataFrame:
    """다른 기기에서 만든 매니페스트의 크롭 경로를 현재 환경에 맞춥니다.

    로컬 PC 에서 전처리 → Drive/Kaggle 업로드 → Colab 학습 흐름에서
    절대경로가 전부 깨지므로, 상대경로(crop_rel)로 다시 조립합니다.
    """
    if "crop_rel" not in df.columns:
        return df
    root = env.work_root() / "crops"
    df = df.copy()
    df["crop_path"] = df["crop_rel"].apply(
        lambda r: str(root / r) if isinstance(r, str) else None
    )
    missing = df["crop_path"].apply(lambda p: p is not None and not Path(p).exists())
    if missing.any():
        print(f"⚠️ 크롭 파일 {missing.sum():,}/{len(df):,}개를 찾을 수 없습니다.")
        print(f"   찾는 위치: {root}")
        print("   압축을 이 경로로 풀었는지, DOG_SKIN_WORK 환경변수가 맞는지 확인하세요.")
    return df


def combine(paths: list[Path] | None = None, pattern: str = "chunk_*.parquet",
            verbose: bool = True) -> pd.DataFrame:
    """청크별 매니페스트를 하나로 합칩니다.

    ⚠️ 왜 필요한가: 90GB zip 을 나눠 받아 청크마다 따로 전처리하면,
       **같은 강아지가 여러 청크에 흩어져** 있을 수 있습니다.
       청크별로 분할하면 개체가 train/val 에 걸쳐 데이터 누수가 생깁니다.
       그래서 크롭까지만 청크별로 하고, **중복제거·개체분할은 전부 합친 뒤 한 번에** 합니다.
    """
    mdir = env.work_root() / "manifests"
    files = [Path(p) for p in paths] if paths else sorted(mdir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"합칠 매니페스트가 없습니다: {mdir}/{pattern}")

    dfs = []
    for f in files:
        d = load(f, rebase=False)
        d["chunk"] = f.stem
        dfs.append(d)
        if verbose:
            print(f"  {f.name}: {len(d):,}행")

    df = pd.concat(dfs, ignore_index=True)
    before = len(df)
    # 같은 청크를 두 번 돌린 경우만 걸러냅니다.
    # ⚠️ image_name 만으로 지우면 안 됩니다 — AI Hub 는 Training 과 Validation 에
    #    같은 파일명을 재사용해서, 실제로는 다른 이미지가 통째로 날아갑니다.
    #    내용이 같은 진짜 중복은 뒤의 phash dedup 이 잡습니다.
    key = [c for c in ("image_name", "src_split", "label", "animal_id", "img_w", "img_h")
           if c in df.columns]
    df = df.drop_duplicates(subset=key, keep="first").reset_index(drop=True)
    df = drop_unlabeled(df, verbose)
    if verbose:
        print(f"\n[labels] 합계 {before:,} → 정리 후 {len(df):,}행 (중복 기준: {key})")
        if "chunk" in df.columns:
            print(f"[labels] 청크별: {df['chunk'].value_counts().to_dict()}")
        print(f"[labels] 개체 {df['animal_id'].nunique():,}마리")
    return rebase_paths(df)


def save(df: pd.DataFrame, name: str) -> Path:
    return _save(df, env.ensure_dirs()["manifests"] / name)
