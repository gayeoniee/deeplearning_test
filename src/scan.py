"""데이터 스캐너 — 스키마를 "가정"하지 않고 "추론"합니다.

이 프로젝트를 만들 때 AI Hub 원본 페이지에 접근할 수 없었기 때문에,
JSON 키 이름이나 폴더 구조를 코드에 박아두면 거의 확실히 틀립니다.
그래서 STEP 2 는 실물을 훑어서 스키마를 알아내는 것부터 시작합니다.

    from src import scan
    rep = scan.run()            # 전체 스캔 (수 분)
    scan.write_dataset_card(rep)

출력:
  - work/reports/scan_report.json   기계가 읽을 결과 (labels.py 가 씀)
  - docs/data/DATASET_CARD.md       사람이 읽을 요약
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src import env

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 우리가 찾고 싶은 필드의 "후보 키워드". 정확한 키 이름을 몰라도
# 이 단어들이 들어간 키를 우선 후보로 올립니다.
FIELD_HINTS: dict[str, list[str]] = {
    "label": ["label", "class", "disease", "lesion", "질환", "증상", "라벨", "code", "diag"],
    "polygon": ["polygon", "poly", "segmentation", "points", "contour", "좌표"],
    "bbox": ["bbox", "box", "rect", "location", "region", "roi", "x1", "xmin"],
    "image_name": ["file", "image", "img", "name", "이미지", "파일"],
    "animal_id": ["pet", "animal", "individual", "subject", "id", "no", "번호", "개체", "환자", "chart"],
    "breed": ["breed", "species", "품종", "견종"],
    "age": ["age", "나이", "birth"],
    "camera": ["camera", "device", "카메라", "장비"],
    "width": ["width", "가로", "너비"],
    "height": ["height", "세로", "높이"],
}

# 폴더 트리에서 우리가 아는 축
AXIS_TOKENS = {
    "species": ["반려견", "반려묘", "dog", "cat"],
    "camera": ["일반카메라", "더모스코프", "microscope", "dermoscope"],
    "symptom": ["유증상", "무증상"],
    "split": ["Training", "Validation", "Test", "TS", "TL", "VS", "VL"],
    "class": ["A1", "A2", "A3", "A4", "A5", "A6", "A7"],
}


# ──────────────────────────────────────────────────────────────
# JSON 평탄화
# ──────────────────────────────────────────────────────────────
def flatten(obj: Any, prefix: str = "", out: dict[str, Any] | None = None,
            max_depth: int = 8) -> dict[str, Any]:
    """중첩 dict/list 를 'a.b[].c' 형태의 키 경로로 폅니다."""
    if out is None:
        out = {}
    if max_depth <= 0:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(v, f"{prefix}.{k}" if prefix else str(k), out, max_depth - 1)
    elif isinstance(obj, list):
        if obj:
            # 리스트는 첫 원소만 대표로 봅니다 (구조 파악이 목적)
            flatten(obj[0], f"{prefix}[]", out, max_depth - 1)
        else:
            out[f"{prefix}[]"] = None
    else:
        out[prefix] = obj
    return out


def _short(v: Any, n: int = 60) -> str:
    s = str(v)
    return s if len(s) <= n else s[: n - 3] + "..."


# ──────────────────────────────────────────────────────────────
# 리포트 자료구조
# ──────────────────────────────────────────────────────────────
@dataclass
class ScanReport:
    root: str = ""
    scanned_at: str = ""
    n_images: int = 0
    n_jsons: int = 0
    size_gb: float = 0.0
    tree: dict[str, list[str]] = field(default_factory=dict)          # depth -> segments
    axes: dict[str, dict[str, int]] = field(default_factory=dict)     # species/camera/... -> count
    has_normal: bool = False
    json_keys: list[dict] = field(default_factory=list)               # 키 경로 통계
    field_guess: dict[str, str | None] = field(default_factory=dict)  # 역할 -> 추정 키
    class_counts: dict[str, int] = field(default_factory=dict)
    resolutions: dict[str, int] = field(default_factory=dict)
    filename_patterns: list[tuple[str, int]] = field(default_factory=list)
    animal_id_stats: dict[str, Any] = field(default_factory=dict)
    lesion_area: dict[str, Any] = field(default_factory=dict)
    dup_estimate: dict[str, Any] = field(default_factory=dict)
    sample_json: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)

    def save(self, path: Path | None = None) -> Path:
        p = path or (env.ensure_dirs()["reports"] / "scan_report.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str),
                     encoding="utf-8")
        print(f"[scan] 저장: {p}")
        return p

    @staticmethod
    def load(path: Path | None = None) -> "ScanReport":
        p = path or (env.work_root() / "reports" / "scan_report.json")
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        r = ScanReport()
        for k, v in data.items():
            if hasattr(r, k):
                setattr(r, k, v)
        return r


# ──────────────────────────────────────────────────────────────
# 개별 스캔 단계
# ──────────────────────────────────────────────────────────────
def _walk(root: Path) -> tuple[list[Path], list[Path], int]:
    imgs, jsons, size = [], [], 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        try:
            size += p.stat().st_size
        except OSError:
            continue
        s = p.suffix.lower()
        if s in IMG_EXT:
            imgs.append(p)
        elif s == ".json":
            jsons.append(p)
    return imgs, jsons, size


def scan_tree(root: Path, files: Iterable[Path], rep: ScanReport) -> None:
    """폴더 깊이별로 어떤 이름들이 나타나는지 집계합니다."""
    depth_seg: dict[int, Counter] = defaultdict(Counter)
    axis_hits: dict[str, Counter] = {k: Counter() for k in AXIS_TOKENS}

    for p in files:
        try:
            parts = p.relative_to(root).parts[:-1]
        except ValueError:
            continue
        for d, seg in enumerate(parts):
            depth_seg[d][seg] += 1
        joined = "/".join(parts)
        for axis, tokens in AXIS_TOKENS.items():
            for t in tokens:
                if t in joined:
                    axis_hits[axis][t] += 1

    rep.tree = {
        f"depth{d}": [f"{name} ({cnt:,})" for name, cnt in c.most_common(15)]
        for d, c in sorted(depth_seg.items())
    }
    rep.axes = {k: dict(v.most_common()) for k, v in axis_hits.items() if v}
    rep.has_normal = axis_hits["symptom"].get("무증상", 0) > 0


def scan_json_schema(jsons: list[Path], rep: ScanReport, sample: int = 300) -> None:
    """JSON 키 경로를 평탄화해 빈도·타입·예시를 모읍니다."""
    if not jsons:
        rep.warnings.append("JSON 라벨 파일을 하나도 못 찾았습니다. 라벨링데이터(TL/VL)를 받았는지 확인하세요.")
        return

    picks = random.sample(jsons, min(sample, len(jsons)))
    key_count: Counter = Counter()
    key_types: dict[str, Counter] = defaultdict(Counter)
    key_examples: dict[str, list] = defaultdict(list)
    first_raw = None

    for jp in picks:
        try:
            data = json.loads(jp.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            rep.warnings.append(f"JSON 파싱 실패 예: {jp.name} — {exc}")
            continue
        if first_raw is None:
            first_raw = data
        # 파일이 리스트로 감싸여 있을 수도 있음
        records = data if isinstance(data, list) else [data]
        for rec in records[:3]:
            flat = flatten(rec)
            for k, v in flat.items():
                key_count[k] += 1
                key_types[k][type(v).__name__] += 1
                if len(key_examples[k]) < 3 and v is not None:
                    key_examples[k].append(_short(v))

    n = max(len(picks), 1)
    rep.json_keys = [
        {
            "key": k,
            "freq": round(c / n, 3),
            "types": dict(key_types[k]),
            "examples": key_examples[k],
        }
        for k, c in key_count.most_common(120)
    ]
    rep.sample_json = first_raw if isinstance(first_raw, dict) else {"_list_first": first_raw}

    # 역할별 후보 지목
    guess: dict[str, str | None] = {}
    for role, hints in FIELD_HINTS.items():
        best, best_score = None, 0.0
        for k, c in key_count.items():
            kl = k.lower()
            leaf = kl.split(".")[-1]
            score = 0.0
            for h in hints:
                if leaf == h.lower():
                    score += 3.0
                elif h.lower() in leaf:
                    score += 1.5
                elif h.lower() in kl:
                    score += 0.5
            if score == 0:
                continue
            score *= c / n                      # 자주 나오는 키에 가중
            score += _role_type_bonus(role, key_types[k], key_examples[k])
            if score > best_score:
                best, best_score = k, score
        guess[role] = best
    rep.field_guess = guess


def _role_type_bonus(role: str, types: Counter, examples: list) -> float:
    """타입/예시값 모양으로 후보 점수를 보정합니다."""
    tnames = set(types)
    ex = " ".join(str(e) for e in examples)
    if role in {"width", "height"} and tnames & {"int", "float"}:
        return 1.0
    if role == "polygon" and ("list" in tnames or re.search(r"\d+\s*,\s*\d+", ex)):
        return 1.5
    if role == "bbox" and (tnames & {"int", "float", "list"}):
        return 1.0
    if role == "image_name" and re.search(r"\.(jpg|jpeg|png)", ex, re.I):
        return 3.0
    if role == "label" and re.search(r"\bA[1-7]\b", ex):
        return 3.0
    if role == "animal_id" and tnames & {"str", "int"} and len(ex) < 40:
        return 0.5
    return 0.0


def scan_classes_and_names(imgs: list[Path], rep: ScanReport) -> None:
    """폴더명과 파일명에서 클래스와 개체ID 후보를 뽑습니다."""
    cls_c: Counter = Counter()
    pattern_c: Counter = Counter()

    for p in imgs:
        m = re.search(r"(?:^|[/_\\-])(A[0-7])(?:[_/\\-]|$)", str(p))
        if m:
            cls_c[m.group(1)] += 1
        # 파일명 구조를 정규화: 숫자→#, 영문→L
        norm = re.sub(r"\d+", "#", p.stem)
        norm = re.sub(r"[A-Za-z]+", "L", norm)
        pattern_c[norm] += 1

    rep.class_counts = dict(cls_c.most_common())
    rep.filename_patterns = pattern_c.most_common(10)

    if not cls_c:
        rep.warnings.append("파일 경로에서 A1~A6 클래스 코드를 못 찾았습니다. JSON 라벨 필드를 써야 합니다.")

    # 파일명에서 개체ID로 쓸 만한 토큰 후보 찾기
    #  - 같은 값이 여러 장에 반복되고(그룹 역할)
    #  - 고유값 수가 이미지 수보다 충분히 적을 것
    tok_sets: dict[int, Counter] = defaultdict(Counter)
    for p in imgs[:50_000]:
        toks = re.split(r"[_\-.]", p.stem)
        for i, t in enumerate(toks[:8]):
            tok_sets[i][t] += 1
    cands = []
    for i, c in tok_sets.items():
        uniq, total = len(c), sum(c.values())
        if total < 50 or uniq < 2:
            continue
        per = total / uniq
        # 개체ID 후보의 조건:
        #  - 개체당 2장 이상 (per<2 면 사실상 이미지 고유값 → 그룹 역할을 못 함)
        #  - 고유값이 전체의 절반 미만
        #  - 클래스 코드(A1..A7)나 종 구분자처럼 값이 몇 개뿐인 토큰은 제외
        if per < 2.0 or uniq >= total * 0.5 or uniq < 5:
            continue
        vals = [k for k, _ in c.most_common(3)]
        if all(re.fullmatch(r"A[0-7]", v) for v in vals):
            continue
        cands.append({"token_index": i, "unique": uniq, "images": total,
                      "avg_per_group": round(per, 1), "examples": vals})
    # 그룹 역할을 하면서도 가장 잘게 나뉘는 토큰이 개체ID 에 가깝습니다.
    # (avg_per_group 이 큰 순이 아니라 unique 가 많은 순으로 골라야 '견/묘' 같은 걸 안 뽑습니다)
    rep.animal_id_stats = {
        "filename_token_candidates": sorted(cands, key=lambda d: -d["unique"])[:5],
        "json_field_guess": rep.field_guess.get("animal_id"),
    }


def scan_images(imgs: list[Path], rep: ScanReport, sample: int = 400) -> None:
    """해상도 분포를 샘플링합니다."""
    try:
        from PIL import Image
    except ImportError:
        rep.warnings.append("Pillow 미설치 — 해상도 스캔을 건너뜁니다.")
        return
    picks = random.sample(imgs, min(sample, len(imgs)))
    res: Counter = Counter()
    for p in picks:
        try:
            with Image.open(p) as im:
                res[f"{im.width}x{im.height}"] += 1
        except Exception:
            continue
    rep.resolutions = dict(res.most_common(15))


def scan_lesion_area(jsons: list[Path], rep: ScanReport, sample: int = 500) -> None:
    """병변이 이미지에서 차지하는 면적 비율 분포.

    선행 프로젝트가 "93%가 이미지의 5% 미만"이라고 보고했는데, 사실이면
    전체 이미지로 학습하면 안 되고 ROI 크롭이 필수가 됩니다. 실물로 검증합니다.
    """
    if not jsons:
        return
    picks = random.sample(jsons, min(sample, len(jsons)))
    ratios: list[float] = []

    for jp in picks:
        try:
            data = json.loads(jp.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        for rec in (data if isinstance(data, list) else [data]):
            r = _area_ratio(rec)
            if r is not None:
                ratios.append(r)

    if not ratios:
        rep.warnings.append("병변 면적을 계산할 좌표를 못 찾았습니다. field_guess 의 polygon/bbox 후보를 확인하세요.")
        return

    ratios.sort()
    def q(p: float) -> float:
        return round(ratios[min(int(len(ratios) * p), len(ratios) - 1)], 4)

    rep.lesion_area = {
        "n": len(ratios),
        "p10": q(0.10), "p25": q(0.25), "median": q(0.50),
        "p75": q(0.75), "p90": q(0.90),
        "under_5pct": round(sum(r < 0.05 for r in ratios) / len(ratios), 3),
        "under_1pct": round(sum(r < 0.01 for r in ratios) / len(ratios), 3),
    }


def _collect_numbers(obj: Any, keys: list[str], out: list) -> None:
    """중첩 구조 어디에 있든 지정 키워드가 든 키의 값을 긁어옵니다."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if any(h in k.lower() for h in keys):
                out.append(v)
            _collect_numbers(v, keys, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_numbers(v, keys, out)


def _area_ratio(rec: dict) -> float | None:
    """레코드 하나에서 (병변 면적 / 이미지 면적) 을 추정합니다."""
    wh: list = []
    _collect_numbers(rec, ["width", "가로"], wh)
    hh: list = []
    _collect_numbers(rec, ["height", "세로"], hh)
    W = next((float(x) for x in wh if isinstance(x, (int, float)) and x > 1), None)
    H = next((float(x) for x in hh if isinstance(x, (int, float)) and x > 1), None)
    if not W or not H:
        return None

    # bbox 먼저
    box: list = []
    _collect_numbers(rec, ["bbox", "box", "rect", "location", "region"], box)
    for b in box:
        pts = _as_xy(b)
        if len(pts) >= 2:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            a = (max(xs) - min(xs)) * (max(ys) - min(ys))
            if a > 0:
                return min(a / (W * H), 1.0)

    # polygon
    poly: list = []
    _collect_numbers(rec, ["polygon", "poly", "points", "segmentation"], poly)
    for pg in poly:
        pts = _as_xy(pg)
        if len(pts) >= 3:
            a = abs(sum(pts[i][0] * pts[(i + 1) % len(pts)][1]
                        - pts[(i + 1) % len(pts)][0] * pts[i][1]
                        for i in range(len(pts)))) / 2
            if a > 0:
                return min(a / (W * H), 1.0)
    return None


def _as_xy(v: Any) -> list[tuple[float, float]]:
    """다양한 좌표 표현을 [(x,y), ...] 로 정규화합니다.

    지원: [[x,y],...] / [{"x":..,"y":..},...] / [x1,y1,x2,y2,...] / "x1,y1 x2,y2"
    """
    pts: list[tuple[float, float]] = []
    if isinstance(v, str):
        nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", v)]
        v = nums
    if isinstance(v, dict):
        keys = {k.lower(): k for k in v}
        # AI Hub 561 폴리곤: {"x1":..,"y1":..,"x2":..,"y2":.., ... "x111":..,"y111":..}
        # 점 목록이 아니라 번호가 붙은 평평한 dict 입니다. 이걸 못 읽어서
        # polygon 추출률이 0% 로 나왔습니다.
        if "x1" in keys and "y1" in keys:
            pts: list[tuple[float, float]] = []
            i = 1
            while f"x{i}" in keys and f"y{i}" in keys:
                try:
                    pts.append((float(v[keys[f"x{i}"]]), float(v[keys[f"y{i}"]])))
                except (TypeError, ValueError):
                    pass
                i += 1
            if len(pts) >= 2:
                return pts
        if "x" in keys and "y" in keys:
            try:
                x, y = float(v[keys["x"]]), float(v[keys["y"]])
            except (TypeError, ValueError):
                return []
            # AI Hub 박스는 {"x","y","width","height"} 형태가 흔합니다 → 두 꼭짓점으로
            if "width" in keys and "height" in keys:
                try:
                    return [(x, y), (x + float(v[keys["width"]]), y + float(v[keys["height"]]))]
                except (TypeError, ValueError):
                    return [(x, y)]
            return [(x, y)]
        # {"x1":..,"y1":..,"x2":..,"y2":..}
        if {"x1", "y1", "x2", "y2"} <= set(keys):
            try:
                return [(float(v[keys["x1"]]), float(v[keys["y1"]])),
                        (float(v[keys["x2"]]), float(v[keys["y2"]]))]
            except (TypeError, ValueError):
                return []
        # AI Hub 는 좌표를 한 겹 감싸는 경우가 흔합니다:
        #   {"polygon": {"location": [{"x":..,"y":..}, ...]}}
        # 알려진 래퍼 키를 먼저 열어보고, 없으면 리스트 값 하나를 열어봅니다.
        for wrapper in ("location", "points", "coordinates", "vertices", "coord", "value", "data"):
            if wrapper in keys:
                pts = _as_xy(v[keys[wrapper]])
                if pts:
                    return pts
        for val in v.values():
            if isinstance(val, (list, tuple)) and val:
                pts = _as_xy(val)
                if pts:
                    return pts
        return []
    if isinstance(v, (list, tuple)):
        if v and isinstance(v[0], dict):
            for d in v:
                pts.extend(_as_xy(d))
            return pts
        if v and isinstance(v[0], (list, tuple)) and len(v[0]) >= 2:
            for p in v:
                try:
                    pts.append((float(p[0]), float(p[1])))
                except (TypeError, ValueError, IndexError):
                    continue
            return pts
        nums = [x for x in v if isinstance(x, (int, float))]
        if len(nums) >= 4 and len(nums) % 2 == 0:
            return [(float(nums[i]), float(nums[i + 1])) for i in range(0, len(nums), 2)]
    return pts


def scan_duplicates(imgs: list[Path], rep: ScanReport, sample: int = 3000) -> None:
    """중복률을 미리 재봅니다 (본 제거는 dedup.py 에서)."""
    try:
        import imagehash
        from PIL import Image
    except ImportError:
        rep.warnings.append("imagehash 미설치 — 중복 추정을 건너뜁니다. pip install imagehash")
        return

    picks = random.sample(imgs, min(sample, len(imgs)))
    seen: dict[str, list[Path]] = defaultdict(list)
    for p in picks:
        try:
            with Image.open(p) as im:
                h = str(imagehash.phash(im.convert("RGB"), hash_size=8))
        except Exception:
            continue
        seen[h].append(p)

    groups = [v for v in seen.values() if len(v) > 1]
    dup_imgs = sum(len(g) - 1 for g in groups)
    # 서로 다른 클래스 폴더에 같은 해시가 걸친 경우 = 가장 위험한 오염
    cross = 0
    for g in groups:
        cls = {m.group(1) for p in g if (m := re.search(r"(A[1-7])", str(p)))}
        if len(cls) > 1:
            cross += 1

    rep.dup_estimate = {
        "sampled": len(picks),
        "exact_phash_groups": len(groups),
        "duplicate_rate": round(dup_imgs / max(len(picks), 1), 4),
        "cross_class_groups": cross,
    }
    if cross:
        rep.warnings.append(
            f"⚠️ 서로 다른 클래스에 동일 이미지가 {cross}건 발견됨 (샘플 {len(picks)}장 기준). "
            "선행 프로젝트가 실패한 원인입니다 — dedup.py 로 반드시 제거하세요."
        )


# ──────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────
def run(root: Path | None = None, seed: int = 42, quick: bool = False) -> ScanReport:
    """전체 스캔. quick=True 면 샘플 수를 줄여 빠르게 훑습니다."""
    random.seed(seed)
    root = Path(root) if root else env.data_root()
    if not root.exists():
        raise FileNotFoundError(f"데이터 폴더가 없습니다: {root}\n  STEP 1(다운로드)을 먼저 실행하세요.")

    rep = ScanReport(root=str(root), scanned_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    print(f"[scan] 훑는 중: {root}")
    imgs, jsons, size = _walk(root)
    rep.n_images, rep.n_jsons, rep.size_gb = len(imgs), len(jsons), round(size / 1024**3, 2)
    print(f"[scan] 이미지 {len(imgs):,}장 / JSON {len(jsons):,}개 / {rep.size_gb} GB")

    if not imgs and not jsons:
        rep.warnings.append("파일을 하나도 못 찾았습니다. 압축 해제 여부를 확인하세요 (aihub.unpack_all()).")
        return rep

    s = 0.3 if quick else 1.0
    print("[scan] 1/6 폴더 구조…");    scan_tree(root, imgs + jsons, rep)
    print("[scan] 2/6 JSON 스키마…");  scan_json_schema(jsons, rep, sample=int(300 * s))
    print("[scan] 3/6 클래스·파일명…"); scan_classes_and_names(imgs, rep)
    print("[scan] 4/6 해상도…");       scan_images(imgs, rep, sample=int(400 * s))
    print("[scan] 5/6 병변 면적…");    scan_lesion_area(jsons, rep, sample=int(500 * s))
    print("[scan] 6/6 중복 추정…");    scan_duplicates(imgs, rep, sample=int(3000 * s))

    rep.save()
    summary(rep)
    return rep


def summary(rep: ScanReport) -> None:
    """콘솔에 사람이 읽을 요약을 찍습니다."""
    p = print
    p("\n" + "=" * 66)
    p(f" 스캔 요약 — 이미지 {rep.n_images:,} / JSON {rep.n_jsons:,} / {rep.size_gb} GB")
    p("=" * 66)

    p("\n[폴더 축]")
    for axis, hits in rep.axes.items():
        p(f"  {axis:9} : {hits}")
    p(f"  무증상(정상) 데이터 존재 여부 → {'있음 ✅' if rep.has_normal else '없음 ❌'}")

    p("\n[클래스별 이미지 수 — 경로 기준]")
    tot = sum(rep.class_counts.values()) or 1
    for c, n in rep.class_counts.items():
        p(f"  {c}: {n:>8,}  ({n / tot:5.1%})")

    p("\n[JSON 필드 자동 추정]")
    for role, key in rep.field_guess.items():
        p(f"  {role:11} → {key}")

    p("\n[JSON 키 상위 25]")
    for k in rep.json_keys[:25]:
        p(f"  {k['freq']:>5.0%}  {k['key']:<44} {list(k['types'])}  예: {k['examples'][:1]}")

    if rep.lesion_area:
        la = rep.lesion_area
        p("\n[병변 면적 비율]")
        p(f"  중앙값 {la['median']:.1%}  |  5% 미만 {la['under_5pct']:.1%}  |  1% 미만 {la['under_1pct']:.1%}")
        if la["under_5pct"] > 0.5:
            p("  → ROI 크롭이 필수입니다. 전체 이미지로 학습하면 배경만 배웁니다.")

    if rep.dup_estimate:
        d = rep.dup_estimate
        p("\n[중복 추정]")
        p(f"  샘플 {d['sampled']:,}장 중 중복률 {d['duplicate_rate']:.2%}, "
          f"클래스 간 중복 그룹 {d['cross_class_groups']}건")

    if rep.animal_id_stats.get("filename_token_candidates"):
        p("\n[개체ID 후보 — 파일명 토큰]")
        for c in rep.animal_id_stats["filename_token_candidates"]:
            p(f"  토큰#{c['token_index']}: 고유 {c['unique']:,}개, 그룹당 평균 {c['avg_per_group']}장, "
              f"예 {c['examples']}")

    if rep.warnings:
        p("\n[경고]")
        for w in rep.warnings:
            p(f"  ⚠️ {w}")

    p("\n" + "=" * 66)
    p(" 다음 할 일: 이 출력을 그대로 복사해서 공유하면 labels.py/split.py 를 확정합니다.")
    p("=" * 66 + "\n")


def write_dataset_card(rep: ScanReport, path: Path | None = None) -> Path:
    """스캔 결과를 docs/data/DATASET_CARD.md 로 씁니다."""
    p = Path(path) if path else env.project_root() / "docs" / "data" / "DATASET_CARD.md"
    p.parent.mkdir(parents=True, exist_ok=True)

    L: list[str] = []
    a = L.append
    a("# 데이터셋 카드 — AI Hub 반려동물 피부 질환 (561)\n")
    a("> ⚠️ 이 파일은 `src/scan.py` 가 실물 데이터를 훑어 자동 생성합니다. 손으로 고치지 마세요.")
    a(f"> 생성 시각: `{rep.scanned_at}`  |  스캔 경로: `{rep.root}`\n")

    a("## 규모\n")
    a("| 항목 | 값 |")
    a("|---|---|")
    a(f"| 이미지 | {rep.n_images:,} 장 |")
    a(f"| 라벨 JSON | {rep.n_jsons:,} 개 |")
    a(f"| 용량 | {rep.size_gb} GB |")
    a(f"| 무증상(정상) 데이터 | {'있음' if rep.has_normal else '없음'} |\n")

    a("## 클래스 분포\n")
    if rep.class_counts:
        tot = sum(rep.class_counts.values()) or 1
        a("| 클래스 | 이미지 수 | 비율 |")
        a("|---|---:|---:|")
        for c, n in rep.class_counts.items():
            a(f"| {c} | {n:,} | {n / tot:.1%} |")
        mx, mn = max(rep.class_counts.values()), min(rep.class_counts.values())
        a(f"\n불균형 비(최다/최소): **{mx / max(mn, 1):.1f}배**\n")
    else:
        a("_경로에서 클래스를 찾지 못했습니다. JSON 라벨 필드를 사용해야 합니다._\n")

    a("## 폴더 축\n")
    for axis, hits in rep.axes.items():
        a(f"- **{axis}**: {hits}")
    a("")

    a("## JSON 스키마 추정\n")
    a("| 역할 | 추정 키 |")
    a("|---|---|")
    for role, key in rep.field_guess.items():
        a(f"| {role} | `{key}` |")
    a("\n<details><summary>키 경로 전체 (상위 60)</summary>\n")
    a("| 출현율 | 키 | 타입 | 예시 |")
    a("|---:|---|---|---|")
    for k in rep.json_keys[:60]:
        ex = str(k["examples"][:1]).replace("|", "\\|")
        a(f"| {k['freq']:.0%} | `{k['key']}` | {list(k['types'])} | {ex} |")
    a("\n</details>\n")

    if rep.lesion_area:
        la = rep.lesion_area
        a("## 병변 면적 비율\n")
        a(f"- 중앙값 **{la['median']:.2%}**, p90 {la['p90']:.2%}")
        a(f"- 이미지의 5% 미만인 비율: **{la['under_5pct']:.1%}**")
        a(f"- 1% 미만: {la['under_1pct']:.1%}")
        if la["under_5pct"] > 0.5:
            a("\n> **→ ROI 크롭 필수.** 전체 이미지를 그대로 넣으면 모델이 배경을 학습합니다.\n")

    if rep.dup_estimate:
        d = rep.dup_estimate
        a("## 중복\n")
        a(f"- 샘플 {d['sampled']:,}장 기준 중복률 **{d['duplicate_rate']:.2%}**")
        a(f"- 서로 다른 클래스에 걸친 중복 그룹: **{d['cross_class_groups']}건**\n")

    a("## 해상도 (샘플)\n")
    for r, n in list(rep.resolutions.items())[:10]:
        a(f"- {r}: {n}")
    a("")

    a("## 파일명 패턴 (숫자→`#`, 영문→`L`)\n")
    for pat, n in rep.filename_patterns[:8]:
        a(f"- `{pat}` × {n:,}")
    a("")

    if rep.animal_id_stats.get("filename_token_candidates"):
        a("## 개체ID 후보\n")
        a("데이터 누수를 막으려면 **개체 단위**로 train/val 을 나눠야 합니다.\n")
        a("| 토큰 위치 | 고유값 | 그룹당 평균 장수 | 예시 |")
        a("|---:|---:|---:|---|")
        for c in rep.animal_id_stats["filename_token_candidates"]:
            a(f"| #{c['token_index']} | {c['unique']:,} | {c['avg_per_group']} | {c['examples']} |")
        a(f"\nJSON 필드 후보: `{rep.animal_id_stats.get('json_field_guess')}`\n")

    if rep.warnings:
        a("## ⚠️ 경고\n")
        for w in rep.warnings:
            a(f"- {w}")
        a("")

    a("## 샘플 JSON\n")
    a("```json")
    a(json.dumps(rep.sample_json, indent=2, ensure_ascii=False, default=str)[:3000])
    a("```")

    p.write_text("\n".join(L), encoding="utf-8")
    print(f"[scan] 데이터셋 카드 작성: {p}")
    return p
