"""중복 이미지 제거 — perceptual hash 기반.

왜 필요한가:
  이 데이터를 먼저 써 본 공개 프로젝트들이 공통으로 보고한 문제가
  "같은 이미지가 여러 클래스 폴더에 중복 존재"였습니다.
  이걸 두고 학습하면 (1) 라벨이 모순되고 (2) train/val 에 같은 사진이 걸쳐
  정확도가 가짜로 올라갑니다.

phash 는 리사이즈·약한 압축·미세한 밝기 변화를 견디므로
파일 해시(md5)로는 못 잡는 "거의 같은 사진"까지 잡아냅니다.

    from src import dedup
    df2, info = dedup.run(df)
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from src import env
from src.config import CFG


def compute_hashes(rows, hash_size: int = 16, workers: int = 4,
                   path_col: str = "image_path") -> dict[str, str]:
    """이미지 → phash 16진 문자열. 키는 `path_col` 값입니다.

    rows 는 DataFrame 이거나 경로 문자열 리스트일 수 있습니다.
    DataFrame 이면 `zip_member` 를 보고 **zip 안에서 직접** 읽습니다
    (압축을 풀지 않는 모드에서도 중복 제거가 동작해야 하므로).
    """
    import imagehash
    from PIL import Image

    from src.crop import _open_source

    Image.MAX_IMAGE_PIXELS = None

    if isinstance(rows, pd.DataFrame):
        items = rows.drop_duplicates(subset=[path_col]).to_dict("records")
    else:
        items = [{path_col: p} for p in rows]

    def one(rec: dict) -> tuple[str, str | None]:
        key = rec[path_col]
        try:
            with _open_source({**rec, "image_path": key}) as im:
                return key, str(imagehash.phash(im.convert("RGB"), hash_size=hash_size))
        except Exception:
            return key, None

    out: dict[str, str] = {}
    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for p, h in tqdm(ex.map(one, items), total=len(items), desc="phash"):
                if h:
                    out[p] = h
    else:
        for rec in tqdm(items, desc="phash"):
            p, h = one(rec)
            if h:
                out[p] = h

    if not out:
        raise RuntimeError(
            f"이미지를 한 장도 읽지 못했습니다 (기준 컬럼: {path_col}).\n"
            "  · zip 모드라면 zip_path/zip_member 컬럼이 살아있는지 확인하세요.\n"
            "  · 크롭본으로 중복을 잡으려면 path_col='crop_path' 를 쓰세요."
        )
    return out


def _hex_to_bits(h: str) -> int:
    return int(h, 16)


def cluster(hashes: dict[str, str], max_hamming: int = 6) -> dict[str, int]:
    """near-duplicate 클러스터 ID를 부여합니다.

    전체 쌍 비교는 O(n²) 라 불가능하므로, 해시를 4등분한 밴드로
    후보를 좁힌 뒤(LSH 유사) 그 안에서만 해밍 거리를 잽니다.
    """
    items = list(hashes.items())
    n = len(items)
    if n == 0:
        return {}

    nbits = len(items[0][1]) * 4
    bands = 4
    band_bits = max(nbits // bands, 1)

    ints = [_hex_to_bits(h) for _, h in items]

    # union-find
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # ── 1단계: 완전히 같은 해시는 무조건 먼저 묶습니다 ──────────────
    # ⚠️ 이걸 빼먹으면 안 됩니다. 아래 LSH 는 과대 버킷을 건너뛰는데,
    #    똑같은 이미지가 수백 장이면 그 버킷이 정확히 과대 버킷이 되어
    #    **완전 동일한 이미지들이 서로 다른 클러스터로 갈라집니다.**
    #    (실측: 동일 이미지 600장 → 클러스터 600개, 중복 제거 0건)
    #    dict 그룹핑이라 O(n) 이고 크기 제한이 필요 없습니다.
    exact: dict[int, int] = {}
    for i, v in enumerate(ints):
        if v in exact:
            union(exact[v], i)
        else:
            exact[v] = i

    # ── 2단계: 비슷하지만 같지는 않은 것들 (LSH 밴딩) ───────────────
    #    대표 하나씩만 비교하면 되므로 후보 수가 크게 줄어듭니다.
    reps = sorted(exact.values())
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i in reps:
        v = ints[i]
        for b in range(bands):
            key = (b, (v >> (b * band_bits)) & ((1 << band_bits) - 1))
            buckets[key].append(i)

    MAX_BUCKET = 2000  # O(n²) 폭발 방지. 대표만 담기므로 이 정도면 충분합니다.
    skipped = 0
    for idxs in buckets.values():
        if len(idxs) < 2:
            continue
        if len(idxs) > MAX_BUCKET:
            skipped += 1
            continue
        for a in range(len(idxs)):
            ia = idxs[a]
            for b in range(a + 1, len(idxs)):
                ib = idxs[b]
                if find(ia) == find(ib):
                    continue
                if bin(ints[ia] ^ ints[ib]).count("1") <= max_hamming:
                    union(ia, ib)
    if skipped:
        print(f"[dedup] ⚠️ 후보가 너무 많은 버킷 {skipped}개는 근사 비교를 건너뛰었습니다 "
              "(완전 동일 이미지는 1단계에서 이미 묶였습니다)")

    return {items[i][0]: find(i) for i in range(n)}


def run(
    df: pd.DataFrame,
    cfg: CFG | None = None,
    keep: str = "first",
    drop_cross_class: bool = True,
    workers: int = 4,
    path_col: str | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """매니페스트에서 중복을 제거하고 `dup_cluster` 컬럼을 붙입니다.

    drop_cross_class=True 면 서로 다른 라벨이 붙은 중복 그룹은
    **전부 버립니다**. 어느 쪽이 맞는지 알 수 없는 오염 데이터이기 때문입니다.
    """
    cfg = cfg or CFG()
    # 기본은 **원본 이미지** 기준입니다.
    #   잡으려는 문제가 "같은 사진이 여러 클래스 폴더에 존재"이므로 원본을 봐야 맞습니다.
    #   크롭본으로 해시하면 서로 다른 사진인데 병변 부위만 비슷한 경우까지 묶여
    #   멀쩡한 데이터가 버려집니다 (합성 데이터 실험에서 291→197 로 과하게 줄었습니다).
    #   zip 모드에서도 _open_source 가 zip 안에서 원본을 읽으므로 문제없습니다.
    if path_col is None:
        path_col = "image_path"
    n_uniq = df[path_col].nunique()
    if verbose:
        print(f"[dedup] {n_uniq:,}장 해시 계산 (기준 {path_col}, hash_size={cfg.phash_size})")

    hashes = compute_hashes(df, hash_size=cfg.phash_size, workers=workers, path_col=path_col)
    clusters = cluster(hashes, max_hamming=cfg.dedup_hamming)

    out = df.copy()
    out["phash"] = out[path_col].map(hashes)
    out["dup_cluster"] = out[path_col].map(clusters)

    # 해시 실패분은 자기 자신을 클러스터로 (전부 실패해도 죽지 않도록 방어)
    miss = out["dup_cluster"].isna()
    if miss.any():
        mx = out["dup_cluster"].max()
        base = 0 if pd.isna(mx) else int(mx) + 1
        out.loc[miss, "dup_cluster"] = range(base, base + int(miss.sum()))
        if verbose:
            print(f"[dedup] ⚠️ {int(miss.sum()):,}장은 해시 실패 — 중복 판정에서 제외됩니다")
    out["dup_cluster"] = out["dup_cluster"].astype(int)

    info: dict = {"n_before": len(out), "n_hashed": len(hashes)}

    # 1) 클래스가 충돌하는 클러스터 처리
    labeled = out[out["label"].notna()]
    conflict = (
        labeled.groupby("dup_cluster")["label"].nunique().pipe(lambda s: s[s > 1]).index.tolist()
    )
    info["cross_class_clusters"] = len(conflict)
    info["cross_class_images"] = int(out["dup_cluster"].isin(conflict).sum())
    if conflict and drop_cross_class:
        if verbose:
            print(f"[dedup] ⚠️ 라벨이 충돌하는 중복 그룹 {len(conflict)}개 "
                  f"({info['cross_class_images']:,}장) 제거")
            ex = out[out["dup_cluster"].isin(conflict)].head(4)
            for _, r in ex.iterrows():
                print(f"      {r['label']}  {Path(r['image_path']).name}")
        out = out[~out["dup_cluster"].isin(conflict)]

    # 2) 같은 클러스터 + 같은 라벨 → 대표 1장만
    before = len(out)
    if keep in ("first", "last"):
        # 병변이 큰 것을 남기는 편이 학습에 유리
        out = out.sort_values("area_ratio", ascending=False, na_position="last")
        out = out.drop_duplicates(subset=["dup_cluster", "label"], keep="first")
    info["removed_near_dup"] = before - len(out)
    info["n_after"] = len(out)
    info["duplicate_rate"] = round(1 - len(out) / max(info["n_before"], 1), 4)

    out = out.sort_index().reset_index(drop=True)

    # 중복 제거가 데이터를 과도하게 날리면 조용히 넘어가면 안 됩니다.
    # (라벨 충돌 클러스터 제거가 폭주하면 학습할 데이터가 사라집니다)
    if info["duplicate_rate"] > 0.5:
        print(f"\n🚨 중복 제거로 {info['duplicate_rate']:.0%} 가 사라졌습니다 "
              f"({info['n_before']:,} → {info['n_after']:,}).")
        print("   정상적인 상황이 아닙니다. 확인할 것:")
        print(f"   · 라벨 충돌로 제거된 그룹: {info['cross_class_clusters']:,}개 "
              f"({info['cross_class_images']:,}장)")
        print(f"   · dedup_hamming={cfg.dedup_hamming} 이 너무 커서 "
              "서로 다른 이미지까지 묶고 있지 않은지")
        print("   · 이미지가 실제로 서로 다른지 (crop.preview 로 눈으로 확인)")
        if info["n_after"] == 0:
            raise RuntimeError(
                "중복 제거 후 데이터가 하나도 남지 않았습니다.\n"
                "  drop_cross_class=False 로 두고 원인을 먼저 확인하세요."
            )

    if verbose:
        print(f"[dedup] {info['n_before']:,} → {info['n_after']:,} "
              f"(중복 {info['duplicate_rate']:.1%} 제거)")
        print("\n[제거 후 클래스 분포]")
        for k, v in out["label"].value_counts(dropna=False).items():
            print(f"  {str(k):>6}: {v:>8,}")
    return out, info


def sanity_check_split(train: pd.DataFrame, val: pd.DataFrame) -> None:
    """분할 후 두 세트에 같은 이미지가 걸쳐 있지 않은지 확인합니다.

    이 assert 가 터지면 정확도 수치는 전부 믿을 수 없습니다.
    """
    if "dup_cluster" in train.columns and "dup_cluster" in val.columns:
        overlap = set(train["dup_cluster"]) & set(val["dup_cluster"])
        assert not overlap, (
            f"❌ 데이터 누수: train/val 에 동일 이미지 클러스터가 {len(overlap)}개 겹칩니다.\n"
            "   split.py 가 dup_cluster 를 그룹에 포함하도록 되어 있는지 확인하세요."
        )
    if "animal_id" in train.columns and "animal_id" in val.columns:
        overlap = set(train["animal_id"]) & set(val["animal_id"])
        assert not overlap, (
            f"❌ 데이터 누수: train/val 에 동일 개체가 {len(overlap)}마리 겹칩니다."
        )
    print("✅ 누수 검사 통과 — train/val 에 겹치는 개체·이미지 없음")
