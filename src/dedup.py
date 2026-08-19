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


def compute_hashes(paths: list[str], hash_size: int = 16, workers: int = 4) -> dict[str, str]:
    """이미지 경로 → phash 16진 문자열."""
    import imagehash
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None  # 큰 이미지 경고 억제

    def one(p: str) -> tuple[str, str | None]:
        try:
            with Image.open(p) as im:
                return p, str(imagehash.phash(im.convert("RGB"), hash_size=hash_size))
        except Exception:
            return p, None

    out: dict[str, str] = {}
    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for p, h in tqdm(ex.map(one, paths), total=len(paths), desc="phash"):
                if h:
                    out[p] = h
    else:
        for p in tqdm(paths, desc="phash"):
            _, h = one(p)
            if h:
                out[p] = h
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

    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    ints = [_hex_to_bits(h) for _, h in items]
    for i, v in enumerate(ints):
        for b in range(bands):
            key = (b, (v >> (b * band_bits)) & ((1 << band_bits) - 1))
            buckets[key].append(i)

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

    MAX_BUCKET = 400  # 과대 버킷은 O(n²) 폭발을 막기 위해 건너뜀
    for idxs in buckets.values():
        if len(idxs) < 2 or len(idxs) > MAX_BUCKET:
            continue
        for a in range(len(idxs)):
            ia = idxs[a]
            for b in range(a + 1, len(idxs)):
                ib = idxs[b]
                if find(ia) == find(ib):
                    continue
                if bin(ints[ia] ^ ints[ib]).count("1") <= max_hamming:
                    union(ia, ib)

    return {items[i][0]: find(i) for i in range(n)}


def run(
    df: pd.DataFrame,
    cfg: CFG | None = None,
    keep: str = "first",
    drop_cross_class: bool = True,
    workers: int = 4,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """매니페스트에서 중복을 제거하고 `dup_cluster` 컬럼을 붙입니다.

    drop_cross_class=True 면 서로 다른 라벨이 붙은 중복 그룹은
    **전부 버립니다**. 어느 쪽이 맞는지 알 수 없는 오염 데이터이기 때문입니다.
    """
    cfg = cfg or CFG()
    paths = df["image_path"].unique().tolist()
    if verbose:
        print(f"[dedup] {len(paths):,}장 해시 계산 (hash_size={cfg.phash_size})")

    hashes = compute_hashes(paths, hash_size=cfg.phash_size, workers=workers)
    clusters = cluster(hashes, max_hamming=cfg.dedup_hamming)

    out = df.copy()
    out["phash"] = out["image_path"].map(hashes)
    out["dup_cluster"] = out["image_path"].map(clusters)

    # 해시 실패분은 자기 자신을 클러스터로
    miss = out["dup_cluster"].isna()
    if miss.any():
        base = int(out["dup_cluster"].max() or 0) + 1
        out.loc[miss, "dup_cluster"] = range(base, base + int(miss.sum()))
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
