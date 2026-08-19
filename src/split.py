"""개체 단위 데이터 분할 — 이 프로젝트에서 가장 중요한 파일.

왜 중요한가:
  이 데이터셋은 강아지 약 1만 마리에서 50만 장을 찍었습니다. 개체당 평균 수십 장.
  같은 강아지의 같은 부위를 각도만 바꿔 찍은 사진들이 train 과 val 에 나뉘어 들어가면,
  모델은 "병변"이 아니라 "이 강아지"를 외워서 맞힙니다.
  → 검증 정확도 95% 가 나와도 처음 보는 강아지에서는 60% 밖에 안 나옵니다.

  이건 버그가 아니라 **평가 설계의 실패**라 더 위험합니다. 코드는 잘 도는데
  숫자만 거짓말을 하기 때문에, 배포하고 나서야 알게 됩니다.

그래서 분할 그룹 = (개체ID) ∪ (중복 클러스터) 를 union-find 로 합칩니다.
개체ID가 없거나 못 믿을 때도 중복 클러스터가 최소한의 방어선이 됩니다.

    from src import split
    df = split.assign(df)              # fold, is_holdout 컬럼 추가
    tr, va = split.get_fold(df, 0)
"""

from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np
import pandas as pd

from src.config import CFG


# ──────────────────────────────────────────────────────────────
# 그룹 만들기
# ──────────────────────────────────────────────────────────────
def build_groups(df: pd.DataFrame, verbose: bool = True) -> pd.Series:
    """animal_id 와 dup_cluster 를 union-find 로 합쳐 최종 그룹 키를 만듭니다."""
    keys: list[tuple] = []
    use_animal = "animal_id" in df.columns and df["animal_id"].notna().any()
    use_dup = "dup_cluster" in df.columns and df["dup_cluster"].notna().any()

    if not use_animal and not use_dup:
        if verbose:
            print("⚠️ animal_id 도 dup_cluster 도 없습니다. 이미지 단위 분할이 되어 "
                  "데이터 누수를 막을 수 없습니다. labels.py/dedup.py 를 먼저 돌리세요.")
        return pd.Series([f"img_{i}" for i in range(len(df))], index=df.index)

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, row in df.iterrows():
        node = f"row::{i}"
        find(node)
        if use_animal and pd.notna(row.get("animal_id")):
            union(node, f"animal::{row['animal_id']}")
        if use_dup and pd.notna(row.get("dup_cluster")):
            union(node, f"dup::{int(row['dup_cluster'])}")

    groups = pd.Series([find(f"row::{i}") for i in df.index], index=df.index)
    # 보기 좋은 짧은 이름으로
    mapping = {g: f"G{n:06d}" for n, g in enumerate(sorted(groups.unique()))}
    groups = groups.map(mapping)

    if verbose:
        n_g = groups.nunique()
        print(f"[split] 그룹 {n_g:,}개 (행 {len(df):,}개, 그룹당 평균 {len(df) / n_g:.1f}장)")
        if use_animal and use_dup:
            print("        구성: 개체ID ∪ 중복클러스터")
        elif use_animal:
            print("        구성: 개체ID만 (중복 클러스터 없음 — dedup.py 를 먼저 돌리는 게 안전)")
        else:
            print("        구성: 중복클러스터만 (개체ID 없음 — 차선책, 누수 위험 잔존)")
        if len(df) / n_g < 1.2:
            print("        ⚠️ 그룹당 평균이 1에 가깝습니다 = 사실상 이미지 단위 분할입니다.")
    return groups


def _hash_bucket(key: str, n: int, salt: str = "") -> int:
    """문자열 키를 안정적으로 0..n-1 버킷에 넣습니다 (실행마다 동일)."""
    h = hashlib.md5(f"{salt}{key}".encode()).hexdigest()
    return int(h[:8], 16) % n


# ──────────────────────────────────────────────────────────────
# 분할
# ──────────────────────────────────────────────────────────────
def assign(
    df: pd.DataFrame,
    cfg: CFG | None = None,
    label_col: str = "label",
    verbose: bool = True,
) -> pd.DataFrame:
    """`group`, `is_holdout`, `fold` 컬럼을 붙여 돌려줍니다.

    - is_holdout : 최종 1회만 보는 테스트셋. 개체 단위로 떼어냅니다.
    - fold       : 나머지에 대한 StratifiedGroupKFold 인덱스 (0..n_folds-1)
    """
    cfg = cfg or CFG()
    out = df.copy()
    out["group"] = build_groups(out, verbose=verbose)

    # --- 1) holdout 분리 (그룹 해시 기반 → 재실행해도 동일) ---
    n_buckets = 1000
    cut = int(cfg.holdout_ratio * n_buckets)
    out["is_holdout"] = out["group"].map(
        lambda g: _hash_bucket(g, n_buckets, salt=f"holdout{cfg.seed}") < cut
    )

    dev = out[~out["is_holdout"]]
    if verbose:
        print(f"[split] holdout {out['is_holdout'].sum():,}장 "
              f"({out['is_holdout'].mean():.1%}) / 개발용 {len(dev):,}장")

    # --- 2) 개발용에 StratifiedGroupKFold ---
    out["fold"] = -1
    if len(dev) == 0:
        return out

    y = dev[label_col].fillna("NA").astype(str).values
    g = dev["group"].values

    try:
        from sklearn.model_selection import StratifiedGroupKFold

        skf = StratifiedGroupKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
        for k, (_, vi) in enumerate(skf.split(np.zeros(len(dev)), y, groups=g)):
            out.loc[dev.index[vi], "fold"] = k
    except Exception as exc:
        if verbose:
            print(f"[split] StratifiedGroupKFold 실패({exc}) → 그룹 해시 분할로 대체")
        out.loc[dev.index, "fold"] = [
            _hash_bucket(x, cfg.n_folds, salt=f"fold{cfg.seed}") for x in g
        ]

    if verbose:
        _report(out, label_col, cfg)
    return out


def _report(df: pd.DataFrame, label_col: str, cfg: CFG) -> None:
    print("\n[fold 별 분포]")
    dev = df[~df["is_holdout"]]
    tbl = pd.crosstab(dev["fold"], dev[label_col].fillna("NA"))
    print(tbl.to_string())
    print("\n[fold 별 클래스 비율 — 편차가 크면 층화가 실패한 것]")
    print((tbl.div(tbl.sum(axis=1), axis=0) * 100).round(1).to_string())

    # 그룹 누수 자체 점검
    bad = 0
    for k in range(cfg.n_folds):
        gv = set(dev[dev["fold"] == k]["group"])
        gt = set(dev[dev["fold"] != k]["group"])
        if gv & gt:
            bad += len(gv & gt)
    print(f"\n[누수 점검] fold 간 그룹 중복: {bad}건 " + ("✅" if bad == 0 else "❌"))
    ho = set(df[df["is_holdout"]]["group"]) & set(dev["group"])
    print(f"[누수 점검] holdout ↔ 개발용 그룹 중복: {len(ho)}건 " + ("✅" if not ho else "❌"))


def get_fold(df: pd.DataFrame, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, val) 을 돌려줍니다. holdout 은 양쪽 모두에서 제외됩니다."""
    dev = df[~df["is_holdout"]]
    tr = dev[dev["fold"] != fold].reset_index(drop=True)
    va = dev[dev["fold"] == fold].reset_index(drop=True)
    return tr, va


def get_holdout(df: pd.DataFrame) -> pd.DataFrame:
    """최종 보고용 테스트셋.

    ⚠️ 이걸 보고 하이퍼파라미터를 고치면 더 이상 holdout 이 아닙니다.
       모든 결정이 끝난 뒤 딱 한 번만 여세요.
    """
    return df[df["is_holdout"]].reset_index(drop=True)


def verify(df: pd.DataFrame, fold: int = 0, strict: bool = True) -> bool:
    """분할이 정말 새는 곳이 없는지 확인합니다."""
    tr, va = get_fold(df, fold)
    ho = get_holdout(df)
    ok = True

    checks = [
        ("train ↔ val   그룹", set(tr["group"]), set(va["group"])),
        ("train ↔ hold  그룹", set(tr["group"]), set(ho["group"])),
        ("val   ↔ hold  그룹", set(va["group"]), set(ho["group"])),
    ]
    if "animal_id" in df.columns:
        checks.append(("train ↔ val   개체", set(tr["animal_id"]), set(va["animal_id"])))
    if "phash" in df.columns:
        checks.append(("train ↔ val   해시", set(tr["phash"].dropna()), set(va["phash"].dropna())))

    print("\n" + "─" * 52)
    for name, a, b in checks:
        n = len(a & b)
        mark = "✅" if n == 0 else "❌"
        print(f"  {mark} {name} 중복: {n}")
        ok &= n == 0
    print(f"\n  train {len(tr):,} / val {len(va):,} / holdout {len(ho):,}")
    print("─" * 52 + "\n")

    if strict and not ok:
        raise AssertionError(
            "데이터 누수가 감지되었습니다. 이 상태로 학습하면 정확도 수치를 신뢰할 수 없습니다.\n"
            "docs/cautions/02_데이터_누수_가장_치명적인_함정.md 를 참고하세요."
        )
    return ok


def compare_with_random(df: pd.DataFrame, cfg: CFG | None = None) -> None:
    """참고용: 이미지 단위(잘못된) 분할이면 얼마나 새는지 보여줍니다.

    "왜 굳이 개체 단위로 나눠야 하나"를 눈으로 확인하는 용도입니다.
    """
    cfg = cfg or CFG()
    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(len(df))
    cutoff = int(len(df) * 0.8)
    tr, va = df.iloc[idx[:cutoff]], df.iloc[idx[cutoff:]]

    shared_g = len(set(tr["group"]) & set(va["group"]))
    leaked = va["group"].isin(set(tr["group"])).mean()
    print("\n[참고] 만약 이미지 단위로 무작위 분할했다면:")
    print(f"  train/val 이 공유하는 그룹: {shared_g:,}개")
    print(f"  val 이미지 중 train 에 같은 개체가 있는 비율: {leaked:.1%}")
    print("  → 이 비율만큼 검증 점수가 부풀려집니다.\n")
