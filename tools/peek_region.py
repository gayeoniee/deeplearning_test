#!/usr/bin/env python3
"""매니페스트의 `region`(부위) 값을 확인합니다. GPU·Kaggle 없이 로컬에서 돕니다.

    uv run python tools/peek_region.py

왜 필요한가 — STEP 11 에서 헛알림 사진 30장을 눈으로 보니 발바닥 패드와 코가
많아 보였습니다. 부위별로 재려면 먼저 **매니페스트에 부위 값이 실제로 들어
있는지** 확인해야 합니다 (`metaData.region` 이 비어 있을 수도 있습니다).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import env, labels                                       # noqa: E402
from src.config import NORMAL_LABEL                               # noqa: E402

mf = env.work_root() / "manifests" / "manifest_final.parquet"
if not mf.exists():
    raise SystemExit(f"❌ {mf} 가 없습니다. prepare_local.py --finalize 를 먼저 돌리세요.")

df = labels.load(mf)
print(f"{len(df):,}행 / 컬럼 {len(df.columns)}개\n")

if "region" not in df.columns:
    raise SystemExit("❌ 'region' 컬럼이 없습니다 — 원본 JSON 에 부위 정보가 없습니다.")

r = df["region"].astype(str).str.strip()
filled = r[(r != "") & (r.str.lower() != "none") & (r != "nan")]
print(f"부위(region) 채워진 행: {len(filled):,} / {len(df):,}  ({len(filled)/len(df):.1%})")
if len(filled) == 0:
    raise SystemExit("\n❌ 전부 비어 있습니다 — 부위별 분석은 못 합니다.")

print(f"고유값 {filled.nunique()}개\n")
print("── 부위별 정상/병변 ──")
sub = df.loc[filled.index].copy()
sub["부위"] = filled
tab = sub.assign(정상=(sub["label"] == NORMAL_LABEL)).groupby("부위", observed=True).agg(
    n=("label", "size"), 정상=("정상", "sum"))
tab["정상비율"] = tab["정상"] / tab["n"]
tab = tab.sort_values("n", ascending=False)
print(f"  {'부위':<22}{'전체':>9}{'정상':>9}{'정상비율':>10}")
for k, row in tab.head(20).iterrows():
    print(f"  {str(k):<22}{int(row['n']):>9,}{int(row['정상']):>9,}{row['정상비율']:>10.1%}")
if len(tab) > 20:
    print(f"  … 그 밖에 {len(tab) - 20}개")

print("\n💡 정상 사진이 몰려 있는 부위가 곧 헛알림 후보입니다.")
print("   실제 헛알림률은 노트북 07 의 4-b 절(errors.by_group)이 잽니다.")
