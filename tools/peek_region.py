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

# ── 부위 코드가 무엇을 뜻하는지 해독 ─────────────────────────────
# ⚠️ A/B/H/L 을 추측으로 풀면 안 됩니다. 이 프로젝트에서 스키마를 추측했다가
#    세 번 틀렸고, 셋 다 에러 없이 그럴듯한 값을 만들어냈습니다.
#    원본에 남아 있는 문자열(경로·파일명)에서 읽습니다.
print("\n── 매니페스트 전체 컬럼 ──")
print("  " + ", ".join(df.columns))

print("\n── 부위 코드 해독 단서 ──")
# ⚠️ 경로는 **뒤쪽**이 중요합니다. 앞은 전부 같은 디렉터리라 정보가 없고,
#    zip 안쪽 폴더 구조(VL01.zip!반려견/피부/일반카메라/…)에 부위가 들어 있습니다.
def _tail(v: str, n: int = 130) -> str:
    for sep in ("!", ".zip"):
        if sep in v:
            v = v.split(sep, 1)[1]
            break
    return v if len(v) <= n else "…" + v[-n:]

cand = [c for c in ("image_path", "symptom_meta", "image_name", "crop_rel")
        if c in sub.columns]
if not cand:
    print("  단서가 될 컬럼이 없습니다")
for code in tab.index:
    rows = sub[sub["부위"] == code]
    print(f"\n  [{code}]  {len(rows):,}장")
    for c in cand:
        vals = rows[c].dropna().astype(str)
        vals = vals[vals.str.strip() != ""]
        if len(vals) == 0:
            continue
        print(f"    {c:<14} 고유 {vals.nunique():,}개")
        for v in vals.drop_duplicates().head(3):
            print(f"      {_tail(v)}")

# 경로에서 폴더 조각만 뽑아 부위와 교차표를 만듭니다 — 이게 제일 확실합니다
if "image_path" in sub.columns:
    import re

    paths = sub["image_path"].dropna().astype(str)
    if len(paths):
        segs = paths.apply(lambda v: [t for t in re.split(r"[\\/!]+", _tail(v, 400))
                                      if t and not t.lower().endswith((".jpg", ".json"))])
        depth = max((len(x) for x in segs), default=0)
        print(f"\n── 경로 폴더 조각별 × 부위 (깊이 {depth}) ──")
        for d in range(depth):
            col = segs.apply(lambda x, d=d: x[d] if len(x) > d else None)
            uniq = col.dropna().nunique()
            if uniq <= 1 or uniq > 12:      # 전부 같거나 너무 잘게 쪼개진 층은 건너뜀
                continue
            ct = sub.assign(_s=col).pivot_table(index="_s", columns="부위",
                                                values="label", aggfunc="size",
                                                fill_value=0)
            print(f"\n  [깊이 {d}]  고유 {uniq}개")
            print("  " + ct.to_string().replace("\n", "\n  "))

print("\n💡 위에서 부위(A/B/H/L)와 **딱 맞아떨어지는 폴더 층**이 있으면 그게 답입니다.")
print("   없으면 코드의 뜻은 사진으로 알아내거나 AI Hub 설명서를 봐야 합니다.")

# ── 부위별 사진을 깔아 눈으로 확인 ──────────────────────────────
# 경로·메타에 부위명이 없으면 남은 방법은 **보는 것**입니다. 크롭이 로컬에 있으니
# 부위별로 정상 사진 12장씩 한 판에 깔면 A/B/H/L 이 무엇인지 바로 보입니다.
if "--sheets" in sys.argv:
    import matplotlib
    matplotlib.use("Agg")

    from src import crop, errors

    tags = crop.available_tags()
    if not tags:
        raise SystemExit("❌ 크롭이 없습니다. prepare_local.py 를 먼저 돌리세요.")
    tag = "f320" if "f320" in tags else tags[0]      # 1단계가 보는 크롭을 우선
    print(f"\n부위별 사진을 저장합니다 (크롭 '{tag}', 붙어 있는 것: {tags})")

    out = env.work_root() / "reports"
    # ⚠️ allow_missing=True — 그림 몇 장 보자는 것이라 95% 커버리지 기준으로
    #    멈출 이유가 없습니다. 실제로 있는 파일만 골라 씁니다.
    switched = crop.switch_tag(sub, tag, verbose=False, allow_missing=True)
    have = switched["crop_path"].apply(lambda v: bool(v) and Path(v).exists())
    print(f"  실제로 있는 크롭 {int(have.sum()):,} / {len(switched):,}장")
    switched = switched[have]
    if len(switched) == 0:
        raise SystemExit(f"❌ '{tag}' 크롭 파일을 하나도 못 찾았습니다.")
    for code in tab.index:
        rows = switched[(switched["region"].astype(str).str.strip() == str(code))
                        & (switched["label"] == NORMAL_LABEL)]
        if len(rows) == 0:
            print(f"  [{code}] 정상 사진이 없습니다 — 건너뜁니다")
            continue
        errors.contact_sheet(rows, n=12, cols=4, seed=0, show=False,
                             title=f"부위 '{code}' — 정상 사진 (크롭 {tag})",
                             save_to=out / f"region_{code}.png")
    print(f"\n→ {out} 안의 region_*.png 를 열어 무슨 부위인지 확인하세요.")
    print("  (머리 / 등 / 배 / 다리 / 발 … 무엇이 보이는지)")
else:
    print("\n💡 부위별 사진을 보려면:  uv run python tools/peek_region.py --sheets")
print("\n💡 정상 사진이 몰려 있는 부위가 곧 헛알림 후보입니다.")
print("   실제 헛알림률은 노트북 07 의 4-b 절(errors.by_group)이 잽니다.")
