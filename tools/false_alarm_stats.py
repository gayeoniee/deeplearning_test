"""헛알림·놓침이 **사진의 무엇** 과 상관 있나 — 눈이 아니라 숫자로.

    uv run --extra train python tools/false_alarm_stats.py --release <릴리스폴더>

왜 이 도구가 있나
-----------------
STEP 11 에서 오류가 A2(비듬·각질)에 몰린다는 건 알아냈는데 **원인은 못 쟀습니다.**
다음 수는 "헛알림 난 정상 사진 30장을 눈으로 보고 공통점 찾기" 였는데,
수의사가 아니면 못 하는 일이라 접었습니다.

접는 게 맞았지만 질문 자체는 남습니다. 그래서 **눈 대신 자를 댑니다.**
털·흐림·조명은 병변 지식 없이도 잴 수 있는 값이고, 그게 헛알림과 상관 있으면
"촬영 가이드" 로 잡을 수 있는 문제, 없으면 모델·라벨 문제입니다.

무엇을 재나 (전부 병변 지식이 필요 없는 값)
-------------------------------------------
| 이름 | 뜻 | 크면 |
|---|---|---|
| `blur` | 라플라시안 분산 | **작을수록** 흐림 |
| `bright` | 밝기 평균 (0~1) | 밝음 |
| `contrast` | 밝기 표준편차 | 대비 큼 |
| `sat` | 채도 평균 | 색이 진함 |
| `hair` | 1픽셀 차분의 크기 (fine texture) | **털처럼 가는 선**이 많음 |
| `warm` | R−B 평균 | 노란 조명 |

⚠️ `hair` 는 **털 검출기가 아닙니다.** 가는 선이 많으면 커지는 값이라
   털·각질·주름을 **구분하지 못합니다.** 그래서 이 값 하나로는
   "털 때문" 이라고 못 박을 수 없습니다 — 그건 견종·부위로 따로 잽니다
   (`--from-saved` 가 같이 보여줍니다).

읽는 법
-------
같은 정상 사진인데 **틀린 쪽과 맞은 쪽의 값이 다른가** 를 봅니다.

* `Δ` = 두 평균의 차이 (그 값의 원래 단위)
* `d`  = 퍼짐으로 나눈 차이(Cohen's d). |d| < 0.2 는 사실상 차이 없음
* `AUROC` = 그 값 **하나만** 보고 오류 난 사진을 골라낼 수 있나.
  0.5 = 못 함. **0.5 미만이면 방향이 반대**라는 뜻이고 세기는 |AUROC − 0.5| 입니다

⚠️ **`d` 하나만 보면 속습니다.** 퍼짐이 아주 작으면 실제 차이가 없다시피 해도
   `d` 가 커집니다 (검증 때 `bright` 가 0.5001 vs 0.4981 인데 d=0.69 였습니다).
   그래서 판정은 **d 와 AUROC 를 둘 다** 넘어야 합니다:

       |d| ≥ 0.5  **그리고**  |AUROC − 0.5| ≥ 0.15   →  차이 있음

둘 다 작으면 "그 사진들엔 공통점이 없다" 가 결론이고, 그것도 결과입니다 —
촬영 가이드로 못 잡는다는 뜻이니까요.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ⚠️ 값의 **정의는 `src/texture.py` 한 곳에만** 둡니다. 여기 따로 적으면
#    학습(`data.hair_sampler`)과 갈라지고, 갈라져도 아무도 모릅니다 —
#    이 리포가 크롭 창에서 이미 당한 실패입니다.
from src.texture import STATS, image_stats                      # noqa: E402,F401

# 매니페스트에서 가져오는 값 — 사진을 다시 안 열어도 됩니다.
#   area     = bbox 면적 ÷ 원본 면적 (`area_ratio`). **얼마나 가까이서 찍었나**
#   megapix  = 원본 화소수. 기기·촬영 설정의 대용
EXTRA = ["area", "megapix"]

def cohens_d(x, y) -> float:
    import numpy as np

    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    s = np.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1))
                / (len(x) + len(y) - 2))
    return float((x.mean() - y.mean()) / s) if s else float("nan")


def auroc(pos, neg) -> float:
    """값 하나만으로 pos 를 neg 와 가를 수 있나 (Mann-Whitney U)."""
    import numpy as np

    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = allv.argsort().argsort().astype(float) + 1
    # 동점 처리 — 각질처럼 값이 뭉치는 경우가 있어서 필요합니다
    order = np.sort(allv)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and order[j + 1] == order[i]:
            j += 1
        if j > i:
            r[(allv >= order[i]) & (allv <= order[j])] = (i + j) / 2 + 1
        i = j + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


# 판정 문턱 — **결과를 보기 전에** 못 박습니다 (CLAUDE.md 규칙 2).
# d 하나로는 안 됩니다: 퍼짐이 작으면 실제 차이가 없어도 d 가 커집니다.
D_STRONG, A_STRONG = 0.5, 0.15         # 둘 **다** 넘어야 "차이 있음"
D_WEAK, A_WEAK = 0.2, 0.06


def verdict(d: float, a: float) -> str:
    import math

    if math.isnan(d) or math.isnan(a):
        return "못 잼"
    da = abs(a - 0.5)
    if abs(d) >= D_STRONG and da >= A_STRONG:
        return "**차이 있음**"
    if abs(d) >= D_WEAK and da >= A_WEAK:
        return "조금"
    return "차이 없음"


def compare(df, mask_bad, mask_good, title: str, n_bad_label: str,
            n_good_label: str, stats: list | None = None) -> None:
    print(f"\n  ── {title} ──")
    nb, ng = int(mask_bad.sum()), int(mask_good.sum())
    print(f"     {n_bad_label} {nb:,}장  vs  {n_good_label} {ng:,}장")
    if nb < 30 or ng < 30:
        print("     ⚠️ 표본이 30장 미만이라 아래 숫자는 못 믿습니다.")
    print(f"\n     {'값':<10}{'틀린 쪽':>10}{'맞은 쪽':>10}{'Δ':>10}"
          f"{'d':>7}{'AUROC':>8}   판정")
    print("     " + "─" * 70)
    for k in (stats or STATS):
        b, g = df.loc[mask_bad, k], df.loc[mask_good, k]
        d, ar = cohens_d(b, g), auroc(b, g)
        print(f"     {k:<10}{b.mean():>10.4f}{g.mean():>10.4f}"
              f"{b.mean() - g.mean():>10.4f}{d:>7.2f}{ar:>8.3f}   {verdict(d, ar)}")


def residualize(target, control, deg: int = 3):
    """`control` 의 영향을 뺀 `target` 만 남깁니다.

    ⚠️ 처음엔 `control` 을 사분위로 **묶어서** 재려 했는데 **틀렸습니다.**
       칸 안에서도 control 이 계속 변하고 target 이 그걸 주워담아서, 독립
       성분이 **0인 합성 데이터에서도** "독립" 이 나왔습니다 (AUROC 0.657).
       칸을 잘게 쪼개도 새는 건 마찬가지라 방법 자체를 바꿨습니다.

    지금 방식 — **순위로 바꾼 뒤 control 의 3차 추세를 빼냅니다.**
    순위를 쓰는 이유: `blur`(라플라시안 분산)와 `hair`(1픽셀 차분)는 대략
    제곱근 관계라 원값에 직선을 맞추면 휘어진 부분이 남습니다. 순위는
    로그·제곱근 같은 **단조 변환에 안 흔들립니다.**

    합성 데이터 검증 (tests/test_false_alarm_stats.py):
      · hair = blur 의 다른 이름   → 0.939 → **0.511** (같은 것)
      · hair 에 독립 성분, 그게 오답 → 0.712 → **0.917** (독립)
      · 독립 성분 있으나 오답은 blur → 0.886 → **0.515** (같은 것)
    """
    import numpy as np
    import pandas as pd

    t = pd.Series(np.asarray(target, float)).rank().values
    c = pd.Series(np.asarray(control, float)).rank().values
    c = (c - c.mean()) / (c.std() or 1.0)
    X = np.vander(c, deg + 1)
    coef, *_ = np.linalg.lstsq(X, t, rcond=None)
    return t - X @ coef


def by_category(df, mask_bad, mask_good, col: str, top: int = 12) -> None:
    """견종·부위별로 헛알림률을 봅니다 — **"털 때문인가" 를 직접 묻는 방법**입니다.

    `hair` 는 털·각질·주름을 구분 못 합니다. 하지만 견종은 털의 양을
    대신 말해줍니다. 털 많은 견종에 헛알림이 몰리면 "털 때문" 이 맞고,
    견종과 무관하게 퍼져 있으면 털이 아니라 다른 것입니다.
    """
    import numpy as np
    import pandas as pd

    sel = mask_bad | mask_good
    d = df[sel]
    if col not in d.columns or d[col].isna().all():
        print(f"\n  ── {col} — 매니페스트에 없어서 건너뜁니다 ──")
        return

    g = pd.DataFrame({
        "n": d.groupby(col, dropna=True).size(),
        "헛알림": d[mask_bad.reindex(d.index).fillna(False)]
                  .groupby(col, dropna=True).size(),
        "hair": d.groupby(col, dropna=True)["hair"].mean(),
    })
    g["헛알림"] = g["헛알림"].fillna(0).astype(int)
    g["헛알림률"] = g["헛알림"] / g["n"]
    g = g[g["n"] >= 30].sort_values("헛알림률", ascending=False)
    if g.empty:
        print(f"\n  ── {col} — 30장 넘는 항목이 없습니다 ──")
        return

    base = int(mask_bad.sum()) / max(int(sel.sum()), 1)
    print(f"\n  ── {col} 별 헛알림률 (정상 사진 30장 이상) ──")
    print(f"     전체 평균 {base:.1%}\n")
    print(f"     {col:<18}{'n':>7}{'헛알림':>8}{'비율':>9}{'hair 평균':>11}")
    print("     " + "─" * 54)
    show = pd.concat([g.head(top // 2), g.tail(top // 2)]).drop_duplicates() \
        if len(g) > top else g
    prev = None
    ko = REGION_KO if col == "region" else {}
    for k, r in show.iterrows():
        if prev is not None and g.index.get_loc(k) - g.index.get_loc(prev) > 1:
            print(f"     {'…':<18}")
        _lab = f"{k} ({ko[k]})" if str(k) in ko else str(k)
        print(f"     {_lab[:17]:<18}{int(r['n']):>7,}{int(r['헛알림']):>8,}"
              f"{r['헛알림률']:>9.1%}{r['hair']:>11.4f}")
        prev = k

    # 항목별 헛알림률과 hair 평균이 같이 움직이나
    if len(g) >= 4:
        c = np.corrcoef(g["헛알림률"], g["hair"])[0, 1]
        print(f"\n     {col} 별 (헛알림률 ↔ hair 평균) 상관 {c:+.3f}  "
              f"[{len(g)}개 항목]")

    # ★ 같은 항목 안에서도 hair 이 가르나 — 항목 평균을 뺀 뒤 다시 잽니다
    res = _center_by(df.loc[sel, "hair"], d[col])
    bad = mask_bad.reindex(d.index).fillna(False).values
    ok = np.isfinite(res)
    if (bad & ok).sum() >= 30 and (~bad & ok).sum() >= 30:
        a = auroc(res[bad & ok], res[~bad & ok])
        raw = auroc(df.loc[mask_bad, "hair"], df.loc[mask_good, "hair"])
        print(f"\n     같은 {col} 안에서만 보면 hair AUROC {raw:.3f} → {a:.3f}")
        if abs(a - 0.5) >= A_STRONG:
            print(f"     ✅ {col} 이 같아도 hair 이 가릅니다 — "
                  f"{col} 만으로 설명이 안 됩니다.")
        elif abs(a - 0.5) >= A_WEAK:
            print(f"     ◐ 약해졌습니다 — 상당 부분이 {col} 였습니다.")
        else:
            print(f"     ❌ {col} 을 고정하니 못 가릅니다 — **{col} 이 원인**입니다.")


def _center_by(vals, groups):
    """항목(견종 등) 평균을 뺀 순위. 범주형 판 `residualize` 입니다."""
    import numpy as np
    import pandas as pd

    r = pd.Series(np.asarray(vals, float)).rank()
    r.index = groups.index
    return (r - r.groupby(groups).transform("mean")).values


def independence(df, mask_bad, mask_good, target: str, control: str) -> float:
    """`control` 을 빼고도 `target` 이 오답을 가르는가.

    왜 필요한가 — `hair`(결)와 `blur`(선명도)는 **같은 걸 재고 있을 수 있습니다.**
    흐리게 하면 결이 사라지니까요 (도구 검증: 흐림 처리로 hair 0.63 → 0.0007).
    그러면 "털이 많다" 는 그냥 "선명하다" 의 다른 말이고, 처방이 달라집니다:

      · hair 가 독립  → 털이 진짜 원인. 견종·부위로 한 번 더 확인합니다
      · blur 로 설명  → 화질 지름길 하나. 촬영으로는 못 잡습니다
    """
    import numpy as np

    sel = mask_bad | mask_good
    sub = df[sel]
    ok = np.isfinite(sub[target]) & np.isfinite(sub[control])
    rows = sub.index[ok]
    bad = mask_bad.reindex(rows).fillna(False).values
    good = mask_good.reindex(rows).fillna(False).values
    if bad.sum() < 30 or good.sum() < 30:
        print(f"\n  ── {target} vs {control} — 표본이 적어 생략 ──")
        return float("nan")

    raw = auroc(df.loc[rows[bad], target], df.loc[rows[good], target])
    r = residualize(df.loc[rows, target], df.loc[rows, control])
    res = auroc(r[bad], r[good])

    print(f"\n  ── {control} 를 빼고 {target} 만 남기면 ──")
    print(f"     그냥 잰 AUROC        {raw:.3f}")
    print(f"     {control} 를 뺀 뒤       {res:.3f}   ({res - raw:+.3f})")
    if abs(res - 0.5) >= A_STRONG:
        print(f"     ✅ **{target} 은 {control} 와 별개입니다.**")
        print(f"        {control} 를 설명하고 나서도 {target} 이 오답을 가릅니다.")
    elif abs(res - 0.5) >= A_WEAK:
        print(f"     ◐ 약해졌습니다 — {target} 의 상당 부분이 {control} 였습니다.")
    else:
        print(f"     ❌ **{target} 은 {control} 의 다른 이름입니다.**")
        print(f"        {control} 를 빼고 나면 아무것도 안 남습니다.")
    return res


# 매니페스트의 **실물** 컬럼 이름. 후보를 여러 개 두는 이유는
# `docs/data/DATASET_CARD.md` 가 "추론이 틀렸던 3곳" 을 따로 적어둘 만큼
# 이 스키마를 한 번 틀린 적이 있기 때문입니다. 지금 맞는 건 앞쪽입니다.
COL_CANDIDATES = {
    "area_ratio": ["area_ratio", "lesion_area_ratio"],
    "img_w": ["img_w", "width"],
    "img_h": ["img_h", "height"],
    "bbox": ["bbox"],
}

# 범주형 — "털 때문인가" 를 값 하나가 아니라 **견종·부위**로 직접 묻습니다.
# `hair` 는 털·각질·주름을 구분 못 하지만, 견종은 털의 양을 대신 말해줍니다.
CAT_CANDIDATES = {"breed": ["breed", "견종"], "region": ["region", "부위"]}

# 부위 코드 — 매니페스트엔 알파벳만 있고 뜻이 우리 문서 어디에도 없었습니다.
# 2026-08-26 확인 (데이터셋 설명). 표에 한글을 같이 찍습니다.
REGION_KO = {"L": "다리", "H": "머리", "B": "몸통", "A": "연접부"}


def _pick(cols, names: list[str]) -> str | None:
    return next((n for n in names if n in cols), None)


def attach_manifest(df):
    """매니페스트에서 `area` / `megapix` 를 붙입니다. 사진을 다시 안 엽니다.

    `area`    = bbox 면적 ÷ 원본 면적. **얼마나 가까이서 찍었나**의 대용
    `megapix` = 원본 화소수. 기기·촬영 설정의 대용

    ⚠️ 못 찾으면 **조용히 NaN 으로 넘어가지 않고 멈춥니다.** 처음 짤 때 컬럼
       이름을 계획 문서에서 베껴 왔다가(`lesion_area_ratio`/`width`/`height`)
       실물(`area_ratio`/`img_w`/`img_h`)과 달라서, 0장인 채로 분석이
       끝까지 돌아갔습니다. 결과가 안 나온 게 아니라 **없는 걸 재고 있었습니다.**
    """
    import numpy as np
    import pandas as pd

    from src import env

    mf = env.work_root() / "manifests" / "manifest_final.parquet"
    m = pd.read_parquet(mf)
    found = {k: _pick(m.columns, v) for k, v in COL_CANDIDATES.items()}
    for k, v in found.items():
        print(f"  {k:<12} → {v or '❌ 없음'}")

    if "image_path" not in m.columns:
        raise SystemExit(f"매니페스트에 image_path 가 없습니다. 있는 것: "
                         f"{sorted(m.columns)[:20]}")

    cats = {k: _pick(m.columns, v) for k, v in CAT_CANDIDATES.items()}
    for k, v in cats.items():
        print(f"  {k:<12} → {v or '❌ 없음'}")

    take = ["image_path"] + [v for v in found.values() if v] \
        + [v for v in cats.values() if v]
    m = m[take].drop_duplicates("image_path")
    out = df.merge(m, on="image_path", how="left")

    matched = int(out[take[1]].notna().sum()) if len(take) > 1 else 0
    if len(take) > 1 and matched == 0:
        raise SystemExit(
            f"❌ 매니페스트와 한 줄도 안 맞았습니다 (image_path 병합 실패).\n"
            f"   저장 파일 예: {df['image_path'].iloc[0]}\n"
            f"   매니페스트 예: {m['image_path'].iloc[0]}")

    # area — 있으면 그대로, 없으면 bbox 로 직접 계산
    if found["area_ratio"]:
        out["area"] = out[found["area_ratio"]]
    elif found["bbox"] and found["img_w"] and found["img_h"]:
        def _a(r):
            b = r[found["bbox"]]
            if b is None or len(b) != 4 or not r[found["img_w"]]:
                return np.nan
            return (((b[2] - b[0]) * (b[3] - b[1]))
                    / (r[found["img_w"]] * r[found["img_h"]]))
        out["area"] = out.apply(_a, axis=1)
        print("  area 를 bbox 에서 직접 계산했습니다.")
    else:
        out["area"] = np.nan

    if found["img_w"] and found["img_h"]:
        out["megapix"] = out[found["img_w"]] * out[found["img_h"]] / 1e6
    else:
        out["megapix"] = np.nan

    for k, v in cats.items():
        out[k] = out[v].astype("string") if v else pd.NA

    for c in EXTRA:
        n = int(out[c].notna().sum())
        mark = "" if n else "   ← 이 값은 빼고 갑니다"
        print(f"  {c:<12} {n:,}/{len(out):,}장{mark}")
    if not out[EXTRA].notna().any().any():
        raise SystemExit(
            f"❌ 붙일 수 있는 값이 하나도 없습니다.\n"
            f"   매니페스트 컬럼: {sorted(m.columns)}\n"
            f"   tools/false_alarm_stats.py 의 COL_CANDIDATES 를 고치세요.")
    return out


def _from_saved(a) -> None:
    """`--save` 로 남긴 값에서 바로 분석합니다. 추론을 안 해서 몇 초면 끝납니다."""
    import pandas as pd

    sp = Path(a.from_saved)
    if not sp.exists():
        raise SystemExit(f"저장 파일이 없습니다: {sp}")
    df = pd.read_csv(sp) if sp.suffix == ".csv" else pd.read_parquet(sp)
    print(f"\n[불러옴] {sp}  ({len(df):,}행) — 추론 안 합니다")

    print("\n[매니페스트에서 값 붙이기]")
    df = attach_manifest(df)
    use = STATS + [c for c in EXTRA if df[c].notna().any()]

    fa = df["is_normal"] & df["said_abnormal"]
    tn = df["is_normal"] & ~df["said_abnormal"]
    fn = ~df["is_normal"] & ~df["said_abnormal"]
    tp = ~df["is_normal"] & df["said_abnormal"]

    print("\n" + "=" * 68)
    print(f" 헛알림 {int(fa.sum()):,} / 놓침 {int(fn.sum()):,}")
    print("=" * 68)
    compare(df, fa, tn, "헛알림 — 정상인데 병원 보낸 사진", "헛알림", "맞게 넘김",
            stats=use)
    compare(df, fn, tp, "놓침 — 병변인데 괜찮다고 한 사진", "놓침", "잡음", stats=use)

    print("\n  상관 (정상 사진에서만)")
    corr = df.loc[df["is_normal"], use].corr()
    print(f"    {'':<10}" + "".join(f"{c:>10}" for c in use))
    for r in use:
        print(f"    {r:<10}" + "".join(f"{corr.loc[r, c]:>10.3f}" for c in use))

    print("\n" + "=" * 68)
    print(' 털 때문인가 — 견종·부위로 직접 묻습니다')
    print("=" * 68)
    for c in ("breed", "region"):
        by_category(df, fa, tn, c)

    print("\n" + "=" * 68)
    print(" 무엇이 무엇의 그림자인가 (연속값끼리)")
    print("=" * 68)
    pairs = [("hair", "blur"), ("blur", "hair")]
    if "area" in use:
        pairs += [("hair", "area"), ("area", "hair"),
                  ("blur", "area"), ("area", "blur")]
    got = {}
    for t, c in pairs:
        got[(t, c)] = independence(df, fa, tn, t, c)
    verdict_shadow(df, fa, tn, got, use)


def verdict_shadow(df, fa, tn, got: dict, use: list) -> None:
    """서로를 뺀 결과가 **대칭이 아니면** 그게 단서입니다.

    ⚠️ 처음엔 (값, 뺀 것) 짝 중 **가장 센 것**으로 판정했습니다. 그러면 상관이
       거의 없는 상대를 골라 이깁니다 — `hair` vs `area`(상관 −0.067)는 당연히
       살아남으니까요. 실제로 그 버전이 "hair 가 원인 ✅" 이라고 했는데,
       정작 진짜 경쟁자인 `blur`(상관 0.896)에게는 문턱을 못 넘었습니다.
       그래서 지금은 **가장 약해진 쪽(최악)**으로 판정합니다 —
       "제일 센 상대를 빼고도 버티는가" 가 물어야 할 질문입니다.
    """
    print("\n  ── 읽기 ──")
    raw = {k: auroc(df.loc[fa, k], df.loc[tn, k]) for k in use}

    rows = [(t, c, raw[t], v, abs(v - 0.5))
            for (t, c), v in got.items() if v == v]
    if not rows:
        print("  판정할 게 없습니다.")
        return

    print(f"\n    {'값':<8}{'뺀 것':<8}{'그냥':>8}{'뺀 뒤':>8}{'남은 세기':>10}   방향")
    print("    " + "─" * 52)
    for t, c, r0, v, st in sorted(rows, key=lambda x: -x[4]):
        flip = "뒤집힘" if (r0 - 0.5) * (v - 0.5) < 0 else "유지"
        print(f"    {t:<8}{c:<8}{r0:>8.3f}{v:>8.3f}{st:>10.3f}   {flip}")

    # ★ 값마다 **최악**(가장 약해진) 잔차로 판정합니다
    worst: dict = {}
    for t, c, r0, v, st in rows:
        if t not in worst or st < worst[t][2]:
            worst[t] = (c, v, st)

    print(f"\n    ── 값마다 **제일 센 상대**를 뺐을 때 ──")
    print(f"\n    {'값':<8}{'제일 센 상대':<14}{'남은 세기':>10}   판정")
    print("    " + "─" * 48)
    order = sorted(worst.items(), key=lambda kv: -kv[1][2])
    for t, (c, v, st) in order:
        j = ("버팀" if st >= A_STRONG else "약함" if st >= A_WEAK else "무너짐")
        print(f"    {t:<8}{c:<14}{st:>10.3f}   {j}")

    best_t, (best_c, best_v, best_st) = order[0]
    print(f"\n  가장 오래 버티는 값: **{best_t}** "
          f"({best_c} 를 빼도 {best_st:.3f} 남음)")
    if best_st >= A_STRONG:
        print(f"  ✅ 제일 센 상대를 빼고도 문턱({A_STRONG})을 넘습니다 — 원인에 가깝습니다.")
    elif best_st >= A_WEAK:
        print(f"  ◐ 문턱({A_STRONG})은 **못 넘습니다**({best_st:.3f}). 다른 값들보다")
        print("     오래 버티지만, '원인' 이라고 못 박진 못합니다. **더 가깝다** 까지입니다.")
    else:
        print("  ❌ 서로를 빼면 전부 무너집니다 — 이 값들은 한 원인의 그림자들입니다.")

    dead = [t for t, (_, _, st) in worst.items() if st < A_WEAK]
    if dead:
        print(f"\n  기각: {', '.join(dead)} — 다른 값을 빼면 아무것도 안 남습니다.")
    print("\n  ⚠️ 상관이 높은 값끼리 잔차를 내면 **둘 다** 약해집니다. 그래서")
    print("     '약해졌다' 자체는 증거가 아닙니다. 봐야 할 건 **누가 더 버티나**와")
    print("     **방향이 뒤집혔나** 입니다. 뒤집혔다면 그 값의 원래 신호는")
    print("     상대를 통해 온 것입니다.")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--release", help="06 이 만든 릴리스 폴더 (--from-saved 면 불필요)")
    ap.add_argument("--tag", default=None, help="쓸 크롭 태그 (기본: 1단계 태그)")
    ap.add_argument("--limit", type=int, default=0, help="맛보기로 N장만")
    ap.add_argument("--out", default="docs/results/헛알림_사진통계_실측.md")
    ap.add_argument("--save", metavar="파일", default=None,
                    help="장별 값을 저장 (.parquet/.csv). 다시 20분 안 돌리려면 쓰세요")
    ap.add_argument("--from-saved", metavar="파일", dest="from_saved", default=None,
                    help="★ --save 로 남긴 값에서 바로 분석 (추론 안 함, 몇 초)")
    a = ap.parse_args(argv)
    if not a.release and not a.from_saved:
        raise SystemExit("--release 또는 --from-saved 중 하나가 필요합니다.")

    import numpy as np
    import pandas as pd

    from src import agent, crop, env, stages

    if a.from_saved:
        _from_saved(a)
        return

    # ── 0) 경로부터 확인합니다 — 20분 돌린 뒤 오타로 죽으면 아깝습니다 ──
    rel = Path(a.release).expanduser()
    if not rel.exists():
        raise SystemExit(
            f"릴리스 폴더가 없습니다: {rel}\n"
            "   탐색기에서 checkpoints 가 들어 있는 폴더의 주소를 복사해 붙이세요.\n"
            "   폴더 이름에 공백이 있으면 \"따옴표\" 로 감싸세요.")
    hits = sorted(rel.glob("**/stage1_*/best.pt"))
    if not hits:
        raise SystemExit(
            f"{rel} 안에서 stage1_*/best.pt 를 못 찾았습니다.\n"
            f"   여기 있는 것: {[p.name for p in sorted(rel.iterdir())[:10]]}\n"
            "   압축을 풀면 한 겹 더 감싸져 있는 일이 흔합니다 (release/release/…).")
    print(f"[릴리스] {hits[0].parent.name}")
    for q in sorted(rel.glob("**/stage2_*/best.pt")):
        print(f"         {q.parent.name}")
    if not (list(rel.glob("**/temperature.json")) or []):
        print("         ⚠️ temperature.json 이 없습니다 — 이 릴리스는 보정 전입니다.")

    tag = a.tag or agent.STAGE1_TAG
    mf = env.work_root() / "manifests" / "manifest_final.parquet"
    if not mf.exists():
        raise SystemExit(f"매니페스트가 없습니다: {mf}")

    df = pd.read_parquet(mf)
    if "is_holdout" not in df.columns:
        raise SystemExit("매니페스트에 is_holdout 이 없습니다 — --finalize 를 돌렸나요?")
    df = df[df["is_holdout"].astype(bool)].reset_index(drop=True)
    if df.empty:
        raise SystemExit("holdout 행이 없습니다. --finalize 를 돌렸나요?")
    df = stages.to_stage1(df)
    print(f"[holdout] {len(df):,}장")

    # 크롭 경로는 매니페스트 컬럼이 아니라 **태그로 다시 계산**합니다 —
    # manifest 의 crop_path 는 --margins 의 첫 항목 것이라 f320 이 아닐 수 있습니다.
    out_dir = env.work_root() / "crops"
    df["path"] = df["image_path"].apply(
        lambda p: str(crop._out_path(p, out_dir, tag)))
    have = df["path"].apply(lambda p: Path(p).exists())
    if not have.all():
        print(f"  ⚠️ 크롭 {tag} 가 {int((~have).sum()):,}장 없습니다 — 그 행은 뺍니다.")
        df = df[have].reset_index(drop=True)
    if a.limit:
        df = df.sample(min(a.limit, len(df)), random_state=0).reset_index(drop=True)
        print(f"  (맛보기 {len(df):,}장)")

    # ── 1) 모델이 뭐라고 했나 ──
    ag = agent.ScreeningAgent.from_release(a.release, device="cpu", stage1_only=True)
    print(f"[모델] 임계값 {ag.thr:.4f}  (CPU 추론 — 시간이 좀 걸립니다)")

    # ⚠️ `ag.screen()` 을 쓰면 안 됩니다. 그건 사진을 **다시 자릅니다** —
    #    여기 있는 건 이미 학습이 쓴 크롭이라, 자르면 두 번 자르는 게 됩니다.
    #    1단계 엔진에 크롭 경로를 바로 넘깁니다.
    scores = np.empty(len(df), dtype=np.float32)
    for i, p in enumerate(df["path"]):
        scores[i] = dict(ag.s1.predict(p).topk).get(ag._ab, 0.0)
        if (i + 1) % 500 == 0:
            print(f"    {i + 1:,}/{len(df):,}")
    df["score"] = scores
    df["said_abnormal"] = df["score"] >= ag.thr
    df["is_normal"] = df["label"] == stages.NORMAL_LABEL

    # ── 2) 사진의 값들 ──
    print("[사진 통계] 재는 중 …")
    st = pd.DataFrame([image_stats(p) for p in df["path"]], index=df.index)
    df = pd.concat([df, st], axis=1)

    # ── 3) 비교 ──
    fa = df["is_normal"] & df["said_abnormal"]          # 헛알림
    tn = df["is_normal"] & ~df["said_abnormal"]         # 맞게 넘긴 정상
    fn = ~df["is_normal"] & ~df["said_abnormal"]        # 놓친 병변
    tp = ~df["is_normal"] & df["said_abnormal"]         # 잡은 병변

    print("\n" + "=" * 68)
    print(f" 헛알림 {int(fa.sum()):,} / 놓침 {int(fn.sum()):,}"
          f"  (정상 {int(df['is_normal'].sum()):,} / 병변 {int((~df['is_normal']).sum()):,})")
    print("=" * 68)
    compare(df, fa, tn, "헛알림 — 정상인데 병원 보낸 사진", "헛알림", "맞게 넘김")
    compare(df, fn, tp, "놓침 — 병변인데 괜찮다고 한 사진", "놓침", "잡음")

    print("\n  ── 어떻게 읽나 ──")
    print(f"  판정 문턱: |d| ≥ {D_STRONG} **그리고** |AUROC − 0.5| ≥ {A_STRONG}")
    print("  (d 만 보면 속습니다 — 퍼짐이 작으면 차이가 없어도 d 가 커집니다)")
    print("\n  '차이 있음' 이 하나라도 있으면 → **촬영 가이드로 잡을 수 있는 문제**입니다.")
    print("     그 값을 촬영 단계에서 걸러내면 헛알림이 줄어듭니다.")
    print("  전부 '차이 없음' 이면 → 사진 조건 문제가 **아닙니다.** 모델이나 라벨 쪽이고,")
    print("     촬영 가이드를 아무리 손봐도 안 줄어듭니다. 이것도 결론입니다.")
    print("  AUROC < 0.5 는 틀림이 아니라 **방향이 반대**라는 뜻입니다 "
          "(예: 헛알림 쪽이 더 흐림).")

    # ── 4) 두 값이 같은 걸 재고 있나 ──
    print("\n" + "=" * 68)
    print(" 헛알림 — 두 값이 같은 걸 재는 건 아닌가")
    print("=" * 68)
    corr = df.loc[df["is_normal"], STATS].corr().loc["hair", "blur"]
    print(f"\n  정상 사진에서 hair 와 blur 의 상관 {corr:+.3f}")
    h_res = independence(df, fa, tn, "hair", "blur")
    b_res = independence(df, fa, tn, "blur", "hair")

    print("\n  ── 둘 중 어느 쪽이 진짜인가 ──")
    if h_res == h_res and b_res == b_res:                   # nan 아님
        hs, bs = abs(h_res - 0.5), abs(b_res - 0.5)
        if hs >= A_STRONG and bs < A_WEAK:
            print("  → **결(hair)이 원인입니다.** 선명도는 결의 그림자였습니다.")
            print("     촬영 가이드 '털을 헤쳐서 피부가 보이게' 가 살아납니다.")
        elif bs >= A_STRONG and hs < A_WEAK:
            print("  → **선명도(blur)가 원인입니다.** 결은 선명도의 다른 이름이었습니다.")
            print("     STEP 6 의 화질 지름길 그대로입니다 — 촬영으로는 못 잡습니다.")
        elif hs >= A_STRONG and bs >= A_STRONG:
            print("  → **둘 다 독립적으로 기여합니다.** 한쪽만 고쳐선 부족합니다.")
        else:
            print("  → 서로를 빼면 둘 다 약해집니다. 얽혀 있어서 이 데이터로는")
            print("     어느 쪽이 원인인지 못 가릅니다. 그것도 결론입니다.")

    if a.save:
        cols = ["image_path", "label", "is_normal", "score", "said_abnormal",
                "path", *STATS]
        sp = Path(a.save)
        sp.parent.mkdir(parents=True, exist_ok=True)
        keep = df[[c for c in cols if c in df.columns]]
        if sp.suffix == ".csv":
            keep.to_csv(sp, index=False, encoding="utf-8-sig")
        else:
            keep.to_parquet(sp)
        print(f"\n  장별 값 저장: {sp}  ({len(keep):,}행)")
        print("     → 다시 20분 안 돌리고 여기서 더 파볼 수 있습니다.")
    print(f"\n  결과를 {a.out} 에 적어주세요 (규칙 1 — 실측만).")


if __name__ == "__main__":
    main()
