"""두 병변 클래스가 **데이터에서** 어떻게 겹치나 — 모델을 안 거치고 잽니다.

    uv run python tools/class_overlap.py --manifest <manifest_final.parquet>
    uv run python tools/class_overlap.py --manifest ... --a A4 --b A1

왜 이 도구가 있나
-----------------
STEP 16 holdout 혼동행렬에서 **A4→A1 이 0.25 로 대각선(0.25)과 동률**입니다.
A4(농포·여드름) 2,606장 중 652장을 A4 로, 똑같이 652장을 A1(구진·플라크)으로
부릅니다. "장수가 적어서" 라면 A6(1,501장 수준)이 recall 0.619 인 게 설명이 안 됩니다.

그럼 왜인가. 가능한 답은 크게 둘이고, **고칠 방법이 서로 다릅니다**:

| 가설 | 뜻 | 맞으면 다음 수 |
|---|---|---|
| ① **촬영·데이터 조건이 다르다** | A4 는 더 가까이/작게/특정 부위에 몰려 찍혔다 | 샘플링·가중·증강 (모델 쪽 지렛대 있음) |
| ② **라벨 정의가 겹친다** | 같은 피부를 누구는 A4, 누구는 A1 이라 적었다 | 병합·계층 분류 (⚠️ 작업 규칙 6 — 후반에) |

이 도구는 **모델을 안 씁니다.** 매니페스트만 읽습니다. 그래서 지금 돌려도
"학습률이 덜 수렴해서 나온 값 아닌가" 라는 반론을 받지 않습니다 —
2단계 학습률 진단 결과와 **무관하게** 유효합니다.

⚠️ 안 하는 것: 사진을 눈으로 보고 A1/A4 를 가르는 일. 수의사가 아니면 못 합니다
   (`false_alarms.png` 30장 판독을 접었던 것과 같은 이유).

무엇을 재나
-----------
**숫자 축** (사진을 열지 않고 매니페스트에서 바로 나옵니다)

| 이름 | 뜻 | 왜 보나 |
|---|---|---|
| `area` | 병변 bbox 면적 ÷ 원본 면적 (`area_ratio`) | 얼마나 가까이·크게 찍혔나 |
| `boxlong` | bbox 긴 변 (px) | 병변 자체의 크기 |
| `boxrel` | 긴 변 ÷ 원본 긴 변 | 화소수와 무관한 상대 크기 |
| `boxaspect` | 긴 변 ÷ 짧은 변 | 둥근가(농포) 넓게 퍼졌나(플라크) |
| `nlesion` | 한 사진의 병변 개수 | 농포·여드름은 여러 개 나는 게 특징 |
| `megapix` | 원본 화소수 | 기기·촬영 설정의 대용 |

**갈래 축** (`region` 부위 · `breed` 견종 · `gender` · `age` · `synthetic`)
→ 분포표 + **총변동거리(TVD)** + 그 축 하나로 가를 수 있나(AUROC).
  ⚠️ 갈래 AUROC 는 **교차 인코딩(out-of-fold)** 으로 냅니다. 그냥 인코딩하면
     갈래가 많을수록(견종 200종) 자기 자신을 보고 맞혀 **부풀어** 오르고,
     leave-one-out 으로 하면 **반대로 무너집니다** (`cat_auroc` 주석 참고).

**동시출현** — 같은 개체(`animal_id`)·같은 부위에서 두 라벨이 같이 나오나.
  15개 병변쌍 전부에 대해 재서 **A–B 가 몇 위인지**로 읽습니다.
  ⚠️ 문턱을 임의로 정하지 않으려고 이렇게 합니다. 한 마리가 농포와 구진을
     동시에 가질 수 있으니, 값 자체보다 **다른 쌍 대비**가 의미 있습니다.
  ⚠️ `dup_cluster`(phash) 는 여기서 볼 필요가 없습니다 — `dedup.run` 이
     라벨 충돌 클러스터를 **이미 전부 버렸기** 때문에 구조상 0 입니다.

판정 (결과를 보기 전에 못 박습니다 — CLAUDE.md 작업 규칙 2)
-----------------------------------------------------------
    ① 숫자·갈래 축 중 **하나라도** |d| ≥ 0.5 이고 |AUROC − 0.5| ≥ 0.15
       →  **가설 ① 촬영·데이터 조건이 다름.** 그 축으로 가중·증강을 걸 수 있습니다

    ② 모든 축이 "차이 없음" 이고, 동시출현이 15쌍 중 **상위 3위 이내**이면서
       중앙값의 **1.5배 이상**
       →  **가설 ② 라벨 정의가 겹침.** 백본·학습률로는 안 풀립니다

    ③ 그 밖 (조금씩만 다름 / 신호가 엇갈림)
       →  **구분 불가.** 데이터 쪽엔 지렛대가 안 보이니 다음 수는 모델 쪽입니다

⚠️ 이 도구가 답하지 **않는** 것: 위 차이가 **모델의 오분류를 만들었는지**.
   그건 개입 실험(가중을 걸고 다시 학습)으로만 알 수 있습니다. 여기서 나오는
   건 관찰입니다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ⚠️ `d` · `AUROC` · 판정 문턱의 정의는 `src/effectsize.py` 한 곳에만 있습니다.
from src.effectsize import A_STRONG, D_STRONG, auroc, cohens_d, verdict  # noqa: E402

LESIONS = ["A1", "A2", "A3", "A4", "A5", "A6"]

# 숫자 축 — (이름, 설명)
NUMERIC = [
    ("area", "병변 면적 비율"),
    ("boxlong", "bbox 긴 변 (px)"),
    ("boxrel", "긴 변 ÷ 원본 긴 변"),
    ("boxaspect", "긴 변 ÷ 짧은 변"),
    ("nlesion", "사진당 병변 수"),
    ("megapix", "원본 화소수 (백만)"),
]
# `chunk`/`src_split` 은 **어느 다운로드 청크에서 왔나** 입니다. 한 클래스가
# 특정 청크에 몰려 있으면 그건 병변의 성질이 아니라 수집 경로의 차이입니다.
CATEGORICAL = ["region", "breed", "gender", "age", "synthetic",
               "chunk", "src_split"]

MIN_ROWS = 200          # 이보다 적으면 아래 숫자를 못 믿습니다
CO_TOP_RANK = 3         # 동시출현 판정: 15쌍 중 상위 몇 위까지
CO_RATIO = 1.5          # 그리고 중앙값의 몇 배 이상


# ──────────────────────────────────────────────────────────────
# 매니페스트 → 축
# ──────────────────────────────────────────────────────────────
def _bbox_xy(b):
    """`bbox` 는 [x1, y1, x2, y2] 입니다 (labels.parse_meta)."""
    try:
        x1, y1, x2, y2 = (float(v) for v in b)
    except Exception:
        return None
    return abs(x2 - x1), abs(y2 - y1)


def build_axes(df):
    """매니페스트에 숫자 축 컬럼을 붙입니다.

    ⚠️ 컬럼 이름을 계획 문서에서 베끼면 안 됩니다. 실물은 `area_ratio` /
       `img_w` / `img_h` 입니다 (`docs/data/DATASET_CARD.md`). 이름이 틀리면
       병합이 조용히 NaN 을 만들고 **없는 걸 재면서 분석이 끝까지 돕니다.**
    """
    import numpy as np
    import pandas as pd

    need = ["label", "area_ratio", "img_w", "img_h", "bbox", "animal_id"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(
            f"매니페스트에 컬럼이 없습니다: {missing}\n"
            f"   있는 컬럼: {list(df.columns)}\n"
            "   → docs/data/DATASET_CARD.md 의 실물 스키마와 대조하세요.")

    out = df.copy()
    out["area"] = pd.to_numeric(out["area_ratio"], errors="coerce")
    out["megapix"] = (pd.to_numeric(out["img_w"], errors="coerce")
                      * pd.to_numeric(out["img_h"], errors="coerce")) / 1e6
    out["nlesion"] = (pd.to_numeric(out["n_lesion"], errors="coerce")
                      if "n_lesion" in out.columns else np.nan)

    wh = out["bbox"].map(lambda b: _bbox_xy(b) if b is not None else None)
    bw = wh.map(lambda t: t[0] if t else np.nan).astype(float)
    bh = wh.map(lambda t: t[1] if t else np.nan).astype(float)
    lo, sh = np.maximum(bw, bh), np.minimum(bw, bh)
    out["boxlong"] = lo
    out["boxrel"] = lo / np.maximum(pd.to_numeric(out["img_w"], errors="coerce"),
                                    pd.to_numeric(out["img_h"], errors="coerce"))
    with np.errstate(divide="ignore", invalid="ignore"):
        out["boxaspect"] = np.where(sh > 0, lo / sh, np.nan)

    # bbox 가 한 장도 안 풀렸으면 **멈춥니다** — 조용한 NaN 이 제일 위험합니다
    if not np.isfinite(out["boxlong"]).any():
        raise SystemExit("bbox 에서 크기를 하나도 못 읽었습니다. "
                         "`bbox` 형식이 [x1,y1,x2,y2] 가 맞는지 확인하세요.")
    return out


def cat_auroc(cat, y, n_folds: int = 5, seed: int = 0):
    """갈래 축 하나로 가를 수 있나 — **교차 인코딩(out-of-fold)**.

    각 행의 점수 = "그 행이 든 fold 를 뺀" 나머지에서 잰 같은 갈래의 양성 비율.
    자기 자신을 못 봐서 갈래가 많아도 부풀지 않습니다 (견종 120종을 그냥
    target-encode 하면 자기 자신을 보고 맞혀 0.75+ 가 나옵니다).

    ⚠️ **leave-one-out 으로 하면 안 됩니다 — 반대로 무너집니다.**
       LOO 점수는 `(합 − 자기 y) / (n − 1)` 이라, 신호가 하나도 없는 축에서도
       양성이 **항상** 음성보다 `1/(n−1)` 만큼 낮게 나옵니다. 순위만 보는
       AUROC 에서는 이 미세한 차이가 **완벽한 분리(AUROC 0.000)** 로 읽힙니다.
       실제로 `synthetic` 처럼 갈래가 하나뿐인 축이 "차이 있음" 으로 찍혔습니다.
       fold 를 섞으면 이 규칙성이 깨집니다.
    """
    import numpy as np
    import pandas as pd

    cat = pd.Series(cat).astype("string").fillna("(없음)").reset_index(drop=True)
    y = pd.Series(np.asarray(y, float)).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, n_folds, len(y))
    score = np.full(len(y), np.nan)
    for k in range(n_folds):
        te, tr = fold == k, fold != k
        if not te.any() or not tr.any():
            continue
        m = y[tr].groupby(cat[tr]).mean()
        score[te] = cat[te].map(m).astype(float).to_numpy()
    ok = np.isfinite(score)
    yv = y.to_numpy()
    return auroc(score[ok & (yv == 1)], score[ok & (yv == 0)])


def tvd(cat_a, cat_b) -> float:
    """총변동거리 — 두 분포가 얼마나 다른가. 0 = 같음, 1 = 겹치는 갈래 없음."""
    import pandas as pd

    pa = pd.Series(cat_a).astype("string").fillna("(없음)").value_counts(normalize=True)
    pb = pd.Series(cat_b).astype("string").fillna("(없음)").value_counts(normalize=True)
    keys = set(pa.index) | set(pb.index)
    return 0.5 * sum(abs(float(pa.get(k, 0.0)) - float(pb.get(k, 0.0))) for k in keys)


# ──────────────────────────────────────────────────────────────
# 동시출현
# ──────────────────────────────────────────────────────────────
def cooccurrence(df, x: str, y: str, keys: list[str]) -> float:
    """`keys` 로 묶었을 때 x·y 를 **둘 다** 가진 묶음의 비율.

    분모 = x 나 y 중 **하나라도** 가진 묶음 (자카드). 한쪽이 드물다고
    비율이 저절로 커지지 않게 하려는 것입니다.
    """
    sub = df[df["label"].isin([x, y])]
    if sub.empty:
        return float("nan")
    g = sub.groupby(keys)["label"].agg(lambda s: frozenset(s))
    both = sum(1 for v in g if x in v and y in v)
    return both / len(g) if len(g) else float("nan")


def cooccurrence_table(df, keys: list[str]) -> list[tuple]:
    rows = []
    for i, x in enumerate(LESIONS):
        for y in LESIONS[i + 1:]:
            rows.append((f"{x}–{y}", cooccurrence(df, x, y, keys)))
    rows.sort(key=lambda r: (-r[1] if r[1] == r[1] else 1))
    return rows


# ──────────────────────────────────────────────────────────────
# 본체
# ──────────────────────────────────────────────────────────────
def analyze(df, a: str, b: str, out_lines: list) -> dict:
    import numpy as np

    def say(s=""):
        print(s)
        out_lines.append(s)

    ma, mb = df["label"] == a, df["label"] == b
    na, nb = int(ma.sum()), int(mb.sum())
    say(f"## {a} vs {b} — 데이터 쪽 차이\n")
    say(f"* {a} {na:,}장 / {b} {nb:,}장  (전체 {len(df):,}행)")
    if min(na, nb) < MIN_ROWS:
        say(f"* ⚠️ 한쪽이 {MIN_ROWS}장 미만이라 아래 숫자는 못 믿습니다.")
    say()

    strong, results = [], {}

    # ── 숫자 축 ──
    say("### 숫자 축\n")
    say(f"| 축 | 뜻 | {a} | {b} | Δ | d | AUROC | 판정 |")
    say("|---|---|---:|---:|---:|---:|---:|---|")
    for k, desc in NUMERIC:
        if k not in df.columns:
            continue
        x, y = df.loc[ma, k], df.loc[mb, k]
        if not np.isfinite(x).any() or not np.isfinite(y).any():
            say(f"| `{k}` | {desc} | — | — | — | — | — | 못 잼 |")
            continue
        d, ar = cohens_d(x, y), auroc(x, y)
        v = verdict(d, ar)
        results[k] = {"d": d, "auroc": ar, "verdict": v,
                      f"mean_{a}": float(np.nanmean(x)), f"mean_{b}": float(np.nanmean(y))}
        if abs(d) >= D_STRONG and abs(ar - 0.5) >= A_STRONG:
            strong.append(k)
        say(f"| `{k}` | {desc} | {np.nanmean(x):.4g} | {np.nanmean(y):.4g} | "
            f"{np.nanmean(x) - np.nanmean(y):+.4g} | {d:+.2f} | {ar:.3f} | {v} |")
    say()

    # ── 갈래 축 ──
    say("### 갈래 축\n")
    say("| 축 | 갈래 수 | TVD | AUROC (교차) | 판정 |")
    say("|---|---:|---:|---:|---|")
    for c in CATEGORICAL:
        if c not in df.columns:
            continue
        sub = df.loc[ma | mb]
        ar = cat_auroc(sub[c], (sub["label"] == a).astype(int))
        t = tvd(df.loc[ma, c], df.loc[mb, c])
        nlev = int(sub[c].astype("string").fillna("(없음)").nunique())
        # 갈래 축엔 d 가 없으므로 AUROC 만으로 판정합니다 (문턱은 같은 값)
        v = ("**차이 있음**" if abs(ar - 0.5) >= A_STRONG
             else "조금" if abs(ar - 0.5) >= 0.06 else "차이 없음")
        if abs(ar - 0.5) >= A_STRONG:
            strong.append(c)
        results[c] = {"auroc": ar, "tvd": t, "levels": nlev, "verdict": v}
        say(f"| `{c}` | {nlev} | {t:.3f} | {ar:.3f} | {v} |")
    say()

    # 부위는 갈래가 적으니 표까지 보여줍니다 (읽는 사람이 방향을 알아야 함)
    if "region" in df.columns:
        say("#### 부위(`region`)별 — 그 부위 안에서 두 라벨의 비중\n")
        say(f"| 부위 | {a} | {b} | {a} 비중 |")
        say("|---|---:|---:|---:|")
        sub = df.loc[ma | mb]
        for r, g in sorted(sub.groupby(sub["region"].astype("string").fillna("(없음)"))):
            ca = int((g["label"] == a).sum())
            cb = int((g["label"] == b).sum())
            if ca + cb == 0:
                continue
            say(f"| {r} | {ca:,} | {cb:,} | {ca / (ca + cb):.1%} |")
        say()

    # ── 동시출현 ──
    say("### 동시출현 — 같은 개체·같은 부위에 두 라벨이 같이 있나\n")
    co_res = {}
    for name, keys in [("개체(`animal_id`)", ["animal_id"]),
                       ("개체+부위", ["animal_id", "region"])]:
        if any(k not in df.columns for k in keys):
            continue
        table = cooccurrence_table(df, keys)
        vals = [v for _, v in table if v == v]
        med = float(np.median(vals)) if vals else float("nan")
        pair = f"{a}–{b}" if LESIONS.index(a) < LESIONS.index(b) else f"{b}–{a}"
        rank = next((i + 1 for i, (p, _) in enumerate(table) if p == pair), None)
        val = next((v for p, v in table if p == pair), float("nan"))
        # ⚠️ 순위는 **다른 쌍이 있어야** 뜻이 있습니다. 두세 쌍만 잴 수 있으면
        #    "15쌍 중 1위" 가 저절로 나옵니다 — 그건 결과가 아닙니다.
        ratio = (val / med) if (med and med == med) else float("nan")
        enough = len(vals) >= 10
        say(f"**{name} 기준** — {pair} = **{val:.1%}**, "
            f"{len(vals)}쌍 중 **{rank}위**, 중앙값 {med:.1%}"
            + (f" (= {ratio:.2f}배)" if ratio == ratio else " (중앙값 0 — 배수 못 냄)")
            + ("" if enough else "  ⚠️ 잴 수 있는 쌍이 적어 순위는 못 믿습니다")
            + "\n")
        say("| 쌍 | 비율 |")
        say("|---|---:|")
        for p, v in table:
            if v != v:                       # 그 쌍이 데이터에 없음 — 줄을 안 냅니다
                continue
            mark = " ←" if p == pair else ""
            say(f"| {p} | {v:.1%}{mark} |")
        say()
        co_res[name] = {"value": val, "rank": rank, "median": med,
                        "ratio": ratio, "n_pairs": len(vals), "enough": enough}

    # ── 판정 ──
    say("### 판정 (문턱은 실행 전에 못 박은 값입니다)\n")
    co = co_res.get("개체+부위") or co_res.get("개체(`animal_id`)") or {}
    _r = co.get("ratio", float("nan"))
    co_hit = (co.get("enough") and co.get("rank") is not None
              and co["rank"] <= CO_TOP_RANK and _r == _r and _r >= CO_RATIO)
    if strong:
        vd = f"① 촬영·데이터 조건이 다름 — 축: {', '.join(sorted(set(strong)))}"
        nxt = ("그 축으로 가중·증강을 걸어 **개입 실험**을 하세요. "
               "관찰만으로는 원인이라고 못 박습니다.")
    elif co_hit:
        vd = "② 라벨 정의가 겹침"
        nxt = ("백본·학습률로는 안 풀립니다. 병합 또는 계층 분류를 검토하되 "
               "⚠️ 작업 규칙 6 — 자동 실험을 다 짜낸 뒤에.")
    else:
        vd = "③ 구분 불가 — 데이터 쪽엔 지렛대가 안 보입니다"
        nxt = "다음 수는 모델 쪽(손실 가중·학습률)입니다."
    say(f"**{vd}**\n")
    say(f"→ {nxt}\n")
    say("⚠️ 이건 **관찰**입니다. 위 차이가 오분류를 *만들었는지* 는 "
        "가중을 걸고 다시 학습해봐야 압니다.")

    return {"a": a, "b": b, "n_a": na, "n_b": nb, "axes": results,
            "cooccurrence": co_res, "verdict": vd, "strong_axes": sorted(set(strong))}


def main(argv=None) -> None:
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", default="data/work/manifests/manifest_final.parquet")
    ap.add_argument("--a", default="A4", help="관심 클래스 (기본 A4 농포·여드름)")
    ap.add_argument("--b", default="A1", help="비교 클래스 (기본 A1 구진·플라크)")
    ap.add_argument("--holdout-only", action="store_true",
                    help="holdout 행만 (기본: 전체 — 매니페스트는 모델을 안 거치니 전체가 맞습니다)")
    ap.add_argument("--out", default="docs/results/A4_A1_데이터겹침_실측.md")
    a = ap.parse_args(argv)

    path = Path(a.manifest)
    if not path.exists():
        raise SystemExit(f"매니페스트가 없습니다: {path}")
    df = pd.read_parquet(path)
    print(f"[class_overlap] {path.name} — {len(df):,}행")

    if a.holdout_only:
        if "is_holdout" not in df.columns:
            raise SystemExit("`is_holdout` 컬럼이 없습니다. split.assign() 을 거친 "
                             "매니페스트인지 확인하세요.")
        df = df[df["is_holdout"]]
        print(f"   holdout 만 — {len(df):,}행")

    df = build_axes(df)
    lines: list[str] = [f"# {a.a} vs {a.b} — 데이터 겹침 실측", "",
                        f"> `tools/class_overlap.py` 출력. 원본 {path.name} ({len(df):,}행).",
                        "> 모델을 안 거칩니다 — 매니페스트만 읽습니다.", ""]
    analyze(df, a.a, a.b, lines)

    outp = ROOT / a.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[class_overlap] 저장: {outp}")


if __name__ == "__main__":
    main()
