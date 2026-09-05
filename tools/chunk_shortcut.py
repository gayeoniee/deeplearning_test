"""클래스와 **청크(다운로드 묶음)** 가 얼마나 얽혀 있나 — 그리고 모델이 그걸 쓰나.

    uv run --extra train python tools/chunk_shortcut.py

왜 이 도구가 있나
-----------------
AI Hub 561 을 세 묶음(TL01·TL02·VL01)으로 받아 합쳤는데, **묶음마다 담긴 클래스가
다릅니다.** A1·A7 은 TL02 에 한 장도 없고, A3~A6 은 TL01 에 한 장도 없습니다.

그래서 원리적으로 모델은 병변 형태를 안 보고 **"어느 묶음에서 온 사진인가"**
만으로 {A1,A7} 과 {A3,A4,A5,A6} 을 가를 수 있습니다. 묶음이 다르면 촬영 시기·
장비·라벨러가 다를 수 있고, 그건 픽셀에 흔적을 남깁니다. 이 리포가 이미 두 번
잡은 지름길(크롭 배율 · 선명도)과 같은 종류입니다.

⚠️ **두 가지를 구분해야 합니다.**

    ① 설계가 얼마나 얽혀 있나   — 청크만으로 라벨을 얼마나 맞히나 (**상한선**)
    ② 모델이 그걸 실제로 쓰나   — 청크를 못 쓰는 상황에서 성능이 떨어지나

①이 커도 ②가 아니면 문제가 없습니다. 청크는 **모델에 입력되지 않으니까요** —
모델이 쓰려면 청크가 픽셀에 시각적 서명을 남겨야 합니다. 그래서 ①만 보고
"지름길이다" 라고 하면 안 됩니다.

②를 재는 법 — 세 갈래를 다 봅니다:

    (a) **한 청크 안에서만** 평가. 거기선 청크가 상수라 단서로 못 씁니다.
        VL01 은 7개 클래스가 다 있는 유일한 청크입니다.
    (b) **주된 청크가 같은 쌍 vs 다른 쌍**의 혼동량 비교.
        청크를 쓰고 있다면 **주된 청크가 다른 쌍이 더 잘 갈려야** 합니다.
        ⚠️ "청크 집합이 분리된 쌍" 으로 재면 안 됩니다 — VL01 에 7개 클래스가
           다 있어서 분리된 쌍이 **0개**입니다 (첫 판에서 겪었습니다).
    (c) (a) 의 macro-F1 이 전체와 얼마나 다른가.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402

# ── 사전등록 판정 (결과를 보고 바꾸지 않습니다 — 작업 규칙 2) ────────────
#   ① 청크만으로 낸 macro-F1 이 모델의 이 비율을 넘으면 "설계가 많이 얽힘"
ENTANGLED_RATIO = 0.50
#   ② 한 청크 안에서 macro-F1 이 이만큼 넘게 떨어지면 "모델이 청크를 쓰고 있음"
#      (experiments.MACRO_F1_NOISE 와 같은 값)
WITHIN_CHUNK_DROP = 0.02
#   (b) 안 겹치는 쌍의 혼동이 겹치는 쌍보다 이 비율 아래로 낮으면 청크를 쓴 정황
CROSS_CHUNK_SUSPECT = 0.70


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """두 범주형 변수의 연관 강도 (0 = 무관, 1 = 완전 결정)."""
    ct = pd.crosstab(a, b).to_numpy(dtype=float)
    n = ct.sum()
    if n == 0:
        return float("nan")
    exp = np.outer(ct.sum(1), ct.sum(0)) / n
    chi2 = float(((ct - exp) ** 2 / np.where(exp == 0, 1, exp)).sum())
    r, k = ct.shape
    denom = n * (min(r, k) - 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else float("nan")


def chunk_only_f1(df: pd.DataFrame, fold: int = 0) -> tuple[float, pd.Series]:
    """청크**만** 보고 라벨을 맞힙니다 — 각 청크의 최빈 클래스로 찍기.

    이보다 잘하는 방법은 없습니다(청크가 유일한 입력이므로). 즉 **상한선**입니다.
    """
    from sklearn.metrics import f1_score

    dev = df[~df["is_holdout"]]
    tr, va = dev[dev["fold"] != fold], dev[dev["fold"] == fold]
    rule = tr.groupby("chunk")["label"].agg(lambda s: s.value_counts().idxmax())
    pred = va["chunk"].map(rule)
    return float(f1_score(va["label"], pred, average="macro", zero_division=0)), rule


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="manifest_final.parquet")
    ap.add_argument("--logits-dir", default="data/work/reports/step18_local",
                    help="local_logits.py 산출물 (한 청크 안 평가에 씁니다)")
    ap.add_argument("--model-macro-f1", type=float, default=0.5999,
                    help="비교 기준. 기본은 STEP 16 전체 val 2단계 macro-F1")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    df = pd.read_parquet(ROOT / a.manifest,
                         columns=["label", "chunk", "fold", "is_holdout"])
    out: dict = {}

    print("=" * 78)
    print(" ① 설계가 얼마나 얽혀 있나 (상한선)")
    print("=" * 78)
    ct = pd.crosstab(df["label"], df["chunk"], normalize="index").mul(100).round(1)
    print(ct.to_string())
    v = cramers_v(df["label"], df["chunk"])
    print(f"\n  Cramer's V(라벨, 청크) = {v:.3f}   (0 = 무관, 1 = 청크가 라벨을 완전 결정)")

    les = df[df["label"] != "A7"]
    f1_chunk, rule = chunk_only_f1(les)
    ratio = f1_chunk / a.model_macro_f1
    print(f"\n  청크만으로 병변 6종 macro-F1 = {f1_chunk:.4f}")
    print(f"    (각 청크의 최빈 클래스로 찍기 — 청크가 유일한 입력일 때의 최선)")
    print(f"    규칙: {rule.to_dict()}")
    print(f"  모델(2단계) macro-F1 {a.model_macro_f1:.4f} 대비 {ratio:.1%}")
    out.update(cramers_v=v, chunk_only_macro_f1=f1_chunk, ratio=ratio)
    print(f"  {'[!]' if ratio >= ENTANGLED_RATIO else '[O]'} "
          f"문턱 {ENTANGLED_RATIO:.0%} — "
          f"{'설계가 많이 얽혀 있습니다' if ratio >= ENTANGLED_RATIO else '상한선 자체는 낮습니다'}")

    print("\n" + "=" * 78)
    print(" ② 모델이 실제로 청크를 쓰나")
    print("=" * 78)
    d = ROOT / a.logits_dir
    if not (d / "stage2_logits.npz").exists():
        print(f"  (건너뜀) {d} 에 로짓이 없습니다 — tools/local_logits.py 를 먼저 돌리세요.")
    else:
        from sklearn.metrics import f1_score

        z = np.load(d / "stage2_logits.npz", allow_pickle=False)
        rows = pd.read_parquet(d / "stage2_rows.parquet")
        classes = [str(c) for c in z["classes"]]
        y, pred = z["y"], z["logits"].argmax(1)

        f1_in = float(f1_score(y, pred, average="macro", zero_division=0))
        drop = a.model_macro_f1 - f1_in
        print(f"\n (a) **한 청크(VL01) 안에서만** 평가 — 청크가 상수라 단서로 못 씀")
        print(f"     macro-F1 {f1_in:.4f}  vs 전체 val {a.model_macro_f1:.4f}  "
              f"(차이 {drop:+.4f})")
        print(f"     {'[!]' if drop > WITHIN_CHUNK_DROP else '[O]'} 문턱 {WITHIN_CHUNK_DROP} — "
              + ("떨어졌습니다. 청크를 쓰고 있을 수 있습니다"
                 if drop > WITHIN_CHUNK_DROP else
                 "잡음 안입니다. 청크 없이도 같은 실력입니다"))
        print("     ⚠️ VL01 은 A5 70장 · A6 126장뿐이라 macro-F1 이 흔들립니다."
              " 방향만 보세요.")
        out.update(within_chunk_macro_f1=f1_in, within_chunk_drop=drop)

        # (b) **주된** 청크가 같은 쌍 vs 다른 쌍
        # ⚠️ "청크 집합이 분리된 쌍" 으로 재면 안 됩니다 — VL01 에 7개 클래스가
        #    다 있어서 **어떤 쌍도 분리되지 않습니다** (첫 판에서 0개가 나왔습니다).
        #    실제로 갈리는 건 각 클래스가 **어디서 대부분 왔나** 입니다.
        full = pd.read_parquet(ROOT / a.manifest, columns=["label", "chunk"])
        dom = full.groupby("label")["chunk"].agg(lambda s: s.value_counts().idxmax())
        print(f"\n (b) 혼동량 — **주된 청크**가 같은 쌍 vs 다른 쌍")
        print(f"     주된 청크: {dom.to_dict()}")
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        np.add.at(cm, (y, pred), 1)
        share, disjoint = [], []
        for i, ci in enumerate(classes):
            for j, cj in enumerate(classes):
                if i == j:
                    continue
                rate = cm[i, j] / max(cm[i].sum(), 1)
                (share if dom[ci] == dom[cj] else disjoint).append((ci, cj, rate))
        ms = float(np.mean([r for *_, r in share])) if share else float("nan")
        md = float(np.mean([r for *_, r in disjoint])) if disjoint else float("nan")
        print(f"     같은 쌍 {len(share):>3}개  평균 혼동률 {ms:.4f}")
        print(f"     다른 쌍 {len(disjoint):>3}개  평균 혼동률 {md:.4f}")
        if disjoint and share:
            rel = md / ms if ms else float("nan")
            print(f"     비 {rel:.2f}   문턱 {CROSS_CHUNK_SUSPECT}")
            print("     청크를 쓰고 있다면 **주된 청크가 다른 쌍이 훨씬 덜 헷갈려야**"
                  f" 합니다 (비가 {CROSS_CHUNK_SUSPECT} 아래).")
            print(f"     {'[!] 청크를 쓴 정황' if rel < CROSS_CHUNK_SUSPECT else '[O] 그런 정황 없음'}")
            out["cross_over_shared"] = rel
            top = sorted(disjoint, key=lambda t: -t[2])[:3]
            print("     주된 청크가 다른데도 많이 헷갈리는 쌍: "
                  + ", ".join(f"{i}->{j} {r:.3f}" for i, j, r in top))

    print("\n" + "=" * 78)
    print(" 판정")
    print("=" * 78)
    ent = out.get("ratio", 0) >= ENTANGLED_RATIO
    used = (out.get("within_chunk_drop", 0) > WITHIN_CHUNK_DROP
            or out.get("cross_over_shared", 1.0) < CROSS_CHUNK_SUSPECT)
    if ent and used:
        v2 = "위험 — 설계도 얽혀 있고 모델도 쓰는 정황이 있습니다"
    elif ent:
        v2 = "주의 — 설계는 얽혀 있으나 모델이 쓰는 정황은 없습니다 (기록하고 진행)"
    elif used:
        v2 = "이상 — 상한선은 낮은데 청크 의존 정황이 있습니다. 다시 보세요"
    else:
        v2 = "문제 없음 — 설계 상한선도 낮고 의존 정황도 없습니다"
    out["verdict"] = v2
    print(f"  -> {v2}")
    print("\n⚠️ 이건 **관찰**입니다. 못 박으려면 '사진으로 청크를 맞히기' 를"
          " 학습시켜 봐야 합니다 (GPU 필요).")

    if a.out:
        p = ROOT / a.out
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float),
                     encoding="utf-8")
        print(f"\n저장: {p}")


if __name__ == "__main__":
    main()
