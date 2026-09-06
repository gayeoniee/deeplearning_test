"""★ STEP 28 — **무엇을** 말할 것인가 (알갱이 크기).

    uv run --extra train python tools/naming_granularity.py

STEP 27 은 *"6종 이름을 말할 것인가"* 를 물어 33.7% 를 얻고 "판단 보류" 로
끝났습니다. 그런데 물어야 했던 건 **"무엇을 말할 것인가"** 입니다. 말할 수
있는 것은 알갱이가 여럿입니다:

    6종 이름     "농포·여드름으로 보입니다"
    형태 계열    "융기·발진 계열로 보입니다"        A1·A4 / A2·A3 / A5·A6
    긴급도       "조기 진료를 권합니다"              관찰 / 진료 권장 / 조기 진료
    두 이름      "구진 또는 농포로 보입니다"
    A6 이진      "덩어리가 의심됩니다"

⚠️ **묶음은 데이터를 보고 만들지 않았습니다.** `config.URGENCY_TIER` 와
   `config.MORPH_GROUP` 은 `docs/data/병변_6종_임상_해설.md` 의 요약표
   "긴급도" 열과 **"🟢 비교적 안전한 혼동"** 목록을 옮긴 것이고, 둘 다 이
   분석보다 먼저 있었습니다. 혼동행렬을 보고 묶으면 무슨 묶음이든 좋아집니다.

★ 그리고 **오답의 정의**도 다시 봅니다. 임상 해설이 꼽은 위험한 혼동
   (A6→A2 · A5→A1 · A6→A1)의 공통점은 *"급한 걸 안 급하다고 말했다"* 입니다.
   반대 방향은 병원에 가게 만들 뿐이라 안전합니다 — 1단계 헛알림도 그렇습니다.
   그래서 **긴급도 하향(under-triage)** 을 따로 셉니다.

⚠️ 판정 기준은 `experiments.granularity_report()` 에 미리 박혀 있습니다.
⚠️ 입력은 `tools/naming_coverage.py` 가 남긴 배열입니다 (추론 다시 안 함).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import experiments  # noqa: E402
from src.config import (CLASSES, MORPH_GROUP, MORPH_GROUP_KEEP_A6,  # noqa: E402
                        URGENCY_TIER, URGENCY_TIER_NAME)
from src.stages import NORMAL_LABEL  # noqa: E402


def _grouped(p: np.ndarray, mapping: dict) -> tuple[np.ndarray, list]:
    """클래스 확률을 묶음 확률로 **더합니다** (argmax 를 묶는 게 아니라)."""
    keys = list(dict.fromkeys(mapping[c] for c in CLASSES))
    g = np.stack([p[:, [i for i, c in enumerate(CLASSES) if mapping[c] == k]].sum(1)
                  for k in keys], 1)
    return g, keys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arrays", default="data/work/reports/step27_arrays.npz")
    ap.add_argument("--out", default="data/work/reports/step28_granularity.json")
    a = ap.parse_args()

    f = ROOT / a.arrays
    if not f.exists():
        raise SystemExit(f"[X] {f} 가 없습니다 — 먼저 tools/naming_coverage.py 를 도세요")
    z = np.load(f, allow_pickle=True)
    p1 = z["p1"]
    truth = z["truth"]
    names = list(z["arm_names"])
    P2 = [z[f"p2_{i}"] for i in range(len(names))]
    ens = np.mean(P2, 0)
    is_norm = truth == NORMAL_LABEL
    n = len(truth)
    print(f"1단계가 넘긴 사진 {n:,}장 · 헛알림 {int(is_norm.sum()):,}장 "
          f"({is_norm.mean():.1%}) · 팔 {len(names)}개\n")

    # 실제 긴급도 등급 (정상 사진은 -1 — 낮출 긴급도가 없습니다)
    tier_true = np.array([URGENCY_TIER.get(t, -1) for t in truth])

    res = {}
    print("=" * 78)
    print(f"  {'알갱이':22} {'커버리지':>8}   {'긴급도 하향':>10}   판정   "
          f"(오답률 {experiments.NAMING_TARGET_ERROR:.0%} 목표)")
    print("=" * 78)

    # ① 6종 이름 (STEP 27 재확인)
    said = np.array(CLASSES, dtype=object)[ens.argmax(1)]
    res["6종 이름"] = experiments.granularity_report("① 6종 이름", {
        "conf": p1 * ens.max(1), "wrong": is_norm | (said != truth),
        "tier_true": tier_true,
        "tier_said": [URGENCY_TIER[s] for s in said]})

    # ② 형태 계열 3군 · ③ 긴급도 3등급
    for tag, mapping, label in (("형태 계열", MORPH_GROUP, "② 형태 계열 3군"),
                                ("긴급도", URGENCY_TIER, "③ 긴급도 3등급")):
        g, keys = _grouped(ens, mapping)
        gi = g.argmax(1)
        said_g = np.array(keys, dtype=object)[gi]
        true_g = np.array([mapping.get(t, "정상") for t in truth], dtype=object)
        # 말한 것의 긴급도: 형태 계열이면 그 묶음 안 최대 등급(보수적으로)
        if tag == "긴급도":
            tier_said = gi                      # keys 가 0/1/2 그대로
        else:
            tier_said = np.array([max(URGENCY_TIER[c] for c in CLASSES
                                      if mapping[c] == k) for k in keys])[gi]
        res[tag] = experiments.granularity_report(label, {
            "conf": p1 * g.max(1), "wrong": is_norm | (said_g != true_g),
            "tier_true": tier_true, "tier_said": tier_said})

    # ④ 두 이름 (top-2)
    o2 = np.argsort(-ens, 1)[:, :2]
    top2 = np.array(CLASSES, dtype=object)[o2]
    hit2 = (top2 == truth[:, None]).any(1)
    # 두 이름의 긴급도는 **높은 쪽**을 말한 것으로 봅니다 (보수적)
    tier2 = np.array([[URGENCY_TIER[c] for c in row] for row in top2]).max(1)
    res["두 이름"] = experiments.granularity_report("④ 두 이름 (top-2)", {
        "conf": p1 * np.take_along_axis(ens, o2, 1).sum(1),
        "wrong": is_norm | ~hit2, "tier_true": tier_true, "tier_said": tier2})

    # ⑤ 긴급도 **2등급** — "지켜봐도 됩니다" 를 아예 안 만들면?
    #    임상 해설이 꼽은 위험은 전부 "급한 걸 안 급하다고" 입니다. 그러면
    #    등급을 둘로 줄여 (관찰+진료권장) vs (조기진료) 로만 말하면 어떤가.
    hi = np.array([URGENCY_TIER[c] == 2 for c in CLASSES])
    p_hi = ens[:, hi].sum(1)
    said_hi = p_hi >= 0.5
    true_hi = np.array([URGENCY_TIER.get(t, -1) == 2 for t in truth])
    res["긴급도 2등급"] = experiments.granularity_report("⑤ 긴급도 2등급", {
        "conf": p1 * np.maximum(p_hi, 1 - p_hi),
        "wrong": is_norm | (said_hi != true_hi),
        "tier_true": tier_true, "tier_said": np.where(said_hi, 2, 0)})

    # ── ⑥ A6 이진 경보는 커버리지가 아니라 정밀도·재현율로 봅니다 ──
    # A6 은 넘긴 사진의 4% 뿐이라 "커버리지" 로 재면 최대치가 4% 입니다.
    # 물어야 할 것은 **"실제 덩어리의 몇 %를 잡으면서 몇 %가 헛말인가"** 입니다.
    print("\n" + "=" * 78)
    print("⑥ A6('덩어리가 의심됩니다') 경보 — 커버리지가 아니라 정밀도·재현율")
    print("=" * 78)
    ia6 = CLASSES.index("A6")
    score = p1 * ens[:, ia6]
    real = truth == "A6"
    print(f"  넘긴 사진 {n:,}장 중 실제 A6 {int(real.sum()):,}장 ({real.mean():.1%})")
    print(f"  {'정밀도 목표':>10}{'재현율':>10}{'경보 수':>10}{'문턱':>10}")
    a6_rows = []
    for prec in (0.9, 0.8, 0.7, 0.6, 0.5):
        o = np.argsort(-score)
        cp = np.cumsum(real[o]) / np.arange(1, n + 1)
        idx = np.flatnonzero(cp >= prec)
        if len(idx) == 0:
            a6_rows.append({"precision": prec, "recall": 0.0, "n": 0})
            print(f"  {prec:>10.0%}{0.0:>10.1%}{0:>10}{'—':>10}")
            continue
        k = int(idx[-1]) + 1
        rec = float(real[o][:k].sum() / max(real.sum(), 1))
        a6_rows.append({"precision": prec, "recall": rec, "n": k,
                        "threshold": float(score[o][k - 1])})
        print(f"  {prec:>10.0%}{rec:>10.1%}{k:>10,}{score[o][k - 1]:>10.3f}")
    res["A6_경보"] = a6_rows
    print("  ⚠️ 임상 해설: 'A6 으로 오탐하는 건 상대적으로 안전합니다"
          " (병원에 가서 확인하면 되니까)'")

    # ── ⑦ 오답의 정의를 바꾸면 — 위험한 혼동만 세기 ─────────────────
    print("\n" + "=" * 78)
    print("⑦ 6종 이름을 그대로 말하되, **긴급도를 낮춰 말한 것만** 오답으로")
    print("=" * 78)
    tier_said6 = np.array([URGENCY_TIER[s] for s in said])
    under = (tier_true >= 0) & (tier_true > tier_said6)
    print(f"  전부 말했을 때  이름 오답률 {float((is_norm | (said != truth)).mean()):.1%}"
          f"  vs  **긴급도 하향 {float(under[~is_norm].mean()):.1%}**")
    for tgt in (0.10, 0.05, 0.02):
        o = np.argsort(-(p1 * ens.max(1)))
        e = np.cumsum(under[o]) / np.arange(1, n + 1)
        k = np.flatnonzero(e <= tgt)
        cov = float((k[-1] + 1) / n) if len(k) else 0.0
        print(f"  긴급도 하향 {tgt:.0%} 목표 → 커버리지 {cov:.1%}")
        res.setdefault("위험오류만", {})[f"under{int(tgt * 100)}"] = cov

    # ── ⑧ 이 알갱이에서도 앙상블이 필요한가 (추론 3배 값) ──────────
    # 굵게 말하면 쉬워지니, 릴리스 단독으로 충분해질 수도 있습니다.
    print("\n" + "=" * 78)
    print("⑧ 알갱이별로 — 릴리스 단독 vs 3팔 앙상블 (커버리지)")
    print("=" * 78)
    print(f"  {'알갱이':22}{'릴리스 단독':>12}{'3팔 앙상블':>12}{'차이':>10}"
          f"{'말한 것 중 헛알림':>16}")
    cmp_rows = {}
    for label, mapping, kind in (("6종 이름", None, "cls"),
                                 ("형태 계열 3군", MORPH_GROUP, "grp"),
                                 ("긴급도 3등급", URGENCY_TIER, "grp"),
                                 ("두 이름 (top-2)", None, "top2")):
        row = {}
        for who, p in (("release", P2[0]), ("ens", ens)):
            if kind == "cls":
                sd = np.array(CLASSES, dtype=object)[p.argmax(1)]
                w, c = is_norm | (sd != truth), p1 * p.max(1)
            elif kind == "grp":
                g, keys = _grouped(p, mapping)
                sd = np.array(keys, dtype=object)[g.argmax(1)]
                tg = np.array([mapping.get(t, "정상") for t in truth], dtype=object)
                w, c = is_norm | (sd != tg), p1 * g.max(1)
            else:
                oo = np.argsort(-p, 1)[:, :2]
                t2 = np.array(CLASSES, dtype=object)[oo]
                w = is_norm | ~(t2 == truth[:, None]).any(1)
                c = p1 * np.take_along_axis(p, oo, 1).sum(1)
            o = np.argsort(-c)
            e = np.cumsum(w[o]) / np.arange(1, n + 1)
            k = np.flatnonzero(e <= experiments.NAMING_TARGET_ERROR)
            kk = int(k[-1]) + 1 if len(k) else 0
            row[who] = kk / n
            if who == "ens":
                row["fa_share"] = float(is_norm[o][:kk].mean()) if kk else 0.0
        cmp_rows[label] = row
        print(f"  {label:22}{row['release']:>12.1%}{row['ens']:>12.1%}"
              f"{row['ens'] - row['release']:>+10.1%}{row['fa_share']:>16.1%}")
    print("  ← '말한 것 중 헛알림' 은 앙상블 기준. 거절이 헛알림을 얼마나 걸러냈나.")
    res["알갱이별_앙상블_비교"] = cmp_rows

    # ── ⑨ ★ 임상 때문인가, 통계적 편의 때문인가 ────────────────────
    # 임상 해설이 "🟢 안전한 혼동" 으로 **명시한 것은 두 쌍뿐**입니다
    # (A1↔A4 · A5↔A6). 3군의 셋째 묶음 A2+A3 은 문서가 인정한 게 아니라
    # **남은 것**입니다. 그러면 이득 중 얼마가 어디서 오나? 갈라 봅니다.
    # ⚠️ 이걸 안 갈라 보면 "임상적으로 비슷해서 묶었다" 고 말하게 됩니다.
    print("\n" + "=" * 78)
    print("⑨ 묶음이 임상 때문인가 — 문서가 인정한 병합만 하면?")
    print("=" * 78)
    SCHEMES = {
        "6종 (병합 없음)": {c: c for c in CLASSES},
        "문서가 인정한 병합만": {"A1": "융기·발진", "A4": "융기·발진",
                         "A5": "손상·덩어리", "A6": "손상·덩어리",
                         "A2": "비듬·각질", "A3": "태선화·색소침착"},
        "A1·A4 만 병합": {"A1": "융기·발진", "A4": "융기·발진", "A2": "비듬·각질",
                      "A3": "태선화·색소침착", "A5": "미란·궤양", "A6": "결절·종괴"},
        "★ A6 만 따로 (4군)": MORPH_GROUP_KEEP_A6,
        "형태 계열 3군": MORPH_GROUP,
        # ★ 수의피부과 **교과서 축** (config.LESION_ORIGIN). 우리가 만든 게
        #   아니라 표준이라 반드시 재봐야 합니다 — 그런데 **과잉에서 걸립니다.**
        "교과서: primary/secondary (2군)": {
            "A1": "primary", "A4": "primary", "A6": "primary",
            "A2": "secondary", "A3": "secondary", "A5": "secondary"},
        "교과서 + 1cm 경계 (3군)": {
            "A1": "작은 융기(≤1cm)", "A4": "작은 융기(≤1cm)", "A6": "덩어리(>1cm)",
            "A2": "표면·이차 변화", "A3": "표면·이차 변화", "A5": "표면·이차 변화"},
    }
    sch = {}
    for nm, mp in SCHEMES.items():
        keys = list(dict.fromkeys(mp[c] for c in CLASSES))
        g, _ = _grouped(ens, mp)
        gi = g.argmax(1)
        said_g = np.array(keys, dtype=object)[gi]
        true_g = np.array([mp.get(t, "정상") for t in truth], dtype=object)
        tier_said = np.array([max(URGENCY_TIER[c] for c in CLASSES if mp[c] == k)
                              for k in keys])[gi]
        sch[nm] = experiments.granularity_report(f"{nm} [{len(keys)}군]", {
            "conf": p1 * g.max(1), "wrong": is_norm | (said_g != true_g),
            "tier_true": tier_true, "tier_said": tier_said})
    res["묶음_변형"] = sch
    print("  → 문서가 인정한 병합만으로는 문턱을 못 넘습니다. 이득의 대부분이")
    print("    **A2+A3** 에서 오는데 그건 문서가 인정한 묶음이 아닙니다.")
    print("  → 'A6 만 따로' 는 3군보다 조금밖에 안 잃습니다 — A6 은 종양 감별이")
    print("    필요한 유일한 클래스라, 이름을 지키는 값이 그보다 큽니다.")
    print("  🚫 **교과서 축(primary/secondary)은 커버리지가 제일 높은데 기각**입니다 —")
    print("     primary 에 A6(조기 진료)이 들어 있어 구진 하나에도 '조기 진료' 가")
    print("     붙습니다. **과잉 열을 안 봤으면 이걸 골랐을 겁니다** (STEP 30 의 관문).")

    # ── ⑩ 앙상블이 알갱이마다 얼마나 값을 하나 (추론 3배) ───────────
    print("\n" + "=" * 78)
    print("⑩ 릴리스 단독으로도 되나 — 추론 3배를 치를지의 근거")
    print("=" * 78)
    print(f"  {'묶음':30}{'릴리스 단독':>12}{'3팔 앙상블':>12}{'앙상블 값어치':>14}")
    rel = P2[0]
    cmp2 = {}
    for nm, mp in SCHEMES.items():
        row = []
        for p in (rel, ens):
            keys = list(dict.fromkeys(mp[c] for c in CLASSES))
            g, _ = _grouped(p, mp)
            gi = g.argmax(1)
            sd = np.array(keys, dtype=object)[gi]
            tr = np.array([mp.get(t, "정상") for t in truth], dtype=object)
            w = is_norm | (sd != tr)
            o = np.argsort(-(p1 * g.max(1)))
            e = np.cumsum(w[o]) / np.arange(1, n + 1)
            k = np.flatnonzero(e <= experiments.NAMING_TARGET_ERROR)
            row.append(float((k[-1] + 1) / n) if len(k) else 0.0)
        cmp2[nm] = {"release": row[0], "ensemble": row[1], "gain": row[1] - row[0]}
        print(f"  {nm:30}{row[0]:>12.1%}{row[1]:>12.1%}{row[1]-row[0]:>+14.1%}")
    res["앙상블_값어치"] = cmp2
    print("  → 굵게 말할수록 앙상블의 몫이 줄어듭니다. 6종 이름일 때만 컸습니다.")

    out = ROOT / a.out
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float),
                   encoding="utf-8")
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
