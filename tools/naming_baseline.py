"""★ STEP 29 — 하한선 대비, 그리고 A6 경보를 **전체 데이터**로.

    uv run --extra train python tools/naming_baseline.py

STEP 28 에서 제가 "계열 정확도 0.8365" 를 앞세웠는데 **하한선을 안 적었습니다.**
굵게 묶으면 *아무것도 안 하는 하한선*도 같이 올라갑니다 — 3군의 최빈 묶음이
전체의 67% 입니다. 작업 규칙 1: **하한선과 비교하지 않은 정확도는 숫자가
아닙니다.**

같이 재는 것:
  ② A6('덩어리가 의심됩니다') 경보를 전체 val 로 — VL01 표본 124장이 너무
     얇았습니다 (전체는 1,540장).
  ③ 4군 안에서 결절·종괴가 얼마나 지켜지나 (거절 없이 전부 말했을 때).

⚠️ 전체 val 로짓에는 **1단계 헛알림이 없습니다** (TL01·TL02 의 1단계 점수가
   로컬에 없음). 그래서 VL01 에서 '헛알림 포함' 과 '병변만' 을 둘 다 재서
   그 차이가 얼마나 되는지 **측정한 뒤** 전체 숫자를 읽습니다 — 경고를
   달아두는 것으로 끝내지 않습니다 (STEP 27 에서 그렇게 데었습니다).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (CLASSES, MORPH_GROUP, MORPH_GROUP_KEEP_A6,  # noqa: E402
                        URGENCY_TIER)
from src.stages import NORMAL_LABEL  # noqa: E402

#: 전체 val 로짓 (병변만). STEP 23·25 가 남긴 것.
FULL_LOGITS = {
    "릴리스 m2.5": "data/work/reports/step18_full/stage2_logits.npz",
    "eff m2.5": ".orca/drops/logits_val.npz",
    "eff f320": ".orca/drops/logits_val copy.npz",
}

SCHEMES = {
    "6종": {c: c for c in CLASSES},
    "문서가 인정한 병합만 (4군)": {"A1": "융기·발진", "A4": "융기·발진",
                          "A5": "손상·덩어리", "A6": "손상·덩어리",
                          "A2": "비듬·각질", "A3": "태선화·색소침착"},
    "★ A6 만 따로 (4군)": MORPH_GROUP_KEEP_A6,
    "형태 계열 3군": MORPH_GROUP,
    "긴급도 3등급": {k: str(v) for k, v in URGENCY_TIER.items()},
}


def _softmax(a):
    e = np.exp(a - a.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def _group(p, mapping):
    keys = list(dict.fromkeys(mapping[c] for c in CLASSES))
    g = np.stack([p[:, [i for i, c in enumerate(CLASSES) if mapping[c] == k]].sum(1)
                  for k in keys], 1)
    return g, keys


def _pr(score, real, targets=(0.9, 0.8, 0.7, 0.6, 0.5)):
    o = np.argsort(-score)
    cp = np.cumsum(real[o]) / np.arange(1, len(o) + 1)
    rows = []
    for t in targets:
        i = np.flatnonzero(cp >= t)
        if len(i) == 0:
            rows.append({"precision": t, "recall": None, "n": 0})
            continue
        k = int(i[-1]) + 1
        rows.append({"precision": t, "recall": float(real[o][:k].sum() / real.sum()),
                     "n": k, "threshold": float(score[o][k - 1])})
    return rows


def main() -> int:
    from sklearn.metrics import accuracy_score, f1_score

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/work/reports/step29_baseline_a6.json")
    a = ap.parse_args()

    P, y = [], None
    for nm, f in FULL_LOGITS.items():
        p = ROOT / f
        if not p.exists():
            raise SystemExit(f"[X] {nm}: {p} 가 없습니다 — STEP 23·25 산출물이 필요합니다")
        z = np.load(p, allow_pickle=False)
        if y is None:
            y = z["y"]
        elif not np.array_equal(y, z["y"]):
            raise SystemExit(f"[X] {nm}: 라벨 순서가 다릅니다 — 짝짓기가 망가집니다")
        P.append(_softmax(z["logits"].astype(np.float64)))
    ens, rel = np.mean(P, 0), P[0]
    truth = np.array(CLASSES, dtype=object)[y]
    res: dict = {"n_val": int(len(y))}
    print(f"전체 val {len(y):,}행 (병변만 — 1단계 헛알림 없음)\n")

    # ── ① 하한선 대비 ────────────────────────────────────────────
    print("=" * 88)
    print("① 굵게 묶으면 **하한선도 같이 오릅니다** (작업 규칙 1)")
    print("=" * 88)
    print(f"  {'묶음':26}{'하한선':>9}{'모델':>9}{'하한선 대비':>12}"
          f"{'macro-F1':>10}{'하한선 F1':>11}")
    base = {}
    for name, mp in SCHEMES.items():
        g, keys = _group(ens, mp)
        pred = np.array(keys, dtype=object)[g.argmax(1)]
        tr = np.array([mp[t] for t in truth], dtype=object)
        u, c = np.unique(tr, return_counts=True)
        share = float(c.max() / len(tr))
        dumb = np.full(len(tr), u[c.argmax()], dtype=object)
        acc = float(accuracy_score(tr, pred))
        base[name] = {
            "majority_share": share, "accuracy": acc, "over_baseline": acc - share,
            "macro_f1": float(f1_score(tr, pred, average="macro", zero_division=0)),
            "baseline_macro_f1": float(f1_score(tr, dumb, average="macro",
                                                zero_division=0)),
            "majority_group": str(u[c.argmax()])}
        b = base[name]
        print(f"  {name:26}{share:>9.1%}{acc:>9.4f}{b['over_baseline']:>+12.1%}"
              f"{b['macro_f1']:>10.4f}{b['baseline_macro_f1']:>11.4f}")
    res["baseline"] = base
    print("\n  → 0.7037 → 0.8365 는 모델이 좋아진 게 아니라 **문제가 쉬워진 것**이고,")
    print("    쉬워진 만큼 하한선도 따라옵니다. 변별 일은 6종에서 더 큽니다.")
    print("  ⚠️ 단 **커버리지는 이 비판에서 자유롭습니다** — 늘 최빈 묶음만 말하면")
    print("    오답률이 어디서나 일정해서 20% 목표를 한 번도 못 지킵니다(커버리지 0%).")

    # ── ② A6 경보 ────────────────────────────────────────────────
    ia6 = CLASSES.index("A6")
    real = truth == "A6"
    print("\n" + "=" * 88)
    print(f"② A6('덩어리가 의심됩니다') 경보 — 전체 데이터 A6 {int(real.sum()):,}장")
    print("=" * 88)
    print(f"  {'정밀도 목표':>12}{'재현율(앙상블)':>16}{'재현율(릴리스)':>16}{'경보 수':>10}")
    a6 = {"ensemble": _pr(ens[:, ia6], real), "release": _pr(rel[:, ia6], real)}
    for e, r in zip(a6["ensemble"], a6["release"]):
        rr = f"{e['recall']:.1%}" if e["recall"] is not None else "—"
        r2 = f"{r['recall']:.1%}" if r["recall"] is not None else "—"
        print(f"  {e['precision']:>12.0%}{rr:>16}{r2:>16}{e['n']:>10,}")
    res["a6_alarm_full"] = a6

    # 헛알림을 넣으면 얼마나 나빠지나 — VL01 에서 **재서** 확인합니다
    arr = ROOT / "data/work/reports/step27_arrays.npz"
    if arr.exists():
        z = np.load(arr, allow_pickle=True)
        p1v, tv = z["p1"], z["truth"]
        ev = np.mean([z[f"p2_{i}"] for i in range(len(z["arm_names"]))], 0)
        les = tv != NORMAL_LABEL
        sc = p1v * ev[:, ia6]
        both = _pr(sc, tv == "A6")
        only = _pr(sc[les], tv[les] == "A6")
        print("\n  헛알림을 넣으면 나빠지나 (VL01 에서 실측)")
        print(f"  {'정밀도 목표':>12}{'헛알림 포함':>14}{'병변만':>12}")
        for x, o in zip(both, only):
            f1 = f"{x['recall']:.1%}" if x["recall"] is not None else "—"
            f2 = f"{o['recall']:.1%}" if o["recall"] is not None else "—"
            print(f"  {x['precision']:>12.0%}{f1:>14}{f2:>12}")
        print("  → 거의 같습니다. 멀쩡한 피부를 덩어리로 보지는 않습니다 —")
        print("    그래서 위 전체 데이터 표를 그대로 읽어도 됩니다.")
        res["a6_alarm_vl01"] = {"with_false_alarms": both, "lesions_only": only}
    else:
        print("\n  ⚠️ step27_arrays.npz 가 없어 헛알림 대조를 못 했습니다")

    # ── ③ 4군 안에서 결절·종괴가 지켜지나 ───────────────────────────
    print("\n" + "=" * 88)
    print("③ 4군(A6 따로) — 묶음별 recall 과 오답 행선지 (거절 없이 전부 말했을 때)")
    print("=" * 88)
    mp = MORPH_GROUP_KEEP_A6
    g, keys = _group(ens, mp)
    pred = np.array(keys, dtype=object)[g.argmax(1)]
    tg = np.array([mp[t] for t in truth], dtype=object)
    print(f"  {'실제':14}{'장수':>8}{'recall':>9}   가장 많이 가는 곳")
    per = {}
    for k in keys:
        m = tg == k
        w = pred[m][pred[m] != k]
        top = ""
        if len(w):
            u, c = np.unique(w, return_counts=True)
            top = f"{u[c.argmax()]} {c.max() / m.sum():.1%}"
        per[k] = {"n": int(m.sum()), "recall": float((pred[m] == k).mean()),
                  "top_confusion": top}
        print(f"  {k:14}{int(m.sum()):>8,}{per[k]['recall']:>9.3f}   {top}")
    tier_of = {k: max(URGENCY_TIER[c] for c in CLASSES if mp[c] == k) for k in keys}
    said_tier = np.array([tier_of[p] for p in pred])
    m6 = tg == MORPH_GROUP_KEEP_A6["A6"]
    under = float((said_tier[m6] < 2).mean())
    to_watch = float((said_tier[m6] == 0).mean())
    print(f"\n  ★ 실제 결절·종괴 {int(m6.sum()):,}장 중 긴급도를 낮춰 부른 비율 "
          f"**{under:.1%}**")
    print(f"     그중 '관찰' 등급으로 부른 것 **{to_watch:.1%}** — "
          "어떤 경로로 틀려도 병원에 가라는 말은 나갑니다")
    print("  ⚠️ 이 값은 **거절 없이 전부 말했을 때**입니다. STEP 28 의 '하향 1.7%' 는")
    print("    **말한 것 중**(거절 뒤) 값이라 다른 숫자입니다 — 섞지 마세요.")
    res["group4_full"] = {"per_group": per, "a6_under_triage_no_abstain": under,
                          "a6_called_watch": to_watch}

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float),
                   encoding="utf-8")
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
