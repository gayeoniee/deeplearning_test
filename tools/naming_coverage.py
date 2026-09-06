"""★ "확신 있을 때만 병변 이름을 말할 것인가" — **정직한** 커버리지-오답 곡선.

    uv run --extra train python tools/naming_coverage.py

왜 다시 재나
------------
STEP 11 의 *"오답률 20% 를 맞추려면 커버리지 18.2%"* 는 **2단계 val(병변만)**
에서 잰 값이고, 그 문서에도 *"그 곡선은 헛알림 1,233건을 뺀 것이라 실제는 더
나쁩니다"* 라고 적혀 있습니다. 그런데 그 뒤로 아무도 **더 나쁜 쪽**을 재지
않았고, 우리는 그 숫자로 "이름을 화면에서 뺀다" 를 결정했습니다.

실제 화면에 뜨는 사진은 **1단계가 "이상" 으로 넘긴 것 전부**입니다. 거기엔
멀쩡한 개(헛알림)가 섞여 있고, 그 사진에 병변 이름이 붙으면 **무조건 오답**
입니다. 이 도구는 그걸 포함해서 다시 잽니다.

    분모      1단계가 넘긴 사진 전부 (헛알림 포함)
    오답      정상인데 이름을 붙였다  OR  병변인데 다른 이름을 붙였다
    확신도    p(이상) × p(그 병변)   ← `infer.Engine` 이 실제로 쓰는 값 그대로
    커버리지  그중 실제로 이름을 말한 비율

⚠️ **판정 기준은 `experiments.naming_report()` 에 미리 박혀 있습니다.**

⚠️ 이 도구는 **val** 만 씁니다 (holdout 은 06 만 엽니다 — 작업 규칙).
   그리고 로컬에 크롭이 있는 **VL01** 만 됩니다. VL01 은 커버리지-위험에서
   **비관적**인 청크입니다 (STEP 25) — 여기서 통과하면 전체에서도 통과할
   가능성이 큽니다. 반대로 여기서 떨어지면 전체로 다시 재야 합니다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import crop, env, experiments, labels, models, split, stages, train  # noqa: E402
from src.config import CFG, CLASSES  # noqa: E402

#: 팔 — (이름, 실험 폴더, 크롭 태그). 첫 번째가 지금 서빙하는 릴리스입니다.
ARMS = [
    ("m2.5(cnx릴리스)", "stage2_convnextv2_base_m2.5_384_n121k_moderate", "m2.5"),
    ("m2.5(eff)", "stage2_effnetv2_s_m2.5_384_moderate", "m2.5"),
    ("f320(eff)", "stage2_effnetv2_s_f320_384_moderate", "f320"),
]
STAGE1_EXP = "stage1_effnetv2_s_f320_384_n233k_moderate_photometric"


def _softmax(a: np.ndarray) -> np.ndarray:
    e = np.exp(a - a.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def _temperature(exp: str) -> float:
    f = train.ckpt_dir(exp) / "temperature.json"
    if not f.exists():
        print(f"  ⚠️ {exp}/temperature.json 이 없습니다 — T=1.0 으로 갑니다")
        return 1.0
    return float(json.loads(f.read_text(encoding="utf-8"))["temperature"])


def _infer(exp: str, frame, dev: str, batch: int) -> np.ndarray:
    """이 실험의 모델로 `frame` 전체를 추론해 **로짓**을 돌려줍니다."""
    import torch

    from src import data

    m = models.build(train.model_key_from_exp(exp), n_classes=len(CLASSES),
                     pretrained=False)
    sd = torch.load(train.ckpt_dir(exp) / "best.pt", map_location="cpu",
                    weights_only=False)
    m.load_state_dict(sd.get("model", sd.get("state_dict", sd)), strict=False)
    m = m.to(dev).eval()

    cfg = CFG()
    cfg.img_size = 384
    cfg.batch_size = batch
    # ⚠️ 반드시 `data.eval_loader` — 행 순서가 보존돼야 1·2단계를 짝지을 수
    #    있습니다 (직접 만든 로더로 갈라졌던 적이 있습니다).
    dl, ds = data.eval_loader(frame, cfg, model=m, classes=CLASSES)
    if len(ds) != len(frame):
        raise SystemExit(f"[X] {exp}: {len(frame) - len(ds):,}행이 빠졌습니다 "
                         "— 행이 어긋나면 짝짓기가 조용히 망가집니다")
    out = []
    with torch.no_grad():
        for x, _ in dl:
            out.append(m(x.to(dev)).float().cpu())
    del m
    torch.cuda.empty_cache()
    return torch.cat(out).numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunk", default="chunk_VL01")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="data/work/reports/step27_naming_coverage.json")
    a = ap.parse_args()

    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"장치: {dev}\n")

    # ── 1단계 val 행 (정상 + 병변 전부) ──────────────────────────────
    df = labels.load(ROOT / "manifest_final.parquet")
    df = df[df["chunk"] == a.chunk].reset_index(drop=True)
    s1 = stages.to_stage1(crop.switch_tag(df, "f320", verbose=False), verbose=False)
    _, va = split.get_fold(s1, 0)
    va = va.reset_index(drop=True)

    z = np.load(train.ckpt_dir(STAGE1_EXP) / f"logits_{a.chunk}_val.npz",
                allow_pickle=False)
    if len(z["y"]) != len(va):
        raise SystemExit(f"[X] 1단계 로짓 {len(z['y']):,}행 vs val {len(va):,}행 — "
                         "같은 분할이 아닙니다")
    y1_saved = z["y"]
    y1 = (va["label"] != stages.NORMAL_LABEL).to_numpy().astype(int)
    if not np.array_equal(y1_saved, y1):
        raise SystemExit("[X] 1단계 라벨 순서가 저장된 로짓과 다릅니다")

    T1 = _temperature(STAGE1_EXP)
    p1 = _softmax(z["logits"].astype(np.float64) / T1)[:, 1]
    thr = float(json.loads((env.work_root() / "stage1_threshold.json")
                           .read_text(encoding="utf-8"))["threshold"])
    flag = p1 >= thr
    n_norm_flag = int((flag & (y1 == 0)).sum())
    print(f"[1단계] val {len(va):,}장 (정상 {int((y1 == 0).sum()):,} / "
          f"병변 {int(y1.sum()):,}) · T={T1:.4f} · 임계값 {thr:.4f}")
    print(f"        넘긴 사진 {int(flag.sum()):,}장 — 그중 **헛알림 {n_norm_flag:,}장** "
          f"({n_norm_flag / max(int(flag.sum()), 1):.1%})")
    print(f"        recall {float(flag[y1 == 1].mean()):.3f} · "
          f"헛알림률 {float(flag[y1 == 0].mean()):.1%}\n")

    # ── 2단계: **넘긴 사진 전부**에 이름을 붙여봅니다 (헛알림 포함) ──
    sub = va[flag].reset_index(drop=True)
    P2 = {}
    for name, exp, tag in ARMS:
        frame = crop.switch_tag(sub, tag, verbose=False)
        miss = int((~frame["crop_path"].map(lambda p: Path(p).exists())).sum())
        if miss:
            raise SystemExit(f"[X] {name}: 크롭 {miss:,}장이 없습니다 (태그 {tag})")
        print(f"[2단계] {name} 추론 {len(frame):,}장 …", flush=True)
        # ⚠️ 서빙(`infer.Engine`)은 softmax(logit / T) 를 돌려줍니다. 여기서
        #    T 를 안 걸면 **서빙과 다른 확률**로 판단하게 됩니다. 온도가 없는
        #    실험은 T=1.0 (릴리스만 보정돼 있습니다).
        T2 = _temperature(exp) if (train.ckpt_dir(exp) / "temperature.json").exists() else 1.0
        if T2 != 1.0:
            print(f"         온도 보정 T={T2:.4f} 적용 (릴리스에만 있습니다)")
        P2[name] = _softmax(_infer(exp, frame, dev, a.batch).astype(np.float64) / T2)

    truth = sub["label_orig"].to_numpy()
    is_norm = truth == stages.NORMAL_LABEL
    a6 = CLASSES.index("A6")
    p1s = p1[flag]

    results: dict = {}
    for label, probs in [(ARMS[0][0] + " 단독", P2[ARMS[0][0]]),
                         ("3팔 앙상블", np.mean(list(P2.values()), 0))]:
        said = probs.argmax(1)
        said_name = np.array(CLASSES, dtype=object)[said]
        wrong = is_norm | (said_name != truth)
        print(f"\n{'=' * 62}\n{label}\n{'=' * 62}")
        print(f"  이름을 다 말하면 오답률 {wrong.mean():.1%} "
              f"(헛알림 {is_norm.mean():.1%} + 이름 틀림 "
              f"{float((~is_norm & (said_name != truth)).mean()):.1%})")
        results[label] = experiments.naming_report(
            {"conf": p1s * probs.max(1), "wrong": wrong,
             "is_a6": truth == "A6", "said_a6": said == a6})

    # ── 릴리스 단독이 왜 못 쓰나 — 확신도 10분위별 오답률 ────────────
    # "정확도가 낮아서" 가 아니라 **자기가 언제 맞는지 몰라서**라는 주장을
    # 그냥 하지 말고 보입니다. 확신도 상위 10% 에서 몇 % 틀리나?
    # ⚠️ **두 열은 다른 것입니다.** 그 칸만(=10분위 각각) 과 거기까지 누적.
    #    판정에 쓰이는 건 **누적**입니다 — 커버리지는 "위에서부터 잘랐을 때"
    #    이니까요. 처음에 이 둘을 같은 이름으로 찍었다가 "상위 20% 오답률
    #    18.0%" 라는 **틀린 문장**을 만들 뻔했습니다 (그 칸만의 값입니다).
    print(f"\n{'=' * 62}\n확신도 10분위별 오답률 (확신도 높은 쪽부터)\n{'=' * 62}")
    heads = ["릴리스 단독", "3팔 앙상블"]
    print(f"  {'구간':>12}" + "".join(f"{h + ' 그칸':>16}{h + ' 누적':>16}" for h in heads))
    deciles = {}
    for label, probs in [(heads[0], P2[ARMS[0][0]]),
                         (heads[1], np.mean(list(P2.values()), 0))]:
        sn = np.array(CLASSES, dtype=object)[probs.argmax(1)]
        w = is_norm | (sn != truth)
        o = np.argsort(-(p1s * probs.max(1)))
        ws = w[o]
        cum = np.cumsum(ws) / np.arange(1, len(ws) + 1)
        deciles[label] = {
            "bin": [float(ws[i * len(ws) // 10:(i + 1) * len(ws) // 10].mean())
                    for i in range(10)],
            "cum": [float(cum[(i + 1) * len(ws) // 10 - 1]) for i in range(10)]}
    for i in range(10):
        lo, hi = i * 10, (i + 1) * 10
        print(f"  {f'{lo}~{hi}%':>12}"
              + "".join(f"{deciles[k]['bin'][i]:>16.1%}{deciles[k]['cum'][i]:>16.1%}"
                        for k in heads))
    print("  → 판정에 쓰이는 건 **누적** 열입니다 (위에서부터 자르니까요).")
    print("  → 누적이 20% 아래로 안 내려가면 '언제 맞는지 모른다' 는 뜻입니다.")
    results["10분위"] = deciles

    # ── 확신도를 무엇으로 재야 하나 ─────────────────────────────────
    # 서빙(`infer.Engine`)은 p(이상) × p(그 병변) 을 씁니다. 그게 최선인지
    # 아무도 안 봤습니다. 헛알림은 p1 이 낮으니 p1 을 곱하는 게 유리할 텐데,
    # 실제로 그런지 잽니다. **재학습도 추론도 추가로 안 듭니다.**
    print(f"\n{'=' * 62}\n확신도를 무엇으로 재나 (3팔 앙상블, 오답률 20% 목표)\n{'=' * 62}")
    ens = np.mean(list(P2.values()), 0)
    said_name = np.array(CLASSES, dtype=object)[ens.argmax(1)]
    wrong = is_norm | (said_name != truth)

    def _cov(conf, tgt=experiments.NAMING_TARGET_ERROR):
        o = np.argsort(-np.asarray(conf))
        e = np.cumsum(wrong[o]) / np.arange(1, len(o) + 1)
        k = np.flatnonzero(e <= tgt)
        return float((k[-1] + 1) / len(o)) if len(k) else 0.0

    conf_defs = {"p1 × p2max (서빙이 쓰는 것)": p1s * ens.max(1),
                 "p2max 만": ens.max(1),
                 "p1 만": p1s,
                 "p2 마진 (1등−2등)": np.sort(ens, 1)[:, -1] - np.sort(ens, 1)[:, -2]}
    conf_res = {k: _cov(v) for k, v in conf_defs.items()}
    for k, v in conf_res.items():
        print(f"  {k:28} 커버리지 {v:>6.1%}")

    # ── 1단계 임계값을 올리면 ────────────────────────────────────────
    # 헛알림이 넘긴 사진의 20.8% 인데 목표 오답률이 20% 입니다. 즉 **헛알림만으로
    # 이미 목표를 넘습니다.** 임계값을 올리면 헛알림이 줄지만 병변을 더 놓칩니다
    # (그건 화면에 "괜찮아요" 로 나갑니다 — 이름 문제보다 훨씬 나쁜 오류).
    print(f"\n{'=' * 62}\n1단계 임계값을 올리면 (3팔 앙상블)\n{'=' * 62}")
    print(f"  {'임계값':>8}{'1단계 recall':>13}{'넘긴 사진':>10}{'헛알림 몫':>10}"
          f"{'이름 커버리지':>13}")
    sweep = []
    for t in (thr, 0.20, 0.30, 0.40, 0.50, 0.70):
        m = p1s >= t                      # 지금 넘긴 것의 부분집합
        if m.sum() < 200:
            continue
        w, tr = wrong[m], truth[m]
        c = (p1s * ens.max(1))[m]
        o = np.argsort(-c)
        e = np.cumsum(w[o]) / np.arange(1, m.sum() + 1)
        k = np.flatnonzero(e <= experiments.NAMING_TARGET_ERROR)
        cov = float((k[-1] + 1) / m.sum()) if len(k) else 0.0
        rec = float((p1[y1 == 1] >= t).mean())
        fa = float((tr == stages.NORMAL_LABEL).mean())
        sweep.append({"threshold": float(t), "stage1_recall": rec,
                      "n_flagged": int(m.sum()), "false_alarm_share": fa,
                      "naming_coverage": cov})
        print(f"  {t:>8.3f}{rec:>13.3f}{int(m.sum()):>10,}{fa:>10.1%}{cov:>13.1%}")
    print("  ⚠️ 임계값을 올리면 이름은 쉬워지지만 **병변을 더 놓칩니다** — "
          "그건 '괜찮아요' 로 나갑니다. 여기서 고르지 마세요.")
    results["확신도_정의"] = conf_res
    results["1단계_임계값_스윕"] = sweep

    # 비교용 — STEP 11 이 쓴 방식 (병변만, 헛알림 뺀 곡선)
    print(f"\n{'=' * 62}\n참고 — 헛알림을 뺀 옛 방식 (STEP 11 과 같은 분모)\n{'=' * 62}")
    les = ~is_norm
    for label, probs in [(ARMS[0][0] + " 단독", P2[ARMS[0][0]]),
                         ("3팔 앙상블", np.mean(list(P2.values()), 0))]:
        p, t = probs[les], truth[les]
        said_name = np.array(CLASSES, dtype=object)[p.argmax(1)]
        conf, wrong = p.max(1), said_name != t
        o = np.argsort(-conf)
        err = np.cumsum(wrong[o]) / np.arange(1, les.sum() + 1)
        ok = np.flatnonzero(err <= experiments.NAMING_TARGET_ERROR)
        cov = (ok[-1] + 1) / les.sum() if len(ok) else 0.0
        print(f"  {label:22} 커버리지 {cov:.1%}   ← 이 숫자와 위를 헷갈리면 안 됩니다")
        results.setdefault("옛_방식_병변만", {})[label] = float(cov)

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"chunk": a.chunk, "n_val": int(len(va)), "n_flagged": int(flag.sum()),
         "n_false_alarm": n_norm_flag, "stage1_threshold": thr,
         "stage1_recall": float(flag[y1 == 1].mean()),
         "stage1_false_alarm_rate": float(flag[y1 == 0].mean()),
         "results": results}, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
