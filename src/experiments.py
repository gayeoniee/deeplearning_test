"""한 번의 실행으로 결론이 나야 하는 비교 실험들.

⚠️ **왜 노트북이 아니라 여기 있나**

노트북 셀은 `git pull` 로 갱신되지 않습니다. 그래서 이 프로젝트는
"바뀔 수 있는 로직" 을 전부 `src/` 에 둡니다 (`src/gates.py` 의 설명 참고).

여기 있는 건 **학습을 여러 번 돌려 비교하는** 코드입니다. 노트북에 복붙하면
같은 30줄이 해상도마다 반복되고, 고칠 일이 생기면 사용자가 .ipynb 를 다시
받아야 합니다. 함수로 두면 한 줄로 부르고 `git pull` 로 고쳐집니다.

핵심 원칙 — **판정은 검증 점수가 아니라 견고성으로 합니다.**
    검증 macro-F1 은 "정답 박스로 잘라준 사진" 점수입니다.
    실제 보호자 사진에는 박스가 없고 배율이 제각각입니다.
    그래서 배율 교란 하락폭을 같이 재고, 그걸로 채택을 결정합니다.
"""

from __future__ import annotations

import gc
import time
from typing import Any

import torch

# 실행 간 잡음. 같은 설정을 두 번 돌렸을 때 30.7% ↔ 27.8% 였습니다.
# 이보다 작은 차이는 "차이 없음" 으로 읽어야 합니다.
NOISE_PP = 0.03

# 배율 하락이 이 아래면 실사용에 내놓을 만합니다.
DROP_WANT = 0.15


def train_and_measure(
    view,
    *,
    stage: int,
    img_size: int,
    crop_tag: str,
    device: str,
    epochs: int,
    model_name: str = "resnet50",
    finetune: str = "moderate",
    aug: str = "default",
    fold: int = 0,
    subset_frac: float = 1.0,
    n_robust: int = 3000,
    measure_robust: bool = True,
    measure_blur: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """한 설정으로 학습하고 **점수와 견고성을 함께** 돌려줍니다.

    stage=1 이면 정상/이상(이진), stage=2 면 병변 6종입니다.

    반환하는 dict 는 `resolution_report()` / `augmentation_report()` 가 그대로 먹습니다.
    모델은 다 쓰고 나면 버립니다 (해상도를 올리면 VRAM 이 빠듯합니다).
    """
    from src import data, evaluate, models, robust, split, stages, train
    from src.config import CLASSES, CLASSES_STAGE1, CFG, with_aug, with_finetune

    classes = CLASSES_STAGE1 if stage == 1 else CLASSES
    cfg = with_aug(
        with_finetune(
            CFG(model_name=model_name, img_size=img_size, epochs=epochs,
                # 1단계는 정상:이상이 5:5 라 가중치가 필요 없습니다.
                balance_strategy="none" if stage == 1 else "class_weight",
                monitor="macro_f1",
                # ⚠️ 해상도를 이름에 넣습니다 — 안 넣으면 224 체크포인트를 384 학습이
                #    "이미 끝난 학습" 으로 착각하고 건너뜁니다.
                #    (증강 프리셋은 with_aug 가 이름 뒤에 자동으로 붙입니다)
                exp_name=f"stage{stage}_{model_name}_{crop_tag}_{img_size}"),
            finetune),
        aug)

    tr, va = split.get_fold(view, fold)

    # ── 빠른 스윕용 부분 학습 ────────────────────────────────────
    # 멘토 피드백 6번: "초반에는 최대한 빠르게 자동으로 시도"
    # ⚠️ **학습셋만** 줄입니다. 검증셋을 줄이면 프리셋 간 점수 비교가 흔들립니다.
    #    클래스별로 같은 비율을 뽑아 불균형 구조를 유지합니다.
    #    ⚠️ 서브셋 순위가 풀 데이터 순위와 같다는 보장은 없습니다.
    #       **후보를 줄이는 용도**이고, 확정은 반드시 풀 스케일로 다시 합니다.
    if subset_frac < 1.0:
        n_before = len(tr)
        # groupby(...).sample 은 그룹별 비율 표본을 바로 줍니다.
        # (groupby.apply 는 pandas 2.2+ 에서 그룹 컬럼 처리로 경고가 납니다)
        tr = tr.loc[tr.groupby("label").sample(
            frac=subset_frac, random_state=fold).index].reset_index(drop=True)
        if verbose:
            print(f"\n  [스윕] 학습셋 {n_before:,} → {len(tr):,}장 "
                  f"({subset_frac:.0%}) · 검증셋 {len(va):,}장은 그대로")

    if verbose:
        print(f"\n{'━' * 66}")
        print(f"  {stage}단계 @ {img_size}px · 증강 '{aug}'")
        print(f"  크롭 '{crop_tag}' / {epochs}에폭 / 배치 {cfg.resolved_batch_size()}")
        print(f"{'━' * 66}")
        print(f"  train {len(tr):,} / val {len(va):,}")

    model = models.build(model_name, n_classes=len(classes),
                         pretrained=True, drop_rate=cfg.drop_rate)
    dl_tr, dl_va, ds_tr, _ = data.build_loaders(tr, va, cfg, model=model, classes=classes)

    t0 = time.time()
    train.print_status(cfg.exp_name)
    res = train.fit(model, dl_tr, dl_va, cfg, ds_train=ds_tr)
    minutes = (time.time() - t0) / 60

    logits, y = train.cached_logits(model, dl_va, key="val", exp=cfg.exp_name,
                                    n_cls=len(classes), device=device,
                                    tta_hflip=cfg.tta_hflip)

    out: dict[str, Any] = {
        "stage": stage, "img_size": img_size, "crop_tag": crop_tag, "aug": aug,
        # ★ 어떤 백본이었는지 남깁니다 — 2×2 비교(stage1_report)가 이걸로 표를 만듭니다
        "model_name": model_name,
        "subset_frac": subset_frac, "n_train": len(tr),
        "exp_name": cfg.exp_name, "epochs": epochs, "minutes": minutes,
        "batch_size": cfg.resolved_batch_size(),
        "best_epoch": res.best_epoch, "n_epochs": len(res.history),
        # 마지막 에폭이 최고면 아직 덜 학습된 것입니다 (에폭을 더 줘야 합니다)
        "converged": res.best_epoch < len(res.history) - 2,
    }

    if stage == 1:
        rep = evaluate.binary_report(stages.stage1_scores(logits),
                                     stages.binary_targets(y),
                                     target_recall=cfg.target_recall_stage1)
        out.update(auroc=rep["auroc"], threshold=rep["threshold"],
                   precision=rep["precision_at_target"], score=rep["auroc"],
                   score_name="AUROC")
    else:
        rep = evaluate.full_report(logits, y, classes, show=False)
        out.update(macro_f1=rep.metrics["macro_f1"],
                   a6_recall=rep.metrics["per_class"]["recall"][classes.index("A6")],
                   score=rep.metrics["macro_f1"], score_name="macro-F1")
        out["report"] = rep

    if measure_robust:
        r = robust.scale_stress(model, va, cfg, classes, n=n_robust, device=device)
        s = r.get("_summary", {})
        # 키 이름은 robust.py 가 정합니다: baseline / worst / rel_drop
        out.update(scale_drop=s.get("rel_drop"), scale_worst=s.get("worst"),
                   scale_worst_at=s.get("worst_condition"))

    # ⚠️ 화질 교란은 **1단계 전용**에 가깝습니다. 정상 사진이 계통적으로 흐려서
    #    (선명도 50 vs 274) 모델이 화질로 맞힐 수 있는데, val 점수만 보면
    #    그게 안 보입니다. photometric 증강 전후를 이 값으로 비교합니다.
    if measure_blur:
        b = robust.blur_stress(model, va, cfg, classes, n=n_robust, device=device)
        s = b.get("_summary", {})
        out.update(blur_drop=s.get("rel_drop"), blur_worst=s.get("worst"),
                   blur_worst_at=s.get("worst_condition"))

    del model, dl_tr, dl_va, ds_tr
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def _compare(runs: list[dict], key: str, base_val, title: str,
             fmt=str) -> dict[str, Any]:
    """`key` 만 다른 실행들을 견고성 기준으로 비교합니다.

    해상도 비교와 증강 비교가 같은 판정 규칙을 쓰도록 한 곳에 모았습니다.
    """
    runs = [r for r in runs if r]
    if not runs:
        return {}

    by_stage: dict[int, list[dict]] = {}
    for r in runs:
        by_stage.setdefault(r["stage"], []).append(r)

    verdicts: dict[str, Any] = {}
    for stage in sorted(by_stage):
        rs = by_stage[stage]
        name = rs[0]["score_name"]
        print(f"\n{'=' * 68}\n {stage}단계 — {title}\n{'=' * 68}")
        print(f"  {'설정':<16}{name:>10}{'배율하락':>10}{'최악조건':>10}{'분':>7}   수렴")
        for r in rs:
            drop = r.get("scale_drop")
            ds = f"{drop:>9.1%}" if drop is not None else f"{'—':>10}"
            print(f"  {fmt(r[key]):<16}{r['score']:>10.4f}{ds}"
                  f"{str(r.get('scale_worst_at', '?')):>10}{r['minutes']:>7.0f}   "
                  f"{'수렴' if r['converged'] else '더 필요'}")

        base = next((r for r in rs if r[key] == base_val), rs[0])
        others = [r for r in rs if r is not base]
        v: dict[str, Any] = {"baseline": base}
        if not others:
            print("\n  (비교 대상이 없어 판정을 생략합니다)")
            verdicts[f"stage{stage}"] = v
            continue

        # 견고성이 가장 좋은 것 = 하락이 가장 작은 것. 점수가 아니라 이걸로 고릅니다.
        scored = [r for r in others if r.get("scale_drop") is not None]
        if not scored or base.get("scale_drop") is None:
            v["verdict"] = "unmeasured"
            print("\n  ⚠️ 배율 하락을 안 재서 판정할 수 없습니다.")
            verdicts[f"stage{stage}"] = v
            continue

        best = min(scored, key=lambda r: r["scale_drop"])
        bd, hd = base["scale_drop"], best["scale_drop"]
        d_drop, d_score = hd - bd, best["score"] - base["score"]
        v.update(best=best, drop_delta=d_drop, score_delta=d_score)

        print(f"\n  가장 견고한 설정: {fmt(best[key])}")
        print(f"  {name}  {base['score']:.4f} → {best['score']:.4f}  ({d_score:+.4f})")
        print(f"  배율 하락  {bd:.1%} → {hd:.1%}  ({d_drop:+.1%})")

        if d_drop < -NOISE_PP and hd <= DROP_WANT:
            v["verdict"] = "adopt"
            print(f"\n  ✅ 채택 — 하락이 잡음(±{NOISE_PP:.0%})보다 크게 줄었고 "
                  f"목표({DROP_WANT:.0%}) 안에 들어왔습니다.")
        elif d_drop < -NOISE_PP:
            v["verdict"] = "improved"
            print(f"\n  🤔 개선됐지만 아직 {hd:.0%} 입니다 (목표 {DROP_WANT:.0%}).")
            print("     방향은 맞습니다 — 더 밀어붙이거나 배포 설계로 보완하세요.")
        else:
            v["verdict"] = "no_effect"
            print(f"\n  ❌ 하락폭이 그대로입니다 (잡음 ±{NOISE_PP:.0%} 안).")
            print("     이 축으로는 안 잡힙니다. 남은 건 배포 설계입니다:")
            print('     "병변이 화면 절반 이상 차지하게 찍어주세요" 로 입력을 제한하고,')
            print("     full 크롭 점수를 정직한 숫자로 보고하는 쪽.")

        if not best["converged"]:
            print(f"\n  📈 {fmt(best[key])} 는 마지막 에폭이 최고였습니다 — 덜 학습됐습니다. "
                  f"에폭을 {int(best['epochs'] * 1.6)} 로 올리면 더 오를 수 있습니다.")
        verdicts[f"stage{stage}"] = v

    return verdicts


def resolution_report(runs: list[dict], *, baseline_size: int = 224) -> dict[str, Any]:
    """해상도 비교. 판정 기준은 `_compare` 에 있습니다."""
    return _compare(runs, "img_size", baseline_size, "해상도 비교",
                    fmt=lambda v: f"{v}px")


def crop_report(runs: list[dict], *, baseline: str = "m1.5") -> dict[str, Any]:
    """크롭 태그 비교 (m1.5 vs m2.5 vs f320 …).

    ⚠️ 크롭은 **다른 축과 성격이 다릅니다.** 증강이나 백본을 바꾸면 모델만
    바뀌지만, 크롭을 바꾸면 **입력 자체가 바뀝니다.** 그래서 이걸 먼저
    확정해야 뒤(촬영 가이드·임계값·백본 순위)가 흔들리지 않습니다.

    비교 지점은 "병변을 크게 보되 흐리게(m1.5)" vs "작게 보되 선명하게(m2.5)"
    입니다. 크롭 창이 좁으면 384 로 늘릴 때 없는 픽셀을 만들어냅니다 —
    실측으로 A1 은 m1.5 에서 2.6배 확대, m2.5 에서 1.6배입니다.
    """
    return _compare(runs, "crop_tag", baseline, "크롭 비교", fmt=str)


def augmentation_report(runs: list[dict], *, baseline: str = "default") -> dict[str, Any]:
    """증강 프리셋 비교.

    ⚠️ **점수가 아니라 배율 하락으로 고릅니다.** 검증 macro-F1 은 "정답 박스로
    잘라준 사진" 점수라서, 증강을 세게 걸면 대개 조금 내려갑니다. 그래도
    하락폭이 크게 줄면 배포에는 그쪽이 낫습니다.
    """
    return _compare(runs, "aug", baseline, "증강 비교", fmt=str)


# ──────────────────────────────────────────────────────────────
# 1단계 2×2 실험 (백본 × 증강)
# ──────────────────────────────────────────────────────────────
# STEP 5 에서 1단계가 holdout 에서 무너졌습니다 (AUROC 0.8143 → 0.7412).
# 용의자가 둘이고, 둘을 한 실행에서 갈라 봅니다:
#
#   ① 화질 지름길  — 정상 사진이 계통적으로 흐림 (선명도 50 vs 274)
#                    → `photometric` 증강으로 막히나?
#   ② 표현력·사전학습 — resnet50(2015, in1k) 이 약한 건가?
#                    → in21k 사전학습 백본이면 나은가?
#
# ⚠️ **holdout 은 여기서 안 봅니다.** 4개 중에 고르는 데 holdout 을 쓰면
#    그 순간 holdout 이 오염되어 "처음 보는 데이터" 가 아니게 됩니다.
#    고른 하나를 풀 데이터로 다시 학습한 뒤에만 엽니다.

# 잡음 폭 — 실측 근거를 달아둡니다 (추정치 금지, 규칙 1)
AUROC_NOISE = 0.01     # 같은 설정 두 실행: 0.8192(08-21) vs 0.8155(08-22)
BLUR_NOISE_PP = 0.05   # 교란 검사 일반 (STEP 5 에서 ±3%p → ±5%p 상향)


def stage1_report(runs: list[dict], *, base_model: str = "resnet50",
                  base_aug: str = "default") -> dict[str, Any]:
    """백본 × 증강 2×2 를 읽고 **미리 정해둔 기준**으로 판정합니다.

    판정 기준 (실험 **전에** 못 박음 — 규칙 2):

      · `photometric` 채택 ← AUROC 가 {AUROC_NOISE} 이상 안 떨어지면서
                             흐림 하락이 {BLUR_NOISE_PP} 이상 줄어들 때
      · 백본 교체 채택     ← AUROC 가 {AUROC_NOISE} 이상 오를 때
      · 둘 다 아니면       → 그 축은 닫고 다음으로
    """
    runs = [r for r in runs if r]
    if not runs:
        return {}

    models_ = sorted({r["model_name"] for r in runs})
    augs = sorted({r["aug"] for r in runs}, key=lambda a: (a != base_aug, a))
    get = {(r["model_name"], r["aug"]): r for r in runs}

    def fmt(v, pct=False, nd=4):
        if v is None:
            return "   못 잼"
        return f"{v:>8.1%}" if pct else f"{v:>8.{nd}f}"

    print("\n" + "=" * 74)
    print(" 1단계 2×2 — 백본 × 증강")
    print("=" * 74)
    print(f"  {'백본':<16}{'증강':<14}{'AUROC':>10}{'흐림하락':>10}"
          f"{'precision':>11}{'분':>6}")
    for m in models_:
        for a in augs:
            r = get.get((m, a))
            if not r:
                continue
            print(f"  {m:<16}{a:<14}{fmt(r.get('auroc'))}{fmt(r.get('blur_drop'), pct=True)}"
                  f"{fmt(r.get('precision'), nd=3)}{r['minutes']:>6.0f}")

    base = get.get((base_model, base_aug))
    if base is None:
        print("\n  ⚠️ 기준 조합을 못 찾아 판정을 생략합니다.")
        return {"runs": runs}

    verdict: dict[str, Any] = {"baseline": f"{base_model}/{base_aug}"}
    print(f"\n  기준: {base_model} / {base_aug}  "
          f"(AUROC {base['auroc']:.4f}, 흐림 하락 {fmt(base.get('blur_drop'), pct=True).strip()})")
    print(f"  잡음 폭: AUROC ±{AUROC_NOISE} · 흐림 하락 ±{BLUR_NOISE_PP:.0%}")

    # ── 증강 축 ────────────────────────────────────────────────
    alt_aug = next((a for a in augs if a != base_aug), None)
    if alt_aug:
        r = get.get((base_model, alt_aug))
        if r:
            d_auroc = r["auroc"] - base["auroc"]
            d_blur = (r["blur_drop"] - base["blur_drop"]
                      if r.get("blur_drop") is not None and base.get("blur_drop") is not None
                      else None)
            print(f"\n  [증강 축] {base_aug} → {alt_aug}")
            print(f"    AUROC     {d_auroc:+.4f}")
            print(f"    흐림 하락  {'못 잼' if d_blur is None else f'{d_blur:+.1%}'}")
            if d_auroc > -AUROC_NOISE and d_blur is not None and d_blur <= -BLUR_NOISE_PP:
                verdict["aug"] = alt_aug
                print(f"    ✅ {alt_aug} 채택 — 점수를 안 깎으면서 화질 의존을 줄였습니다.")
            elif d_auroc <= -AUROC_NOISE:
                verdict["aug"] = base_aug
                print(f"    ❌ {base_aug} 유지 — {alt_aug} 이 점수를 깎습니다 "
                      "(2단계에서와 같은 이유일 수 있습니다: 신호를 지움).")
            else:
                verdict["aug"] = base_aug
                print(f"    ➖ {base_aug} 유지 — 차이가 잡음 안입니다. 이 축은 닫습니다.")

    # ── 백본 축 ────────────────────────────────────────────────
    alt_model = next((m for m in models_ if m != base_model), None)
    if alt_model:
        r = get.get((alt_model, base_aug))
        if r:
            d = r["auroc"] - base["auroc"]
            print(f"\n  [백본 축] {base_model} → {alt_model}")
            print(f"    AUROC     {d:+.4f}")
            if d >= AUROC_NOISE:
                verdict["model"] = alt_model
                print(f"    ✅ {alt_model} 채택")
            elif d <= -AUROC_NOISE:
                verdict["model"] = base_model
                print(f"    ❌ {base_model} 유지 — {alt_model} 이 더 나쁩니다.")
            else:
                verdict["model"] = base_model
                print(f"    ➖ {base_model} 유지 — 차이가 잡음 안입니다.")

    # ── 상호작용 ────────────────────────────────────────────────
    if alt_model and alt_aug:
        both = get.get((alt_model, alt_aug))
        if both:
            best = max(runs, key=lambda r: r["auroc"])
            print(f"\n  [둘 다]   {alt_model} / {alt_aug}   AUROC {both['auroc']:.4f}")
            print(f"  가장 높은 조합: {best['model_name']} / {best['aug']} "
                  f"(AUROC {best['auroc']:.4f})")
            verdict["best"] = {"model": best["model_name"], "aug": best["aug"],
                               "auroc": best["auroc"], "exp_name": best["exp_name"]}

    print("\n  ⚠️ 여기서 고른 조합은 **후보**입니다. 서브셋·짧은 에폭이라")
    print("     절대값은 풀 학습과 다릅니다 (STEP 4B 에서 확인). 풀 학습으로 확정하세요.")
    print("  ⚠️ holdout 은 아직 안 봤습니다 — 풀 학습 뒤에 한 번만 엽니다.")
    print("=" * 74)
    return verdict


def estimate_runtime(model_names: list[str], img_size: int, n_train: int,
                     epochs: int, n_conditions: int | None = None,
                     device: str | None = None) -> dict[str, Any]:
    """학습을 시작하기 **전에** 총 예상 시간을 찍습니다.

    "몇 시간 걸릴지 모르고 돌렸다가 뒤통수" 를 여러 번 맞아서 넣었습니다.
    합성 텐서로 GPU 속도만 재므로 백본당 20초 안쪽입니다.

    ⚠️ 데이터 로딩이 병목이면 실제는 이보다 느립니다. **하한 추정**입니다.
    """
    from src import bench
    from src.config import CFG, MODEL_BY_KEY

    n_conditions = n_conditions or len(model_names)
    rows, total_min = [], 0.0
    print("\n" + "=" * 66)
    print(" 시작 전 시간 추정 (GPU 속도 실측, 백본당 ~20초)")
    print("=" * 66)
    for key in model_names:
        spec = MODEL_BY_KEY[key]
        cfg = CFG(model_name=spec.timm_name, img_size=img_size)
        g = bench.gpu_speed(cfg, n_classes=2, steps=20)
        ips = float(g.get("img_per_sec") or 0.0)
        # GPU 가 없으면 img_per_sec 이 NaN 입니다 (NaN 은 비교가 전부 False 라 따로 봅니다)
        if not (ips > 0) or ips != ips:
            print(f"  {key:<16} 속도를 못 쟀습니다 ({g.get('note', '이유 불명')}) — 추정 생략")
            continue
        epoch_min = n_train / ips / 60
        run_min = epoch_min * epochs
        rows.append({"model": key, "img_per_sec": ips, "batch": g.get("batch"),
                     "peak_vram_gb": g.get("peak_vram_gb"),
                     "epoch_min": epoch_min, "run_min": run_min})
        total_min += run_min
        print(f"  {key:<16}{ips:>7.0f} img/s  배치 {g.get('batch', '?'):>3}  "
              f"VRAM {g.get('peak_vram_gb', 0):>4.1f}GB   "
              f"1에폭 {epoch_min:>5.1f}분   {epochs}에폭 {run_min:>6.0f}분")

    # 조건 수가 백본 수보다 많으면(2×2 처럼) 백본별로 같은 횟수만큼 돕니다
    reps = max(1, n_conditions // max(len(rows), 1))
    total_min *= reps
    print("-" * 66)
    print(f"  조건 {n_conditions}개 → 학습 총 예상 **{total_min / 60:.1f}시간**  "
          f"(+ 교란 검사·크롭 확인 별도)")
    print("  ⚠️ GPU 속도만 잰 **하한**입니다. 데이터 로딩이 병목이면 더 걸립니다.")
    print("  → 너무 길면 여기서 멈추고 서브셋을 줄이거나 백본을 바꾸세요.")
    print("=" * 66)
    return {"rows": rows, "total_hours": total_min / 60, "n_conditions": n_conditions}
