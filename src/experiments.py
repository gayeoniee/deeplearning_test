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

# 교란 검사(배율/위치/화질) 하락의 잡음 폭.
#
# ⚠️ **짝지은 비교 전용입니다** — 두 조건을 *같은 실행·같은 표본*으로 나란히
#    돌렸을 때만 이 값을 쓰세요 (03c 방식). 그러면 표본 추출 잡음이 상쇄됩니다.
#
# **실행이 다르면 ±8%p 로 보세요.** 2단계 설정을 하나도 안 바꾸고 네 번 쟀더니
# 배율 하락이 이렇게 나왔습니다:
#     STEP 4D 16.0%  ·  STEP 5 21.6%(n=2000) / 16.8%(n=3000)  ·  STEP 7 23.8%
# 폭이 7.8%p 입니다. 처음엔 ±3%p, STEP 5 에서 ±5%p 로 올렸는데 그것도 낙관적이었습니다.
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
    balance: str | None = None,
    hair_alpha: float = 1.0,
    lr: float | None = None,
    backbone_lr_mult: float | None = None,
    warmup_epochs: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """한 설정으로 학습하고 **점수와 견고성을 함께** 돌려줍니다.

    stage=1 이면 정상/이상(이진), stage=2 면 병변 6종입니다.

    반환하는 dict 는 `resolution_report()` / `augmentation_report()` 가 그대로 먹습니다.
    모델은 다 쓰고 나면 버립니다 (해상도를 올리면 VRAM 이 빠듯합니다).

    Args:
        balance: 불균형 대응 전략을 덮어씁니다. 기본은 1단계 `"none"` /
            2단계 `"class_weight"`. `"hair_weighted"` 를 주면 **털처럼 가는 선이
            많은 정상 사진**을 더 자주 뽑습니다 (`data.hair_sampler`).
        hair_alpha: `balance="hair_weighted"` 일 때의 세기. `0` 이면 균등.
        lr / backbone_lr_mult / warmup_epochs: 학습률 축을 비교할 때 덮어씁니다.
            ★ 셋 중 하나라도 주면 **실험 이름에 붙습니다.** 안 붙이면 네 판이
            같은 폴더를 쓰고, `train.fit` 이 뒤의 세 판을 "이미 끝난 학습" 으로
            조용히 건너뜁니다 — 해상도·샘플러에서 이미 두 번 당한 실패입니다.
    """
    from src import data, evaluate, models, robust, split, stages, train
    from src.config import CLASSES, CLASSES_STAGE1, CFG, with_aug, with_finetune

    classes = CLASSES_STAGE1 if stage == 1 else CLASSES
    cfg = with_aug(
        with_finetune(
            CFG(model_name=model_name, img_size=img_size, epochs=epochs,
                # 1단계는 정상:이상이 5:5 라 가중치가 필요 없습니다.
                balance_strategy=balance or ("none" if stage == 1 else "class_weight"),
                hair_alpha=hair_alpha,
                monitor="macro_f1",
                # ⚠️ 해상도를 이름에 넣습니다 — 안 넣으면 224 체크포인트를 384 학습이
                #    "이미 끝난 학습" 으로 착각하고 건너뜁니다.
                #    (증강 프리셋은 with_aug 가 이름 뒤에 자동으로 붙입니다)
                # ⚠️ 샘플러가 다르면 **이름도 달라야** 합니다. 안 그러면
                #    train.fit 이 기준선 체크포인트를 보고 "이미 끝난 학습" 으로
                #    착각해 처치를 통째로 건너뜁니다 — 해상도에서 이미 당한 실패입니다.
                exp_name=f"stage{stage}_{model_name}_{crop_tag}_{img_size}"
                         + (f"_hair{hair_alpha:g}"
                            if balance == "hair_weighted" else "")),
            finetune),
        aug)

    # ── 학습률 축 (주었을 때만) ──────────────────────────────────
    # ⚠️ 이름을 먼저 붙이고 값을 넣습니다. 순서가 바뀌어도 결과는 같지만,
    #    **이름을 안 붙이면** 네 판이 한 폴더를 공유해 뒤의 세 판이 통째로
    #    건너뛰어집니다. 그래도 표는 그럴듯하게 나오고 아무 에러도 안 납니다.
    _lr_tag = ""
    if lr is not None:
        _lr_tag += f"_lr{lr:g}"
    if backbone_lr_mult is not None:
        _lr_tag += f"_bb{backbone_lr_mult:g}"
    if warmup_epochs is not None:
        _lr_tag += f"_wu{warmup_epochs}"
    if _lr_tag:
        over = {"exp_name": cfg.exp_name + _lr_tag}
        if lr is not None:
            over["lr"] = lr
        if backbone_lr_mult is not None:
            over["backbone_lr_mult"] = backbone_lr_mult
        if warmup_epochs is not None:
            over["warmup_epochs"] = warmup_epochs
        cfg = CFG.from_dict({**cfg.to_dict(), **over})

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

    # ⚠️ img_size 를 넘깁니다. CNN 은 해상도가 자유로워 무시되지만, ViT 계열은
    #    위치 임베딩 크기가 고정이라 안 넘기면 **학습 도중에** shape 오류가 납니다.
    #    (models.build 는 필요할 때만 timm 에 전달합니다)
    model = models.build(model_name, n_classes=len(classes),
                         pretrained=True, drop_rate=cfg.drop_rate,
                         img_size=img_size)
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
        # ★ 샘플러 설정 — 기준선/처치를 표에서 갈라 보려면 이게 있어야 합니다
        "balance": cfg.balance_strategy,
        "hair_alpha": hair_alpha if cfg.balance_strategy == "hair_weighted" else 0.0,
        # ★ 어떤 백본이었는지 남깁니다 — 2×2 비교(stage1_report)가 이걸로 표를 만듭니다
        "model_name": model_name,
        "subset_frac": subset_frac, "n_train": len(tr),
        "exp_name": cfg.exp_name, "epochs": epochs, "minutes": minutes,
        "batch_size": cfg.resolved_batch_size(),
        # ★ 학습률 축 — 표에서 판을 갈라 보려면 이게 있어야 합니다
        "lr": cfg.lr, "backbone_lr_mult": cfg.backbone_lr_mult,
        "backbone_lr": cfg.lr * cfg.backbone_lr_mult,
        "warmup_epochs": cfg.warmup_epochs,
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
# 학습률 축 (STEP 17) — "best 가 0에폭" 을 설명할 수 있나
# ──────────────────────────────────────────────────────────────
# STEP 16 에서 2단계(convnextv2_base, 89M)의 best 가 **0에폭**이었습니다.
# 14에폭까지 돌려도 못 넘었고, 그 사이 train loss 가 **올라갔습니다**
# (1.0292 → 1.2321, warmup 2에폭 구간). 설명이 둘입니다:
#
#   ① warmup 이라 정상이다 — 학습률이 꼭대기를 찍을 때 흔들렸다가 회복 중
#   ② 학습률이 너무 높다   — 백본 9e-5(3e-4 × 0.3) 를 7,569스텝/에폭 먹여
#                            사전학습 표현이 흐트러졌고 끝내 회복 못 함
#
# ②가 맞으면 지금 macro-F1 0.599 는 천장이 아니고, **A4 recall 0.264 와
# 배율 하락 26.8% 도 "덜 배운 모델에서 잰 값"** 이 됩니다. STEP 9→12 에서
# 이미 같은 일을 당했습니다 (미수렴 기준선이 교란 검사를 왜곡함).
#
# ⚠️ **판정 기준을 여기 못 박아 둡니다** (규칙 2). 결과를 보고 기준을 고르면
#    무슨 숫자가 나와도 성공담이 됩니다.
LR_MIN_BEST_EPOCH = 3      # ← 1차 기준. 점수보다 이게 먼저입니다
MACRO_F1_NOISE = 0.02      # 같은 설정 두 실행: 0.4862 vs 0.5024 (STEP 12·13)


def lr_report(runs: list[dict], *, baseline_lr: float = 3e-4,
              baseline_mult: float = 0.3) -> dict[str, Any]:
    """학습률 비교 — **1차 기준은 점수가 아니라 `best_epoch` 입니다.**

    묻는 것은 "어느 판이 제일 높나" 가 아니라 **"학습이 되긴 하나"** 입니다.
    best 가 0에폭이면 그 판도 같은 병에 걸린 것이고, 점수가 조금 높아도
    설명이 안 됩니다.

    판정 (실험 전 고정):
      1차 — `best_epoch >= LR_MIN_BEST_EPOCH` (=3)
      2차 — macro-F1 이 기준선 대비 `+MACRO_F1_NOISE`(0.02) 이상
      둘 다 만족하는 판이 없으면 → **학습률 축을 닫습니다.**
      "0에폭이 진짜 최고" 도 결론이고, 그러면 원인은 다른 데 있습니다.
    """
    runs = [r for r in runs if r]
    if not runs:
        return {}

    def key(r):
        return (r.get("lr"), r.get("backbone_lr_mult"), r.get("warmup_epochs"))

    base = next((r for r in runs
                 if r.get("lr") == baseline_lr
                 and r.get("backbone_lr_mult") == baseline_mult), runs[0])

    print(f"\n{'=' * 74}\n 2단계 학습률 — 1차 기준은 best_epoch >= "
          f"{LR_MIN_BEST_EPOCH}\n{'=' * 74}")
    print(f"  {'헤드lr':>8}{'×배수':>7}{'백본lr':>10}{'warmup':>8}"
          f"{'macro-F1':>10}{'best':>6}{'/에폭':>6}{'배율하락':>9}{'분':>6}")
    for r in runs:
        drop = r.get("scale_drop")
        ds = f"{drop:>8.1%}" if drop is not None else f"{'—':>9}"
        mark = " ★" if r is base else ""
        print(f"  {r.get('lr', 0):>8.0e}{r.get('backbone_lr_mult', 0):>7.2f}"
              f"{r.get('backbone_lr', 0):>10.0e}{r.get('warmup_epochs', 0):>8}"
              f"{r['score']:>10.4f}{r.get('best_epoch', -1):>6}"
              f"{r.get('n_epochs', 0):>6}{ds}{r['minutes']:>6.0f}{mark}")

    # ★ **먼저: 기준선에서 현상이 재현됐는가.**
    #   "best 가 0에폭" 은 한 에폭의 스텝 수가 많아서 생긴 현상입니다.
    #   서브셋으로 줄이면 스텝/에폭도 줄어 **현상 자체가 안 나타날 수 있습니다.**
    #   그 상태에서 B·C·D 가 좋아 보이면 "학습률이 고쳤다" 가 아니라
    #   "원래 문제가 없었다" 입니다 — 아무것도 못 배우고 시간만 씁니다.
    if base.get("best_epoch", -1) >= LR_MIN_BEST_EPOCH:
        print(f"\n  🚨 **기준선 A 의 best_epoch 이 이미 "
              f"{base['best_epoch']} 입니다 (≥ {LR_MIN_BEST_EPOCH}).**")
        print("     STEP 16 에서 보려던 '0에폭 best' 가 이 규모에서는 재현되지")
        print("     않았습니다. 다른 판이 좋아 보여도 학습률 덕분이라고 말할 수")
        print("     없습니다 — **이 실험은 무효입니다.**")
        print(f"     스텝/에폭을 STEP 16(7,569)에 가깝게 올려서 다시 하세요"
              f" — 지금 학습 {base.get('n_train', 0):,}장.")
        return {"verdict": "무효 — 기준선에서 현상 미재현",
                "baseline": key(base), "trained": [], "winner": None,
                "min_best_epoch": LR_MIN_BEST_EPOCH, "noise": MACRO_F1_NOISE}

    trained = [r for r in runs if r.get("best_epoch", -1) >= LR_MIN_BEST_EPOCH]
    better = [r for r in trained if r["score"] >= base["score"] + MACRO_F1_NOISE]

    print(f"\n  기준선: macro-F1 {base['score']:.4f} (best epoch "
          f"{base.get('best_epoch', -1)}) — 현상 재현됨 ✅")
    print(f"  1차 통과 (best_epoch >= {LR_MIN_BEST_EPOCH}): "
          f"{len(trained)}/{len(runs)}판")

    if not trained:
        verdict = "축 닫힘"
        print("\n  ❌ **어느 판도 0~2에폭을 못 벗어났습니다.**")
        print("     학습률 축을 닫습니다 — 0에폭 best 는 학습률 탓이 아닙니다.")
        print("     다음 용의자: 데이터(라벨 노이즈) · 백본 크기 · 증강 강도")
    elif not better:
        verdict = "학습은 되나 점수는 그대로"
        print(f"\n  ⚠️ 학습은 되는데 점수가 잡음(±{MACRO_F1_NOISE}) 안입니다.")
        print("     '0에폭 best' 는 고쳤지만 성능은 안 올랐습니다 —")
        print("     지금 0.599 가 데이터 천장이라는 쪽으로 기웁니다.")
        for r in trained:
            print(f"     · lr {r['lr']:.0e} ×{r['backbone_lr_mult']:.2f}: "
                  f"{r['score']:.4f} ({r['score'] - base['score']:+.4f}), "
                  f"best epoch {r['best_epoch']}")
    else:
        win = max(better, key=lambda r: r["score"])
        verdict = "채택"
        print(f"\n  ✅ **채택: 헤드 lr {win['lr']:.0e} × 배수 "
              f"{win['backbone_lr_mult']:.2f} (백본 {win['backbone_lr']:.0e})**")
        print(f"     macro-F1 {base['score']:.4f} → {win['score']:.4f} "
              f"({win['score'] - base['score']:+.4f}, 잡음 ±{MACRO_F1_NOISE} 밖)")
        print(f"     best epoch {base.get('best_epoch', -1)} → {win['best_epoch']}")
        print("\n  ⚠️ **서브셋 결과입니다.** 확정하려면 06 을 전체 데이터로 다시")
        print("     돌려야 합니다 — 서브셋 순위가 풀 순위와 같다는 보장은 없습니다.")

    return {"verdict": verdict, "baseline": key(base),
            "trained": [key(r) for r in trained],
            "winner": key(max(better, key=lambda r: r["score"])) if better else None,
            "min_best_epoch": LR_MIN_BEST_EPOCH,
            "noise": MACRO_F1_NOISE}


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
    # ⚠️ 전에는 대체 백본을 **하나만** 봤습니다 (2×2 전용). 3종 이상을 비교하면
    #    나머지가 조용히 무시됩니다. 이제 전부 보고, 그중 가장 많이 오른 것을
    #    고릅니다. 후보가 하나뿐이면 예전과 똑같이 동작합니다.
    alts = [m for m in models_ if m != base_model]
    gains: list[tuple[str, float]] = []
    for alt_model in alts:
        r = get.get((alt_model, base_aug))
        if not r:
            continue
        d = r["auroc"] - base["auroc"]
        gains.append((alt_model, d))
        print(f"\n  [백본 축] {base_model} → {alt_model}")
        print(f"    AUROC     {d:+.4f}")
        if d >= AUROC_NOISE:
            print(f"    ✅ 기준선을 넘습니다 (+{d:.4f} ≥ 잡음 {AUROC_NOISE})")
        elif d <= -AUROC_NOISE:
            print(f"    ❌ 더 나쁩니다")
        else:
            print(f"    ➖ 차이가 잡음({AUROC_NOISE}) 안입니다 — 구분 불가")

    if gains:
        best_alt, best_d = max(gains, key=lambda t: t[1])
        if best_d >= AUROC_NOISE:
            verdict["model"] = best_alt
            # 2등과의 차이도 잡음 안이면 "이겼다" 고 말하면 안 됩니다.
            others = [d for m, d in gains if m != best_alt]
            runner = max(others) if others else None
            if runner is not None and abs(best_d - runner) < AUROC_NOISE:
                verdict["model_tie"] = True
                print(f"\n    ⚠️ {best_alt} 이 1등이지만 2등과 차이가 "
                      f"{abs(best_d - runner):.4f} 로 잡음({AUROC_NOISE}) 안입니다 — "
                      "**구분 불가**. 싼 쪽을 고르세요.")
            print(f"\n    ★ 백본 축: {best_alt} (+{best_d:.4f})")
        else:
            verdict["model"] = base_model
            print(f"\n    ★ 백본 축: {base_model} 유지 — 아무도 잡음을 못 넘었습니다.")

    # ── 채택 조합 ──────────────────────────────────────────────
    # ⚠️ **val AUROC 가 가장 높은 조합을 고르면 안 됩니다.** 그게 바로 우리가
    #    당한 실수입니다 — val 0.8143 로 골랐는데 holdout 에서 0.7412 였습니다.
    #    val 점수는 지름길을 쓰는 모델도 높게 나옵니다. 그래서 **두 축의 판정을
    #    그대로 합친 조합**을 채택합니다. 판정에는 흐림 하락이 들어갑니다.
    pick_m, pick_a = verdict.get("model", base_model), verdict.get("aug", base_aug)
    picked = get.get((pick_m, pick_a))
    if picked:
        verdict["best"] = {"model": pick_m, "aug": pick_a,
                           "auroc": picked["auroc"], "blur_drop": picked.get("blur_drop"),
                           "exp_name": picked["exp_name"]}
        print(f"\n  ★ 채택: {pick_m} / {pick_a}   "
              f"AUROC {picked['auroc']:.4f} · 흐림 하락 "
              f"{fmt(picked.get('blur_drop'), pct=True).strip()}")

    top = max(runs, key=lambda r: r["auroc"])
    if picked and (top["model_name"], top["aug"]) != (pick_m, pick_a):
        print(f"\n  ⚠️ val AUROC 가 가장 높은 건 {top['model_name']} / {top['aug']} "
              f"({top['auroc']:.4f}) 이지만 채택하지 않습니다.")
        print(f"     흐림 하락이 {fmt(top.get('blur_drop'), pct=True).strip()} 로 "
              f"지름길에 기대고 있습니다 (채택안 "
              f"{fmt(picked.get('blur_drop'), pct=True).strip()}).")
        print(f"     AUROC 차이 {top['auroc'] - picked['auroc']:+.4f} 는 "
              f"잡음(±{AUROC_NOISE}) 안입니다."
              if abs(top["auroc"] - picked["auroc"]) < AUROC_NOISE else
              f"     ⚠️ AUROC 차이 {top['auroc'] - picked['auroc']:+.4f} 는 잡음 밖입니다 "
              "— 어느 쪽을 택할지 사람이 판단하세요.")

    print("\n  ⚠️ 여기서 고른 조합은 **후보**입니다. 서브셋·짧은 에폭이라")
    print("     절대값은 풀 학습과 다릅니다 (STEP 4B 에서 확인). 풀 학습으로 확정하세요.")
    print("  ⚠️ holdout 은 아직 안 봤습니다 — 풀 학습 뒤에 한 번만 엽니다.")
    print("=" * 74)
    return verdict


# ──────────────────────────────────────────────────────────────
# 2단계 백본 비교 (STEP 9)
# ──────────────────────────────────────────────────────────────
# 2단계는 지금까지 resnet50 하나로만 돌았습니다. **한 번도 비교한 적이 없습니다.**
# 1단계는 STEP 6 에서 effnetv2_s 로 바꿨지만, 그 결과를 2단계에 옮겨 적으면
# 안 됩니다 — 같은 `photometric` 증강이 1단계에는 약이고 2단계에는 독이었습니다.

# macro-F1 잡음 폭 — 실측 근거 (추정치 금지, 규칙 1)
#   같은 설정 두 실행: m2.5 0.5395 / 0.5313 (Δ0.008), m1.5 0.5697 / 0.5536 (Δ0.016)
#   부트스트랩 95% CI 반폭: 0.5456 → 0.5243~0.5663 (±0.021)
# 둘을 합쳐 ±0.02 로 잡습니다.
F1_NOISE = 0.02

# 배율 하락으로 후보를 떨어뜨릴 때 쓰는 문턱.
#
# 원래 8%p 였습니다 (BLUR_NOISE_PP + 0.03). 그런데 우리가 **따로** 잰 교란 검사
# 잡음이 그것보다 큽니다:
#   · 설정을 하나도 안 바꾼 2단계를 일곱 번 재서 배율 하락 15.4% ~ 25.0% (폭 9.6%p)
#   · STEP 10 한 실행 안에서 표본 수만 2,000 → 3,000 으로 바꿨더니
#     위치 하락이 9.3% → 20.2% (10.9%p)
#
# 즉 8%p 짜리 문턱은 **잡음만으로도 걸립니다.** 실제로 STEP 9 에서
# convnextv2_base 가 +8.1%p 로 탈락했는데, 그건 판정이 아니라 동전 던지기였습니다.
#
# ⚠️ 이 값을 STEP 9 결과를 **보고 나서** 올리는 게 아닙니다. 근거는 STEP 10
#    (다른 실험)에서 나온 잡음 측정이고, 04 를 다시 돌리기 **전에** 못 박습니다
#    (규칙 2). 이 사이 구간은 탈락도 통과도 아닌 **구분 불가**로 적습니다.
SCALE_DROP_REJECT_PP = 0.12


def backbone_report(runs: list[dict], *, base_model: str = "resnet50") -> dict[str, Any]:
    """2단계 백본 비교를 **미리 정해둔 기준**으로 판정합니다.

    판정 기준 (실험 **전에** 못 박음 — 규칙 2)
    -------------------------------------------
    1. macro-F1 이 기준선보다 **+0.02(F1_NOISE) 넘게** 높아야 교체 후보입니다.
       그 안이면 "구분 불가" 이고, 구분 불가일 때는 **기준선을 유지**합니다
       (바꿀 이유가 없으면 안 바꿉니다 — 바꾸면 비교 이력이 끊깁니다).
    2. 점수가 올라도 **배율 하락이 12%p(SCALE_DROP_REJECT_PP) 넘게 나빠지면**
       채택하지 않습니다. STEP 5·6 에서 val 최고점을 골랐다가 holdout 에서
       무너진 실패를 반복하지 않기 위해서입니다.
       5~12%p 사이는 **구분 불가**로 적고 판정하지 않습니다 — 우리가 잰
       교란 검사 잡음이 그만큼 큽니다 (아래 상수 주석 참고).
    3. 1·2 를 모두 만족하는 후보가 여럿이면 **macro-F1 이 가장 높은 것**.

    ⚠️ 여기서 고른 건 **후보**입니다. 서브셋·짧은 에폭 결과라 절대값이
       풀 학습과 다릅니다. 확정은 풀 학습으로 다시 합니다.
    ⚠️ holdout 은 여기서 안 봅니다.
    """
    runs = [r for r in runs if r]
    if not runs:
        print("⚠️ 성공한 실행이 없습니다 — 판정할 것이 없습니다.")
        return {}

    base = next((r for r in runs if r.get("model_name") == base_model), None)
    if base is None:
        print(f"⚠️ 기준선 '{base_model}' 실행이 없어 상대 비교를 못 합니다.")
        base = max(runs, key=lambda r: r["score"])
        print(f"   가장 높은 '{base['model_name']}' 를 임시 기준선으로 씁니다.")

    # ⚠️ 기준선이 수렴하지 않았으면 이 비교 전체가 흔들립니다.
    #    "서브셋 하락률을 풀에 대입해도 되나 → 아니오 — 약한 모델은 잃을 것도 적어
    #    덜 떨어집니다" (CLAUDE.md) 와 같은 함정입니다. 덜 학습된 기준선은 배운 게
    #    적어 교란 검사에서 잃을 것도 적고, 그래서 배율 하락이 실제보다 낮게 나와
    #    다른 백본이 부당하게 나빠 보입니다. 실제로 04 첫 실행에서 resnet50 이
    #    12에폭에서도 계속 오르는 중이었고, 그 배율 하락(12.2%)이 이 프로젝트의
    #    다른 풀 학습 실행들(23.8~25.0%)과 안 맞았습니다.
    if not base.get("converged", True):
        print(f"\n  ⚠️ 기준선 '{base['model_name']}' 이 수렴하지 않았습니다 "
              "(마지막 에폭이 최고였습니다). 덜 학습된 기준선은 교란 검사에서 잃을 게 "
              "적어 배율 하락이 실제보다 낮게 나옵니다. 이 비교의 '탈락' 판정은 "
              "재확인이 필요합니다 — 기준선을 더 학습시키거나 풀 데이터로 다시 재세요.")

    print("\n" + "=" * 78)
    print(" 2단계 백본 비교 — 판정")
    print("=" * 78)
    print(f"  {'백본':<18}{'해상도':>7}{'macro-F1':>10}{'기준선대비':>11}"
          f"{'배율하락':>10}{'분':>7}   수렴")
    for r in sorted(runs, key=lambda r: -r["score"]):
        d = r["score"] - base["score"]
        drop = r.get("scale_drop")
        ds = f"{drop:>9.1%}" if drop is not None else f"{'—':>10}"
        mark = "  ← 기준선" if r is base else ""
        print(f"  {r['model_name']:<18}{r['img_size']:>6}p{r['score']:>10.4f}"
              f"{d:>+11.4f}{ds}{r['minutes']:>7.0f}   "
              f"{'수렴' if r['converged'] else '더 필요'}{mark}")

    base_drop = base.get("scale_drop")
    ok, rejected, unclear = [], [], []
    for r in runs:
        if r is base:
            continue
        gain = r["score"] - base["score"]
        drop = r.get("scale_drop")
        gap = (drop - base_drop) if (drop is not None and base_drop is not None) else None
        if gain <= F1_NOISE:
            rejected.append((r, f"macro-F1 차이 {gain:+.4f} 가 잡음(±{F1_NOISE}) 안"))
        elif gap is not None and gap > SCALE_DROP_REJECT_PP:
            rejected.append((r, f"배율 하락이 {gap:+.1%}p 나빠짐 "
                                f"(문턱 {SCALE_DROP_REJECT_PP:.0%}p)"))
        else:
            if gap is not None and gap > BLUR_NOISE_PP:
                unclear.append((r, gap))
            ok.append(r)

    print("\n" + "-" * 78)
    for r, why in rejected:
        print(f"  ✗ {r['model_name']:<18} {why}")
    # 잡음보다는 크고 탈락 문턱보다는 작은 구간 — 판정하지 말고 그대로 적습니다.
    for r, gap in unclear:
        print(f"  ◐ {r['model_name']:<18} 배율 하락 {gap:+.1%}p — 잡음(±{BLUR_NOISE_PP:.0%}p)"
              f"보다 크지만 탈락 문턱({SCALE_DROP_REJECT_PP:.0%}p)에는 못 미칩니다. "
              "구분 불가로 적고 풀 학습에서 다시 재세요.")

    verdict: dict[str, Any] = {"baseline": base, "candidates": ok, "rejected": rejected}
    if ok:
        best = max(ok, key=lambda r: r["score"])
        verdict["best"] = best
        print(f"\n  ✅ 채택 후보: **{best['model_name']}** "
              f"(macro-F1 {best['score']:.4f}, 기준선 {base['score']:.4f} 대비 "
              f"{best['score'] - base['score']:+.4f})")
    else:
        verdict["best"] = base
        print(f"\n  ◐ 기준선 **{base_model}** 유지 — 잡음 밖으로 이긴 백본이 없습니다.")
        print("     바꿀 이유가 없으면 안 바꿉니다 (비교 이력이 끊깁니다).")

    top = max(runs, key=lambda r: r["score"])
    if top is not verdict["best"]:
        print(f"\n  ⚠️ macro-F1 이 가장 높은 건 {top['model_name']} ({top['score']:.4f}) 인데")
        print("     위 기준에서 걸러졌습니다. val 최고점을 그냥 고르지 않습니다"
              " (STEP 5·6 의 실패).")

    print("\n  ⚠️ 서브셋·짧은 에폭 결과입니다. 순위가 풀 학습과 같다는 보장은 없습니다.")
    print("  ⚠️ holdout 은 아직 안 봤습니다 — 풀 학습 뒤 05 에서 한 번만 엽니다.")
    print("=" * 78)
    return verdict



# ── 1단계 고정 창 **크기** 비교 (STEP 20) ────────────────────────────────
#
# STEP 18 실측: 1단계는 병변이 **클수록** 점수가 낮습니다 (rho −0.201, 6개 클래스
# 전부 음수). 2단계(`m2.5`, 병변에 비례하는 창)에서는 그 효과가 없어서, 원인이
# "큰 병변이 어렵다" 가 아니라 **고정 창이 병변으로 꽉 차서 주변 정상 피부가
# 안 보이는 것** 쪽으로 좁혀졌습니다. 창 안 놓침 1.8% vs 창 밖 4.6% (2.5배).
#
# 전체 184,735 병변 중 **23.2%** 가 320px 창을 넘칩니다 (A6 38.7% · A2 27.9%).
# 창을 키우면 넘침은 줄지만(448 에서 12.4%) **작은 병변이 그만큼 작아집니다.**
# 맞바꿈이라 실측이 필요합니다.
#
# ⚠️ 창을 **비례**로 바꾸면 안 됩니다 — STEP 9-A 에서 ROI 크롭이 정답을 흘려
#    (정상 bbox 중앙값 0.71% vs 병변 1.25%) AUROC 가 0.8272 → 0.9477 로 갈렸습니다.
#    **고정이라는 성질을 유지한 채 크기만** 바꿉니다.

#   넘치던 병변의 놓침이 이만큼은 줄어야 "창 때문" 이 확인됩니다.
#   ⚠️ 고정값이 아니라 **그 실행에서 계산한 부트스트랩 CI 반폭**과 비교합니다
#      (evaluate.bootstrap_ci(metric="recall")). 작은 부분집합에서 잡음에
#      속지 않으려는 것입니다 — A6 가 val 256장에서 2σ ±0.085 였던 전례.
WINDOW_MIN_OVERFLOW_GAIN = "ci"
#   헛알림이 이보다 나빠지면 탈락. STEP 15 가 "2%p 이상 줄어야 채택" 을 쓴 것과
#   같은 폭을 반대 방향으로 씁니다.
WINDOW_FALSE_ALARM_TOL_PP = 0.02



# ── 2단계 크롭: 비례 창 vs 고정 창 (STEP 22) ─────────────────────────────
#
# STEP 21 실측: VL01 안에서 A4 병변이 A1 보다 **1.68배** 큽니다. A1·A4 를 둘 다
# 가진 개 72마리로 좁혀도 1.88배(Wilcoxon p=8.1e-05)라 "다른 개를 다르게 찍었다"
# 로는 설명이 안 됩니다. 그런데 `m2.5` 는 창 = 병변 × 2.5 라 병변이 언제나
# 프레임의 40% 를 차지합니다 — **절대 크기가 지워집니다.**
#
#     A1 117px → 창 292px → 384 입력에서 154px
#     A4 196px → 창 490px → 384 입력에서 154px   ← 똑같아짐
#
# 고정 창(f320/f448)은 그 크기를 남깁니다. 그럼 A4↔A1 이 덜 헷갈릴까요?
#
# ⚠️ **좋아져도 바로 채택할 수 없습니다.** 절대 크기를 쓰면 배율 교란에 약해지고,
#    배포에서 보호자의 촬영 거리는 통제되지 않습니다. 그래서 판정을 **세 갈래**로
#    둡니다 — "정보는 있으나 배율 의존" 이 나오면 그건 진단이지 채택이 아닙니다.

#   A4 recall 개선은 **그 실행에서 계산한 CI 반폭**과 비교합니다 (고정값 금지).
S2SIZE_MAIN = "ci"
#   A4→A1 혼동이 이만큼은 줄어야 "크기 신호가 실제로 쓰였다" 로 봅니다.
S2SIZE_A4A1_DROP = 0.03


def stage2_size_report(runs: list[dict], *, base_crop: str = "m2.5",
                       focus: str = "A4", other: str = "A1") -> dict[str, Any]:
    """2단계 **비례 창 vs 고정 창** 비교를 미리 정해둔 기준으로 판정합니다.

    각 run 은 `train_and_measure` 결과에 아래를 더한 dict 를 기대합니다:

        macro_f1, scale_drop           (train_and_measure 가 이미 넣습니다)
        focus_recall, focus_recall_ci  관심 클래스 recall 과 CI 반폭
        f2o_rate                       관심→비교 클래스 혼동률 (A4→A1)

    판정 (실험 **전에** 못 박음 — 규칙 2)
    -------------------------------------
    1. **1차**: `focus_recall` 이 CI 반폭보다 크게 오를 것
    2. `f2o_rate` 가 {S2SIZE_A4A1_DROP} 이상 줄 것
    3. macro-F1 이 {MACRO_F1_NOISE} 넘게 떨어지지 않을 것
    4. **배율 하락**이 {SCALE_DROP_REJECT_PP} 넘게 나빠지지 않을 것

    1·2 를 만족하고 4 가 걸리면 **"정보는 있으나 배율 의존"** — 진단으로는
    성공이지만 배포에는 못 씁니다 (보호자 촬영 거리가 통제되지 않으므로).
    """
    runs = [r for r in runs if r]
    if not runs:
        print("⚠️ 성공한 실행이 없습니다 — 판정할 것이 없습니다.")
        return {}
    base = next((r for r in runs if r.get("crop_tag") == base_crop), None)
    if base is None:
        print(f"⚠️ 기준 '{base_crop}' 실행이 없어 상대 비교를 못 합니다.")
        return {"runs": runs}

    def f(v, pct=False, nd=4):
        if v is None:
            return "   못 잼"
        return f"{v:>9.1%}" if pct else f"{v:>9.{nd}f}"

    print("\n" + "=" * 84)
    print(f" 2단계 크롭 — 비례 창(크기 지움) vs 고정 창(크기 보존)")
    print("=" * 84)
    print(f"  {'크롭':<9}{'macro-F1':>10}{focus + ' recall':>12}"
          f"{focus + '→' + other:>10}{'배율하락':>10}{'분':>6}")
    for r in sorted(runs, key=lambda r: r["crop_tag"] != base_crop):
        print(f"  {r['crop_tag']:<9}{f(r.get('macro_f1'))}{f(r.get('focus_recall'), nd=3)}"
              f"{f(r.get('f2o_rate'), pct=True)}{f(r.get('scale_drop'), pct=True)}"
              f"{r.get('minutes', 0):>6.0f}")

    verdict: dict[str, Any] = {"baseline": base_crop, "candidates": []}
    for r in runs:
        if r is base:
            continue
        d_rec = (r.get("focus_recall") or 0) - (base.get("focus_recall") or 0)
        half = r.get("focus_recall_ci") or base.get("focus_recall_ci") or 0.0
        d_f2o = (base.get("f2o_rate") or 0) - (r.get("f2o_rate") or 0)   # +면 개선
        d_f1 = (r.get("macro_f1") or 0) - (base.get("macro_f1") or 0)
        d_scale = (r.get("scale_drop") or 0) - (base.get("scale_drop") or 0)

        ok_main = d_rec > half
        ok_f2o = d_f2o >= S2SIZE_A4A1_DROP
        ok_f1 = d_f1 > -MACRO_F1_NOISE
        ok_scale = d_scale < SCALE_DROP_REJECT_PP

        print(f"\n  [{base_crop} → {r['crop_tag']}]")
        print(f"    {'[O]' if ok_main else '[X]'} 1차 — {focus} recall {d_rec:+.3f} "
              f"(CI 반폭 ±{half:.3f})")
        print(f"    {'[O]' if ok_f2o else '[X]'} {focus}→{other} 혼동 {-d_f2o:+.1%} "
              f"(문턱 −{S2SIZE_A4A1_DROP:.0%})")
        print(f"    {'[O]' if ok_f1 else '[X]'} macro-F1 {d_f1:+.4f} "
              f"(허용 −{MACRO_F1_NOISE})")
        print(f"    {'[O]' if ok_scale else '[X]'} 배율 하락 {d_scale:+.1%} "
              f"(허용 +{SCALE_DROP_REJECT_PP:.0%})")

        if ok_main and ok_f2o and ok_f1 and ok_scale:
            v = "채택 후보"
        elif ok_main and ok_f2o:
            v = "정보는 있으나 **배율 의존** — 진단 성공, 배포엔 못 씀"
        else:
            v = "기각 — 크기 신호가 A4 를 살리지 못합니다"
        print(f"    → {v}")
        verdict["candidates"].append({
            "crop_tag": r["crop_tag"], "verdict": v, "d_focus_recall": d_rec,
            "ci_half": half, "d_f2o": d_f2o, "d_macro_f1": d_f1,
            "d_scale_drop": d_scale})

    adopted = [c for c in verdict["candidates"] if c["verdict"] == "채택 후보"]
    diag = [c for c in verdict["candidates"] if "배율 의존" in c["verdict"]]
    verdict["verdict"] = ("채택 후보: " + adopted[0]["crop_tag"]) if adopted else (
        ("배율 의존: " + diag[0]["crop_tag"]) if diag else "채택 없음")
    print(f"\n  판정: {verdict['verdict']}")
    print("  ⚠️ VL01 만 · 백본은 빠른 effnetv2_s 입니다. 효과가 크면 확정 백본"
          "(convnextv2_base)으로 다시 확인해야 합니다.")
    print("  ⚠️ 고정 창은 배포에서 **촬영 거리에 의존**합니다 — 채택 전에"
          " 그 위험을 따로 판단하세요.")
    print("=" * 84)
    return verdict


def stage1_window_report(runs: list[dict], *, base_crop: str = "f320") -> dict[str, Any]:
    """1단계 **고정 창 크기** 비교를 미리 정해둔 기준으로 판정합니다.

    각 run 은 `train_and_measure` 결과에 아래를 더한 dict 를 기대합니다:

        auroc                 전체 AUROC
        blur_drop             흐림 교란 하락 (화질 지름길 감시)
        false_alarm           recall 0.95 에서의 헛알림률
        overflow_miss         **기준 창을 넘치던 병변**의 놓침률
        overflow_miss_ci      그 값의 부트스트랩 CI 반폭
        within_miss           넘치지 않던 병변의 놓침률

    ⚠️ `overflow` 는 **기준 창(f320)** 기준으로 정의합니다. 창을 키우면 정의가
       같이 움직이면 안 됩니다 — 같은 사진 집합을 두 모델이 어떻게 다루는지를
       봐야 하니까요.

    판정 (실험 **전에** 못 박음 — 규칙 2)
    -------------------------------------
    1. **1차**: `overflow_miss` 가 CI 반폭보다 크게 줄어야 합니다.
       이게 가설의 핵심입니다. 점수보다 이게 먼저입니다.
    2. AUROC 가 {AUROC_NOISE} 넘게 떨어지면 탈락.
    3. 흐림 하락이 {BLUR_NOISE_PP} 넘게 나빠지면 탈락 (화질 지름길 재개방).
    4. 헛알림이 {WINDOW_FALSE_ALARM_TOL_PP} 넘게 나빠지면 탈락.

    1번만 만족하고 2~4 중 하나가 걸리면 **맞바꿈**으로 적고 채택하지 않습니다.
    """
    runs = [r for r in runs if r]
    if not runs:
        print("⚠️ 성공한 실행이 없습니다 — 판정할 것이 없습니다.")
        return {}
    base = next((r for r in runs if r.get("crop_tag") == base_crop), None)
    if base is None:
        print(f"⚠️ 기준 '{base_crop}' 실행이 없어 상대 비교를 못 합니다.")
        return {"runs": runs}

    def f(v, pct=False, nd=4):
        if v is None:
            return "   못 잼"
        return f"{v:>9.1%}" if pct else f"{v:>9.{nd}f}"

    print("\n" + "=" * 82)
    print(" 1단계 고정 창 **크기** 비교 — 큰 병변이 창을 넘치는 문제")
    print("=" * 82)
    print(f"  {'크롭':<9}{'AUROC':>9}{'넘친것 놓침':>12}{'안넘친것':>10}"
          f"{'헛알림':>9}{'흐림하락':>10}{'분':>6}")
    for r in sorted(runs, key=lambda r: r["crop_tag"] != base_crop):
        print(f"  {r['crop_tag']:<9}{f(r.get('auroc'))}{f(r.get('overflow_miss'), pct=True)}"
              f"{f(r.get('within_miss'), pct=True)}{f(r.get('false_alarm'), pct=True)}"
              f"{f(r.get('blur_drop'), pct=True)}{r.get('minutes', 0):>6.0f}")
    print(f"\n  ⚠️ '넘친것' 은 **{base_crop} 기준**으로 정의합니다 (창을 키워도 같은 사진 집합).")

    if not base.get("converged", True):
        print(f"  ⚠️ 기준 '{base_crop}' 이 수렴하지 않았습니다 — 재확인 필요"
              " (STEP 9 와 같은 함정).")

    verdict: dict[str, Any] = {"baseline": base_crop, "candidates": []}
    for r in runs:
        if r is base:
            continue
        d_of = base.get("overflow_miss", 0) - r.get("overflow_miss", 0)   # +면 개선
        half = r.get("overflow_miss_ci") or base.get("overflow_miss_ci") or 0.0
        d_auroc = (r.get("auroc") or 0) - (base.get("auroc") or 0)
        d_blur = ((r.get("blur_drop") or 0) - (base.get("blur_drop") or 0))
        d_fa = ((r.get("false_alarm") or 0) - (base.get("false_alarm") or 0))

        ok_main = d_of > half
        ok_auroc = d_auroc > -AUROC_NOISE
        ok_blur = d_blur < BLUR_NOISE_PP
        ok_fa = d_fa < WINDOW_FALSE_ALARM_TOL_PP

        print(f"\n  [{base_crop} → {r['crop_tag']}]")
        print(f"    {'[O]' if ok_main else '[X]'} 1차 — 넘친 병변 놓침 {d_of:+.1%} "
              f"(CI 반폭 ±{half:.1%})")
        print(f"    {'[O]' if ok_auroc else '[X]'} AUROC {d_auroc:+.4f} "
              f"(허용 −{AUROC_NOISE})")
        print(f"    {'[O]' if ok_blur else '[X]'} 흐림 하락 {d_blur:+.1%} "
              f"(허용 +{BLUR_NOISE_PP:.0%})")
        print(f"    {'[O]' if ok_fa else '[X]'} 헛알림 {d_fa:+.1%} "
              f"(허용 +{WINDOW_FALSE_ALARM_TOL_PP:.0%})")

        if ok_main and ok_auroc and ok_blur and ok_fa:
            v = "채택"
        elif ok_main:
            v = "맞바꿈 — 넘친 병변은 좋아지나 다른 데서 잃습니다"
        else:
            v = "기각 — 창 크기가 원인이 아닙니다"
        print(f"    → {v}")
        verdict["candidates"].append({"crop_tag": r["crop_tag"], "verdict": v,
                                      "d_overflow_miss": d_of, "ci_half": half,
                                      "d_auroc": d_auroc, "d_blur": d_blur,
                                      "d_false_alarm": d_fa})

    adopted = [c for c in verdict["candidates"] if c["verdict"] == "채택"]
    verdict["verdict"] = ("채택: " + adopted[0]["crop_tag"]) if adopted else "채택 없음"
    print(f"\n  판정: {verdict['verdict']}")
    print("  ⚠️ VL01 만으로 낸 결과라면 전체 데이터에서 다시 확인해야 합니다"
          " (서브셋 순위가 풀 학습과 같다는 보장은 없습니다).")
    print("=" * 82)
    return verdict


def stage1_crop_report(runs: list[dict], *, base_crop: str = "full") -> dict[str, Any]:
    """1단계 입력(크롭 태그) 비교를 **미리 정해둔 기준**으로 판정합니다.

    STEP 8 에서 남은 위험: 1단계가 `full`(강아지 전신)로 학습했는데, 배포에서
    보호자는 촬영 가이드대로 병변에 다가가서 찍습니다. `f320`(고정 픽셀 창)은
    창 크기가 병변 크기와 무관해 창 크기 지름길이 없고, 구도도 근접 사진에
    가깝습니다. 여기서는 **모델·증강은 STEP 6·7 이 정한 대로 고정**하고
    (`effnetv2_s` + `photometric`) 입력만 바꿔 비교합니다.

    판정 기준 (실험 **전에** 못 박음 — 규칙 2)
    -------------------------------------------
    1. AUROC 가 {AUROC_NOISE} 이상 오르거나, 최소한 안 떨어져야 후보입니다.
       (holdout 변별력 부족이 STEP 8 의 핵심 문제라 AUROC 를 우선합니다)
    2. 흐림 하락이 {BLUR_NOISE_PP} 넘게 나빠지면 탈락합니다 — f320 이 화질
       지름길을 다시 열면 안 됩니다.
    3. 스크리닝 recall 이 얼마나 나오는지는 여기서 안 봅니다. 임계값은
       val 로 다시 잡아야 하는 값이라, 이 비교 단계에서는 신호가 아닙니다.

    ⚠️ 여기서 고른 건 **후보**입니다. holdout 은 안 봅니다 — 풀 학습 뒤 05 에서
    한 번만 엽니다.
    """
    runs = [r for r in runs if r]
    if not runs:
        print("⚠️ 성공한 실행이 없습니다 — 판정할 것이 없습니다.")
        return {}

    base = next((r for r in runs if r.get("crop_tag") == base_crop), None)
    if base is None:
        print(f"⚠️ 기준 '{base_crop}' 실행이 없어 상대 비교를 못 합니다.")
        return {"runs": runs}

    def fmt(v, pct=False, nd=4):
        if v is None:
            return "   못 잼"
        return f"{v:>8.1%}" if pct else f"{v:>8.{nd}f}"

    print("\n" + "=" * 74)
    print(" 1단계 입력 비교 — full vs f320")
    print("=" * 74)
    print(f"  {'크롭':<12}{'AUROC':>10}{'흐림하락':>10}{'precision':>11}{'분':>6}   수렴")
    for r in sorted(runs, key=lambda r: r["crop_tag"] != base_crop):
        print(f"  {r['crop_tag']:<12}{fmt(r.get('auroc'))}{fmt(r.get('blur_drop'), pct=True)}"
              f"{fmt(r.get('precision'), nd=3)}{r['minutes']:>6.0f}   "
              f"{'수렴' if r['converged'] else '더 필요'}")

    if not base.get("converged", True):
        print(f"\n  ⚠️ 기준 '{base_crop}' 이 수렴하지 않았습니다. 이 비교는 재확인이 "
              "필요합니다 (STEP 9 의 2단계 백본 비교와 같은 함정).")

    verdict: dict[str, Any] = {"baseline": base}
    print(f"\n  기준: {base_crop}  (AUROC {base['auroc']:.4f}, "
          f"흐림 하락 {fmt(base.get('blur_drop'), pct=True).strip()})")
    print(f"  잡음 폭: AUROC ±{AUROC_NOISE} · 흐림 하락 ±{BLUR_NOISE_PP:.0%}")

    candidates = []
    for r in runs:
        if r is base:
            continue
        d_auroc = r["auroc"] - base["auroc"]
        d_blur = (r["blur_drop"] - base["blur_drop"]
                  if r.get("blur_drop") is not None and base.get("blur_drop") is not None
                  else None)
        print(f"\n  [{base_crop} → {r['crop_tag']}]")
        print(f"    AUROC     {d_auroc:+.4f}")
        print(f"    흐림 하락  {'못 잼' if d_blur is None else f'{d_blur:+.1%}'}")
        worse_blur = d_blur is not None and d_blur > BLUR_NOISE_PP
        if d_auroc >= -AUROC_NOISE and not worse_blur:
            candidates.append(r)
            print(f"    ✅ {r['crop_tag']} 후보 — AUROC 를 깎지 않으면서 화질 의존을 "
                  "늘리지 않았습니다.")
        elif d_auroc < -AUROC_NOISE:
            print(f"    ❌ 탈락 — AUROC 가 잡음 밖으로 떨어졌습니다.")
        else:
            print(f"    ❌ 탈락 — 흐림 하락이 {d_blur:+.1%}p 나빠졌습니다.")

    if candidates:
        best = max(candidates, key=lambda r: r["auroc"])
        verdict["best"] = best
        print(f"\n  ✅ 채택 후보: **{best['crop_tag']}** (AUROC {best['auroc']:.4f}, "
              f"기준 {base['auroc']:.4f} 대비 {best['auroc'] - base['auroc']:+.4f})")
    else:
        verdict["best"] = base
        print(f"\n  ◐ 기준 **{base_crop}** 유지 — 후보가 없습니다.")

    print("\n  ⚠️ 서브셋·짧은 에폭 결과입니다. 순위가 풀 학습과 같다는 보장은 없습니다.")
    print("  ⚠️ holdout 은 아직 안 봤습니다 — 풀 학습 뒤 05 에서 한 번만 엽니다.")
    print("=" * 74)
    return verdict


# ── 1단계 놓침이 병변 탓인가 창 배치 탓인가 (STEP 24) ────────────────────
#
# 배경: rho(병변 크기, 1단계 점수) = -0.201 로 **큰 병변을 더 놓칩니다**
# (STEP 17·18). STEP 20 이 "창이 작아서" 를 기각했고(f448 로 넓히니 오히려
# 악화), 남은 설명이 둘이었습니다:
#
#   H-A 병변    큰 병변이 원래 어렵다 — 창이 병변으로만 꽉 차 경계가 안 보인다
#   H-B' 창배치  네모 중심이 병변을 벗어난다 — 네모는 폴리곤의 **외접사각형**이라
#                길쭉·굽은 윤곽에서 중심이 빈 곳에 놓인다 (우리 파이프라인 탓)
#
# 두 가설은 같은 값(`occ` = 창 안 병변 점유율)에 **정반대**를 예측합니다.
# 그래서 결과를 보고 기준을 고를 수 없게 여기 먼저 박아 둡니다 (작업 규칙 2).
#
# ⚠️ 이 판정은 **관찰**입니다. H-B' 가 지지돼도 "창을 옮기면 낫다" 는 아직
#    아닙니다 — 그건 폴리곤 무게중심으로 다시 잘라 재는 **개입 실험**이 말합니다
#    (`PLACEMENT_RECOVER_MIN` 아래).

PLACEMENT_BINS = 5           # occ 를 몇 층으로 나눠 볼 것인가
PLACEMENT_RATIO_HB = 2.0     # 최저층 놓침률 ÷ 최고층 ≥ 이 값이면 H-B'
PLACEMENT_RATIO_HA = 0.5     # 그 비가 이 값 이하면 H-A (놓침이 꽉 찬 쪽에 몰림)
PLACEMENT_RHO_RESIDUAL = 0.08   # 층 안에서 크기 상관이 이보다 작으면 "크기 효과는 배치로 설명됨"
PLACEMENT_MIN_MISS = 40      # 놓침이 이보다 적으면 **판정하지 않습니다**

# ⚠️ **이 잠금장치는 제가 실제로 밟은 함정에서 나왔습니다** (2026-09-06).
#    위 H-A/H-B' 판정을 `occ`(창 안 병변 점유율)로 설계했는데, 막상 재 보니
#    `rho(occ, 폴리곤 √넓이) = +0.995` 였습니다 — **occ 는 병변 크기의 다른
#    이름**입니다. 그러면 "occ 가 높을수록 놓친다" 는 "크면 놓친다" 의 재진술이지
#    두 가설을 가른 게 아닙니다. 그런데 코드는 아무 말 없이 "H-A" 를 찍었습니다.
#    가르는 변수가 가르려는 원인과 붙어 있으면 **판정 자체가 성립하지 않습니다.**
PLACEMENT_MAX_COLLINEARITY = 0.80   # |rho(층 변수, 크기)| 가 이 이상이면 판정 거부

# 개입 실험(폴리곤 무게중심으로 창을 옮김) 채택 기준
PLACEMENT_RECOVER_MIN = 0.30  # 놓쳤던 것 중 이만큼 되살아나야 채택 후보
PLACEMENT_LOSS_TOL = 0.01     # 이미 맞히던 것을 이보다 많이 잃으면 기각


def stage1_placement_report(rows, *, bins: int = PLACEMENT_BINS,
                            boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """`occ`(창 안 병변 점유율) 층별 놓침률로 H-A / H-B' 를 가릅니다.

    `rows` 에 필요한 열: ``occ`` · ``miss``(bool) · ``box_px`` · ``score``.

    돌려주는 값의 ``verdict`` 는 셋 중 하나입니다:
      "창 배치(H-B')" · "병변 자체(H-A)" · "구분 불가"

    ⚠️ **`occ` 가 크기와 붙어 있으면 판정을 거부합니다** — 그게 실제로 벌어졌고
       (rho +0.995) 코드가 조용히 "H-A" 를 찍었습니다. `PLACEMENT_MAX_COLLINEARITY`.

    ⚠️ 놓침이 `PLACEMENT_MIN_MISS` 미만이면 무조건 "표본 부족" 입니다.
       VL01 val 은 놓침이 77건뿐이라 층마다 15건꼴입니다 — 부트스트랩 CI 를
       같이 내고, **CI 가 겹치면 비가 아무리 커도 판정하지 않습니다.**
    """
    import numpy as np
    import pandas as pd

    d = pd.DataFrame(rows).dropna(subset=["occ", "miss"]).reset_index(drop=True)
    n_miss = int(d["miss"].sum())
    out: dict[str, Any] = {"n": len(d), "n_miss": n_miss, "bins": [],
                           "verdict": "표본 부족", "why": ""}
    # 층을 가르는 변수가 크기와 붙어 있으면 이 검사는 아무것도 못 가릅니다
    if "box_px" in d:
        from scipy.stats import spearmanr
        col = float(spearmanr(d["occ"], d["box_px"]).statistic)
        out["collinearity"] = col
        if abs(col) >= PLACEMENT_MAX_COLLINEARITY:
            out["verdict"] = "판정 불가(공선성)"
            out["why"] = (
                f"층 변수 occ 가 크기와 rho={col:+.3f} 로 붙어 있습니다 "
                f"(>= {PLACEMENT_MAX_COLLINEARITY}). occ 로 나눈 층은 크기로 나눈 "
                f"층과 같아서, 어떤 결과가 나와도 '크면 놓친다' 의 재진술입니다. "
                f"크기와 분리되는 **모양** 변수(slack · center_off · 길쭉함)로 "
                f"크기를 고정한 채 재세요.")
            print(f"[placement] {out['why']}")
            return out

    if n_miss < PLACEMENT_MIN_MISS:
        out["why"] = (f"놓침 {n_miss}건 < {PLACEMENT_MIN_MISS}건 — 판정하지 않습니다. "
                      f"층마다 {n_miss / bins:.0f}건꼴이라 어떤 비도 잡음입니다.")
        print(f"[placement] {out['why']}")
        return out

    d["층"] = pd.qcut(d["occ"], bins, labels=False, duplicates="drop")
    rng = np.random.default_rng(seed)
    for b in sorted(d["층"].dropna().unique()):
        g = d[d["층"] == b]
        m = g["miss"].to_numpy().astype(float)
        bs = np.array([rng.choice(m, len(m), replace=True).mean() for _ in range(boot)])
        out["bins"].append({
            "층": int(b), "n": len(g), "occ_중앙값": float(g["occ"].median()),
            "놓침률": float(m.mean()),
            "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "box_px_중앙값": float(g["box_px"].median()) if "box_px" in g else float("nan"),
        })

    lo, hi = out["bins"][0], out["bins"][-1]     # occ 최저층 / 최고층
    ratio = lo["놓침률"] / hi["놓침률"] if hi["놓침률"] > 0 else float("inf")
    overlap = not (lo["ci"][0] > hi["ci"][1] or hi["ci"][0] > lo["ci"][1])
    out["ratio_lowest_over_highest"] = ratio
    out["ci_overlap"] = overlap

    # 크기 효과가 배치로 설명되는가 — 층 **안**에서 상관이 남는지
    resid = []
    if "score" in d and "box_px" in d:
        from scipy.stats import spearmanr
        for b in sorted(d["층"].dropna().unique()):
            g = d[d["층"] == b].dropna(subset=["box_px", "score"])
            if len(g) >= 50:
                resid.append(float(spearmanr(g["box_px"], g["score"]).statistic))
    out["rho_within_bins"] = resid
    out["rho_within_max_abs"] = max((abs(r) for r in resid), default=float("nan"))
    out["size_effect_explained"] = bool(
        resid and out["rho_within_max_abs"] < PLACEMENT_RHO_RESIDUAL)

    if overlap:
        out["verdict"] = "구분 불가"
        out["why"] = (f"최저층 {lo['놓침률']:.1%} {lo['ci']} vs 최고층 "
                      f"{hi['놓침률']:.1%} {hi['ci']} — CI 가 겹칩니다 (비 {ratio:.2f}).")
    elif ratio >= PLACEMENT_RATIO_HB:
        out["verdict"] = "창 배치(H-B')"
        out["why"] = (f"병변이 적게 담긴 창에서 {ratio:.2f}배 더 놓칩니다 "
                      f"({lo['놓침률']:.1%} vs {hi['놓침률']:.1%}, CI 안 겹침).")
    elif ratio <= PLACEMENT_RATIO_HA:
        out["verdict"] = "병변 자체(H-A)"
        out["why"] = (f"창이 병변으로 꽉 찰수록 더 놓칩니다 (비 {ratio:.2f}) — "
                      f"경계가 안 보이는 쪽이 어렵다는 뜻입니다.")
    else:
        out["verdict"] = "구분 불가"
        out["why"] = (f"비 {ratio:.2f} 가 {PLACEMENT_RATIO_HA}~{PLACEMENT_RATIO_HB} "
                      f"사이입니다 — 어느 쪽도 아닙니다.")

    print(f"\n[placement] n={out['n']:,} · 놓침 {n_miss}건")
    print(f"{'층':>3} {'n':>6} {'occ 중앙':>9} {'네모px':>7} {'놓침률':>8}   95% CI")
    for b in out["bins"]:
        print(f"{b['층']:>3} {b['n']:>6,} {b['occ_중앙값']:>9.3f} "
              f"{b['box_px_중앙값']:>7.0f} {b['놓침률']:>8.1%}   "
              f"[{b['ci'][0]:.1%}, {b['ci'][1]:.1%}]")
    if resid:
        print(f"층 안 rho(네모 크기, 점수): "
              f"{', '.join(f'{r:+.3f}' for r in resid)}  "
              f"→ 크기 효과가 배치로 설명되나: "
              f"{'예' if out['size_effect_explained'] else '아니오'}")
    print(f"판정: {out['verdict']} — {out['why']}")
    return out


# ⚠️ **아래는 사전등록이 아니라 탐색적(exploratory) 분석입니다.** 위 `occ` 판정이
#    공선성으로 무너진 뒤에 만들었으므로, 문턱을 결과를 보고 고른 것으로
#    의심해야 합니다. 다만 결론이 **"신호 없음"** 쪽이라 문턱을 낮춰 잡을수록
#    불리해집니다 — 그 방향의 자기기만은 어렵습니다. 그래도 확증으로 쓰지 말고
#    **H-B'(창 배치)를 기각하는 근거**로만 쓰세요.
SHAPE_RHO_MIN = 0.10          # 크기 층 안에서 이만큼은 돼야 "신호" (크기 자체는 0.20)
SHAPE_MIN_CONSISTENT = 3      # 4개 층 중 몇 개가 같은 방향이어야 하나


def stage1_shape_report(rows, *, size_col: str = "box_px",
                        shape_cols=("slack", "center_off", "ecc"),
                        bins: int = 4) -> dict[str, Any]:
    """**크기를 고정한 채** 모양이 1단계 놓침을 예측하는가.

    `occ` 판정이 못 하는 일을 합니다 — 모양 변수들은 크기와 거의 안 붙어 있어
    (rho 0.10~0.26) 크기 층 안에서 따로 움직일 수 있습니다.

    ``slack``       1 - 폴리곤/네모 넓이. 네모로 줄이며 생긴 빈 곳
    ``center_off``  네모 중심 ↔ 폴리곤 무게중심 거리 / 네모 긴 변
    ``ecc``         네모 긴 변 / 짧은 변 (길쭉함)

    판정: 어느 모양 변수든 4개 크기 층 중 `SHAPE_MIN_CONSISTENT` 개 이상에서
    같은 방향이고 |rho| >= `SHAPE_RHO_MIN` 이면 "모양도 관여".
    아니면 **"크기가 전부"** — 즉 창 배치 가설(H-B') 기각.
    """
    import pandas as pd
    from scipy.stats import spearmanr

    d = pd.DataFrame(rows).dropna(subset=[size_col, "score"]).reset_index(drop=True)
    d["_q"] = pd.qcut(d[size_col], bins, labels=False, duplicates="drop")
    out: dict[str, Any] = {"n": len(d), "size_col": size_col, "shape": {}}

    print(f"\n[shape] 크기({size_col}) {bins}층 안에서 모양이 점수를 예측하나  n={len(d):,}")
    print(f"{'변수':<12}" + "".join(f"{'층'+str(i):>9}" for i in range(bins))
          + f"{'같은방향':>9}{'판정':>10}")
    hit = []
    for c in shape_cols:
        if c not in d:
            continue
        g = d.dropna(subset=[c])
        rs = [float(spearmanr(x[c], x["score"]).statistic)
              for _, x in g.groupby("_q", observed=True) if len(x) >= 50]
        pos, neg = sum(r > 0 for r in rs), sum(r < 0 for r in rs)
        same = max(pos, neg)
        strong = sum(abs(r) >= SHAPE_RHO_MIN for r in rs)
        ok = same >= SHAPE_MIN_CONSISTENT and strong >= SHAPE_MIN_CONSISTENT
        hit.append(ok)
        out["shape"][c] = {"rho_by_size_bin": rs, "same_direction": same,
                           "strong": strong, "signal": ok}
        print(f"{c:<12}" + "".join(f"{r:>+9.3f}" for r in rs)
              + f"{same:>9}" + f"{'신호' if ok else '없음':>10}")

    # 대조군 — 크기 자체는 같은 자리에서 얼마나 강한가
    rs_size = [float(spearmanr(x[size_col], x["score"]).statistic)
               for _, x in d.groupby("_q", observed=True) if len(x) >= 50]
    out["rho_size_within_own_bins"] = rs_size
    print(f"{'(대조)크기':<12}" + "".join(f"{r:>+9.3f}" for r in rs_size)
          + "   ← 층 안에서도 크기는 남습니다" if rs_size else "")

    out["verdict"] = "모양도 관여" if any(hit) else "크기가 전부 (창 배치 H-B' 기각)"
    out["why"] = (
        f"모양 변수 {len(hit)}개 중 신호 {sum(hit)}개. "
        f"{SHAPE_MIN_CONSISTENT}개 층 이상에서 |rho| >= {SHAPE_RHO_MIN} 이고 방향이 "
        f"같아야 신호로 셉니다.")
    print(f"판정: {out['verdict']} — {out['why']}")
    return out


def estimate_runtime(model_names: list[str] | list[tuple[str, int]], img_size: int,
                     n_train: int, epochs: int, n_conditions: int | None = None,
                     device: str | None = None) -> dict[str, Any]:
    """학습을 시작하기 **전에** 총 예상 시간을 찍습니다.

    "몇 시간 걸릴지 모르고 돌렸다가 뒤통수" 를 여러 번 맞아서 넣었습니다.
    합성 텐서로 GPU 속도만 재므로 백본당 20초 안쪽입니다.

    `model_names` 는 이름 목록이거나 **(이름, 해상도) 목록**입니다. 뒤엣것을 쓰면
    백본마다 다른 해상도로 잽니다 — ViT 계열은 해상도가 고정이라 CNN 과 같은
    384 로 재면 안 됩니다 (판 B 가 그 경우입니다).

    ⚠️ 데이터 로딩이 병목이면 실제는 이보다 느립니다. **하한 추정**입니다.
    """
    from src import bench
    from src.config import CFG, MODEL_BY_KEY

    pairs = [(m, img_size) if isinstance(m, str) else (m[0], m[1]) for m in model_names]
    n_conditions = n_conditions or len(pairs)
    rows, total_min = [], 0.0
    print("\n" + "=" * 66)
    print(" 시작 전 시간 추정 (GPU 속도 실측, 백본당 ~20초)")
    print("=" * 66)
    for key, size in pairs:
        spec = MODEL_BY_KEY[key]
        cfg = CFG(model_name=spec.timm_name, img_size=size)
        # ⚠️ 하나가 터져도 나머지 추정은 보여줘야 합니다. 추정하다 죽으면
        #    정작 돌 수 있는 백본들의 시간도 못 보고 셀이 멈춥니다.
        try:
            g = bench.gpu_speed(cfg, n_classes=2, steps=20)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  {key:<16} 속도 측정 실패 ({type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:60]}) — 추정 생략")
            continue
        ips = float(g.get("img_per_sec") or 0.0)
        # GPU 가 없으면 img_per_sec 이 NaN 입니다 (NaN 은 비교가 전부 False 라 따로 봅니다)
        if not (ips > 0) or ips != ips:
            print(f"  {key:<16} 속도를 못 쟀습니다 ({g.get('note', '이유 불명')}) — 추정 생략")
            continue
        epoch_min = n_train / ips / 60
        run_min = epoch_min * epochs
        rows.append({"model": key, "img_size": size, "img_per_sec": ips,
                     "batch": g.get("batch"),
                     "peak_vram_gb": g.get("peak_vram_gb"),
                     "epoch_min": epoch_min, "run_min": run_min})
        total_min += run_min
        print(f"  {key:<16}{size:>4}px{ips:>7.0f} img/s  배치 {g.get('batch', '?'):>3}  "
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
