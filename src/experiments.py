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


def augmentation_report(runs: list[dict], *, baseline: str = "default") -> dict[str, Any]:
    """증강 프리셋 비교.

    ⚠️ **점수가 아니라 배율 하락으로 고릅니다.** 검증 macro-F1 은 "정답 박스로
    잘라준 사진" 점수라서, 증강을 세게 걸면 대개 조금 내려갑니다. 그래도
    하락폭이 크게 줄면 배포에는 그쪽이 낫습니다.
    """
    return _compare(runs, "aug", baseline, "증강 비교", fmt=str)
