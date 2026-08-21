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
    fold: int = 0,
    n_robust: int = 3000,
    measure_robust: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """한 설정으로 학습하고 **점수와 견고성을 함께** 돌려줍니다.

    stage=1 이면 정상/이상(이진), stage=2 면 병변 6종입니다.

    반환하는 dict 는 `resolution_report()` 가 그대로 먹습니다.
    모델은 다 쓰고 나면 버립니다 (해상도를 올리면 VRAM 이 빠듯합니다).
    """
    from src import data, evaluate, models, robust, split, stages, train
    from src.config import CLASSES, CLASSES_STAGE1, CFG, with_finetune

    classes = CLASSES_STAGE1 if stage == 1 else CLASSES
    cfg = with_finetune(
        CFG(model_name=model_name, img_size=img_size, epochs=epochs,
            # 1단계는 정상:이상이 5:5 라 가중치가 필요 없습니다.
            balance_strategy="none" if stage == 1 else "class_weight",
            monitor="macro_f1",
            # ⚠️ 해상도를 이름에 넣습니다 — 안 넣으면 224 체크포인트를 384 학습이
            #    "이미 끝난 학습" 으로 착각하고 건너뜁니다.
            exp_name=f"stage{stage}_{model_name}_{crop_tag}_{img_size}"),
        finetune)

    tr, va = split.get_fold(view, fold)
    if verbose:
        print(f"\n{'━' * 66}")
        print(f"  {stage}단계 @ {img_size}px  (크롭 '{crop_tag}', {epochs}에폭, "
              f"배치 {cfg.resolved_batch_size()})")
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
        "stage": stage, "img_size": img_size, "crop_tag": crop_tag,
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


def resolution_report(runs: list[dict], *, baseline_size: int = 224) -> dict[str, Any]:
    """해상도 비교 표를 찍고 **채택 여부까지** 판정합니다.

    판정 기준을 실험 **전에** 못 박아 둡니다 (결과를 보고 기준을 고르면
    무슨 숫자가 나와도 성공담을 쓸 수 있습니다):

        · 배율 하락이 잡음(±3%p)보다 크게 줄었다  → 채택
        · 점수는 올랐는데 하락폭이 그대로다        → 보류, 배포 설계로 넘김
        · 둘 다 그대로다                          → 해상도 문제가 아님
    """
    runs = [r for r in runs if r]
    if not runs:
        return {}

    by_stage: dict[int, list[dict]] = {}
    for r in runs:
        by_stage.setdefault(r["stage"], []).append(r)

    verdicts: dict[str, Any] = {}
    for stage in sorted(by_stage):
        rs = sorted(by_stage[stage], key=lambda r: r["img_size"])
        name = rs[0]["score_name"]
        print(f"\n{'=' * 68}\n {stage}단계 — 해상도 비교\n{'=' * 68}")
        print(f"  {'해상도':<9}{name:>10}{'배율하락':>10}{'배치':>7}{'분':>8}   수렴")
        for r in rs:
            drop = r.get("scale_drop")
            ds = f"{drop:>9.1%}" if drop is not None else f"{'—':>10}"
            print(f"  {str(r['img_size']) + 'px':<9}{r['score']:>10.4f}{ds}"
                  f"{r['batch_size']:>7}{r['minutes']:>8.0f}   "
                  f"{'✅' if r['converged'] else '📈 더 필요'}")

        base = next((r for r in rs if r["img_size"] == baseline_size), rs[0])
        best = max(rs, key=lambda r: r["score"])
        v: dict[str, Any] = {"baseline": base, "best_score": best}

        # ── 판정 ────────────────────────────────────────────────
        others = [r for r in rs if r is not base]
        if not others:
            print("\n  (비교 대상이 없어 판정을 생략합니다)")
            verdicts[f"stage{stage}"] = v
            continue

        hi = max(others, key=lambda r: r["img_size"])
        d_score = hi["score"] - base["score"]
        print(f"\n  {name}  {base['score']:.4f} → {hi['score']:.4f}  ({d_score:+.4f})")

        bd, hd = base.get("scale_drop"), hi.get("scale_drop")
        if bd is not None and hd is not None:
            d_drop = hd - bd
            print(f"  배율 하락  {bd:.1%} → {hd:.1%}  ({d_drop:+.1%})")
            v["drop_delta"] = d_drop

            if d_drop < -NOISE_PP and hd <= DROP_WANT:
                v["verdict"] = "adopt"
                print(f"\n  ✅ 채택 — 하락이 잡음(±{NOISE_PP:.0%})보다 크게 줄었고 "
                      f"목표({DROP_WANT:.0%}) 안에 들어왔습니다.")
                print(f"     노트북의 IMG_SIZE 를 {hi['img_size']} 로 바꾸세요.")
            elif d_drop < -NOISE_PP:
                v["verdict"] = "improved"
                print(f"\n  🤔 개선됐지만 아직 {hd:.0%} 입니다 (목표 {DROP_WANT:.0%}).")
                print("     방향은 맞습니다 — 해상도를 더 올리거나 배포 설계로 보완하세요.")
            else:
                v["verdict"] = "no_effect"
                print(f"\n  ❌ 하락폭이 그대로입니다 (잡음 ±{NOISE_PP:.0%} 안).")
                print("     해상도로도 안 잡힙니다. 남은 건 모델링이 아니라 배포 설계입니다:")
                print('     "병변이 화면 절반 이상 차지하게 찍어주세요" 로 입력을 제한하고,')
                print("     full 크롭 점수를 정직한 숫자로 보고하는 쪽.")
        else:
            v["verdict"] = "unmeasured"
            print("\n  ⚠️ 배율 하락을 안 재서 판정할 수 없습니다.")

        if not hi["converged"]:
            print(f"\n  📈 {hi['img_size']}px 는 마지막 에폭이 최고였습니다 — "
                  f"덜 학습됐습니다. 에폭을 {int(hi['epochs'] * 1.6)} 로 올리면 더 오를 수 있습니다.")
        verdicts[f"stage{stage}"] = v

    return verdicts
