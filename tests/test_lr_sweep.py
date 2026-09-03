"""학습률 스윕(STEP 17)이 **조용히 틀리지 않는가**.

    uv run python tests/test_lr_sweep.py

두 가지를 못 박습니다.

1. **네 판이 서로 다른 실험 이름을 쓰는가.**
   안 그러면 `train.fit` 이 뒤의 세 판을 "이미 끝난 학습" 으로 건너뛰고,
   표는 그럴듯하게 나오는데 사실은 **같은 모델 네 번**입니다. 에러도 안 납니다.
   해상도(224 체크포인트를 384 가 재사용)와 샘플러에서 이미 두 번 당했습니다.

2. **판정 규칙이 결과를 보고 바뀌지 않는가.**
   1차 기준은 점수가 아니라 `best_epoch` 입니다 — 묻는 것이 "어느 판이 높나"
   가 아니라 "학습이 되긴 하나" 이기 때문입니다.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAILS: list[str] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {extra}" if extra else ""))
    if not cond:
        FAILS.append(f"{name} {extra}".strip())


def _quiet(fn, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*a, **kw)
    return out, buf.getvalue()


# ──────────────────────────────────────────────────────────────
# 1. 실험 이름 — 네 판이 갈라지는가
# ──────────────────────────────────────────────────────────────
def test_each_plan_gets_its_own_experiment_name():
    print("\n[이름] 학습률이 다르면 실험 이름도 다른가")
    import numpy as np
    import pandas as pd

    from src import data, evaluate, experiments, models, split, train

    seen: list = []

    class _Res:
        best_epoch, history = 4, [{}] * 12

    class _Rep:
        metrics = {"macro_f1": 0.5, "per_class": {"recall": [0.5] * 6}}

    orig = (split.get_fold, models.build, data.build_loaders,
            train.print_status, train.fit, train.cached_logits,
            evaluate.full_report)

    rows = pd.DataFrame({"label": ["A1"] * 12, "crop_path": ["x"] * 12})
    split.get_fold = lambda v, f: (rows, rows)
    models.build = lambda *a, **k: object()
    data.build_loaders = lambda *a, **k: (None, None, None, None)
    train.print_status = lambda *a, **k: None
    train.cached_logits = lambda *a, **k: (np.zeros((12, 6)), np.zeros(12, int))
    evaluate.full_report = lambda *a, **k: _Rep()

    def fake_fit(model, dl_tr, dl_va, cfg, **kw):
        seen.append(cfg)
        return _Res()

    train.fit = fake_fit
    try:
        plans = [
            {},                                                        # 기준선
            {"lr": 1e-4, "backbone_lr_mult": 0.3, "warmup_epochs": 2},
            {"lr": 3e-4, "backbone_lr_mult": 0.1, "warmup_epochs": 2},
            {"lr": 1e-4, "backbone_lr_mult": 0.1, "warmup_epochs": 1},
        ]
        outs = []
        for p in plans:
            o, _ = _quiet(experiments.train_and_measure,
                          rows, stage=2, img_size=384, crop_tag="m2.5",
                          device="cpu", epochs=12, model_name="convnextv2_base",
                          measure_robust=False, verbose=False, **p)
            outs.append(o)
    finally:
        (split.get_fold, models.build, data.build_loaders, train.print_status,
         train.fit, train.cached_logits, evaluate.full_report) = orig

    names = [o["exp_name"] for o in outs]
    check("네 판의 이름이 전부 다르다", len(set(names)) == 4, str(names))
    check("기준선에는 lr 꼬리표가 안 붙는다",
          "_lr" not in names[0] and "_bb" not in names[0], names[0])
    check("헤드 lr 이 이름에 들어간다", "lr0.0001" in names[1], names[1])
    check("백본 배수가 이름에 들어간다", "bb0.1" in names[2], names[2])
    check("warmup 이 이름에 들어간다", "wu1" in names[3], names[3])

    # 값이 실제로 cfg 에 들어갔는가 (이름만 바뀌고 값이 그대로면 최악입니다)
    check("cfg.lr 이 실제로 바뀐다", seen[1].lr == 1e-4, str(seen[1].lr))
    check("cfg.backbone_lr_mult 이 실제로 바뀐다",
          seen[2].backbone_lr_mult == 0.1, str(seen[2].backbone_lr_mult))
    check("cfg.warmup_epochs 가 실제로 바뀐다",
          seen[3].warmup_epochs == 1, str(seen[3].warmup_epochs))
    check("결과에 백본 lr 이 남는다",
          abs(outs[3]["backbone_lr"] - 1e-5) < 1e-12, str(outs[3]["backbone_lr"]))


# ──────────────────────────────────────────────────────────────
# 2. 판정 — 기준이 결과를 따라가지 않는가
# ──────────────────────────────────────────────────────────────
def _run(lr, mult, score, best_epoch, wu=2):
    return {"lr": lr, "backbone_lr_mult": mult, "backbone_lr": lr * mult,
            "warmup_epochs": wu, "score": score, "score_name": "macro-F1",
            "best_epoch": best_epoch, "n_epochs": 12, "minutes": 40.0,
            "scale_drop": 0.25, "stage": 2}


def test_verdict_rules_are_fixed_before_the_run():
    print("\n[판정] 1차 기준이 점수가 아니라 best_epoch 인가")
    from src import experiments

    check("문턱이 코드에 있다", experiments.LR_MIN_BEST_EPOCH == 3)
    check("잡음 폭이 코드에 있다", experiments.MACRO_F1_NOISE == 0.02)

    # ① 전부 0~2에폭 → 점수가 올라도 축을 닫습니다
    v, log = _quiet(experiments.lr_report, [
        _run(3e-4, 0.3, 0.55, 0), _run(1e-4, 0.3, 0.60, 1),
        _run(3e-4, 0.1, 0.61, 2), _run(1e-4, 0.1, 0.59, 0)])
    check("전부 0~2에폭이면 축을 닫는다", v["verdict"] == "축 닫힘", str(v["verdict"]))
    check("점수가 +0.06 이어도 안 채택한다", v["winner"] is None)
    check("다음 용의자를 말해준다", "데이터" in log)

    # ② 학습은 되는데 점수가 잡음 안
    v, log = _quiet(experiments.lr_report, [
        _run(3e-4, 0.3, 0.55, 0), _run(1e-4, 0.3, 0.56, 6)])
    check("잡음 안이면 채택 안 한다", v["winner"] is None, str(v["winner"]))
    check("그래도 '학습은 된다' 고 구분해 준다",
          v["verdict"] == "학습은 되나 점수는 그대로", str(v["verdict"]))
    check("1차는 통과했다고 센다", len(v["trained"]) == 1)

    # ③ 둘 다 만족
    v, log = _quiet(experiments.lr_report, [
        _run(3e-4, 0.3, 0.55, 0), _run(1e-4, 0.1, 0.58, 7)])
    check("잡음 밖 + 수렴이면 채택", v["verdict"] == "채택", str(v["verdict"]))
    check("이긴 설정을 돌려준다", v["winner"] == (1e-4, 0.1, 2), str(v["winner"]))
    check("서브셋이라고 경고한다", "서브셋" in log)

    # ④ 경계 — 딱 문턱이면 통과, 하나 아래면 탈락
    v, _ = _quiet(experiments.lr_report,
                  [_run(3e-4, 0.3, 0.55, 0), _run(1e-4, 0.3, 0.57, 3)])
    check("best_epoch 3 은 통과", len(v["trained"]) == 1)
    v, _ = _quiet(experiments.lr_report,
                  [_run(3e-4, 0.3, 0.55, 0), _run(1e-4, 0.3, 0.57, 2)])
    check("best_epoch 2 는 탈락", len(v["trained"]) == 0)


# NOTE 노트북 셀이 판정을 베껴 갔는지 보던 검사가 여기 있었습니다. 03h 노트북은
#      아직 결과가 안 나와서 main 에 올리지 않았고, 검사만 남으면 없는 파일을
#      읽다 죽습니다. 노트북은 작업 브랜치에 있습니다 — 그쪽에서 돌리세요.
#      판정 규칙 자체(experiments.lr_report)는 위 검사가 그대로 봅니다.


if __name__ == "__main__":
    print("2단계 학습률 스윕 (STEP 17)")
    for fn in (test_each_plan_gets_its_own_experiment_name,
               test_verdict_rules_are_fixed_before_the_run):
        fn()
    print()
    if FAILS:
        print(f"{len(FAILS)} check(s) failed:")
        for f in FAILS:
            print(f"  - {f}")
        sys.exit(1)
    print("all checks passed")
