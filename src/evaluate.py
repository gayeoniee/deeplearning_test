"""평가 — 정확도 하나로 속지 않기.

이 프로젝트에서 accuracy 만 보면 안 되는 이유:
  클래스가 6개인데 A2 가 40%, A5 가 3% 라고 해봅시다.
  "전부 A2 라고 찍는" 모델도 accuracy 40% 가 나옵니다. 쓸모는 0인데요.
  그리고 A5(미란·궤양)는 임상적으로 더 급한 병변입니다 — 정확히 그 클래스를
  못 맞히는 모델이 높은 accuracy 를 받습니다.

그래서 기본 보고 지표는 **macro-F1** 과 **클래스별 recall** 입니다.
(docs/basics/07_평가지표_의료AI_관점.md)

    from src import evaluate as ev
    rep = ev.full_report(logits, y, classes)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from src.config import CLASS_KO


# ──────────────────────────────────────────────────────────────
# 지표
# ──────────────────────────────────────────────────────────────
def pad_ko(s: str, width: int) -> str:
    """한글은 터미널에서 두 칸을 차지해서 f-string 의 :<26 이 어긋납니다.

    표시 폭 기준으로 채워 클래스별 표가 삐뚤어지지 않게 합니다.
    """
    import unicodedata

    w = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)
    return s + " " * max(width - w, 0)


def softmax_np(logits: torch.Tensor | np.ndarray) -> np.ndarray:
    x = logits.numpy() if isinstance(logits, torch.Tensor) else np.asarray(logits)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray | None,
            n_classes: int) -> dict:
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                                 f1_score, precision_recall_fscore_support, roc_auc_score)

    out: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0
    )
    out["per_class"] = {
        "precision": p.tolist(), "recall": r.tolist(), "f1": f.tolist(), "support": s.tolist()
    }
    if probs is not None and len(np.unique(y_true)) > 1:
        try:
            out["macro_auroc"] = float(
                roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
            )
        except ValueError:
            out["macro_auroc"] = float("nan")
    return out


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, metric: str = "macro_f1",
                 n: int = 1000, alpha: float = 0.05, seed: int = 0,
                 cls: int | None = None) -> tuple[float, float, float]:
    """부트스트랩 신뢰구간.

    "macro-F1 0.74" 보다 "0.74 (95% CI 0.71–0.77)" 이 정직합니다.
    검증셋이 작으면 구간이 넓게 나오는데, 그게 사실입니다.

    `metric="recall"` + `cls=<클래스 번호>` 로 **클래스 하나의 recall** 구간도 냅니다.
    A4·A6 처럼 표본이 적은 클래스는 이게 없으면 개선인지 주사위인지 알 수 없습니다
    (A6 는 val 표본 ~256장에서 2σ 가 ±0.085 로, 목표 개선폭과 비슷했습니다).
    """
    from sklearn.metrics import balanced_accuracy_score, f1_score

    if metric == "recall":
        if cls is None:
            raise ValueError("metric='recall' 에는 cls=<클래스 번호> 가 필요합니다")

        def fn(a, b):
            m = a == cls
            return float((b[m] == cls).mean()) if m.any() else float("nan")
    else:
        fn = {"macro_f1": lambda a, b: f1_score(a, b, average="macro", zero_division=0),
              "balanced_accuracy": balanced_accuracy_score}[metric]

    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        # ⚠️ 클래스별 recall 은 그 클래스가 뽑혀야 잽니다. 전체 클래스 수로
        #    거르면(<2) 그 클래스가 0장인 표본이 통과해 nan 이 섞입니다.
        if metric == "recall":
            if not (y_true[idx] == cls).any():
                continue
        elif len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(fn(y_true[idx], y_pred[idx]))
    vals = np.array(vals)
    return float(fn(y_true, y_pred)), float(np.quantile(vals, alpha / 2)), \
        float(np.quantile(vals, 1 - alpha / 2))


# ──────────────────────────────────────────────────────────────
# 리포트
# ──────────────────────────────────────────────────────────────
@dataclass
class EvalReport:
    classes: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    ci: tuple[float, float, float] = (0.0, 0.0, 0.0)
    confusion: list = field(default_factory=list)
    y_true: np.ndarray | None = None
    y_pred: np.ndarray | None = None
    probs: np.ndarray | None = None

    def print(self) -> None:
        m = self.metrics
        print("\n" + "=" * 64)
        print(" 평가 결과")
        print("=" * 64)
        print(f"  macro-F1          : {m['macro_f1']:.4f}   "
              f"(95% CI {self.ci[1]:.4f} – {self.ci[2]:.4f})   ← 주 지표")
        print(f"  balanced accuracy : {m['balanced_accuracy']:.4f}")
        print(f"  accuracy          : {m['accuracy']:.4f}   ← 불균형이라 참고만")
        print(f"  Cohen's kappa     : {m['cohen_kappa']:.4f}")
        if "macro_auroc" in m:
            print(f"  macro AUROC       : {m['macro_auroc']:.4f}")

        print("\n  클래스별 (recall 이 낮은 클래스 = 놓치는 병변)")
        print(f"  {pad_ko('클래스', 30)}{'precision':>10}{'recall':>9}{'f1':>8}{'n':>8}")
        pc = m["per_class"]
        for i, c in enumerate(self.classes):
            name = pad_ko(f"{c} {CLASS_KO.get(c, '')}", 30)
            print(f"  {name}{pc['precision'][i]:>10.3f}{pc['recall'][i]:>9.3f}"
                  f"{pc['f1'][i]:>8.3f}{pc['support'][i]:>8,}")

        # 검증셋에 한 장도 없는 클래스는 recall 0 으로 나오므로 최저 판정에서 제외합니다.
        present = [i for i, s in enumerate(pc["support"]) if s > 0]
        empty = [self.classes[i] for i, s in enumerate(pc["support"]) if s == 0]
        if empty:
            print(f"\n  ℹ️ 검증셋에 표본이 없는 클래스: {', '.join(empty)} "
                  "— 이 클래스의 지표는 의미가 없습니다.")
            print("     fold 를 바꾸거나 층화 분할이 제대로 됐는지 확인하세요.")
        if present:
            worst = min(present, key=lambda i: pc["recall"][i])
            print(f"\n  ⚠️ recall 최저: {self.classes[worst]} "
                  f"({CLASS_KO.get(self.classes[worst], '')}) = {pc['recall'][worst]:.3f}")
            print(f"     → 이 병변이 있는 강아지 {1 - pc['recall'][worst]:.0%} 를 놓칩니다.")
        print("=" * 64 + "\n")

    def plot_confusion(self, normalize: bool = True) -> None:
        import matplotlib.pyplot as plt

        cm = np.array(self.confusion, dtype=float)
        if normalize:
            cm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
        ax.set_xticks(range(len(self.classes)), self.classes)
        ax.set_yticks(range(len(self.classes)), self.classes)
        ax.set_xlabel("예측"); ax.set_ylabel("정답")
        ax.set_title("혼동행렬 (행 정규화)" if normalize else "혼동행렬")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                v = cm[i, j]
                ax.text(j, i, f"{v:.2f}" if normalize else f"{int(v)}",
                        ha="center", va="center",
                        color="white" if v > (0.5 if normalize else cm.max() / 2) else "black",
                        fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout(); plt.show()
        print("💡 대각선 밖의 큰 값 = 자주 헷갈리는 쌍입니다.")
        print("   임상적으로 비슷한 병변끼리 헷갈리는 건 어느 정도 자연스럽지만,")
        print("   A6(결절·종괴)를 A2(비듬)로 보는 것처럼 위험도가 다른 혼동은 문제입니다.")

    def plot_per_class(self) -> None:
        import matplotlib.pyplot as plt

        pc = self.metrics["per_class"]
        x = np.arange(len(self.classes))
        fig, ax = plt.subplots(figsize=(8, 3.6))
        ax.bar(x - 0.2, pc["precision"], 0.4, label="precision")
        ax.bar(x + 0.2, pc["recall"], 0.4, label="recall")
        ax.set_xticks(x, [f"{c}\n(n={s:,})" for c, s in zip(self.classes, pc["support"])],
                      fontsize=8)
        ax.axhline(0.5, color="grey", ls=":", lw=1)
        ax.set_ylim(0, 1); ax.legend(); ax.grid(axis="y", alpha=.3)
        ax.set_title("클래스별 precision / recall")
        plt.tight_layout(); plt.show()


def full_report(logits: torch.Tensor | np.ndarray, y_true: torch.Tensor | np.ndarray,
                classes: list[str], show: bool = True) -> EvalReport:
    from sklearn.metrics import confusion_matrix

    probs = softmax_np(logits)
    yt = y_true.numpy() if isinstance(y_true, torch.Tensor) else np.asarray(y_true)
    yp = probs.argmax(1)

    rep = EvalReport(
        classes=classes,
        metrics=metrics(yt, yp, probs, len(classes)),
        ci=bootstrap_ci(yt, yp),
        confusion=confusion_matrix(yt, yp, labels=list(range(len(classes)))).tolist(),
        y_true=yt, y_pred=yp, probs=probs,
    )
    if show:
        rep.print()
    return rep


# ──────────────────────────────────────────────────────────────
# 1단계(정상/이상) 전용
# ──────────────────────────────────────────────────────────────
def binary_report(scores: np.ndarray, y_bin: np.ndarray,
                  target_recall: float = 0.95) -> dict:
    """정상/이상 이진 판정에서 "놓치지 않는" 임계값을 찾습니다.

    보호자용 스크리닝에서는 오탐(정상인데 병원 가보라고 함)보다
    미탐(병변인데 괜찮다고 함)이 훨씬 나쁩니다. 그래서 recall 을 먼저 고정하고
    그 조건에서 precision 이 얼마나 나오는지를 봅니다.
    """
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

    prec, rec, thr = precision_recall_curve(y_bin, scores)
    ok = np.where(rec[:-1] >= target_recall)[0]
    if len(ok):
        i = ok[np.argmax(prec[:-1][ok])]
        chosen_thr, chosen_p, chosen_r = float(thr[i]), float(prec[i]), float(rec[i])
    else:
        chosen_thr, chosen_p, chosen_r = 0.5, float("nan"), float("nan")

    out = {
        "auroc": float(roc_auc_score(y_bin, scores)),
        "ap": float(average_precision_score(y_bin, scores)),
        "threshold": chosen_thr,
        "precision_at_target": chosen_p,
        "recall_at_target": chosen_r,
        "target_recall": target_recall,
    }
    print(f"\n[1단계 정상/이상]  AUROC {out['auroc']:.4f}  AP {out['ap']:.4f}")
    print(f"  recall ≥ {target_recall:.0%} 를 만족하는 임계값: {chosen_thr:.4f}")
    print(f"  그때 precision = {chosen_p:.3f}, recall = {chosen_r:.3f}")
    if chosen_p == chosen_p:  # not nan
        fp_rate = 1 - chosen_p
        print(f"  → 이상이라고 알린 것 중 {fp_rate:.0%} 는 실제로는 정상입니다.")
        print("     스크리닝 보조 목적이라면 감수할 만한 비용인지 판단하세요.")
    return out


def compare_models(results: dict[str, EvalReport]) -> None:
    """여러 모델 결과를 한 표로."""
    import pandas as pd

    rows = []
    for name, r in results.items():
        m = r.metrics
        rows.append({
            "model": name,
            "macro_F1": round(m["macro_f1"], 4),
            "CI_low": round(r.ci[1], 4),
            "CI_high": round(r.ci[2], 4),
            "bal_acc": round(m["balanced_accuracy"], 4),
            "accuracy": round(m["accuracy"], 4),
            "AUROC": round(m.get("macro_auroc", float("nan")), 4),
            "min_recall": round(min(m["per_class"]["recall"]), 4),
        })
    df = pd.DataFrame(rows).sort_values("macro_F1", ascending=False)
    print(df.to_string(index=False))
    print("\n💡 CI 가 겹치는 모델끼리는 '더 낫다'고 말할 수 없습니다.")
    print("   min_recall(최악 클래스 재현율)이 낮으면 평균이 좋아도 위험합니다.")
    return df
