"""2단계 구조 — ① 정상/이상 → ② 병변 6종.

왜 2단계인가:
  이 기능의 목적은 "보호자에게 의심된다까지 알려주기" 입니다.
  가장 중요한 판단은 **"병원에 가봐야 하나?"** 이고, 그건 이진 문제입니다.
  병변 종류를 6개로 나누는 건 그 다음 이야기입니다.

  한 번에 7클래스로 풀면 두 문제가 섞입니다:
    · A7(정상)을 A2(비듬)로 틀리는 것과 A2 를 A3 로 틀리는 것은
      임상적 무게가 완전히 다른데, 7클래스 손실함수는 둘을 똑같이 취급합니다.
    · 1단계에만 걸고 싶은 "재현율 우선" 임계값을 걸 수가 없습니다.

  그래서 나눕니다:
    1단계  A7 vs 나머지        재현율 ≥ 0.95 (놓치는 게 오탐보다 나쁨)
    2단계  A1~A6 만            병변 종류. 1단계를 통과한 것만 들어옵니다

데이터 규모 (VL01 기준, 46,483장):
    1단계   정상 22,815 / 이상 23,070   ← 거의 5:5, 학습이 수월합니다
    2단계   A2 7,693 … A5 1,464         ← 5.3배 불균형, class weight 필요

    from src import stages
    s1 = stages.to_stage1(df)      # 전체, 라벨이 A7/ABNORMAL 로 바뀜
    s2 = stages.to_stage2(df)      # A1~A6 만
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CLASSES, CLASSES_STAGE1, NORMAL_LABEL

ABNORMAL_LABEL = "ABNORMAL"

# 사용자가 실제로 보는 최종 출력 7종. A7 이 **마지막** 인덱스인 게 중요합니다
# (2단계 모델의 클래스 인덱스 0~5 를 그대로 재사용하려면 그래야 합니다).
PIPELINE_CLASSES: list[str] = CLASSES + [NORMAL_LABEL]


def to_stage1(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """1단계용 뷰. `label` 을 A7 / ABNORMAL 로 바꿉니다.

    fold / is_holdout / group 은 그대로 물려받으므로 두 단계가
    같은 분할을 씁니다 — 이게 중요합니다. 단계별로 따로 나누면
    1단계 val 에 쓰인 개체가 2단계 train 에 들어가 누수가 생깁니다.
    """
    out = df.copy()
    out["label_orig"] = out["label"]
    out["label"] = np.where(out["label"] == NORMAL_LABEL, NORMAL_LABEL, ABNORMAL_LABEL)
    if verbose:
        vc = out["label"].value_counts()
        n_ab, n_no = vc.get(ABNORMAL_LABEL, 0), vc.get(NORMAL_LABEL, 0)
        print(f"[stage1] 정상 {n_no:,} / 이상 {n_ab:,}  "
              f"(비율 {n_no / max(len(out), 1):.1%} : {n_ab / max(len(out), 1):.1%})")
    return out


def to_stage2(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """2단계용 뷰. 병변(A1~A6)만 남깁니다."""
    out = df[df["label"].isin(CLASSES)].reset_index(drop=True)
    if verbose:
        vc = out["label"].value_counts()
        print(f"[stage2] 병변 {len(out):,}장 / {vc.to_dict()}")
        if len(vc) > 1:
            print(f"[stage2] 불균형 비 {vc.max() / max(vc.min(), 1):.1f}배 "
                  "→ class_weight 권장")
    return out


def stage1_scores(logits, classes: list[str] | None = None) -> np.ndarray:
    """1단계 logits 에서 '이상일 확률' 을 뽑습니다.

    임계값 조정과 binary_report 에 넣을 점수입니다.
    """
    from src.evaluate import softmax_np

    classes = classes or CLASSES_STAGE1
    probs = softmax_np(logits)
    i_ab = classes.index(ABNORMAL_LABEL)
    return probs[:, i_ab]


def binary_targets(y, classes: list[str] | None = None) -> np.ndarray:
    """정답 인덱스 → 이상=1 / 정상=0."""
    classes = classes or CLASSES_STAGE1
    yy = y.numpy() if hasattr(y, "numpy") else np.asarray(y)
    return (yy == classes.index(ABNORMAL_LABEL)).astype(int)


def pipeline_predict(s1_scores: np.ndarray, s2_logits, threshold: float):
    """두 단계를 실제 서비스처럼 이어붙여 **최종 라벨 하나**를 만듭니다.

        1단계 '이상 확률' < threshold  →  A7(정상). 2단계는 아예 안 봄
        1단계 통과                     →  2단계의 argmax (A1~A6)

    두 인자는 **같은 행 집합, 같은 순서**여야 합니다. 즉 정상 사진까지 포함한
    전체 검증셋에 두 모델을 각각 돌린 결과를 넣으세요.

    돌려주는 것: (pred_idx  PIPELINE_CLASSES 기준 0~6,  conf  최종 확률)
    """
    from src.evaluate import softmax_np

    s1 = np.asarray(s1_scores, dtype=float)
    p2 = softmax_np(s2_logits)
    if len(p2) != len(s1):
        raise ValueError(
            f"행 수가 다릅니다: 1단계 {len(s1):,} vs 2단계 {len(p2):,}.\n"
            "두 모델을 **같은 데이터프레임**에 같은 순서로 돌려야 합니다.\n"
            "(2단계 모델도 정상 사진을 포함한 전체 검증셋에 돌리세요 — "
            "실제 서비스에서는 정상 사진도 2단계로 넘어올 수 있습니다)"
        )

    passed = s1 >= threshold
    i_normal = len(CLASSES)                    # PIPELINE_CLASSES 에서 A7 의 위치
    pred = np.where(passed, p2.argmax(1), i_normal)
    conf = np.where(passed, s1 * p2.max(1), 1.0 - s1)
    return pred, conf


def pipeline_report(
    s1_scores: np.ndarray,
    s2_logits,
    y_true_labels,
    threshold: float,
    show: bool = True,
) -> dict:
    """**사용자가 실제로 겪는 성능.** 최종 7종 기준으로 한 번에 평가합니다.

    `report_pipeline()` 은 두 단계를 따로 재서 곱한 추정치라 낙관적입니다.
    이 함수는 정상 사진까지 포함한 같은 집합에 두 모델을 이어붙여 돌린
    실제 결과를 봅니다. 최종 보고에는 **이 숫자**를 쓰세요.

    y_true_labels: 문자열 라벨 배열 (A1~A7).
    """
    from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

    y_lab = np.asarray([str(v) for v in y_true_labels])
    unknown = sorted(set(y_lab) - set(PIPELINE_CLASSES))
    if unknown:
        raise ValueError(f"모르는 라벨이 있습니다: {unknown} (기대: {PIPELINE_CLASSES})")

    idx = {c: i for i, c in enumerate(PIPELINE_CLASSES)}
    y = np.array([idx[c] for c in y_lab])
    pred, conf = pipeline_predict(s1_scores, s2_logits, threshold)

    i_normal = len(CLASSES)
    is_lesion = y != i_normal
    # ★ 이 프로젝트에서 가장 중요한 숫자: 병변인데 "정상" 이라고 안심시킨 비율
    missed = int((is_lesion & (pred == i_normal)).sum())
    missed_rate = float(missed / max(is_lesion.sum(), 1))
    # 병변을 병변으로 알린 것 중, 종류까지 맞힌 비율
    routed = is_lesion & (pred != i_normal)
    kind_acc = float((pred[routed] == y[routed]).mean()) if routed.any() else float("nan")
    false_alarm = float(((~is_lesion) & (pred != i_normal)).sum() / max((~is_lesion).sum(), 1))

    p, r, f, s = precision_recall_fscore_support(
        y, pred, labels=list(range(len(PIPELINE_CLASSES))), zero_division=0
    )
    out = {
        "threshold": float(threshold),
        "final_macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "lesion_missed": missed,
        "lesion_missed_rate": missed_rate,
        "lesion_screening_recall": 1.0 - missed_rate,
        "false_alarm_rate": false_alarm,
        "kind_accuracy_given_routed": kind_acc,
        "classes": PIPELINE_CLASSES,
        "per_class": {"precision": p.tolist(), "recall": r.tolist(),
                      "f1": f.tolist(), "support": s.tolist()},
        "confusion": confusion_matrix(
            y, pred, labels=list(range(len(PIPELINE_CLASSES)))).tolist(),
        "mean_confidence": float(np.mean(conf)),
    }

    if show:
        from src.config import CLASS_KO
        from src.evaluate import pad_ko

        print("\n" + "=" * 66)
        print(" 전체 파이프라인 — 사용자가 실제로 겪는 성능 (최종 7종)")
        print("=" * 66)
        print(f"  1단계 임계값                : {threshold:.4f}")
        print(f"  ★ 병변을 놓친 비율          : {missed_rate:.2%}  ({missed:,}/{int(is_lesion.sum()):,}장)")
        print(f"     = 병원 가야 하는데 '정상' 이라고 안심시킨 경우 ← 가장 위험한 오류")
        print(f"  ★ 스크리닝 재현율           : {1 - missed_rate:.4f}   (목표 ≥ 0.95)")
        print(f"  오탐 비율(정상→이상)        : {false_alarm:.2%}")
        print(f"     = 괜히 병원 가보라고 한 경우. 스크리닝에서는 감수하는 쪽입니다")
        print(f"  병변 종류 정확도(통과분)    : {kind_acc:.4f}")
        print(f"  최종 macro-F1 (7종)         : {out['final_macro_f1']:.4f}")
        print(f"\n  {pad_ko('클래스', 30)}{'precision':>10}{'recall':>9}{'f1':>8}{'n':>8}")
        for i, c in enumerate(PIPELINE_CLASSES):
            name = pad_ko(f"{c} {CLASS_KO.get(c, '')}", 30)
            print(f"  {name}{p[i]:>10.3f}{r[i]:>9.3f}{f[i]:>8.3f}{s[i]:>8,}")
        print("\n  💡 임계값을 낮추면 놓치는 병변은 줄고 오탐은 늘어납니다.")
        print("     이 프로젝트 목적('의심된다까지 알려주기')에서는 그 교환이 맞는 방향입니다.")
        print("=" * 66 + "\n")
    return out


def plot_pipeline_confusion(rep: dict, normalize: bool = True) -> None:
    """pipeline_report() 결과의 최종 7종 혼동행렬."""
    import matplotlib.pyplot as plt

    cm = np.array(rep["confusion"], dtype=float)
    if normalize:
        cm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    cls = rep["classes"]
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(cls)), cls)
    ax.set_yticks(range(len(cls)), cls)
    ax.set_xlabel("파이프라인 최종 예측")
    ax.set_ylabel("정답")
    ax.set_title("2단계 파이프라인 최종 혼동행렬")
    for i in range(len(cls)):
        for j in range(len(cls)):
            v = cm[i, j]
            ax.text(j, i, f"{v:.2f}" if normalize else f"{int(v)}", ha="center", va="center",
                    color="white" if v > (0.5 if normalize else cm.max() / 2) else "black",
                    fontsize=9)
    # 가장 위험한 칸을 표시: 병변(행 0~5) → A7(마지막 열)
    ax.add_patch(plt.Rectangle((len(cls) - 1.5, -0.5), 1, len(cls) - 1,
                               fill=False, edgecolor="red", lw=2.2))
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout(); plt.show()
    print("🟥 빨간 테두리 = 병변인데 '정상' 이라고 한 칸. 이 열이 이 프로젝트의 실패 지점입니다.")


def report_pipeline(
    s1_scores: np.ndarray,
    s1_true: np.ndarray,
    s2_logits,
    s2_true,
    threshold: float,
    verbose: bool = True,
) -> dict:
    """두 단계를 이어붙인 파이프라인 성능의 **빠른 추정치**.

    두 단계를 서로 다른 집합에서 재고 곱하는 방식이라 낙관적입니다.
    학습 중 감을 잡는 용도로만 쓰고, 최종 보고에는 `pipeline_report()` 를 쓰세요.

    ⚠️ 각 단계를 따로 잘 하는 것과, 이어붙여서 잘 하는 것은 다릅니다.
       1단계가 놓친 병변은 2단계가 볼 기회조차 없습니다.
    """
    from sklearn.metrics import f1_score, recall_score

    from src.evaluate import softmax_np

    passed = s1_scores >= threshold            # 1단계가 '이상' 이라고 판단
    真_이상 = s1_true == 1

    s1_recall = float(recall_score(s1_true, passed.astype(int), zero_division=0))
    missed = int((真_이상 & ~passed).sum())    # 병변인데 1단계에서 놓침

    s2_pred = softmax_np(s2_logits).argmax(1)
    s2_yy = s2_true.numpy() if hasattr(s2_true, "numpy") else np.asarray(s2_true)
    s2_f1 = float(f1_score(s2_yy, s2_pred, average="macro", zero_division=0))

    # 전체 파이프라인: 1단계 통과율 × 2단계 정확도
    end_to_end = s1_recall * float((s2_pred == s2_yy).mean())

    out = {
        "threshold": threshold,
        "stage1_recall": s1_recall,
        "stage1_missed": missed,
        "stage2_macro_f1": s2_f1,
        "pipeline_estimate": end_to_end,
    }
    if verbose:
        print("\n" + "=" * 60)
        print(" 전체 파이프라인 (사용자가 실제로 겪는 성능)")
        print("=" * 60)
        print(f"  1단계 임계값        : {threshold:.4f}")
        print(f"  1단계 재현율        : {s1_recall:.4f}")
        print(f"  1단계에서 놓친 병변 : {missed:,}장  ← 2단계가 볼 기회조차 없음")
        print(f"  2단계 macro-F1      : {s2_f1:.4f}")
        print(f"  파이프라인 추정치   : {end_to_end:.4f}")
        print("\n  💡 1단계 재현율이 낮으면 2단계를 아무리 잘 만들어도 소용없습니다.")
        print("=" * 60)
    return out
