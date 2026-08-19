"""확률 보정 (calibration) — "의심된다"를 말할 자격을 만드는 단계.

문제:
  신경망은 자기 확신이 과합니다. "확률 95%" 라고 출력한 예측 100건 중
  실제로는 70건만 맞는 게 흔합니다(overconfidence).
  우리 서비스는 "신뢰도 72% 로 의심됩니다" 같은 문구를 보호자에게 보여줄 건데,
  그 72% 가 거짓말이면 안 됩니다.

해결:
  온도 스케일링(temperature scaling). logits 를 T 로 나누기만 하는,
  파라미터 1개짜리 보정입니다. 순위(=예측 결과)는 전혀 바뀌지 않고
  확률의 "자신감"만 조정되므로 정확도는 그대로 두고 신뢰도만 정직해집니다.

  ⚠️ 반드시 **검증셋**으로 T 를 학습하고 **테스트셋**에서 효과를 확인하세요.
     테스트셋으로 T 를 맞추면 그건 또 다른 형태의 과적합입니다.

    from src import calibrate
    T = calibrate.fit_temperature(val_logits, val_y)
    cal = calibrate.apply(test_logits, T)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# 온도 스케일링
# ──────────────────────────────────────────────────────────────
def fit_temperature(logits: torch.Tensor, y: torch.Tensor, max_iter: int = 200,
                    t_min: float = 0.25, t_max: float = 10.0, verbose: bool = True) -> float:
    """NLL 을 최소화하는 T 를 찾습니다. T>1 이면 과신을 낮춘 것.

    T 는 [t_min, t_max] 로 제한합니다. 모델이 아직 아무것도 못 배웠거나
    검증셋이 너무 작으면 NLL 곡면이 평평해져 T 가 수천까지 발산하는데,
    그 T 를 쓰면 모든 확률이 1/n 로 뭉개져 서비스가 아무 말도 못 하게 됩니다.
    """
    logits = logits.detach().float().cpu()
    y = y.detach().long().cpu()
    log_T = torch.zeros(1, requires_grad=True)  # T = exp(log_T) → 항상 양수
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_T.exp().clamp(t_min, t_max), y)
        loss.backward()
        return loss

    before = float(F.cross_entropy(logits, y))
    opt.step(closure)
    raw_T = float(log_T.exp().item())
    T = float(min(max(raw_T, t_min), t_max))
    after = float(F.cross_entropy(logits / T, y))

    if verbose:
        print(f"[calibrate] T = {T:.4f}   NLL {before:.4f} → {after:.4f}")
        if raw_T > t_max or raw_T < t_min:
            print(f"  ⚠️ 최적 T 가 {raw_T:.1f} 로 발산해 {T} 로 잘랐습니다.")
            print("     보통 (1) 모델이 아직 제대로 학습되지 않았거나 (2) 검증셋이 너무 작을 때 "
                  "생깁니다. 보정 결과를 믿지 말고 학습부터 다시 확인하세요.")
        elif T > 1.05:
            print("  → 모델이 과신하고 있었습니다 (T>1 로 확률을 낮춤). 정상적인 결과입니다.")
        elif T < 0.95:
            print("  → 모델이 과소평가하고 있었습니다 (T<1 로 확률을 높임). 드문 경우입니다.")
        else:
            print("  → 원래 보정이 잘 되어 있었습니다.")
    return T


def apply(logits: torch.Tensor | np.ndarray, T: float) -> np.ndarray:
    """온도를 적용한 확률을 돌려줍니다."""
    x = torch.as_tensor(np.asarray(logits), dtype=torch.float32) / T
    return F.softmax(x, dim=1).numpy()


# ──────────────────────────────────────────────────────────────
# 보정 품질 측정
# ──────────────────────────────────────────────────────────────
def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error.

    확률을 구간으로 나눠 "그 구간에서 주장한 확신"과 "실제 정답률"의
    차이를 가중평균한 값. 0 에 가까울수록 정직합니다.
    실무적으로 0.05 미만이면 양호, 0.15 이상이면 확률을 보여주면 안 됩니다.
    """
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def mce(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Maximum Calibration Error — 최악의 구간에서 얼마나 벌어지는가."""
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    worst = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() < 5:
            continue
        worst = max(worst, abs(correct[m].mean() - conf[m].mean()))
    return float(worst)


def reliability_diagram(probs_before: np.ndarray, probs_after: np.ndarray | None,
                        y: np.ndarray, n_bins: int = 15) -> None:
    """신뢰도 다이어그램. 대각선에 붙을수록 정직한 확률입니다."""
    import matplotlib.pyplot as plt

    def bins(p):
        conf, pred = p.max(1), p.argmax(1)
        correct = (pred == y).astype(float)
        edges = np.linspace(0, 1, n_bins + 1)
        xs, ys, ns = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() < 3:
                continue
            xs.append(conf[m].mean()); ys.append(correct[m].mean()); ns.append(int(m.sum()))
        return xs, ys, ns

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot([0, 1], [0, 1], "k:", label="완벽한 보정")
    for p, lbl, style in ((probs_before, "보정 전", "o-"), (probs_after, "보정 후", "s-")):
        if p is None:
            continue
        xs, ys, _ = bins(p)
        ax.plot(xs, ys, style, label=f"{lbl} (ECE={ece(p, y):.3f})")
    ax.set_xlabel("모델이 주장한 확신"); ax.set_ylabel("실제 정답률")
    ax.set_title("신뢰도 다이어그램"); ax.legend(); ax.grid(alpha=.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout(); plt.show()
    print("💡 곡선이 대각선 아래 = 과신(주장한 확률보다 실제로 덜 맞음).")
    print("   보정 후 곡선이 대각선에 붙어야 '신뢰도 72%' 라는 문구를 쓸 수 있습니다.")


def report(logits_val, y_val, logits_test, y_test, verbose: bool = True) -> dict:
    """검증셋으로 T 를 학습하고 테스트셋에서 효과를 확인합니다."""
    from src.evaluate import softmax_np

    T = fit_temperature(logits_val, y_val, verbose=verbose)
    yt = np.asarray(y_test.numpy() if hasattr(y_test, "numpy") else y_test)
    before = softmax_np(logits_test)
    after = apply(logits_test, T)

    out = {
        "temperature": T,
        "ece_before": ece(before, yt), "ece_after": ece(after, yt),
        "mce_before": mce(before, yt), "mce_after": mce(after, yt),
        "acc_before": float((before.argmax(1) == yt).mean()),
        "acc_after": float((after.argmax(1) == yt).mean()),
    }
    if verbose:
        print(f"\n  ECE  {out['ece_before']:.4f} → {out['ece_after']:.4f}")
        print(f"  MCE  {out['mce_before']:.4f} → {out['mce_after']:.4f}")
        print(f"  정확도 {out['acc_before']:.4f} → {out['acc_after']:.4f}  (변하지 않는 게 정상)")
        if out["ece_after"] > 0.10:
            print("  ⚠️ 보정 후에도 ECE 가 큽니다. 확률 숫자를 사용자에게 노출하지 마세요.")
    return out


# ──────────────────────────────────────────────────────────────
# 임계값 / 거절(abstention)
# ──────────────────────────────────────────────────────────────
def coverage_risk_curve(probs: np.ndarray, y: np.ndarray, show: bool = True) -> dict:
    """"자신 없는 건 답하지 않기"의 효과를 봅니다.

    coverage = 답한 비율, risk = 답한 것 중 틀린 비율.
    "80% 만 답하면 오답률이 12% → 5% 로 떨어진다" 같은 판단 근거가 됩니다.
    """
    conf = probs.max(1)
    correct = (probs.argmax(1) == y)
    order = np.argsort(-conf)
    c_sorted = correct[order]

    cov = np.arange(1, len(y) + 1) / len(y)
    risk = 1 - np.cumsum(c_sorted) / np.arange(1, len(y) + 1)

    out = {}
    for target in (1.0, 0.9, 0.8, 0.7, 0.5):
        i = min(int(target * len(y)) - 1, len(y) - 1)
        out[f"risk@cov{int(target * 100)}"] = float(risk[i])
        out[f"thr@cov{int(target * 100)}"] = float(conf[order][i])

    if show:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(5.6, 3.8))
        plt.plot(cov, risk)
        plt.xlabel("coverage (답한 비율)"); plt.ylabel("risk (답한 것 중 오답률)")
        plt.title("coverage–risk 곡선"); plt.grid(alpha=.3)
        plt.tight_layout(); plt.show()
        print("\n  답한 비율 → 오답률")
        for t in (100, 90, 80, 70, 50):
            print(f"    {t:>3}%  →  {out[f'risk@cov{t}']:.1%}   "
                  f"(신뢰도 임계값 {out[f'thr@cov{t}']:.3f})")
        print("\n💡 '판단 어려움 — 사진을 다시 찍어주세요' 로 돌릴 비율을 여기서 정하세요.")
    return out


def suggest_abstain_threshold(probs: np.ndarray, y: np.ndarray,
                              max_risk: float = 0.20) -> float:
    """오답률을 max_risk 이하로 유지하면서 최대한 많이 답하는 임계값."""
    conf = probs.max(1)
    correct = (probs.argmax(1) == y)
    best_thr, best_cov = 1.0, 0.0
    for thr in np.linspace(0.2, 0.99, 80):
        m = conf >= thr
        if m.sum() < 20:
            continue
        risk = 1 - correct[m].mean()
        if risk <= max_risk and m.mean() > best_cov:
            best_thr, best_cov = float(thr), float(m.mean())
    print(f"[calibrate] 오답률 ≤ {max_risk:.0%} 조건에서 임계값 {best_thr:.3f} "
          f"→ 전체의 {best_cov:.1%} 에 답변, 나머지는 '판단 어려움'")
    return best_thr
