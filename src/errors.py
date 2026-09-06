"""오답 분석 — **어디로** 틀리는지 숫자로 보고, 그 사진을 눈으로 봅니다.

왜 필요한가:
    STEP 10 holdout 에서 정상 사진의 33.3% 가 병원으로 갔습니다. 클래스별
    precision·recall 만 봐서는 그 1,233장이 **어느 병변으로** 갔는지 알 수 없습니다.
    주변합(marginal)에서 역산하면 순 흐름은 보이지만 칸(cell)은 안 보입니다.

    그리고 숫자로 끝나면 안 됩니다. 멘토 피드백 5번 — 틀린 사진 30장을 실제로
    보면 공통점이 나옵니다 (털이 길다 / 실내 조명이다 / 피부 접힌 자국이다).
    공통점이 있으면 촬영 가이드나 데이터 정제로 잡히고, 없으면 모델을 손봐야
    한다는 뜻입니다. 어느 쪽인지가 다음 실험을 정합니다.

⚠️ **여기 숫자를 보고 설정을 고르면 안 됩니다.** 진단용입니다.
   그래서 기본 대상이 val 입니다 — holdout 은 확인용으로만 같이 찍습니다.

    from src import errors
    errors.confusion_pairs(rep)          # 어느 칸으로 몇 장
    errors.false_alarm_targets(rep)      # 정상이 어느 병변으로 갔나
    errors.contact_sheet(df, mask, ...)  # 그 사진 30장을 한 판에
"""

from __future__ import annotations

import numpy as np

from src.stages import PIPELINE_CLASSES


def _cm(rep: dict) -> tuple[np.ndarray, list[str]]:
    cm = np.asarray(rep["confusion"], dtype=int)
    classes = list(rep.get("classes") or PIPELINE_CLASSES)
    if cm.shape != (len(classes), len(classes)):
        raise ValueError(f"혼동행렬 {cm.shape} 가 클래스 {len(classes)}개와 안 맞습니다")
    return cm, classes


def confusion_pairs(rep: dict, top: int = 15, show: bool = True) -> list[dict]:
    """대각선 밖 칸을 **장수 순으로** 줄 세웁니다.

    반환: [{"true", "pred", "n", "share_of_true"}, ...]
    `share_of_true` = 그 정답 클래스 전체에서 이 칸이 차지하는 비율.
    """
    cm, classes = _cm(rep)
    from src.config import CLASS_KO
    from src.evaluate import pad_ko

    row_tot = cm.sum(axis=1)
    pairs = [
        {"true": classes[i], "pred": classes[j], "n": int(cm[i, j]),
         "share_of_true": float(cm[i, j] / row_tot[i]) if row_tot[i] else 0.0}
        for i in range(len(classes)) for j in range(len(classes))
        if i != j and cm[i, j] > 0
    ]
    pairs.sort(key=lambda d: -d["n"])

    if show:
        print("\n" + "=" * 74)
        print(" 자주 헷갈리는 칸 — 정답 → 예측")
        print("=" * 74)
        print(f"  {pad_ko('정답', 26)}{pad_ko('예측', 26)}{'장수':>8}{'그 클래스의':>12}")
        for d in pairs[:top]:
            t = pad_ko(f"{d['true']} {CLASS_KO.get(d['true'], '')}", 26)
            p = pad_ko(f"{d['pred']} {CLASS_KO.get(d['pred'], '')}", 26)
            print(f"  {t}{p}{d['n']:>8,}{d['share_of_true']:>11.1%}")
        if len(pairs) > top:
            print(f"  … 그 밖에 {len(pairs) - top}칸")
        print("=" * 74)
    return pairs


def false_alarm_targets(rep: dict, normal: str = "A7", show: bool = True) -> dict:
    """헛알림이 **어느 병변으로** 갔는지. 정상 행만 떼어 봅니다."""
    cm, classes = _cm(rep)
    from src.config import CLASS_KO
    from src.evaluate import pad_ko

    i = classes.index(normal)
    row = cm[i]
    total, wrong = int(row.sum()), int(row.sum() - row[i])
    out = {"normal_total": total, "false_alarms": wrong,
           "by_class": {classes[j]: int(row[j]) for j in range(len(classes)) if j != i}}

    if show:
        print("\n" + "=" * 74)
        print(f" 헛알림의 행선지 — 실제 '{normal}' {total:,}장 중 {wrong:,}장이 병원행")
        print("=" * 74)
        if wrong:
            for c, n in sorted(out["by_class"].items(), key=lambda kv: -kv[1]):
                if not n:
                    continue
                bar = "█" * max(1, round(30 * n / wrong))
                name = pad_ko(f"{c} {CLASS_KO.get(c, '')}", 28)
                print(f"  {name}{n:>7,}{n / wrong:>8.1%}  {bar}")
        print("=" * 74)
        print("  💡 한두 클래스에 몰려 있으면 그 클래스의 경계 문제입니다.")
        print("     고르게 퍼져 있으면 1단계가 정상을 못 가리는 것입니다.")
    return out


def miss_sources(rep: dict, normal: str = "A7", show: bool = True) -> dict:
    """**가장 위험한 오류** — 병변인데 '정상' 이라고 한 것이 어느 병변인지."""
    cm, classes = _cm(rep)
    from src.config import CLASS_KO
    from src.evaluate import pad_ko

    j = classes.index(normal)
    col = cm[:, j]
    out = {"total": int(col.sum() - col[j]),
           "by_class": {classes[i]: int(col[i]) for i in range(len(classes)) if i != j},
           "rate_by_class": {classes[i]: (float(col[i] / cm[i].sum()) if cm[i].sum() else 0.0)
                             for i in range(len(classes)) if i != j}}

    if show:
        print("\n" + "=" * 74)
        print(f" 놓친 병변 — '{normal}(정상)' 이라고 안심시킨 {out['total']:,}장")
        print("=" * 74)
        print(f"  {pad_ko('병변', 28)}{'장수':>7}{'그 병변의':>11}")
        for c, n in sorted(out["by_class"].items(), key=lambda kv: -kv[1]):
            name = pad_ko(f"{c} {CLASS_KO.get(c, '')}", 28)
            print(f"  {name}{n:>7,}{out['rate_by_class'][c]:>10.1%}")
        print("=" * 74)
        print("  ⚠️ 이 열이 이 프로젝트의 실패 지점입니다. 비율이 높은 병변부터 봅니다.")
    return out


# ── 멘토 지적을 수치로: "짝 혼동인가, 여러 클래스로 흩어지는가" ──────────
#
# 멘토(2026-09-05): *"짝(pairwise) 혼동이 아니라 한 클래스가 여러 클래스로
# 동시에 흩어진다 — 그 클래스의 핵심 표현을 못 잡은 것"*.
#
# `confusion_pairs` 는 **칸을 장수 순으로** 줄 세웁니다. 그건 짝을 보는 눈이라
# 이 질문에 답을 못 합니다 — A4 가 다섯 클래스로 골고루 흩어져도 각 칸은 작아서
# 상위 목록에 안 올라옵니다. 그래서 **행의 모양**을 따로 잽니다.
#
# ⚠️ 흩어짐만 보면 절반만 보는 것입니다. 흩어지는 데도 **방향**이 있고,
#    그 방향이 클래스 빈도 순서와 같으면 원인이 "표현 부재" 가 아니라
#    **사전확률(prior)** 일 수 있습니다. prior 라면 재학습 없이 로짓 보정으로
#    고쳐지므로, 둘을 갈라 놓는 게 다음 실험을 정합니다.

# ── 사전등록 판정 기준 (결과를 보고 바꾸지 않습니다 — 작업 규칙 2) ──────
#   정규화 엔트로피: 대각선 밖 분포가 균등하면 1, 한 칸에 몰리면 0
DISPERSION_SCATTER = 0.85     # 이 위면 "흩어짐"
DISPERSION_PAIRWISE = 0.60    # 이 아래 + 최다 행선지 비중이 아래 값 이상이면 "짝 혼동"
PAIRWISE_TOP_SHARE = 0.50
#   빈도 큰 쪽으로 흐르는 양 ÷ **우연히 그렇게 될 양**. 이 위면 prior 를 의심합니다.
#   ⚠️ 날 것의 비(더 흔한 쪽 ÷ 덜 흔한 쪽)를 쓰면 안 됩니다 — 제일 드문 클래스는
#      다른 모든 클래스가 "더 흔한 쪽" 이라 **모델과 무관하게 무한대**가 나옵니다
#      (A6 에서 실제로 그랬습니다). 그래서 균등 분산일 때의 기대값으로 나눕니다.
PRIOR_LIFT_SUSPECT = 1.25


def class_dispersion(rep: dict, show: bool = True) -> list[dict]:
    """클래스마다 오답이 **흩어지는지 한 곳으로 몰리는지**, 그리고 어느 쪽으로.

    반환 항목 하나가 정답 클래스 하나입니다:
        recall        대각선 ÷ 행 합
        h_norm        대각선 **뺀** 행 분포의 엔트로피 ÷ ln(칸 수). 1 = 완전 균등
        top_pred      최다 행선지, top_share 그 비중 (오답 중에서)
        to_larger     오답 중 **더 흔한** 클래스로 간 비율
        exp_larger    오답이 균등하게 흩어졌을 때의 to_larger (= 더 흔한 클래스 수 ÷ 칸 수)
        lift          to_larger ÷ exp_larger. **1 이면 우연과 같음**, 크면 소수→다수 편향
        fp, fn        열 합 − 대각선 / 행 합 − 대각선. sink 인지 source 인지
        shape         "흩어짐" / "짝 혼동" / "그 사이"
        prior_suspect lift 가 문턱을 넘는가

    ⚠️ 빈도는 **이 행렬의 행 합**(정답 장수)으로 봅니다. 학습 장수가 아닙니다 —
       보통 같은 방향이지만 평가셋 구성이 다르면 갈립니다.
    """
    cm, classes = _cm(rep)
    from src.config import CLASS_KO
    from src.evaluate import pad_ko

    n = len(classes)
    support = cm.sum(axis=1)
    out: list[dict] = []
    for i, c in enumerate(classes):
        row = cm[i].astype(float)
        diag = row[i]
        off = np.delete(row, i)
        names = [x for j, x in enumerate(classes) if j != i]
        sup_other = np.delete(support, i)
        tot = off.sum()

        if tot <= 0:                       # 오답이 없으면 모양을 말할 수 없습니다
            out.append({"cls": c, "recall": float(diag / support[i]) if support[i] else 0.0,
                        "h_norm": float("nan"), "top_pred": None, "top_share": 0.0,
                        "to_larger": 0.0, "exp_larger": float("nan"), "lift": float("nan"),
                        "fp": int(cm[:, i].sum() - diag), "fn": 0,
                        "shape": "오답 없음", "prior_suspect": False})
            continue

        p = off / tot
        nz = p[p > 0]
        h = float(-(nz * np.log(nz)).sum() / np.log(n - 1)) if n > 2 else 0.0
        k = int(p.argmax())
        larger = float(p[sup_other > support[i]].sum())
        # 균등하게 흩어졌다면 이만큼이 "더 흔한 쪽" 으로 갑니다. 이걸로 나눠야
        # 드문 클래스가 자동으로 prior 의심을 받는 일이 없습니다.
        exp_larger = float((sup_other > support[i]).mean())
        # ⚠️ exp_larger 가 0 이거나 1 이면 lift 는 **정보가 없습니다.**
        #    제일 흔한 클래스는 갈 "더 흔한 쪽" 이 없고(0), 제일 드문 클래스는
        #    다른 전부가 더 흔해서(1) lift 가 무조건 1.0 으로 나옵니다.
        #    그걸 1.00 으로 찍으면 "우연과 같다" 는 판정처럼 읽혀서 위험합니다.
        lift = (larger / exp_larger) if 0 < exp_larger < 1 else float("nan")

        if h >= DISPERSION_SCATTER:
            shape = "흩어짐"
        elif h <= DISPERSION_PAIRWISE and p[k] >= PAIRWISE_TOP_SHARE:
            shape = "짝 혼동"
        else:
            shape = "그 사이"

        out.append({
            "cls": c, "recall": float(diag / support[i]) if support[i] else 0.0,
            "h_norm": h, "top_pred": names[k], "top_share": float(p[k]),
            "to_larger": larger, "exp_larger": exp_larger, "lift": lift,
            "fp": int(cm[:, i].sum() - diag), "fn": int(tot),
            "shape": shape,
            "prior_suspect": bool(lift == lift and lift >= PRIOR_LIFT_SUSPECT),
        })

    if show:
        print("\n" + "=" * 88)
        print(" 오답이 흩어지는가 · 어느 쪽으로 — 행에서 대각선을 뺀 모양")
        print("=" * 88)
        print(f"  {pad_ko('정답', 24)}{'recall':>8}{'H':>7}{'최다행선지':>12}{'비중':>7}"
              f"{'→흔한쪽':>9}{'우연히':>8}{'lift':>7}  모양")
        for d in out:
            t = pad_ko(f"{d['cls']} {CLASS_KO.get(d['cls'], '')}", 24)
            lf = "—" if d["lift"] != d["lift"] else f"{d['lift']:.2f}"
            mark = " ⚠️prior" if d["prior_suspect"] else ""
            print(f"  {t}{d['recall']:>8.3f}{d['h_norm']:>7.3f}"
                  f"{str(d['top_pred'] or '—'):>12}{d['top_share']:>7.2f}"
                  f"{d['to_larger']:>9.2f}{d['exp_larger']:>8.2f}{lf:>7}  {d['shape']}{mark}")
        print("-" * 88)
        print(f"  H = 대각선 뺀 행 분포의 정규화 엔트로피. ≥{DISPERSION_SCATTER} 흩어짐 /"
              f" ≤{DISPERSION_PAIRWISE} 이면서 최다 ≥{PAIRWISE_TOP_SHARE:.0%} 면 짝 혼동")
        print(f"  lift = 흔한 쪽으로 간 비율 ÷ **우연히 그럴 비율**."
              f" 1.00 = 우연과 같음, ≥{PRIOR_LIFT_SUSPECT} 면 prior 의심")
        print("  ⚠️ lift 가 '—' 인 클래스는 **제일 흔하거나 제일 드물어서** 잴 수가"
              " 없는 것입니다 (해석 금지).")
        print("  ⚠️ '흩어짐' 은 표현 부재의 **증상**이지 증거가 아닙니다. prior 로도"
              " 같은 모양이 나옵니다 —")
        print("     로짓 보정으로 쏠림이 사라지는지 보면 갈립니다 (재학습 불필요).")
        sinks = [d["cls"] for d in out if d["fp"] > d["fn"] * 1.5]
        sources = [d["cls"] for d in out if d["fn"] > d["fp"] * 1.5]
        print(f"  빨아들이는 쪽(sink, FP≫FN): {sinks or '없음'}")
        print(f"  새어나가는 쪽(source, FN≫FP): {sources or '없음'}")
        print("=" * 88)
    return out


def contact_sheet(df, *, path_col: str = "crop_path", n: int = 30, cols: int = 6,
                  title: str = "", seed: int = 0, save_to=None, show: bool = True):
    """사진을 한 판에 깔아 **눈으로** 봅니다 (CAM 없이 원본만).

    df 는 이미 걸러진 것을 주세요 (예: 헛알림 난 정상 사진만).
    `note` 컬럼이 있으면 각 칸 제목으로 씁니다.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    sub = df[df[path_col].notna()]
    if len(sub) == 0:
        print(f"[contact_sheet] 보여줄 사진이 없습니다{' — ' + title if title else ''}")
        return None
    picks = sub.sample(min(n, len(sub)), random_state=seed)

    cols = max(1, min(cols, len(picks)))
    rows = (len(picks) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.8 * rows))
    axes = [axes] if len(picks) == 1 else list(np.array(axes).flat)

    for ax, (_, r) in zip(axes, picks.iterrows()):
        try:
            with Image.open(r[path_col]) as im:
                ax.imshow(im.convert("RGB").resize((256, 256)))
        except Exception as exc:                                  # noqa: BLE001
            ax.text(0.5, 0.5, str(exc)[:50], ha="center", wrap=True, fontsize=6)
        ax.set_title(str(r.get("note", ""))[:34], fontsize=7.5)
        ax.axis("off")
    for ax in axes[len(picks):]:
        ax.axis("off")
    if title:
        fig.suptitle(f"{title}  (n={len(picks)} / 전체 {len(sub):,})", fontsize=11)
    plt.tight_layout()
    if save_to is not None:
        from pathlib import Path
        p = Path(save_to)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=110, bbox_inches="tight")
        print(f"저장: {p}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return picks

def by_group(df, col: str = "region", *, normal: str = "A7", top: int = 12,
             show: bool = True) -> dict:
    """오답을 **부위 같은 메타 컬럼**으로 쪼갭니다.

    왜 필요한가 — STEP 11 에서 헛알림 사진 30장을 눈으로 보니 발바닥 패드와 코가
    유난히 많았습니다. 그런데 "많아 보인다" 는 인상이지 숫자가 아닙니다. 부위별로
    헛알림 비율을 재면 그게 **정말 그 부위 문제인지**, 아니면 원래 그 부위 사진이
    많아서 그렇게 보이는 것인지 갈립니다.

    df 에 `label_orig`(정답), `pred`(예측), 그리고 `col` 이 있어야 합니다.
    돌려주는 것: 부위별 n / 헛알림 수 / 헛알림률, 전체 평균과의 차이.
    """
    import pandas as pd

    if col not in df.columns:
        print(f"[errors] '{col}' 컬럼이 없습니다 — 건너뜁니다 "
              f"(있는 것: {sorted(df.columns)[:12]}…)")
        return {}
    sub = df[df[col].notna() & (df[col].astype(str).str.strip() != "")]
    if len(sub) == 0:
        print(f"[errors] '{col}' 이 전부 비어 있습니다 — 원본 JSON 에 값이 없는 듯합니다")
        return {}

    norm = sub[sub["label_orig"] == normal]
    if len(norm) == 0:
        print(f"[errors] 정답이 '{normal}' 인 행이 없습니다")
        return {}
    base = float((norm["pred"] != normal).mean())

    g = norm.groupby(col, observed=True).agg(
        n=("pred", "size"), 헛알림=("pred", lambda v: int((v != normal).sum())))
    g["헛알림률"] = g["헛알림"] / g["n"]
    g["전체대비"] = g["헛알림률"] - base
    g = g.sort_values("헛알림", ascending=False)

    if show:
        from src.evaluate import pad_ko

        print("\n" + "=" * 74)
        print(f" 부위별 헛알림 — 정답 '{normal}' {len(norm):,}장, 전체 헛알림률 {base:.1%}")
        print("=" * 74)
        print(f"  {pad_ko(str(col), 24)}{'n':>7}{'헛알림':>8}{'비율':>8}{'전체대비':>10}")
        for k, r in g.head(top).iterrows():
            mark = " ←" if abs(r["전체대비"]) > 0.10 else ""
            print(f"  {pad_ko(str(k), 24)}{int(r['n']):>7,}{int(r['헛알림']):>8,}"
                  f"{r['헛알림률']:>8.1%}{r['전체대비']:>+10.1%}{mark}")
        if len(g) > top:
            print(f"  … 그 밖에 {len(g) - top}개 부위")
        print("=" * 74)
        print("  💡 '전체대비' 가 크게 +인 부위 = 그 부위가 유난히 헛알림이 납니다.")
        print("     n 이 작으면 흔들리니 n 도 같이 보세요.")
    return {"baseline": base, "table": g.reset_index().to_dict("records")}
