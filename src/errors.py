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
