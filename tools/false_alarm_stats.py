"""헛알림·놓침이 **사진의 무엇** 과 상관 있나 — 눈이 아니라 숫자로.

    uv run --extra train python tools/false_alarm_stats.py --release <릴리스폴더>

왜 이 도구가 있나
-----------------
STEP 11 에서 오류가 A2(비듬·각질)에 몰린다는 건 알아냈는데 **원인은 못 쟀습니다.**
다음 수는 "헛알림 난 정상 사진 30장을 눈으로 보고 공통점 찾기" 였는데,
수의사가 아니면 못 하는 일이라 접었습니다.

접는 게 맞았지만 질문 자체는 남습니다. 그래서 **눈 대신 자를 댑니다.**
털·흐림·조명은 병변 지식 없이도 잴 수 있는 값이고, 그게 헛알림과 상관 있으면
"촬영 가이드" 로 잡을 수 있는 문제, 없으면 모델·라벨 문제입니다.

무엇을 재나 (전부 병변 지식이 필요 없는 값)
-------------------------------------------
| 이름 | 뜻 | 크면 |
|---|---|---|
| `blur` | 라플라시안 분산 | **작을수록** 흐림 |
| `bright` | 밝기 평균 (0~1) | 밝음 |
| `contrast` | 밝기 표준편차 | 대비 큼 |
| `sat` | 채도 평균 | 색이 진함 |
| `hair` | 고주파 에너지 (털·질감 대용) | 결이 많음 |
| `warm` | R−B 평균 | 노란 조명 |

⚠️ `hair` 는 **털 검출기가 아닙니다.** 가는 선이 많으면 커지는 값이고,
   각질·주름도 같이 올립니다. "질감이 많다" 까지만 말할 수 있습니다.

읽는 법
-------
같은 정상 사진인데 **틀린 쪽과 맞은 쪽의 값이 다른가** 를 봅니다.

* `Δ` = 두 평균의 차이 (그 값의 원래 단위)
* `d`  = 퍼짐으로 나눈 차이(Cohen's d). |d| < 0.2 는 사실상 차이 없음
* `AUROC` = 그 값 **하나만** 보고 오류 난 사진을 골라낼 수 있나.
  0.5 = 못 함. **0.5 미만이면 방향이 반대**라는 뜻이고 세기는 |AUROC − 0.5| 입니다

⚠️ **`d` 하나만 보면 속습니다.** 퍼짐이 아주 작으면 실제 차이가 없다시피 해도
   `d` 가 커집니다 (검증 때 `bright` 가 0.5001 vs 0.4981 인데 d=0.69 였습니다).
   그래서 판정은 **d 와 AUROC 를 둘 다** 넘어야 합니다:

       |d| ≥ 0.5  **그리고**  |AUROC − 0.5| ≥ 0.15   →  차이 있음

둘 다 작으면 "그 사진들엔 공통점이 없다" 가 결론이고, 그것도 결과입니다 —
촬영 가이드로 못 잡는다는 뜻이니까요.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STATS = ["blur", "bright", "contrast", "sat", "hair", "warm"]


def image_stats(path: str) -> dict:
    """사진 한 장의 값들. torch 없이 numpy + PIL 로만."""
    import numpy as np
    from PIL import Image

    try:
        im = Image.open(path).convert("RGB")
    except Exception:                                          # noqa: BLE001
        return {k: float("nan") for k in STATS}
    a = np.asarray(im, dtype=np.float32) / 255.0
    g = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)

    # 라플라시안 = 2차 미분. 초점이 나가면 급변이 사라져 분산이 줄어듭니다.
    lap = (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
           - 4 * g[1:-1, 1:-1])
    mx, mn = a.max(axis=2), a.min(axis=2)
    return {
        "blur": float(lap.var()),
        "bright": float(g.mean()),
        "contrast": float(g.std()),
        "sat": float(((mx - mn) / (mx + 1e-6)).mean()),
        # 가로/세로 1픽셀 차분의 크기 — 가는 결이 많을수록 커집니다
        "hair": float(np.abs(np.diff(g, axis=0)).mean()
                      + np.abs(np.diff(g, axis=1)).mean()),
        "warm": float((a[:, :, 0] - a[:, :, 2]).mean()),
    }


def cohens_d(x, y) -> float:
    import numpy as np

    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    s = np.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1))
                / (len(x) + len(y) - 2))
    return float((x.mean() - y.mean()) / s) if s else float("nan")


def auroc(pos, neg) -> float:
    """값 하나만으로 pos 를 neg 와 가를 수 있나 (Mann-Whitney U)."""
    import numpy as np

    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = allv.argsort().argsort().astype(float) + 1
    # 동점 처리 — 각질처럼 값이 뭉치는 경우가 있어서 필요합니다
    order = np.sort(allv)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and order[j + 1] == order[i]:
            j += 1
        if j > i:
            r[(allv >= order[i]) & (allv <= order[j])] = (i + j) / 2 + 1
        i = j + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


# 판정 문턱 — **결과를 보기 전에** 못 박습니다 (CLAUDE.md 규칙 2).
# d 하나로는 안 됩니다: 퍼짐이 작으면 실제 차이가 없어도 d 가 커집니다.
D_STRONG, A_STRONG = 0.5, 0.15         # 둘 **다** 넘어야 "차이 있음"
D_WEAK, A_WEAK = 0.2, 0.06


def verdict(d: float, a: float) -> str:
    import math

    if math.isnan(d) or math.isnan(a):
        return "못 잼"
    da = abs(a - 0.5)
    if abs(d) >= D_STRONG and da >= A_STRONG:
        return "**차이 있음**"
    if abs(d) >= D_WEAK and da >= A_WEAK:
        return "조금"
    return "차이 없음"


def compare(df, mask_bad, mask_good, title: str, n_bad_label: str,
            n_good_label: str) -> None:
    print(f"\n  ── {title} ──")
    nb, ng = int(mask_bad.sum()), int(mask_good.sum())
    print(f"     {n_bad_label} {nb:,}장  vs  {n_good_label} {ng:,}장")
    if nb < 30 or ng < 30:
        print("     ⚠️ 표본이 30장 미만이라 아래 숫자는 못 믿습니다.")
    print(f"\n     {'값':<10}{'틀린 쪽':>10}{'맞은 쪽':>10}{'Δ':>10}"
          f"{'d':>7}{'AUROC':>8}   판정")
    print("     " + "─" * 70)
    for k in STATS:
        b, g = df.loc[mask_bad, k], df.loc[mask_good, k]
        d, ar = cohens_d(b, g), auroc(b, g)
        print(f"     {k:<10}{b.mean():>10.4f}{g.mean():>10.4f}"
              f"{b.mean() - g.mean():>10.4f}{d:>7.2f}{ar:>8.3f}   {verdict(d, ar)}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--release", required=True, help="06 이 만든 릴리스 폴더")
    ap.add_argument("--tag", default=None, help="쓸 크롭 태그 (기본: 1단계 태그)")
    ap.add_argument("--limit", type=int, default=0, help="맛보기로 N장만")
    ap.add_argument("--out", default="docs/results/헛알림_사진통계_실측.md")
    a = ap.parse_args(argv)

    import numpy as np
    import pandas as pd

    from src import agent, crop, env, stages

    # ── 0) 경로부터 확인합니다 — 20분 돌린 뒤 오타로 죽으면 아깝습니다 ──
    rel = Path(a.release).expanduser()
    if not rel.exists():
        raise SystemExit(
            f"릴리스 폴더가 없습니다: {rel}\n"
            "   탐색기에서 checkpoints 가 들어 있는 폴더의 주소를 복사해 붙이세요.\n"
            "   폴더 이름에 공백이 있으면 \"따옴표\" 로 감싸세요.")
    hits = sorted(rel.glob("**/stage1_*/best.pt"))
    if not hits:
        raise SystemExit(
            f"{rel} 안에서 stage1_*/best.pt 를 못 찾았습니다.\n"
            f"   여기 있는 것: {[p.name for p in sorted(rel.iterdir())[:10]]}\n"
            "   압축을 풀면 한 겹 더 감싸져 있는 일이 흔합니다 (release/release/…).")
    print(f"[릴리스] {hits[0].parent.name}")
    for q in sorted(rel.glob("**/stage2_*/best.pt")):
        print(f"         {q.parent.name}")
    if not (list(rel.glob("**/temperature.json")) or []):
        print("         ⚠️ temperature.json 이 없습니다 — 이 릴리스는 보정 전입니다.")

    tag = a.tag or agent.STAGE1_TAG
    mf = env.work_root() / "manifests" / "manifest_final.parquet"
    if not mf.exists():
        raise SystemExit(f"매니페스트가 없습니다: {mf}")

    df = pd.read_parquet(mf)
    if "is_holdout" not in df.columns:
        raise SystemExit("매니페스트에 is_holdout 이 없습니다 — --finalize 를 돌렸나요?")
    df = df[df["is_holdout"].astype(bool)].reset_index(drop=True)
    if df.empty:
        raise SystemExit("holdout 행이 없습니다. --finalize 를 돌렸나요?")
    df = stages.to_stage1(df)
    print(f"[holdout] {len(df):,}장")

    # 크롭 경로는 매니페스트 컬럼이 아니라 **태그로 다시 계산**합니다 —
    # manifest 의 crop_path 는 --margins 의 첫 항목 것이라 f320 이 아닐 수 있습니다.
    out_dir = env.work_root() / "crops"
    df["path"] = df["image_path"].apply(
        lambda p: str(crop._out_path(p, out_dir, tag)))
    have = df["path"].apply(lambda p: Path(p).exists())
    if not have.all():
        print(f"  ⚠️ 크롭 {tag} 가 {int((~have).sum()):,}장 없습니다 — 그 행은 뺍니다.")
        df = df[have].reset_index(drop=True)
    if a.limit:
        df = df.sample(min(a.limit, len(df)), random_state=0).reset_index(drop=True)
        print(f"  (맛보기 {len(df):,}장)")

    # ── 1) 모델이 뭐라고 했나 ──
    ag = agent.ScreeningAgent.from_release(a.release, device="cpu", stage1_only=True)
    print(f"[모델] 임계값 {ag.thr:.4f}  (CPU 추론 — 시간이 좀 걸립니다)")

    # ⚠️ `ag.screen()` 을 쓰면 안 됩니다. 그건 사진을 **다시 자릅니다** —
    #    여기 있는 건 이미 학습이 쓴 크롭이라, 자르면 두 번 자르는 게 됩니다.
    #    1단계 엔진에 크롭 경로를 바로 넘깁니다.
    scores = np.empty(len(df), dtype=np.float32)
    for i, p in enumerate(df["path"]):
        scores[i] = dict(ag.s1.predict(p).topk).get(ag._ab, 0.0)
        if (i + 1) % 500 == 0:
            print(f"    {i + 1:,}/{len(df):,}")
    df["score"] = scores
    df["said_abnormal"] = df["score"] >= ag.thr
    df["is_normal"] = df["label"] == stages.NORMAL_LABEL

    # ── 2) 사진의 값들 ──
    print("[사진 통계] 재는 중 …")
    st = pd.DataFrame([image_stats(p) for p in df["path"]], index=df.index)
    df = pd.concat([df, st], axis=1)

    # ── 3) 비교 ──
    fa = df["is_normal"] & df["said_abnormal"]          # 헛알림
    tn = df["is_normal"] & ~df["said_abnormal"]         # 맞게 넘긴 정상
    fn = ~df["is_normal"] & ~df["said_abnormal"]        # 놓친 병변
    tp = ~df["is_normal"] & df["said_abnormal"]         # 잡은 병변

    print("\n" + "=" * 68)
    print(f" 헛알림 {int(fa.sum()):,} / 놓침 {int(fn.sum()):,}"
          f"  (정상 {int(df['is_normal'].sum()):,} / 병변 {int((~df['is_normal']).sum()):,})")
    print("=" * 68)
    compare(df, fa, tn, "헛알림 — 정상인데 병원 보낸 사진", "헛알림", "맞게 넘김")
    compare(df, fn, tp, "놓침 — 병변인데 괜찮다고 한 사진", "놓침", "잡음")

    print("\n  ── 어떻게 읽나 ──")
    print(f"  판정 문턱: |d| ≥ {D_STRONG} **그리고** |AUROC − 0.5| ≥ {A_STRONG}")
    print("  (d 만 보면 속습니다 — 퍼짐이 작으면 차이가 없어도 d 가 커집니다)")
    print("\n  '차이 있음' 이 하나라도 있으면 → **촬영 가이드로 잡을 수 있는 문제**입니다.")
    print("     그 값을 촬영 단계에서 걸러내면 헛알림이 줄어듭니다.")
    print("  전부 '차이 없음' 이면 → 사진 조건 문제가 **아닙니다.** 모델이나 라벨 쪽이고,")
    print("     촬영 가이드를 아무리 손봐도 안 줄어듭니다. 이것도 결론입니다.")
    print("  AUROC < 0.5 는 틀림이 아니라 **방향이 반대**라는 뜻입니다 "
          "(예: 헛알림 쪽이 더 흐림).")
    print(f"\n  결과를 {a.out} 에 적어주세요 (규칙 1 — 실측만).")


if __name__ == "__main__":
    main()
