"""사진의 질감·화질을 재는 값들 — **여기가 유일한 정의입니다.**

왜 별도 모듈인가
----------------
`hair`(털처럼 가는 선의 양)는 두 곳에서 씁니다:

* `tools/false_alarm_stats.py` — 헛알림이 무엇과 상관 있나 (분석)
* `src/data.py` — 털 많은 정상 사진을 더 자주 뽑기 (학습)

두 곳에 따로 적으면 **갈라지고, 갈라져도 아무도 모릅니다.** 이 리포는
크롭 창(`crop.crop_window`)에서 이미 같은 실패를 했습니다 — 서빙이 "중앙 몇 %"
를 따로 계산하다 학습과 어긋났었죠. 그래서 정의를 하나만 둡니다.

⚠️ `hair` 는 **털 검출기가 아닙니다.** 1픽셀 차분의 크기라 털·각질·주름을
   구분하지 못합니다. "털처럼 가는 선이 얼마나 있나" 까지만 말합니다.
   실측 근거는 `docs/results/헛알림_사진통계_실측.md`.
"""

from __future__ import annotations

from pathlib import Path

STATS = ["blur", "bright", "contrast", "sat", "hair", "warm"]


def image_stats(path: str | Path) -> dict:
    """사진 한 장의 값들. **torch 없이** numpy + PIL 로만 계산합니다.

    | 이름 | 뜻 | 크면 |
    |---|---|---|
    | `blur` | 라플라시안 분산 | **선명** (작을수록 흐림) |
    | `bright` | 밝기 평균 (0~1) | 밝음 |
    | `contrast` | 밝기 표준편차 | 대비 큼 |
    | `sat` | 채도 평균 | 색이 진함 |
    | `hair` | 1픽셀 차분의 크기 | **털처럼 가는 선**이 많음 |
    | `warm` | R−B 평균 | 노란 조명 |

    못 여는 파일은 죽지 않고 전부 `nan` 을 돌려줍니다 — 3만 장을 돌리는데
    한 장 때문에 멈추면 안 되기 때문입니다.
    """
    import numpy as np
    from PIL import Image

    try:
        with Image.open(path) as im:
            a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    except Exception:                                          # noqa: BLE001
        return {k: float("nan") for k in STATS}

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
        "hair": float(np.abs(np.diff(g, axis=0)).mean()
                      + np.abs(np.diff(g, axis=1)).mean()),
        "warm": float((a[:, :, 0] - a[:, :, 2]).mean()),
    }


def hair_of(path: str | Path) -> float:
    """`hair` 하나만. 학습 샘플러가 3만 장을 돌릴 때 씁니다."""
    import numpy as np
    from PIL import Image

    try:
        with Image.open(path) as im:
            g = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
    except Exception:                                          # noqa: BLE001
        return float("nan")
    return float(np.abs(np.diff(g, axis=0)).mean()
                 + np.abs(np.diff(g, axis=1)).mean())


def hair_index(paths, cache: Path | None = None, workers: int = 8,
               verbose: bool = True):
    """경로 목록 → `hair` 배열. **한 번 재면 캐시에 남깁니다.**

    3만 장에 몇 분 걸리는데, 실험을 여러 번 돌리므로 매번 재면 낭비입니다.
    캐시는 경로를 키로 쓰므로 크롭이 그대로면 계속 유효합니다.
    """
    import numpy as np
    import pandas as pd

    paths = [str(p) for p in paths]
    have: dict[str, float] = {}
    if cache and Path(cache).exists():
        c = pd.read_parquet(cache)
        have = dict(zip(c["path"], c["hair"]))
        if verbose:
            print(f"[texture] 캐시 {len(have):,}장 — {cache}")

    todo = [p for p in paths if p not in have]
    if todo:
        if verbose:
            print(f"[texture] hair 계산 {len(todo):,}장 …")
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for p, v in zip(todo, ex.map(hair_of, todo)):
                have[p] = v
        if cache:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"path": list(have), "hair": list(have.values())}
                         ).to_parquet(cache)
            if verbose:
                print(f"[texture] 캐시 저장 {len(have):,}장 → {cache}")

    out = np.array([have.get(p, np.nan) for p in paths], dtype=np.float64)
    bad = int(np.isnan(out).sum())
    if bad and verbose:
        print(f"[texture] ⚠️ {bad:,}장을 못 읽었습니다 — 중앙값으로 채웁니다.")
    if bad:
        out[np.isnan(out)] = np.nanmedian(out) if bad < len(out) else 0.0
    return out
