"""tools/class_overlap.py — 두 클래스의 데이터 겹침 재기.

여기서 못 박는 것 (전부 실제로 당할 수 있는 실패입니다):

1. **컬럼 이름이 틀리면 멈춘다.** 계획서엔 `lesion_area_ratio` 로 적혀 있지만
   실물은 `area_ratio` 입니다. 틀린 이름으로 조용히 NaN 을 만들면
   **없는 걸 재면서 분석이 끝까지 돕니다.**
2. **bbox 를 하나도 못 읽으면 멈춘다.** 위와 같은 종류의 조용한 실패입니다.
3. **갈래 AUROC 가 부풀지 않는다.** 견종처럼 갈래가 많은 축을 그냥
   target-encode 하면 자기 자신을 보고 맞혀 1.0 이 나옵니다.
4. **판정이 문턱대로 나온다.** 심어둔 차이 → ①, 안 심으면 ③.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402

from tools.class_overlap import (build_axes, cat_auroc,          # noqa: E402
                                 cooccurrence, tvd)
import tools.class_overlap as co                                # noqa: E402


def make_df(n=600, seed=0, separate=False, co_pairs=0):
    """가짜 매니페스트. `separate=True` 면 A4 를 A1 과 다르게 찍은 것으로 만듭니다."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        lab = "A4" if i % 2 else "A1"
        w, h = 1920, 1080
        # separate 면 A4 의 bbox 를 뚜렷하게 작게
        base = 40 if (separate and lab == "A4") else 200
        bw = float(max(4, rng.normal(base, 10)))
        bh = bw * float(rng.uniform(0.9, 1.1))
        x, y = rng.uniform(0, w - bw), rng.uniform(0, h - bh)
        rows.append({
            "label": lab,
            "animal_id": f"dog{i // 6}",
            "region": rng.choice(["H", "B", "L", "A"]),
            "breed": f"b{rng.integers(0, 120)}",   # 갈래가 많고 라벨과 **무관**한 축
            "gender": rng.choice(["M", "F"]),
            "age": int(rng.integers(1, 15)),
            "synthetic": False,
            "img_w": w, "img_h": h,
            "bbox": [x, y, x + bw, y + bh],
            "area_ratio": (bw * bh) / (w * h),
            "n_lesion": int(rng.integers(1, 4)),
        })
    df = pd.DataFrame(rows)
    # 동시출현을 심습니다 — 같은 개체+부위에 두 라벨을 같이 둡니다
    for k in range(co_pairs):
        for lab in ("A1", "A4"):
            r = df.iloc[0].to_dict()
            r.update({"label": lab, "animal_id": f"both{k}", "region": "H"})
            df = pd.concat([df, pd.DataFrame([r])], ignore_index=True)
    return df


def test_missing_column_stops():
    df = make_df(60).rename(columns={"area_ratio": "lesion_area_ratio"})
    try:
        build_axes(df)
    except SystemExit as e:
        assert "area_ratio" in str(e), str(e)
    else:
        raise AssertionError("컬럼이 없는데 그냥 지나갔습니다 — 조용한 NaN 이 됩니다")
    print("✅ 컬럼 이름이 틀리면 멈춥니다")


def test_unreadable_bbox_stops():
    df = make_df(60)
    df["bbox"] = None
    try:
        build_axes(df)
    except SystemExit as e:
        assert "bbox" in str(e), str(e)
    else:
        raise AssertionError("bbox 를 하나도 못 읽었는데 그냥 지나갔습니다")
    print("✅ bbox 를 못 읽으면 멈춥니다")


def test_axes_are_real_numbers():
    d = build_axes(make_df(200))
    for k, _ in co.NUMERIC:
        assert np.isfinite(d[k]).mean() > 0.9, f"{k} 가 대부분 NaN 입니다"
    # boxaspect 는 0.9~1.1 비율로 만들었으니 1 근처여야 합니다
    assert 0.95 < np.nanmedian(d["boxaspect"]) < 1.15
    print("✅ 축이 실제 숫자로 채워집니다")


def test_cat_auroc_does_not_inflate():
    """견종처럼 갈래가 120개면 그냥 인코딩은 부풀어 오릅니다 — 교차 인코딩은 0.5 근처."""
    df = make_df(600)
    y = (df["label"] == "A4").astype(int)
    naive = df.groupby("breed")["label"].transform(lambda s: (s == "A4").mean())
    from src.effectsize import auroc
    naive_a = auroc(naive[y == 1], naive[y == 0])
    loo_a = cat_auroc(df["breed"], y)
    assert naive_a > 0.75, f"부풀림 재현 실패 {naive_a}"
    assert abs(loo_a - 0.5) < 0.12, f"LOO 가 부풀었습니다 {loo_a}"
    print(f"✅ 갈래 AUROC 부풀림 차단: 그냥 {naive_a:.3f} → 교차 {loo_a:.3f}")


def test_cat_auroc_constant_column_is_half():
    """갈래가 **하나뿐**인 축은 0.5 여야 합니다.

    leave-one-out 으로 짜면 여기서 **0.000** 이 나옵니다 — 자기 y 를 빼는 바람에
    양성이 항상 음성보다 낮아지고, 순위만 보는 AUROC 는 그걸 완벽한 분리로 읽습니다.
    실제로 `synthetic`(전부 False)이 "차이 있음" 으로 찍혔던 버그입니다.
    """
    df = make_df(400)
    y = (df["label"] == "A4").astype(int)
    a = cat_auroc(df["synthetic"], y)
    # 문턱 0.10 — LOO 버그(0.000)를 잡되 fold 잡음에는 안 걸리게
    assert abs(a - 0.5) < 0.10, f"갈래 하나짜리 축이 {a:.3f} 입니다"
    print(f"✅ 갈래가 하나뿐인 축은 0.5 근처: {a:.3f}")


def test_tvd_and_cooccurrence():
    assert tvd(["a"] * 10, ["a"] * 10) == 0.0
    assert abs(tvd(["a"] * 10, ["b"] * 10) - 1.0) < 1e-9
    df = make_df(120, co_pairs=0)
    df["animal_id"] = "solo" + df.index.astype(str)     # 한 묶음에 한 장씩
    assert cooccurrence(df, "A1", "A4", ["animal_id"]) == 0.0
    print("✅ TVD·동시출현이 극단값에서 맞습니다")


def test_verdict_separated_vs_not(tmp=None):
    import io
    from contextlib import redirect_stdout

    for sep, want in [(True, "①"), (False, "③")]:
        d = build_axes(make_df(600, separate=sep))
        buf, lines = io.StringIO(), []
        with redirect_stdout(buf):
            res = co.analyze(d, "A4", "A1", lines)
        assert res["verdict"].startswith(want), \
            f"separate={sep} 인데 판정이 {res['verdict']}"
    print("✅ 판정이 문턱대로 나옵니다 (심으면 ①, 안 심으면 ③)")


def test_cli_runs(tmp_path=None):
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.parquet"
        make_df(400, separate=True).to_parquet(p)
        out = Path(td) / "r.md"
        r = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "class_overlap.py"),
             "--manifest", str(p), "--out", str(out)],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stdout + r.stderr
        # --out 은 ROOT 기준 상대경로로 붙습니다 — 절대경로면 그대로
        text = out.read_text(encoding="utf-8")
        assert "판정" in text and "A4 vs A1" in text
    print("✅ CLI 가 끝까지 돌고 마크다운을 씁니다")


if __name__ == "__main__":
    test_missing_column_stops()
    test_unreadable_bbox_stops()
    test_axes_are_real_numbers()
    test_cat_auroc_does_not_inflate()
    test_cat_auroc_constant_column_is_half()
    test_tvd_and_cooccurrence()
    test_verdict_separated_vs_not()
    test_cli_runs()
    print("\n모두 통과")
