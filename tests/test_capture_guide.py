"""촬영 가이드(capture guideline) 도출 검증.

배율 강건성(scale robustness)을 모델링으로 못 잡으면 입력을 제한해야 하고,
그러려면 "얼마나 가까이" 를 숫자로 말해야 합니다. 그 숫자를 뽑는 코드가
`robust.usable_range` 인데, **밴드 계산이 틀리면 가이드 문구가 틀립니다.**

실행: uv run python tests/test_capture_guide.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        if detail:
            print(f"        {detail}")
        FAILS.append(name)


def band_of(rows, tol):
    """usable_range 의 밴드 계산과 같은 규칙 (peak 에서 **연속으로** 만족하는 구간)."""
    peak_z, peak_f1 = max(rows, key=lambda r: r[1])
    floor = peak_f1 * (1 - tol)
    ok = [z for z, f in rows if f >= floor]
    srt = sorted(z for z, _ in rows)
    i = srt.index(peak_z)
    lo = hi = peak_z
    for j in range(i - 1, -1, -1):
        if srt[j] in ok:
            lo = srt[j]
        else:
            break
    for j in range(i + 1, len(srt)):
        if srt[j] in ok:
            hi = srt[j]
        else:
            break
    return lo, hi


def test_band_is_contiguous_around_peak():
    """peak 에서 떨어진 곳이 우연히 기준을 넘어도 밴드에 넣으면 안 됩니다.

    가이드는 "이 구간 안에서 찍으세요" 라서 **연속**이어야 합니다.
    0.5x 가 우연히 잘 나와도 0.7x 가 무너지면 0.5x 를 허용하면 안 됩니다.
    """
    rows = [(0.5, 0.98), (0.7, 0.50), (1.0, 1.00), (1.4, 0.97), (2.0, 0.60)]
    lo, hi = band_of(rows, 0.05)
    check("띄엄띄엄 만족하는 점은 제외", (lo, hi) == (1.0, 1.4), f"{lo}~{hi}")


def test_band_widens_with_tolerance():
    rows = [(0.5, 0.70), (0.7, 0.88), (1.0, 1.00), (1.4, 0.94), (2.0, 0.80)]
    tight = band_of(rows, 0.05)
    loose = band_of(rows, 0.20)
    check("허용치를 키우면 구간이 넓어진다",
          (loose[0] <= tight[0] and loose[1] >= tight[1]) and loose != tight,
          f"5%={tight}  20%={loose}")


def test_peak_need_not_be_1x():
    """최고점이 1x 가 아닐 수도 있습니다. 그때도 그 지점 기준으로 잡아야 합니다."""
    rows = [(0.5, 0.60), (0.7, 0.80), (1.0, 0.90), (1.4, 0.95), (2.0, 0.70)]
    lo, hi = band_of(rows, 0.10)
    check("최고점이 1.4x 일 때 그 주변으로 잡힌다", lo <= 1.4 <= hi, f"{lo}~{hi}")


def test_occupancy_conversion():
    """배율 → 화면 점유율. 보호자는 '1.2배' 를 모르지만 '화면 절반' 은 압니다.

    m1.5 크롭이면 1x 에서 병변이 화면 가로의 1/1.5 = 67% 입니다.
    """
    def occ(z, margin=1.5):
        return min(z / margin, 1.0)

    check("m1.5 · 1x → 67%", abs(occ(1.0) - 0.667) < 0.01, f"{occ(1.0):.3f}")
    check("m1.5 · 0.7x → 47%", abs(occ(0.7) - 0.467) < 0.01, f"{occ(0.7):.3f}")
    check("점유율은 100% 를 안 넘는다", occ(2.0) == 1.0, f"{occ(2.0):.3f}")


def test_real_measured_curve():
    """03 실측(384px, 2단계)으로 계산해봅니다 — 5개 점이라 성깁니다."""
    rows = [(0.5, 0.4066), (0.71, 0.4920), (1.0, 0.5489), (1.41, 0.5205), (2.0, 0.4632)]
    lo5, hi5 = band_of(rows, 0.05)
    lo10, hi10 = band_of(rows, 0.10)

    # ⚠️ 5% 밴드가 **한 점으로 무너집니다.** 1.41x 가 -5.2% 로 기준을 0.2%p 차이로
    #    놓치기 때문입니다. 10% 밴드도 0.71x 가 -10.4% 라 아래쪽이 막힙니다.
    #    가이드로 쓸 수 없는 결과이고, 이게 촘촘한 격자가 필요한 이유입니다.
    check("실측 5% 밴드가 한 점으로 무너진다 (격자가 성겨서)",
          (lo5, hi5) == (1.0, 1.0), f"{lo5}~{hi5}")
    check("실측 10% 밴드도 위쪽만 열린다",
          (lo10, hi10) == (1.0, 1.41), f"{lo10}~{hi10}")
    print("        → 5개 점(간격 √2)으로는 가이드를 못 씁니다.")
    print("           1.41x 가 -5.2%, 0.71x 가 -10.4% 로 둘 다 기준을 아슬하게 놓칩니다.")
    print("           0.85x / 1.2x 를 재봐야 실제 경계가 나옵니다.")


def test_usable_range_is_inference_only():
    """가이드 측정에 학습이 끼면 안 됩니다 (비싸고, 매번 답이 달라집니다)."""
    import inspect

    from src import robust

    src = inspect.getsource(robust.usable_range)
    check("학습 호출이 없다",
          "fit(" not in src and "backward" not in src and "optimizer" not in src)
    check("scale_stress 와 같은 뷰를 쓴다", "ZoomView" in src)


if __name__ == "__main__":
    print("촬영 가이드 도출 검증\n")
    for fn in (test_band_is_contiguous_around_peak,
               test_band_widens_with_tolerance,
               test_peak_need_not_be_1x,
               test_occupancy_conversion,
               test_real_measured_curve,
               test_usable_range_is_inference_only):
        fn()
    print()
    if FAILS:
        print(f"{len(FAILS)} check(s) failed:")
        for f in FAILS:
            print(f"  - {f}")
        sys.exit(1)
    print("all checks passed")
