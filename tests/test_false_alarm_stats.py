"""tools/false_alarm_stats.py — 통계가 맞나 · 문턱이 안 흔들리나.

⚠️ 이 검사의 요점은 **판정 문턱을 못 박는 것**입니다. 결과를 보고 문턱을
고치면 무슨 숫자가 나와도 성공담이 됩니다 (CLAUDE.md 규칙 2).

    uv run python tests/test_false_alarm_stats.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np                                              # noqa: E402

import false_alarm_stats as F                                   # noqa: E402

ok = fail = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {name}")
    else:
        fail += 1
        print(f"  FAIL  {name}  {detail}")


print("\n[1] 통계 함수가 아는 답을 맞히나")
rng = np.random.default_rng(0)
x, y = rng.normal(1.0, 1, 2000), rng.normal(0.0, 1, 2000)
check("d(1σ 차이) ≈ 1", 0.85 <= F.cohens_d(x, y) <= 1.15, str(F.cohens_d(x, y)))
check("AUROC(1σ 차이) ≈ 0.76", 0.72 <= F.auroc(x, y) <= 0.80, str(F.auroc(x, y)))
z = rng.normal(0.0, 1, 2000)
check("d(차이 없음) ≈ 0", abs(F.cohens_d(z, y)) < 0.15, str(F.cohens_d(z, y)))
check("AUROC(차이 없음) ≈ 0.5", abs(F.auroc(z, y) - 0.5) < 0.05, str(F.auroc(z, y)))
check("전부 동점이면 0.5", F.auroc(np.ones(50), np.ones(50)) == 0.5)
check("표본이 없으면 nan", np.isnan(F.auroc(np.array([]), y)))
check("한쪽이 1개면 d 는 nan", np.isnan(F.cohens_d(np.array([1.0]), y)))

print("\n[2] 판정 문턱 — 결과 보고 못 바꾸게 못 박습니다")
check("문턱 값 자체", (F.D_STRONG, F.A_STRONG, F.D_WEAK, F.A_WEAK)
      == (0.5, 0.15, 0.2, 0.06))
check("진짜 신호 → 차이 있음", F.verdict(1.16, 0.228) == "**차이 있음**")
check("방향이 반대여도 잡는다 (AUROC<0.5)", F.verdict(-1.16, 0.228) == "**차이 있음**")
# ★ 이게 이 검사의 핵심입니다. 퍼짐이 작으면 실제 차이가 없어도 d 가 큽니다.
check("d 만 크고 AUROC 는 0.5 근처 → 차이 없음", F.verdict(0.69, 0.52) == "차이 없음")
check("AUROC 만 크고 d 는 작음 → 차이 없음", F.verdict(0.05, 0.72) == "차이 없음")
check("둘 다 애매 → 조금", F.verdict(0.3, 0.10 + 0.5) == "조금")
check("아무것도 아님", F.verdict(0.05, 0.51) == "차이 없음")
check("못 잰 건 못 잼", F.verdict(float("nan"), 0.5) == "못 잼")

print("\n[3] 사진 통계가 방향을 맞게 재나")
import tempfile                                                 # noqa: E402

from PIL import Image, ImageFilter                              # noqa: E402

td = Path(tempfile.mkdtemp())
lines = np.tile(np.array([40, 200], np.uint8), (256, 128))
sharp = td / "sharp.jpg"
Image.fromarray(np.dstack([lines] * 3)).save(sharp, quality=95)
blur = td / "blur.jpg"
Image.open(sharp).filter(ImageFilter.GaussianBlur(4)).save(blur, quality=95)
dark = td / "dark.jpg"
Image.fromarray(np.full((256, 256, 3), 30, np.uint8)).save(dark, quality=95)
warm = td / "warm.jpg"
Image.fromarray(np.dstack([np.full((256, 256), 200, np.uint8),
                           np.full((256, 256), 170, np.uint8),
                           np.full((256, 256), 90, np.uint8)])).save(warm, quality=95)

S, B, D, W = (F.image_stats(str(p)) for p in (sharp, blur, dark, warm))
check("흐리게 하면 blur 가 준다", B["blur"] < S["blur"] * 0.1,
      f"{S['blur']:.3f} → {B['blur']:.3f}")
check("흐리게 하면 hair 도 준다", B["hair"] < S["hair"] * 0.1,
      f"{S['hair']:.3f} → {B['hair']:.3f}")
check("어두운 사진은 bright 가 낮다", D["bright"] < 0.2, str(D["bright"]))
check("노란 조명은 warm 이 양수", W["warm"] > 0.3, str(W["warm"]))
check("노란 조명은 sat 이 높다", W["sat"] > 0.3, str(W["sat"]))
check("못 여는 파일은 nan (죽지 않음)", np.isnan(F.image_stats("/없음.jpg")["blur"]))
check("재는 값은 6개 그대로", set(S) == set(F.STATS) and len(F.STATS) == 6)

print("\n[4] 두 값이 같은 걸 재는지 가려내나")
# ⚠️ 처음엔 control 을 사분위로 묶었는데, 독립 성분이 **0인 데이터에서도**
#    "독립" 이 나왔습니다 (AUROC 0.657). 칸 안에서 control 이 계속 변해서
#    target 이 그걸 주워담기 때문입니다. 그래서 순위 잔차로 바꿨습니다.
#    이 검사는 그 실패가 되돌아오지 않게 막습니다.
import pandas as pd                                             # noqa: E402

rng2 = np.random.default_rng(0)
N = 4000
for why, indep_w, bad_from, expect in [
    ("hair 가 blur 의 다른 이름 (독립 성분 0)", 0.0, "blur", "같은 것"),
    ("hair 의 독립 성분이 오답을 만듦",          1.0, "indep", "독립"),
    ("독립 성분은 있지만 오답은 blur 가 만듦",   1.0, "blur", "같은 것"),
]:
    blur = rng2.normal(0, 1, N)
    indep = rng2.normal(0, 1, N) * indep_w
    # 제곱근 관계 — 원값에 직선을 맞추면 휘어진 부분이 남습니다 (순위를 쓰는 이유)
    hair = np.sqrt(np.abs(blur)) * np.sign(blur) * 2 + indep + rng2.normal(0, .2, N)
    drv = blur if bad_from == "blur" else indep
    bad_m = (drv + rng2.normal(0, .5, N)) > .6

    d = pd.DataFrame({"blur": blur, "hair": hair})
    fa_m = pd.Series(bad_m, index=d.index)
    tn_m = ~fa_m
    res = F.residualize(d["hair"], d["blur"])
    got = F.auroc(res[bad_m], res[~bad_m])
    raw = F.auroc(d.loc[bad_m, "hair"], d.loc[~bad_m, "hair"])
    indep_call = abs(got - 0.5) >= F.A_STRONG
    check(f"{why} → {expect}",
          indep_call == (expect == "독립"), f"raw {raw:.3f} → resid {got:.3f}")

check("단조 변환에 안 흔들린다 (로그를 씌워도 같은 답)",
      abs(F.auroc(F.residualize(np.exp(hair), np.exp(blur))[bad_m],
                  F.residualize(np.exp(hair), np.exp(blur))[~bad_m])
          - F.auroc(F.residualize(hair, blur)[bad_m],
                    F.residualize(hair, blur)[~bad_m])) < 0.02)

print("\n" + "=" * 60)
print(f" 통과 {ok} / {ok + fail}")
sys.exit(1 if fail else 0)
