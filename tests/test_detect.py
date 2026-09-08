"""STEP 42 — 검출기 판정 기준과 배선을 못 박습니다 (작업 규칙 2·3).

제일 중요한 감시: **자극이 정답에서 파생되면 안 됩니다.** STEP 36 에서 사람에게
병변이 가운데 놓인 크롭을 보여주고 "네모를 그려보라" 했다가 결과를 통째로
버렸습니다. 검출 데이터셋도 같은 함정에 빠질 수 있습니다 — 정답이 항상
한가운데면 모델이 "가운데" 만 외우고 밴드 판정이 거짓말합니다.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import detect as D
from src.config import ZOOM_ALLOW, ZOOM_CENTER_MAX
from src.experiments import DETECT_MIN_COVERAGE_GAIN, DETECT_MIN_USABLE

ok = fail = 0


def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {name}")
    else:
        fail += 1
        print(f"  FAIL  {name}" + (f"\n        {detail}" if detail else ""))


print("[1] 판정 기준이 코드에 있고, 다른 갈래와 **같은 잣대**인가")
from src.experiments import CAM_BOX_MIN_USABLE, FIXEDSCALE_MIN_COVERAGE_GAIN  # noqa: E402

check("DETECT_MIN_USABLE 가 창탐지기와 같은 값",
      DETECT_MIN_USABLE == CAM_BOX_MIN_USABLE,
      f"{DETECT_MIN_USABLE} vs {CAM_BOX_MIN_USABLE}")
check("DETECT_MIN_COVERAGE_GAIN 이 고정배율과 같은 값",
      DETECT_MIN_COVERAGE_GAIN == FIXEDSCALE_MIN_COVERAGE_GAIN,
      f"{DETECT_MIN_COVERAGE_GAIN} vs {FIXEDSCALE_MIN_COVERAGE_GAIN}")

print("\n[2] 밴드를 **베껴 적지 않고** 출처에서 끌어 쓰는가")
src = Path(__file__).resolve().parents[1] / "src" / "detect.py"
text = src.read_text(encoding="utf-8")
check("ZOOM_ALLOW 를 import 한다", "ZOOM_ALLOW" in text)
check("ZOOM_CENTER_MAX 를 import 한다", "ZOOM_CENTER_MAX" in text)
# 밴드 숫자를 직접 적어두면 출처가 바뀌어도 안 따라옵니다 (촬영 밴드에서 당함)
for lit in ("0.71", "1.67", "0.7,", "1.4,"):
    check(f"밴드 숫자 '{lit}' 를 코드에 안 박았다", lit not in text.split('"""')[-1])

print("\n[3] ★ 자극이 정답에서 파생되면 잡아내는가")
# 정답이 **항상 한가운데**인 가짜 데이터 — 모델이 "가운데" 만 외우면 만점
n = 200
true_c = np.tile([0.4, 0.4, 0.6, 0.6], (n, 1))          # 전부 정중앙
pred_c = np.tile([0.4, 0.4, 0.6, 0.6], (n, 1))
rep_c = D.band_report(pred_c, true_c)
check("가운데 고정이면 둘 다 100% 로 나온다 (그래서 위험합니다)",
      rep_c["both"] == 1.0, str(rep_c["both"]))
check("그때 중심 어긋남이 0 이다 — **이 값이 0 이면 자극을 의심하세요**",
      rep_c["off_median"] == 0.0)

# 흩어진 정답 — 진짜 측정
rng = np.random.default_rng(0)
cx = rng.uniform(0.2, 0.8, n)
cy = rng.uniform(0.2, 0.8, n)
w = rng.uniform(0.08, 0.5, n)
true = np.stack([cx - w / 2, cy - w / 2, cx + w / 2, cy + w / 2], 1)
rep_s = D.band_report(true, true)                        # 완벽한 예측
check("완벽히 맞히면 둘 다 100%", rep_s["both"] == 1.0)
check("흩어진 자극은 중심 이탈이 0 이 아니다", rep_s["off_median"] >= 0.0)

print("\n[4] 밴드 계산이 촬영 가이드와 **같은 규칙**인가")
# 네모를 r 배로 그리면 환산 줌은 1/r → 밴드를 뒤집습니다
lo, hi = 1 / ZOOM_ALLOW[1], 1 / ZOOM_ALLOW[0]
check("보고가 뒤집은 밴드를 쓴다",
      abs(rep_s["band"][0] - lo) < 1e-9 and abs(rep_s["band"][1] - hi) < 1e-9,
      f"{rep_s['band']} vs [{lo}, {hi}]")
check("중심 문턱이 ZOOM_CENTER_MAX", rep_s["center_max"] == ZOOM_CENTER_MAX)

# 밴드 밖 크기는 걸러야 합니다
big = true.copy()
c = (big[:, :2] + big[:, 2:]) / 2
half = (big[:, 2:] - big[:, :2]) / 2 * 3.0               # 3배로 그림
rep_b = D.band_report(np.concatenate([c - half, c + half], 1), true)
check("★ 3배로 그리면 배율 밴드에서 걸린다", rep_b["in_size"] == 0.0,
      f"{rep_b['in_size']:.1%}")

print("\n[5] 크기를 **로그**로 재는가 (작은 병변이 묻히지 않게)")
check("box_loss 가 log 를 쓴다", "torch.log" in text)
# 같은 **비율** 오차면 큰 병변이든 작은 병변이든 손실이 비슷해야 합니다
import torch  # noqa: E402

def loss_for(size, ratio):
    t = torch.tensor([[.5 - size / 2, .5 - size / 2, .5 + size / 2, .5 + size / 2]])
    p = torch.tensor([[.5, .5, size * ratio, size * ratio]])
    return float(D.box_loss(p, t))

small, large = loss_for(0.06, 1.5), loss_for(0.50, 1.5)
check("★ 같은 비율 오차면 크기와 무관하게 손실이 비슷하다",
      abs(small - large) < 0.02, f"작은 {small:.4f} vs 큰 {large:.4f}")

print("\n[6] 헤드 초기값이 **가운데 · 병변 중앙값 크기**인가")
from src.experiments import FIXEDSCALE_LESION_FRAC  # noqa: E402

net = D.BoxHead(pretrained=False, img_size=384)
with torch.no_grad():
    out = net(torch.zeros(1, 3, 384, 384))[0].tolist()
check("초기 중심이 화면 가운데", abs(out[0] - .5) < 1e-3 and abs(out[1] - .5) < 1e-3,
      str(out[:2]))
check("초기 크기가 실측 병변 중앙값",
      abs(out[2] - FIXEDSCALE_LESION_FRAC) < 1e-3, str(out[2]))

print("\n[7] 노트북 13 이 규약을 지키는가")
nb = (Path(__file__).resolve().parents[1] / "notebooks" / "13_병변_검출기.ipynb")
check("노트북이 있다", nb.is_file())
if nb.is_file():
    t = nb.read_text(encoding="utf-8")
    check("holdout 을 안 연다고 적혀 있다", "holdout 을 **안 엽니다**" in t)
    check("개체 단위 분할을 하고 겹침을 assert 한다", "겹칩니다" in t)
    check("크롭 데이터셋을 쓰면 안 된다고 적혀 있다", "가운데 놓고 자른" in t)
    check("★ 밴드만 통과해도 채택 아니라고 적혀 있다", "커버리지가 안 오르면" in t)

print()
print("[8] ★ 하한선을 **같이 찍는가** (STEP 42 에서 당한 것)")
# 검출기 배율 75.0% 를 보고 "배율은 풀렸다" 로 읽을 뻔했습니다 —
# 같은 val 에서 하한선이 72.3% 입니다 (순이득 +2.7%p).
rng = np.random.default_rng(0)
c = rng.uniform(0.2, 0.8, (400, 2))
s = rng.uniform(0.08, 0.45, 400)
t_rand = np.stack([c[:, 0] - s / 2, c[:, 1] - s / 2,
                   c[:, 0] + s / 2, c[:, 1] + s / 2], 1)
rep_r = D.band_report(t_rand.copy(), t_rand)          # 완벽한 예측
check("band_report 가 baseline 을 같이 낸다", "baseline" in rep_r)
b = rep_r.get("baseline", {})
check("하한선 크기가 실측 병변 중앙값이다",
      b.get("size_frac") == FIXEDSCALE_LESION_FRAC, str(b.get("size_frac")))
check("완벽한 예측은 1.0 인데 하한선은 그보다 낮다",
      rep_r["both"] == 1.0 and b.get("both", 1.0) < 1.0,
      str((rep_r["both"], b.get("both"))))
# 하한선을 그대로 예측으로 넣으면 순이득이 0 이어야 합니다
hf = FIXEDSCALE_LESION_FRAC / 2
p_base = np.tile([0.5 - hf, 0.5 - hf, 0.5 + hf, 0.5 + hf], (len(t_rand), 1))
rep_b = D.band_report(p_base, t_rand)
check("★ 하한선을 예측으로 넣으면 순이득 0",
      abs(rep_b["both"] - rep_b["baseline"]["both"]) < 1e-9,
      str((rep_b["both"], rep_b["baseline"]["both"])))
src_t = (Path(__file__).resolve().parents[1] / "src" / "detect.py").read_text(encoding="utf-8")
check("print_report 가 하한선을 찍는다", "하한선" in src_t.split("def print_report")[-1])

print("\n" + "=" * 60)
print(f" 통과 {ok} / {ok + fail}")
raise SystemExit(1 if fail else 0)
