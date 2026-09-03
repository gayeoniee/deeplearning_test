"""보정값이 **실제로 서빙까지 도착하는가.**

두 번 놓친 곳이라 테스트로 못 박습니다:
  1. `export_release` 가 temperature.json 을 안 옮겨서, 릴리스로 서빙하면
     `Engine.load` 가 조용히 T=1.0 으로 돌아갔습니다 — 보정 안 된 확률이
     보호자에게 갑니다. 아무 에러도 안 납니다.
  2. `calibrate.apply()` 는 **이미 확률**을 돌려주는데 `stages.stage1_scores()`
     가 softmax 를 또 겁니다. 이중 softmax 로 짰다가 ECE 가 0.03 → 0.20 으로
     나빠져서 잡았습니다.

    uv run --extra train python tests/test_calibration_wiring.py
"""
from __future__ import annotations

import inspect
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import agent, calibrate, evaluate, stages, train          # noqa: E402

ok = fail = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {name}")
    else:
        fail += 1
        print(f"  FAIL  {name}  {detail}")


# ── 1. apply() 는 확률이지 logits 가 아님 ────────────────────
print("\n[1] 이중 softmax 함정")
lg = torch.tensor([[-2.0, 2.0], [1.0, -1.0], [0.2, 0.1]])
out = calibrate.apply(lg, 1.5)
check("apply 는 확률을 돌려줌 (행 합 = 1)",
      np.allclose(out.sum(1), 1.0), str(out.sum(1)))
check("apply 결과를 다시 softmax 하면 뭉개짐 (그래서 하면 안 됨)",
      abs(stages.stage1_scores(out)[0] - 0.5) < abs(stages.stage1_scores(lg / 1.5)[0] - 0.5))
check("올바른 방식은 logits/T",
      abs(stages.stage1_scores(lg / 1.5)[0] - float(torch.softmax(lg[0] / 1.5, 0)[1])) < 1e-6)

# ── 2. 온도는 판정을 바꾸지 않는다 (단조 변환) ────────────────
print("\n[2] 온도는 순위를 안 바꾼다")
rng = np.random.default_rng(0)
n = 4000
y = rng.binomial(1, 0.45, n)
gap = rng.normal(1.6 * (y * 2 - 1), 1.4)
L = torch.tensor(np.stack([-gap / 2, gap / 2], 1) * 2.2, dtype=torch.float32)

s_raw = stages.stage1_scores(L.numpy())
b_raw = evaluate.binary_report(s_raw, y, target_recall=0.95, verbose=False) \
    if "verbose" in inspect.signature(evaluate.binary_report).parameters \
    else evaluate.binary_report(s_raw, y, target_recall=0.95)
T = calibrate.fit_temperature(L, torch.tensor(y), verbose=False) \
    if "verbose" in inspect.signature(calibrate.fit_temperature).parameters \
    else calibrate.fit_temperature(L, torch.tensor(y))
s_cal = stages.stage1_scores(L / T)
b_cal = evaluate.binary_report(s_cal, y, target_recall=0.95, verbose=False) \
    if "verbose" in inspect.signature(evaluate.binary_report).parameters \
    else evaluate.binary_report(s_cal, y, target_recall=0.95)

same = ((s_raw >= b_raw["threshold"]) == (s_cal >= b_cal["threshold"])).mean()
check("보정 전후 판정이 100% 같음", same == 1.0, f"{same:.4%}")
check("recall 도 같음", abs(b_raw["recall_at_target"] - b_cal["recall_at_target"]) < 1e-9)
check("임계값 숫자는 옮겨감", abs(b_raw["threshold"] - b_cal["threshold"]) > 1e-3)

e_b = calibrate.ece(np.stack([1 - s_raw, s_raw], 1), y)
e_a = calibrate.ece(np.stack([1 - s_cal, s_cal], 1), y)
check("ECE 가 좋아짐", e_a < e_b, f"{e_b:.4f} → {e_a:.4f}")
bad = stages.stage1_scores(calibrate.apply(L, T))
check("이중 softmax 는 ECE 를 망침 (반례)",
      calibrate.ece(np.stack([1 - bad, bad], 1), y) > e_b)

# ── 3. export_release 가 temperature.json 을 옮기는가 ────────
print("\n[3] 릴리스가 온도를 싣고 가는가")
src = inspect.getsource(train.export_release)
check("export_release 가 temperature.json 을 복사함", "temperature.json" in src)
check("없으면 경고함", "보정 안 된 확률" in src)
check("best.pt 와 같은 방식으로 복사", src.count("_copy_atomic") >= 3, str(src.count("_copy_atomic")))

# ── 4. 계약이 보정 여부를 진짜로 보고하는가 ──────────────────
print("\n[4] 계약")
check("calibrated 를 하드코딩하지 않음",
      '"calibrated": False' not in inspect.getsource(agent.contract))
check("인자로 받은 값을 씀",
      agent.contract("normal", calibrated=True)["stage1"]["calibrated"] is True
      and agent.contract("normal")["stage1"]["calibrated"] is False)
sc = inspect.getsource(agent.ScreeningAgent.screen)
check("엔진의 실제 T 를 보고 정함", 'getattr(self.s1, "T"' in sc)
check("T 를 meta 에도 실음", "stage1_temperature" in sc)

# ── 5. 노트북 06 이 1단계를 보정하는가 ───────────────────────
print("\n[5] 노트북 06")
nb = json.loads((Path(__file__).resolve().parents[1] /
                 "notebooks" / "06_확정재학습_홀드아웃.ipynb").read_text(encoding="utf-8"))
cells = ["".join(c["source"]) for c in nb["cells"]]
allsrc = "\n".join(cells)
check("1단계 온도를 맞춤", "T1 = calibrate.fit_temperature(lg1_va, y1_va)" in allsrc)
check("보정 눈금으로 임계값을 다시 뽑음", "THR1_RAW, THR1 = THR1," in allsrc)
check("이중 softmax 를 피함",
      "stages.stage1_scores(lg1_va / T1)" in allsrc
      and "stage1_scores(calibrate.apply(lg1_va" not in allsrc)
check("판정 일치율을 확인함", "판정 일치율" in allsrc)
check("체크포인트 옆에 저장", 'ckpt_dir(cfg1.exp_name) / "temperature.json"' in allsrc)
check("파이프라인이 1단계 T 를 하드코딩하지 않음",
      "CLASSES_STAGE1, temperature=1.0)" not in allsrc)
check("파이프라인이 저장된 T 를 읽음", "T1_SERVE" in allsrc)

print("\n" + "=" * 60)
print(f" 통과 {ok} / {ok + fail}")
sys.exit(1 if fail else 0)
