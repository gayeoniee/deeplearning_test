"""★ 최종 출력 문구가 병변 이름을 단정하지 않는가.

2026-08-26 멘토 피드백으로 확정된 규칙입니다. 코드로 못 박아 두지 않으면
누군가 "1등만 보여주면 깔끔한데" 하고 되돌립니다 — 그게 정확히 하면 안 되는
일입니다. holdout 에서 그 1등 이름이 틀린 비율이 **56.6%** 였습니다.

    uv run python tests/test_screening_message.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import infer                                             # noqa: E402
from src.config import CLASS_KO, NORMAL_LABEL, URGENCY_HINT       # noqa: E402

ok = fail = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {name}")
    else:
        fail += 1
        print(f"  FAIL  {name}  {detail}")


LESION = ["A1", "A2", "A3", "A4", "A5", "A6"]
RAW = [("A2", 0.31), ("A3", 0.22), ("A1", 0.18), ("A5", 0.13), ("A4", 0.09), ("A6", 0.07)]


def abnormal_pred(abnormal: float = 0.62) -> infer.Prediction:
    """2단계 파이프라인이 내놓는 모양 그대로 만듭니다."""
    p = infer.Prediction(topk=[(c, v * abnormal) for c, v in RAW],
                         confidence_band=infer.band(RAW[0][1] * abnormal), image="x.jpg")
    p.stage2_probs = list(RAW)
    p.stage1_abnormal = abnormal
    return p


# ── 1. 이름을 단정하지 않는다 ──────────────────────────────────
print("\n[1] 이름 단정 금지")
msg = infer.compose_screening_message(abnormal_pred())

check("병변 이름을 굵게 쓰지 않음",
      not any(f"**{CLASS_KO[c]}**" in msg for c in LESION),
      next((CLASS_KO[c] for c in LESION if f"**{CLASS_KO[c]}**" in msg), ""))
check("'의심됩니다' 로 이름을 지목하지 않음", "형태의 병변이 의심됩니다" not in msg)
hints = [h for h in URGENCY_HINT.values() if h and h != "관찰"]
check("긴급도 문구(URGENCY_HINT) 없음",
      not any(h in msg for h in hints),
      next((h for h in hints if h in msg), ""))

# ── 2. 여섯 개를 전부 보여준다 ────────────────────────────────
print("\n[2] 분포 전체 노출")
for c in LESION:
    check(f"{c}({CLASS_KO[c]}) 가 화면에 있음", CLASS_KO[c] in msg)
check("여섯 줄 모두 백분율이 붙음",
      sum(1 for ln in msg.splitlines() if ln.startswith("    ") and "%" in ln) == 6)

# ── 3. 순서 — "판단할 수 없습니다" 가 숫자보다 위 ──────────────
print("\n[3] 순서")
lines = msg.splitlines()
i_cant = next(i for i, ln in enumerate(lines) if "판단할 수 없습니다" in ln)
i_num = next(i for i, ln in enumerate(lines) if ln.startswith("    ") and "%" in ln)
check("'판단할 수 없습니다' 가 숫자 위", i_cant < i_num, f"{i_cant} vs {i_num}")
check("진료 권유가 숫자 아래", msg.rindex("수의사 진료를 받아보시기를 권합니다") > msg.index("%"))
check("면책 문구 포함", infer.DISCLAIMER in msg)

# ── 4. 깎기 전 원본 분포를 쓴다 ───────────────────────────────
print("\n[4] 원본 분포 (합 = 1)")
check("A2 는 31% 로 뜸 (깎은 19% 아님)", "31%" in msg and "19%" not in msg)
check("1단계 이상 확률을 머리말에 씀", "이상 가능성 62%" in msg)
check("abnormal_p 를 직접 주면 그게 우선",
      "이상 가능성 80%" in infer.compose_screening_message(abnormal_pred(), 0.80))
check("stage2_probs 가 없으면 topk 로 대체",
      "%" in infer.compose_screening_message(
          infer.Prediction(topk=RAW, confidence_band="보통")))

# ── 5. 정상 · 거절은 기존 문구 그대로 ─────────────────────────
print("\n[5] 정상 / 거절")
norm = infer.compose_screening_message(
    infer.Prediction(topk=[(NORMAL_LABEL, 0.91)], confidence_band="높음",
                     stage1_abnormal=0.09))
check("정상은 병변 목록을 안 띄움", not any(CLASS_KO[c] in norm for c in LESION))
check("정상도 단정하지 않음", "병원에 가보시는 것을 권합니다" in norm)
check("정상 확률 표기", "91%" in norm)

ab = infer.compose_screening_message(infer.Prediction(topk=[], abstain=True))
check("거절은 '다시 찍어주세요'", "다시 찍어주세요" in ab)
check("거절에 병변 목록 없음", not any(CLASS_KO[c] in ab for c in LESION))

abstained = abnormal_pred()
abstained.abstain = True
check("이상이어도 abstain 이면 재촬영 안내",
      "판단이 어려운 사진입니다" in infer.compose_screening_message(abstained))

# ── 6. 표 정렬 (한글은 두 칸) ─────────────────────────────────
print("\n[6] 표 정렬")
check("_cells 는 한글을 2 로 셈", infer._cells("비듬") == 4 and infer._cells("ab") == 2)
rows = [ln for ln in lines if ln.startswith("    ") and "%" in ln]
cols = {infer._cells(ln[:ln.index("%") + 1]) for ln in rows}
check("백분율이 같은 칸에서 끝남", len(cols) == 1, str(cols))

# ── 7. 엔진이 이 함수를 쓰는가 ────────────────────────────────
print("\n[7] 배선")
import inspect                                                    # noqa: E402

src_ex = inspect.getsource(infer.TwoStageEngine.explain)
check("TwoStageEngine.explain 이 스크리닝 문구를 씀",
      "compose_screening_message" in src_ex and "compose_message(" not in src_ex)
src_show = inspect.getsource(infer.Engine.show)
check("Engine.show 가 파이프라인 판정이면 스크리닝 문구로 분기",
      "compose_screening_message" in src_show and "stage1_abnormal" in src_show)
check("단일 Engine.explain 은 기존 문구 유지",
      "compose_message(" in inspect.getsource(infer.Engine.explain))

d = abnormal_pred().to_dict()
check("to_dict 에 stage2_probs 원본이 남음",
      d["stage2_probs"][0]["prob"] == 0.31, str(d["stage2_probs"][0]))
check("to_dict 에 stage1_abnormal 이 남음", d["stage1_abnormal"] == 0.62)

print("\n" + "=" * 60)
print(f" 통과 {ok} / {ok + fail}")
sys.exit(1 if fail else 0)
