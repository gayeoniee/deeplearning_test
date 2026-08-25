"""src/agent.py — 앱이 받는 계약이 병변 이름을 흘리지 않는가.

이 파일의 절반은 **금지 목록**입니다. 계약에 "1등 병변" 필드가 생기는 순간
안드로이드 쪽은 그걸 화면 제일 크게 띄웁니다 — holdout 에서 56.6% 틀린 이름을요.
그래서 사람 리뷰가 아니라 테스트로 막습니다.

    uv run python tests/test_agent.py           (torch 불필요)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import agent                                            # noqa: E402
from src.config import CLASS_KO, CLASSES                         # noqa: E402

ok = fail = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {name}")
    else:
        fail += 1
        print(f"  FAIL  {name}  {detail}")


# ── 1. 계약에 "1등" 이 없다 ───────────────────────────────────
print("\n[1] 금지 필드")

# 앱이 "이거 하나만 보여주면 되겠네" 하고 집을 수 있는 이름들
BANNED = ["top1", "top_1", "topk", "top_k", "predicted", "prediction", "predicted_class",
          "diagnosis", "diagnosed", "best", "best_class", "winner", "label", "answer",
          "most_likely", "primary", "urgency", "hint"]

full = agent.contract("abnormal", abnormal_p=0.62, threshold=0.1823,
                      stage2=[(c, 1 / 6) for c in CLASSES], text="…")
flat = json.dumps(full, ensure_ascii=False)


def keys_of(o, acc=None):
    acc = acc if acc is not None else set()
    if isinstance(o, dict):
        for k, v in o.items():
            acc.add(k)
            keys_of(v, acc)
    elif isinstance(o, list):
        for v in o:
            keys_of(v, acc)
    return acc


ks = keys_of(full)
hit = sorted(k for k in ks if k.lower() in BANNED)
check("계약에 1등류 키가 없음", not hit, str(hit))
check("JSON 직렬화 됨", isinstance(flat, str) and len(flat) > 100)
check("헤드라인이 병변 이름을 안 씀",
      not any(CLASS_KO[c] in full["headline"] for c in CLASSES))
check("action 이 병변 이름을 안 씀",
      not any(CLASS_KO[c] in full["action"] for c in CLASSES))

# ── 2. 분포는 여섯 개 전부 ────────────────────────────────────
print("\n[2] 분포")
d = full["stage2"]["distribution"]
check("여섯 개 전부", len(d) == 6, str(len(d)))
check("여섯 클래스가 다 있음", {x["code"] for x in d} == set(CLASSES))
check("합이 1", abs(sum(x["prob"] for x in d) - 1) < 0.01, str(sum(x["prob"] for x in d)))
check("한국어·영어 이름이 붙어 있음", all(x["name_ko"] and x["name_en"] for x in d))
check("percent 도 같이 줌", all("percent" in x for x in d))
check("확률 내림차순", [x["prob"] for x in d] == sorted((x["prob"] for x in d), reverse=True))

# ── 3. verdict 3종 ────────────────────────────────────────────
print("\n[3] verdict")
for v in ("normal", "abnormal", "retake"):
    c = agent.contract(v, abnormal_p=0.5)
    check(f"{v} 이 만들어짐", c["verdict"] == v and c["headline"])
try:
    agent.contract("suspicious")
    check("모르는 verdict 는 에러", False, "안 막힘")
except ValueError:
    check("모르는 verdict 는 에러", True)
check("normal 은 분포를 안 실음", not agent.contract("normal")["stage2"]["shown"])
check("retake 는 분포를 안 실음", not agent.contract("retake")["stage2"]["shown"])
check("면책 문구가 항상 있음",
      all(agent.contract(v)["disclaimer"] for v in ("normal", "abnormal", "retake")))
check("1단계 미보정을 계약이 실어 나름",
      agent.contract("normal")["stage1"]["calibrated"] is False)
check("서빙 크롭이 미검증임을 계약이 실어 나름",
      "실측하지 않았습니다" in agent.contract("normal")["meta"]["crop_untested"])

# ── 4. 서빙 크롭 ──────────────────────────────────────────────
print("\n[4] 서빙 크롭")
from PIL import Image                                            # noqa: E402

check("f320 비율은 320/1080", abs(agent.STAGE1_FRAC - 320 / 1080) < 1e-9)
im = Image.new("RGB", (1920, 1080))
c1 = agent.center_square(im, agent.STAGE1_FRAC)
check("1920×1080 에서 f320 은 320px 정사각", c1.size == (320, 320), str(c1.size))
c2 = agent.center_square(im, 1.0)
check("frac=1 은 짧은 변 정사각", c2.size == (1080, 1080), str(c2.size))
# 휴대폰 사진(4032×3024)에서도 **같은 화각**이 나와야 합니다 — 픽셀이 아니라 비율
phone = agent.center_square(Image.new("RGB", (4032, 3024)), agent.STAGE1_FRAC)
check("휴대폰 사진은 비율로 잘림 (픽셀 320 아님)",
      phone.size == (896, 896), str(phone.size))
check("세로 사진도 짧은 변 기준",
      agent.center_square(Image.new("RGB", (1080, 1920)), 1.0).size == (1080, 1080))
check("중앙에서 잘림",
      agent.center_square(Image.new("RGB", (100, 100)), 0.5).size == (50, 50))

# ── 5. MockAgent — 진짜와 같은 모양 ───────────────────────────
print("\n[5] MockAgent")
tmp = Path(__file__).resolve().parents[1] / "README.md"
m = agent.MockAgent()
r1, r2 = m.screen(tmp), m.screen(tmp)


def decision(r: dict) -> dict:
    """판정만 남깁니다 — elapsed_ms 는 매번 달라도 되는 값입니다."""
    return {**r, "meta": {k: v for k, v in r["meta"].items() if k != "elapsed_ms"}}


check("같은 입력 → 같은 판정 (시연 중 안 흔들림)", decision(r1) == decision(r2))
check("흔들려도 되는 건 소요시간뿐",
      {k for k in r1["meta"] if r1["meta"][k] != r2["meta"][k]} <= {"elapsed_ms"})
check("mock 임을 밝힘", r1["meta"]["mock"] is True)
check("진짜와 같은 키", set(r1) == set(full))
check("mock 도 금지 키 없음", not [k for k in keys_of(r1) if k.lower() in BANNED])
check("text 가 여섯 줄 분포를 담음",
      r1["verdict"] != "abnormal" or
      sum(1 for ln in r1["text"].splitlines() if ln.startswith("    ") and "%" in ln) == 6)
check("못 읽는 파일은 retake",
      m.screen("존재하지-않는-파일.jpg")["verdict"] == "retake")

# normal 로 떨어지는 입력도 하나 찾아 확인합니다
found = None
for i in range(300):
    p = Path(__file__).resolve().parents[1] / "src" / "agent.py"
    r = agent.MockAgent(threshold=0.99).screen(p)
    found = r
    break
check("임계값을 올리면 normal 로", found["verdict"] == "normal", found["verdict"])
check("normal 이면 분포가 비어 있음", not found["stage2"]["distribution"])

# ── 6. torch 없이 import 되는가 ───────────────────────────────
print("\n[6] torch 의존")
check("agent 가 torch 를 import 하지 않음", "torch" not in sys.modules)
import src.message                                               # noqa: E402

check("message 도 torch 없이 됨", "torch" not in sys.modules)
check("infer 가 message 를 재수출함",
      "from src.message import" in (Path(__file__).resolve().parents[1]
                                    / "src" / "infer.py").read_text(encoding="utf-8"))

# ── 7. ScreeningAgent 안전장치 ────────────────────────────────
print("\n[7] 임계값 안전장치")
import inspect                                                   # noqa: E402

src_load = inspect.getsource(agent.ScreeningAgent.load)
check("임계값을 못 찾으면 에러 (기본값으로 조용히 안 감)",
      "FileNotFoundError" in src_load)
sig = inspect.signature(agent.ScreeningAgent.load)
check("load 의 threshold 기본값이 None (숫자가 아님)",
      sig.parameters["threshold"].default is None,
      str(sig.parameters["threshold"].default))
sig2 = inspect.signature(agent.ScreeningAgent.__init__)
check("__init__ 의 threshold 는 필수 인자",
      sig2.parameters["threshold"].default is inspect.Parameter.empty)
check("stage1_threshold.json 을 찾아봄", "stage1_threshold.json" in src_load)
src_screen = inspect.getsource(agent.ScreeningAgent.screen)
check("깎기 전 원본을 stage2 로 넘김",
      "stage2=raw" in src_screen and "stage2_probs = raw" in src_screen)
check("두 단계가 다른 크롭을 씀",
      "self.f1" in src_screen and "self.f2" in src_screen)

print("\n" + "=" * 60)
print(f" 통과 {ok} / {ok + fail}")
sys.exit(1 if fail else 0)
