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
import inspect                                                   # noqa: E402

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
check("crop_note 두 갈래가 다 있음", set(agent.CROP_NOTE) == {"user_box", "center"})

# ── 4. 서빙 크롭 = 학습 크롭 ──────────────────────────────────
print("\n[4] 서빙 크롭이 학습 크롭과 같은가")
from PIL import Image                                            # noqa: E402

from src import crop                                             # noqa: E402

W, H = 1920, 1080                       # AI Hub 원본
BBOX = [900.0, 470.0, 1020.0, 610.0]    # 라벨 bbox (120×140)
train1 = crop.crop_window({"bbox": BBOX, "img_w": W, "img_h": H}, tag="f320")
train2 = crop.crop_window({"bbox": BBOX, "img_w": W, "img_h": H}, tag="m2.5")

# 같은 구도를 휴대폰이 4032×2268 로 찍고, 가이드 프레임을 정규화로 넘겼다고 가정
box = [BBOX[0] / W, BBOX[1] / H, (BBOX[2] - BBOX[0]) / W, (BBOX[3] - BBOX[1]) / H]
sp = agent.to_train_space(Image.new("RGB", (4032, 2268)))
check("짧은 변을 1080 으로 맞춤", sp.size == (1920, 1080), str(sp.size))
bb = agent.box_to_px(box, *sp.size)
serve1 = crop.crop_window({"bbox": bb, "img_w": sp.size[0], "img_h": sp.size[1]}, tag="f320")
serve2 = crop.crop_window({"bbox": bb, "img_w": sp.size[0], "img_h": sp.size[1]}, tag="m2.5")

check("★ 1단계 창이 학습과 동일", max(abs(a - b) for a, b in zip(train1, serve1)) <= 2,
      f"{train1} vs {serve1}")
check("★ 2단계 창이 학습과 동일", max(abs(a - b) for a, b in zip(train2, serve2)) <= 2,
      f"{train2} vs {serve2}")
check("학습이 쓰는 crop.crop_window 를 그대로 부름",
      "crop_window" in inspect.getsource(agent.crop_for))

# 1단계는 네모 **크기**에 둔감해야 합니다 (f320 은 중심만 씀)
loose = [box[0] - .06, box[1] - .10, box[2] + .12, box[3] + .20]
lb = agent.box_to_px(loose, *sp.size)
l1 = crop.crop_window({"bbox": lb, "img_w": sp.size[0], "img_h": sp.size[1]}, tag="f320")
l2 = crop.crop_window({"bbox": lb, "img_w": sp.size[0], "img_h": sp.size[1]}, tag="m2.5")
# ⚠️ 320 과 정확히 같지는 않습니다 — `fixed_box` 가 좌변은 int(), 우변은
#    int(round()) 을 써서 중심이 반 픽셀에 걸리면 321 이 나옵니다. 학습도 같은
#    함수를 쓰므로 학습 크롭에도 똑같이 있는 오차입니다 (320px 에서 0.3%).
#    중요한 건 **네모 크기에 안 흔들린다**는 것이라, 그걸 봅니다.
check("네모가 헐렁해도 1단계 창 크기는 그대로 (±1px)",
      abs((l1[2] - l1[0]) - (train1[2] - train1[0])) <= 1
      and abs((l1[2] - l1[0]) - 320) <= 1,
      f"{train1[2]-train1[0]} → {l1[2]-l1[0]}")
check("네모가 헐렁하면 2단계 창은 커짐 (크기를 쓰므로)",
      (l2[2] - l2[0]) > (train2[2] - train2[0]) * 1.3,
      f"{train2[2]-train2[0]} → {l2[2]-l2[0]}")

# 네모가 없으면 물러섭니다
c1 = agent.crop_for(Image.new("RGB", (W, H)), None, "f320")
check("네모 없으면 f320 은 중앙 320px", c1.size == (320, 320), str(c1.size))
c2 = agent.crop_for(Image.new("RGB", (W, H)), None, "m2.5")
check("네모 없으면 m2.5 는 중앙 정사각", c2.size == (1080, 1080), str(c2.size))

check("잘못된 box 는 None", agent.box_to_px([0, 0, 0, 0], W, H) is None)
check("길이가 안 맞는 box 는 None", agent.box_to_px([0.1, 0.2], W, H) is None)
check("체크포인트 이름에서 크롭 태그를 읽음",
      agent.crop_tag_from_exp("stage1_effnetv2_s_f320_384_moderate_photometric") == "f320"
      and agent.crop_tag_from_exp("stage2_convnextv2_base_m2.5_384_moderate") == "m2.5")

# ── 4-b. 촬영 가이드 밴드 ─────────────────────────────────────
print("\n[4-b] 촬영 가이드 밴드")
check("밴드 값이 STATUS 실측과 같음",
      agent.GUIDE_RECOMMEND == (0.28, 0.48) and agent.GUIDE_ALLOW == (0.24, 0.56))
mid = [0.5 - 0.44 / 2, 0.5 - 0.44 / 2, 0.44, 0.44]
check("권장 안이면 통과", agent.check_guide(mid)["ok"])
small = [0.5 - .10 / 2, 0.5 - .10 / 2, .10, .10]
check("너무 작으면 막고 이유를 말함",
      not agent.check_guide(small)["ok"] and "가까이" in agent.check_guide(small)["reason"])
big = [0.5 - .80 / 2, 0.5 - .80 / 2, .80, .80]
check("너무 크면 막고 이유를 말함",
      not agent.check_guide(big)["ok"] and "멀리" in agent.check_guide(big)["reason"])
off = [0.0, 0.0, 0.44, 0.44]
check("가운데서 벗어나면 막음",
      not agent.check_guide(off)["ok"] and "가운데" in agent.check_guide(off)["reason"])
check("허용 경계 안쪽(28%)은 통과",
      agent.check_guide([0.5 - .29 / 2, 0.5 - .29 / 2, .29, .29])["ok"])

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
check("mock 도 가이드 검사를 진짜로 함",
      m.screen(tmp, box=small)["verdict"] == "retake"
      and "가까이" in m.screen(tmp, box=small)["meta"]["retake_reason"])
check("밴드 안이면 mock 도 통과", m.screen(tmp, box=mid)["verdict"] != "retake")
check("box 를 주면 box_source=user", m.screen(tmp, box=mid)["meta"]["box_source"] == "user")
check("box 가 없으면 box_source=center", m.screen(tmp)["meta"]["box_source"] == "center")
check("한계를 계약에 실어 보냄",
      "분포가 다릅니다" in m.screen(tmp, box=mid)["meta"]["crop_note"]
      and "어긋납니다" in m.screen(tmp)["meta"]["crop_note"])

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
check("두 단계가 다른 크롭 태그를 씀",
      "self.tag1" in src_screen and "self.tag2" in src_screen)
check("밴드 밖이면 모델을 돌리기 전에 돌려보냄",
      src_screen.index("check_guide") < src_screen.index("self.s1.predict"))
check("학습 픽셀 공간으로 먼저 맞춤", "to_train_space" in src_screen)

# ── 8. 네모 오차 → 이미 잰 교란으로 환산 (tools/box_error.py) ──
print("\n[8] 네모 오차 환산")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import box_error as be                                          # noqa: E402

T = [0.45, 0.45, 0.55, 0.55]          # 정답: 가운데 0.10 크기
check("딱 맞으면 줌 1.0", abs(be.to_perturbation([.45, .45, .10, .10], T)["zoom"] - 1) < 1e-9)
check("네모를 2배로 그리면 줌 0.5 (크롭이 넓어짐)",
      abs(be.to_perturbation([.40, .40, .20, .20], T)["zoom"] - 0.5) < 1e-9)
check("네모를 절반으로 그리면 줌 2.0",
      abs(be.to_perturbation([.475, .475, .05, .05], T)["zoom"] - 2.0) < 1e-9)
check("중심만 어긋나면 크기비는 1",
      abs(be.to_perturbation([.50, .50, .10, .10], T)["size_ratio"] - 1) < 1e-9)
check("중심 어긋남은 2단계 크롭 폭 기준",
      abs(be.to_perturbation([.50, .50, .10, .10], T)["shift_frac"]
          - (0.05 * 2 ** .5) / 0.25) < 1e-6)
check("빈 네모는 빈 dict", be.to_perturbation([.5, .5, 0, 0], T) == {})

check("네모 허용 밴드는 줌 밴드의 역수",
      abs(be.BOX_ALLOW[0] - 1 / be.ZOOM_ALLOW[1]) < 1e-9
      and abs(be.BOX_ALLOW[1] - 1 / be.ZOOM_ALLOW[0]) < 1e-9)
check("밴드 값이 agent 와 같은 실측에서 옴",
      be.SHIFT_MAX == agent.GUIDE_CENTER_MAX)
check("1.5배로 그리면 허용 밖 (0.59~1.43배)",
      not (be.BOX_ALLOW[0] <= 1.5 <= be.BOX_ALLOW[1]))
check("1.3배는 허용 안", be.BOX_ALLOW[0] <= 1.3 <= be.BOX_ALLOW[1])

sm = be.summarize([be.to_perturbation([.45, .45, .10, .10], T), {"user": None}])
check("요약이 건너뛴 장수를 셈", "건너뜀 1" in sm, sm.split("\n")[0])
check("요약이 병명 정확도가 아님을 밝힘", "병명 정확도가 아닙니다" in sm)
check("표본이 없으면 그렇게 말함", "쓸 수 있는 표본이 없습니다" in be.summarize([]))

# ── 9. release 폴더 자동 탐색 + 1단계만 구성 ────────────────
print("\n[9] 가중치 붙이기")
import json as _json                                             # noqa: E402
import tempfile                                                  # noqa: E402

with tempfile.TemporaryDirectory() as td:
    rel = Path(td) / "release"
    s1 = rel / "checkpoints" / "stage1_effnetv2_s_f320_384_moderate_photometric"
    s2 = rel / "checkpoints" / "stage2_convnextv2_base_m2.5_384_moderate"
    for d in (s1, s2):
        d.mkdir(parents=True); (d / "best.pt").touch()
    (rel / "stage1_threshold.json").write_text(_json.dumps({"threshold": 0.1823}))

    seen: dict = {}
    real = agent.ScreeningAgent.load
    agent.ScreeningAgent.load = classmethod(
        lambda cls, c1, c2=None, thr=None, dev=None, stage1_only=False:
        seen.update(c1=Path(c1).parent.name, c2=(Path(c2).parent.name if c2 else None),
                    thr=thr, only=stage1_only))
    try:
        agent.ScreeningAgent.from_release(rel)
        check("release 에서 1단계를 이름으로 찾음", seen["c1"].startswith("stage1_"), str(seen))
        check("release 에서 2단계를 이름으로 찾음", seen["c2"].startswith("stage2_"), str(seen))
        check("stage1_threshold.json 을 같이 읽음", seen["thr"] == 0.1823, str(seen["thr"]))

        seen.clear()
        agent.ScreeningAgent.from_release(rel, stage1_only=True)
        check("--stage1-only 를 전달함", seen["only"] is True)

        import shutil                                            # noqa: E402
        shutil.rmtree(s2)
        try:
            agent.ScreeningAgent.from_release(rel)
            check("2단계가 없으면 에러", False, "안 막힘")
        except FileNotFoundError as e:
            check("2단계가 없으면 에러 + --stage1-only 를 안내", "--stage1-only" in str(e))
        seen.clear()
        agent.ScreeningAgent.from_release(rel, stage1_only=True)
        check("2단계가 없어도 1단계만은 뜸", seen["c1"].startswith("stage1_"))

        shutil.rmtree(s1)
        try:
            agent.ScreeningAgent.from_release(rel, stage1_only=True)
            check("1단계도 없으면 에러", False, "안 막힘")
        except FileNotFoundError:
            check("1단계도 없으면 에러", True)
    finally:
        agent.ScreeningAgent.load = real

# 1단계만일 때의 문구
from src import message as _msg                                  # noqa: E402

only_txt = _msg.compose_screening_message(
    _msg.Prediction(topk=[], stage1_abnormal=0.62, confidence_band="보통"))
check("1단계만이어도 재촬영으로 안 보냄", "판단이 어려운 사진" not in only_txt)
check("1단계만도 이상을 말함", "이상 소견이 보입니다" in only_txt)
check("1단계만도 확률을 보여줌", "62%" in only_txt)
check("1단계만도 진료를 권함", "수의사 진료를 받아보시기를 권합니다" in only_txt)
check("1단계만이면 병변 이름이 하나도 안 나옴",
      not any(CLASS_KO[c] in only_txt for c in CLASSES))
check("진짜 거절은 그대로 재촬영",
      "판단이 어려운 사진" in _msg.compose_screening_message(
          _msg.Prediction(topk=[], abstain=True)))

# ── 10. 기권이 판정을 뒤집지 않는가 ────────────────────────────
#
# 2026-08-31 배포에서 터진 것: 1단계가 95.6% 로 이상이라고 본 사진이
# 2단계 기권 때문에 통째로 "다시 찍어주세요" 로 나갔습니다. 분포까지
# 같이 사라졌고요. 기권은 병변 **종류**를 말할지의 판단이지 1단계 판정을
# 뒤집는 장치가 아닙니다 (노트북 06 §5 에 그렇게 적혀 있습니다).
print("\n[10] 기권이 1단계 판정을 뒤집지 않는가")

_after = src_screen[src_screen.index("pred.abstain ="):]
check("기권해도 retake 로 안 보냄", 'contract("retake"' not in _after)
check("1단계가 이상이면 결론은 abnormal 하나뿐",
      _after.count('contract("abnormal"') == 1)
check("기권이어도 분포를 실어 보냄", "stage2=raw" in _after)
check("기권 사실을 meta 로 알림", "stage2_low_confidence" in src_screen)

# 기권 판정은 **깎기 전** 확률로 — 1단계 확률을 곱한 값과 비교하면 이중 감점입니다
check("기권을 깎기 전 확률로 판정", "raw[0][1] < self.s2.cfg.abstain_threshold" in src_screen)
check("깎은 값(topk)으로 기권 판정하지 않음",
      "pred.topk[0][1] < self.s2.cfg.abstain_threshold" not in src_screen)

# 문구 쪽에도 같은 구멍이 있었습니다 — 한쪽만 고치면 verdict 와 text 가 어긋납니다
_src_msg = inspect.getsource(_msg.compose_screening_message)
check("문구도 기권을 재촬영으로 안 바꿈", "if pred.abstain or not pred.topk:" not in _src_msg)

_ab = _msg.Prediction(topk=[("A2", 0.31)], abstain=True, stage1_abnormal=0.956,
                      confidence_band="낮음")
_ab.stage2_probs = [("A2", 0.31), ("A3", 0.22)]
_txt = _msg.compose_screening_message(_ab)
check("기권이어도 이상 소견을 말함", "이상 소견이 보입니다" in _txt)
check("기권이어도 재촬영 문구가 안 나옴", "판단이 어려운 사진" not in _txt)
check("기권이어도 분포가 보임", CLASS_KO["A2"] in _txt)
check("기권이어도 진료를 권함", "수의사 진료를 받아보시기를 권합니다" in _txt)
check("기권일 때 다시 찍으라고 하지 않음",
      "더 선명하게 다시 찍으면" not in _txt)

print("\n" + "=" * 60)
print(f" 통과 {ok} / {ok + fail}")
sys.exit(1 if fail else 0)
