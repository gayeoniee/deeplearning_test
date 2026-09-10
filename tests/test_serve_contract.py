"""데모 서버를 **실제로 띄워** 앱 계약을 확인합니다 (문서 말고 응답).

    uv run --extra serve python tests/test_serve_contract.py

왜 필요한가 — 계약이 갈라지는 경로가 셋입니다:

1. **mock 과 진짜가 다른 키를 냅니다.** 앱은 둘을 구분 못 합니다. 앱을 mock
   으로 만들었다가 진짜 모델에서 `undefined` 를 만납니다.
   실제로 `meta.stage1_temperature` 가 **진짜에만** 있었습니다.
2. **데모 화면이 없는 필드를 읽습니다.** 화면은 조용히 빈칸이 됩니다.
3. **금지 필드가 슬쩍 들어옵니다.** "1등 병변" 을 주면 앱은 그걸 제일 크게
   띄웁니다 — holdout 에서 46.3% 틀린 이름을요.

`tests/test_agent.py` 는 함수 단위로 봅니다. 여기는 **HTTP 를 통과한 실제
응답**을 봅니다 — `serve.py` 가 계약을 갈아버릴 수도 있으니까요.

⚠️ 가중치가 없으면 mock 만 검사하고 **그 사실을 크게 찍습니다** (조용히
   건너뛰면 "통과" 가 거짓말이 됩니다).
"""

from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ok = fail = 0


def check(name, cond, msg=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {name}")
    else:
        fail += 1
        print(f"  FAIL  {name}" + (f"\n        {msg}" if msg else ""))


try:
    import numpy as np
    from fastapi.testclient import TestClient
    from PIL import Image
except ImportError as exc:
    print(f"[skip] 서빙 의존성이 없습니다 ({exc}) — `uv sync --extra serve`")
    raise SystemExit(0)

import serve  # noqa: E402
from src.agent import GUIDE_ALLOW, MockAgent  # noqa: E402

#: 계약에 **절대** 들어오면 안 되는 이름 (앱이 제일 크게 띄웁니다)
BANNED = {"top1", "predicted", "diagnosis", "disease", "label_top", "best", "answer"}


def photo(seed: int, w: int = 1920, h: int = 1080) -> bytes:
    b = io.BytesIO()
    Image.fromarray(np.random.default_rng(seed).integers(
        0, 255, (h, w, 3), dtype=np.uint8)).save(b, format="JPEG")
    return b.getvalue()


def call(app, seed: int, box=None):
    c = TestClient(app)
    data = {"box": json.dumps(box)} if box is not None else {}
    r = c.post("/v1/screen",
               files={"photo": (f"{seed}.jpg", photo(seed), "image/jpeg")}, data=data)
    return r.status_code, r.json()


def shape(o, path: str = "") -> set[str]:
    """응답의 **키 구조**만 뽑습니다 (값은 안 봅니다)."""
    if isinstance(o, dict):
        out: set[str] = set()
        for k, v in o.items():
            out |= {f"{path}.{k}"} | shape(v, f"{path}.{k}")
        return out
    if isinstance(o, list) and o:
        return shape(o[0], path + "[]")
    return set()


#: 밴드 **안쪽** 네모 (권장 구간, 중앙)
GOOD = [0.32, 0.32, 0.36, 0.36]


def survey(app, label: str) -> tuple[set[str], set[str]]:
    """여러 사진으로 normal·abnormal 을 **둘 다** 만들어 키 구조를 모읍니다.

    ⚠️ 한 장만 보면 안 됩니다 — `verdict` 에 따라 `stage2.distribution` 이
       비어 있어서, 빈 배열끼리 비교하면 차이가 안 보입니다. 처음에 이걸로
       "mock 에만 있는 키" 를 잘못 봤습니다.
    """
    keys, verdicts = set(), set()
    for seed in range(12):
        code, j = call(app, seed, GOOD)
        if code != 200:
            continue
        keys |= shape(j)
        verdicts.add(j.get("verdict"))
    return keys, verdicts


print("[1] mock 서버 — 계약의 기본형")
mock_app = serve.build_app(MockAgent(), mock=True)
mk, mv = survey(mock_app, "mock")
check("HTTP 200 으로 응답한다", bool(mk))
check("normal 과 abnormal 을 둘 다 만든다 (빈 분포로 비교하면 안 됩니다)",
      {"normal", "abnormal"} <= mv, f"나온 verdict: {sorted(mv)}")
for k in (".contract_version", ".verdict", ".headline", ".body", ".action",
          ".disclaimer", ".stage1.abnormal_prob", ".stage1.threshold",
          ".stage1.calibrated", ".stage2.shown", ".stage2.distribution",
          ".meta.mock", ".meta.box_source", ".meta.crop_note",
          ".meta.stage1_temperature"):
    check(f"계약에 {k} 가 있다", k in mk)

print("\n[2] ★ 금지 필드 — 주는 순간 앱이 제일 크게 띄웁니다")
hit = sorted(k for k in mk if k.rsplit(".", 1)[-1].rstrip("[]") in BANNED)
check("'1등 병변' 계열 필드가 없다", not hit, str(hit))

print("\n[3] ★ 가이드 밴드 밖이면 **모델 돌리기 전에** 돌려보낸다")
lo, hi = GUIDE_ALLOW
for why, box in (("허용보다 큼", [0.05, 0.05, hi + 0.1, hi + 0.1]),
                 ("허용보다 작음", [0.48, 0.48, lo - 0.05, lo - 0.05])):
    _, j = call(mock_app, 0, box)
    check(f"{why} → retake",
          j["verdict"] == "retake" and bool(j["meta"].get("retake_reason")),
          f"{j['verdict']} / {j['meta'].get('retake_reason')}")
_, j = call(mock_app, 0, GOOD)
check("밴드 안쪽은 통과한다", j["verdict"] != "retake", j["verdict"])
_, j = call(mock_app, 0, [0.0, 0.0, 0.3, 0.3])
check("화면 가장자리의 병변도 크기가 맞으면 통과한다", j["verdict"] != "retake", j["verdict"])

print("\n[4] box 를 안 주면 중앙으로 물러서고 **그걸 밝힌다**")
_, j = call(mock_app, 0, None)
check("box_source 가 center", j["meta"].get("box_source") == "center")
check("crop_note 로 한계를 실어 보낸다", bool(j["meta"].get("crop_note")))

print("\n[5] ★ demo/index.html 이 읽는 필드가 응답에 실제로 있는가")
html = (ROOT / "demo" / "index.html").read_text(encoding="utf-8")
reads = sorted(set(re.findall(r"\bj\.([a-zA-Z_]+(?:\.[a-zA-Z_]+)*)", html)))
check("데모가 응답을 읽고 있다 (형식이 안 바뀌었다)", len(reads) >= 5, str(reads))
#: retake 때만 오는 것 — 데모가 `j.meta && j.meta.retake_reason` 으로 지킵니다
OPTIONAL = {"meta.retake_reason", "detail"}
for r in reads:
    if r in OPTIONAL:
        continue
    check(f"j.{r} 가 응답에 있다",
          any(k.lstrip(".").startswith(r) for k in mk))
_, jr = call(mock_app, 0, [0.05, 0.05, 0.9, 0.9])
check("j.meta.retake_reason 은 retake 일 때 온다",
      "retake_reason" in jr["meta"])

print("\n[6] ★ mock 과 진짜가 **같은 키**를 내는가")
CK = ROOT / "data" / "work" / "checkpoints"
c1 = CK / "stage1_effnetv2_s_f320_384_n233k_moderate_photometric" / "best.pt"
c2 = CK / "stage2_convnextv2_base_m2.5_384_n121k_moderate" / "best.pt"
thr_f = ROOT / "data" / "work" / "stage1_threshold.json"
if not (c1.exists() and c2.exists() and thr_f.exists()):
    print("  ⚠️ 가중치가 없어 **이 검사를 못 했습니다** (mock 만 봤습니다).")
    print("     릴리스가 있는 환경에서 한 번은 돌려주세요 — 여기서 실제로")
    print("     `meta.stage1_temperature` 가 진짜에만 있는 걸 잡았습니다.")
else:
    from src.agent import ScreeningAgent

    real = ScreeningAgent.load(
        c1, c2, threshold=json.loads(thr_f.read_text(encoding="utf-8"))["threshold"],
        device="cpu")
    rk, rv = survey(serve.build_app(real, mock=False), "real")
    check("진짜도 normal·abnormal 을 둘 다 만든다", {"normal", "abnormal"} <= rv,
          f"나온 verdict: {sorted(rv)} — 표본이 한쪽으로만 나오면 비교가 무의미합니다")
    only_m, only_r = sorted(mk - rk), sorted(rk - mk)
    check("mock 에만 있는 키가 없다", not only_m, str(only_m))
    check("진짜에만 있는 키가 없다", not only_r, str(only_r))
    hit = sorted(k for k in rk if k.rsplit(".", 1)[-1].rstrip("[]") in BANNED)
    check("진짜 응답에도 금지 필드가 없다", not hit, str(hit))

print("\n[7] ★ 계열(`stage2.group`) — 기본은 null, 켜도 이름이 안 샌다")
import src.message as _M                                        # noqa: E402
from src.agent import lesion_group                              # noqa: E402
from src.config import CLASS_KO, CLASSES, MORPH_GROUP_KEEP_A6   # noqa: E402

_PEAK = [("A1", .45), ("A4", .30), ("A2", .10), ("A3", .06), ("A5", .05), ("A6", .04)]
_FLAT = [("A1", .2), ("A2", .2), ("A3", .2), ("A4", .2), ("A5", .1), ("A6", .1)]

_r = [j for _, j in [call(mock_app, s, GOOD) for s in range(6)]]
check("계약에 stage2.group 이 있다 (꺼져 있어도 키는 있어야 함)",
      all("group" in x["stage2"] for x in _r))
check("기본값에서는 group 이 null",
      all(x["stage2"]["group"] is None for x in _r) if not _M.SHOW_GROUP else True)

_was = _M.SHOW_GROUP
_M.SHOW_GROUP = True
try:
    g = lesion_group(_PEAK, 0.90)
    check("켜고 확신이 높으면 group 이 온다", g is not None)
    check("확신이 낮으면 **null** — 말하지 않는다", lesion_group(_FLAT, 0.60) is None)
    check("1단계가 애매하면 **null**", lesion_group(_PEAK, 0.55) is None)
    if g:
        # ★ 2026-09-10 — `labels` 가 생겼습니다. 거기엔 6종 이름이 **일부러**
        #   들어갑니다 (`(구진·플라크·농포·여드름)`) — `솟아오른 변화` 만 들고
        #   병원에 가면 수의사가 못 알아듣기 때문입니다.
        #   ⚠️ 금지된 건 여전히 *하나를 골라 단정하는 것*이라, `labels` 를 뺀
        #      나머지에서 이름이 새는지 봅니다.
        blob = json.dumps({k: v for k, v in g.items() if k != "labels"},
                          ensure_ascii=False)
        leaked = [CLASS_KO[c] for c in CLASSES if CLASS_KO[c] in blob]
        check("★ 6종 이름이 group 에 (labels 밖으로) 안 들어간다", not leaked, str(leaked))
        # ★★ `labels` 는 **코드순 고정**이어야 합니다 — 확률순이면 첫 이름이
        #    "1등" 으로 읽혀서 그때는 진짜 top1 부활입니다.
        from src.config import GROUP_LABELS                        # noqa: E402
        check("★★ labels 가 코드순 고정본과 같다",
              g.get("labels") == GROUP_LABELS.get(g["name"]),
              f"{g.get('labels')!r} vs {GROUP_LABELS.get(g['name'])!r}")
        check("계열 이름만 쓴다",
              g["name"] in set(MORPH_GROUP_KEEP_A6.values()))
        check("진단이 아니라고 같이 보낸다", "진단이 아닙니다" in g["caveat"])
        check("문장을 서버가 준다 (앱·콘솔이 지어 쓰지 않게)", bool(g["text"]))
        check("★ group 에도 금지 필드가 없다", not any(k in g for k in BANNED))
finally:
    _M.SHOW_GROUP = _was

print("\n[7] ★ /healthz 가 **몇 팔인지** 말하는가 (2026-09-07 배포 사고)")
# 배포에서 앙상블이 조용히 1팔로 줄었는데 응답은 200 이었습니다. 알아챌 단서가
# 로그에 성공 줄이 **안 찍힌 것**뿐이라 아무도 못 봤습니다. 실제로 확인하는 데
# 찌르기 6번 + 합성 병변 사진이 들었습니다 — 사진 없이 curl 한 번이어야 합니다.
import importlib  # noqa: E402
import inspect  # noqa: E402

import src.agent as _A  # noqa: E402

_hz = TestClient(serve.build_app(MockAgent(0.1823), mock=True)).get("/healthz").json()
for _k in ("stage2_arms", "stage2_crops", "stage2_experiments",
           "release_dir", "stage1_crop", "stage2_crop"):
    check(f"/healthz 에 {_k} 가 있다", _k in _hz, str(sorted(_hz)))
check("stage2_arms 가 정수다", isinstance(_hz.get("stage2_arms"), int))
check("stage2_crops 길이 == stage2_arms",
      len(_hz.get("stage2_crops", [])) == _hz.get("stage2_arms"),
      f"{_hz.get('stage2_crops')} vs {_hz.get('stage2_arms')}")
check("헬스체크가 모델을 안 올린다 (describe 가 예외를 안 냄)",
      "describe_error" not in _hz, _hz.get("describe_error", ""))

# ⚠️ mock 과 real 이 **같은 키**를 내야 합니다 (여섯 번 어긋났습니다).
_src = inspect.getsource(_A.ScreeningAgent.describe)
for _k in ("stage2_arms", "stage2_crops", "stage2_experiments", "release_dir"):
    check(f"ScreeningAgent.describe 도 {_k} 를 낸다", f'"{_k}"' in _src)

print("\n[8] export_release 가 앙상블 팔을 빠뜨리지 않는가")
# from_release() 는 릴리스의 stage2_* 를 **전부** 훑습니다. export 가 덜 담으면
# 릴리스를 다시 만드는 순간 팔이 조용히 사라집니다 (커버리지 67.9% → 58.4%).
_T = importlib.import_module("src.train")
check("stage2_arms_on_disk 가 있다", hasattr(_T, "stage2_arms_on_disk"))
_es = inspect.getsource(_T.export_release)
check("export_release 가 그것을 부른다", "stage2_arms_on_disk" in _es)
check("끄는 스위치가 있다 (include_stage2_arms)",
      "include_stage2_arms" in inspect.signature(_T.export_release).parameters)

print("\n" + "=" * 60)
print(f" 통과 {ok} / {ok + fail}")
raise SystemExit(1 if fail else 0)
