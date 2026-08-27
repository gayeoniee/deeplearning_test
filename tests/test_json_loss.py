"""JSON 을 못 읽었을 때 **조용히 넘어가지 않는가**.

    uv run python tests/test_json_loss.py

실제로 당한 것 (2026-08-27): TL01.zip 90GB 를 받아 처리했더니

    [labels] TL01.zip: JSON 263,340개 / 이미지 263,341장
    [labels] 원시 행 6,280개 (이미지 매칭 실패 0건)
    ✅ 청크 TL01 완료 — 6,278행

**263,340개 중 6,280개만 남았는데 아무 말도 없었습니다.** 그 사이에 있던 건
카운터도 메시지도 없는 `except Exception: continue` 한 줄이었습니다.
같은 코드로 TL02 는 177,605개 중 1개만 놓쳤으니, 코드가 늘 틀린 게 아니라
**청크마다 다른 일이 벌어지는데 그걸 볼 수가 없었던** 것입니다.

⚠️ "매칭 실패 0건" 이 사람을 안심시켰습니다. 그 카운터는 JSON 을 읽는 데
   성공한 것들만 세기 때문에, 앞에서 다 죽으면 당연히 0 입니다.
   **아무 일도 안 일어났을 때와 전부 실패했을 때가 같은 숫자로 보입니다.**
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import labels                                          # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {extra}" if extra else ""))
    if not cond:
        FAILS.append(f"{name} {extra}".strip())


def _capture(fn, *a, **kw):
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*a, **kw)
    return out, buf.getvalue()


# ──────────────────────────────────────────────────────────────
def test_counts_and_names_the_reason():
    print("\n[세기] 못 읽은 것을 세고 이유를 말하는가")
    sk = labels.JsonSkips()
    check("처음엔 0", sk.n == 0)

    good = json.dumps({"a": 1}).encode()
    check("정상 JSON 은 읽힌다",
          labels._load_json_bytes(good, "ok.json", sk) == {"a": 1})
    check("정상은 안 센다", sk.n == 0)

    check("깨진 JSON 은 None", labels._load_json_bytes(b"{not json", "x.json", sk) is None)
    check("이유를 남긴다", any("JSON파싱" in k for k in sk.by_kind), str(sk.by_kind))

    # cp949 로 쓰인 한글 — utf-8 로는 못 읽습니다
    cp = json.dumps({"라벨": "비듬"}, ensure_ascii=False).encode("cp949")
    got = labels._load_json_bytes(cp, "cp949.json", sk)
    check("cp949 도 읽어낸다 (같은 버그로 세 번 막혔던 자리)",
          got == {"라벨": "비듬"}, str(got))

    # 어느 인코딩으로도 안 되는 바이트
    check("정말 못 읽으면 None",
          labels._load_json_bytes(b"\xff\xfe\x00\x80" * 8, "bad.json", sk) is None)
    check("인코딩 실패도 따로 센다", "인코딩" in sk.by_kind, str(sk.by_kind))


def test_stops_when_the_loss_is_big():
    print("\n[멈춤] 많이 잃으면 멈추는가")
    sk = labels.JsonSkips()
    for i in range(3):
        sk.add("JSON파싱(ValueError)", f"f{i}.json")

    # 3/1000 = 0.3% — 문턱 아래라 경고만
    _, log = _capture(sk.report, 1000, True)
    check("문턱 아래는 경고만", "⚠️" in log and "0.3%" in log, log.strip()[:60])

    # 3/10 = 30% — 멈춰야 합니다
    died = ""
    try:
        sk.report(10, verbose=True)
    except RuntimeError as exc:
        died = str(exc)
    check("문턱 위는 멈춘다", "못 읽었습니다" in died, died[:60])
    check("비율을 말한다", "30.0%" in died)
    check("종류를 말한다", "JSON파싱" in died)
    check("무엇을 못 읽었는지 예를 준다", "f0.json" in died)

    check("문턱은 2%", labels.JSON_LOSS_STOP == 0.02)


def test_build_from_zip_refuses_a_mostly_unreadable_zip():
    print("\n[통합] TL01 상황 재현 — zip 을 97% 못 읽으면")
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        zp = Path(tmp) / "TL01.zip"
        n_good, n_bad = 3, 97
        with zipfile.ZipFile(zp, "w") as z:
            for i in range(n_good):
                name = f"반려견/피부/일반카메라/무증상/IMG_{i:04d}"
                z.writestr(f"{name}.json", json.dumps({
                    "image_name": f"IMG_{i:04d}.jpg", "label": "A7"}))
                z.writestr(f"{name}.jpg", b"\xff\xd8\xff\xe0")
            for i in range(n_bad):
                name = f"반려견/피부/일반카메라/유증상/IMG_{i:05d}"
                # 읽히긴 하는데 JSON 이 아닌 바이트 (깨진 조각을 흉내)
                z.writestr(f"{name}.json", b"\x00\x1f\x8b\x08garbage\xff")
                z.writestr(f"{name}.jpg", b"\xff\xd8\xff\xe0")

        died = ""
        try:
            _capture(labels.build_from_zip, zp, save=False, verbose=True)
        except RuntimeError as exc:
            died = str(exc)
        check("멈춘다", "못 읽었습니다" in died, died.splitlines()[0][:70] if died else "안 멈춤")
        check("몇 개인지 말한다", f"{n_bad}/{n_good + n_bad}" in died, died[:80])
        check("예전엔 여기서 '✅ 완료' 가 찍혔습니다", bool(died))


if __name__ == "__main__":
    print("JSON 손실을 조용히 넘기지 않는가")
    for fn in (test_counts_and_names_the_reason,
               test_stops_when_the_loss_is_big,
               test_build_from_zip_refuses_a_mostly_unreadable_zip):
        fn()
    print()
    if FAILS:
        print(f"{len(FAILS)} check(s) failed:")
        for f in FAILS:
            print(f"  - {f}")
        sys.exit(1)
    print("all checks passed")
