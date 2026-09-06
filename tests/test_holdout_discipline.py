"""★ **holdout 을 여는 노트북이 늘어나지 않는가.**

    uv run python tests/test_holdout_discipline.py

holdout 은 "최종 1회만 보는" 데이터입니다. 그런데 이 규칙은 지금까지
`CLAUDE.md` 와 `notebooks/README.md` 에 **문장으로만** 있었습니다 —
검사가 없었습니다. 문장은 안 지켜져도 아무 일도 안 일어납니다.

**왜 위험한가**: holdout 을 두 번 보면 두 번째부터는 val 입니다. 결과가
나쁘다고 설정을 바꿔 다시 돌리는 순간, 우리가 "일반화된다" 고 말할 근거가
사라집니다. 그런데 그 순간에는 **아무 에러도 안 납니다** — 오히려 숫자가
좋아져서 기분이 좋습니다. 이 프로젝트가 계속 당한 종류입니다.

그래서 **누가 holdout 을 열 수 있는지**를 목록으로 못 박습니다. 새 노트북이
holdout 을 열면 여기서 실패하고, 그때 사람이 "정말 열어야 하나" 를 한 번
생각하게 됩니다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: holdout 을 열어도 되는 노트북. **여기 추가하려면 이유를 커밋 메시지에 적으세요.**
#:
#: ⚠️ **이 목록을 처음 만들면서 알았습니다 — 문서가 사실이 아니었습니다.**
#:    `CLAUDE.md` 와 `notebooks/README.md` 는 *"holdout 은 06 만 엽니다"* 라고
#:    적어뒀는데 실제로는 **05·06·07 셋이** 열고 있었습니다. 규칙이 05 → 06 으로
#:    바뀌던 때 05 를 안 고쳤고, 07 은 그 뒤에 생기면서 그냥 따라 열었습니다.
#:    **검사가 없으니 아무 일도 안 일어났습니다.**
#:
#:    이미 돌아간 것을 되돌릴 수는 없습니다(본 것을 안 본 것으로 못 만듭니다).
#:    그래서 **있는 그대로 적고**, 앞으로 **새 노트북이 여는 것**을 막습니다.
#:    ⚠️ 그 대가: 우리 holdout 숫자는 "한 번만 봤다" 보다 **노출이 큽니다.**
ALLOWED = {
    # 확정 재학습. 학습을 마치고 그 자리에서 최종 평가까지 합니다.
    "06_확정재학습_홀드아웃.ipynb",
    # holdout 확인 전용. 학습 없이 추론만, 결정이 굳은 뒤 한 번.
    "12_홀드아웃_확인.ipynb",
    # ── 아래 둘은 **이미 돌아간 것**입니다. 새로 여는 걸 허용하는 게 아닙니다 ──
    # STEP 5 의 보정·거절·Grad-CAM 평가가 holdout 에서 이뤄졌습니다.
    "05_평가_보정_GradCAM.ipynb",
    # STEP 16 혼동행렬·부위별 놓침을 holdout 에서 읽었습니다 (STEP 17 의 전제).
    "07_오답분석_혼동쌍.ipynb",
}

#: holdout 을 여는 신호. `split.get_holdout()` 이 정식 통로이고,
#: `is_holdout` 컬럼을 직접 거르는 것도 같은 일입니다.
SIGNALS = ("get_holdout(", 'is_holdout"]', "is_holdout']", ".is_holdout")

ok = fail = 0


def check(name: str, cond: bool, msg: str = "") -> None:
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {name}")
    else:
        fail += 1
        print(f"  FAIL  {name}" + (f"\n        {msg}" if msg else ""))


def opens_holdout(nb_path: Path) -> list[str]:
    """그 노트북의 **코드 셀**에서 holdout 을 여는 줄을 찾습니다."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    hits = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue                      # 설명에 적힌 건 여는 게 아닙니다
        for ln in "".join(cell["source"]).splitlines():
            s = ln.strip()
            if s.startswith("#"):
                continue                  # 주석도 아닙니다
            if any(sig in s for sig in SIGNALS):
                hits.append(s[:90])
    return hits


print("[1] holdout 을 여는 노트북이 목록 안에 있는가")
nbs = sorted((ROOT / "notebooks").glob("*.ipynb"))
check("노트북을 찾았다", len(nbs) >= 10, f"{len(nbs)}개")

offenders = {}
for f in nbs:
    hits = opens_holdout(f)
    if hits and f.name not in ALLOWED:
        offenders[f.name] = hits
    elif hits:
        print(f"        (허용) {f.name} — {len(hits)}줄")

check("목록 밖에서 holdout 을 여는 노트북이 없다", not offenders,
      "\n        ".join(f"{k}: {v[0]}" for k, v in offenders.items())
      + "\n        → 정말 열어야 하면 ALLOWED 에 추가하고 **이유를 커밋에 적으세요**")

print("\n[2] 허용 목록의 노트북이 실제로 존재하는가")
for name in sorted(ALLOWED):
    check(f"{name} 이 있다", (ROOT / "notebooks" / name).is_file())

print("\n[3] ★ 12 는 **학습을 하지 않는다** (확인 전용)")
nb12 = ROOT / "notebooks" / "12_홀드아웃_확인.ipynb"
if nb12.is_file():
    src = "\n".join("".join(c["source"]) for c in
                    json.loads(nb12.read_text(encoding="utf-8"))["cells"]
                    if c.get("cell_type") == "code")
    for bad in ("train.fit(", "experiments.train_and_measure("):
        check(f"학습을 안 부른다 ({bad[:-1]})", bad not in src)
    check("holdout 을 연다 (확인 전용이니 당연히)", "get_holdout(" in src)
    check("판정 기준을 노트북에서 새로 안 만든다 (src 를 부른다)",
          "experiments.naming_report(" in src
          and "experiments.granularity_report(" in src)

print("\n[4] 문서와 검사가 같은 말을 하는가")
claude = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
readme = (ROOT / "notebooks" / "README.md").read_text(encoding="utf-8")
# 06 만이라고 적혀 있으면 12 를 추가한 지금과 어긋납니다
check("CLAUDE.md 가 '06 만' 이라고 말하지 않는다",
      "holdout 은 06 만 엽니다" not in claude,
      "12 가 생겼으므로 문서도 같이 고쳐야 합니다")
check("notebooks/README 가 '06 하나뿐' 이라고 말하지 않는다",
      "06 하나뿐입니다" not in readme,
      "12 가 생겼으므로 문서도 같이 고쳐야 합니다")

print("\n" + "=" * 60)
print(f" 통과 {ok} / {ok + fail}")
raise SystemExit(1 if fail else 0)
