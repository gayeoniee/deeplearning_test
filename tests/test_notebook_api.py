"""노트북이 부르는 `src/` 함수가 **실제로 존재하고 인자가 맞는가** (실행 없이).

    uv run --extra train python tests/test_notebook_api.py

왜 필요한가
-----------
`tests/test_notebook_names.py` 는 이름 오타를 잡습니다. 그런데 **인자가 바뀐
경우**는 못 잡습니다 — 이름은 그대로인데 시그니처만 달라지면 노트북은
캐글에서 **30분을 돌린 뒤** TypeError 로 죽습니다.

2026-09-05 실측: `experiments.estimate_runtime` 이
`(model_names, img_size, n_train, epochs, ...)` 로 바뀌었는데 노트북 09 가 옛
호출 `[(MODEL, EPOCHS)]` 를 쓰고 있었습니다. 목록의 튜플은 **(모델, 해상도)**
이지 에폭이 아닙니다. 크롭 연결에만 31분이 걸리므로 실패 한 번이 비쌉니다.
(03h 는 같은 걸 이미 겪고 주석까지 남겼는데 그걸 안 읽고 베낀 게 원인입니다.)

⚠️ 이 검사는 **정적**입니다 — 인자 이름·개수만 봅니다. 값의 의미가 바뀐 경우
   (예: 튜플의 두 번째가 에폭 → 해상도)는 못 잡습니다. 그건 주석과 눈이 할 일입니다.
"""
from __future__ import annotations

import ast
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ok = fail = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {name}")
    else:
        fail += 1
        print(f"  FAIL  {name}  {detail}")


# 검사할 모듈 — 노트북이 `from src import ...` 로 쓰는 것들
import src.crop as crop            # noqa: E402
import src.evaluate as evaluate    # noqa: E402
import src.experiments as experiments  # noqa: E402
import src.labels as labels        # noqa: E402
import src.split as split          # noqa: E402
import src.stages as stages        # noqa: E402
import src.train as train          # noqa: E402

MODULES = {"crop": crop, "evaluate": evaluate, "experiments": experiments,
           "labels": labels, "split": split, "stages": stages, "train": train}


def calls_in(source: str):
    """`모듈.함수(...)` 호출을 뽑아 (모듈, 함수, 위치인자수, 키워드이름들) 로."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    out = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
            mod = f.value.id
            if mod in MODULES:
                kw = [k.arg for k in n.keywords if k.arg]
                has_star = any(k.arg is None for k in n.keywords)
                out.append((mod, f.attr, len(n.args), kw, has_star))
    return out


print("\n[1] 노트북의 src 호출이 실제 시그니처와 맞는가")
nbs = sorted((ROOT / "notebooks").glob("*.ipynb"))
check("노트북을 찾았다", len(nbs) > 0, str(len(nbs)))

seen = 0
for nb_path in nbs:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    src_all = "\n".join("".join(c["source"]) for c in nb["cells"]
                        if c["cell_type"] == "code")
    for mod, fname, npos, kwnames, has_star in calls_in(src_all):
        seen += 1
        fn = getattr(MODULES[mod], fname, None)
        if fn is None:
            check(f"{nb_path.name}: {mod}.{fname} 가 존재", False, "없는 이름")
            continue
        if not callable(fn):
            continue
        sig = inspect.signature(fn)
        if has_star:            # **kwargs 로 넘기면 정적으로 못 봅니다
            continue
        try:
            sig.bind_partial(*([None] * npos), **{k: None for k in kwnames})
        except TypeError as e:
            check(f"{nb_path.name}: {mod}.{fname}(…)", False,
                  f"{e}  실제: {sig}")
        else:
            ok += 1
print(f"  (검사한 호출 {seen}개)")
check("호출을 실제로 하나 이상 찾았다", seen > 0, str(seen))

print("\n[2] 자주 틀리는 것 — estimate_runtime 은 epochs 가 **별도 인자**")
sig = inspect.signature(experiments.estimate_runtime)
check("epochs 가 필수 인자로 있다", "epochs" in sig.parameters)
check("epochs 에 기본값이 없다 (빠뜨리면 죽어야 함)",
      sig.parameters["epochs"].default is inspect.Parameter.empty)
check("model_names 가 첫 인자", list(sig.parameters)[0] == "model_names")

print("\n" + "=" * 60)
print(f" 통과 {ok} / {ok + fail}")
sys.exit(1 if fail else 0)
