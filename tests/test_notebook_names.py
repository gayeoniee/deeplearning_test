"""노트북 셀의 **이름 오타** 검사.

⚠️ 문법 검사(`compile`)로는 `NameError` 를 못 잡습니다. 실제로 05 의 촬영 가이드
셀이 `cfg2` 를 참조했는데, 그건 노트북 03 의 변수 이름이었습니다. 문법은
멀쩡했고 Kaggle 에서 **15분을 돌린 뒤** 그 셀에서 죽었습니다.

여기서는 노트북 전체에서 **한 번도 대입되지 않은 이름**을 씁니다.
셀 순서까지 보지는 않습니다 (함수 안에서 나중 전역을 쓰는 건 정상이므로).
오타·다른 노트북 변수 가져다 쓰기를 잡는 게 목적입니다.

    uv run python tests/test_notebook_names.py
"""

from __future__ import annotations

import ast
import builtins
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# IPython 이 넣어주는 것들 + 노트북에서만 존재하는 이름
EXTRA = {"get_ipython", "display", "In", "Out", "exit", "quit", "__file__"}


def _clean(src: str) -> str:
    """`!pip ...`, `%%time` 같은 IPython 전용 줄을 지웁니다."""
    out = []
    for line in src.splitlines():
        st = line.lstrip()
        out.append("" if st.startswith(("!", "%", "?")) else line)
    return "\n".join(out)


class _Names(ast.NodeVisitor):
    def __init__(self) -> None:
        self.bound: set[str] = set()
        self.used: set[str] = set()

    def visit_Name(self, n: ast.Name) -> None:
        (self.bound if isinstance(n.ctx, (ast.Store, ast.Del)) else self.used).add(n.id)

    def visit_alias(self, n: ast.alias) -> None:
        name = n.asname or n.name.split(".")[0]
        self.bound.add(name)

    def _bind_args(self, a: ast.arguments) -> None:
        for arg in [*a.posonlyargs, *a.args, *a.kwonlyargs,
                    *([a.vararg] if a.vararg else []),
                    *([a.kwarg] if a.kwarg else [])]:
            self.bound.add(arg.arg)

    def _fn(self, n) -> None:
        self.bound.add(n.name)
        self._bind_args(n.args)
        self.generic_visit(n)

    visit_FunctionDef = visit_AsyncFunctionDef = _fn

    # ⚠️ lambda 인자도 묶어야 합니다. 안 그러면
    #       sorted(d.items(), key=lambda kv: -kv[1])
    #    의 `kv` 가 "정의되지 않은 이름" 으로 잡힙니다 (07 에서 실제로 걸렸습니다).
    def visit_Lambda(self, n: ast.Lambda) -> None:
        self._bind_args(n.args)
        self.generic_visit(n)

    def visit_ClassDef(self, n: ast.ClassDef) -> None:
        self.bound.add(n.name)
        self.generic_visit(n)

    def visit_ExceptHandler(self, n: ast.ExceptHandler) -> None:
        if n.name:
            self.bound.add(n.name)
        self.generic_visit(n)

    def visit_Global(self, n: ast.Global) -> None:
        self.bound.update(n.names)

    visit_Nonlocal = visit_Global


def check_notebook(path: Path) -> list[str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    v = _Names()
    unparsed = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = _clean("".join(c["source"]))
        try:
            v.visit(ast.parse(src))
        except SyntaxError as e:
            unparsed.append(f"셀 {i}: 파싱 실패 {e}")
    known = v.bound | set(dir(builtins)) | EXTRA
    bad = sorted(v.used - known)
    return unparsed + [f"정의되지 않은 이름: {n}" for n in bad]


# ──────────────────────────────────────────────────────────────
# 인자 이름 검사
#
# 이름 검사로는 `robust.usable_range(tolerance=...)` 같은 오타를 못 잡습니다
# (실제 인자는 tolerances). 실행해야 TypeError 가 나는데 그때는 이미 15분을
# 쓴 뒤입니다. src.* 호출의 키워드를 실제 시그니처와 대조합니다.
# ──────────────────────────────────────────────────────────────
SRC_MODULES = {q.stem for q in (ROOT / "src").glob("*.py") if q.stem != "__init__"}


def check_calls(path: Path) -> list[str]:
    import inspect

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    nb = json.loads(path.read_text(encoding="utf-8"))
    problems: list[str] = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        try:
            tree = ast.parse(_clean("".join(c["source"])))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            owner = node.func.value
            if not isinstance(owner, ast.Name) or owner.id not in SRC_MODULES:
                continue
            try:
                mod = __import__(f"src.{owner.id}", fromlist=["_"])
            except Exception:                                    # noqa: BLE001
                continue                      # torch 가 없는 환경 등 — 조용히 넘어감
            fn = getattr(mod, node.func.attr, None)
            if fn is None:
                problems.append(f"셀 {i}: src.{owner.id} 에 '{node.func.attr}' 가 없습니다")
                continue
            if not callable(fn):
                continue
            try:
                sig = inspect.signature(fn)
            except (TypeError, ValueError):
                continue
            if any(prm.kind is prm.VAR_KEYWORD for prm in sig.parameters.values()):
                continue
            names = set(sig.parameters)
            for kw in node.keywords:
                if kw.arg and kw.arg not in names:
                    problems.append(
                        f"셀 {i}: {owner.id}.{node.func.attr}(...) 에 "
                        f"'{kw.arg}' 인자가 없습니다 — 있는 것: {sorted(names)}")
    return problems



# ──────────────────────────────────────────────────────────────
# 정의하기 **전에** 쓰는 이름이 있나 (셀 순서대로 실행한다고 보고)
#
# ⚠️ 위쪽 check_notebook 은 노트북 전체에서 한 번이라도 대입되면 통과시킵니다.
#    그래서 "뒤 셀에서 import 하는 걸 앞 셀에서 쓰는" 경우를 못 잡습니다.
#    실제로 06 에서 두 번 당했습니다 — `calibrate` 는 40분짜리 학습이 끝난
#    뒤에 NameError 로 죽었고, `json`/`np` 도 같은 셀에 숨어 있었습니다.
#    임대 GPU 는 시간당 과금이라 이런 건 돈으로 셉니다.
# ──────────────────────────────────────────────────────────────
def defined_before_use(path: Path) -> list[tuple[int, str]]:
    """(셀 번호, 이름) — 그 셀 전에 정의된 적이 없는데 쓰는 것."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    known = set(dir(builtins)) | EXTRA
    bad: list[tuple[int, str]] = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        try:
            tree = ast.parse(_clean("".join(c["source"])))
        except SyntaxError:
            continue                      # 문법은 위쪽 검사가 봅니다
        v = _Names()
        v.visit(tree)
        bad += [(i, u) for u in sorted(v.used - known - v.bound)]
        known |= v.bound
    return bad


def _check_order() -> int:
    print("\n[실행 순서] 정의하기 전에 쓰는 이름이 있는가")
    n = 0
    for p in sorted((ROOT / "notebooks").glob("*.ipynb")):
        bad = defined_before_use(p)
        if bad:
            n += len(bad)
            for i, name in bad:
                print(f"  ❌ {p.name} 셀 {i}: '{name}' — 그 전에 정의가 없습니다")
        else:
            print(f"  ✅ {p.name}")
    if n:
        print(f"\n❌ {n}건 — 몇십 분 돌린 뒤에 NameError 로 죽습니다")
    return n


if __name__ == "__main__":
    fails = 0
    for nb_path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        problems = check_notebook(nb_path) + check_calls(nb_path)
        if problems:
            fails += len(problems)
            print(f"❌ {nb_path.name}")
            for p in problems:
                print(f"     {p}")
        else:
            print(f"✅ {nb_path.name}")
    print(f"\n{'문제 없음' if not fails else f'문제 {fails}건'}")
    fails += _check_order()
    sys.exit(1 if fails else 0)
