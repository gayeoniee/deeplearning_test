"""학습 루프.

딥러닝 학습 루프는 결국 이 네 줄의 반복입니다:

    pred = model(x)                # 1. 순전파: 예측
    loss = criterion(pred, y)      # 2. 얼마나 틀렸나
    loss.backward()                # 3. 역전파: 각 가중치의 책임(기울기) 계산
    optimizer.step()               # 4. 그 반대 방향으로 조금 이동

나머지는 전부 "이걸 빠르고 안정적으로 하는 장치"입니다.
AMP=빠르게, 스케줄러=lr 조절, EMA=가중치 평균, clip=폭주 방지, accum=배치 늘리기.
(docs/basics/05_학습루프_옵티마이저_스케줄러.md 에서 자세히 설명합니다)

    from src import train
    res = train.fit(model, dl_tr, dl_va, cfg)
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from src import env
from src.config import CFG
from src.data import class_weights, mixup_cutmix
from src.models import ModelEMA, param_groups


# ──────────────────────────────────────────────────────────────
# 손실
# ──────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """쉬운 샘플의 기여를 줄여 어려운 샘플에 집중시키는 손실.

    극심한 불균형에서 유용하지만 만능은 아닙니다. gamma=0 이면 그냥 CE 입니다.
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ls = label_smoothing

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight,
                             label_smoothing=self.ls, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def build_criterion(cfg: CFG, ds_train=None, device: str = "cuda") -> nn.Module:
    w = None
    if cfg.balance_strategy == "class_weight" and ds_train is not None:
        w = class_weights(ds_train).to(device)
        print(f"[train] 클래스 가중치: {[round(float(x), 2) for x in w]}")
    if cfg.focal_gamma > 0:
        return FocalLoss(cfg.focal_gamma, w, cfg.label_smoothing)
    return nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.label_smoothing)


# ──────────────────────────────────────────────────────────────
# 스케줄러
# ──────────────────────────────────────────────────────────────
def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int, min_ratio: float = 0.01):
    """앞부분은 lr 을 0→목표까지 올리고(warmup), 그 뒤 cosine 으로 내립니다.

    warmup 이 필요한 이유: 학습 초반 랜덤 헤드가 만드는 큰 기울기가
    사전학습된 백본을 망가뜨리는 걸 막습니다.
    """
    def fn(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


# ──────────────────────────────────────────────────────────────
# 중단 대비 — 체크포인트 영속화
# ──────────────────────────────────────────────────────────────
# ⚠️ Colab 의 `/content` 는 **세션이 끊기면 통째로 사라집니다.** 25에폭 학습이
#    23에폭에서 끊기면 처음부터 다시입니다. 그래서 두 가지를 합니다:
#
#    1) 매 에폭 `last.pt` 를 저장합니다 (모델 + EMA + 옵티마이저 + 스케줄러 + RNG)
#    2) 그걸 살아남는 저장소(Drive)로 복사합니다
#
#    다음 세션에서 `fit(..., resume=True)` 를 부르면 Drive → 로컬로 되돌린 뒤
#    끊긴 에폭 다음부터 이어갑니다. 이미 끝난 학습이면 아예 건너뜁니다.
#
# 비용: resnet50 기준 last.pt ≈ 400MB, best.pt ≈ 200MB. Drive 쓰기가
#       에폭당 10~30초 붙습니다. 80분을 날리는 것보다 훨씬 쌉니다.
#       `persist_every=2` 로 줄일 수 있습니다 (그만큼 되돌아가는 폭도 커집니다).

_SYNC_FILES = ("best.pt", "last.pt", "history.csv", "config.json", "result.json")


def ckpt_dir(exp: str) -> Path:
    """로컬(빠른 디스크) 체크포인트 폴더."""
    p = env.ensure_dirs()["checkpoints"] / exp
    p.mkdir(parents=True, exist_ok=True)
    return p


def persist_dir(exp: str) -> Path | None:
    """세션이 끊겨도 남는 체크포인트 폴더. 없으면 None."""
    root = env.persist_root()
    if root is None:
        return None
    p = Path(root) / "checkpoints" / exp
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return p


def _same_dir(a: Path | None, b: Path | None) -> bool:
    """로컬 실행에서는 영속 저장소가 작업 폴더와 같은 곳입니다 — 복사할 게 없습니다."""
    if a is None or b is None:
        return False
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return False


def _copy_atomic(src: Path, dst: Path) -> None:
    """중간에 끊겨도 반쪽 파일이 남지 않게 임시 이름으로 쓰고 바꿔치기합니다."""
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def sync_to_persist(exp: str, files: tuple[str, ...] = _SYNC_FILES,
                    verbose: bool = False) -> float:
    """로컬 → 영속 저장소. 걸린 초를 돌려줍니다 (영속 저장소가 없으면 0)."""
    dst_dir = persist_dir(exp)
    src_dir = ckpt_dir(exp)
    if dst_dir is None or _same_dir(dst_dir, src_dir):
        return 0.0
    t0 = time.time()
    done = []
    for name in files:
        src = src_dir / name
        if not src.exists():
            continue
        try:
            _copy_atomic(src, dst_dir / name)
            done.append(name)
        except OSError as exc:
            # Drive 가 잠깐 끊기는 일은 흔합니다. 학습을 죽이지는 않습니다.
            print(f"⚠️ [train] '{name}' 백업 실패 — {type(exc).__name__}: {exc}")
    dt = time.time() - t0
    if verbose and done:
        print(f"[train] 백업 {len(done)}개 → {dst_dir}  ({dt:.1f}초)")
    return dt


def restore_from_persist(exp: str, verbose: bool = True) -> bool:
    """영속 저장소 → 로컬. 되돌릴 게 있었으면 True."""
    src_dir = persist_dir(exp)
    dst_dir = ckpt_dir(exp)
    if src_dir is None or not src_dir.exists() or _same_dir(src_dir, dst_dir):
        return False
    n = 0
    # 고정 목록 + 이 실험이 남긴 logits 캐시 전부
    names = list(_SYNC_FILES) + sorted(p.name for p in src_dir.glob("logits_*.npz"))
    for name in names:
        src = src_dir / name
        if not src.exists():
            continue
        dst = dst_dir / name
        # 로컬이 더 새것이면 덮어쓰지 않습니다 (같은 세션에서 재실행한 경우)
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            continue
        try:
            _copy_atomic(src, dst)
            n += 1
        except OSError as exc:
            print(f"⚠️ [train] '{name}' 복원 실패 — {type(exc).__name__}: {exc}")
    if n and verbose:
        print(f"[train] 이전 세션 체크포인트 {n}개 복원 ← {src_dir}")
    return n > 0


# ──────────────────────────────────────────────────────────────
# 노트북 사이의 인계 (Kaggle 은 노트북마다 세션이 따로입니다)
# ──────────────────────────────────────────────────────────────
# 03 이 세 시간 걸려 만든 것들은 전부 그 세션의 /kaggle/working 안에 있습니다.
# 05 를 열면 그건 이미 없습니다. 05 가 필요로 하는 건 두 종류입니다:
#
#   · 가중치      checkpoints/<실험>/best.pt
#   · 실행 결과   stage1_threshold.json (임계값·크롭·실험 이름), reports/*.json
#
# 둘 중 하나만 가져오면 05 는 여전히 죽습니다. 그래서 `import_previous_run()`
# 하나로 묶었습니다. 03 을 'Save & Run All (Commit)' 로 돌린 뒤
# 05 에서 [Add Input → Notebook Output] 으로 붙이면 경로는 알아서 찾습니다.

# 실행 결과 파일 — 작고, 없으면 05 가 시작 직후 KeyError 로 죽는 것들입니다.
_RUN_FILES = ("stage1_threshold.json", "best_crop.txt", "best_model.json")
# 훑지 않을 폴더. crops 는 4만 장이라 들어가면 탐색이 하염없이 느려집니다.
_SKIP_WALK = {"crops", "manifests", ".git", "__pycache__", "site-packages", ".venv"}


def _walk_inputs(max_depth: int = 4):
    """붙어 있는 입력을 얕게 훑습니다. 무거운 폴더와 작업 폴더는 건너뜁니다."""
    # ⚠️ 작업 폴더 자신은 제외합니다 — 이번 세션이 방금 쓴 파일을
    #    "이전 실행에서 가져온 것" 으로 착각하면 안 됩니다.
    try:
        here = env.work_root().resolve()
    except OSError:
        here = None
    for root in env._search_roots():
        stack = [(root, 0)]
        while stack:
            d, depth = stack.pop()
            if here is not None and d.resolve() == here:
                continue
            yield d
            if depth >= max_depth:
                continue
            try:
                kids = sorted(p for p in d.iterdir()
                              if p.is_dir() and p.name not in _SKIP_WALK)
            except OSError:
                continue
            stack += [(k, depth + 1) for k in kids]


def find_checkpoint_sources() -> list[Path]:
    """붙어 있는 입력에서 `checkpoints/<실험>/best.pt` 를 담은 폴더를 찾습니다."""
    out: list[Path] = []
    for d in _walk_inputs():
        ck = d / "checkpoints"
        if ck.is_dir() and any(ck.glob("*/best.pt")) and ck not in out:
            out.append(ck)
    return out


def _has_run_files(d: Path) -> bool:
    if any((d / f).is_file() for f in _RUN_FILES):
        return True
    rep = d / "reports"
    return rep.is_dir() and any(rep.glob("*.json"))


def _run_roots_under(base: Path, max_depth: int = 4) -> list[Path]:
    """`base` 아래에서 결과 파일이 있는 폴더 (경로를 직접 줬을 때)."""
    out, stack = [], [(base, 0)]
    while stack:
        d, depth = stack.pop()
        if _has_run_files(d) and d not in out:
            out.append(d)
        if depth >= max_depth:
            continue
        try:
            stack += [(k, depth + 1) for k in sorted(d.iterdir())
                      if k.is_dir() and k.name not in _SKIP_WALK]
        except OSError:
            continue
    return out


def find_run_output_roots() -> list[Path]:
    """이전 실행이 남긴 결과 파일(`stage1_threshold.json` 등)이 있는 폴더."""
    out: list[Path] = []
    for d in _walk_inputs():
        if d not in out and _has_run_files(d):
            out.append(d)
    return out


def import_run_files(src: str | Path | None = None, verbose: bool = True) -> list[str]:
    """이전 실행의 결과 JSON 을 작업 폴더로 가져옵니다.

    ⚠️ **이미 있는 파일은 덮지 않습니다.** 이번 세션에서 방금 만든 결과를
       예전 입력이 덮어쓰면, 낡은 임계값으로 평가하고도 눈치채지 못합니다.
    """
    dst_root = env.work_root()
    got: list[str] = []
    roots = find_run_output_roots() if src is None else _run_roots_under(Path(src))
    for src in roots:
        pairs = [(src / f, dst_root / f) for f in _RUN_FILES if (src / f).is_file()]
        pairs += [(p, dst_root / "reports" / p.name)
                  for p in sorted((src / "reports").glob("*.json"))
                  if (src / "reports").is_dir()]
        for s, d in pairs:
            if d.exists():
                if verbose:
                    print(f"  ⏭️  {d.name} — 이미 있어서 건너뜀 (이번 세션 것이 우선)")
                continue
            d.parent.mkdir(parents=True, exist_ok=True)
            try:
                _copy_atomic(s, d)
                got.append(d.name)
            except OSError as exc:
                print(f"  ⚠️ {d.name} 복사 실패 — {type(exc).__name__}")
    if verbose:
        print(f"[train] 실행 결과 {len(got)}개 가져옴" + (f" {got}" if got else ""))
    return got


def infer_run_settings() -> dict:
    """체크포인트 **폴더 이름**에서 실행 설정을 되살립니다.

    `stage1_threshold.json` 이 안 넘어와도 크롭과 실험 이름만은 살릴 수 있습니다 —
    이름에 다 적혀 있으니까요:

        stage1_resnet50_full_moderate  → stage 1 / resnet50 / full 크롭
        stage2_resnet50_m2.5_moderate  → stage 2 / resnet50 / m2.5 크롭

    가중치는 무겁고 JSON 은 가벼운데, 인계에서 **가벼운 쪽이 더 잘 빠집니다.**
    (Kaggle 출력을 데이터셋으로 만들 때 실제로 그랬습니다)
    """
    from src import crop as _crop

    roots = [env.ensure_dirs()["checkpoints"]]
    p = env.persist_root()
    if p:
        roots.append(p / "checkpoints")

    names: set[str] = set()
    for r in roots:
        if r.is_dir():
            names |= {d.name for d in r.iterdir()
                      if d.is_dir() and (d / "best.pt").exists()}

    out: dict = {}
    for n in sorted(names):
        for stage in (1, 2):
            if not n.startswith(f"stage{stage}_"):
                continue
            toks = n.split("_")[1:]
            ci = next((i for i, t in enumerate(toks)
                       if t == "full" or _crop.margin_of_tag(t) or _crop.fixed_of_tag(t)), None)
            if ci is None:
                continue
            out[f"stage{stage}_exp"] = n
            out[f"stage{stage}_crop"] = toks[ci]
            out[f"stage{stage}_model"] = "_".join(toks[:ci]) or "resnet50"
    return out


def explain_handoff() -> str:
    """인계가 왜 안 됐는지 **눈으로 확인할 수 있게** 현재 상태를 적습니다.

    "파일이 없습니다" 만 보면 원인이 세 갈래로 갈립니다 —
    입력을 안 붙였나 / 붙였는데 출력이 비었나 / 붙었는데 못 찾았나.
    셋을 구분하려면 붙어 있는 게 뭔지 실제로 보여줘야 합니다.
    """
    W = env.work_root()
    lines = ["", "── 인계 진단 ──────────────────────────────────────"]
    for root in env._search_roots():
        try:
            kids = sorted(p.name for p in root.iterdir())
        except OSError as exc:
            lines.append(f"  {root}  (읽기 실패: {type(exc).__name__})")
            continue
        lines.append(f"  {root}  →  {kids[:12]}" + (" …" if len(kids) > 12 else ""))
    ck = find_checkpoint_sources()
    rr = find_run_output_roots()
    lines.append(f"  checkpoints/ 를 담은 폴더 : {[str(p) for p in ck] or '없음'}")
    lines.append(f"  결과 JSON 을 담은 폴더    : {[str(p) for p in rr] or '없음'}")
    here = sorted(p.name for p in W.iterdir()) if W.exists() else []
    lines.append(f"  작업 폴더 {W} : {here or '비어 있음'}")
    lines.append("───────────────────────────────────────────────────")
    if not ck and not rr:
        lines += [
            "붙어 있는 입력에 이전 실행의 산출물이 없습니다.",
            "  1) 03 을 [Save Version → Save & Run All (Commit)] 로 돌렸는지",
            "  2) 05 의 [Add Input] → **Notebooks** 탭에서 그 03 노트북을 골랐는지",
            "     (이 탭이 곧 '노트북 출력' 입니다 — 다운로드할 필요 없습니다)",
            "  3) 그래도 안 보이면 03 의 Output 을 Private 데이터셋으로 만들어 붙이세요",
            "  4) 경로를 직접 주려면:  train.import_previous_run(src='/kaggle/input/<이름>')",
        ]
    return "\n".join(lines)


def import_previous_run(src: str | Path | None = None, verbose: bool = True) -> dict:
    """이전 노트북 실행의 **가중치 + 결과 파일**을 한 번에 가져옵니다.

        train.import_previous_run()                        # 알아서 찾기
        train.import_previous_run("/kaggle/input/xxx")     # 경로를 직접 줄 때

    붙어 있는 게 없으면 멈추지는 않되, **왜 없는지 진단을 출력**합니다.
    같은 세션에서 방금 학습했다면 가져올 게 없는 게 정상입니다.
    """
    ck = import_checkpoints(src, verbose=verbose)
    files = import_run_files(src, verbose=verbose)
    # ⚠️ 한쪽만 와도 진단을 찍습니다. 가중치는 왔는데 JSON 이 안 온 경우가
    #    실제로 있었고, 그때 05 는 한참 뒤에야 "임계값이 없다" 로 죽었습니다.
    if verbose and not (ck and files):
        print(explain_handoff())
    return {"checkpoints": ck, "files": files}


def import_checkpoints(src: str | Path | None = None, exps: list[str] | None = None,
                       verbose: bool = True) -> list[str]:
    """다른 환경에서 만든 체크포인트를 가져옵니다 (Colab → Kaggle 이주 등).

    `src` 는 `checkpoints/` 를 담은 폴더이거나 `checkpoints/` 자체입니다.
    Kaggle 이면 보통 `/kaggle/input/<데이터셋이름>` 입니다.

        train.import_checkpoints("/kaggle/input/dogskin-ckpt")
        train.import_checkpoints()          # 붙어 있는 입력에서 알아서 찾기

    `src` 를 생략하면 `find_checkpoint_sources()` 가 찾은 곳을 **전부** 가져옵니다.
    하나도 없으면 조용히 빈 목록을 돌려줍니다 — 같은 세션에서 방금 학습했다면
    가져올 게 없는 게 정상이라, 여기서 멈추면 안 됩니다.

    ⚠️ 학습 환경이 달라도 가중치는 그대로 쓸 수 있습니다. 다만 `last.pt` 로
       이어서 학습할 때는 배치 크기·워커 수가 달라져 배치 순서가 바뀝니다
       (결과가 소수점 셋째 자리에서 흔들립니다 — cautions/09 참고).
    """
    if src is None:
        found = find_checkpoint_sources()
        if not found:
            if verbose:
                print("[train] 붙어 있는 입력에 체크포인트가 없습니다 "
                      "(같은 세션에서 학습했다면 정상입니다)")
            return []
        out: list[str] = []
        for f in found:
            out += import_checkpoints(f, exps=exps, verbose=verbose)
        return out

    src = Path(src)
    # Kaggle 에 zip 으로 올렸다면(자동 해제가 안 됐다면) 먼저 풉니다
    if src.is_file() and src.suffix == ".zip":
        import zipfile

        out = env.work_root() / "_imported_ckpt"
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(src) as z:
            z.extractall(out)
        print(f"[train] {src.name} 해제 → {out}")
        src = out

    if (src / "checkpoints").is_dir():
        root = src
    else:
        # 경로를 직접 줬는데 한두 단계 더 들어가 있는 경우 (노트북 출력의 data/work/…)
        deeper = next((c for c in _run_roots_under(src)
                       if (c / "checkpoints").is_dir()
                       and any((c / "checkpoints").glob("*/best.pt"))), None)
        root = deeper or src
    root = root / "checkpoints" if (root / "checkpoints").is_dir() else root
    if not root.is_dir():
        raise FileNotFoundError(
            f"{src} 안에서 체크포인트 폴더를 찾지 못했습니다.\n"
            f"  기대한 구조: {src}/checkpoints/<실험이름>/best.pt  또는\n"
            f"                {src}/<실험이름>/best.pt"
        )

    done = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if exps and d.name not in exps:
            continue
        dst = ckpt_dir(d.name)
        n = 0
        for f in d.iterdir():
            if f.is_file():
                try:
                    _copy_atomic(f, dst / f.name)
                    n += 1
                except OSError as exc:
                    print(f"⚠️ [train] '{d.name}/{f.name}' 복사 실패 — {type(exc).__name__}")
        if n:
            done.append(d.name)
            sync_to_persist(d.name)          # 이 환경의 영속 저장소에도 남깁니다
            if verbose:
                st = training_state(d.name, check_persist=False)
                mark = "✅ 완료" if st["completed"] else f"⏸️ {st['epochs_done']}에폭"
                print(f"  {mark}  {d.name}  ({n}개 파일)")
    if verbose:
        print(f"[train] 체크포인트 {len(done)}개 실험을 가져왔습니다 ← {root}")
        if not done:
            print(f"  ⚠️ 가져온 게 없습니다. {root} 안에 실험 폴더가 있는지 확인하세요.")
    return done


def training_state(exp: str, check_persist: bool = True) -> dict:
    """이 실험이 어디까지 갔는지. 학습을 시작하기 전에 물어봅니다."""
    if check_persist:
        restore_from_persist(exp, verbose=False)
    d = ckpt_dir(exp)
    out = {"exp": exp, "dir": str(d), "completed": False,
           "epochs_done": 0, "best_score": None, "best_epoch": None,
           "target_epochs": 0, "early_stopped": False,
           "has_last": (d / "last.pt").exists(), "has_best": (d / "best.pt").exists(),
           "persist": str(persist_dir(exp) or "")}
    rj = d / "result.json"
    if rj.exists():
        try:
            r = json.loads(rj.read_text(encoding="utf-8"))
            out["completed"] = bool(r.get("completed"))
            out["best_score"] = r.get("best_score")
            out["best_epoch"] = r.get("best_epoch")
            out["epochs_done"] = len(r.get("history") or [])
            out["early_stopped"] = bool(r.get("early_stopped"))
            # 예전 형식(target_epochs 없음)은 돌린 만큼이 목표였다고 봅니다.
            out["target_epochs"] = int(r.get("target_epochs") or out["epochs_done"])
        except (OSError, json.JSONDecodeError):
            pass
    if out["has_last"]:
        try:
            ck = torch.load(d / "last.pt", map_location="cpu", weights_only=False)
            out["epochs_done"] = max(out["epochs_done"], int(ck.get("epoch", -1)) + 1)
            out["best_score"] = ck.get("best", out["best_score"])
        except Exception:
            pass
    return out


def _rng_state() -> dict:
    st = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
          "python": random.getstate()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st: dict | None) -> None:
    if not st:
        return
    try:
        torch.set_rng_state(st["torch"].cpu() if torch.is_tensor(st["torch"]) else st["torch"])
        np.random.set_state(st["numpy"])
        random.setstate(st["python"])
        if torch.cuda.is_available() and st.get("cuda") is not None:
            torch.cuda.set_rng_state_all(st["cuda"])
    except Exception as exc:
        print(f"⚠️ [train] 난수 상태 복원 실패 (계속 진행) — {type(exc).__name__}")


# ──────────────────────────────────────────────────────────────
# 결과 컨테이너
# ──────────────────────────────────────────────────────────────
@dataclass
class FitResult:
    best_score: float = 0.0
    best_epoch: int = -1
    best_ckpt: str = ""
    history: list[dict] = field(default_factory=list)
    cfg: dict = field(default_factory=dict)
    elapsed_sec: float = 0.0
    resumed_from: int = 0        # 이어서 시작한 에폭 (0 이면 처음부터)
    skipped: bool = False        # 이미 끝나 있어서 학습을 아예 건너뜀

    @classmethod
    def from_dir(cls, d: Path | str, cfg: dict | None = None) -> "FitResult":
        """result.json 으로부터 되살립니다 (이미 끝난 학습을 건너뛸 때)."""
        d = Path(d)
        r = json.loads((d / "result.json").read_text(encoding="utf-8"))
        return cls(best_score=float(r.get("best_score", 0.0)),
                   best_epoch=int(r.get("best_epoch", -1)),
                   best_ckpt=str(d / "best.pt"),
                   history=list(r.get("history") or []),
                   cfg=cfg or r.get("cfg") or {},
                   elapsed_sec=float(r.get("elapsed_sec", 0.0)),
                   skipped=True)

    def summary(self) -> None:
        print(f"\n최고 {self.cfg.get('monitor', 'score')} = {self.best_score:.4f} "
              f"(epoch {self.best_epoch})  |  {self.elapsed_sec / 60:.1f}분")
        print(f"체크포인트: {self.best_ckpt}")

    def plot(self) -> None:
        import matplotlib.pyplot as plt

        if not self.history:
            return
        h = self.history
        ep = [r["epoch"] for r in h]
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
        ax[0].plot(ep, [r["train_loss"] for r in h], label="train")
        ax[0].plot(ep, [r["val_loss"] for r in h], label="val")
        ax[0].set_title("loss"); ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].grid(alpha=.3)
        ax[1].plot(ep, [r["val_macro_f1"] for r in h], label="macro-F1")
        ax[1].plot(ep, [r["val_acc"] for r in h], label="accuracy", ls="--")
        ax[1].set_title("val 성능"); ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].grid(alpha=.3)
        plt.tight_layout(); plt.show()
        print("💡 train loss 는 계속 내려가는데 val loss 가 올라가면 = 과적합 시작 지점입니다.")


# ──────────────────────────────────────────────────────────────
# 한 에폭
# ──────────────────────────────────────────────────────────────
def _train_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg, ema, device, n_cls):
    model.train()
    total, n = 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    amp_dtype = torch.bfloat16 if env.device_info().bf16 else torch.float16
    use_amp = cfg.amp and device == "cuda"

    pbar = tqdm(loader, desc="train", leave=False)
    for step, (x, y) in enumerate(pbar):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x, ya, yb, lam = mixup_cutmix(x, y, cfg, n_cls)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            out = model(x)
            loss = (lam * criterion(out, ya) + (1 - lam) * criterion(out, yb)
                    if lam < 1.0 else criterion(out, y))
            loss = loss / cfg.grad_accum

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % cfg.grad_accum == 0:
            if cfg.clip_grad_norm:
                if scaler is not None and scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            if scaler is not None and scaler.is_enabled():
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            if ema is not None:
                ema.update(model)

        bs = x.size(0)
        total += loss.item() * cfg.grad_accum * bs
        n += bs
        # log_every 가 전체 스텝 수보다 크면 진행바가 멈춰 보이므로 항상 갱신합니다.
        if step % max(min(cfg.log_every, len(loader) // 10 or 1), 1) == 0:
            pbar.set_postfix(loss=f"{total / max(n, 1):.4f}",
                             lr=f"{optimizer.param_groups[0]['lr']:.2e}")
    return total / max(n, 1)


@torch.no_grad()
def evaluate_loader(model, loader, criterion, device, n_cls: int, tta_hflip: bool = False):
    """검증. logits/labels 를 함께 돌려주므로 보정·임계값 탐색에 바로 씁니다."""
    model.eval()
    logits_all, y_all = [], []
    total, n = 0.0, 0
    amp_dtype = torch.bfloat16 if env.device_info().bf16 else torch.float16
    use_amp = device == "cuda"

    for x, y in tqdm(loader, desc="val", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            out = model(x)
            if tta_hflip:
                out = (out + model(torch.flip(x, dims=[3]))) / 2
        out = out.float()
        if criterion is not None:
            total += criterion(out, y).item() * x.size(0)
        n += x.size(0)
        logits_all.append(out.cpu())
        y_all.append(y.cpu())

    logits = torch.cat(logits_all)
    ys = torch.cat(y_all)
    return total / max(n, 1), logits, ys


def quick_metrics(logits: torch.Tensor, y: torch.Tensor) -> dict:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    pred = logits.argmax(1).numpy()
    yy = y.numpy()
    return {
        "acc": float(accuracy_score(yy, pred)),
        "balanced_acc": float(balanced_accuracy_score(yy, pred)),
        "macro_f1": float(f1_score(yy, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(yy, pred, average="weighted", zero_division=0)),
    }


# ──────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────
def fit(
    model: nn.Module,
    dl_train,
    dl_val,
    cfg: CFG,
    ds_train=None,
    device: str | None = None,
    exp_name: str | None = None,
    verbose: bool = True,
    resume: bool = True,
    persist: bool = True,
    persist_every: int = 1,
    restore_best: bool = True,
) -> FitResult:
    """학습 루프.

    resume=True (기본): 같은 `exp_name` 의 체크포인트가 있으면 이어서 합니다.
        - 이미 끝난 학습이면 **아무것도 안 하고** best 가중치만 model 에 얹어 돌려줍니다.
        - 중간에 끊긴 학습이면 다음 에폭부터 이어갑니다.
        처음부터 다시 하려면 `resume=False` (기존 기록을 덮어씁니다).

    persist=True (기본): 매 에폭 체크포인트를 세션 밖(Drive 등)으로 복사합니다.
        Colab 세션이 끊겨도 살아남습니다. → `env.persist_root()`

    restore_best=True (기본): 학습이 끝나면 **best 에폭의 가중치를 model 에 되돌립니다.**
        ⚠️ 이게 없으면 model 은 마지막 에폭 가중치를 들고 있어서, 저장된 best.pt 와
           노트북에서 평가하는 대상이 서로 다른 모델이 됩니다.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    exp = exp_name or cfg.exp_name
    n_cls = getattr(dl_train.dataset, "classes", None)
    n_cls = len(n_cls) if n_cls else int(max(dl_train.dataset.targets)) + 1

    ck_dir = ckpt_dir(exp)
    if resume and persist:
        restore_from_persist(exp, verbose=verbose)

    # ── 이미 끝난 학습이면 건너뜁니다 ────────────────────────────
    # 단, cfg.epochs 를 늘려서 다시 부른 경우는 "더 돌려보자" 는 뜻이므로
    # 건너뛰지 않고 이어서 갑니다 (조기 종료로 끝난 건 이어가도 의미 없음).
    if resume:
        st = training_state(exp, check_persist=False)
        finished = st["completed"] and st["has_best"] and (
            st["early_stopped"] or cfg.epochs <= st["target_epochs"])
        if finished:
            res = FitResult.from_dir(ck_dir, cfg=cfg.to_dict())
            if verbose:
                why = " (조기 종료)" if st["early_stopped"] else ""
                print(f"⏭️  '{exp}' 은 이미 끝난 학습입니다{why} — 건너뜁니다.\n"
                      f"    최고 {cfg.monitor} = {res.best_score:.4f} "
                      f"(epoch {res.best_epoch}, {res.elapsed_sec / 60:.1f}분 소요)\n"
                      f"    다시 학습하려면 resume=False, 더 돌리려면 epochs 를 늘리세요.")
            if restore_best:
                _load_into(model, ck_dir / "best.pt", verbose=verbose)
            model.to(device)
            return res
        if st["completed"] and st["has_last"] and cfg.epochs > st["target_epochs"]:
            if verbose:
                print(f"🔁 '{exp}' 을 {st['target_epochs']} → {cfg.epochs} 에폭으로 "
                      f"연장합니다.")
    elif (ck_dir / "history.csv").exists():
        # resume=False 는 "처음부터" 라는 뜻이므로 이전 기록을 치웁니다.
        for name in ("history.csv", "last.pt", "result.json"):
            (ck_dir / name).unlink(missing_ok=True)

    model = model.to(device)
    if device == "cuda":
        model = model.to(memory_format=torch.channels_last)

    criterion = build_criterion(cfg, ds_train or dl_train.dataset, device)
    optimizer = torch.optim.AdamW(param_groups(model, cfg))
    steps_per_epoch = max(len(dl_train) // cfg.grad_accum, 1)
    scheduler = cosine_with_warmup(
        optimizer, cfg.warmup_epochs * steps_per_epoch, cfg.epochs * steps_per_epoch
    )
    use_scaler = cfg.amp and device == "cuda" and not env.device_info().bf16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    ema = ModelEMA(model, cfg.ema_decay) if cfg.ema_decay > 0 else None

    cfg.save(ck_dir / "config.json")
    log_path = ck_dir / "history.csv"

    res = FitResult(cfg=cfg.to_dict())
    best, bad_epochs = -1.0, 0
    start_epoch = 0
    prior_sec = 0.0

    # ── 중간에 끊긴 학습 이어받기 ────────────────────────────────
    if resume and (ck_dir / "last.pt").exists():
        try:
            ck = torch.load(ck_dir / "last.pt", map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            if ema is not None and ck.get("ema") is not None:
                ema.ema.load_state_dict(ck["ema"])
            optimizer.load_state_dict(ck["optimizer"])
            if ck.get("scheduler") is not None:
                scheduler.load_state_dict(ck["scheduler"])
            if ck.get("scaler") is not None:
                scaler.load_state_dict(ck["scaler"])
            _set_rng_state(ck.get("rng"))
            start_epoch = int(ck["epoch"]) + 1
            best = float(ck.get("best", -1.0))
            bad_epochs = int(ck.get("bad_epochs", 0))
            res.history = list(ck.get("history") or [])
            res.best_score = float(ck.get("best_score", max(best, 0.0)))
            res.best_epoch = int(ck.get("best_epoch", -1))
            res.best_ckpt = str(ck_dir / "best.pt")
            res.resumed_from = start_epoch
            prior_sec = float(ck.get("elapsed_sec", 0.0))
            if verbose:
                print(f"▶️  '{exp}' 을 epoch {start_epoch} 부터 이어서 합니다 "
                      f"(이전 최고 {cfg.monitor} {res.best_score:.4f}, "
                      f"누적 {prior_sec / 60:.1f}분)")
        except Exception as exc:
            print(f"⚠️ [train] last.pt 를 읽지 못해 처음부터 시작합니다 — "
                  f"{type(exc).__name__}: {exc}")
            start_epoch, best, bad_epochs, prior_sec = 0, -1.0, 0, 0.0
            res = FitResult(cfg=cfg.to_dict())

    t0 = time.time()

    if verbose:
        print(f"\n{'=' * 60}\n 실험: {exp}  |  {cfg.model_name}  |  {device}")
        print(f" epochs={cfg.epochs} batch={cfg.resolved_batch_size()} "
              f"accum={cfg.grad_accum} lr={cfg.lr} amp={cfg.amp} ema={cfg.ema_decay > 0}")
        print(f" 조기종료 기준: val {cfg.monitor} (patience={cfg.early_stop_patience})")
        pd_ = persist_dir(exp) if persist else None
        if _same_dir(pd_, ck_dir):
            print(" 체크포인트가 이미 영속 디스크에 있습니다 — 별도 백업 없음")
        elif pd_ is not None:
            print(f" 중단 대비 백업: {pd_}  (매 {persist_every} 에폭)")
        elif persist:
            print(" ⚠️ 중단 대비 백업 없음 — 세션이 끊기면 체크포인트가 사라집니다.\n"
                  "    Colab 이면 env.mount_drive() 를 먼저 실행하세요.")
        print("=" * 60)

    if start_epoch >= cfg.epochs and verbose:
        print(f"이미 {start_epoch} 에폭까지 끝났습니다 (목표 {cfg.epochs}). 학습 생략.")

    early_stopped = False
    for epoch in range(start_epoch, cfg.epochs):
        tl = _train_epoch(model, dl_train, criterion, optimizer, scheduler,
                          scaler, cfg, ema, device, n_cls)
        eval_model = ema.ema if ema is not None else model
        vl, logits, ys = evaluate_loader(eval_model, dl_val, criterion, device, n_cls)
        m = quick_metrics(logits, ys)
        score = m.get(cfg.monitor, m["macro_f1"])

        row = {"epoch": epoch, "train_loss": round(tl, 5), "val_loss": round(vl, 5),
               "val_acc": round(m["acc"], 5), "val_macro_f1": round(m["macro_f1"], 5),
               "val_balanced_acc": round(m["balanced_acc"], 5),
               "lr": optimizer.param_groups[0]["lr"]}
        res.history.append(row)

        # 이어받기 후에도 헤더가 두 번 찍히지 않게 파일 존재 여부로 판단합니다.
        need_header = not log_path.exists() or log_path.stat().st_size == 0
        with log_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if need_header:
                w.writeheader()
            w.writerow(row)

        flag = ""
        if score > best:
            best, bad_epochs = score, 0
            res.best_score, res.best_epoch = score, epoch
            ckpt = ck_dir / "best.pt"
            torch.save({
                "model": model.state_dict(),
                "ema": ema.ema.state_dict() if ema else None,
                "epoch": epoch, "score": score, "cfg": cfg.to_dict(),
                "classes": getattr(dl_train.dataset, "classes", None),
            }, ckpt)
            res.best_ckpt = str(ckpt)
            flag = "  ★ best"
        else:
            bad_epochs += 1

        stop = bad_epochs >= cfg.early_stop_patience
        res.elapsed_sec = prior_sec + (time.time() - t0)

        # ── 중단 대비: 매 에폭 전체 상태를 남깁니다 ──────────────
        torch.save({
            "model": model.state_dict(),
            "ema": ema.ema.state_dict() if ema else None,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "rng": _rng_state(),
            "epoch": epoch, "best": best, "bad_epochs": bad_epochs,
            "best_score": res.best_score, "best_epoch": res.best_epoch,
            "history": res.history, "elapsed_sec": res.elapsed_sec,
            "cfg": cfg.to_dict(),
            "classes": getattr(dl_train.dataset, "classes", None),
        }, ck_dir / "last.pt")
        _write_result(ck_dir, res, completed=False, target_epochs=cfg.epochs)

        sync_sec = 0.0
        last_epoch = stop or epoch == cfg.epochs - 1
        if persist and (epoch % max(persist_every, 1) == 0 or last_epoch):
            sync_sec = sync_to_persist(exp)

        if verbose:
            extra = f" | 백업 {sync_sec:.0f}초" if sync_sec >= 1 else ""
            print(f"[{epoch:>2}/{cfg.epochs - 1}] train {tl:.4f} | val {vl:.4f} | "
                  f"acc {m['acc']:.4f} | macroF1 {m['macro_f1']:.4f} | "
                  f"balAcc {m['balanced_acc']:.4f}{flag}{extra}")

        if stop:
            early_stopped = True
            if verbose:
                print(f"\n조기 종료 — {cfg.early_stop_patience} 에폭 동안 개선 없음")
            break

    res.elapsed_sec = prior_sec + (time.time() - t0)
    _write_result(ck_dir, res, completed=True, target_epochs=cfg.epochs,
                  early_stopped=early_stopped)
    if persist:
        sync_to_persist(exp)

    # ⚠️ 여기까지 model 은 **마지막 에폭** 가중치입니다. 저장된 best.pt 와 다릅니다.
    #    노트북은 fit() 이 끝난 뒤 이 model 로 평가하므로, best 를 되돌려 놓습니다.
    if restore_best and res.best_ckpt and Path(res.best_ckpt).exists():
        _load_into(model, Path(res.best_ckpt), verbose=verbose)

    if verbose:
        res.summary()
    return res


def _write_result(ck_dir: Path, res: FitResult, completed: bool,
                  target_epochs: int = 0, early_stopped: bool = False) -> None:
    (ck_dir / "result.json").write_text(
        json.dumps({"best_score": res.best_score, "best_epoch": res.best_epoch,
                    "history": res.history, "elapsed_sec": res.elapsed_sec,
                    "completed": completed, "target_epochs": target_epochs,
                    "early_stopped": early_stopped, "cfg": res.cfg},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _load_into(model: nn.Module, ckpt_path: Path, use_ema: bool = True,
               verbose: bool = True) -> nn.Module:
    """체크포인트의 가중치를 **주어진 model 객체에** 그대로 얹습니다.

    EMA 를 썼다면 EMA 가중치가 best 로 저장된 것이므로 그쪽을 씁니다.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = (ck.get("ema") if use_ema else None) or ck.get("model") or ck
    missing, unexpected = model.load_state_dict(state, strict=False)
    if (missing or unexpected) and verbose:
        print(f"⚠️ [train] state_dict 불일치 — missing={len(missing)}, "
              f"unexpected={len(unexpected)}")
    which = "EMA" if (use_ema and ck.get("ema") is not None) else "raw"
    if verbose:
        print(f"[train] best 가중치({which}, epoch {ck.get('epoch', '?')}) 를 "
              f"model 에 되돌렸습니다.")
    return model


# ──────────────────────────────────────────────────────────────
# 평가 결과 캐시
# ──────────────────────────────────────────────────────────────
# 세션이 끊겨 노트북을 처음부터 다시 돌리면, 학습은 건너뛰어도 **검증 추론은
# 매번 다시** 합니다. 노트북 03 기준 전부 합쳐 7만 장 넘는 순전파 = 10분 이상.
# 모델도 데이터도 그대로인데 같은 숫자를 다시 계산하는 건 낭비입니다.
#
# 그래서 logits 를 파일로 남기고, **입력이 같을 때만** 재사용합니다.
# "같다" 의 판정(지문)에 들어가는 것:
#   · 체크포인트 파일의 크기·수정시각  → 모델이 바뀌면 무효
#   · 이미지 경로 목록의 해시·행 수     → 데이터·순서가 바뀌면 무효
#   · TTA 여부, 클래스 수              → 추론 방식이 바뀌면 무효
# 하나라도 다르면 조용히 다시 계산합니다. 낡은 숫자를 쓰는 사고를 막는 게 우선입니다.


def _fingerprint(dataset, tta: bool, n_cls: int, ckpt: Path | None) -> str:
    import hashlib

    h = hashlib.sha256()
    # ⚠️ 행 수는 **항상** 넣습니다. paths 같은 선택 속성에만 기대면, 그게 없는
    #    Dataset 에서 지문이 데이터를 아예 안 보게 되어 낡은 캐시를 재사용합니다.
    h.update(f"{len(dataset)}|{tta}|{n_cls}|".encode())
    for p in getattr(dataset, "paths", []) or []:
        h.update(str(p).encode())
        h.update(b"\0")
    t = getattr(dataset, "targets", None)
    if t is not None:
        h.update(np.asarray(t).tobytes())
    if ckpt is not None and ckpt.exists():
        st = ckpt.stat()
        h.update(f"|{st.st_size}|{int(st.st_mtime)}|{_file_digest(ckpt)}".encode())
    return h.hexdigest()[:32]


def _file_digest(p: Path, head: int = 1 << 20) -> str:
    """파일 앞 1MB 의 해시. mtime 만 보면 같은 초에 덮어쓴 체크포인트를 놓칩니다."""
    import hashlib

    try:
        with p.open("rb") as f:
            return hashlib.sha256(f.read(head)).hexdigest()[:16]
    except OSError:
        return ""


def cached_logits(model, loader, key: str, exp: str, n_cls: int,
                  device: str | None = None, tta_hflip: bool = False,
                  ckpt: str | Path | None = None,
                  use_cache: bool = True, verbose: bool = True):
    """`evaluate_loader` 의 캐시판. `(logits, y)` 를 돌려줍니다.

        lg, y = train.cached_logits(m1, dl_va1, key="val", exp=cfg1.exp_name,
                                    n_cls=len(CLASSES_STAGE1), tta_hflip=True)

    캐시가 유효하면 즉시, 아니면 계산 후 저장 + 영속 저장소로 백업합니다.
    강제로 다시 계산하려면 `use_cache=False`.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    d = ckpt_dir(exp)
    name = f"logits_{key}.npz"
    path = d / name
    ckpt_p = Path(ckpt) if ckpt else (d / "best.pt")
    fp = _fingerprint(loader.dataset, tta_hflip, n_cls, ckpt_p)

    if use_cache:
        if not path.exists():
            restore_from_persist(exp, verbose=False)
        if path.exists():
            try:
                z = np.load(path, allow_pickle=False)
                if str(z["fingerprint"]) == fp:
                    if verbose:
                        print(f"⏭️  [{exp}/{key}] 캐시 재사용 — {len(z['y']):,}행 "
                              f"(다시 계산 안 함)")
                    return torch.from_numpy(z["logits"]), torch.from_numpy(z["y"])
                if verbose:
                    print(f"[{exp}/{key}] 캐시가 현재 입력과 안 맞습니다 — 다시 계산합니다.")
            except Exception as exc:
                print(f"⚠️ [{exp}/{key}] 캐시를 읽지 못해 다시 계산합니다 — "
                      f"{type(exc).__name__}")

    t0 = time.time()
    _, logits, ys = evaluate_loader(model, loader, None, device, n_cls, tta_hflip=tta_hflip)
    dt = time.time() - t0
    try:
        np.savez_compressed(path, logits=logits.numpy(), y=ys.numpy(), fingerprint=fp)
        sync_to_persist(exp, files=(name,))
        if verbose:
            print(f"[{exp}/{key}] {len(ys):,}행 추론 {dt:.0f}초 → 캐시 저장 "
                  f"(다음 실행부터는 건너뜁니다)")
    except OSError as exc:
        print(f"⚠️ [{exp}/{key}] 캐시 저장 실패 — {type(exc).__name__}: {exc}")
    return logits, ys


def print_status(*exps: str) -> list[dict]:
    """실험들이 어디까지 갔는지 한눈에. 학습 셀을 돌리기 전에 확인용."""
    root = env.persist_root()
    if root is None:
        print("영속 저장소: ❌ 없음 — 세션이 끊기면 체크포인트가 사라집니다 "
              "(Colab 이면 env.mount_drive())")
    elif _same_dir(Path(root), Path(env.work_root())):
        print(f"영속 저장소: {root} (작업 폴더와 동일 — 휘발성 디스크가 아닙니다)")
    else:
        print(f"영속 저장소: {root}")
    rows = []
    for e in exps:
        st = training_state(e)
        rows.append(st)
        if st["completed"]:
            mark, note = "✅", f"완료 — 최고 {st['best_score']:.4f} (epoch {st['best_epoch']})"
        elif st["has_last"]:
            mark, note = "⏸️", f"{st['epochs_done']} 에폭까지 진행 — 이어서 합니다"
        else:
            mark, note = "🆕", "처음부터"
        print(f"  {mark} {e:<32} {note}")
    return rows


def load_best(result: FitResult, spec, n_classes: int, device: str = "cuda",
              use_ema: bool = True) -> nn.Module:
    from src.models import build

    ckpt = torch.load(result.best_ckpt, map_location="cpu", weights_only=False)
    state = (ckpt.get("ema") if use_ema else None) or ckpt["model"]
    model = build(spec, n_classes, pretrained=False, verbose=False)
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()
