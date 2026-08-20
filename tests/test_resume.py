"""중단·재개 테스트 — 세션이 끊겨도 학습을 이어갈 수 있는가.

Colab 세션은 예고 없이 끊깁니다. 25에폭 중 23에폭에서 끊겼을 때 처음부터
다시 하지 않으려면, 체크포인트가 (1) 세션 밖에 남아 있고 (2) 옵티마이저·
스케줄러 상태까지 복원되어야 합니다. 그 두 가지를 여기서 검증합니다.

    python tests/test_resume.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ⚠️ src.env 는 호출 시점에 환경변수를 읽으므로, import 전에 잡아둡니다.
_TMP = Path(tempfile.mkdtemp(prefix="dogskin_resume_"))
os.environ["DOG_SKIN_WORK"] = str(_TMP / "work")
os.environ["DOG_SKIN_PERSIST"] = str(_TMP / "persist")

import torch                                                    # noqa: E402
import torch.nn as nn                                           # noqa: E402
from torch.utils.data import DataLoader, Dataset                 # noqa: E402

from src import env, train                                      # noqa: E402
from src.config import CFG                                       # noqa: E402

PASS, FAIL = [], []


def check(name: str, cond: bool, msg: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"{'✅' if cond else '❌'} {name}" + (f"\n     {msg}" if msg and not cond else ""))


# ──────────────────────────────────────────────────────────────
# 아주 작은 학습 재료 (이미지 없이 텐서로)
# ──────────────────────────────────────────────────────────────
class TinyDS(Dataset):
    classes = ["c0", "c1", "c2"]

    def __init__(self, n: int = 48, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.targets = [i % 3 for i in range(n)]
        # 클래스마다 다른 평균 → 학습이 실제로 진행됩니다
        self.x = torch.stack([
            torch.randn(3, 8, 8, generator=g) + float(t) for t in self.targets
        ])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.x[i], self.targets[i]


class TinyNet(nn.Module):
    """param_groups 가 'head' 를 찾을 수 있도록 이름을 맞춥니다."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 4, 3, padding=1)
        self.head = nn.Linear(4 * 8 * 8, 3)

    def forward(self, x):
        return self.head(torch.relu(self.stem(x)).flatten(1))


def loaders():
    ds = TinyDS()
    dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0)
    dl_va = DataLoader(TinyDS(n=24, seed=1), batch_size=16, num_workers=0)
    return dl, dl_va, ds


def tiny_cfg(exp: str, epochs: int) -> CFG:
    return CFG(model_name="tiny", img_size=8, epochs=epochs, batch_size=16,
               num_workers=0, warmup_epochs=1, amp=False, ema_decay=0.9,
               mixup_alpha=0.0, cutmix_alpha=0.0, balance_strategy="none",
               monitor="macro_f1", early_stop_patience=99, exp_name=exp)


def fit_once(exp: str, epochs: int, **kw):
    dl, dl_va, ds = loaders()
    m = TinyNet()
    res = train.fit(m, dl, dl_va, tiny_cfg(exp, epochs), ds_train=ds,
                    device="cpu", verbose=False, **kw)
    return res, m


# ──────────────────────────────────────────────────────────────
# 1. 영속 저장소
# ──────────────────────────────────────────────────────────────
def test_persist_root_override():
    p = env.persist_root()
    check("persist_root 가 DOG_SKIN_PERSIST 를 존중한다",
          p is not None and str(p) == os.environ["DOG_SKIN_PERSIST"], f"{p}")
    check("persist_root 는 work_root 와 다른 곳이다",
          p != env.work_root(), "같으면 세션이 끊길 때 함께 사라집니다")


def test_sync_restore_roundtrip():
    exp = "t_sync"
    d = train.ckpt_dir(exp)
    (d / "history.csv").write_text("epoch\n0\n", encoding="utf-8")
    (d / "best.pt").write_bytes(b"\x00" * 32)
    train.sync_to_persist(exp)

    pd_ = train.persist_dir(exp)
    check("sync_to_persist 가 파일을 영속 저장소로 복사한다",
          pd_ is not None and (pd_ / "history.csv").exists() and (pd_ / "best.pt").exists())
    check("임시 파일(.tmp)을 남기지 않는다",
          pd_ is not None and not list(pd_.glob("*.tmp")))

    # 세션이 죽은 상황: 로컬만 날아갑니다
    shutil.rmtree(d)
    check("복원 전에는 로컬이 비어 있다", not (train.ckpt_dir(exp) / "best.pt").exists())
    ok = train.restore_from_persist(exp, verbose=False)
    check("restore_from_persist 가 되돌린다",
          ok and (train.ckpt_dir(exp) / "best.pt").exists())


def test_sync_without_persist_is_noop():
    saved = os.environ.pop("DOG_SKIN_PERSIST")
    try:
        # 로컬 환경에서는 work_root 가 이미 영속이라 persist_root 가 그걸 돌려줍니다.
        # 그래도 크래시 없이 동작해야 합니다.
        train.sync_to_persist("t_noop")
        train.restore_from_persist("t_noop", verbose=False)
        check("영속 저장소가 지정되지 않아도 죽지 않는다", True)
    except Exception as exc:                                    # noqa: BLE001
        check("영속 저장소가 지정되지 않아도 죽지 않는다", False, repr(exc))
    finally:
        os.environ["DOG_SKIN_PERSIST"] = saved


# ──────────────────────────────────────────────────────────────
# 2. 상태 조회
# ──────────────────────────────────────────────────────────────
def test_state_fresh():
    st = train.training_state("t_never_run")
    check("처음 보는 실험은 completed=False, epochs_done=0",
          st["completed"] is False and st["epochs_done"] == 0, f"{st}")


def test_state_after_run():
    exp = "t_state"
    fit_once(exp, 2)
    st = train.training_state(exp)
    check("끝난 학습은 completed=True", st["completed"] is True, f"{st}")
    check("끝난 학습은 epochs_done=2", st["epochs_done"] == 2, f"{st}")
    check("best.pt / last.pt 가 둘 다 있다", st["has_best"] and st["has_last"])


# ──────────────────────────────────────────────────────────────
# 3. 재개 ★ 핵심
# ──────────────────────────────────────────────────────────────
def test_resume_continues_from_next_epoch():
    exp = "t_resume"
    res_a, _ = fit_once(exp, 2)
    check("1차 학습이 2에폭 돌았다", len(res_a.history) == 2, f"{len(res_a.history)}")

    # 목표 에폭을 늘려 다시 부릅니다 = "끊긴 학습 이어받기"
    res_b, _ = fit_once(exp, 5)
    check("이어받기: resumed_from == 2", res_b.resumed_from == 2, f"{res_b.resumed_from}")
    check("이어받기: 히스토리가 누적된다 (5행)", len(res_b.history) == 5,
          f"{[r['epoch'] for r in res_b.history]}")
    check("이어받기: 에폭 번호가 0..4 로 이어진다",
          [r["epoch"] for r in res_b.history] == [0, 1, 2, 3, 4])
    check("이어받기: 재학습이 아니다 (skipped=False)", res_b.skipped is False)


def test_resume_csv_has_one_header():
    exp = "t_csv"
    fit_once(exp, 2)
    fit_once(exp, 4)
    lines = (train.ckpt_dir(exp) / "history.csv").read_text(encoding="utf-8").strip().splitlines()
    heads = [ln for ln in lines if ln.startswith("epoch,")]
    check("history.csv 헤더가 한 번만 찍힌다", len(heads) == 1, f"{heads}")
    check("history.csv 에 4에폭이 다 들어간다", len(lines) == 5, f"{len(lines)}행")


def test_completed_run_is_skipped():
    exp = "t_skip"
    res_a, _ = fit_once(exp, 2)
    res_b, m_b = fit_once(exp, 2)
    check("이미 끝난 학습은 건너뛴다 (skipped=True)", res_b.skipped is True)
    check("건너뛰어도 점수가 그대로다",
          abs(res_b.best_score - res_a.best_score) < 1e-9,
          f"{res_a.best_score} vs {res_b.best_score}")
    check("건너뛰어도 model 에 best 가중치가 얹힌다",
          _same_as_best(m_b, train.ckpt_dir(exp) / "best.pt"))


def test_resume_false_starts_over():
    exp = "t_scratch"
    fit_once(exp, 3)
    res, _ = fit_once(exp, 2, resume=False)
    check("resume=False 는 처음부터 (resumed_from=0)", res.resumed_from == 0)
    check("resume=False 는 히스토리를 새로 쓴다", len(res.history) == 2,
          f"{[r['epoch'] for r in res.history]}")


def test_resume_restores_optimizer_state():
    """옵티마이저 모멘텀까지 살아있는가 — 이게 없으면 이어받아도 학습이 흔들립니다."""
    exp = "t_optim"
    fit_once(exp, 2)
    ck = torch.load(train.ckpt_dir(exp) / "last.pt", map_location="cpu", weights_only=False)
    check("last.pt 에 옵티마이저 상태가 있다", "optimizer" in ck and bool(ck["optimizer"]))
    check("last.pt 에 스케줄러 상태가 있다", ck.get("scheduler") is not None)
    check("last.pt 에 난수 상태가 있다", "rng" in ck and "torch" in ck["rng"])
    exp_avg = [s.get("exp_avg") for s in ck["optimizer"]["state"].values()]
    check("AdamW 모멘텀(exp_avg)이 실제로 채워져 있다",
          bool(exp_avg) and all(t is not None and float(t.abs().sum()) > 0 for t in exp_avg))


# ──────────────────────────────────────────────────────────────
# 4. best 가중치 되돌리기 (원래 있던 불일치)
# ──────────────────────────────────────────────────────────────
def _same_as_best(model: nn.Module, ckpt: Path) -> bool:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = ck.get("ema") or ck["model"]
    for k, v in model.state_dict().items():
        if k not in state:
            return False
        if not torch.allclose(v.cpu().float(), state[k].cpu().float(), atol=1e-6):
            return False
    return True


def test_restore_best_puts_best_weights_in_model():
    exp = "t_best"
    # 3에폭 중 마지막이 최고가 아니게 만들기 어렵기 때문에, 저장된 best 와
    # 반환된 model 이 **같은지**만 봅니다. 같으면 평가 대상이 일치합니다.
    _, m = fit_once(exp, 3)
    check("fit 이 돌려준 model == 저장된 best.pt",
          _same_as_best(m, train.ckpt_dir(exp) / "best.pt"),
          "이게 다르면 노트북이 best 가 아닌 마지막 에폭을 평가합니다")


def test_restore_best_off():
    exp = "t_nobest"
    _, m = fit_once(exp, 2, restore_best=False)
    ck = torch.load(train.ckpt_dir(exp) / "last.pt", map_location="cpu", weights_only=False)
    same_last = all(torch.allclose(v.cpu().float(), ck["model"][k].cpu().float(), atol=1e-6)
                    for k, v in m.state_dict().items())
    check("restore_best=False 면 마지막 에폭 가중치를 유지한다", same_last)


# ──────────────────────────────────────────────────────────────
# 5. 세션 종료 시뮬레이션 (로컬 디스크만 날림)
# ──────────────────────────────────────────────────────────────
def test_survives_wiped_local_disk():
    exp = "t_crash"
    dl, dl_va, ds = loaders()
    m = TinyNet()
    # 2에폭까지 진행 → 여기서 "세션이 끊겼다"
    train.fit(m, dl, dl_va, tiny_cfg(exp, 6), ds_train=ds, device="cpu",
              verbose=False, resume=False)
    d = train.ckpt_dir(exp)
    # result.json 의 completed 를 지워 '중간에 끊긴' 상태로 만듭니다
    r = json.loads((d / "result.json").read_text(encoding="utf-8"))
    r["completed"] = False
    r["history"] = r["history"][:2]
    (d / "result.json").write_text(json.dumps(r), encoding="utf-8")
    ck = torch.load(d / "last.pt", map_location="cpu", weights_only=False)
    ck["epoch"] = 1
    ck["history"] = ck["history"][:2]
    torch.save(ck, d / "last.pt")
    train.sync_to_persist(exp)

    shutil.rmtree(d)                     # ← /content 가 날아간 상황
    check("세션 종료 후 로컬 체크포인트가 없다", not (train.ckpt_dir(exp) / "last.pt").exists())

    res, _ = fit_once(exp, 6)
    check("드라이브에서 되살려 epoch 2 부터 이어간다", res.resumed_from == 2,
          f"{res.resumed_from}")
    check("최종적으로 6에폭이 채워진다", len(res.history) == 6,
          f"{[r['epoch'] for r in res.history]}")


def test_corrupt_last_pt_falls_back():
    exp = "t_corrupt"
    fit_once(exp, 2)
    d = train.ckpt_dir(exp)
    (d / "last.pt").write_bytes(b"not a checkpoint")
    (d / "result.json").write_text(json.dumps({"completed": False}), encoding="utf-8")
    try:
        res, _ = fit_once(exp, 2)
        check("깨진 last.pt 는 처음부터 다시 (죽지 않음)",
              res.resumed_from == 0 and len(res.history) == 2,
              f"resumed_from={res.resumed_from}, history={len(res.history)}")
    except Exception as exc:                                    # noqa: BLE001
        check("깨진 last.pt 는 처음부터 다시 (죽지 않음)", False, repr(exc))


def test_early_stopped_is_not_extended():
    """조기 종료로 끝난 학습은 에폭을 늘려도 이어가지 않습니다.

    patience 를 넘겨 멈춘 건 "더 돌려도 안 좋아진다" 는 결론이므로,
    노트북을 다시 돌릴 때마다 같은 학습을 반복하면 안 됩니다.
    """
    exp = "t_early"
    fit_once(exp, 2)
    d = train.ckpt_dir(exp)
    r = json.loads((d / "result.json").read_text(encoding="utf-8"))
    r["early_stopped"] = True
    (d / "result.json").write_text(json.dumps(r), encoding="utf-8")
    train.sync_to_persist(exp)

    res, _ = fit_once(exp, 10)              # 에폭을 크게 늘려도
    check("조기 종료된 학습은 연장하지 않는다", res.skipped is True,
          f"skipped={res.skipped}, history={len(res.history)}")


def test_extend_records_new_target():
    exp = "t_extend"
    fit_once(exp, 2)
    fit_once(exp, 4)
    st = train.training_state(exp)
    check("연장 후 target_epochs 가 갱신된다", st["target_epochs"] == 4, f"{st}")
    res, _ = fit_once(exp, 4)
    check("연장이 끝나면 그 다음부터는 건너뛴다", res.skipped is True)


# ──────────────────────────────────────────────────────────────
# 6. 평가 캐시 — 낡은 값을 쓰면 캐시가 없느니만 못합니다
# ──────────────────────────────────────────────────────────────
def _val_loader():
    return DataLoader(TinyDS(n=24, seed=1), batch_size=8, num_workers=0)


def test_cache_hit_and_reuse():
    exp = "t_cache"
    _, m = fit_once(exp, 2)
    dl = _val_loader()
    lg1, y1 = train.cached_logits(m, dl, key="val", exp=exp, n_cls=3,
                                  device="cpu", verbose=False)
    p = train.ckpt_dir(exp) / "logits_val.npz"
    check("캐시 파일이 만들어진다", p.exists())

    lg2, y2 = train.cached_logits(m, dl, key="val", exp=exp, n_cls=3,
                                  device="cpu", verbose=False)
    check("두 번째 호출이 같은 값을 준다",
          torch.allclose(lg1, lg2) and torch.equal(y1, y2))


def test_cache_invalidated_by_new_weights():
    """모델이 바뀌면 캐시를 버려야 합니다. 이게 안 되면 옛 점수를 보게 됩니다."""
    exp = "t_cache_w"
    _, m = fit_once(exp, 2)
    dl = _val_loader()
    lg1, _ = train.cached_logits(m, dl, key="val", exp=exp, n_cls=3,
                                 device="cpu", verbose=False)

    # best.pt 를 다른 가중치로 덮어씁니다 (= 재학습 후 상황)
    m2 = TinyNet()
    with torch.no_grad():
        for p in m2.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    ck = train.ckpt_dir(exp) / "best.pt"
    saved = torch.load(ck, map_location="cpu", weights_only=False)
    saved["model"] = m2.state_dict()
    saved["ema"] = m2.state_dict()
    torch.save(saved, ck)

    lg2, _ = train.cached_logits(m2, dl, key="val", exp=exp, n_cls=3,
                                 device="cpu", verbose=False)
    check("체크포인트가 바뀌면 다시 계산한다",
          not torch.allclose(lg1, lg2),
          "같은 값이 나왔습니다 = 낡은 캐시를 재사용했습니다")


def test_cache_invalidated_by_different_rows():
    exp = "t_cache_r"
    _, m = fit_once(exp, 2)
    lg_a, y_a = train.cached_logits(m, _val_loader(), key="v", exp=exp, n_cls=3,
                                    device="cpu", verbose=False)
    other = DataLoader(TinyDS(n=16, seed=7), batch_size=8, num_workers=0)
    lg_b, y_b = train.cached_logits(m, other, key="v", exp=exp, n_cls=3,
                                    device="cpu", verbose=False)
    check("행 수가 다르면 다시 계산한다", len(y_b) == 16 and len(y_a) == 24,
          f"{len(y_a)} vs {len(y_b)}")


def test_cache_invalidated_by_tta():
    exp = "t_cache_t"
    _, m = fit_once(exp, 2)
    dl = _val_loader()
    a, _ = train.cached_logits(m, dl, key="v", exp=exp, n_cls=3, device="cpu",
                               tta_hflip=False, verbose=False)
    b, _ = train.cached_logits(m, dl, key="v", exp=exp, n_cls=3, device="cpu",
                               tta_hflip=True, verbose=False)
    check("TTA 여부가 바뀌면 다시 계산한다", not torch.allclose(a, b))


def test_cache_survives_wiped_local_disk():
    exp = "t_cache_p"
    _, m = fit_once(exp, 2)
    dl = _val_loader()
    lg1, _ = train.cached_logits(m, dl, key="val", exp=exp, n_cls=3,
                                 device="cpu", verbose=False)
    pd_ = train.persist_dir(exp)
    check("캐시가 영속 저장소로 백업된다",
          pd_ is not None and (pd_ / "logits_val.npz").exists())

    shutil.rmtree(train.ckpt_dir(exp))          # 세션 종료
    train.restore_from_persist(exp, verbose=False)
    check("복원 시 logits 캐시도 같이 돌아온다",
          (train.ckpt_dir(exp) / "logits_val.npz").exists())


def test_cache_off():
    exp = "t_cache_off"
    _, m = fit_once(exp, 2)
    dl = _val_loader()
    train.cached_logits(m, dl, key="v", exp=exp, n_cls=3, device="cpu", verbose=False)
    p = train.ckpt_dir(exp) / "logits_v.npz"
    p.write_bytes(b"garbage")                    # 깨진 캐시
    try:
        lg, y = train.cached_logits(m, dl, key="v", exp=exp, n_cls=3,
                                    device="cpu", verbose=False)
        check("깨진 캐시는 무시하고 다시 계산한다", len(y) == 24)
    except Exception as exc:                                    # noqa: BLE001
        check("깨진 캐시는 무시하고 다시 계산한다", False, repr(exc))


# ──────────────────────────────────────────────────────────────
# 7. 디코딩 draft — 검증 변환에서는 결과가 사실상 같아야 합니다
# ──────────────────────────────────────────────────────────────
def test_draft_matches_full_decode_for_eval():
    """draft 로 절반 크기 디코딩해도 검증 점수가 바뀌면 안 됩니다."""
    import numpy as np
    import pandas as pd
    from PIL import Image

    from src.config import CFG as _CFG
    from src.data import SkinDataset, build_transforms

    d = _TMP / "draftimgs"
    d.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(6):
        # 부드러운 그라디언트 + 약한 노이즈 = 실제 사진과 비슷한 주파수 특성
        yy, xx = np.mgrid[0:512, 0:512]
        base = ((xx + yy) / 4 % 255).astype(np.uint8)
        arr = np.stack([base, np.roll(base, 40, 0), np.roll(base, 80, 1)], -1)
        arr = np.clip(arr.astype(int) + rng.integers(-6, 6, arr.shape), 0, 255).astype(np.uint8)
        p = d / f"g{i}.jpg"
        Image.fromarray(arr).save(p, quality=92)
        rows.append({"crop_path": str(p), "label": "A1"})
    df = pd.DataFrame(rows)

    cfg = _CFG(model_name="tiny", img_size=224)
    tf = build_transforms(cfg, train=False)
    full = SkinDataset(df, tf, classes=["A1"])
    drafted = SkinDataset(df, tf, classes=["A1"], draft_size=int(224 * 1.14))

    diffs = [float((full[i][0] - drafted[i][0]).abs().mean()) for i in range(len(df))]
    worst = max(diffs)
    # 정규화된 텐서 기준. 0.05 는 8비트로 약 1.3 수준 — 모델 판정에 영향 없는 크기입니다.
    check("draft 디코딩이 검증 이미지를 거의 안 바꾼다", worst < 0.05,
          f"평균절대차 최대 {worst:.4f}")

    same_shape = all(full[i][0].shape == drafted[i][0].shape for i in range(len(df)))
    check("draft 후에도 텐서 크기가 같다", same_shape)


def test_print_status_runs():
    try:
        rows = train.print_status("t_resume", "t_never_run")
        check("print_status 가 실험 상태를 돌려준다", len(rows) == 2)
    except Exception as exc:                                    # noqa: BLE001
        check("print_status 가 실험 상태를 돌려준다", False, repr(exc))


if __name__ == "__main__":
    print(f"작업 폴더: {_TMP}\n")
    torch.manual_seed(0)
    for fn in [test_persist_root_override, test_sync_restore_roundtrip,
               test_sync_without_persist_is_noop,
               test_state_fresh, test_state_after_run,
               test_resume_continues_from_next_epoch, test_resume_csv_has_one_header,
               test_completed_run_is_skipped, test_resume_false_starts_over,
               test_resume_restores_optimizer_state,
               test_restore_best_puts_best_weights_in_model, test_restore_best_off,
               test_survives_wiped_local_disk, test_corrupt_last_pt_falls_back,
               test_early_stopped_is_not_extended, test_extend_records_new_target,
               test_cache_hit_and_reuse, test_cache_invalidated_by_new_weights,
               test_cache_invalidated_by_different_rows, test_cache_invalidated_by_tta,
               test_cache_survives_wiped_local_disk, test_cache_off,
               test_draft_matches_full_decode_for_eval,
               test_print_status_runs]:
        print(f"\n── {fn.__name__} ──")
        fn()

    print(f"\n{'=' * 60}\n통과 {len(PASS)} / {len(PASS) + len(FAIL)}")
    if FAIL:
        print("실패:", ", ".join(FAIL))
    shutil.rmtree(_TMP, ignore_errors=True)
    sys.exit(1 if FAIL else 0)
