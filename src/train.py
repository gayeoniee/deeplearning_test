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
) -> FitResult:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    exp = exp_name or cfg.exp_name
    n_cls = getattr(dl_train.dataset, "classes", None)
    n_cls = len(n_cls) if n_cls else int(max(dl_train.dataset.targets)) + 1

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

    ck_dir = env.ensure_dirs()["checkpoints"] / exp
    ck_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(ck_dir / "config.json")
    log_path = ck_dir / "history.csv"

    res = FitResult(cfg=cfg.to_dict())
    best, bad_epochs = -1.0, 0
    t0 = time.time()

    if verbose:
        print(f"\n{'=' * 60}\n 실험: {exp}  |  {cfg.model_name}  |  {device}")
        print(f" epochs={cfg.epochs} batch={cfg.resolved_batch_size()} "
              f"accum={cfg.grad_accum} lr={cfg.lr} amp={cfg.amp} ema={cfg.ema_decay > 0}")
        print(f" 조기종료 기준: val {cfg.monitor} (patience={cfg.early_stop_patience})\n{'=' * 60}")

    for epoch in range(cfg.epochs):
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

        with log_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if epoch == 0:
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

        if verbose:
            print(f"[{epoch:>2}/{cfg.epochs - 1}] train {tl:.4f} | val {vl:.4f} | "
                  f"acc {m['acc']:.4f} | macroF1 {m['macro_f1']:.4f} | "
                  f"balAcc {m['balanced_acc']:.4f}{flag}")

        if bad_epochs >= cfg.early_stop_patience:
            if verbose:
                print(f"\n조기 종료 — {cfg.early_stop_patience} 에폭 동안 개선 없음")
            break

    res.elapsed_sec = time.time() - t0
    (ck_dir / "result.json").write_text(
        json.dumps({"best_score": res.best_score, "best_epoch": res.best_epoch,
                    "history": res.history, "elapsed_sec": res.elapsed_sec},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if verbose:
        res.summary()
    return res


def load_best(result: FitResult, spec, n_classes: int, device: str = "cuda",
              use_ema: bool = True) -> nn.Module:
    from src.models import build

    ckpt = torch.load(result.best_ckpt, map_location="cpu", weights_only=False)
    state = (ckpt.get("ema") if use_ema else None) or ckpt["model"]
    model = build(spec, n_classes, pretrained=False, verbose=False)
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()
