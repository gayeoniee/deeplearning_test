"""학습이 느린 이유를 **숫자로** 가리는 진단기.

배치 크기를 16 → 32 로 올렸는데 에폭 시간이 그대로였습니다.
그럼 GPU 가 병목이 아닙니다. 그런데 어디가 병목인지는 추측하면 안 됩니다 —
고칠 곳을 잘못 짚으면 시간만 버립니다. 그래서 셋을 따로 잽니다:

    ① 이미지 1장 읽고 변환하는 데 걸리는 시간      (CPU, 1 프로세스)
    ② DataLoader 가 실제로 내주는 초당 장수         (CPU × 워커 수)
    ③ GPU 가 데이터를 안 기다릴 때의 초당 장수      (합성 텐서로 학습 스텝)

②와 ③ 중 작은 쪽이 실제 처리량입니다. 어느 쪽인지에 따라 처방이 다릅니다:

    ② < ③ 이면  → 입력 파이프라인 (워커·디코딩·디스크)
    ③ < ② 이면  → GPU (배치·해상도·모델 크기)

    from src import bench
    bench.report(df, CFG(model_name="resnet50", img_size=224), classes=CLASSES)

⚠️ **학습이 돌고 있는 동안 재지 마세요.** GPU/CPU 를 나눠 쓰게 되어 둘 다 틀립니다.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch

from src import env
from src.config import CFG, CLASSES


# ──────────────────────────────────────────────────────────────
# ① 한 장 읽는 비용
# ──────────────────────────────────────────────────────────────
def decode_speed(paths: list[str], cfg: CFG, n: int = 200) -> dict:
    """워커 없이, 한 프로세스에서 이미지를 읽고 변환만 합니다."""
    from src.data import build_transforms

    tf = build_transforms(cfg, train=True)
    use = paths[:n]
    if not use:
        raise ValueError("경로가 비었습니다.")

    from PIL import Image

    # torchvision 변환은 첫 호출에서 초기화 비용이 붙습니다 — 빼고 잽니다.
    for p in use[:5]:
        with Image.open(p) as im:
            tf(im.convert("RGB"))

    t_open = t_tf = 0.0
    sizes = []
    for p in use:
        t0 = time.perf_counter()
        with Image.open(p) as im:
            img = im.convert("RGB")
        t1 = time.perf_counter()
        tf(img)
        t2 = time.perf_counter()
        t_open += t1 - t0
        t_tf += t2 - t1
        sizes.append(img.size)

    k = len(use)
    w = float(np.mean([s[0] for s in sizes]))
    h = float(np.mean([s[1] for s in sizes]))
    return {"n": k,
            "decode_ms": 1000 * t_open / k,
            "transform_ms": 1000 * t_tf / k,
            "img_per_sec_1proc": k / (t_open + t_tf),
            "mean_src_px": (round(w), round(h))}


# ──────────────────────────────────────────────────────────────
# ② DataLoader 처리량
# ──────────────────────────────────────────────────────────────
def loader_speed(df, cfg: CFG, classes: list[str] | None = None,
                 batches: int = 30, workers: int | None = None) -> dict:
    """실제 DataLoader 로 몇 장/초가 나오는지. GPU 는 쓰지 않습니다."""
    from torch.utils.data import DataLoader

    from src.data import SkinDataset, build_transforms

    nw = cfg.resolved_num_workers() if workers is None else workers
    ds = SkinDataset(df, transform=build_transforms(cfg, train=True),
                     classes=classes or CLASSES)
    bs = cfg.resolved_batch_size()
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=nw,
                    pin_memory=torch.cuda.is_available(),
                    persistent_workers=nw > 0, drop_last=True)

    it = iter(dl)
    next(it)                              # 워커 시작 비용은 빼고 잽니다
    t0, seen = time.perf_counter(), 0
    for _ in range(batches):
        try:
            x, _y = next(it)
        except StopIteration:
            break
        seen += x.size(0)
    dt = time.perf_counter() - t0
    del it, dl
    return {"workers": nw, "batch": bs, "n": seen,
            "img_per_sec": seen / max(dt, 1e-9), "sec": dt}


# ──────────────────────────────────────────────────────────────
# ③ GPU 계산 처리량 (데이터를 안 기다릴 때)
# ──────────────────────────────────────────────────────────────
def gpu_speed(cfg: CFG, n_classes: int = 6, steps: int = 30) -> dict:
    """같은 배치를 반복해서 학습 스텝만 돕니다 — 디스크·CPU 를 완전히 배제."""
    from src.models import build, param_groups

    if not torch.cuda.is_available():
        return {"img_per_sec": float("nan"), "note": "GPU 없음"}

    dev = "cuda"
    # ⚠️ img_size 를 넘겨야 합니다. ViT 계열은 위치 임베딩·윈도우 크기가 고정이라
    #    안 넘기면 기본 해상도(예: swinv2 256)로 만들어놓고 cfg.img_size(384) 짜리
    #    텐서를 먹여 shape 오류로 죽습니다. 실제로 판 B(해상도 혼합)에서 시간 추정
    #    셀이 학습 시작 전에 터질 뻔했습니다.
    model = build(cfg.model_name, n_classes, pretrained=False, verbose=False,
                  img_size=cfg.img_size)
    model = model.to(dev).to(memory_format=torch.channels_last).train()
    opt = torch.optim.AdamW(param_groups(model, cfg))
    crit = torch.nn.CrossEntropyLoss()
    bs = cfg.resolved_batch_size()

    amp_dtype = torch.bfloat16 if env.device_info().bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and amp_dtype is torch.float16)

    def one():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=cfg.amp):
            loss = crit(model(x), y)
        if scaler.is_enabled():
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()

    # ⚠️ 추천 배치가 안 들어가면 **여기서 알아야 합니다.** 학습 3시간째에 OOM 으로
    #    죽는 것보다 20초짜리 추정에서 배치를 낮춰 잡는 게 낫습니다.
    #    실제로 판 B 에서 swinv2_base 가 배치 32 로 OOM 났습니다.
    x = y = None
    for attempt in range(4):
        try:
            x = torch.randn(bs, 3, cfg.img_size, cfg.img_size, device=dev
                            ).to(memory_format=torch.channels_last)
            y = torch.randint(0, n_classes, (bs,), device=dev)
            for _ in range(5):            # warmup (cudnn 알고리즘 선택 포함)
                one()
            break
        except torch.cuda.OutOfMemoryError:
            del x, y
            x = y = None
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if bs <= 4 or attempt == 3:
                return {"img_per_sec": float("nan"),
                        "note": f"배치 {bs} 로도 VRAM 부족"}
            bs = max(4, bs // 2)
            print(f"    [bench] VRAM 부족 → 배치를 {bs} 로 낮춰 다시 잽니다")
    # 실패한 시도의 최대치가 섞이지 않게 여기서 초기화합니다
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        one()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    peak = torch.cuda.max_memory_allocated() / 1024**3
    del model, opt, x, y
    torch.cuda.empty_cache()
    return {"batch": bs, "img_per_sec": bs * steps / dt, "sec": dt,
            "peak_vram_gb": round(peak, 2),
            "amp_dtype": "bf16" if amp_dtype is torch.bfloat16 else "fp16"}


# ──────────────────────────────────────────────────────────────
# 종합
# ──────────────────────────────────────────────────────────────
def report(df, cfg: CFG, classes: list[str] | None = None,
           n_classes: int | None = None, verbose: bool = True) -> dict:
    """①②③ 을 다 재고, 어디가 병목인지와 처방을 출력합니다."""
    paths = [p for p in df["crop_path"].dropna().astype(str).tolist()[:400]]
    d1 = decode_speed(paths, cfg)
    d2 = loader_speed(df, cfg, classes=classes)
    d3 = gpu_speed(cfg, n_classes=n_classes or len(classes or CLASSES))

    has_gpu = np.isfinite(d3["img_per_sec"])
    real = min(d2["img_per_sec"], d3["img_per_sec"]) if has_gpu else d2["img_per_sec"]
    n_train = len(df)
    per_epoch = n_train / max(real, 1e-9) / 60

    out = {"cpu_count": os.cpu_count(), "decode": d1, "loader": d2, "gpu": d3,
           "est_min_per_epoch": per_epoch}

    if not verbose:
        return out

    print(f"\n{'=' * 62}\n 처리량 진단  |  {cfg.model_name} @ {cfg.img_size}px\n{'=' * 62}")
    print(f"  CPU 코어 수                    {os.cpu_count()}")
    print(f"  DataLoader 워커 수             {d2['workers']}")
    print(f"  배치 크기                      {d2['batch']}")
    print()
    print(f"① 한 장 읽기 (1 프로세스)")
    print(f"     디코딩          {d1['decode_ms']:6.1f} ms   (원본 평균 {d1['mean_src_px'][0]}×{d1['mean_src_px'][1]}px)")
    print(f"     증강·변환       {d1['transform_ms']:6.1f} ms")
    print(f"     → {d1['img_per_sec_1proc']:6.0f} img/s  × 워커 {d2['workers']} = "
          f"이론상 {d1['img_per_sec_1proc'] * d2['workers']:.0f} img/s")
    print()
    print(f"② DataLoader 실측                {d2['img_per_sec']:6.0f} img/s")
    if has_gpu:
        print(f"③ GPU 만 (데이터 대기 없음)      {d3['img_per_sec']:6.0f} img/s"
              f"   [{d3.get('amp_dtype', '?')}, VRAM {d3.get('peak_vram_gb', '?')}GB]")
    else:
        print("③ GPU 만                         — GPU 가 없어 못 쟀습니다")
    print()
    print(f"  실제 처리량 ≈ {real:.0f} img/s"
          f"  → {n_train:,}장 기준 에폭당 약 {per_epoch:.1f}분")
    print()

    if not has_gpu:
        print("⚠️ GPU 가 없어 병목 판정을 못 합니다. Colab GPU 런타임에서 다시 돌리세요.")
        print("=" * 62)
        return out

    slower = "loader" if d2["img_per_sec"] < d3["img_per_sec"] else "gpu"
    ratio = max(d2["img_per_sec"], d3["img_per_sec"]) / max(
        min(d2["img_per_sec"], d3["img_per_sec"]), 1e-9)

    if slower == "loader":
        print(f"🚨 병목은 **입력 파이프라인** 입니다 (GPU 가 {ratio:.1f}배 더 빠름).")
        print("   GPU 는 데이터를 기다리며 놀고 있습니다. 배치나 해상도를 건드려도 안 빨라집니다.")
        if d1["decode_ms"] > d1["transform_ms"]:
            print(f"   비용의 대부분이 **JPEG 디코딩**({d1['decode_ms']:.1f}ms) 입니다.")
            print("   → 처방 1: 크롭을 학습 해상도에 맞게 더 작게 저장 "
                  f"(지금 원본 평균 {d1['mean_src_px'][0]}px → {cfg.img_size}px 근처로)")
            print("             로컬에서 `prepare_local.py --save-crop-size` 로 다시 패키징")
            print("   → 처방 2: Colab 런타임을 CPU 코어가 더 많은 것으로 (유료 티어)")
        else:
            print(f"   비용의 대부분이 **증강·변환**({d1['transform_ms']:.1f}ms) 입니다.")
            print("   → 처방: 증강 단계를 줄이거나 img_size 를 낮추세요.")
        if os.cpu_count() and os.cpu_count() <= 2:
            print(f"   ⚠️ CPU 코어가 {os.cpu_count()}개뿐입니다 — 워커를 늘려도 한계입니다.")
    else:
        print(f"✅ 병목은 **GPU** 입니다 (입력이 {ratio:.1f}배 더 빠름).")
        print("   입력 파이프라인은 충분합니다. 여기서 더 빠르게 하려면")
        print("   모델을 줄이거나 해상도를 낮추는 것 말고는 없습니다.")
        print("   반대로 말하면 **해상도를 올린 만큼 정직하게 느려집니다.**")
    print("=" * 62)
    return out
