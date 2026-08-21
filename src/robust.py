"""실사용 견고성 검사 — 검증 점수가 실제로 버틸 숫자인가.

검증셋 점수가 높아도 실사용에서 무너지는 두 가지 이유가 있습니다.
둘 다 **우리가 크롭을 만든 방식** 때문에 생깁니다:

  ① 배율   크롭은 병변 박스에 맞춰 잘라서 배율이 클래스마다 다릅니다.
            (실측: A1 0.47% ~ A6 3.08%, 6.5배)
            보호자 사진의 배율은 우리가 모릅니다. 모델이 배율에 의존했으면 무너집니다.

  ② 위치   크롭은 병변을 항상 화면 정중앙에 둡니다.
            보호자 사진에서 병변이 정중앙에 올 이유가 없습니다.

이 모듈은 검증셋을 일부러 그렇게 망가뜨려서 점수 하락폭을 잽니다.
**하락폭이 곧 실사용 위험도**입니다. 학습은 하지 않으니 몇 분이면 됩니다.

    from src import robust
    robust.scale_stress(model, va_df, cfg, CLASSES)     # 배율 교란
    robust.shift_stress(model, va_df, cfg, CLASSES)     # 위치 교란
"""

from __future__ import annotations

import numpy as np
import torch

from src.config import CFG


# ──────────────────────────────────────────────────────────────
# 교란 변환
# ──────────────────────────────────────────────────────────────
class ZoomView:
    """같은 사진을 z 배 확대/축소해서 본 것처럼 만듭니다.

    z = 2.0  → 절반 영역만 크게 (가까이서 찍은 셈)
    z = 1.0  → 평소 검증과 동일
    z = 0.5  → 두 배 넓게 (멀리서 찍은 셈)

    z < 1 일 때 크롭 밖의 픽셀이 필요해지는데 우리에겐 없습니다.
    그래서 **반사 패딩**으로 채웁니다 — 인공적이지만 모든 클래스에 똑같이
    적용되므로 클래스 간 비교는 공정합니다.
    """

    def __init__(self, img_size: int, z: float, pad_mode: str = "reflect"):
        self.img_size = img_size
        self.z = float(z)
        self.pad_mode = pad_mode

    def __call__(self, pil):
        from PIL import Image
        from torchvision.transforms import functional as F

        base = max(int(round(self.img_size * 1.14 * self.z)), 8)
        im = F.resize(pil, base, interpolation=Image.BICUBIC)

        w, h = im.size
        need = self.img_size
        if min(w, h) < need:
            # 반사 패딩은 이미지 크기보다 큰 패딩을 못 하므로 여러 번 나눠 넣습니다
            while min(im.size) < need:
                w, h = im.size
                px = min(max((need - w + 1) // 2, 0), w - 1) if w < need else 0
                py = min(max((need - h + 1) // 2, 0), h - 1) if h < need else 0
                if px == 0 and py == 0:
                    break
                im = F.pad(im, [px, py, px, py], padding_mode=self.pad_mode)
        return F.center_crop(im, [need, need])


class ShiftView:
    """병변을 화면 중앙에서 밀어냅니다 (프레임 폭의 frac 만큼).

    크롭은 병변을 항상 정중앙에 둡니다. 실제 사진은 그렇지 않으니,
    중앙에만 반응하는 모델인지 확인합니다.
    """

    def __init__(self, img_size: int, frac: float, direction: str = "diag",
                 pad_mode: str = "reflect"):
        self.img_size = img_size
        self.frac = float(frac)
        self.direction = direction
        self.pad_mode = pad_mode

    def __call__(self, pil):
        from PIL import Image
        from torchvision.transforms import functional as F

        need = self.img_size
        im = F.resize(pil, int(round(need * 1.14)), interpolation=Image.BICUBIC)
        im = F.center_crop(im, [need, need])

        dx = int(round(need * self.frac))
        dy = dx if self.direction == "diag" else 0
        if self.direction == "x":
            dy = 0
        elif self.direction == "y":
            dx, dy = 0, dx
        if dx == 0 and dy == 0:
            return im

        pad = [min(abs(dx), need - 1), min(abs(dy), need - 1)] * 2
        im = F.pad(im, pad, padding_mode=self.pad_mode)
        # 패딩된 이미지에서 중앙을 dx,dy 만큼 옮긴 위치를 잘라냅니다
        left = pad[0] + dx
        top = pad[1] + dy
        return F.crop(im, top, left, need, need)


# ──────────────────────────────────────────────────────────────
# 실행
# ──────────────────────────────────────────────────────────────
def _mean_std(model):
    from src.data import IMAGENET_MEAN, IMAGENET_STD

    d = getattr(model, "pretrained_cfg", None) or {}
    if not isinstance(d, dict):
        d = {}
    return tuple(d.get("mean", IMAGENET_MEAN)), tuple(d.get("std", IMAGENET_STD))


@torch.no_grad()
def _score(model, df, view, cfg: CFG, classes: list[str], device: str,
           path_col: str) -> tuple[float, float]:
    """(macro_f1, balanced_acc) — 주어진 교란 변환으로 평가합니다."""
    from sklearn.metrics import balanced_accuracy_score, f1_score
    from torch.utils.data import DataLoader
    from torchvision import transforms as T

    from src.data import SkinDataset

    mean, std = _mean_std(model)
    tf = T.Compose([view, T.ToTensor(), T.Normalize(mean, std)])
    ds = SkinDataset(df, tf, path_col, classes=classes)
    dl = DataLoader(ds, batch_size=max(cfg.resolved_batch_size() * 2, 1), shuffle=False,
                    num_workers=cfg.resolved_num_workers(), pin_memory=torch.cuda.is_available())

    model = model.to(device).eval()
    preds, ys = [], []
    for x, y in dl:
        out = model(x.to(device, non_blocking=True)).float().cpu()
        preds.append(out.argmax(1))
        ys.append(y)
    p = torch.cat(preds).numpy()
    t = torch.cat(ys).numpy()
    keep = t >= 0
    if keep.sum() == 0:
        return float("nan"), float("nan")
    return (float(f1_score(t[keep], p[keep], average="macro", zero_division=0)),
            float(balanced_accuracy_score(t[keep], p[keep])))


def _zoom_name(z: float) -> str:
    return f"{'원본' if z == 1.0 else '배율'}({z:g}x)"


def _shift_name(f: float) -> str:
    return f"{'중앙' if f == 0 else '이동'}({f:.0%})"


def _run(model, df, cfg, classes, views: list[tuple[str, object]], n: int, seed: int,
         path_col: str, device: str | None, title: str, hint: str,
         baseline_name: str | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sub = df.dropna(subset=[path_col])
    if n and len(sub) > n:
        sub = sub.sample(n, random_state=seed)

    print("=" * 68)
    print(f" {title}")
    print("=" * 68)
    print(f"  표본 {len(sub):,}장  |  {len(views)}가지 조건")

    rows, out = [], {}
    for name, view in views:
        f1, bacc = _score(model, sub, view, cfg, classes, device, path_col)
        out[name] = {"macro_f1": f1, "balanced_acc": bacc}
        rows.append((name, f1, bacc))
        print(f"  {name:<14} macro-F1 {f1:.4f}   balanced-acc {bacc:.4f}")

    base = out.get(baseline_name) if baseline_name else None
    if base and base["macro_f1"] == base["macro_f1"]:
        b = base["macro_f1"]
        worst_name, worst = min(
            ((k, v["macro_f1"]) for k, v in out.items() if v["macro_f1"] == v["macro_f1"]),
            key=lambda kv: kv[1])
        drop = b - worst
        rel = drop / max(b, 1e-9)
        out["_summary"] = {"baseline": b, "worst": worst, "worst_condition": worst_name,
                           "abs_drop": drop, "rel_drop": rel}
        print(f"\n  기준 {b:.4f}  →  최악 {worst:.4f} ({worst_name})")
        print(f"  하락 {drop:.4f}  ({rel:.1%})")
        if rel > 0.30:
            print(f"  🚨 {rel:.0%} 하락 — 실사용에서 무너집니다.")
            print(f"     {hint}")
        elif rel > 0.15:
            print(f"  ⚠️ {rel:.0%} 하락 — 상당히 의존하고 있습니다. 개선 여지가 큽니다.")
        else:
            print(f"  ✅ {rel:.0%} 하락 — 견고합니다.")
    else:
        print("\n  ⚠️ 기준 조건이 없어 하락폭을 못 냅니다 "
              "(zooms 에 1.0, fracs 에 0.0 을 포함하세요).")
    print("=" * 68 + "\n")
    return out


def scale_stress(model, df, cfg: CFG | None = None, classes: list[str] | None = None,
                 zooms=(0.5, 0.71, 1.0, 1.41, 2.0), n: int = 3000, seed: int = 0,
                 path_col: str = "crop_path", device: str | None = None) -> dict:
    """★ 배율 교란 — 보호자가 더 멀리/가까이 찍었을 때도 버티는가.

    크롭이 배율로 정답을 흘리고 있으면(A1 0.47% vs A6 3.08%), 모델은 배율을
    단서로 씁니다. 그런데 실사용에서는 배율이 무작위입니다. 여기서 무너집니다.

    ⚠️ 이 검사가 `crop.shortcut_baseline()` 보다 결정적입니다.
       하한선은 "데이터에 상관이 있나"를 재고, 이건 "모델이 그걸 썼나"를 잽니다.
       고정 픽셀 크롭(f320)의 효과도 하한선이 아니라 **이 숫자로** 판정하세요.
    """
    from src.config import CLASSES

    cfg = cfg or CFG()
    classes = classes or CLASSES
    views = [(_zoom_name(z), ZoomView(cfg.img_size, z)) for z in zooms]
    return _run(model, df, cfg, classes, views, n, seed, path_col, device,
                "배율 교란 검사 — 확대/축소에 견디는가",
                "→ 고정 픽셀 크롭(f320) + 넓은 배율 증강(rrc_scale)을 쓰세요.",
                baseline_name=_zoom_name(1.0) if 1.0 in tuple(zooms) else None)


def shift_stress(model, df, cfg: CFG | None = None, classes: list[str] | None = None,
                 fracs=(0.0, 0.1, 0.2, 0.3), n: int = 3000, seed: int = 0,
                 path_col: str = "crop_path", device: str | None = None) -> dict:
    """위치 교란 — 병변이 화면 중앙에 없어도 버티는가.

    크롭은 병변을 늘 정중앙에 둡니다. 모델이 "가운데만 보면 된다"를 배웠는지 확인합니다.
    """
    from src.config import CLASSES

    cfg = cfg or CFG()
    classes = classes or CLASSES
    views = [(_shift_name(f), ShiftView(cfg.img_size, f)) for f in fracs]
    return _run(model, df, cfg, classes, views, n, seed, path_col, device,
                "위치 교란 검사 — 병변이 중앙에 없어도 되는가",
                "→ 학습 증강에 위치 흔들기를 넣으세요 (넓은 rrc_scale 이 함께 해결).",
                baseline_name=_shift_name(0.0) if 0.0 in tuple(fracs) else None)


# ──────────────────────────────────────────────────────────────
# 촬영 가이드 (capture guideline) 도출
# ──────────────────────────────────────────────────────────────
def usable_range(
    model, df, cfg: CFG | None = None, classes: list[str] | None = None,
    zooms=(0.5, 0.6, 0.7, 0.85, 1.0, 1.2, 1.4, 1.7, 2.0),
    tolerances=(0.05, 0.10),
    crop_margin: float = 1.5,
    n: int = 3000, seed: int = 0, path_col: str = "crop_path",
    device: str | None = None,
) -> dict:
    """**보호자에게 뭐라고 안내할지**를 실측에서 뽑아냅니다.

    배율 강건성(scale robustness)을 모델링으로 못 잡으면, 남은 길은 애초에
    나쁜 배율이 안 들어오게 **입력을 제한**하는 것입니다. 그러려면 "얼마나
    가까이" 를 숫자로 말할 수 있어야 하는데, 그 숫자가 여기서 나옵니다.

    학습은 전혀 안 합니다 — 이미 학습된 모델에 배율만 바꿔 추론할 뿐입니다.

    기본 격자가 `scale_stress` 보다 촘촘합니다. 5개 점(간격 √2)으로는
    "5% 이내로 버티는 구간" 의 경계가 0.85x 인지 1.2x 인지 알 수 없습니다.

    Args:
        tolerances: 최고점 대비 허용 하락률. 각각에 대해 연속 구간을 찾습니다.
        crop_margin: 학습 크롭의 margin (m1.5 면 1.5). 배율을 **화면 점유율**로
            바꾸는 데 씁니다 — 보호자는 "1.2배" 를 모르지만 "화면 절반" 은 압니다.

    Returns:
        {"table": [...], "peak": {...}, "bands": {0.05: (lo, hi), ...},
         "occupancy": {...}}  — occupancy 는 화면 가로 점유율(%)
    """
    from src.config import CLASSES

    cfg = cfg or CFG()
    classes = classes or CLASSES
    views = [(_zoom_name(z), ZoomView(cfg.img_size, z)) for z in zooms]
    res = _run(model, df, cfg, classes, views, n, seed, path_col, device,
               "촬영 가이드 측정 — 어느 배율까지 버티는가",
               "→ 이 구간을 벗어나지 않도록 촬영 UI 로 유도하세요.",
               baseline_name=_zoom_name(1.0) if 1.0 in tuple(zooms) else None)

    rows = [(z, res[_zoom_name(z)]["macro_f1"]) for z in zooms
            if res.get(_zoom_name(z), {}).get("macro_f1") == res.get(_zoom_name(z), {}).get("macro_f1")]
    if not rows:
        return {"table": [], "peak": None, "bands": {}, "occupancy": {}}

    peak_z, peak_f1 = max(rows, key=lambda r: r[1])

    # 배율 → 화면 가로 점유율. 학습 크롭이 m1.5 면 1x 에서 병변이 1/1.5 = 67%.
    def occ(z: float) -> float:
        return min(z / crop_margin, 1.0)

    bands: dict[float, tuple[float, float]] = {}
    for tol in tolerances:
        floor_f1 = peak_f1 * (1 - tol)
        # peak 에서 양옆으로 **연속으로** 기준을 만족하는 구간만 인정합니다.
        # 띄엄띄엄 만족하는 건 가이드로 쓸 수 없습니다.
        ok = [z for z, f in rows if f >= floor_f1]
        lo = hi = peak_z
        srt = sorted(z for z, _ in rows)
        i = srt.index(peak_z)
        for j in range(i - 1, -1, -1):
            if srt[j] in ok:
                lo = srt[j]
            else:
                break
        for j in range(i + 1, len(srt)):
            if srt[j] in ok:
                hi = srt[j]
            else:
                break
        bands[tol] = (lo, hi)

    print(f"\n{'=' * 68}\n 촬영 가이드 — 실측에서 뽑은 허용 구간\n{'=' * 68}")
    print(f"  최고점: 배율 {peak_z}x  (macro-F1 {peak_f1:.4f})")
    print(f"  학습 크롭 margin {crop_margin} → 1x 에서 병변이 화면 가로의 "
          f"{occ(1.0):.0%}")
    print(f"\n  {'배율':>6}{'macro-F1':>11}{'최고점 대비':>12}{'화면 점유율':>12}")
    for z, f in sorted(rows):
        print(f"  {z:>5}x{f:>11.4f}{(f / peak_f1 - 1):>11.1%}{occ(z):>11.0%}")

    print()
    for tol, (lo, hi) in bands.items():
        print(f"  하락 {tol:.0%} 이내 : 배율 {lo}x ~ {hi}x  "
              f"→ 병변이 화면 가로의 **{occ(lo):.0%} ~ {occ(hi):.0%}**")
    print("=" * 68)

    return {"table": rows, "peak": {"zoom": peak_z, "macro_f1": peak_f1},
            "bands": bands,
            "occupancy": {z: occ(z) for z, _ in rows},
            "crop_margin": crop_margin, "raw": res}


def usable_shift(
    model, df, cfg: CFG | None = None, classes: list[str] | None = None,
    fracs=(0.0, 0.05, 0.10, 0.15, 0.20, 0.30),
    tolerance: float = 0.05,
    n: int = 3000, seed: int = 0, path_col: str = "crop_path",
    device: str | None = None,
) -> dict:
    """병변이 화면 중앙에서 얼마나 벗어나도 되는지 — 촬영 가이드의 두 번째 축."""
    from src.config import CLASSES

    cfg = cfg or CFG()
    classes = classes or CLASSES
    views = [(_shift_name(f), ShiftView(cfg.img_size, f)) for f in fracs]
    res = _run(model, df, cfg, classes, views, n, seed, path_col, device,
               "촬영 가이드 측정 — 병변이 중앙에서 벗어나도 되는가",
               "→ 촬영 UI 에 중앙 가이드 프레임을 두세요.",
               baseline_name=_shift_name(0.0) if 0.0 in tuple(fracs) else None)

    rows = [(f, res[_shift_name(f)]["macro_f1"]) for f in fracs
            if res.get(_shift_name(f), {}).get("macro_f1") == res.get(_shift_name(f), {}).get("macro_f1")]
    if not rows:
        return {"table": [], "max_shift": None}

    base = dict(rows).get(0.0, max(f for _, f in rows))
    limit = base * (1 - tolerance)
    allowed = 0.0
    for f, sc in sorted(rows):
        if sc >= limit:
            allowed = f
        else:
            break

    print(f"\n{'=' * 68}\n 촬영 가이드 — 위치 허용 범위\n{'=' * 68}")
    print(f"  {'이동':>6}{'macro-F1':>11}{'중앙 대비':>11}")
    for f, sc in sorted(rows):
        print(f"  {f:>5.0%}{sc:>11.4f}{(sc / base - 1):>10.1%}")
    print(f"\n  하락 {tolerance:.0%} 이내 : 중앙에서 화면의 **{allowed:.0%} 까지** 벗어나도 됨")
    print("=" * 68)
    return {"table": rows, "max_shift": allowed, "baseline": base, "raw": res}


def report(model, df, cfg: CFG | None = None, classes: list[str] | None = None,
           n: int = 2000, **kw) -> dict:
    """배율 + 위치를 한 번에. 배포 판단 직전에 부르세요."""
    return {
        "scale": scale_stress(model, df, cfg, classes, n=n, **kw),
        "shift": shift_stress(model, df, cfg, classes, n=n, **kw),
    }
