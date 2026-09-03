"""털 가중 샘플러 — **총량을 안 바꾸는가**가 핵심입니다.

가중치를 그냥 올리면 정상 사진이 전체적으로 더 뽑혀서 **클래스 균형이 같이
바뀝니다.** 그러면 헛알림이 줄어도 "털 가중치 덕분" 인지 "정상을 더 봐서" 인지
못 가릅니다. 그래서 클래스별 총 가중치 보존을 검사로 박아둡니다.

    uv run --extra train python tests/test_hair_sampler.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402
from PIL import Image                                           # noqa: E402

from src import data, texture                                   # noqa: E402

ok = fail = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {name}")
    else:
        fail += 1
        print(f"  FAIL  {name}  {detail}")


td = Path(tempfile.mkdtemp())
rows = []
for i in range(600):
    lab = "A7" if i % 2 == 0 else "ABNORMAL"
    hairy = lab == "A7" and i % 4 == 0            # 정상의 절반은 잔선 많게
    a = (np.tile(np.array([40, 200], np.uint8), (64, 32)) if hairy
         else np.full((64, 64), 128, np.uint8))
    p = td / f"{i}.jpg"
    Image.fromarray(np.dstack([a] * 3)).save(p, quality=95)
    rows.append({"crop_path": str(p), "label": lab, "hairy": hairy})
df = pd.DataFrame(rows)
ds = data.SkinDataset(df, None, "crop_path", classes=["A7", "ABNORMAL"])
tgt = np.asarray(ds.targets)
hairy = df["hairy"].values


def weights(alpha: float) -> np.ndarray:
    s = data.hair_sampler(ds, alpha=alpha, cache=td / "c.parquet", verbose=False)
    return np.asarray(s.weights, dtype=float)


print("\n[1] hair 정의가 한 곳에서만 나온다")
one = texture.image_stats(td / "0.jpg")
check("image_stats 와 hair_of 가 같은 값",
      abs(one["hair"] - texture.hair_of(td / "0.jpg")) < 1e-9)
src = (ROOT / "tools" / "false_alarm_stats.py").read_text(encoding="utf-8")
check("도구가 src.texture 를 가져다 쓴다 (복붙 아님)",
      "from src.texture import" in src or "from src import texture" in src)
check("도구가 자기만의 image_stats 를 다시 정의하지 않는다",
      "def image_stats" not in src)

print("\n[2] ★ 클래스별 총량이 보존된다")
for alpha in (0.0, 0.5, 1.0, 3.0):
    w = weights(alpha)
    a7, ab = w[tgt == 0].sum(), w[tgt == 1].sum()
    check(f"alpha={alpha:g} — A7 총합 == 장수 ({a7:.1f} vs {(tgt == 0).sum()})",
          abs(a7 - (tgt == 0).sum()) < 1e-6, f"{a7}")
    check(f"alpha={alpha:g} — ABNORMAL 총합 == 장수",
          abs(ab - (tgt == 1).sum()) < 1e-6, f"{ab}")

print("\n[3] 잔선 많은 정상만 더 자주 뽑힌다")
w0, w1, w3 = weights(0.0), weights(1.0), weights(3.0)
check("alpha=0 이면 전부 균등", np.allclose(w0, 1.0))
check("alpha=1 — 잔선 많은 정상이 매끈한 정상보다 자주",
      w1[(tgt == 0) & hairy].mean() > w1[(tgt == 0) & ~hairy].mean())
check("alpha 를 키우면 격차가 커진다",
      (w3[(tgt == 0) & hairy].mean() / w3[(tgt == 0) & ~hairy].mean())
      > (w1[(tgt == 0) & hairy].mean() / w1[(tgt == 0) & ~hairy].mean()))
check("★ 이상(ABNORMAL) 쪽은 안 건드린다 — 놓침엔 hair 신호가 없었습니다",
      np.allclose(w1[tgt == 1], w1[tgt == 1][0]))

print("\n[4] 기준선과 처치가 **다른 이름**을 받는다")
# 같은 이름이면 train.fit 이 기준선을 보고 처치를 통째로 건너뜁니다
from src.config import CFG, with_aug, with_finetune                # noqa: E402


def exp_name(balance: str, alpha: float) -> str:
    return (f"stage1_effnetv2_s_f320_384"
            + (f"_hair{alpha:g}" if balance == "hair_weighted" else ""))


check("기준선 != 처치", exp_name("none", 0) != exp_name("hair_weighted", 1))
check("alpha 가 다르면 이름도 다름",
      exp_name("hair_weighted", 1) != exp_name("hair_weighted", 3))
src_e = (ROOT / "src" / "experiments.py").read_text(encoding="utf-8")
check("experiments 가 exp_name 에 alpha 를 붙인다", '_hair{hair_alpha:g}' in src_e)
check("experiments 가 balance 를 덮어쓸 수 있다",
      'balance or ("none" if stage == 1' in src_e)

print("\n[5] 못 읽는 파일이 있어도 안 죽는다")
bad_df = pd.concat([df, pd.DataFrame([{"crop_path": "/없음.jpg", "label": "A7",
                                       "hairy": False}])], ignore_index=True)
bad_ds = data.SkinDataset(bad_df, None, "crop_path", classes=["A7", "ABNORMAL"])
w = np.asarray(data.hair_sampler(bad_ds, alpha=1.0, verbose=False).weights, float)
check("가중치에 nan 이 없다", np.isfinite(w).all())
check("총량은 여전히 보존", abs(w[np.asarray(bad_ds.targets) == 1].sum()
                          - (np.asarray(bad_ds.targets) == 1).sum()) < 1e-6)

print("\n" + "=" * 60)
print(f" 통과 {ok} / {ok + fail}")
sys.exit(1 if fail else 0)
