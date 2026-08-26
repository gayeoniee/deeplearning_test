"""털 가중 샘플러 — **총량을 보존하는가**가 이 검사의 전부입니다.

가중치를 그냥 올리면 정상 사진이 전체적으로 더 많이 뽑혀서 클래스 균형이
같이 바뀝니다. 그러면 헛알림이 줄어도 "털 가중치 덕분" 인지 "정상을 더 봐서"
인지 **못 가릅니다.** 교란(confound)이고, 실험이 통째로 무의미해집니다.

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
for i in range(400):
    lab = "A7" if i % 2 == 0 else "ABNORMAL"
    hairy = lab == "A7" and i % 4 == 0          # 정상의 절반만 잔선 많음
    a = (np.tile(np.array([40, 200], np.uint8), (64, 32)) if hairy
         else np.full((64, 64), 128, np.uint8))
    p = td / f"{i}.jpg"
    Image.fromarray(np.dstack([a] * 3)).save(p, quality=95)
    rows.append({"crop_path": str(p), "label": lab, "hairy": hairy})
df = pd.DataFrame(rows)
ds = data.SkinDataset(df, None, "crop_path", classes=["A7", "ABNORMAL"])
t = np.asarray(ds.targets)
hairy = df["hairy"].values

print("\n[1] 값의 정의가 한 곳인가")
check("도구가 src.texture 를 쓴다",
      "from src.texture import" in (ROOT / "tools" / "false_alarm_stats.py")
      .read_text(encoding="utf-8"))
check("image_stats 와 hair_of 가 같은 값",
      abs(texture.image_stats(rows[0]["crop_path"])["hair"]
          - texture.hair_of(rows[0]["crop_path"])) < 1e-9)

print("\n[2] ★ 클래스 총량을 보존한다 (안 하면 실험이 무의미)")
s = data.hair_sampler(ds, alpha=1.0, cache=td / "c.parquet", verbose=False)
w = s.weights.numpy()
for name, m in [("A7", t == 0), ("ABNORMAL", t == 1)]:
    check(f"{name} 총 가중치 = 장수", abs(w[m].sum() - m.sum()) < 1e-6,
          f"{w[m].sum():.3f} vs {m.sum()}")
check("전체 합도 보존", abs(w.sum() - len(ds)) < 1e-6)

print("\n[3] 털 많은 정상을 실제로 더 뽑는다")
hi, lo = w[hairy].mean(), w[(t == 0) & ~hairy].mean()
check("털 많은 쪽이 더 무겁다", hi > lo * 1.2, f"{hi:.3f} vs {lo:.3f}")
check("병변 쪽은 안 건드린다", np.allclose(w[t == 1], w[t == 1][0]))
check("alpha 를 키우면 격차도 커진다",
      (data.hair_sampler(ds, alpha=3.0, cache=td / "c.parquet", verbose=False)
       .weights.numpy()[hairy].mean()) > hi)

print("\n[4] 끄면 아무 일도 없다")
w0 = data.hair_sampler(ds, alpha=0.0, verbose=False).weights.numpy()
check("alpha=0 은 완전 균등", np.allclose(w0, 1.0))

print("\n[5] 캐시가 두 번째 호출을 건너뛴다")
import io                                                       # noqa: E402
import contextlib                                               # noqa: E402

# ⚠️ hair_sampler 는 **대상 클래스(A7)만** 계산합니다. 캐시가 그 200장만
#    갖고 있으므로, 400장 전부로 부르면 나머지 200장은 새로 재는 게 정상입니다.
#    같은 200장으로 불러야 캐시 재사용을 확인할 수 있습니다.
a7_paths = [r["crop_path"] for r in rows if r["label"] == "A7"]
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    texture.hair_index(a7_paths, cache=td / "c.parquet")
check("'계산' 이 아니라 '캐시' 를 쓴다",
      "캐시" in buf.getvalue() and "hair 계산" not in buf.getvalue(),
      buf.getvalue()[:120])

print("\n[6] 못 읽는 파일이 있어도 안 죽는다")
bad = texture.hair_index([rows[0]["crop_path"], "/없음.jpg"], verbose=False)
check("길이가 맞고 nan 이 안 남는다",
      len(bad) == 2 and np.isfinite(bad).all(), str(bad))

print("\n" + "=" * 60)
print(f" 통과 {ok} / {ok + fail}")
sys.exit(1 if fail else 0)
