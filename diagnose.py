#!/usr/bin/env python3
"""실물 JSON 을 들여다보는 진단 도구.

스캔 결과가 이상할 때 (개체ID 가 이미지마다 고유하다거나, 병변 면적이 100% 라거나)
"실제 JSON 이 어떻게 생겼는지" 를 봐야 고칠 수 있습니다.

    py diagnose.py

출력을 그대로 복사해서 공유하면 추출 로직을 실물에 맞춰 확정할 수 있습니다.
"""

from __future__ import annotations

import json
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

if os.name == "nt":
    try:
        import ctypes

        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    except Exception:
        pass
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from src import env  # noqa: E402
from src.scan import flatten  # noqa: E402

SAMPLE = 800
LINE = "=" * 72


def find_zip() -> Path:
    cands = sorted(env.data_root().rglob("*.zip"))
    if not cands:
        print(f"zip 을 찾지 못했습니다: {env.data_root()}")
        sys.exit(1)
    return max(cands, key=lambda p: p.stat().st_size)


def main() -> None:
    zp = find_zip()
    print(LINE)
    print(f" 진단 대상: {zp}  ({zp.stat().st_size / 1024**3:.1f}GB)")
    print(LINE)

    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        jsons = [n for n in names if n.lower().endswith(".json")]

        # ── 1. 경로 구조 ──────────────────────────────────────
        print("\n[1] 경로 구조 — 유증상/무증상 각각의 예시\n")
        for tag in ("유증상", "무증상"):
            hits = [n for n in jsons if tag in n]
            print(f"  {tag}: {len(hits):,}개")
            for e in hits[:2]:
                print(f"    {e}")
        print("\n  경로 깊이별 고유 세그먼트:")
        seg: dict[int, Counter] = defaultdict(Counter)
        for n in jsons[:20000]:
            for d, s in enumerate(Path(n).parts[:-1]):
                seg[d][s] += 1
        for d in sorted(seg):
            top = seg[d].most_common(6)
            if len(seg[d]) <= 12:
                print(f"    depth{d}: {[t[0] for t in top]}")

        # ── 2. 유증상 / 무증상 JSON 원본 ──────────────────────
        print(f"\n{LINE}\n[2] JSON 원본 — 이게 제일 중요합니다\n")
        for tag in ("유증상", "무증상"):
            pick = next((n for n in jsons if tag in n), None)
            if not pick:
                print(f"  ({tag} 없음)")
                continue
            print(f"  ── {tag} ── {pick}")
            data = json.loads(z.read(pick).decode("utf-8-sig"))
            txt = json.dumps(data, indent=2, ensure_ascii=False)
            print("\n".join("    " + l for l in txt.split("\n")[:70]))
            if txt.count("\n") > 70:
                print("    ... (생략)")
            print()

        # ── 3. 키별 고유값 개수 — 개체ID 후보 찾기 ────────────
        print(f"{LINE}\n[3] 키별 고유값 개수 — 개체ID 후보 찾기\n")
        import random

        random.seed(0)
        picks = random.sample(jsons, min(SAMPLE, len(jsons)))
        vals: dict[str, list] = defaultdict(list)
        for jn in picks:
            try:
                rec = json.loads(z.read(jn).decode("utf-8-sig"))
            except Exception:
                continue
            for k, v in flatten(rec if isinstance(rec, dict) else rec[0]).items():
                if isinstance(v, (str, int, float)) and not isinstance(v, bool):
                    vals[k].append(v)

        n = len(picks)
        print(f"  샘플 {n}개 기준")
        print(f"  {'고유값':>7} {'평균장수':>8}  키                                     예시")
        print("  " + "-" * 88)
        rows = []
        for k, vs in vals.items():
            u = len(set(vs))
            rows.append((u, k, vs[:2]))
        for u, k, ex in sorted(rows):
            per = len(picks) / max(u, 1)
            mark = ""
            if 1.5 <= per <= 200 and u > 3:
                mark = "  <<< 개체ID 후보"
            print(f"  {u:>7} {per:>8.1f}  {k:<38} {str(ex)[:26]}{mark}")

        # ── 4. 좌표와 이미지 크기 비교 ────────────────────────
        print(f"\n{LINE}\n[4] 좌표 vs 이미지 크기 — 병변 면적이 100% 인 이유\n")
        for tag in ("유증상", "무증상"):
            sub = [x for x in picks if tag in x][:3]
            print(f"  ── {tag} ──")
            for jn in sub:
                try:
                    rec = json.loads(z.read(jn).decode("utf-8-sig"))
                except Exception:
                    continue
                flat = flatten(rec if isinstance(rec, dict) else rec[0])
                geo = {k: v for k, v in flat.items()
                       if re.search(r"box|poly|point|loca|coord|x|y|width|height|size",
                                    k, re.I)}
                print(f"    {Path(jn).name}")
                for k, v in list(geo.items())[:14]:
                    print(f"      {k:<44} = {v}")
            print()

        # ── 5. 파일명 패턴 ────────────────────────────────────
        print(f"{LINE}\n[5] 파일명 패턴 — 개체ID 가 파일명에 있는지\n")
        pat = Counter()
        for jn in jsons[:20000]:
            stem = Path(jn).stem
            pat[re.sub(r"[A-Za-z]+", "L", re.sub(r"\d+", "#", stem))] += 1
        for p, c in pat.most_common(6):
            print(f"    {p:<44} x{c:,}")
        print("\n  실제 파일명 예시:")
        for jn in jsons[:6]:
            print(f"    {Path(jn).stem}")
        print("\n  토큰 분해 (구분자 _ - .):")
        for jn in jsons[:3]:
            toks = re.split(r"[_\-.]", Path(jn).stem)
            print(f"    {toks}")

    print(f"\n{LINE}")
    print(" 이 출력을 통째로 복사해서 공유해주세요.")
    print(" 확정할 것: ① 개체ID 필드  ② 병변 좌표 필드  ③ 무증상 라벨 규칙")
    print(LINE)


if __name__ == "__main__":
    main()
