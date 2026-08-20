#!/usr/bin/env python3
"""로컬(한국) PC 에서 데이터 받고 전처리까지 끝내는 스크립트.

왜 필요한가:
  AI Hub 는 **해외 IP 다운로드를 차단**합니다. Colab / Kaggle VM 은 한국 밖이라
  클라우드에서는 데이터를 받을 수 없습니다 (HTTP 502).

  한국 PC : 다운로드 + 전처리 (CPU 만, GPU 불필요)
  클라우드: 학습 + 평가 (GPU 필요)

────────────────────────────────────────────────────────────────
사용법 A — 작게 시작 (VL01 21GB 하나만)
────────────────────────────────────────────────────────────────
    export AIHUB_API_KEY="발급받은키"
    python prepare_local.py --chunk VL01
    python prepare_local.py --finalize
    python prepare_local.py --package

────────────────────────────────────────────────────────────────
사용법 B — 청크를 이어붙여 데이터 늘리기 (드라이브 여유가 큰 경우)
────────────────────────────────────────────────────────────────
  받고 → 정제하고 → 원본 버리고 → 다음 청크. 로컬 디스크를 돌려씁니다.

    python prepare_local.py --chunk VL01      # 21GB → 크롭 후 원본 삭제
    python prepare_local.py --chunk TL01      # 90GB → 크롭 후 원본 삭제
    python prepare_local.py --chunk TL02      # 80GB → 크롭 후 원본 삭제
    python prepare_local.py --finalize        # ★ 전부 합쳐서 중복제거 + 개체분할
    python prepare_local.py --package

  ⚠️ --finalize 를 반드시 마지막에 한 번 돌리세요.
     같은 강아지가 여러 청크에 흩어져 있을 수 있어서, 청크별로 분할하면
     개체가 train/val 에 걸쳐 데이터 누수가 생깁니다.
     크롭까지만 청크별로 하고, 중복제거·분할은 합친 뒤 한 번에 합니다.

결과물:
    dogskin_prepared.zip   ← 이것만 Drive / Kaggle 에 올리면 됩니다
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Windows cmd(cp949)에서 이모지·박스문자 출력이 UnicodeEncodeError 로 죽는 것을 방지.
# src 를 import 하기 전에 먼저 걸어둡니다.
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

# 청크 별칭 → filekey
CHUNKS: dict[str, dict] = {
    "VL01": {"filekey": "517022", "gb": 21, "desc": "Validation 라벨"},
    "VS01": {"filekey": "517021", "gb": 21, "desc": "Validation 원천"},
    "TL01": {"filekey": "517019", "gb": 90, "desc": "Training 라벨 1"},
    "TL02": {"filekey": "517020", "gb": 80, "desc": "Training 라벨 2"},
    "TS01": {"filekey": "517017", "gb": 90, "desc": "Training 원천 1"},
    "TS02": {"filekey": "517018", "gb": 80, "desc": "Training 원천 2"},
}


def _die(msg: str) -> None:
    print(f"\n❌ {msg}")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────
def step_chunk(name: str, margins: list[float], keep_raw: bool = False,  # noqa: C901
               mode: str = "zip") -> None:
    """청크 하나: 다운로드 → 매니페스트 → 크롭 → 원본 삭제.

    mode="zip"     : zip 을 **풀지 않고** 안에서 바로 읽습니다. 디스크를 가장 적게 씁니다.
    mode="extract" : 반려묘·더모스코프를 뺀 선택 해제 후 처리 (디스크 여유가 있을 때)
    """
    from src import aihub, crop, env, labels, scan
    from src.config import CFG

    if name not in CHUNKS:
        _die(f"알 수 없는 청크 '{name}'. 가능: {', '.join(CHUNKS)}")
    info = CHUNKS[name]
    cfg = CFG()

    print("\n" + "=" * 68)
    print(f" 청크 {name} — {info['desc']} ({info['gb']}GB), 방식: {mode}")
    print("=" * 68)

    free = env.free_disk_gb()
    # zip 방식은 zip 자체 + 크롭본만 필요. extract 방식은 해제본까지.
    need = info["gb"] * (1.15 if mode == "zip" else 1.7)
    print(f" 여유 디스크 {free:.0f}GB / 필요 약 {need:.0f}GB")
    if free < need:
        print(" ⚠️ 공간이 부족합니다.")
        if mode != "zip":
            print("    --mode zip 을 쓰면 압축 해제 없이 처리해 훨씬 적게 듭니다.")
        print("    그래도 진행하려면 Enter, 중단하려면 Ctrl+C")
        input()

    # 1) 다운로드
    key = env.secret("AIHUB_API_KEY")
    aihub.install()
    raw = env.data_root() / name
    raw.mkdir(parents=True, exist_ok=True)
    if aihub.download(key, [info["filekey"]], dest=raw, chunk=1):
        _die("다운로드 실패. 한국에서 실행 중인지, 활용신청이 승인됐는지 확인하세요.")

    # 실패한 시도가 남긴 download_*.tar 백업 정리 (21GB 씩 쌓입니다)
    aihub.cleanup_backups(raw)

    archives = aihub.find_archives(raw)
    zips = [a for a in archives if a.suffix.lower() == ".zip"]
    tars = [a for a in archives if a.suffix.lower() != ".zip"]

    # aihubshell 은 download.tar 로 받습니다. 보통 스스로 풀지만 남아 있으면 먼저 풉니다.
    if tars:
        print(f"\n[해제] tar {len(tars)}개를 먼저 풉니다 (aihubshell 이 남긴 것)")
        for t in tars:
            try:
                shutil.unpack_archive(str(t), str(raw))
                t.unlink()
                print(f"  {t.name} 해제 완료")
            except Exception as exc:
                print(f"  ✗ {t.name}: {exc}")
        zips = [a for a in aihub.find_archives(raw) if a.suffix.lower() == ".zip"]

    if mode == "zip" and zips:
        # 2) 압축을 풀지 않고 zip 안에서 직접 읽습니다
        print("\n[매니페스트] zip 직접 읽기 — 압축 해제 안 함")
        dfs = [labels.build_from_zip(a, save=False) for a in zips]
        import pandas as pd
        df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    else:
        # 2') 선택 해제 후 디스크에서 처리
        #     (zip 이 없다 = aihubshell 이 이미 다 풀어놨다는 뜻일 수 있음)
        for a in zips:
            aihub.extract_selective(a, dest=raw, remove_archive=True)
        aihub.peek(raw)
        print("\n[매니페스트]")
        rep = scan.run(raw, quick=True)
        df = labels.build(root=raw, report=rep, save=False)

    # 3) 크롭
    print("\n[크롭]")
    first = True
    for m in margins:
        # 음수는 "고정 픽셀 창" 을 뜻합니다: -320 → f320
        # margin 크롭은 병변 크기에 따라 확대 배율이 달라져 그 배율이 정답을 흘립니다.
        # 고정 창은 피부 1mm 가 항상 같은 픽셀 수라 그 경로를 막습니다.
        if m < 0:
            d = crop.run(df, cfg, fixed_px=int(-m))
        else:
            d = crop.run(df, cfg, margin=m, tag=f"m{m:g}" if m > 0 else "full")
        if first:
            df = d          # 첫 항목 결과를 대표 매니페스트로 저장
            first = False
    labels.save(df, f"chunk_{name}.parquet")

    # 4) 원본 삭제 — 다음 청크를 위해 공간 확보
    if not keep_raw:
        print(f"\n[정리] 원본 삭제: {raw}")
        shutil.rmtree(raw, ignore_errors=True)
        print(f"       여유 디스크 {env.free_disk_gb():.0f}GB")

    print(f"\n✅ 청크 {name} 완료 — {len(df):,}행, 개체 {df['animal_id'].nunique():,}마리")
    print("   다음 청크를 받거나, 다 받았으면 --finalize 를 실행하세요.")


# ──────────────────────────────────────────────────────────────
def step_finalize(cfg_margins: list[float]) -> None:
    """모든 청크를 합쳐 중복제거 + 개체 단위 분할. **반드시 마지막에 한 번.**"""
    from src import dedup, labels, split
    from src.config import CFG

    print("\n" + "=" * 68)
    print(" FINALIZE — 전 청크 통합 → 중복제거 → 개체 단위 분할")
    print("=" * 68)
    print(" ⚠️ 이 단계를 건너뛰면 청크 경계를 넘는 데이터 누수를 막을 수 없습니다.\n")

    cfg = CFG()
    df = labels.combine()

    print("\n[중복 제거] — 청크 경계를 넘는 중복까지 함께 처리합니다")
    df, info = dedup.run(df, cfg)

    print("\n[개체 단위 분할]")
    df = split.assign(df, cfg)
    split.verify(df, fold=0, strict=True)

    labels.save(df, "manifest_final.parquet")
    print(f"\n✅ 최종 {len(df):,}행 / 개체 {df['animal_id'].nunique():,}마리")
    print("   저장: manifests/manifest_final.parquet")


# ──────────────────────────────────────────────────────────────
def step_download(filekeys: list[str]) -> None:
    from src import aihub, env

    key = env.secret("AIHUB_API_KEY")
    aihub.install()
    aihub.recommend_plan()
    if aihub.download(key, filekeys, chunk=1):
        _die("다운로드 실패")
    aihub.unpack_all()
    aihub.peek()


def step_scan() -> None:
    from src import scan

    print("\n" + "=" * 68)
    print(" 스캔 (스키마 추론)")
    print("=" * 68)
    rep = scan.run()
    scan.write_dataset_card(rep)
    print("\n📋 이 출력을 공유해주세요. 특히:")
    print("   · 무증상(정상) 데이터 존재 여부   · 개체ID 필드")
    print("   · polygon/bbox 키                · 클래스별 이미지 수")


def step_package(out: Path, tags: str | None = None) -> None:
    """업로드용 zip. `tags` 로 크롭 태그를 골라 담을 수 있습니다.

    ⚠️ 전체 zip 은 쉽게 5GB 를 넘습니다. Kaggle/Drive 업로드가 오래 걸리고
       중간에 끊기면 처음부터입니다. 학습에 실제로 쓰는 태그만 담으면
       크게 줄일 수 있습니다:

           py prepare_local.py --package --tags m1.5 --out dogskin_m15.zip
           py prepare_local.py --package --tags full --out dogskin_full.zip

       나눠 올려도 `env.load_prepared()` 가 여러 입력을 합쳐서 인식합니다.
    """
    from src import env

    print("\n" + "=" * 68)
    print(" 업로드용 패키지 만들기")
    print("=" * 68)

    work = env.work_root()
    want = [t.strip() for t in tags.split(",")] if tags else None

    crops = work / "crops"
    if crops.exists():
        have = sorted(p.name for p in crops.iterdir() if p.is_dir())
        print(f"  가진 크롭 태그: {have}")
        if want:
            missing = [t for t in want if t not in have]
            if missing:
                raise SystemExit(f"❌ 없는 태그: {missing}. 가진 것: {have}")
            print(f"  담을 태그      : {want}  (나머지는 제외)")
        # 태그별 용량을 보여줍니다 — 무엇이 큰지 알아야 판단이 됩니다
        for t in have:
            n = sum(1 for _ in (crops / t).rglob("*.jpg"))
            mb = sum(f.stat().st_size for f in (crops / t).rglob("*.jpg")) / 1024**2
            mark = "→ 담음" if (want is None or t in want) else "  제외"
            print(f"    {mark}  {t:<8} {n:>7,}장  {mb:>8,.0f} MB")

    stage = work.parent / "_pkg"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    for sub in ("crops", "manifests", "reports"):
        s = work / sub
        if not s.exists():
            continue
        if sub == "crops" and want:
            (stage / "crops").mkdir(parents=True, exist_ok=True)
            for t in want:
                print(f"  담는 중: crops/{t}")
                shutil.copytree(s / t, stage / "crops" / t)
        else:
            print(f"  담는 중: {sub}")
            shutil.copytree(s, stage / sub)

    print("\n  압축 중… (몇 분 걸립니다)")
    made = shutil.make_archive(str(out.with_suffix("")), "zip", str(stage))
    shutil.rmtree(stage)

    gb = Path(made).stat().st_size / 1024**3
    print(f"\n✅ 완성: {made}  ({gb:.2f} GB)")
    print("\n올리는 곳")
    print("  · Google Drive — 업로드 후 Colab 에서 마운트")
    print("  · Kaggle       — Datasets → New Dataset (Private)")
    print("\n⚠️ 반드시 **비공개**로 올리세요. AI Hub 데이터는 재배포 금지입니다.")
    print("⚠️ Colab 에서는 Drive 에서 직접 읽지 말고 /content 로 풀어서 쓰세요 (10배 차이).")


# ──────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(
        description="한국 PC 에서 AI Hub 데이터 받고 전처리 (해외 IP 차단 우회)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="청크: " + ", ".join(f"{k}({v['gb']}GB)" for k, v in CHUNKS.items()),
    )
    p.add_argument("--chunk", metavar="이름",
                   help="청크 하나를 받아 크롭까지 (예: VL01, TL01). 끝나면 원본 삭제")
    p.add_argument("--finalize", action="store_true",
                   help="★ 모든 청크를 합쳐 중복제거 + 개체 단위 분할. 마지막에 한 번")
    p.add_argument("--package", action="store_true", help="업로드용 zip 생성")
    p.add_argument("--tags", default=None, metavar="목록",
                   help="패키지에 담을 크롭 태그 (콤마 구분, 예: m1.5,full). "
                        "생략하면 전부. 업로드 용량을 줄일 때 씁니다")
    p.add_argument("--all", action="store_true", help="VL01 만 받아 전 과정 (가장 간단)")
    p.add_argument("--download", action="store_true", help="(단독) 다운로드만")
    p.add_argument("--scan", action="store_true", help="(단독) 스캔만")
    p.add_argument("--filekey", default="517022")
    p.add_argument("--margins", default="1.5,2.5,0,-320",
                   help="크롭 방식 목록. 양수=margin 배율(m1.5), 0=중앙 정사각(full), "
                        "음수=고정 픽셀 창(-320 → f320, 배율이 일정해 지름길 차단)")
    p.add_argument("--keep-raw", action="store_true", help="원본을 지우지 않음")
    p.add_argument("--mode", choices=["zip", "extract"], default="zip",
                   help="zip: 압축을 풀지 않고 바로 읽음 (디스크 최소, 기본값) / "
                        "extract: 선택 해제 후 처리 (디스크 여유가 있을 때)")
    p.add_argument("--out", default="dogskin_prepared.zip")
    a = p.parse_args()

    if not any([a.all, a.chunk, a.finalize, a.package, a.download, a.scan]):
        p.print_help()
        sys.exit(0)

    needs_key = a.all or a.chunk or a.download
    if needs_key and not os.environ.get("AIHUB_API_KEY"):
        _die('환경변수 AIHUB_API_KEY 가 없습니다.\n'
             '   Linux/Mac : export AIHUB_API_KEY="키"\n'
             "   Windows   : set AIHUB_API_KEY=키")

    from src import env
    env.describe()
    env.set_seed(42)
    margins = [float(x) for x in a.margins.split(",")]

    if a.all:
        step_chunk("VL01", margins, a.keep_raw, a.mode)
        step_finalize(margins)
        step_package(Path(a.out), a.tags)
    else:
        if a.download:
            step_download(a.filekey.split(","))
        if a.scan:
            step_scan()
        if a.chunk:
            step_chunk(a.chunk, margins, a.keep_raw, a.mode)
        if a.finalize:
            step_finalize(margins)
        if a.package:
            step_package(Path(a.out), a.tags)

    print("\n" + "=" * 68)
    print(" 끝났습니다.")
    print("=" * 68)


if __name__ == "__main__":
    main()
