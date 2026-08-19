#!/usr/bin/env python3
"""로컬(한국) PC 에서 데이터 받고 전처리까지 끝내는 스크립트.

왜 필요한가:
  AI Hub 는 **해외 IP 다운로드를 차단**합니다. Colab / Kaggle VM 은 한국 밖에 있어서
  클라우드에서는 데이터를 받을 수 없습니다 (HTTP 502 + "해외에서의 데이터 다운로드를 제한").

  그래서 역할을 나눕니다:
     한국 PC : 다운로드 + 전처리 (CPU 만 있으면 됨, GPU 불필요)
     클라우드: 학습 + 평가 (GPU 필요)

  전처리 후 크롭본은 원본의 1/5~1/10 이라 업로드가 현실적입니다.
     원본 21GB  →  크롭본 2~5GB 정도

사용법:
    # 1. 필요한 패키지
    pip install -r requirements.txt

    # 2. API 키를 환경변수로 (코드에 절대 쓰지 마세요)
    export AIHUB_API_KEY="발급받은키"          # Windows: set AIHUB_API_KEY=...

    # 3. 전체 실행
    python prepare_local.py --all

    # 또는 단계별로
    python prepare_local.py --download
    python prepare_local.py --scan
    python prepare_local.py --preprocess
    python prepare_local.py --package

결과물:
    dogskin_prepared.zip   ← 이것만 Kaggle / Drive 에 올리면 됩니다
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# 리포 루트를 import 경로에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_FILEKEY = "517022"   # VL01.zip — Validation 라벨, 21GB


def step_download(filekeys: list[str]) -> None:
    from src import aihub, env

    print("\n" + "=" * 68)
    print(" STEP 1 — 다운로드")
    print("=" * 68)

    key = env.secret("AIHUB_API_KEY")
    aihub.install()
    aihub.recommend_plan()

    failed = aihub.download(key, filekeys, chunk=1)
    if failed:
        print("\n❌ 다운로드 실패. 아래를 확인하세요:")
        print("   · 지금 한국에서 실행 중인가? (VPN/해외 서버면 차단됩니다)")
        print("   · AI Hub 활용신청이 승인됐는가?")
        print("   · API Key 가 유효한가? (마이페이지에서 재발급 가능)")
        sys.exit(1)

    aihub.unpack_all()
    aihub.peek()


def step_scan() -> None:
    from src import scan

    print("\n" + "=" * 68)
    print(" STEP 2 — 스캔 (스키마 추론)")
    print("=" * 68)
    rep = scan.run()
    scan.write_dataset_card(rep)

    print("\n📋 이 출력을 그대로 복사해서 공유해주세요.")
    print("   특히 다음 4가지가 중요합니다:")
    print("     · 무증상(정상) 데이터 존재 여부")
    print("     · 개체ID 필드")
    print("     · polygon/bbox 키")
    print("     · 클래스별 이미지 수")


def step_preprocess(margins: list[float]) -> None:
    from src import crop, dedup, labels, scan, split
    from src.config import CFG

    print("\n" + "=" * 68)
    print(" STEP 3 — 전처리 (CPU 만 사용, GPU 불필요)")
    print("=" * 68)

    cfg = CFG()
    rep = scan.ScanReport.load()

    print("\n[1/4] 매니페스트 생성")
    df = labels.build(report=rep)

    print("\n[2/4] 중복 제거")
    df, _ = dedup.run(df, cfg)

    print("\n[3/4] 개체 단위 분할")
    df = split.assign(df, cfg)
    split.verify(df, fold=0, strict=True)

    print("\n[4/4] ROI 크롭")
    for m in margins:
        tag = f"m{m:g}" if m > 0 else "full"
        d = crop.run(df, cfg, margin=m, tag=tag)
        labels.save(d, f"manifest_{tag}.parquet")


def step_package(out: Path) -> None:
    """클라우드로 옮길 것만 골라 zip 으로 묶습니다."""
    from src import env

    print("\n" + "=" * 68)
    print(" STEP 4 — 업로드용 패키지 만들기")
    print("=" * 68)

    work = env.work_root()
    stage = work.parent / "_pkg"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    for sub in ("crops", "manifests", "reports"):
        src = work / sub
        if src.exists():
            print(f"  담는 중: {sub}")
            shutil.copytree(src, stage / sub)

    out = out.with_suffix("")
    print("\n  압축 중… (몇 분 걸립니다)")
    made = shutil.make_archive(str(out), "zip", str(stage))
    shutil.rmtree(stage)

    gb = Path(made).stat().st_size / 1024**3
    print(f"\n✅ 완성: {made}  ({gb:.2f} GB)")
    print("\n다음 중 하나로 클라우드에 올리세요:")
    print("  · Kaggle  (권장) — Datasets → New Dataset → 이 zip 업로드 (비공개)")
    print("               학습 노트북에서 /kaggle/input/<이름> 으로 붙습니다")
    if gb < 14:
        print("  · Google Drive — 무료 15GB 안에 들어갑니다")
    else:
        print(f"  · Google Drive — {gb:.1f}GB 라 무료 15GB 를 넘습니다. Kaggle 을 쓰세요")
    print("\n⚠️ 이 zip 에는 AI Hub 데이터가 들어 있습니다. 반드시 **비공개**로 올리세요.")
    print("   공개 데이터셋으로 올리면 재배포가 되어 이용약관 위반입니다.")


def main() -> None:
    p = argparse.ArgumentParser(
        description="한국 PC 에서 AI Hub 데이터 받고 전처리까지 (해외 IP 차단 우회)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--all", action="store_true", help="전 단계 순서대로 실행")
    p.add_argument("--download", action="store_true")
    p.add_argument("--scan", action="store_true")
    p.add_argument("--preprocess", action="store_true")
    p.add_argument("--package", action="store_true")
    p.add_argument("--filekey", default=DEFAULT_FILEKEY,
                   help=f"받을 filekey (콤마 구분). 기본 {DEFAULT_FILEKEY} = VL01 21GB")
    p.add_argument("--margins", default="1.5,2.5,0",
                   help="크롭 margin 목록. 0 은 크롭 없이 중앙 정사각")
    p.add_argument("--out", default="dogskin_prepared.zip")
    a = p.parse_args()

    if not any([a.all, a.download, a.scan, a.preprocess, a.package]):
        p.print_help()
        sys.exit(0)

    if not os.environ.get("AIHUB_API_KEY") and (a.all or a.download):
        print("❌ 환경변수 AIHUB_API_KEY 가 없습니다.")
        print('   Linux/Mac : export AIHUB_API_KEY="발급받은키"')
        print("   Windows   : set AIHUB_API_KEY=발급받은키")
        sys.exit(1)

    from src import env
    env.describe()
    env.set_seed(42)

    if a.all or a.download:
        step_download(a.filekey.split(","))
    if a.all or a.scan:
        step_scan()
    if a.all or a.preprocess:
        step_preprocess([float(x) for x in a.margins.split(",")])
    if a.all or a.package:
        step_package(Path(a.out))

    print("\n" + "=" * 68)
    print(" 끝났습니다.")
    print("=" * 68)


if __name__ == "__main__":
    main()
