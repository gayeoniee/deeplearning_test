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
import zipfile
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
               mode: str = "zip", only: list[str] | None = None) -> None:
    """청크 하나: 다운로드 → 매니페스트 → 크롭 → 원본 삭제.

    mode="zip"     : zip 을 **풀지 않고** 안에서 바로 읽습니다. 디스크를 가장 적게 씁니다.
    mode="extract" : 반려묘·더모스코프를 뺀 선택 해제 후 처리 (디스크 여유가 있을 때)
    only=["A5","A6"] : 그 라벨만 크롭합니다. **매니페스트는 전부 만듭니다** —
                       나중에 마음이 바뀌면 `--recrop` 으로 zip 없이는 못 되살리니
                       무엇이 있었는지는 기록에 남겨둡니다.

    ⚠️ `--only` 는 **다운로드를 줄이지 못합니다.** AI Hub 가 zip 6개 통짜로만 주기 때문에
       (`--list` 로 실측) 80GB 는 그대로 받아야 하고, 줄어드는 건 크롭 용량뿐입니다.
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
    # ⚠️ 끝나고 남는 양이 아니라 **받는 도중 최대치**를 봐야 합니다.
    #    aihubshell 이 download.tar + 푼 내용 + 합친 파일을 겹쳐 놓습니다
    #    (소스를 읽어 확인한 2~3배. src/aihub.DL_PEAK).
    #    예전엔 여기서 1.15배로 봤는데, 그러면 "여유 93GB / 필요 92GB → 통과" 를
    #    찍고 받다가 디스크가 찹니다. 통과시키면 안 되는 걸 통과시켰습니다.
    lo, hi = (info["gb"] * m for m in aihub.DL_PEAK)
    keep = info["gb"] * 1.15                      # 다 받고 나면 남는 양 (zip + 크롭)
    print(f" 여유 디스크 {free:.0f}GB")
    print(f" 받는 도중 최대 {lo:.0f}~{hi:.0f}GB  ·  다 받고 나면 약 {keep:.0f}GB")
    if free < hi:
        worst = "❌ 최대치가 디스크보다 큽니다" if free < lo else "⚠️ 최대치에 못 미칩니다"
        print(f" {worst} — 중간에 'No space left' 로 죽을 수 있습니다.")
        print("    조각(.part*)으로 안 쪼개진 청크면 2배에서 끝나 될 수도 있습니다.")
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
    #    --only 는 **크롭할 행만** 줄입니다. 매니페스트(df)는 통째로 저장해서
    #    "그때 뭐가 있었는지" 는 남깁니다 — 나중에 클래스 구성을 다시 재려면 필요합니다.
    print("\n[크롭]")
    df_all = df
    if only:
        keep = df["label"].astype(str).isin(only)
        print(f"  --only {','.join(only)} → {len(df):,}행 중 {int(keep.sum()):,}행만 크롭")
        for lab, n in df["label"].value_counts().items():
            mark = "크롭" if str(lab) in only else "  ·  "
            print(f"     {mark}  {lab:<6}{n:>9,}장")
        if not keep.any():
            _die(f"--only {','.join(only)} 에 해당하는 행이 없습니다. "
                 f"있는 라벨: {', '.join(map(str, df['label'].unique()))}")
        df = df[keep].reset_index(drop=True)
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
    if only:
        # 크롭한 행의 crop_path 를 전체 매니페스트에 되붙입니다.
        # (크롭 안 한 행은 crop_path 가 비어 있고, 학습 때 걸러집니다)
        import pandas as pd
        df = df_all.merge(
            df[[c for c in df.columns if c == "image_path" or c.startswith("crop_path")]],
            on="image_path", how="left", suffixes=("", "_new"))
        for c in [c for c in df.columns if c.endswith("_new")]:
            df[c[:-4]] = df[c]
            df = df.drop(columns=[c])
    # 3-b) ★ phash — **원본을 지우기 전에** 재 둡니다
    #
    # 중복 제거는 --finalize 에서 전 청크를 합쳐 한 번에 합니다. 그런데 그때는
    # 원본이 이미 없습니다. 이 PC 에서 계속 작업하면 zip 이 남아 있어 우연히
    # 됐지만, **다른 PC 에서 청크를 처리해 크롭만 가져오면** 그 청크는 영영
    # 못 읽습니다 — 그리고 dedup 은 그 경우 에러 없이 넘어갑니다
    # (src/dedup.py 의 주석 참고). 여기서 미리 재서 매니페스트에 넣어둡니다.
    #
    # 비용은 zip 을 한 번 더 읽는 것뿐이고, 대신 --finalize 가 그만큼 빨라집니다.
    print("\n[phash] 원본을 지우기 전에 지문을 재 둡니다 (--finalize 가 이걸 씁니다)")
    try:
        from src import dedup
        hashes = dedup.compute_hashes(df_all, hash_size=cfg.phash_size, workers=8)
        df["phash"] = df["image_path"].map(hashes)
        got = int(df["phash"].notna().sum())
        print(f"       {got:,}/{len(df):,}장 ({got / max(len(df), 1):.1%})")
        if got < len(df) * 0.98:
            print("  ⚠️ 못 읽은 사진이 있습니다. 그대로 두면 --finalize 가 멈춥니다 "
                  "(멈추는 게 맞습니다 — 조용히 넘어가면 중복을 못 잡습니다).")
    except Exception as exc:                                       # noqa: BLE001
        # 여기서 죽으면 다운로드부터 다시 해야 합니다 — 크롭은 이미 끝났으니
        # 매니페스트는 저장하고, 대신 무엇이 빠졌는지 크게 알립니다.
        print(f"  ⚠️ phash 계산 실패: {type(exc).__name__}: {exc}")
        print("     크롭과 매니페스트는 저장합니다. 다만 원본을 지우기 전에")
        print("     --keep-raw 로 다시 돌리거나, 이 PC 에서 --finalize 까지 끝내세요.")

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
def step_recrop(tag: str, raw_root: str | None = None, workers: int = 8) -> None:
    """★ 분할을 건드리지 않고 **크롭 태그 하나만** 추가로 만듭니다.

    `--chunk` 와 결정적으로 다른 점: **매니페스트를 다시 만들지 않습니다.**

    왜 전용 명령이 필요한가
    -----------------------
    나중에 새 크롭(예: f320)이 필요해졌을 때 `--chunk` 를 다시 돌리면
    스캔·중복제거·분할이 전부 다시 돌아갑니다. 그러면 **어떤 개체가 holdout 에
    가는지가 바뀔 수 있고**, 그 순간 지금까지 잰 숫자와 비교가 불가능해집니다.
    holdout 에 있던 개체가 학습셋으로 넘어가면 시험지 자체가 오염됩니다.

    이 명령은 `manifest_final.parquet` 을 **읽기만** 하고, 거기 적힌 경로로
    크롭 파일만 새로 만듭니다.

    경로가 바뀌어도 되는 이유
    -------------------------
    크롭 파일 이름은 `md5(image_path)` 이고, 실제 읽기는 `zip_path`+`zip_member`
    가 담당합니다. 둘이 분리돼 있어서, 재다운로드가 다른 폴더에 떨어져도
    `--raw` 로 zip 위치만 알려주면 **파일 이름은 예전 그대로** 나옵니다.
    이미 있는 full/m1.5/m2.5 와 짝이 맞습니다.
    """
    import pandas as pd

    from src import crop, env
    from src.config import CFG

    mdir = env.work_root() / "manifests"
    mf = mdir / "manifest_final.parquet"
    if not mf.exists():
        _die(f"분할이 끝난 매니페스트가 없습니다: {mf}\n"
             "   --recrop 은 기존 분할을 재사용하는 명령입니다.\n"
             "   아직 --finalize 를 한 번도 안 돌렸다면 --chunk 부터 하세요.")

    fixed_px, margin = crop.fixed_of_tag(tag), crop.margin_of_tag(tag)
    if not fixed_px and not margin and tag != "full":
        _die(f"모르는 크롭 태그 '{tag}'. 예: f320, m2.5, full")

    print("\n" + "=" * 68)
    print(f" RECROP — 태그 '{tag}' 만 추가 (분할은 그대로)")
    print("=" * 68)

    before = _tag_counts()
    df = pd.read_parquet(mf)
    print(f"\n[매니페스트] {len(df):,}행 / 개체 {df['animal_id'].nunique():,}마리 (읽기 전용)")
    if tag in before:
        print(f"⚠️ 태그 '{tag}' 가 이미 {before[tag]:,}장 있습니다. 있는 건 건너뜁니다.")

    # ── zip 위치 갈아끼우기 (파일 이름의 근거인 image_path 는 절대 안 건드립니다)
    if raw_root:
        new_root = Path(raw_root).resolve()
        if "zip_path" not in df.columns:
            _die("이 매니페스트는 zip 방식이 아니라 --raw 로 위치를 옮길 수 없습니다.\n"
                 "   원본을 매니페스트에 적힌 경로 그대로 되돌려 놓으세요.")
        def _rebase(old: str) -> str:
            return str(new_root / Path(str(old)).name)
        df = df.copy()
        df["zip_path"] = df["zip_path"].apply(_rebase)
        print(f"[재지정] zip 위치 → {new_root}  (image_path 는 그대로 두므로 파일 이름이 안 바뀝니다)")

    # ── 먼저 열어봅니다. 몇 시간 크롭한 뒤에 전부 실패했다는 걸 알면 늦습니다.
    print("\n[사전 확인] 원본이 실제로 열리는지 확인합니다 …")
    sample = df.sample(min(20, len(df)), random_state=0)
    ok = 0
    first_err = None
    for _, row in sample.iterrows():
        try:
            crop._open_source(row).close()
            ok += 1
        except Exception as exc:                                  # noqa: BLE001
            first_err = first_err or f"{type(exc).__name__}: {exc}"
    print(f"   표본 {len(sample)}장 중 {ok}장 열림")
    if ok < len(sample):
        _die("원본을 못 엽니다. 크롭을 시작하지 않았습니다.\n"
             f"   첫 오류: {first_err}\n"
             f"   매니페스트가 기대하는 zip: {df['zip_path'].iloc[0]}\n"
             "   재다운로드한 위치가 다르면 --raw <폴더> 로 알려주세요.")

    cfg = CFG()
    if fixed_px:
        crop.run(df, cfg, fixed_px=fixed_px, workers=workers)
    else:
        crop.run(df, cfg, margin=margin, tag=tag, workers=workers)

    # ── 기존 태그가 그대로인지, 분할 파일이 안 바뀌었는지 확인합니다
    after = _tag_counts()
    print("\n[대조] 크롭 태그별 장수")
    for k in sorted(set(before) | set(after)):
        b, a2 = before.get(k, 0), after.get(k, 0)
        mark = "  ← 새로 만듦" if k == tag else (" ✅" if b == a2 else "  ⚠️ 바뀌었습니다")
        print(f"   {k:<8} {b:>7,} → {a2:>7,}{mark}")

    lost = [k for k in before if after.get(k, 0) < before[k]]
    if lost:
        _die(f"기존 크롭이 줄었습니다: {lost}. 확인이 필요합니다.")
    print(f"\n✅ '{tag}' 완료. manifest_final.parquet 은 건드리지 않았습니다.")
    print(f"   다음: py prepare_local.py --package --tags {tag} --out dogskin_{tag}.zip")


def _tag_counts() -> dict[str, int]:
    """크롭 폴더의 태그별 파일 수."""
    from src import env

    d = env.work_root() / "crops"
    if not d.exists():
        return {}
    return {p.name: sum(1 for _ in p.rglob("*.jpg")) for p in sorted(d.iterdir()) if p.is_dir()}


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


def step_list() -> None:
    """★ AI Hub 가 **어느 단위까지** filekey 를 주는지 봅니다.

    이게 왜 중요한가 — TL02 는 80GB 짜리 zip 한 덩어리라, 통째로 받으려면
    디스크가 그만큼 비어 있어야 합니다. 그런데 AI Hub 가 zip **안쪽**까지
    filekey 를 준다면 조각조각 받아서 크롭하고 지우기를 반복할 수 있습니다.
    그러면 디스크 문제가 통째로 사라집니다.

    `KNOWN_FILES_561` 에 하드코딩된 6개는 2026-08 에 **관측된** 것일 뿐,
    API 가 그것만 준다는 뜻은 아닙니다. 실제 출력을 봐야 압니다.
    """
    from src import aihub, env

    key = env.secret("AIHUB_API_KEY")
    aihub.install()

    print("\n" + "=" * 68)
    print(" AI Hub 파일 목록 (-mode l 원본)")
    print("=" * 68)
    text = aihub.raw_listing(key)
    print(text)

    files = aihub.parse_listing(text)
    print("\n" + "=" * 68)
    print(f" 파싱 결과: {len(files)}개")
    print("=" * 68)
    if not files:
        print(" ⚠️ 파싱 0건 — 위 원본 출력을 그대로 공유해주세요.")
        return

    for f in files:
        print(f"   {f.filekey:>8}  {str(f.size):>10}  {f.path}/{f.name}" if f.path
              else f"   {f.filekey:>8}  {str(f.size):>10}  {f.name}")

    big = [f for f in files if "GB" in str(f.size)]
    print(f"\n 총 {len(files)}개 중 GB 단위 {len(big)}개")
    if len(files) > 10:
        print(" ✅ zip 안쪽까지 filekey 가 있습니다 — **조각내서 받을 수 있습니다.**")
        print("    → 디스크가 부족해도 조금씩 받고 크롭하고 지우기를 반복하면 됩니다.")
    else:
        print(" ❌ 큰 zip 단위로만 줍니다 — 통째로 받는 수밖에 없습니다.")
        print("    → 그 크기만큼 디스크가 비어 있어야 합니다.")


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

    # ⚠️ 예전에는 임시 폴더(_pkg)로 전부 복사한 뒤 압축했습니다. 3.7GB 를 복사했다가
    #    지우는 셈이라 디스크도 시간도 두 배로 들고, 탐색기로 그 폴더를 열어두면
    #    삭제될 때 "위치를 사용할 수 없습니다" 가 뜹니다.
    #    이제 원본에서 zip 으로 **바로** 씁니다.
    #
    # ⚠️ JPEG 은 이미 압축돼 있어서 deflate 를 걸어도 1~2% 밖에 안 줄어드는데
    #    CPU 시간은 몇 배로 씁니다. 이미지는 무압축(STORED)으로 담습니다.
    files: list[tuple[Path, str]] = []
    for sub in ("crops", "manifests", "reports"):
        s = work / sub
        if not s.exists():
            continue
        roots = [(s / t, f"crops/{t}") for t in want] if (sub == "crops" and want) \
            else [(s, sub)]
        for root, arc_base in roots:
            for f in sorted(root.rglob("*")):
                if f.is_file():
                    files.append((f, f"{arc_base}/{f.relative_to(root).as_posix()}"))

    total_mb = sum(f.stat().st_size for f, _ in files) / 1024**2
    print(f"\n  담을 파일 {len(files):,}개 / {total_mb:,.0f} MB")
    print("  압축 중… (이미지는 무압축으로 담아 빠릅니다)")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".zip.part")      # 중간에 끊겨도 반쪽 zip 을 안 남깁니다
    done_mb = 0.0
    step = max(len(files) // 20, 1)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for i, (f, arc) in enumerate(files):
            store = f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".zip")
            z.write(f, arc, compress_type=zipfile.ZIP_STORED if store else None)
            done_mb += f.stat().st_size / 1024**2
            if i % step == 0 or i == len(files) - 1:
                pct = done_mb / total_mb if total_mb > 0 else 1.0
                print(f"    {pct:5.0%}  ({done_mb:,.0f} / {total_mb:,.0f} MB)", flush=True)
    out.unlink(missing_ok=True)
    tmp.rename(out)
    made = str(out)

    gb = out.stat().st_size / 1024**3
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
    p.add_argument("--list", action="store_true", dest="list_files",
                   help="★ AI Hub 가 어느 단위까지 filekey 를 주는지 확인 "
                        "(조각내서 받을 수 있는지 판단)")
    p.add_argument("--filekey", default="517022")
    p.add_argument("--margins", default="1.5,2.5,0,-320",
                   help="크롭 방식 목록. 양수=margin 배율(m1.5), 0=중앙 정사각(full), "
                        "음수=고정 픽셀 창(-320 → f320, 배율이 일정해 지름길 차단)")
    p.add_argument("--recrop", metavar="태그",
                   help="★ 분할을 건드리지 않고 크롭 태그 하나만 추가 (예: f320). "
                        "매니페스트를 다시 만들지 않으므로 holdout 이 오염되지 않습니다")
    p.add_argument("--raw", metavar="폴더", default=None,
                   help="--recrop 전용. 재다운로드한 zip 이 예전과 다른 폴더면 그 위치")
    p.add_argument("--only", metavar="라벨",
                   help="★ 그 라벨만 크롭 (콤마 구분, 예: A5,A6). 매니페스트는 전부 만듭니다. "
                        "다운로드는 안 줄어듭니다 — AI Hub 가 zip 통짜로만 줍니다")
    p.add_argument("--keep-raw", action="store_true", help="원본을 지우지 않음")
    p.add_argument("--mode", choices=["zip", "extract"], default="zip",
                   help="zip: 압축을 풀지 않고 바로 읽음 (디스크 최소, 기본값) / "
                        "extract: 선택 해제 후 처리 (디스크 여유가 있을 때)")
    p.add_argument("--out", default="dogskin_prepared.zip")
    a = p.parse_args()

    if not any([a.all, a.chunk, a.finalize, a.package, a.download,
                a.scan, a.recrop, a.list_files]):
        p.print_help()
        sys.exit(0)

    needs_key = a.all or a.chunk or a.download or a.list_files
    if needs_key and not os.environ.get("AIHUB_API_KEY"):
        _die('환경변수 AIHUB_API_KEY 가 없습니다.\n'
             '   Linux/Mac : export AIHUB_API_KEY="키"\n'
             "   Windows   : set AIHUB_API_KEY=키")

    # 목록만 보는 건 환경 설명·시드 없이 바로 끝냅니다
    if a.list_files:
        step_list()
        return

    from src import env
    env.describe()
    env.set_seed(42)
    margins = [float(x) for x in a.margins.split(",")]
    only = [x.strip() for x in a.only.split(",") if x.strip()] if a.only else None

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
            step_chunk(a.chunk, margins, a.keep_raw, a.mode, only)
        if a.recrop:
            # ⚠️ --recrop 과 --finalize 를 같이 쓰면 분할이 다시 계산돼
            #    기존 크롭·기존 측정값과의 짝이 깨집니다. 애초에 못 쓰게 막습니다.
            if a.finalize:
                _die("--recrop 과 --finalize 는 같이 쓸 수 없습니다.\n"
                     "   --recrop 은 '기존 분할을 그대로 두고 크롭만 추가' 하는 명령이고,\n"
                     "   --finalize 는 그 분할을 다시 계산합니다. 목적이 정반대입니다.")
            step_recrop(a.recrop, a.raw)
        if a.finalize:
            step_finalize(margins)
        if a.package:
            step_package(Path(a.out), a.tags)

    print("\n" + "=" * 68)
    print(" 끝났습니다.")
    print("=" * 68)


if __name__ == "__main__":
    main()
