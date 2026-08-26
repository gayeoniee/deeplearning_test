"""팟을 켜자마자 **가장 먼저** 돌리세요 — 몇 초, GPU 과금 방어용.

    uv run --extra train python tools/preflight.py

왜 이게 있나
------------
런팟은 **켜져 있는 매 순간 과금**됩니다 ($0.75/hr). 그런데 우리 노트북은
Colab/Kaggle 을 전제로 짜여 있어서, 처음 오는 환경에서는 첫 셀부터 막힐 수
있습니다. 두 시간짜리 학습이 5분 뒤에 죽는 것보다, **10초 만에 죽는 게**
훨씬 쌉니다.

여기서 ✅ 가 다 나오면 노트북을 열어도 됩니다.
❌ 가 하나라도 있으면 **학습을 시작하지 마세요** — 고칠 방법을 같이 찍습니다.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAIL: list[str] = []
WARN: list[str] = []


def ok(name: str, detail: str = "") -> None:
    print(f"  ✅ {name}" + (f"   {detail}" if detail else ""))


def bad(name: str, how: str) -> None:
    print(f"  ❌ {name}")
    for line in how.splitlines():
        print(f"       {line}")
    FAIL.append(name)


def warn(name: str, how: str) -> None:
    print(f"  ⚠️  {name}")
    for line in how.splitlines():
        print(f"       {line}")
    WARN.append(name)


def check_gpu() -> None:
    print("\n[1] GPU")
    try:
        import torch
    except ImportError:
        bad("torch 가 없습니다", "uv sync --extra train")
        return
    if not torch.cuda.is_available():
        bad("CUDA 를 못 씁니다",
            "GPU 없는 팟이거나 드라이버가 안 맞습니다.\n"
            "nvidia-smi 가 되는지 먼저 보세요.\n"
            "⚠️ CPU 로 돌리면 20~30배 느립니다 — 시작하지 마세요.")
        return
    n = torch.cuda.get_device_properties(0)
    ok(f"{n.name}  {n.total_memory / 1024 ** 3:.1f}GB  x{torch.cuda.device_count()}",
       f"torch {torch.__version__}")
    if n.total_memory / 1024 ** 3 < 15:
        warn("VRAM 이 16GB 미만입니다",
             "배치가 자동으로 줄어 학습이 느려집니다.")


DATA_OK = False


def check_data() -> None:
    global DATA_OK
    print("\n[2] 데이터")
    from src import env

    w = env.work_root()
    print(f"     work_root = {w}")
    mf = w / "manifests" / "manifest_final.parquet"
    if not mf.exists():
        bad(f"매니페스트가 없습니다: {mf}",
            "크롭 데이터를 work_root 아래에 놓으세요:\n"
            "  <work_root>/crops/f320/... , <work_root>/manifests/*.parquet\n"
            "환경변수로 위치를 바꿀 수 있습니다:  export DOG_SKIN_WORK=/workspace/data/work")
        return

    import pandas as pd

    df = pd.read_parquet(mf)
    need = {"image_path", "label", "is_holdout", "animal_id"}
    miss = need - set(df.columns)
    if miss:
        bad(f"매니페스트에 컬럼이 없습니다: {sorted(miss)}",
            f"있는 것: {sorted(df.columns)[:15]}")
        return
    ok(f"매니페스트 {len(df):,}행", f"holdout {int(df['is_holdout'].sum()):,}")
    DATA_OK = True

    from src import crop

    tags = crop.available_tags()
    if not tags:
        bad("크롭이 하나도 없습니다",
            f"{w / 'crops'} 아래에 태그 폴더(f320 등)가 있어야 합니다.")
        return
    ok(f"크롭 태그 {tags}")

    # 실제로 파일이 있나 — 경로만 맞고 파일이 없는 경우가 실제로 있었습니다
    from src import agent

    for tag in (agent.STAGE1_TAG, agent.STAGE2_TAG):
        if tag not in tags:
            warn(f"크롭 '{tag}' 가 없습니다",
                 f"1단계={agent.STAGE1_TAG} / 2단계={agent.STAGE2_TAG} 입니다.\n"
                 "1단계 실험만 할 거면 f320 만 있어도 됩니다.")
            continue
        d = crop.switch_tag(df.head(200), tag, verbose=False)
        have = d["crop_path"].apply(lambda p: Path(p).exists()).mean()
        (ok if have > 0.99 else bad)(
            f"'{tag}' 파일 존재 {have:.1%} (표본 200장)",
            "" if have > 0.99 else "업로드가 덜 끝났을 수 있습니다.")


def check_split() -> None:
    print("\n[3] 분할 — 누수가 있으면 여기서 멈춥니다")
    if not DATA_OK:
        print("     (앞에서 막혀서 건너뜁니다)")
        return
    from src import env, split, stages

    import pandas as pd

    df = pd.read_parquet(env.work_root() / "manifests" / "manifest_final.parquet")
    try:
        split.verify(stages.to_stage1(df, verbose=False), fold=0, strict=True)
        ok("train / val / holdout 겹침 없음")
    except Exception as exc:                                    # noqa: BLE001
        bad(f"분할 검증 실패: {exc}", "매니페스트가 --finalize 를 거쳤는지 확인하세요.")


def check_persist() -> None:
    print("\n[4] 세션이 끊겨도 남는 곳")
    from src import env

    p = env.persist_root()
    if p is None:
        bad("영속 저장소가 없습니다",
            "체크포인트가 세션과 함께 사라집니다.\n"
            "export DOG_SKIN_PERSIST=/workspace/persist")
        return
    try:
        t = Path(p) / ".preflight"
        t.write_text("ok", encoding="utf-8")
        t.unlink()
    except OSError as exc:
        bad(f"쓰기가 안 됩니다: {p} ({exc})", "권한이나 볼륨 마운트를 확인하세요.")
        return
    ok(f"{p}", "쓰기 확인")
    if "runpod" in str(p).lower() or str(p).startswith("/workspace"):
        warn("런팟 네트워크 볼륨으로 보입니다",
             "팟을 지워도 볼륨은 **계속 과금**됩니다. 다 끝나면 볼륨도 지우세요.")


def check_disk() -> None:
    print("\n[5] 디스크")
    import shutil

    from src import env

    for name, p in (("work_root", env.work_root()),
                    ("persist", env.persist_root() or env.work_root())):
        try:
            du = shutil.disk_usage(p)
        except OSError:
            continue
        free = du.free / 1024 ** 3
        # 릴리스 하나가 800MB 를 넘고, 체크포인트는 실험마다 쌓입니다
        (ok if free >= 15 else warn if free >= 5 else bad)(
            f"{name} 여유 {free:.1f}GB",
            "" if free >= 15 else "체크포인트·릴리스가 쌓이면 모자랄 수 있습니다.")


def check_code() -> None:
    print("\n[6] 코드가 서로 맞는가")
    from src import agent, config, texture

    ok(f"1단계 크롭 {agent.STAGE1_TAG} / 2단계 {agent.STAGE2_TAG}")
    ok(f"노트북 버전 {config.NOTEBOOK_VERSION}")
    # hair 정의가 갈라졌는지 — 두 곳에서 같은 값이 나와야 합니다
    src = (ROOT / "tools" / "false_alarm_stats.py").read_text(encoding="utf-8")
    if "from src.texture import" in src:
        ok("hair 정의가 한 곳 (src/texture.py)")
    else:
        bad("hair 정의가 갈라졌습니다",
            "tools/false_alarm_stats.py 가 src.texture 를 import 해야 합니다.")
    if hasattr(texture, "hair_of") and hasattr(config.CFG(), "hair_alpha"):
        ok("털 가중 샘플러 준비됨", f"기본 alpha={config.CFG().hair_alpha}")
    else:
        bad("hair 샘플러 배선이 빠졌습니다", "git pull 을 했는지 확인하세요.")


def main() -> None:
    print("=" * 66)
    print(" PREFLIGHT — 학습을 시작해도 되는가")
    print("=" * 66)
    from src import env

    print(f"\n환경: {env.detect()}   작업폴더: {env.workspace()}")
    print("이 검사가 통과해야 학습을 시작합니다. 몇 초면 끝납니다.")

    for fn in (check_gpu, check_data, check_split, check_persist,
               check_disk, check_code):
        try:
            fn()
        except SystemExit:
            raise
        except Exception as exc:                                # noqa: BLE001
            bad(f"{fn.__name__} 도중 예외: {type(exc).__name__}: {exc}",
                "위 메시지를 그대로 알려주세요.")

    print("\n" + "=" * 66)
    if FAIL:
        print(f" ❌ 막힌 것 {len(FAIL)}개 — **학습을 시작하지 마세요**")
        for f in FAIL:
            print(f"    · {f}")
        print("\n 지금 팟이 과금 중입니다. 위를 고치고 다시 도세요.")
        sys.exit(1)
    if WARN:
        print(f" ✅ 막힌 것 없음 (주의 {len(WARN)}개)")
        for w in WARN:
            print(f"    · {w}")
    else:
        print(" ✅ 전부 통과 — 노트북을 열어도 됩니다.")
    print("=" * 66)


if __name__ == "__main__":
    main()
