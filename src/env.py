"""실행 환경(Colab / Kaggle / 로컬) 자동 감지 + 경로·시크릿 통합.

노트북 첫 셀에서 이것만 부르면 나머지 코드는 환경을 몰라도 됩니다.

    from src import env
    E = env.describe()          # 환경 요약 출력
    ROOT = env.data_root()      # 데이터가 놓일 곳
    KEY  = env.secret("AIHUB_API_KEY")

⚠️ API 키를 코드나 노트북에 하드코딩하지 마세요.
   Colab  : 왼쪽 사이드바 🔑 Secrets 에 AIHUB_API_KEY 등록
   Kaggle : Add-ons → Secrets 에 AIHUB_API_KEY 등록
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

EnvName = Literal["colab", "kaggle", "local"]


def _fix_console_encoding() -> None:
    """Windows 콘솔에서 한글·이모지 출력이 죽지 않게 합니다.

    한국어 Windows 의 cmd 는 기본 코드페이지가 cp949 라서, 이 프로젝트가 쓰는
    ✅ ⚠️ ★ ─ 같은 문자를 인코딩하지 못하고 UnicodeEncodeError 로 죽습니다.
    출력 스트림을 UTF-8 로 바꾸고, 그래도 못 찍는 글자는 대체 문자로 넘깁니다
    (로그 한 줄 때문에 몇십 분짜리 전처리가 죽는 것보다 낫습니다).
    """
    if os.name != "nt":
        return
    try:
        import ctypes

        ctypes.windll.kernel32.SetConsoleOutputCP(65001)   # UTF-8 코드페이지
    except Exception:
        pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_fix_console_encoding()

# GPU 이름 → 대략적인 VRAM(GB). 배치 크기 자동 추천에만 씁니다.
_VRAM_HINT = {
    "T4": 16, "P100": 16, "V100": 16, "L4": 24, "A100": 40,
    "A10": 24, "RTX 3090": 24, "RTX 4090": 24, "H100": 80,
}


# ──────────────────────────────────────────────────────────────
# 환경 감지
# ──────────────────────────────────────────────────────────────
def detect() -> EnvName:
    """현재 실행 환경을 반환합니다."""
    if "google.colab" in sys.modules:
        return "colab"
    # Colab 은 import 되기 전일 수도 있으므로 경로로도 한 번 더 확인
    if Path("/content").is_dir() and os.environ.get("COLAB_RELEASE_TAG"):
        return "colab"
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/working").is_dir():
        return "kaggle"
    return "local"


def is_colab() -> bool:
    return detect() == "colab"


def is_kaggle() -> bool:
    return detect() == "kaggle"


# ──────────────────────────────────────────────────────────────
# 경로
# ──────────────────────────────────────────────────────────────
def project_root() -> Path:
    """리포지토리 루트(= 이 파일의 부모의 부모)."""
    return Path(__file__).resolve().parent.parent


def workspace() -> Path:
    """쓰기 가능한 작업 루트. 환경마다 다릅니다."""
    e = detect()
    if e == "colab":
        return Path("/content")
    if e == "kaggle":
        # /kaggle/input 은 읽기 전용이므로 working 을 씁니다.
        return Path("/kaggle/working")
    return project_root()


def persist_root() -> Path | None:
    """**세션이 끊겨도 살아남는** 저장소. 없으면 None.

    ⚠️ Colab 의 `/content` 는 휘발성입니다. 세션이 끊기면 체크포인트까지 전부
       사라집니다. 90분짜리 학습이 80분에 끊기면 처음부터 다시입니다.

    그래서 체크포인트는 Drive 로 복사해 둡니다. Drive 가 마운트돼 있어야 하므로
    `env.mount_drive()` 를 먼저 부르세요 (노트북 첫 셀이 합니다).

    환경변수 `DOG_SKIN_PERSIST` 로 직접 지정할 수 있습니다.
    """
    override = os.environ.get("DOG_SKIN_PERSIST")
    if override:
        p = Path(override)
        p.mkdir(parents=True, exist_ok=True)
        return p

    if is_colab():
        drive = Path("/content/drive/MyDrive")
        if drive.exists():
            p = drive / "dogskin_work"
            p.mkdir(parents=True, exist_ok=True)
            return p
        return None            # Drive 미마운트 — 호출부가 경고합니다

    if detect() == "kaggle":
        # Kaggle 은 /kaggle/working 이 세션 종료 시 출력으로 보존됩니다
        return workspace()

    # 로컬은 애초에 휘발성이 아닙니다
    return work_root()


def data_root() -> Path:
    """원본 데이터(압축 해제본)가 놓일 곳. 환경변수 DOG_SKIN_DATA 로 덮어쓸 수 있습니다."""
    override = os.environ.get("DOG_SKIN_DATA")
    if override:
        return Path(override)
    return workspace() / "data" / "raw"


def work_root() -> Path:
    """전처리 산출물(크롭 이미지, 매니페스트, 체크포인트)이 놓일 곳."""
    override = os.environ.get("DOG_SKIN_WORK")
    if override:
        return Path(override)
    return workspace() / "data" / "work"


def ensure_dirs() -> dict[str, Path]:
    """필요한 디렉터리를 만들고 경로 사전을 돌려줍니다."""
    paths = {
        "data_root": data_root(),
        "work_root": work_root(),
        "manifests": work_root() / "manifests",
        "crops": work_root() / "crops",
        "checkpoints": work_root() / "checkpoints",
        "reports": work_root() / "reports",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def mount_drive(mountpoint: str = "/content/drive") -> Path | None:
    """Colab 에서만 Google Drive 를 마운트합니다. 다른 환경에서는 None."""
    if not is_colab():
        print("[env] Colab 이 아니므로 Drive 마운트를 건너뜁니다.")
        return None
    from google.colab import drive  # type: ignore[import-not-found]

    drive.mount(mountpoint)
    return Path(mountpoint) / "MyDrive"


# ──────────────────────────────────────────────────────────────
# 시크릿
# ──────────────────────────────────────────────────────────────
def secret(name: str, required: bool = True) -> str | None:
    """환경에 맞는 시크릿 저장소에서 값을 읽습니다.

    우선순위: 환경변수 → Colab userdata → Kaggle UserSecrets
    """
    val = os.environ.get(name)
    if val:
        return val

    e = detect()
    if e == "colab":
        try:
            from google.colab import userdata  # type: ignore[import-not-found]

            val = userdata.get(name)
        except Exception as exc:  # 미등록 / 접근 거부 모두 여기로
            val = None
            _hint = f"({type(exc).__name__})"
        if val:
            return val
    elif e == "kaggle":
        try:
            from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]

            val = UserSecretsClient().get_secret(name)
        except Exception:
            val = None
        if val:
            return val

    if required:
        raise RuntimeError(
            f"시크릿 '{name}' 을(를) 찾을 수 없습니다.\n"
            f"  현재 환경: {e}\n"
            "  Colab  → 왼쪽 사이드바 🔑 Secrets 에 추가 후 '노트북 액세스' 토글 ON\n"
            "  Kaggle → Add-ons → Secrets 에 추가 후 이 노트북에 Attach\n"
            f"  로컬   → export {name}=...\n"
            "  ⚠️ 절대 코드나 노트북 셀에 직접 붙여넣지 마세요."
        )
    return None


# ──────────────────────────────────────────────────────────────
# 하드웨어
# ──────────────────────────────────────────────────────────────
@dataclass
class Device:
    kind: str  # "cuda" | "mps" | "cpu"
    name: str = "CPU"
    vram_gb: float = 0.0
    bf16: bool = False       # 네이티브 bf16 (Ampere 8.0+). 에뮬레이션은 False 로 둡니다.
    count: int = 0
    capability: str = ""

    @property
    def amp_dtype(self) -> str:
        """AMP(혼합정밀) 에 쓸 dtype. bf16 이 되면 bf16 이 안전합니다."""
        if self.kind != "cuda":
            return "none"
        return "bfloat16" if self.bf16 else "float16"


def device_info() -> Device:
    try:
        import torch
    except ImportError:
        return Device(kind="cpu")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        # ⚠️ torch.cuda.is_bf16_supported() 는 소프트웨어 에뮬레이션까지 True 로 칩니다.
        #    T4(Turing, 7.5)에서 True 가 나오는데, 실제로 bf16 을 쓰면 fp16 보다 훨씬 느립니다.
        #    네이티브 bf16 은 Ampere(8.0) 이상에만 있으므로 compute capability 로 판정합니다.
        real_bf16 = props.major >= 8
        return Device(
            kind="cuda",
            name=props.name,
            vram_gb=round(props.total_memory / 1024**3, 1),
            bf16=real_bf16,
            count=torch.cuda.device_count(),
            capability=f"{props.major}.{props.minor}",
        )
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return Device(kind="mps", name="Apple MPS")
    return Device(kind="cpu")


def suggest_batch_size(img_size: int, model_scale: str = "base") -> int:
    """VRAM 과 입력 해상도로 배치 크기를 대충 추천합니다.

    보수적으로 잡습니다 — OOM 으로 3시간짜리 학습이 죽는 것보다
    배치가 작아 조금 느린 편이 낫습니다. 부족하면 config 에서 직접 올리세요.
    """
    dev = device_info()
    if dev.kind != "cuda":
        return 8
    vram = dev.vram_gb or _VRAM_HINT.get(dev.name.split()[-1], 16)

    # 224px / base 모델 / 16GB 를 기준점 48 로 두고 스케일링
    # (ResNet50@224 를 AMP 로 돌리면 배치 48 이 8GB 남짓입니다. 여전히 보수적입니다)
    scale_factor = {"tiny": 2.0, "small": 1.4, "base": 1.0, "large": 0.5}.get(model_scale, 1.0)
    px_factor = (224 / max(img_size, 64)) ** 2
    bs = 48 * (vram / 16) * scale_factor * px_factor

    # ⚠️ 2의 거듭제곱으로 내림하면 최대 절반을 버립니다.
    #    실제로 T4(14.7GB)에서 계산값 44 가 32 도 아닌 **16** 이 됐습니다
    #    (이전 기준점 32 × 0.92 = 29.4 → 16). 학습이 2~3배 느려집니다.
    #    중간 단계를 둔 사다리로 내림합니다.
    ladder = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256]
    out = ladder[0]
    for v in ladder:
        if v <= bs:
            out = v
    return out


def suggest_workers() -> int:
    """DataLoader 워커 수. CPU 코어에 맞춥니다.

    왜 중요한가: 512px JPEG 을 디코딩+리사이즈하는 건 CPU 일입니다.
    워커가 부족하면 GPU 가 놀면서 데이터를 기다립니다 — 배치를 키워도 안 빨라집니다.
    실측: Colab T4 에서 74 img/s 였는데, ResNet50@224 AMP 의 계산 능력은
          150~200 img/s 입니다. 절반 이상을 데이터 로딩에서 흘리고 있었습니다.
    """
    import os

    n = os.cpu_count() or 2
    # 메인 프로세스 몫을 남기고, 너무 많으면 오히려 컨텍스트 스위칭 비용이 큽니다
    return max(2, min(n - 1, 8))


def free_disk_gb(path: Path | None = None) -> float:
    p = path or workspace()
    p.mkdir(parents=True, exist_ok=True)
    return round(shutil.disk_usage(p).free / 1024**3, 1)


# ──────────────────────────────────────────────────────────────
# 요약 출력
# ──────────────────────────────────────────────────────────────
@dataclass
class EnvSummary:
    env: EnvName
    python: str
    torch: str
    device: Device
    paths: dict[str, Path] = field(default_factory=dict)
    free_disk_gb: float = 0.0


def describe(verbose: bool = True) -> EnvSummary:
    """환경을 감지하고 디렉터리를 만든 뒤 요약을 출력합니다."""
    try:
        import torch

        tv = torch.__version__
    except ImportError:
        tv = "(미설치)"

    s = EnvSummary(
        env=detect(),
        python=sys.version.split()[0],
        torch=tv,
        device=device_info(),
        paths=ensure_dirs(),
        free_disk_gb=free_disk_gb(),
    )

    if verbose:
        d = s.device
        print("─" * 58)
        print(f" 실행 환경   : {s.env}")
        print(f" Python      : {s.python}   |  torch {s.torch}")
        if d.kind == "cuda":
            amp = "bf16" if d.bf16 else "fp16 + GradScaler"
            print(f" GPU         : {d.name}  {d.vram_gb}GB  x{d.count}  "
                  f"(sm_{d.capability}, AMP={amp})")
        else:
            print(f" GPU         : 없음 ({d.kind}) — 학습은 매우 느립니다")
        print(f" 여유 디스크 : {s.free_disk_gb} GB")
        print(f" 원본 데이터 : {s.paths['data_root']}")
        print(f" 작업 폴더   : {s.paths['work_root']}")
        print("─" * 58)
        if s.env == "colab" and s.free_disk_gb < 60:
            print("⚠️  디스크 여유가 적습니다. STEP 1 에서 부분 다운로드를 꼭 쓰세요.")
        if d.kind == "cpu":
            print("⚠️  런타임 유형을 GPU 로 바꿔주세요. Colab: 런타임 → 런타임 유형 변경 → T4 GPU")
    return s


def load_prepared(
    zip_path: str | Path | None = None,
    dest: Path | None = None,
    force: bool = False,
) -> Path:
    """로컬에서 만든 `dogskin_prepared.zip` 을 클라우드 작업 폴더로 풉니다.

    한국 PC 에서 `prepare_local.py` 로 전처리한 결과를 Drive/Kaggle 에 올린 뒤,
    학습 노트북 첫 부분에서 이걸 부르면 됩니다.

    zip_path 를 생략하면 흔한 위치를 자동으로 뒤집니다.

    ⚠️ 반드시 **로컬 디스크**로 풉니다. Drive 에 마운트된 채로 이미지를 읽으면
       네트워크 왕복 때문에 학습이 10배 가까이 느려집니다.
    """
    import zipfile

    dest = Path(dest) if dest else work_root()

    if zip_path is None:
        cands: list[Path] = []
        for base in (Path("/content/drive/MyDrive"), Path("/kaggle/input"),
                     Path("/content"), Path.cwd()):
            if base.exists():
                cands += sorted(base.rglob("dogskin_prepared*.zip"))[:5]
        if not cands:
            raise FileNotFoundError(
                "dogskin_prepared.zip 을 찾지 못했습니다.\n"
                "  · Colab  : Drive 를 마운트했는지 확인 → env.mount_drive()\n"
                "  · Kaggle : 우측 Add Input 으로 데이터셋을 붙였는지 확인\n"
                "  · 경로를 직접 주려면 env.load_prepared('/경로/dogskin_prepared.zip')"
            )
        zip_path = cands[0]
        print(f"[env] 자동 탐색: {zip_path}")

    zip_path = Path(zip_path)
    marker = dest / ".prepared_from"
    if marker.exists() and marker.read_text().strip() == str(zip_path) and not force:
        print(f"[env] 이미 풀려 있습니다: {dest}  (다시 풀려면 force=True)")
    else:
        dest.mkdir(parents=True, exist_ok=True)
        size = zip_path.stat().st_size / 1024**3
        print(f"[env] 압축 해제 {zip_path.name} ({size:.2f}GB) → {dest}")
        print("      (Drive 에서 직접 읽지 않고 로컬 디스크로 풉니다 — 학습 속도 때문)")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dest)
        marker.write_text(str(zip_path))

    crops = dest / "crops"
    mans = sorted((dest / "manifests").glob("*.parquet")) if (dest / "manifests").exists() else []
    n_crop = sum(1 for _ in crops.rglob("*.jpg")) if crops.exists() else 0
    tags = sorted(p.name for p in crops.iterdir() if p.is_dir()) if crops.exists() else []

    print(f"[env] 크롭 {n_crop:,}장, 크롭 태그 {tags}")
    print(f"[env] 매니페스트 {[m.name for m in mans]}")
    if n_crop == 0:
        print("⚠️ 크롭이 하나도 없습니다. zip 내용을 확인하세요.")
    if not any("final" in m.name for m in mans):
        print("⚠️ manifest_final.parquet 이 없습니다.")
        print("   로컬에서 `python prepare_local.py --finalize` 를 돌렸는지 확인하세요.")
        print("   이게 없으면 개체 단위 분할(fold/holdout)이 안 되어 있습니다.")
    return dest


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """재현성을 위한 시드 고정.

    deterministic=True 면 cuDNN 을 결정론 모드로 두는데, 느려집니다.
    최종 결과 재현이 필요할 때만 켜세요.
    """
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
    except ImportError:
        pass


def pip_install(packages: str = "timm imagehash pyarrow grad-cam", quiet: bool = True) -> None:
    """노트북에서 필요한 패키지를 설치합니다 (이미 있으면 빠르게 통과)."""
    flags = ["-q"] if quiet else []
    subprocess.run(
        [sys.executable, "-m", "pip", "install", *flags, *packages.split()],
        check=False,
    )
