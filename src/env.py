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
    """현재 실행 환경을 반환합니다.

    ⚠️ **Kaggle 을 먼저 봅니다.** 예전에는 `"google.colab" in sys.modules` 를
       가장 먼저 봤는데, Kaggle 이미지에도 `google-colab` 패키지와 `/content`
       가 있어서 **Kaggle 세션이 Colab 으로 오판**됐습니다. 그러면 노트북이
       `drive.mount()` 를 부르고, Kaggle 에는 실제 Colab VM 이 없으니
       `NotImplementedError` 로 죽습니다.

       `google.colab` 을 import 할 수 있다는 것과 **Colab VM 위에 있다는 것은
       다릅니다.** 진짜 Colab VM 의 표식은 `/var/colab/hostname` 입니다 —
       구글 자신의 `drive.mount()` 도 이걸로 판정합니다.
    """
    # 1) Kaggle — 가장 확실한 신호부터
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE"):
        return "kaggle"
    if Path("/kaggle/working").is_dir() or Path("/kaggle/input").is_dir():
        return "kaggle"

    # 2) 진짜 Colab VM
    if Path("/var/colab/hostname").exists() or os.environ.get("COLAB_RELEASE_TAG"):
        return "colab"
    # 위 표식이 없는데 google.colab 이 **이미 로드돼 있으면** Colab 계열로 봅니다
    # (마운트는 못 할 수 있고, mount_drive 가 알아서 넘어갑니다)
    if "google.colab" in sys.modules and Path("/content").is_dir():
        return "colab"

    return "local"


def can_mount_drive() -> bool:
    """Drive 를 실제로 마운트할 수 있는 환경인가.

    구글의 `drive.mount()` 가 쓰는 표식과 같은 것을 봅니다. 이게 False 인데
    마운트를 시도하면 NotImplementedError 가 납니다.
    """
    return is_colab() and Path("/var/colab/hostname").exists()


def diagnose() -> dict:
    """환경 감지가 이상할 때 근거를 전부 찍습니다.

        env.diagnose()

    "Kaggle 인데 Colab 이라고 나온다" 같은 상황에서 뭘 보고 그렇게 판단했는지
    확인할 수 있습니다.
    """
    sig = {
        "detect()": detect(),
        "KAGGLE_KERNEL_RUN_TYPE": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
        "KAGGLE_URL_BASE": os.environ.get("KAGGLE_URL_BASE"),
        "COLAB_RELEASE_TAG": os.environ.get("COLAB_RELEASE_TAG"),
        "/kaggle/working 있음": Path("/kaggle/working").is_dir(),
        "/kaggle/input 있음": Path("/kaggle/input").is_dir(),
        "/content 있음": Path("/content").is_dir(),
        "/var/colab/hostname 있음": Path("/var/colab/hostname").exists(),
        "google.colab 로드됨": "google.colab" in sys.modules,
        "can_mount_drive()": can_mount_drive(),
        "workspace()": str(workspace()),
        "work_root()": str(work_root()),
        "persist_root()": str(persist_root() or "None"),
    }
    print("── 환경 감지 근거 " + "─" * 40)
    for k, v in sig.items():
        print(f"  {k:<26} {v}")
    print("─" * 58)
    return sig


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


def mount_drive(mountpoint: str = "/content/drive", strict: bool = False) -> Path | None:
    """Colab 에서만 Google Drive 를 마운트합니다. 다른 환경에서는 None.

    ⚠️ **실패해도 예외를 던지지 않습니다.** Drive 는 체크포인트를 세션 밖에
       남기기 위한 **편의 기능**이지, 학습의 전제가 아닙니다. 여기서 죽으면
       노트북 전체가 멈추는데, 정작 데이터가 다른 곳에 있으면 그냥 진행하면
       됩니다. 못 붙었다는 사실은 호출부(`persist_root()` 가 None)가 알립니다.

    실제로 겪은 경우: `/var/colab/hostname` 이 없는 Colab 계열 환경
    (로컬 런타임 / Colab Enterprise / 일부 프록시 세션)에서
    `NotImplementedError: Mounting drive is unsupported in this environment`.

    strict=True 로 주면 예외를 그대로 올립니다.
    """
    if not is_colab():
        print("[env] Colab 이 아니므로 Drive 마운트를 건너뜁니다.")
        return None
    try:
        from google.colab import drive  # type: ignore[import-not-found]

        drive.mount(mountpoint)
    except Exception as exc:                                    # noqa: BLE001
        if strict:
            raise
        print(f"⚠️ [env] Drive 마운트 실패 — {type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:120]}")
        print("   이 환경에서는 Drive 를 쓸 수 없습니다. **계속 진행할 수 있습니다.**")
        print("   다만 두 가지가 달라집니다:")
        print("     · 데이터를 Drive 에서 못 읽습니다 → zip 을 다른 경로에 두거나 Kaggle 사용")
        print("     · 체크포인트가 세션 밖에 안 남습니다 → 끊기면 학습을 처음부터")
        print("   체크포인트만 살리려면: os.environ['DOG_SKIN_PERSIST'] = '/어딘가/영구경로'")
        return None
    p = Path(mountpoint) / "MyDrive"
    return p if p.exists() else None


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


def require_gpu(hard: bool = True) -> Device:
    """GPU 가 붙어 있는지 확인하고, 없으면 **크게** 알립니다.

    왜 필요한가: GPU 가 없어도 코드는 그냥 돕니다 — 다만 20~30배 느립니다.
    `DEV = "cuda" if torch.cuda.is_available() else "cpu"` 는 조용히 CPU 로
    떨어지고, `suggest_batch_size` 도 8 을 돌려주기 때문에 겉보기엔 정상입니다.
    실측: 검증 7,751장에 GPU 1~2분 vs CPU **40분**. 학습은 며칠입니다.

    Colab 무료 티어는 GPU 사용량 한도를 넘기면 **말없이 CPU 런타임을 줍니다.**
    그래서 사람이 알아채기 전에 몇 시간을 버리게 됩니다.
    """
    d = device_info()
    if d.kind == "cuda":
        return d

    msg = (
        "\n" + "🚨" * 20 + "\n"
        "  GPU 가 없습니다 — 지금 학습/추론하면 20~30배 느립니다.\n"
        + "🚨" * 20 + "\n\n"
        "  실측 비교 (검증 7,751장):\n"
        "     GPU(T4) 1~2분   ↔   CPU 약 40분\n"
        "     학습은 CPU 로 며칠 걸립니다. 기다리지 마세요.\n\n"
        "  · Colab  : 런타임 → 런타임 유형 변경 → T4 GPU → 저장\n"
        "             이미 GPU 로 되어 있는데 이 메시지가 뜬다면\n"
        "             **무료 GPU 한도를 다 쓴 것**입니다 (보통 12~24시간 뒤 회복).\n"
        "             → Kaggle 로 옮기거나(주 30시간 별도) Colab Pro 를 보세요.\n"
        "  · Kaggle : 우측 Settings → Accelerator → GPU T4 x2\n\n"
        "  그래도 CPU 로 진행하려면: env.require_gpu(hard=False)\n"
    )
    if hard:
        raise RuntimeError(msg)
    print(msg)
    return d


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
        elif s.env in ("colab", "kaggle"):
            print(f" GPU         : 🚨 없음 ({d.kind})")
            print(" " * 15 + "학습·추론이 20~30배 느려집니다. 지금 멈추세요.")
            print(" " * 15 + "(런타임 유형이 이미 GPU 인데도 이러면 무료 한도 소진입니다)")
        else:
            # 로컬: 전처리·패키징은 CPU 로 하는 게 맞습니다
            print(f" GPU         : 없음 ({d.kind}) — 전처리·패키징에는 필요 없습니다")
        print(f" 여유 디스크 : {s.free_disk_gb} GB")
        print(f" 원본 데이터 : {s.paths['data_root']}")
        print(f" 작업 폴더   : {s.paths['work_root']}")
        print("─" * 58)
        if s.env == "colab" and s.free_disk_gb < 60:
            print("⚠️  디스크 여유가 적습니다. STEP 1 에서 부분 다운로드를 꼭 쓰세요.")
        # ⚠️ 로컬 PC 는 GPU 가 없는 게 정상입니다 (다운로드·전처리는 CPU 작업).
        #    거기서까지 경고하면 진짜 경고까지 같이 무시하게 됩니다.
        if d.kind == "cpu" and s.env in ("colab", "kaggle"):
            print("⚠️  GPU 런타임이 아닙니다 — 학습·추론이 20~30배 느립니다.")
            print("    Colab : 런타임 → 런타임 유형 변경 → T4 GPU")
            print("    Kaggle: 우측 Settings → Accelerator → GPU T4 x2")
    return s


def _search_roots() -> list[Path]:
    return [p for p in (Path("/kaggle/input"), Path("/content/drive/MyDrive"),
                        Path("/content"), Path.cwd()) if p.exists()]


def _looks_prepared(d: Path) -> bool:
    return (d / "crops").is_dir() and (d / "manifests").is_dir()


def _looks_partial(d: Path) -> bool:
    """`crops/` 만 있고 매니페스트는 없는 폴더 — 태그를 나눠 올린 경우."""
    return (d / "crops").is_dir() and not (d / "manifests").is_dir()


def find_prepared(dest: Path | None = None) -> tuple[Path, str]:
    """전처리 결과를 하나 찾습니다. `(경로, "zip" | "dir")`."""
    src, kind = find_prepared_all(dest)[0]
    return src, kind


def find_prepared_all(dest: Path | None = None) -> list[tuple[Path, str]]:
    """전처리 결과를 **전부** 찾습니다. `[(경로, "zip" | "dir"), ...]`.

    ⚠️ **Kaggle 은 업로드한 zip 을 데이터셋에 넣을 때 자동으로 풀어버립니다.**
       그래서 `/kaggle/input/<데이터셋>/crops/...` 만 있고 zip 은 없습니다.
       zip 만 찾으면 여기서 막힙니다 — 풀려 있는 폴더도 같이 봅니다.

    ⚠️ 전체 zip 은 5GB 를 넘습니다. 업로드가 자주 끊겨서 **태그별로 나눠**
       올리는 경우가 있습니다 (`crops/m1.5` 하나, `crops/full` 하나).
       그래서 하나만 찾고 멈추지 않고 다 모아서 합칩니다.
    """
    dest = Path(dest) if dest else work_root()
    zips: list[tuple[Path, str]] = []
    dirs: list[tuple[Path, str]] = []
    seen: set[Path] = set()

    for base in _search_roots():
        for d in _walk(base, depth=3):
            r = d.resolve()
            # zip 은 이 폴더 바로 아래만 봅니다 (rglob 은 심볼릭 링크로 안 들어갑니다)
            for z in sorted(d.glob("dogskin*.zip")):
                if z.resolve() not in seen:
                    seen.add(z.resolve())
                    zips.append((z, "zip"))
            if r in seen or r == dest.resolve():
                continue
            if _looks_prepared(d) or _looks_partial(d):
                seen.add(r)
                dirs.append((d, "dir"))

    out = dirs or zips              # 풀려 있는 쪽을 우선 (복사 없이 링크만 하면 됨)
    if not out:
        raise FileNotFoundError(
            "전처리 결과(dogskin_prepared)를 찾지 못했습니다.\n\n"
            + _what_is_there()
            + "\n찾는 것 (둘 중 하나):\n"
            "  · dogskin*.zip 파일\n"
            "  · crops/ 폴더를 가진 폴더 (Kaggle 은 zip 을 자동으로 풀어둡니다)\n\n"
            "확인할 것:\n"
            "  · Kaggle : 우측 패널 [Add Input] 으로 데이터셋을 붙였는지\n"
            "             (붙였으면 위 목록의 /kaggle/input 아래에 보여야 합니다)\n"
            "  · Colab  : Drive 를 마운트했는지 → env.mount_drive()\n"
            "  · 경로를 직접 주려면 env.load_prepared('/kaggle/input/<데이터셋이름>')"
        )
    return out


def _walk(base: Path, depth: int = 3) -> list[Path]:
    """base 아래를 depth 단계까지. **심볼릭 링크 폴더도 들어갑니다.**

    ⚠️ `Path.rglob` 은 심볼릭 링크인 하위 폴더로 들어가지 않습니다.
       Kaggle 의 `/kaggle/input/<데이터셋>` 이 링크면 rglob 으로는 안 보입니다.
       `iterdir()` + `is_dir()` 은 링크를 따라가므로 직접 훑습니다.
    """
    out = [base]
    frontier = [base]
    for _ in range(depth):
        nxt: list[Path] = []
        for d in frontier:
            try:
                for p in sorted(d.iterdir()):
                    if p.is_dir():
                        out.append(p)
                        nxt.append(p)
            except OSError:
                continue
        frontier = nxt
    return out


def _what_is_there(max_items: int = 12) -> str:
    """검색 경로에 **실제로 뭐가 있는지** 보여줍니다.

    "못 찾았다"만 알려주면 사용자가 다음에 뭘 볼지 알 수 없습니다.
    """
    lines = ["실제로 있는 것:"]
    for base in _search_roots():
        lines.append(f"  {base}")
        try:
            items = sorted(base.iterdir())
        except OSError as exc:
            lines.append(f"      (읽을 수 없음: {type(exc).__name__})")
            continue
        if not items:
            lines.append("      (비어 있음)")
            continue
        for p in items[:max_items]:
            if p.is_dir():
                try:
                    inner = sorted(x.name for x in p.iterdir())[:6]
                except OSError:
                    inner = ["(읽을 수 없음)"]
                lines.append(f"      📁 {p.name}/   → {inner}")
            else:
                mb = p.stat().st_size / 1024**2
                lines.append(f"      📄 {p.name}  ({mb:,.0f} MB)")
        if len(items) > max_items:
            lines.append(f"      … 외 {len(items) - max_items}개")
    return "\n".join(lines) + "\n"


def _link_tags(src_crops: Path, dst_crops: Path) -> dict[str, str]:
    """크롭을 **태그 단위로** 연결합니다. 여러 입력을 합칠 수 있습니다.

    Kaggle 의 `/kaggle/input` 은 읽기 전용이고, 크롭 45,885장을
    `/kaggle/working`(20GB 제한) 으로 복사하는 건 시간도 용량도 낭비입니다.
    폴더 통째로가 아니라 태그별로 링크해야 `crops/m1.5` 와 `crops/full` 을
    **서로 다른 데이터셋에서** 가져와 합칠 수 있습니다.
    """
    dst_crops.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    for tag_dir in sorted(p for p in src_crops.iterdir() if p.is_dir()):
        dst = dst_crops / tag_dir.name
        if dst.is_symlink():
            out[tag_dir.name] = ("이미 연결됨" if dst.resolve() == tag_dir.resolve()
                                 else "다른 곳에 연결됨(유지)")
            continue
        if dst.exists():
            out[tag_dir.name] = "이미 있음"
            continue
        try:
            dst.symlink_to(tag_dir.resolve(), target_is_directory=True)
            out[tag_dir.name] = "링크"
        except OSError:
            shutil.copytree(tag_dir, dst)
            out[tag_dir.name] = "복사"
    return out


def load_prepared(
    zip_path: str | Path | None = None,
    dest: Path | None = None,
    force: bool = False,
) -> Path:
    """로컬에서 만든 전처리 결과를 클라우드 작업 폴더에 붙입니다.

    한국 PC 에서 `prepare_local.py` 로 전처리한 결과를 Drive/Kaggle 에 올린 뒤,
    학습 노트북 첫 부분에서 이걸 부르면 됩니다. 두 가지 형태를 다 받습니다:

      · `dogskin_prepared.zip`  → 로컬 디스크로 풉니다 (Colab + Drive)
      · 이미 풀린 `crops/ manifests/` 폴더 → 링크만 겁니다 (Kaggle)

    ⚠️ zip 은 반드시 **로컬 디스크**로 풉니다. Drive 에 마운트된 채로 이미지를
       읽으면 네트워크 왕복 때문에 학습이 10배 가까이 느려집니다.
    """
    import zipfile

    dest = Path(dest) if dest else work_root()
    dest.mkdir(parents=True, exist_ok=True)

    if zip_path is None:
        sources = find_prepared_all(dest)
    elif isinstance(zip_path, (list, tuple)):
        sources = [(Path(p), "zip" if Path(p).suffix == ".zip" else "dir") for p in zip_path]
    else:
        p = Path(zip_path)
        sources = [(p, "zip" if p.suffix == ".zip" else "dir")]

    if len(sources) > 1:
        print(f"[env] 입력 {len(sources)}개를 합칩니다: {[s.name for s, _ in sources]}")

    marker = dest / ".prepared_from"
    seen_before = set(marker.read_text().splitlines()) if marker.exists() else set()
    now: list[str] = []

    for src, kind in sources:
        # 마커에 **여러 줄**로 적습니다. zip 을 두 개 올린 경우
        # 한 줄만 쓰면 재실행 때 앞의 zip 을 또 풉니다.
        already = str(src) in seen_before and not force and kind == "zip"

        if kind == "dir":
            # 읽기 전용일 수 있으므로 크롭은 태그별 링크, 매니페스트는 복사
            how = _link_tags(src / "crops", dest / "crops")
            for tag, act in how.items():
                print(f"[env] 크롭 {act}: {src.name}/crops/{tag}")
            man = dest / "manifests"
            if (src / "manifests").is_dir():
                if force and man.exists() and not man.is_symlink():
                    shutil.rmtree(man)
                if not man.exists():
                    shutil.copytree(src / "manifests", man)
        elif already:
            print(f"[env] 이미 풀려 있습니다: {dest}  (다시 풀려면 force=True)")
        else:
            size = src.stat().st_size / 1024**3
            print(f"[env] 압축 해제 {src.name} ({size:.2f}GB) → {dest}")
            print("      (Drive 에서 직접 읽지 않고 로컬 디스크로 풉니다 — 학습 속도 때문)")
            with zipfile.ZipFile(src) as z:
                z.extractall(dest)
        now.append(str(src))

    marker.write_text("\n".join(dict.fromkeys(now)))

    crops = dest / "crops"
    mans = sorted((dest / "manifests").glob("*.parquet")) if (dest / "manifests").exists() else []
    # ⚠️ `crops.rglob()` 은 **심볼릭 링크 하위 폴더로 들어가지 않습니다.**
    #    Kaggle 에서는 태그마다 링크를 걸므로 여기서 0장이 나옵니다.
    #    태그 폴더를 하나씩(= 링크 자체를 시작점으로) 훑어야 합니다.
    tags = sorted(p.name for p in crops.iterdir() if p.is_dir()) if crops.exists() else []
    n_crop = sum(sum(1 for _ in (crops / t).rglob("*.jpg")) for t in tags)

    print(f"[env] 크롭 {n_crop:,}장, 크롭 태그 {tags}")
    print(f"[env] 매니페스트 {[m.name for m in mans]}")
    if n_crop == 0:
        print("⚠️ 크롭이 하나도 없습니다. 데이터셋 내용을 확인하세요.")
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
