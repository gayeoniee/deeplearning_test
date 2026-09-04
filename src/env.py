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


def suggest_batch_size(img_size: int, model_scale: str = "base",
                       mem_factor: float = 1.0) -> int:
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
    # mem_factor: 백본별 보정 (ModelSpec.mem_factor). 이 공식은 ResNet 기준이라
    # 어텐션 계열의 활성값 메모리를 과소평가합니다 — 실측으로 0.4 를 씁니다.
    bs = 48 * (vram / 16) * scale_factor * px_factor * mem_factor

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
    """전처리 결과를 찾아볼 곳.

    ⚠️ `/workspace` 는 **RunPod 등 임대 GPU** 의 표준 마운트입니다. 이게 없어서
       런팟에서 `load_prepared()` 가 바로 FileNotFoundError 로 죽었습니다.
       `DOG_SKIN_PREPARED` 로 직접 지정할 수도 있습니다.
    """
    roots = []
    override = os.environ.get("DOG_SKIN_PREPARED")
    if override:
        roots.append(Path(override))
    roots += [Path("/kaggle/input"), Path("/content/drive/MyDrive"),
              Path("/content"), Path("/workspace"), Path.cwd()]
    seen, out = set(), []
    for p in roots:
        if p.exists() and p.resolve() not in seen:
            seen.add(p.resolve())
            out.append(p)
    return out


#: 우리가 쓰는 크롭 태그. `crops/` 층 없이 올라온 폴더를 알아볼 때만 씁니다
CROP_TAGS = ("m2.5", "m1.5", "f320", "full")


def _tag_has_jpg(t: Path) -> bool:
    try:
        return t.is_dir() and next(t.rglob("*.jpg"), None) is not None
    except OSError:
        return False


def _crops_dir(d: Path) -> Path | None:
    """`d` 안에서 **태그 폴더들을 담고 있는 폴더**를 찾습니다. 없으면 None.

    ⚠️ 캐글 데이터셋을 만들 때 `crops/` 층이 사라지는 일이 있습니다.
       `data/work/crops/m2.5` 를 그대로 올리면 데이터셋 안이
       `<데이터셋>/m2.5/00/....jpg` 가 됩니다 — `crops/` 가 없습니다.
       `d/crops` 만 보면 **붙여놓고도** "전처리 결과를 찾지 못했습니다" 로
       죽습니다 (실제로 당했습니다). 그래서 `d` 자신도 봅니다.

       단 `d` 자신을 볼 때는 **이름이 아는 태그인 것만** 인정합니다. 아무
       폴더나 태그로 받으면 남의 데이터셋이 크롭으로 잡힙니다.
    """
    c = d / "crops"
    try:
        if c.is_dir() and any(_tag_has_jpg(t) for t in c.iterdir()):
            return c
    except OSError:
        pass
    try:
        if d.is_dir() and any(t.name in CROP_TAGS and _tag_has_jpg(t) for t in d.iterdir()):
            return d
    except OSError:
        pass
    return None


def _has_crops(d: Path) -> bool:
    """크롭이 **한 장이라도** 들어 있는 태그 폴더가 있는가."""
    return _crops_dir(d) is not None


def _manifest_files(d: Path) -> list[Path]:
    """`d` **바로 아래**의 매니페스트 파일들 (parquet · csv)."""
    try:
        return sorted(p for p in d.iterdir()
                      if p.is_file() and not p.name.startswith(".")
                      and p.suffix in (".parquet", ".csv")
                      and "manifest" in p.name.lower())
    except OSError:
        return []


def _manifests_dir(d: Path, depth: int = 5) -> Path | None:
    """`d` 안에서 **매니페스트 파일이 실제로 들어 있는 폴더**를 찾습니다.

    ⚠️ `d / "manifests"` 로 못 박지 마세요. 캐글 데이터셋을 만들면 층이
       하나 늘거나(`manifests/manifests/*.parquet`,
       `manifests/data/work/manifests/*.parquet`) 아예 없어져
       (`<데이터셋>/manifest_final.parquet`) 있는 일이 실제로 있습니다.
       예전 코드는 `manifests/` **폴더만** 보고 "매니페스트 복사" 를 찍은 뒤
       0개를 복사했습니다 — 8분 뒤 다음 셀에서
       `FileNotFoundError: manifest_final.parquet` 로 죽었습니다.

    크롭 태그 폴더로는 안 내려갑니다 (36만 장짜리라 훑으면 몇 분 걸립니다).
    """
    skip = set(CROP_TAGS) | {"crops", "reports", "checkpoints",
                             "__pycache__", "lost+found"}
    frontier = [d / "manifests", d]
    seen: set[Path] = set()
    for _ in range(depth):
        nxt: list[Path] = []
        for p in frontier:
            if not p.is_dir():
                continue
            try:
                r = p.resolve()
            except OSError:
                continue
            if r in seen:
                continue
            seen.add(r)
            if _manifest_files(p):
                return p
            try:
                nxt += [c for c in sorted(p.iterdir())
                        if c.is_dir() and c.name not in skip
                        and not c.name.startswith(".")]
            except OSError:
                continue
        if not nxt:
            break
        frontier = nxt
    return None


def _has_manifest(d: Path) -> bool:
    return _manifests_dir(d) is not None


def _looks_prepared(d: Path) -> bool:
    """크롭과 매니페스트가 **내용까지** 있는가.

    ⚠️ 폴더 존재만 보면 안 됩니다. `ensure_dirs()` 가 `crops/` 와 `manifests/`
       를 **빈 폴더로 미리 만듭니다.** 노트북 첫 셀의 `env.describe()` 가 그걸
       부르므로, 존재만 보면 zip 을 받아놓고도 "이미 준비됨" 으로 건너뛰고
       나중에 `manifest_final.parquet` 을 못 찾아 죽습니다 (런팟에서 당했습니다).
    """
    return _has_crops(d) and _has_manifest(d)


def _looks_partial(d: Path) -> bool:
    """크롭만 있고 매니페스트는 없는 폴더 — 태그를 나눠 올린 경우."""
    return _has_crops(d) and not _has_manifest(d)


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
        for d in _walk(base, depth=5):
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
            "  · crops/ 폴더를 가진 폴더 (Kaggle 은 zip 을 자동으로 풀어둡니다)\n"
            "  · 또는 m2.5/f320/full/m1.5 를 바로 담은 폴더 (crops/ 층 없이 올린 데이터셋)\n\n"
            "확인할 것:\n"
            "  · Kaggle : 우측 패널 [Add Input] 으로 데이터셋을 붙였는지\n"
            "             (붙였으면 위 목록의 /kaggle/input 아래에 보여야 합니다)\n"
            "  · Colab  : Drive 를 마운트했는지 → env.mount_drive()\n"
            "  · 경로를 직접 주려면 env.load_prepared('/kaggle/input/<데이터셋이름>')"
        )
    return out


# 데이터가 들어 있을 리 없는데 크고 느린 폴더들 — 여기로 내려가면 탐색이 오래 걸립니다
_SKIP_DIRS = {"crops", "manifests", "reports", "checkpoints", ".git",
              "__pycache__", "node_modules", "site-packages", "lost+found"}


def _walk(base: Path, depth: int = 5, max_dirs: int = 3000) -> list[Path]:
    """base 아래를 depth 단계까지. **심볼릭 링크 폴더도 들어갑니다.**

    ⚠️ `Path.rglob` 은 심볼릭 링크인 하위 폴더로 들어가지 않습니다.
       Kaggle 의 `/kaggle/input/<데이터셋>` 이 링크면 rglob 으로는 안 보입니다.
       `iterdir()` + `is_dir()` 은 링크를 따라가므로 직접 훑습니다.

    ⚠️ Kaggle 의 데이터셋 경로가 생각보다 깊습니다. 실측:
           /kaggle/input/datasets/<사용자>/<데이터셋>/crops/...
       `/kaggle/input` 기준으로 **4단계**입니다. 얕게 보면 못 찾습니다.

    크롭 폴더(4만 장) 안으로는 내려가지 않습니다 — 거기엔 찾을 게 없고 느립니다.
    """
    out = [base]
    frontier = [base]
    for _ in range(depth):
        nxt: list[Path] = []
        for d in frontier:
            if len(out) >= max_dirs:
                return out
            try:
                for p in sorted(d.iterdir()):
                    if not p.is_dir() or p.name in _SKIP_DIRS or p.name.startswith("."):
                        continue
                    out.append(p)
                    # 여기가 이미 전처리 폴더면 더 내려갈 이유가 없습니다
                    if not (_looks_prepared(p) or _looks_partial(p)):
                        nxt.append(p)
            except OSError:
                continue
        if not nxt:
            break
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


def _peek(d: Path, depth: int = 3, max_items: int = 10) -> list[str]:
    """`d` 안을 몇 층까지 훑어 사람이 읽을 줄로 만듭니다 (진단용).

    크롭 태그 폴더로는 안 들어갑니다 — 36만 장을 세면 몇 분 걸립니다.
    """
    skip = set(CROP_TAGS) | {"crops"}
    out: list[str] = []

    def walk(p: Path, level: int, pad: str) -> None:
        if level > depth:
            return
        try:
            items = sorted(p.iterdir())
        except OSError as exc:
            out.append(f"{pad}(읽을 수 없음: {type(exc).__name__})")
            return
        if not items:
            out.append(f"{pad}(비어 있음)")
            return
        for q in items[:max_items]:
            if q.is_dir():
                mark = " …" if q.name in skip else "/"
                out.append(f"{pad}📁 {q.name}{mark}")
                if q.name not in skip:
                    walk(q, level + 1, pad + "   ")
            else:
                mb = q.stat().st_size / 1024**2
                out.append(f"{pad}📄 {q.name}  ({mb:,.1f} MB)")
        if len(items) > max_items:
            out.append(f"{pad}… 외 {len(items) - max_items}개")

    walk(d, 1, "")
    return out


def _link_tags(src_crops: Path, dst_crops: Path) -> dict[str, str]:
    """크롭을 **태그 단위로** 연결합니다. 여러 입력을 합칠 수 있습니다.

    Kaggle 의 `/kaggle/input` 은 읽기 전용이고, 크롭 45,885장을
    `/kaggle/working`(20GB 제한) 으로 복사하는 건 시간도 용량도 낭비입니다.
    폴더 통째로가 아니라 태그별로 링크해야 `crops/m1.5` 와 `crops/full` 을
    **서로 다른 데이터셋에서** 가져와 합칠 수 있습니다.
    """
    dst_crops.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    # ⚠️ `crops/` 층 없이 올라온 폴더(`_crops_dir` 이 폴더 자신을 돌려준 경우)에는
    #    `manifests/` 같은 형제 폴더가 같이 있습니다. 이름으로 걸러내지 않으면
    #    매니페스트가 **크롭 태그로 링크**됩니다.
    for tag_dir in sorted(p for p in src_crops.iterdir()
                          if p.is_dir() and p.name not in _SKIP_DIRS):
        # ⚠️ **빈 태그 폴더는 건너뜁니다.** 노트북 출력을 데이터셋으로 만들면
        #    work/crops/ 의 심볼릭 링크가 빈 폴더로 남는 일이 있습니다.
        #    먼저 연결된 쪽이 이기므로, 그 빈 폴더가 진짜 크롭 데이터셋을 가로막고
        #    "크롭이 0% 밖에 없습니다" 로 죽습니다. 이름 순서 운에 맡길 일이 아닙니다.
        if next(tag_dir.iterdir(), None) is None:
            out[tag_dir.name] = "비어 있어 건너뜀"
            continue
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

    # ★ 이미 work_root 에 크롭+매니페스트가 있으면 **할 일이 없습니다.**
    #    ⚠️ 이 검사는 find_prepared_all() **앞에** 있어야 합니다. 그 함수는
    #       dest 자신을 후보에서 빼기 때문에(아래 `r == dest.resolve()`),
    #       런팟처럼 데이터를 work_root 에 직접 넣은 경우 "못 찾았습니다" 로
    #       **먼저 죽습니다.** 뒤에 두면 이 줄에 영영 도달하지 못합니다.
    if zip_path is None and _looks_prepared(dest):
        n = sum(1 for _ in (dest / "crops").iterdir() if _.is_dir())
        print(f"[env] 이미 준비돼 있습니다: {dest}  (크롭 태그 {n}종) — 그대로 씁니다.")
        return dest

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
            # ⚠️ `src / "crops"` 로 못 박지 마세요 — 캐글 데이터셋은 `crops/` 층이
            #    빠진 채 올라올 수 있습니다 (`_crops_dir` 주석 참고).
            src_crops = _crops_dir(src) or (src / "crops")
            where = src.name if src_crops == src else f"{src.name}/crops"
            how = _link_tags(src_crops, dest / "crops")
            for tag, act in how.items():
                print(f"[env] 크롭 {act}: {where}/{tag}")
            # ⚠️ `ensure_dirs()` 가 work_root()/manifests 를 **빈 폴더로 미리 만듭니다.**
            #    "없으면 복사" 로 조건을 걸면 그 빈 폴더 때문에 영원히 복사가 안 되고,
            #    나중에 manifest_final.parquet 을 못 찾아 죽습니다. 비어 있으면 채웁니다.
            man = dest / "manifests"
            src_man = _manifests_dir(src)
            if src_man is not None:
                if force and man.exists() and not man.is_symlink():
                    shutil.rmtree(man)
                have = sorted(man.glob("*")) if man.exists() else []
                if not have:
                    man.mkdir(parents=True, exist_ok=True)
                    n = 0
                    for f in sorted(src_man.iterdir()):
                        if f.is_file():
                            shutil.copy2(f, man / f.name)
                            n += 1
                    # ⚠️ 몇 개를 복사했는지 **반드시 찍습니다.** 예전엔 개수 없이
                    #    "복사" 만 찍어서 0개를 복사하고도 정상처럼 보였습니다.
                    tail = src_man.name if src_man != src else "(최상위)"
                    print(f"[env] 매니페스트 {n}개 복사: {src.name}/{tail} → {man}")
            else:
                print(f"[env] {src.name} 안에 매니페스트 파일이 없습니다 (크롭만 있는 입력)")
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

    # ★ 매니페스트가 없으면 **여기서 멈춥니다.** 크롭 세기 전에요.
    #
    # ⚠️ 예전에는 경고만 찍고 진행했습니다. 크롭 36만 장을 세는 데 13분이
    #    걸리고, 그 뒤 다음 셀이 `FileNotFoundError: manifest_final.parquet` 로
    #    죽었습니다 — 20분을 버리고, 정작 원인을 알려주는 줄은 스크롤 위로
    #    밀려 올라가 보이지도 않았습니다. 실제로 두 번 당했습니다.
    #    **예외 메시지에 데이터셋 안을 같이 넣습니다** — 사람들이 복사해 오는
    #    건 로그 꼬리(=트레이스백)뿐이라, 거기 없으면 전달되지 않습니다.
    if not any("final" in m.name for m in mans):
        found = [m.name for m in mans] or ["(하나도 없음)"]
        lines = [
            "manifest_final.parquet 이 없습니다. (크롭은 붙었는데 라벨이 없습니다)",
            "",
            f"작업 폴더 {dest / 'manifests'} 에 있는 것: {found}",
            "",
            "붙인 입력 안을 열어봤습니다 ↓ — 여기에 parquet 이 안 보이면",
            "데이터셋에 매니페스트가 **안 들어간 것**입니다:",
        ]
        for s, kind in sources:
            lines.append(f"  ── {s}  ({kind}) ──")
            if kind == "dir":
                lines += [f"     {t}" for t in _peek(s)]
            else:
                lines.append("     (zip)")
        lines += [
            "",
            "할 일 — 둘 중 하나:",
            "  · manifest_final.parquet 을 Private 데이터셋으로 따로 올려",
            "    [Add Input] 으로 붙이세요. 크롭 데이터셋은 그대로 두면 됩니다",
            "    (매니페스트는 100MB 안팎이라 몇 분이면 올라갑니다)",
            "  · 이미 붙였는데 위 목록에 안 보이면 업로드가 덜 끝난 것입니다",
        ]
        raise FileNotFoundError("\n".join(lines))
    # ⚠️ `crops.rglob()` 은 **심볼릭 링크 하위 폴더로 들어가지 않습니다.**
    #    Kaggle 에서는 태그마다 링크를 걸므로 여기서 0장이 나옵니다.
    #    태그 폴더를 하나씩(= 링크 자체를 시작점으로) 훑어야 합니다.
    tags = sorted(p.name for p in crops.iterdir() if p.is_dir()) if crops.exists() else []

    # ⚠️ 태그별로 세어 보여줍니다. 하나만 덜 올라간 경우가 실제로 있었습니다
    #    (full 30% / m1.5 100%) — 합계만 보면 안 보입니다.
    #
    # ⚠️ **한 번만 훑습니다.** 예전에는 합계용으로 한 번, 태그별로 또 한 번 훑어서
    #    같은 일을 두 번 했습니다. Kaggle 입력은 네트워크 마운트라 9만 장을 세는 데
    #    8분 30초가 걸렸고, 그동안 아무 출력이 없어 멈춘 것처럼 보였습니다.
    per_tag: dict[str, int] = {}
    for t in tags:
        print(f"[env] 크롭 세는 중 … {t}", flush=True)
        per_tag[t] = sum(1 for _ in (crops / t).rglob("*.jpg"))
        print(f"[env]   {t}: {per_tag[t]:,}장", flush=True)
    n_crop = sum(per_tag.values())
    print(f"[env] 크롭 {n_crop:,}장, 태그별 {per_tag}")
    if len(per_tag) > 1:
        hi = max(per_tag.values())
        short = {t: n for t, n in per_tag.items() if n < hi * 0.95}
        if short:
            print(f"🚨 태그마다 장수가 다릅니다 — {short} (가장 많은 태그: {hi:,}장)")
            print("   업로드가 덜 끝났을 가능성이 큽니다. 이대로 쓰면 그 태그를 쓰는")
            print("   단계만 **부분 데이터**로 학습되고, 숫자를 다른 실행과 비교할 수 없습니다.")
    print(f"[env] 매니페스트 {[m.name for m in mans]}")
    if n_crop == 0:
        print("⚠️ 크롭이 하나도 없습니다. 데이터셋 내용을 확인하세요.")
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
