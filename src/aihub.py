"""aihubshell 래퍼 — 필요한 부분만 골라 받기.

AI Hub 561번(반려동물 피부 질환)은 50만장이 넘습니다. 전부 받으면 Colab 디스크가
터지므로 `-mode l` 로 파일 목록을 먼저 보고, "반려견 + 일반카메라"에 해당하는
filekey 만 골라 받습니다.

    from src import aihub, env
    key = env.secret("AIHUB_API_KEY")
    aihub.install()
    listing = aihub.list_files(key)                  # 전체 목록 파싱
    picks   = aihub.select_files(listing)            # 반려견+일반카메라만
    aihub.download(key, picks, dest=env.data_root()) # 청크 단위 다운로드

⚠️ API 키는 절대 인자로 하드코딩하지 말고 env.secret() 로만 꺼내세요.
⚠️ 받은 데이터는 재배포 금지입니다. .gitignore 가 data/ 를 막고 있습니다.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src import env
from src.config import (
    AIHUB_DATASET_KEY,
    EXCLUDE_CAMERA,
    EXCLUDE_SPECIES,
    INCLUDE_CAMERA,
    INCLUDE_SPECIES,
)

AIHUBSHELL_URL = "https://api.aihub.or.kr/api/aihubshell.do"

# ──────────────────────────────────────────────────────────────
# 561 데이터셋의 실제 파일 구성 (2026-08 확인)
#
# ⚠️ 파일이 6개 대용량 zip 으로만 나뉘어 있습니다.
#    "반려견+일반카메라만" 같은 선별 다운로드가 **파일 단위로는 불가능**합니다.
#    → Validation 만 받아서 우리가 직접 개체 단위로 재분할하는 편이 낫습니다.
#      (어차피 AI Hub 의 Training/Validation 구분은 개체 누수를 보장하지 않으므로
#       그대로 쓰면 안 되고, 우리가 split.py 로 다시 나눕니다)
# ──────────────────────────────────────────────────────────────
KNOWN_FILES_561: list[dict] = [
    {"filekey": "517021", "name": "VS01.zip", "gb": 21, "split": "Validation", "kind": "원천"},
    {"filekey": "517022", "name": "VL01.zip", "gb": 21, "split": "Validation", "kind": "라벨"},
    {"filekey": "517017", "name": "TS01.zip", "gb": 90, "split": "Training", "kind": "원천"},
    {"filekey": "517018", "name": "TS02.zip", "gb": 80, "split": "Training", "kind": "원천"},
    {"filekey": "517019", "name": "TL01.zip", "gb": 90, "split": "Training", "kind": "라벨"},
    {"filekey": "517020", "name": "TL02.zip", "gb": 80, "split": "Training", "kind": "라벨"},
]


# ──────────────────────────────────────────────────────────────
# 설치
# ──────────────────────────────────────────────────────────────
def shell_path() -> Path:
    return env.workspace() / "aihubshell"


def install(force: bool = False) -> Path:
    """aihubshell 을 내려받아 실행 권한을 줍니다."""
    p = shell_path()
    if p.exists() and not force:
        print(f"[aihub] 이미 설치됨: {p}")
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    print(f"[aihub] 내려받는 중: {AIHUBSHELL_URL}")
    r = subprocess.run(
        ["curl", "-sSL", "-o", str(p), AIHUBSHELL_URL],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or not p.exists() or p.stat().st_size < 100:
        raise RuntimeError(
            "aihubshell 다운로드 실패.\n"
            f"  stderr: {r.stderr[:500]}\n"
            "  네트워크가 막혀 있거나 AI Hub 점검 중일 수 있습니다."
        )
    p.chmod(0o755)
    print(f"[aihub] 설치 완료: {p} ({p.stat().st_size} bytes)")
    return p


# aihubshell 은 실패해도 종료 코드 0 을 돌려줍니다. 출력 본문을 봐야 합니다.
_ERROR_PATTERNS: list[tuple[str, str]] = [
    ("해외에서의 데이터 다운로드를 제한",
     "AI Hub 가 해외 IP 다운로드를 차단합니다. Colab/Kaggle VM 은 한국 밖에 있어\n"
     "     클라우드에서는 받을 수 없습니다. → docs/cautions/06_해외IP_다운로드_차단_우회.md"),
    ("Download failed with HTTP status", "다운로드 실패 (HTTP 오류)"),
    ("승인", "활용신청이 아직 승인되지 않았을 수 있습니다"),
    ("Invalid", "API Key 가 올바르지 않습니다"),
    ("권한", "이 데이터셋에 대한 권한이 없습니다"),
]


def _detect_error(output: str) -> str | None:
    """aihubshell 출력에서 실패 신호를 찾습니다."""
    for pat, msg in _ERROR_PATTERNS:
        if pat in output:
            return msg
    return None


def _run(args: list[str], apikey: str, timeout: int | None = None,
         cwd: Path | None = None, stream: bool = False) -> subprocess.CompletedProcess:
    """aihubshell 실행. 키는 인자로만 넘기고 로그에는 남기지 않습니다."""
    cmd = [str(shell_path()), *args, "-aihubapikey", apikey]
    # 로그에 키가 남지 않도록 마지막 인자(키)만 가립니다.
    print("[aihub] $ " + " ".join(cmd[:-1] + ["****"]))
    if stream:
        # 다운로드처럼 오래 걸리는 작업은 출력을 흘려보냅니다.
        proc = subprocess.Popen(
            cmd, cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        lines: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)
            sys.stdout.write(line)
        proc.wait(timeout=timeout)
        return subprocess.CompletedProcess(cmd, proc.returncode, "".join(lines), "")
    return subprocess.run(
        cmd, cwd=str(cwd) if cwd else None,
        capture_output=True, text=True, timeout=timeout,
    )


# ──────────────────────────────────────────────────────────────
# 파일 목록 파싱
# ──────────────────────────────────────────────────────────────
@dataclass
class RemoteFile:
    filekey: str
    name: str
    size: str
    path: str = ""      # 목록에서 추정한 폴더 경로 (있으면)

    @property
    def size_gb(self) -> float:
        m = re.search(r"([\d.]+)\s*([KMGT]?B)", self.size, re.I)
        if not m:
            return 0.0
        v, unit = float(m.group(1)), m.group(2).upper()
        return v * {"B": 1e-9, "KB": 1e-6, "MB": 1e-3, "GB": 1.0, "TB": 1e3}.get(unit, 0.0)

    @property
    def full(self) -> str:
        return f"{self.path}/{self.name}" if self.path else self.name


def raw_listing(apikey: str, dataset_key: str = AIHUB_DATASET_KEY) -> str:
    """`aihubshell -mode l` 원본 출력. 파싱이 실패하면 이걸 눈으로 보세요."""
    r = _run(["-mode", "l", "-datasetkey", dataset_key], apikey, timeout=180)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0 and not out.strip():
        raise RuntimeError(f"목록 조회 실패 (rc={r.returncode}). 활용신청 승인 여부와 API 키를 확인하세요.")
    return out


def parse_listing(text: str) -> list[RemoteFile]:
    """`-mode l` 출력에서 [파일명 | 용량 | filekey] 를 뽑아냅니다.

    AI Hub 출력 포맷은 공지 없이 바뀔 수 있으므로, 파싱이 0건이면
    raw_listing() 을 그대로 보고 select_files() 를 손으로 채우세요.
    """
    files: list[RemoteFile] = []
    cur_path = ""

    for line in text.splitlines():
        stripped = line.rstrip()
        if not stripped.strip():
            continue

        # 파이프로 구분된 데이터 행: "... 파일명.zip | 1.2 GB | 51937"
        if stripped.count("|") >= 2:
            parts = [p.strip() for p in stripped.split("|")]
            key = parts[-1]
            if re.fullmatch(r"\d{2,}", key):
                name = re.sub(r"^[\s│├└─|+\\-]*", "", parts[-3] if len(parts) >= 3 else parts[0])
                files.append(RemoteFile(filekey=key, name=name, size=parts[-2], path=cur_path))
                continue

        # 트리 형태의 폴더 행 (파일키가 없는 줄)
        if not re.search(r"\|\s*\d{2,}\s*$", stripped):
            folder = re.sub(r"^[\s│├└─|+\\-]*", "", stripped).strip()
            # 우리가 아는 축(반려견/피부/일반카메라/유증상/A1..)이 보이면 경로로 기억
            if folder and len(folder) < 120 and not folder.lower().startswith(("total", "sum", "---")):
                cur_path = folder

    return files


def list_files(apikey: str, dataset_key: str = AIHUB_DATASET_KEY,
               show: bool = True, detail: bool = True) -> list[RemoteFile]:
    text = raw_listing(apikey, dataset_key)
    files = parse_listing(text)
    if not show:
        return files

    if not files:
        print("⚠️ 파싱 결과가 0건입니다. 아래 원본 출력을 확인하고 filekey 를 직접 지정하세요.\n")
        print(text[:4000])
        return files

    total = sum(f.size_gb for f in files)
    print(f"[aihub] 총 {len(files)}개 파일, 합계 약 {total:.1f} GB")
    if detail:
        print(f"[aihub] 여유 디스크 {env.free_disk_gb()} GB\n")
        print(f"  {'filekey':>9}  {'용량':>10}   경로/파일명")
        print("  " + "-" * 76)
        for f in files:
            print(f"  {f.filekey:>9}  {f.size:>10}   {f.full[:60]}")
        # 파일이 몇 개 안 되면 통짜 묶음일 가능성이 큽니다 → 선별 다운로드가 불가능
        if len(files) <= 10 and total > 100:
            print(
                "\n  ⚠️ 파일이 소수의 대용량 묶음으로만 나뉘어 있습니다.\n"
                "     '반려견+일반카메라만 골라 받기'가 파일 단위로는 불가능할 수 있습니다.\n"
                "     아래로 원본 목록을 확인해 하위 항목이 더 있는지 보세요:\n"
                "        print(aihub.raw_listing(APIKEY))"
            )
    return files


# ──────────────────────────────────────────────────────────────
# 필요한 것만 고르기
# ──────────────────────────────────────────────────────────────
def select_files(
    files: list[RemoteFile],
    include_species: list[str] | None = None,
    exclude_species: list[str] | None = None,
    include_camera: list[str] | None = None,
    exclude_camera: list[str] | None = None,
    include_labels_only: bool = False,
    max_gb: float | None = None,
    verbose: bool = True,
) -> list[RemoteFile]:
    """반려견 + 일반카메라에 해당하는 파일만 남깁니다.

    max_gb 를 주면 그 용량을 넘지 않게 앞에서부터 자릅니다
    (Colab 디스크가 작을 때 나눠 받기 위함).
    """
    inc_s = include_species if include_species is not None else INCLUDE_SPECIES
    exc_s = exclude_species if exclude_species is not None else EXCLUDE_SPECIES
    inc_c = include_camera if include_camera is not None else INCLUDE_CAMERA
    exc_c = exclude_camera if exclude_camera is not None else EXCLUDE_CAMERA

    def keep(f: RemoteFile) -> bool:
        hay = f.full
        if any(x in hay for x in exc_s) or any(x in hay for x in exc_c):
            return False
        # 포함 키워드가 경로/파일명 어디에도 안 보이면, 판단 불가로 보고 남깁니다.
        # (파일명이 압축 단위라 종/카메라가 안 드러나는 경우가 있음)
        s_ok = (not inc_s) or any(x in hay for x in inc_s) or not any(
            x in hay for x in inc_s + exc_s
        )
        c_ok = (not inc_c) or any(x in hay for x in inc_c) or not any(
            x in hay for x in inc_c + exc_c
        )
        if include_labels_only and "라벨" not in hay and "TL" not in hay and "VL" not in hay:
            return False
        return s_ok and c_ok

    picked = [f for f in files if keep(f)]

    if max_gb is not None:
        acc, capped = 0.0, []
        for f in picked:
            if acc + f.size_gb > max_gb:
                break
            capped.append(f)
            acc += f.size_gb
        if verbose and len(capped) < len(picked):
            print(f"[aihub] max_gb={max_gb} 제한으로 {len(picked)}개 중 {len(capped)}개만 선택")
        picked = capped

    if verbose:
        total = sum(f.size_gb for f in picked)
        print(f"[aihub] 선택: {len(picked)}개 / 약 {total:.1f} GB")
        print(f"[aihub] 여유 디스크: {env.free_disk_gb()} GB")
        if total > env.free_disk_gb() * 0.8:
            print("⚠️ 디스크가 부족할 수 있습니다. max_gb 를 줄여 나눠 받으세요.")
        for f in picked[:20]:
            print(f"    {f.filekey:>8}  {f.size:>10}  {f.full}")
        if len(picked) > 20:
            print(f"    ... 외 {len(picked) - 20}개")
    return picked


def recommend_plan(free_gb: float | None = None) -> list[dict]:
    """디스크 여유에 맞춰 어떤 순서로 받을지 제안합니다.

    핵심 판단: 라벨 zip 과 원천 zip 의 크기가 정확히 같아서(90=90, 80=80, 21=21)
    라벨 zip 안에 이미지가 함께 들어있을 가능성이 높습니다.
    그래서 **VL01(라벨) 하나를 먼저 받아 내용물을 확인**하는 것이 가장 저렴한 수순입니다.
    """
    free = free_gb if free_gb is not None else env.free_disk_gb()
    print("=" * 68)
    print(" AI Hub 561 다운로드 전략")
    print("=" * 68)
    print(f" 여유 디스크: {free:.1f} GB\n")
    print(" 파일 구성 (6개 통짜 zip, 총 382GB) — 선별 다운로드 불가")
    print(f"   {'filekey':>8}  {'파일':<10}{'용량':>6}  {'구분'}")
    for f in KNOWN_FILES_561:
        print(f"   {f['filekey']:>8}  {f['name']:<10}{f['gb']:>4}GB  {f['split']} / {f['kind']}")

    print("\n 권장 순서")
    print("   1단계  VL01.zip (517022, 21GB)  ← 라벨. 먼저 이것만 받아 내용 확인")
    print("            · 라벨 zip 이 원천 zip 과 크기가 같음 → 이미지가 함께 들어있을 수 있음")
    print("            · 들어있다면 VS01 은 안 받아도 됩니다")
    print("            · JSON 만이라면 2단계로 VS01 을 받습니다")
    print("   2단계  VS01.zip (517021, 21GB)  ← 이미지. 1단계 결과에 따라 결정")
    print("   3단계  Training 은 받지 않습니다 — 물리적으로 불가능합니다")

    # "라벨만 받으면 Training 도 되지 않나?" — 자주 나오는 질문이라 숫자로 답해둡니다.
    label_all = sum(f["gb"] for f in KNOWN_FILES_561 if f["kind"] == "라벨")
    tl01 = next(f["gb"] for f in KNOWN_FILES_561 if f["name"] == "TL01.zip")
    print(f"            · 라벨 zip 전체(TL01+TL02+VL01) = {label_all}GB > 여유 {free:.0f}GB")
    print(f"            · TL01 하나만 해도 {tl01}GB 로 이미 초과")
    print("            · zip 을 통째로 받은 뒤 풀어야 해서 쪼개 받는 것도 불가")
    print(f"            · 코랩에서 다룰 수 있는 라벨 파일은 VL01({21}GB) 뿐입니다")

    # 규모 감각: 전체 382GB / 이미지 50만 장 ≈ 장당 0.76MB
    est = int(21 * 1024 / 0.76)
    print(f"\n   그래도 충분한 이유")
    print(f"            · VL01 21GB ≈ 이미지 약 {est:,}장 (전체 평균 장당 0.76MB 기준)")
    print("            · 반려견+일반카메라 필터 후 대략 1만~1.3만 장 예상")
    print("            · 전이학습에는 충분한 규모 (밑바닥 학습이 아님)")
    print("            · 'Validation' 이라는 이름은 무시하세요. 우리는 이걸 전체 데이터로 보고")
    print("              split.py 로 train/val/holdout 을 개체 단위로 다시 나눕니다")

    if free < 45:
        print(f"\n ⚠️ 여유 {free:.0f}GB 로는 VL01+VS01 동시 보관이 빠듯합니다.")
        print("    1단계 결과를 보고 크롭 후 원본을 지우는 방식으로 진행하세요.")
    print("=" * 68)
    return KNOWN_FILES_561


def peek(root: Path | None = None, n: int = 30) -> dict:
    """받은 zip 을 푼 뒤, 안에 무엇이 들어있는지 빠르게 확인합니다.

    라벨 zip 에 이미지가 함께 들어있는지 판단하는 용도입니다.
    """
    root = root or env.data_root()
    ext: Counter = Counter()
    samples: list[str] = []
    total = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        ext[p.suffix.lower()] += 1
        total += p.stat().st_size
        if len(samples) < n:
            samples.append(str(p.relative_to(root)))

    print(f"[aihub] {root}")
    print(f"  총 {sum(ext.values()):,}개 파일 / {total / 1024**3:.2f} GB\n")
    print("  확장자별 개수")
    for e, c in ext.most_common(10):
        print(f"    {e or '(없음)':<8} {c:>10,}")

    imgs = sum(c for e, c in ext.items() if e in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    jsons = ext.get(".json", 0)
    print()
    if imgs and jsons:
        print(f"  ✅ 이미지({imgs:,})와 JSON({jsons:,})이 함께 있습니다 → 이 zip 만으로 진행 가능")
    elif jsons and not imgs:
        print(f"  ℹ️ JSON({jsons:,})만 있습니다 → 이미지가 든 원천 zip 을 추가로 받아야 합니다")
    elif imgs and not jsons:
        print(f"  ℹ️ 이미지({imgs:,})만 있습니다 → 라벨 zip 을 추가로 받아야 합니다")

    print("\n  경로 샘플")
    for s in samples[:12]:
        print(f"    {s}")
    return {"ext": dict(ext), "images": imgs, "jsons": jsons,
            "size_gb": round(total / 1024**3, 2), "samples": samples}


# ──────────────────────────────────────────────────────────────
# 다운로드
# ──────────────────────────────────────────────────────────────
def download(
    apikey: str,
    files: list[RemoteFile] | list[str],
    dest: Path | None = None,
    dataset_key: str = AIHUB_DATASET_KEY,
    chunk: int = 1,
    skip_existing: bool = True,
    min_free_gb: float = 5.0,
) -> list[str]:
    """filekey 들을 청크 단위로 내려받습니다.

    한 번에 다 받지 않고 chunk 개씩 끊는 이유: 중간에 세션이 끊겨도
    이미 받은 것은 남고, 다시 호출하면 skip_existing 으로 건너뜁니다.

    반환값: 실패한 filekey 목록 (비어 있으면 전부 성공)
    """
    dest = dest or env.data_root()
    dest.mkdir(parents=True, exist_ok=True)

    keys = [f if isinstance(f, str) else f.filekey for f in files]
    done_marker = dest / ".downloaded_keys"
    already: set[str] = set()
    if skip_existing and done_marker.exists():
        already = set(done_marker.read_text(encoding="utf-8").split())
        if already:
            print(f"[aihub] 이미 받은 {len(already)}개는 건너뜁니다.")

    todo = [k for k in keys if k not in already]
    failed: list[str] = []

    # 다운로드 전 용량 점검: zip 을 받은 뒤 압축을 풀므로 잠깐 2배가 필요합니다.
    known = {f["filekey"]: f for f in KNOWN_FILES_561}
    need = sum(known[k]["gb"] for k in todo if k in known)
    if need:
        free_now = env.free_disk_gb(dest)
        print(f"[aihub] 받을 용량 {need}GB, 압축 해제까지 순간 최대 약 {need * 2}GB 필요 "
              f"(여유 {free_now:.1f}GB)")
        if need * 2 > free_now:
            print("  ⚠️ 여유 공간이 부족할 수 있습니다. 한 번에 하나씩 받고,")
            print("     전처리(크롭) 후 원본을 지우며 진행하세요.")

    for i in range(0, len(todo), chunk):
        batch = todo[i : i + chunk]
        free = env.free_disk_gb(dest)
        if free < min_free_gb:
            print(f"⚠️ 여유 디스크 {free}GB < {min_free_gb}GB — 중단합니다. "
                  "전처리로 용량을 줄이거나 Drive 로 옮긴 뒤 다시 실행하세요.")
            failed.extend(todo[i:])
            break

        print(f"\n[aihub] ({i + len(batch)}/{len(todo)}) filekey={','.join(batch)}  여유 {free}GB")
        before = _dir_size_gb(dest)
        r = _run(
            ["-mode", "d", "-datasetkey", dataset_key, "-filekey", ",".join(batch)],
            apikey, cwd=dest, stream=True, timeout=None,
        )
        # ⚠️ aihubshell 은 실패해도 rc=0 을 돌려줍니다.
        #    종료 코드만 믿으면 실패를 성공이라고 보고하게 됩니다.
        err = _detect_error(r.stdout or "")
        gained = _dir_size_gb(dest) - before

        if r.returncode != 0 or err or gained < 0.01:
            reason = err or (f"rc={r.returncode}" if r.returncode else
                             f"받아진 데이터가 없음 ({gained:.2f}GB)")
            print(f"\n  ✗ 실패 — {reason}")
            failed.extend(batch)
            continue

        print(f"  ✓ {gained:.1f}GB 확보")
        already.update(batch)
        done_marker.write_text("\n".join(sorted(already)), encoding="utf-8")

    if failed:
        print(f"\n❌ 다운로드 실패 {len(failed)}건: {failed}")
        print("   해결되기 전까지 다음 단계로 넘어가지 마세요.")
    else:
        print(f"\n✅ 다운로드 완료 — {_dir_size_gb(dest):.1f}GB @ {dest}")
    return failed


def _dir_size_gb(path: Path) -> float:
    """디렉터리 실제 사용량. 다운로드가 정말 됐는지 확인하는 데 씁니다."""
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total / 1024**3


def unpack_all(root: Path | None = None, remove_archives: bool = True) -> int:
    """aihubshell 이 자동 해제하지 못한 zip/tar 를 마저 풉니다."""
    root = root or env.data_root()
    n = 0
    for arc in sorted(root.rglob("*")):
        if arc.suffix.lower() not in {".zip", ".tar", ".gz", ".tgz"}:
            continue
        try:
            shutil.unpack_archive(str(arc), str(arc.parent))
            n += 1
            if remove_archives:
                arc.unlink()
        except Exception as exc:
            print(f"  ✗ {arc.name}: {exc}")
    print(f"[aihub] 압축 해제 {n}건")
    return n


def verify(root: Path | None = None) -> dict:
    """받은 데이터의 대략적인 규모를 확인합니다."""
    root = root or env.data_root()
    imgs = jsons = 0
    total_bytes = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        total_bytes += p.stat().st_size
        s = p.suffix.lower()
        if s in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            imgs += 1
        elif s == ".json":
            jsons += 1
    out = {
        "root": str(root),
        "images": imgs,
        "jsons": jsons,
        "size_gb": round(total_bytes / 1024**3, 2),
    }
    print(f"[aihub] 이미지 {imgs:,}장 / JSON {jsons:,}개 / {out['size_gb']} GB  @ {root}")
    if imgs == 0:
        print("⚠️ 이미지가 0장입니다. 압축이 안 풀렸거나 경로가 다를 수 있습니다 → unpack_all() 실행")
    return out
