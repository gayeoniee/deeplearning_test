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


def _run(args: list[str], apikey: str, timeout: int | None = None,
         cwd: Path | None = None, stream: bool = False) -> subprocess.CompletedProcess:
    """aihubshell 실행. 키는 인자로만 넘기고 로그에는 남기지 않습니다."""
    cmd = [str(shell_path()), *args, "-aihubapikey", apikey]
    safe = " ".join(cmd[:-1]) + " -aihubapikey ****"
    print(f"[aihub] $ {safe}")
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
               show: bool = True) -> list[RemoteFile]:
    text = raw_listing(apikey, dataset_key)
    files = parse_listing(text)
    if show:
        if not files:
            print("⚠️ 파싱 결과가 0건입니다. 아래 원본 출력을 확인하고 filekey 를 직접 지정하세요.\n")
            print(text[:4000])
        else:
            print(f"[aihub] 총 {len(files)}개 파일, 합계 약 {sum(f.size_gb for f in files):.1f} GB")
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

    for i in range(0, len(todo), chunk):
        batch = todo[i : i + chunk]
        free = env.free_disk_gb(dest)
        if free < min_free_gb:
            print(f"⚠️ 여유 디스크 {free}GB < {min_free_gb}GB — 중단합니다. "
                  "전처리로 용량을 줄이거나 Drive 로 옮긴 뒤 다시 실행하세요.")
            failed.extend(todo[i:])
            break

        print(f"\n[aihub] ({i + len(batch)}/{len(todo)}) filekey={','.join(batch)}  여유 {free}GB")
        r = _run(
            ["-mode", "d", "-datasetkey", dataset_key, "-filekey", ",".join(batch)],
            apikey, cwd=dest, stream=True, timeout=None,
        )
        if r.returncode != 0:
            print(f"  ✗ 실패 (rc={r.returncode})")
            failed.extend(batch)
            continue

        already.update(batch)
        done_marker.write_text("\n".join(sorted(already)), encoding="utf-8")

    if failed:
        print(f"\n⚠️ 실패 {len(failed)}건: {failed}\n   같은 셀을 다시 실행하면 성공분은 건너뛰고 재시도합니다.")
    else:
        print("\n✅ 다운로드 완료")
    return failed


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
