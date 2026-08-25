"""★ 스크리닝 API 서버 + 데모 화면.

안드로이드 앱(DAENGS_APP)이 붙을 자리입니다. 로직은 `src/agent.py` 에 있고
여기는 HTTP 만 봅니다.

    # 가중치 없이 화면만 (torch 불필요)
    uv run --extra serve python serve.py --mock

    # 진짜 모델로
    uv run --extra train --extra serve python serve.py \
        --ckpt1 runs/stage1_.../best.pt \
        --ckpt2 runs/stage2_.../best.pt

    http://127.0.0.1:8000/          데모 화면
    http://127.0.0.1:8000/docs      API 문서 (FastAPI 자동 생성)

엔드포인트:
    GET  /healthz          살아있나 + 어떤 가중치를 물고 있나
    POST /v1/screen        multipart 사진 한 장 → 판정 JSON

⚠️ 이건 **데모 서버**입니다. 인증·업로드 크기 제한·레이트 리밋·HTTPS 가 없습니다.
   공개 주소에 그대로 띄우지 마세요. 사진은 디스크에 저장하지 않고 메모리에서
   처리한 뒤 버립니다 (보호자 사진을 동의 없이 모으지 않기 — cautions/03 §8).
"""

# ⚠️ `from __future__ import annotations` 를 넣지 마세요.
# 그러면 UploadFile 이 문자열 어노테이션이 되는데, 라우트 함수가 build_app 안에
# 정의돼 있어 pydantic 이 모듈 전역에서 그 이름을 못 찾고 500 으로 죽습니다.
import argparse
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

MAX_BYTES = 12 * 1024 * 1024          # 휴대폰 사진 한 장이면 충분합니다


def build_app(agent, mock: bool):
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse, JSONResponse
    from PIL import Image

    from src.agent import CONTRACT_VERSION

    app = FastAPI(
        title="반려견 피부 스크리닝 API",
        version=CONTRACT_VERSION,
        description=("사진 한 장(+ 가이드 프레임) → 정상/이상 + 병변 6종 **분포**. "
                     "병변 이름은 단정하지 않습니다 — 응답에 '1등' 필드가 없는 건 "
                     "실수가 아닙니다 (docs/cautions/03 §7-B)."),
    )

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "mock": mock, "contract_version": CONTRACT_VERSION,
                "threshold": getattr(agent, "thr", None)}

    @app.post("/v1/screen")
    async def screen(photo: UploadFile = File(...), box: str = Form(default="")):
        """box: 앱의 **가이드 프레임**. 정규화 JSON `[x, y, w, h]` (0~1).

        주면 학습과 같은 함수로 자릅니다 (`src/agent.crop_for`). 안 주면
        화면 중앙으로 물러섭니다 — 1단계는 큰 차이가 없지만 2단계는 어긋납니다.
        """
        raw = await photo.read()
        if not raw:
            raise HTTPException(400, "빈 파일입니다.")
        if len(raw) > MAX_BYTES:
            raise HTTPException(413, f"사진이 너무 큽니다 ({len(raw) / 1e6:.1f}MB > 12MB).")
        try:
            im = Image.open(io.BytesIO(raw))
            im.load()
        except Exception:
            raise HTTPException(415, "이미지로 열리지 않는 파일입니다.")

        b = None
        if box:
            try:
                b = json.loads(box)
                if not (isinstance(b, list) and len(b) == 4):
                    raise ValueError
                b = [float(v) for v in b]
            except Exception:
                raise HTTPException(422, "box 는 정규화 [x, y, w, h] JSON 배열이어야 합니다.")

        if mock:
            # MockAgent 는 바이트 해시로 값을 만듭니다 — 파일을 그대로 넘깁니다
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=Path(photo.filename or "x.jpg").suffix,
                                             delete=True) as tf:
                tf.write(raw); tf.flush()
                return JSONResponse(agent.screen(tf.name, box=b))
        return JSONResponse(agent.screen(im, box=b))

    demo = ROOT / "demo" / "index.html"

    @app.get("/")
    def index():
        if not demo.exists():
            return {"hint": "데모 화면이 없습니다. POST /v1/screen 을 직접 쓰세요."}
        return FileResponse(demo)

    return app


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt1", help="1단계(정상/이상) 체크포인트")
    ap.add_argument("--ckpt2", help="2단계(병변 6종) 체크포인트")
    ap.add_argument("--threshold", type=float, default=None,
                    help="1단계 임계값. 생략하면 stage1_threshold.json 을 찾습니다")
    ap.add_argument("--mock", action="store_true",
                    help="모델 없이 같은 모양의 가짜 응답 (화면 확인용)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)

    if a.mock:
        from src.agent import MockAgent

        agent = MockAgent(a.threshold or 0.1823)
        print("⚠️  mock 모드입니다 — 숫자는 모델이 낸 것이 아닙니다.")
    else:
        if not (a.ckpt1 and a.ckpt2):
            ap.error("--ckpt1 과 --ckpt2 가 필요합니다 (또는 --mock).")
        from src.agent import ScreeningAgent

        agent = ScreeningAgent.load(a.ckpt1, a.ckpt2, a.threshold)
        print(f"✅ 1단계 {Path(a.ckpt1).parent.name} / 2단계 {Path(a.ckpt2).parent.name} "
              f"/ 임계값 {agent.thr:.4f}")

    import uvicorn

    print(f"→ http://{a.host}:{a.port}/       데모 화면")
    print(f"→ http://{a.host}:{a.port}/docs   API 문서")
    uvicorn.run(build_app(agent, a.mock), host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()
