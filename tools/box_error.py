"""사람이 그린 네모가 정답 bbox 와 얼마나 다른가 — 로컬에서 재는 도구.

    uv run python tools/box_error.py --n 40
    uv run python tools/box_error.py --replay          # 저장된 결과만 다시 요약

**수의사가 필요 없습니다.** 수의사는 *병명*을 붙이는 사람인데, 여기서 재는 건
*네모를 얼마나 잘 그리는가* 입니다. 그 정답은 이미 `manifest_final.parquet` 안의
라벨 bbox 로 들어 있습니다. 병명은 이미 알고 있으니 다시 물을 필요가 없습니다.

왜 이걸 재나 — 앱은 촬영 가이드 프레임을 bbox 로 넘깁니다(`src/agent.py`).
크롭 **구성**은 학습과 0px 차이로 같아졌지만, 네모를 **누가** 그렸는지가 다릅니다:

    라벨러  답을 알고 병변에 딱 맞춰 그림
    사용자  답을 모르는 채 대충 맞춤

2단계 `m2.5` 는 네모의 **크기**로 배율이 정해지므로 이 차이에 그대로 노출됩니다
(1단계 `f320` 은 중심만 쓰므로 거의 무관합니다).

그리고 **얼마나 어긋나면 얼마나 나빠지는지는 이미 재 뒀습니다** — STEP 10 의
`robust.usable_range()` 가 그 곡선입니다. 남은 건 사람이 실제로 얼마나
어긋나게 그리는가 하나뿐이고, 그건 이 도구로 30분이면 됩니다.

의존성이 없습니다 (표준 라이브러리 + pandas/Pillow). 한국 PC 에서
`uv sync` 만 해도 돕니다 — 측정 도구에 설치 장벽이 있으면 안 됩니다.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# STEP 10 실측 밴드 (STATUS.md "촬영 가이드"). `robust.usable_range()` 가 뽑았습니다.
ZOOM_RECOMMEND = (0.85, 1.4)      # 하락 5% 이내
ZOOM_ALLOW = (0.7, 1.7)           # 하락 10% 이내
SHIFT_MAX = 0.10                  # 병변이 화면 중앙에서 이만큼 이내

# 위 밴드를 **네모 오차**로 뒤집은 값 — 앱이 실제로 알아야 하는 숫자입니다.
# 네모를 r 배로 그리면 크롭이 r 배 넓어지므로 환산 줌은 1/r 입니다.
#   zoom 0.7~1.7 안에 있으려면  r ∈ [1/1.7, 1/0.7] = [0.59, 1.43]
BOX_ALLOW = (1 / ZOOM_ALLOW[1], 1 / ZOOM_ALLOW[0])          # 0.59 ~ 1.43배
BOX_RECOMMEND = (1 / ZOOM_RECOMMEND[1], 1 / ZOOM_RECOMMEND[0])   # 0.71 ~ 1.18배

OUT = ROOT / "reports" / "box_error.json"

PAGE = """<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>네모 오차 측정</title>
<style>
body{margin:0;background:#15171a;color:#e8ebef;font-family:system-ui,-apple-system,sans-serif;
     display:flex;flex-direction:column;align-items:center;padding:18px 14px 40px}
.bar{width:100%;max-width:640px;display:flex;align-items:baseline;gap:10px;margin-bottom:12px}
h1{font-size:16px;margin:0;font-weight:600}
.n{margin-left:auto;font-variant-numeric:tabular-nums;color:#98a1ac;font-size:13px}
p.help{max-width:640px;margin:0 0 14px;color:#98a1ac;font-size:13.5px;line-height:1.6}
#stage{position:relative;max-width:640px;width:100%;border-radius:10px;overflow:hidden;
       background:#000;touch-action:none;user-select:none;cursor:move}
#stage img{display:block;width:100%;opacity:.72}
#hole{position:absolute;border:2.5px solid #fff;border-radius:6px;
      box-shadow:0 0 0 9999px rgba(0,0,0,.55);cursor:move}
#hole::after{content:'';position:absolute;right:-9px;bottom:-9px;width:19px;height:19px;
      background:#f0a0a0;border:2.5px solid #fff;border-radius:50%;cursor:nwse-resize}
#truth{position:absolute;border:2px dashed #7fc98f;border-radius:4px;display:none;
       pointer-events:none}
.acts{display:flex;gap:9px;max-width:640px;width:100%;margin-top:12px}
button{flex:1;font:inherit;font-size:14px;font-weight:600;padding:12px;border:0;border-radius:9px;
       background:#2c3037;color:#e8ebef;cursor:pointer}
button.go{background:#f0a0a0;color:#2a1a1a;flex:2}
button:disabled{opacity:.45;cursor:default}
#done{max-width:640px;font-size:14px;line-height:1.7;white-space:pre-wrap;
      font-family:ui-monospace,monospace}
</style>
<div class="bar"><h1>병변이라고 생각하는 곳에 네모를 맞춰주세요</h1>
  <span class="n" id="count"></span></div>
<p class="help">끌어서 옮기고, 오른쪽 아래 동그라미로 크기를 바꿉니다.
  정답은 넘긴 <b>뒤에</b> 초록 점선으로 보여드립니다 — 먼저 보면 측정이 안 됩니다.</p>
<div id="stage"><img id="photo" alt=""><div id="hole"></div><div id="truth"></div></div>
<div class="acts">
  <button class="go" id="next">이 위치로 확정</button>
  <button id="skip">병변이 안 보임 (건너뛰기)</button>
</div>
<pre id="done"></pre>
<script>
let items = [], i = 0, b = {x:.3,y:.3,w:.4,h:.4}, aspect = 1, out = [], showing = false;
const $ = s => document.querySelector(s);
const stage = $('#stage'), hole = $('#hole'), photo = $('#photo'), truth = $('#truth');

fetch('/items').then(r=>r.json()).then(j=>{ items = j; show(); });

function draw(){
  hole.style.left = b.x*100+'%'; hole.style.top = b.y*100+'%';
  hole.style.width = b.w*100+'%'; hole.style.height = b.h*100+'%';
}
function show(){
  if(i >= items.length) return finish();
  showing = false; truth.style.display='none'; hole.style.display='';
  $('#count').textContent = (i+1)+' / '+items.length;
  $('#next').textContent = '이 위치로 확정'; $('#next').disabled = false;
  photo.src = '/img/'+i;
  photo.onload = () => {
    aspect = photo.naturalWidth/photo.naturalHeight;
    b = {x:.3, y:.3, w:.4, h:.4*aspect};
    if(b.y+b.h>1){ b.h=.9-b.y; b.w=b.h/aspect; }
    draw();
  };
}
function reveal(box){
  const t = items[i].truth;
  truth.style.left=t[0]*100+'%'; truth.style.top=t[1]*100+'%';
  truth.style.width=(t[2]-t[0])*100+'%'; truth.style.height=(t[3]-t[1])*100+'%';
  truth.style.display='block'; showing = true;
  $('#next').textContent = '다음 →';
}
$('#next').addEventListener('click', ()=>{
  if(showing){ i++; show(); return; }
  out.push({idx: items[i].idx, box: [b.x, b.y, b.w, b.h]});
  reveal();
});
$('#skip').addEventListener('click', ()=>{
  out.push({idx: items[i].idx, box: null}); i++; show();
});

let mode=null, s0=null;
stage.addEventListener('pointerdown', e=>{
  if(showing) return;
  const R = stage.getBoundingClientRect();
  const px=(e.clientX-R.left)/R.width, py=(e.clientY-R.top)/R.height;
  mode = Math.hypot(px-(b.x+b.w), py-(b.y+b.h)) < .06 ? 'size' : 'move';
  s0 = {px, py, ...b}; stage.setPointerCapture(e.pointerId);
});
stage.addEventListener('pointermove', e=>{
  if(!mode) return;
  const R = stage.getBoundingClientRect();
  const dx=(e.clientX-R.left)/R.width-s0.px, dy=(e.clientY-R.top)/R.height-s0.py;
  if(mode==='move'){
    b.x=Math.min(Math.max(s0.x+dx,0),1-b.w); b.y=Math.min(Math.max(s0.y+dy,0),1-b.h);
  } else {
    const k=Math.min(Math.max(s0.w+Math.max(dx,dy),.02), 1-b.x, (1-b.y)/aspect);
    b.w=k; b.h=k*aspect;
  }
  draw();
});
['pointerup','pointercancel'].forEach(ev=>stage.addEventListener(ev,()=>{mode=null;}));

function finish(){
  stage.style.display='none'; document.querySelector('.acts').style.display='none';
  $('#count').textContent='';
  $('#done').textContent='저장하는 중…';
  fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},
                 body:JSON.stringify(out)})
    .then(r=>r.text()).then(t=>{ $('#done').textContent = t; });
}
</script></html>"""


# ──────────────────────────────────────────────────────────────
# 오차 → 이미 재 둔 교란으로 환산
# ──────────────────────────────────────────────────────────────
def to_perturbation(user, truth) -> dict:
    """사람 네모와 정답 네모의 차이를 **줌/이동 배율**로 바꿉니다.

    이래야 STEP 10 의 `usable_range()` 곡선에 그대로 대입할 수 있습니다.
    새 실험을 하는 게 아니라, **이미 잰 곡선을 읽는 것**입니다.

    Args:
        user:  [x, y, w, h]        (정규화, 프레임 기준)
        truth: [x1, y1, x2, y2]    (정규화, 프레임 기준)

    Returns:
        size_ratio  사람 네모 / 정답 네모 (긴 변)
        zoom        환산 줌. 네모를 크게 그리면 크롭이 넓어지므로 1/size_ratio
        shift_frac  중심 어긋남 ÷ 2단계 크롭 폭 (= ShiftView 의 frac 과 같은 단위)
    """
    ux, uy, uw, uh = user
    tw, th = truth[2] - truth[0], truth[3] - truth[1]
    t_long, u_long = max(tw, th), max(uw, uh)
    if t_long <= 0 or u_long <= 0:
        return {}

    ratio = u_long / t_long
    ucx, ucy = ux + uw / 2, uy + uh / 2
    tcx, tcy = (truth[0] + truth[2]) / 2, (truth[1] + truth[3]) / 2
    # 2단계 크롭은 정답 네모 긴 변의 2.5배입니다 — 그 창 기준의 어긋남
    shift = math.hypot(ucx - tcx, ucy - tcy) / (2.5 * t_long)
    return {"size_ratio": ratio, "zoom": 1 / ratio, "shift_frac": shift}


def summarize(rows: list[dict]) -> str:
    """측정 결과를 밴드에 대입해 사람이 읽을 요약으로."""
    good = [r for r in rows if r.get("size_ratio")]
    skipped = len(rows) - len(good)
    if not good:
        return "쓸 수 있는 표본이 없습니다."

    ratios = sorted(r["size_ratio"] for r in good)
    zooms = [r["zoom"] for r in good]
    shifts = sorted(r["shift_frac"] for r in good)

    def pct(vals, q):
        return vals[min(int(q * len(vals)), len(vals) - 1)]

    in_rec = sum(ZOOM_RECOMMEND[0] <= z <= ZOOM_RECOMMEND[1] for z in zooms) / len(zooms)
    in_allow = sum(ZOOM_ALLOW[0] <= z <= ZOOM_ALLOW[1] for z in zooms) / len(zooms)
    in_shift = sum(s <= SHIFT_MAX for s in shifts) / len(shifts)
    both = sum(ZOOM_ALLOW[0] <= r["zoom"] <= ZOOM_ALLOW[1] and r["shift_frac"] <= SHIFT_MAX
               for r in good) / len(good)

    L = [
        f"표본 {len(good)}장 (건너뜀 {skipped})",
        "",
        "■ 네모 크기 (사람 ÷ 정답)",
        f"    중앙값 {st.median(ratios):.2f}배   "
        f"10~90% {pct(ratios,.1):.2f} ~ {pct(ratios,.9):.2f}배",
        "",
        "■ 중심 어긋남 (2단계 크롭 폭 대비)",
        f"    중앙값 {st.median(shifts):.3f}   "
        f"10~90% {pct(shifts,.1):.3f} ~ {pct(shifts,.9):.3f}",
        "",
        "■ 허용되는 네모 오차 (위 밴드를 뒤집은 값)",
        f"    권장 {BOX_RECOMMEND[0]:.2f} ~ {BOX_RECOMMEND[1]:.2f}배   "
        f"허용 {BOX_ALLOW[0]:.2f} ~ {BOX_ALLOW[1]:.2f}배",
        "",
        "■ STEP 10 밴드에 대입 (usable_range 실측)",
        f"    배율 권장({ZOOM_RECOMMEND[0]}~{ZOOM_RECOMMEND[1]}x, 하락 5% 이내) 안  {in_rec:6.1%}",
        f"    배율 허용({ZOOM_ALLOW[0]}~{ZOOM_ALLOW[1]}x, 하락 10% 이내) 안  {in_allow:6.1%}",
        f"    위치 허용(중앙 {SHIFT_MAX:.0%} 이내) 안                {in_shift:6.1%}",
        f"    둘 다 허용 안                                  {both:6.1%}",
        "",
    ]
    if both >= 0.9:
        L.append("→ 사람이 그린 네모의 90% 이상이 허용 밴드 안입니다.")
        L.append("  2단계에 대한 추가 하락은 잡음 수준으로 볼 수 있습니다.")
    elif both >= 0.7:
        L.append("→ 대부분 허용 안이지만 꼬리가 있습니다.")
        L.append("  앱에서 밴드 밖을 막는 게(agent.check_guide) 실제로 일을 합니다.")
    else:
        L.append("→ ⚠️ 절반 가까이가 밴드 밖입니다. 네모를 그리게 하는 것만으로는 부족합니다.")
        L.append("  가이드 프레임의 기본 크기·안내 문구를 고치고 다시 재세요.")
    L += ["",
          "⚠️ 이건 **네모 그리기 오차**만 잰 것입니다. 병명 정확도가 아닙니다.",
          f"원본: {OUT.relative_to(ROOT)}"]
    return "\n".join(L)


# ──────────────────────────────────────────────────────────────
def pick(n: int, seed: int) -> list[dict]:
    """정답 bbox 가 있는 사진을 n 장 고릅니다. 가장 **넓은** 크롭을 씁니다.

    원본은 크롭 후 지웠으므로(`data/raw`), 남아 있는 크롭 중 화면이 제일 넓은
    것을 쓰고 정답 bbox 를 그 좌표계로 옮깁니다(`crop.bbox_in_crop`).
    """
    import pandas as pd

    from src import crop, env

    mf = env.work_root() / "manifests" / "manifest_final.parquet"
    if not mf.exists():
        raise SystemExit(f"매니페스트가 없습니다: {mf}\n"
                         "한국 PC 에서 `uv run python prepare_local.py --finalize` 뒤에 도세요.")
    df = pd.read_parquet(mf)

    tags = crop.available_tags()
    tag = next((t for t in ("full", "m2.5", "m1.5") if t in tags), tags[0] if tags else None)
    if tag is None:
        raise SystemExit("크롭이 하나도 없습니다.")

    df = crop.switch_tag(df, tag, verbose=False, allow_missing=True)
    df = df[df["crop_path"].notna()]
    df = df[df.get("label_orig", df.get("label")) != "A7"]      # 병변만 (정상엔 병변이 없음)

    picked: list[dict] = []
    for _, r in df.sample(min(len(df), n * 4), random_state=seed).iterrows():
        bb = crop.bbox_in_crop(r, tag=tag)
        if not bb:
            continue
        picked.append({"idx": len(picked), "path": str(r["crop_path"]), "truth": bb})
        if len(picked) >= n:
            break
    if not picked:
        raise SystemExit("정답 bbox 를 옮길 수 있는 사진이 없습니다.")
    print(f"크롭 태그 '{tag}' 에서 {len(picked)}장 골랐습니다.")
    return picked


def serve(items: list[dict], port: int) -> None:
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):        # 조용히
            pass

        def _send(self, code, body, ctype="text/plain; charset=utf-8"):
            data = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                return self._send(200, PAGE, "text/html; charset=utf-8")
            if self.path == "/items":
                # 정답은 넘긴 뒤에만 보여주지만, 어차피 클라이언트에 내려갑니다 —
                # 혼자 재는 도구라 여기서 더 숨기는 건 의미가 없습니다.
                return self._send(200, json.dumps(items), "application/json")
            if self.path.startswith("/img/"):
                try:
                    p = Path(items[int(self.path[5:])]["path"])
                    return self._send(200, p.read_bytes(), "image/jpeg")
                except Exception:
                    return self._send(404, "없음")
            return self._send(404, "없음")

        def do_POST(self):
            if self.path != "/save":
                return self._send(404, "없음")
            n = int(self.headers.get("Content-Length", 0))
            got = json.loads(self.rfile.read(n) or b"[]")

            rows = []
            for g in got:
                it = items[g["idx"]]
                row = {"path": it["path"], "truth": it["truth"], "user": g["box"]}
                if g["box"]:
                    row.update(to_perturbation(g["box"], it["truth"]))
                rows.append(row)

            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
            text = summarize(rows)
            print("\n" + text)
            self._send(200, text)
            # 저장했으면 할 일이 끝났습니다
            import threading

            threading.Timer(1.0, self.server.shutdown).start()

    srv = HTTPServer(("127.0.0.1", port), H)
    print(f"→ http://127.0.0.1:{port}/  (다 하면 자동으로 닫힙니다)")
    srv.serve_forever()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=40, help="몇 장을 재나 (기본 40)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--replay", action="store_true", help="저장된 결과만 다시 요약")
    a = ap.parse_args(argv)

    if a.replay:
        if not OUT.exists():
            raise SystemExit(f"{OUT} 이 없습니다. 먼저 --replay 없이 도세요.")
        print(summarize(json.loads(OUT.read_text(encoding="utf-8"))))
        return

    serve(pick(a.n, a.seed), a.port)


if __name__ == "__main__":
    main()
