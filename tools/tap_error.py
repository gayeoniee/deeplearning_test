"""사람이 병변을 **점 하나로** 찍으면 얼마나 정확한가 — 로컬에서 재는 도구.

    uv run python tools/tap_error.py --n 40

왜 이걸 재나 — `tools/box_error.py` 로 재보니 사람은 네모 **크기**를 병변에
안 맞춥니다 (상관 −0.05, 크기 중앙값 정답의 2.80배). 앱 밴드는 `네모 ÷ 화면`
을 보는데 모델이 필요한 건 `네모 ÷ 병변` 이라, **앱은 그 오차를 검사할 수
없습니다.**

그런데 `f320`(1단계 크롭)은 **중심만** 씁니다. 크기를 아예 안 물어보면 그
문제가 통째로 사라지고 **위치 하나만** 남습니다. 시뮬레이션(STEP 36)에서:

    탭 오차 0.02  f320 0.4521   ← 라벨 네모 수준을 유지
    탭 오차 0.10  f320 0.3830   ← m2.5+네모(0.3609)보다 위
    탭 오차 0.161 f320 0.3363   ← 이건 **네모 중심**이지 탭이 아닙니다

판정 기준은 **재기 전에** 박았습니다: `experiments.TAP_ERROR_MAX = 0.10`.

⚠️ 자극은 `box_error.py` 와 **같은 방식**입니다 — 원본 zip 에서 꺼내 병변
주변을 넉넉히 자르고 창 위치를 무작위로 둡니다. 크롭을 그대로 쓰면 정답이
한가운데 놓여 **잴 것이 없어집니다** (1차에서 당한 것).
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from src.experiments import TAP_ERROR_MAX  # noqa: E402

OUT = ROOT / "reports" / "tap_error.json"

PAGE = """<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>탭 오차 측정</title>
<style>
body{margin:0;background:#15171a;color:#e8ebef;font-family:system-ui,sans-serif;
     display:flex;flex-direction:column;align-items:center;padding:18px 14px 40px}
.bar{width:100%;max-width:640px;display:flex;align-items:baseline;gap:10px;margin-bottom:12px}
h1{font-size:16px;margin:0;font-weight:600}
.n{margin-left:auto;color:#98a1ac;font-size:13px;font-variant-numeric:tabular-nums}
p.help{max-width:640px;margin:0 0 14px;color:#98a1ac;font-size:13.5px;line-height:1.6}
#stage{position:relative;max-width:640px;width:100%;border-radius:10px;overflow:hidden;
       background:#000;touch-action:none;user-select:none;cursor:crosshair}
#stage img{display:block;width:100%}
#dot,#ans{position:absolute;width:16px;height:16px;margin:-8px 0 0 -8px;border-radius:50%;
     border:2.5px solid #fff;background:#f0a0a0;display:none;pointer-events:none}
#ans{background:#7fc98f;border-style:dashed}
#truth{position:absolute;border:2px dashed #7fc98f;border-radius:4px;display:none;
       pointer-events:none}
.acts{display:flex;gap:9px;max-width:640px;width:100%;margin-top:12px}
button{flex:1;font:inherit;font-size:14px;font-weight:600;padding:12px;border:0;
       border-radius:9px;background:#2c3037;color:#e8ebef;cursor:pointer}
button.go{background:#f0a0a0;color:#2a1a1a;flex:2}
button:disabled{opacity:.45;cursor:default}
#done{max-width:640px;font-size:14px;line-height:1.7;white-space:pre-wrap;
      font-family:ui-monospace,monospace}
</style>
<div class="bar"><h1>병변이라고 생각하는 곳을 <b>콕</b> 찍어주세요</h1>
  <span class="n" id="count"></span></div>
<p class="help"><b>크기는 안 물어봅니다 — 점 하나만.</b>
  정답은 찍은 <b>뒤에</b> 초록으로 보여드립니다.</p>
<div id="stage"><img id="photo" alt=""><div id="dot"></div><div id="truth"></div>
  <div id="ans"></div></div>
<div class="acts">
  <button class="go" id="next" disabled>다음 →</button>
  <button id="skip">병변이 안 보임</button>
</div>
<pre id="done"></pre>
<script>
let items=[], i=0, p=null, out=[], showing=false;
const $=s=>document.querySelector(s);
const stage=$('#stage'), photo=$('#photo'), dot=$('#dot'), truth=$('#truth'), ans=$('#ans');
fetch('/items').then(r=>r.json()).then(j=>{items=j; show();});

function show(){
  if(i>=items.length) return finish();
  showing=false; p=null;
  dot.style.display='none'; truth.style.display='none'; ans.style.display='none';
  $('#next').disabled=true; $('#next').textContent='다음 →';
  $('#count').textContent=(i+1)+' / '+items.length;
  photo.src='/img/'+i;
}
stage.addEventListener('click', e=>{
  if(showing) return;
  const R=stage.getBoundingClientRect();
  p=[(e.clientX-R.left)/R.width, (e.clientY-R.top)/R.height];
  dot.style.left=p[0]*100+'%'; dot.style.top=p[1]*100+'%'; dot.style.display='block';
  $('#next').disabled=false;
});
$('#next').addEventListener('click', ()=>{
  if(showing){ i++; show(); return; }
  if(!p) return;
  out.push({idx:items[i].idx, tap:p});
  const t=items[i].truth;
  truth.style.left=t[0]*100+'%'; truth.style.top=t[1]*100+'%';
  truth.style.width=(t[2]-t[0])*100+'%'; truth.style.height=(t[3]-t[1])*100+'%';
  truth.style.display='block';
  ans.style.left=((t[0]+t[2])/2)*100+'%'; ans.style.top=((t[1]+t[3])/2)*100+'%';
  ans.style.display='block';
  showing=true; $('#next').textContent='다음 →'; $('#next').disabled=false;
});
$('#skip').addEventListener('click', ()=>{ out.push({idx:items[i].idx, tap:null}); i++; show(); });
function finish(){
  stage.style.display='none'; document.querySelector('.acts').style.display='none';
  $('#count').textContent=''; $('#done').textContent='저장하는 중…';
  fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},
                 body:JSON.stringify(out)}).then(r=>r.text()).then(t=>{$('#done').textContent=t;});
}
</script></html>"""


def summarize(rows: list[dict]) -> str:
    """탭 오차 = 찍은 점과 정답 중심의 거리 ÷ 화면 폭 (긴 축 기준)."""
    good = [r for r in rows if r.get("tap")]
    if not good:
        return "표본이 없습니다."
    errs = []
    for r in good:
        t, p = r["truth"], r["tap"]
        cx, cy = (t[0] + t[2]) / 2, (t[1] + t[3]) / 2
        errs.append(max(abs(p[0] - cx), abs(p[1] - cy)))
    errs.sort()
    med = st.median(errs)
    q = lambda f: errs[min(len(errs) - 1, int(len(errs) * f))]  # noqa: E731
    inside = sum(1 for e in errs if e <= TAP_ERROR_MAX) / len(errs)
    L = [f"\n표본 {len(good)}장 (건너뜀 {len(rows) - len(good)})",
         "",
         "■ 탭 오차 (찍은 점 ↔ 정답 중심, 화면 대비)",
         f"    중앙값 {med:.3f}   10~90% {q(.1):.3f} ~ {q(.9):.3f}",
         "",
         f"■ 사전등록 문턱 {TAP_ERROR_MAX}  (재기 전에 박았습니다)",
         f"    문턱 안 {inside:.1%}",
         ""]
    if med <= TAP_ERROR_MAX:
        L += ["⭕ **통과** — 위치만 받는 설계(탭 + f320)가 유리합니다.",
              "   ⚠️ 다만 '바로 바꾸자' 가 아닙니다. 라벨 네모 기준으로는 m2.5 가",
              "      여전히 앞섭니다 (0.4913 vs 0.4516) — 완벽한 네모를 받을 수",
              "      있으면 m2.5 가 낫고, 못 받으니까 뒤집히는 것입니다.",
              "      3팔 앙상블도 다시 짜야 합니다."]
    else:
        L += ["❌ **미달** — 탭도 충분히 정확하지 않습니다.",
              "   이 길도 닫고 '자른 결과 미리보기' 쪽으로 가세요."]
    L += ["",
          "⚠️ VL01 기준 · n 이 작고 한 사람입니다. 방향을 보는 용도입니다.",
          f"원본: {OUT}"]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="127.0.0.1",
                    help="바인딩 주소. 폰에서 하려면 tailscale 주소를 주세요 "
                         "(`tailscale ip -4`). ⚠️ 0.0.0.0 은 같은 네트워크 전체에 "
                         "열립니다 — AI Hub 사진이 뜨는 화면이라 권하지 않습니다.")
    ap.add_argument("--replay", action="store_true")
    a = ap.parse_args()

    if a.replay:
        print(summarize(json.loads(OUT.read_text(encoding="utf-8"))))
        return

    import box_error  # 자극을 **같은 함수**로 만듭니다 (두 곳에 안 적습니다)

    items = box_error.pick(a.n, a.seed)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                return self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            if self.path == "/items":
                return self._send(200, json.dumps(items).encode("utf-8"),
                                  "application/json")
            if self.path.startswith("/img/"):
                k = int(self.path.rsplit("/", 1)[1])
                return self._send(200, Path(items[k]["path"]).read_bytes(), "image/jpeg")
            self._send(404, b"", "text/plain")

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            got = json.loads(self.rfile.read(n) or b"[]")
            by = {g["idx"]: g.get("tap") for g in got}
            rows = [{"path": it["path"], "truth": it["truth"],
                     "tap": by.get(it["idx"])} for it in items]
            OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                           encoding="utf-8")
            text = summarize(rows)
            print(text)
            self._send(200, text.encode("utf-8"), "text/plain; charset=utf-8")
            raise SystemExit(0)

    print(f"→ http://{a.host}:{a.port}/   (다 하면 자동으로 닫힙니다)")
    try:
        HTTPServer((a.host, a.port), H).serve_forever()
    except SystemExit:
        pass


if __name__ == "__main__":
    main()
