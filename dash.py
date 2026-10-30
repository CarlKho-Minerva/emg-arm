"""Served With a Smile: the demo screen that stands beside the arm.

    .venv/bin/python dash.py            # then open http://localhost:8790  (or: open run_dash.command)

One screen, two halves. Left, what the robot sees: the wrist camera picture (from the coach on the
robot's box, else the last data/live.jpg, else "robot offline") and the honest log, the last lines
of data/serve.log as the arm wrote them. Right, your face: the frame smile.py shows in its window
(data/face_live.jpg), the control table, the live cheek and mouth numbers against Carl's calibrated
thresholds, and the group being heard, in large type.

Stdlib only. Nothing here touches the robot; the only network call is a GET of the box's live.jpg.
"""
import json
import os
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SERVE_LOG = os.path.join(DATA, "serve.log")
FACE_LOG = os.path.join(DATA, "face.log")
FACE_STATUS = os.path.join(DATA, "face_status.json")
FACE_CALIB = os.path.join(DATA, "face_calib.json")
FACE_JPG = os.path.join(DATA, "face_live.jpg")
LIVE_JPG = os.path.join(DATA, "live.jpg")
ROBOT_URL = os.environ.get("DASH_ROBOT_URL", "http://192.168.31.130:8787/live.jpg")
ROBOT_POLL_S = 0.15          # how often the box is asked for a picture while it answers
ROBOT_RETRY_S = 2.0          # how often it is asked again once it has stopped answering
ROBOT_FRESH_S = 1.5          # a box picture older than this is not shown as live
LOG_LINES = 12

FACE_WORDS = {"L": "smile left", "R": "smile right", "B": "full smile", "M": "mouth open"}

# the controls as serve.py acts on them (DEMO.md, Face controls, plus the open-mouth feed of 19:53)
CONTROLS = [
    ("Fetch mode (the start state, at greet)", None, None),
    ("L", "smile left", "fetch the cube, prove the grip, back to greet"),
    ("R", "smile right", "hand it over, into your palm"),
    ("M", "open mouth", "feed me: cube held at the mouth 2 s, then put back"),
    ("B", "full smile", "let go"),
    ("BB", "smile twice", "nudge mode"),
    ("BBB", "smile three times", "home, gripper open"),
    ("Nudge mode (30 mm a step)", None, None),
    ("L / R", "smile left / right", "one step along the current axis"),
    ("BB", "smile twice", "next axis: left/right, toward/away, up/down"),
    ("B", "full smile", "grab if empty, let go if holding"),
    ("BBB", "smile three times", "back to fetch mode, at greet"),
]

robot = {"jpeg": None, "t": 0.0, "source": "none", "error": ""}


def placeholder(text, sub=""):
    """A dark picture with a line of text, as SVG (no image library needed)."""
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360" viewBox="0 0 640 360">'
           f'<rect width="640" height="360" fill="#0a0a0a"/>'
           f'<text x="320" y="176" fill="#888" font-family="-apple-system,Inter,system-ui,sans-serif" font-size="26" text-anchor="middle">{text}</text>'
           f'<text x="320" y="210" fill="#555" font-family="-apple-system,Inter,system-ui,sans-serif" font-size="15" text-anchor="middle">{sub}</text></svg>')
    return svg.encode(), "image/svg+xml"


def robot_poll():
    """Keep the latest picture from the box in memory, so no page request ever waits on the LAN."""
    while True:
        try:
            with urllib.request.urlopen(ROBOT_URL, timeout=1.0) as r:
                body = r.read()
            if body[:2] == b"\xff\xd8":
                robot.update(jpeg=body, t=time.time(), source="box", error="")
            else:
                robot.update(error="box answered with something that is not a jpeg")
            time.sleep(ROBOT_POLL_S)
        except Exception as e:
            robot.update(error=str(e)[:80])
            time.sleep(ROBOT_RETRY_S)


def robot_picture():
    """(bytes, content type, source, age in seconds)."""
    now = time.time()
    if robot["jpeg"] and now - robot["t"] < ROBOT_FRESH_S:
        return robot["jpeg"], "image/jpeg", "box", now - robot["t"]
    try:
        age = now - os.path.getmtime(LIVE_JPG)
        return open(LIVE_JPG, "rb").read(), "image/jpeg", "file", age
    except OSError:
        pass
    body, ctype = placeholder("robot offline", "no picture from the box, none on disk")
    return body, ctype, "none", None


def face_picture():
    try:
        age = time.time() - os.path.getmtime(FACE_JPG)
        return open(FACE_JPG, "rb").read(), "image/jpeg", age
    except OSError:
        body, ctype = placeholder("no face camera", "start smile.py (open run_smile.command)")
        return body, ctype, None


def tail(path, n):
    try:
        with open(path) as f:
            lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    except OSError:
        return []
    return lines[-n:]


def face_words(group):
    """"RBBBR" becomes "smile right, full smile x3, smile right"."""
    out, i = [], 0
    while i < len(group):
        j = i
        while j < len(group) and group[j] == group[i]:
            j += 1
        word = FACE_WORDS.get(group[i], group[i])
        out.append(word if j - i == 1 else f"{word} x{j - i}")
        i = j
    return ", ".join(out)


def serve_lines():
    """Every line of serve.log, parsed: {t, time, group, face, action}."""
    rows = []
    for ln in tail(SERVE_LOG, 100000):
        parts = ln.split(" ", 2)
        if len(parts) < 3:
            continue
        try:
            t = float(parts[0])
        except ValueError:
            continue
        rows.append({"t": t, "time": time.strftime("%H:%M:%S", time.localtime(t)), "group": parts[1],
                     "face": face_words(parts[1]), "action": parts[2]})
    return rows


def counts(rows):
    picks = sum(1 for r in rows if "fetch" in r["action"])
    hands = sum(1 for r in rows if "feed" in r["action"] or "hand it over" in r["action"])
    return {"picks": picks, "handovers": hands, "lines": len(rows)}


def read_json(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


def state():
    rows = serve_lines()
    face = read_json(FACE_STATUS) or {}
    calib = read_json(FACE_CALIB) or {}
    now = time.time()
    last_face = tail(FACE_LOG, 1)
    last_heard = None
    if last_face:
        p = last_face[0].split()
        if len(p) == 2:
            try:
                last_heard = {"t": float(p[0]), "group": p[1], "face": face_words(p[1]), "age": now - float(p[0])}
            except ValueError:
                pass
    face_age = (now - os.path.getmtime(FACE_JPG)) if os.path.exists(FACE_JPG) else None
    return {
        "t": now,
        "serve": rows[-LOG_LINES:],
        "counts": counts(rows),
        "face": face,
        "face_age": (now - face["t"]) if isinstance(face.get("t"), (int, float)) else None,
        "face_picture_age": face_age,
        "last_heard": last_heard,
        "calib": {k: calib.get(k) for k in ("up_left", "up_right", "down", "jaw_open", "at")},
        "robot": {"source": robot_picture()[2], "age": robot_picture()[3], "url": ROBOT_URL, "error": robot["error"]},
    }


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Served With a Smile</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--fg:#111;--mut:#666;--line:#e5e5e5;--bg:#fff;--card:#fafafa;--ok:#0a7d3b;--acc:#000;--warn:#b45309}
@media(prefers-color-scheme:dark){:root{--fg:#ededed;--mut:#888;--line:#2a2a2a;--bg:#0a0a0a;--card:#111;--ok:#2fbf71;--acc:#fff;--warn:#f59e0b}}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 -apple-system,Inter,system-ui,sans-serif;display:flex;flex-direction:column;overflow:hidden}
header{display:flex;align-items:baseline;gap:20px;padding:12px 24px;border-bottom:1px solid var(--line);flex:0 0 auto}
header h1{font-size:22px;font-weight:600;letter-spacing:-.02em;margin:0;white-space:nowrap}
header .line{color:var(--mut);font-size:15px}
header .line b{color:var(--fg);font-weight:500}
header .right{margin-left:auto;color:var(--mut);font-size:13px;white-space:nowrap}
main{flex:1 1 auto;display:grid;grid-template-columns:1fr 1fr;gap:0;min-height:0}
section{display:flex;flex-direction:column;min-height:0;padding:14px 24px 16px;gap:10px}
section+section{border-left:1px solid var(--line)}
.k{font-size:12px;color:var(--mut);letter-spacing:.04em;text-transform:uppercase;display:flex;align-items:baseline;gap:10px}
.k .tag{margin-left:auto;text-transform:none;letter-spacing:0;font-variant-numeric:tabular-nums}
.k .tag.ok{color:var(--ok)}.k .tag.warn{color:var(--warn)}
h2{font-size:17px;font-weight:600;letter-spacing:-.01em;margin:0}
.pic{flex:1 1 auto;min-height:140px;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:#000;position:relative}
.pic img{display:block;width:100%;height:100%;object-fit:contain;background:#000}
.card{border:1px solid var(--line);border-radius:8px;background:var(--card);padding:10px 14px;min-height:0}
.log{flex:0 1 auto;max-height:56%;display:flex;flex-direction:column;min-height:0}
.log table{width:100%;border-collapse:collapse;font-size:13.5px;font-variant-numeric:tabular-nums}
.log td{padding:3px 6px 3px 0;border-bottom:1px solid var(--line);vertical-align:top;white-space:nowrap}
.log td:last-child{white-space:normal}
.log tr:last-child td{border-bottom:0}
.log td.t{color:var(--mut);font:12.5px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;width:72px}
.log td.f{font-weight:500;width:150px}
.log td.a{color:var(--mut)}
.log tr:last-child td.a,.log tr:last-child td.f{color:var(--fg)}
.log .rows{flex:1 1 auto;overflow:hidden;display:flex;flex-direction:column;justify-content:flex-end;margin-top:6px;-webkit-mask-image:linear-gradient(to bottom,transparent 0,#000 28px);mask-image:linear-gradient(to bottom,transparent 0,#000 28px)}
.count{margin-top:8px;padding-top:8px;border-top:1px solid var(--line);font-size:14px;color:var(--mut)}
.count b{color:var(--fg);font-weight:600;font-size:16px}
.empty{color:var(--mut);font-size:13.5px}
.controls{overflow:hidden}.controls table{width:100%;border-collapse:collapse;font-size:12.5px;line-height:1.3}
.controls td{padding:2px 8px 2px 0;border-bottom:1px solid var(--line);vertical-align:top}
.controls tr:last-child td{border-bottom:0}
.controls td.m{color:var(--mut);font-size:11px;letter-spacing:.04em;text-transform:uppercase;padding-top:7px}
.controls td.g{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;width:44px;color:var(--mut);white-space:nowrap}
.controls td.f{font-weight:500;width:128px;white-space:nowrap}
.controls td.a{color:var(--mut)}
.bottom{flex:0 1 auto;display:grid;grid-template-columns:3fr 2fr;gap:10px;min-height:0;overflow:hidden}
.bars{display:flex;flex-direction:column;gap:8px;margin-top:4px}
.bar{display:grid;grid-template-columns:52px 1fr 44px;align-items:center;gap:10px;font-size:13px}
.bar .lab{color:var(--mut)}
.bar .val{text-align:right;font-variant-numeric:tabular-nums;color:var(--mut)}
.bar .tr{position:relative;height:10px;background:var(--line);border-radius:5px;overflow:visible}
.bar .tr i{position:absolute;left:0;top:0;height:100%;background:var(--mut);border-radius:5px;width:0;transition:width .1s linear}
.bar .tr i.on{background:var(--ok)}
.bar .tr b{position:absolute;top:-4px;width:2px;height:18px;background:var(--fg);margin-left:-1px;border-radius:1px}
.heard{display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;flex:1 1 auto;min-height:0}
.heard .big{font-size:72px;font-weight:700;letter-spacing:-.04em;line-height:1;font-variant-numeric:tabular-nums;color:var(--ok);min-height:72px}
.heard .big.idle{color:var(--line);font-size:48px}
.heard .sub{color:var(--mut);font-size:13px;margin-top:8px;min-height:1.4em}
.heard .last{font-size:14px;margin-top:4px}
@media(max-width:900px){main{grid-template-columns:1fr;overflow:auto}section+section{border-left:0;border-top:1px solid var(--line)}body{overflow:auto}}
</style></head><body>
<header>
  <h1>Served With a Smile</h1>
  <div class="line">An arm that feeds you. <b>Smile left:</b> fetch. <b>Smile right:</b> hand it over. <b>Open your mouth:</b> feed me. <b>Full smile:</b> let go.</div>
  <div class="right" id="clock"></div>
</header>
<main>
<section>
  <div class="k"><span>What the robot sees</span><span class="tag" id="robot_tag">-</span></div>
  <div class="pic"><img id="robot" src="/robot.jpg" alt="robot camera"></div>
  <div class="card log">
    <div class="k"><span>What it did, as it wrote it (data/serve.log)</span><span class="tag" id="log_tag"></span></div>
    <div class="rows"><table><tbody id="rows"></tbody></table><div class="empty" id="log_empty" style="display:none">nothing yet; serve.py writes a line for every face group it acts on</div></div>
    <div class="count" id="count">picks confirmed <b>0</b>, handovers <b>0</b></div>
  </div>
</section>
<section>
  <div class="k"><span>Your face</span><span class="tag" id="face_tag">-</span></div>
  <div class="pic"><img id="face" src="/face.jpg" alt="your face"></div>
  <div class="bottom">
    <div class="card controls">
      <div class="k">Controls</div>
      <table><tbody id="controls"></tbody></table>
    </div>
    <div style="display:flex;flex-direction:column;gap:10px;min-height:0">
      <div class="card">
        <div class="k"><span>Live, against your calibration</span><span class="tag" id="calib_tag"></span></div>
        <div class="bars">
          <div class="bar"><span class="lab">left</span><span class="tr"><i id="b_left"></i><b id="t_left"></b></span><span class="val" id="v_left">-</span></div>
          <div class="bar"><span class="lab">right</span><span class="tr"><i id="b_right"></i><b id="t_right"></b></span><span class="val" id="v_right">-</span></div>
          <div class="bar"><span class="lab">mouth</span><span class="tr"><i id="b_jaw"></i><b id="t_jaw"></b></span><span class="val" id="v_jaw">-</span></div>
        </div>
      </div>
      <div class="card heard">
        <div class="k" style="align-self:stretch">Hearing</div>
        <div class="big idle" id="heard">...</div>
        <div class="sub" id="heard_sub">listening</div>
        <div class="last" id="last"></div>
      </div>
    </div>
  </div>
</section>
</main>
<script>
const CONTROLS=%CONTROLS%;
document.getElementById('controls').innerHTML=CONTROLS.map(c=>c[1]==null?`<tr><td class="m" colspan="3">${c[0]}</td></tr>`:`<tr><td class="g">${c[0]}</td><td class="f">${c[1]}</td><td class="a">${c[2]}</td></tr>`).join('');
const $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function bar(name,val,thr){
  const b=$('b_'+name),t=$('t_'+name),v=$('v_'+name);
  const have=typeof val==='number';
  b.style.width=have?(Math.min(val,1)*100)+'%':'0';
  b.className=(have&&typeof thr==='number'&&val>=thr)?'on':'';
  t.style.display=typeof thr==='number'?'':'none'; if(typeof thr==='number')t.style.left=(Math.min(thr,1)*100)+'%';
  v.textContent=have?val.toFixed(2):'-';
}
async function tick(){
  let s; try{s=await (await fetch('/state')).json();}catch(e){return;}
  const rows=s.serve||[];
  $('rows').innerHTML=rows.map(r=>`<tr><td class="t">${r.time}</td><td class="f">${esc(r.face)}</td><td class="a">${esc(r.action)}</td></tr>`).join('');
  $('log_empty').style.display=rows.length?'none':'';
  $('log_tag').textContent=rows.length?(s.counts.lines>rows.length?`last ${rows.length} of ${s.counts.lines}`:`${rows.length} lines`):'';
  $('count').innerHTML=`picks confirmed <b>${s.counts.picks}</b>, handovers <b>${s.counts.handovers}</b>`;
  const rb=s.robot||{};
  $('robot_tag').textContent=rb.source==='box'?'live from the box':rb.source==='file'?`last picture on disk, ${fmtAge(rb.age)} old`:'robot offline';
  $('robot_tag').className='tag '+(rb.source==='box'?'ok':'warn');
  const f=s.face||{}, c=s.calib||{};
  const fresh=typeof s.face_age==='number'&&s.face_age<2;
  $('face_tag').textContent=fresh?(f.left==null?'camera on, no face seen':'face seen'):(s.face_age==null?'smile.py not running':`smile.py stopped ${fmtAge(s.face_age)} ago`);
  $('face_tag').className='tag '+(fresh&&f.left!=null?'ok':'warn');
  bar('left',fresh?f.left:null,c.up_left); bar('right',fresh?f.right:null,c.up_right); bar('jaw',fresh?f.jaw:null,c.jaw_open);
  $('calib_tag').textContent=c.at?`calibrated ${c.at.slice(0,10)===new Date().toISOString().slice(0,10)||c.at.slice(0,10)===localDay()?c.at.slice(11):c.at}`:'no calibration (defaults)';
  const g=fresh?(f.group||''):'', holding=fresh?f.holding:null;
  const h=$('heard');
  if(g){h.textContent=g;h.className='big';$('heard_sub').textContent=wordsOf(g);}
  else if(holding){h.textContent=holding;h.className='big';$('heard_sub').textContent='holding '+wordsOf(holding);}
  else{h.textContent='...';h.className='big idle';$('heard_sub').textContent=fresh?'listening':'';}
  const l=s.last_heard; $('last').textContent=l?`last heard ${l.group} (${l.face}) ${fmtAge(l.age)} ago`:'';
}
function wordsOf(g){const w={L:'smile left',R:'smile right',B:'full smile',M:'mouth open'};return g.split('').map(ch=>w[ch]||ch).join(', ');}
function localDay(){const d=new Date();return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`;}
function fmtAge(a){if(a==null)return '?';if(a<60)return a.toFixed(0)+' s';if(a<3600)return (a/60).toFixed(0)+' min';return (a/3600).toFixed(1)+' h';}
function swap(id,url){const img=$(id);const n=new Image();n.onload=()=>{img.src=n.src;};n.src=url+'?'+Date.now();}
setInterval(()=>swap('robot','/robot.jpg'),150);
setInterval(()=>swap('face','/face.jpg'),120);
setInterval(tick,250);tick();
setInterval(()=>{$('clock').textContent=new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});},1000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype="application/json", extra=None):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        try:
            if path == "/robot.jpg":
                body, ctype, source, age = robot_picture()
                self._send(body, ctype, {"X-Source": source, "X-Age": f"{age:.1f}" if age is not None else "none"})
            elif path == "/face.jpg":
                body, ctype, age = face_picture()
                self._send(body, ctype, {"X-Age": f"{age:.1f}" if age is not None else "none"})
            elif path == "/state":
                self._send(json.dumps(state()))
            elif path == "/":
                self._send(PAGE.replace("%CONTROLS%", json.dumps(CONTROLS)), "text/html; charset=utf-8")
            else:
                self.send_response(404); self.send_header("Content-Length", "0"); self.end_headers()
        except (BrokenPipeError, ConnectionResetError):
            pass


if __name__ == "__main__":
    port = int(os.environ.get("DASH_PORT", "8790"))
    threading.Thread(target=robot_poll, daemon=True).start()
    print(f"Served With a Smile: demo screen on http://localhost:{port}  (robot picture from {ROBOT_URL})", flush=True)
    ThreadingHTTPServer((os.environ.get("DASH_BIND", "0.0.0.0"), port), H).serve_forever()
