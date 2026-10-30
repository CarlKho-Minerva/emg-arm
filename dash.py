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
    ("Face controls (one state, no modes)", None, None),
    ("L", "left cheek up", "fetch: home, find the green cube, pick, prove the grip, greet"),
    ("R", "right cheek up", "feed Skully: look twice; if he held still and is in reach, cube into the mouth, let go, back"),
    ("B", "full smile", "let go"),
    ("BB", "smile twice", "home, gripper open (stop)"),
    ("Why the robot can refuse", None, None),
    ("-", "Skully moved", "more than 20 mm between two looks a second apart: refused, cube kept"),
    ("-", "out of reach", "nearer than 530 mm or further than 800 mm from the base: refused"),
    ("-", "empty hand", "feed asked with nothing in the gripper: refused"),
]

robot = {"jpeg": None, "t": 0.0, "source": "none", "error": ""}

# ---- the processes the screen can start and stop (Carl, 30 Oct 14:43: "I don't want to run both smile.py and serve.py myself")
import signal
import subprocess
import sys
PY = os.path.join(HERE, ".venv", "bin", "python")
PROGRAMS = {
    "smile": {"label": "Face reader", "cmd": [PY, "smile.py", "--no-window"], "log": "smile.out"},
    "serve": {"label": "Arm loop", "cmd": [PY, "serve.py", "--live", "--cube", os.environ.get("CUBE", "green")], "log": "serve.out"},
    "view": {"label": "Wrist camera", "cmd": [PY, "view.py"], "log": "view.out"},
}
procs = {}
plock = threading.Lock()


def _alive(name):
    pr = procs.get(name)
    return pr is not None and pr.poll() is None


def _others_running(name):
    """Pids of the same program started elsewhere (a Terminal window), so the screen never runs two face readers."""
    try:
        out = subprocess.run(["pgrep", "-f", f"python {PROGRAMS[name]['cmd'][1]}"], capture_output=True, text=True).stdout.split()
    except Exception:
        return []
    mine = procs.get(name).pid if _alive(name) else None
    return [int(x) for x in out if x.isdigit() and int(x) != mine and int(x) != os.getpid()]


def start(name):
    with plock:
        if _alive(name):
            return "already running"
        others = _others_running(name)
        if others:
            return f"already running elsewhere (pid {others[0]}); stop that window first"
        env = dict(os.environ, GLOG_minloglevel="2", TF_CPP_MIN_LOG_LEVEL="2", PYTHONUNBUFFERED="1")
        log = open(os.path.join(DATA, PROGRAMS[name]["log"]), "ab")
        procs[name] = subprocess.Popen(PROGRAMS[name]["cmd"], cwd=HERE, stdout=log, stderr=subprocess.STDOUT, env=env)
        return f"started pid {procs[name].pid}"


def stop(name):
    with plock:
        pr = procs.get(name)
        pids = ([pr.pid] if _alive(name) else []) + _others_running(name)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        t0 = time.time()
        while time.time() - t0 < 3 and any(_pid_alive(x) for x in pids):
            time.sleep(0.1)
        for pid in pids:
            if _pid_alive(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        return f"stopped {len(pids)}" if pids else "not running"


def _pid_alive(pid):
    try:
        os.kill(pid, 0); return True
    except OSError:
        return False


def programs_status():
    out = {}
    for name, pg in PROGRAMS.items():
        others = _others_running(name)
        out[name] = {"label": pg["label"], "running": _alive(name) or bool(others), "pid": (procs[name].pid if _alive(name) else (others[0] if others else None)),
                     "where": "screen" if _alive(name) else ("terminal" if others else None), "log": tail(os.path.join(DATA, pg["log"]), 2)}
    return out


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


def episode_stats():
    try:
        import episodes
        return episodes.stats()
    except Exception as e:
        return {"error": str(e)[:80]}


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
        "episodes": episode_stats(),
        "skully": read_json(os.path.join(DATA, "skully.json")),
        "smile_out": tail(os.path.join(DATA, "smile.out"), 4),
        "mode": read_json(os.path.join(DATA, "mode.json")),
        "programs": programs_status(),
    }


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Served With a Smile</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--fg:#111;--mut:#666;--line:#e5e5e5;--bg:#fff;--card:#fafafa;--ok:#0a7d3b;--warn:#b45309;--bad:#b3261e;--acc:#2a78d6;--surf:#fcfcfb}
@media(prefers-color-scheme:dark){:root{--fg:#ededed;--mut:#8a8a8a;--line:#2a2a2a;--bg:#0a0a0a;--card:#121212;--ok:#2fbf71;--warn:#f59e0b;--bad:#f0756b;--acc:#3987e5;--surf:#1a1a19}}
*{box-sizing:border-box}html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 -apple-system,Inter,system-ui,sans-serif;display:grid;grid-template-columns:232px 1fr;overflow:hidden}
nav{border-right:1px solid var(--line);padding:18px 14px;display:flex;flex-direction:column;gap:4px}
nav h1{font-size:19px;font-weight:600;letter-spacing:-.02em;margin:0 8px 2px}
nav .sub{color:var(--mut);font-size:12.5px;margin:0 8px 14px;line-height:1.35}
nav button{all:unset;cursor:pointer;display:flex;align-items:center;justify-content:space-between;padding:9px 10px;border-radius:8px;font-size:14.5px;color:var(--fg)}
nav button:hover{background:var(--card)}nav button.on{background:var(--fg);color:var(--bg)}
nav button .dot{width:8px;height:8px;border-radius:50%;background:var(--line)}nav button .dot.ok{background:var(--ok)}nav button .dot.warn{background:var(--warn)}
nav button.on .dot.ok{background:var(--ok)}
.runbox{margin-top:auto;border-top:1px solid var(--line);padding-top:4px}
.runrow{display:grid;grid-template-columns:10px 1fr auto auto;gap:6px;align-items:center;padding:3px 8px;font-size:13px}
.rb{all:unset;cursor:pointer;font-size:11.5px;padding:3px 8px;border:1px solid var(--line);border-radius:6px;color:var(--fg)}.rb:hover{background:var(--card)}.rb.all{border-color:var(--ok);color:var(--ok)}
nav .foot{color:var(--mut);font-size:12px;line-height:1.4;padding:0 8px}
nav .foot b{color:var(--fg);font-weight:500}
main{min-height:0;overflow:auto;padding:20px 28px 28px}
section{display:none;flex-direction:column;gap:14px;min-height:100%}section.on{display:flex}
.k{font-size:12px;color:var(--mut);letter-spacing:.04em;text-transform:uppercase;display:flex;align-items:baseline;gap:10px}
.k .tag{margin-left:auto;text-transform:none;letter-spacing:0;font-variant-numeric:tabular-nums}.k .tag.ok{color:var(--ok)}.k .tag.warn{color:var(--warn)}
h2{font-size:22px;font-weight:600;letter-spacing:-.02em;margin:0}
.lead{color:var(--mut);font-size:15px;max-width:72ch;margin:0}
.pic{border:1px solid var(--line);border-radius:10px;overflow:hidden;background:#000;aspect-ratio:4/3;max-height:62vh}
.pic img{display:block;width:100%;height:100%;object-fit:contain;background:#000}
.card{border:1px solid var(--line);border-radius:10px;background:var(--card);padding:12px 16px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:14px}
table{width:100%;border-collapse:collapse;font-size:13.5px;font-variant-numeric:tabular-nums}
td{padding:4px 8px 4px 0;border-bottom:1px solid var(--line);vertical-align:top;white-space:nowrap}td:last-child{white-space:normal}tr:last-child td{border-bottom:0}
td.t{color:var(--mut);font:12.5px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;width:70px}
td.f{font-weight:500;width:150px}td.a{color:var(--mut)}
td.ok{color:var(--ok);font-weight:600;width:16px}td.no{color:var(--bad);font-weight:600;width:16px}td.na{color:var(--mut);width:16px}
td.g{font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace;width:42px;color:var(--mut)}
.checks td.name{font-weight:500;width:110px}
.tiles{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}
.tile{border:1px solid var(--line);border-radius:8px;padding:10px 12px;background:var(--bg)}
.tile .n{font-size:28px;font-weight:600;letter-spacing:-.02em;font-variant-numeric:tabular-nums}.tile .l{font-size:11.5px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
.bars{display:flex;flex-direction:column;gap:10px}
.bar{display:grid;grid-template-columns:56px 1fr 48px;align-items:center;gap:10px;font-size:13.5px}.bar .lab{color:var(--mut)}.bar .val{text-align:right;font-variant-numeric:tabular-nums;color:var(--mut)}
.bar .tr{position:relative;height:10px;background:var(--line);border-radius:5px}.bar .tr i{position:absolute;left:0;top:0;height:100%;background:var(--mut);border-radius:5px;width:0;transition:width .1s linear}.bar .tr i.on{background:var(--ok)}
.bar .tr b{position:absolute;top:-4px;width:2px;height:18px;background:var(--fg);margin-left:-1px;border-radius:1px}
.heard{text-align:center;padding:18px 0 8px}.heard .big{font-size:64px;font-weight:700;letter-spacing:-.04em;line-height:1;color:var(--ok);min-height:64px}.heard .big.idle{color:var(--line);font-size:44px}.heard .sub{color:var(--mut);font-size:13px;margin-top:8px}
svg text{fill:var(--mut);font:11px -apple-system,Inter,system-ui,sans-serif}svg .ax{stroke:var(--line)}svg .pt{fill:var(--acc)}svg .pt.h{fill:none;stroke:var(--acc);stroke-width:1.5}
svg .ok{fill:var(--ok)}svg .no{fill:var(--bad)}svg .na{fill:var(--mut)}
.cap{color:var(--mut);font-size:12.5px;max-width:80ch;margin:4px 0 0}
.cc{display:grid;grid-template-columns:1.2fr 1.2fr 1fr;gap:16px;align-items:start}
.cc .pic{aspect-ratio:4/3;max-height:40vh}
.cc .heard{padding:10px 0 6px}.cc .heard .big{font-size:44px;min-height:44px}.cc .heard .big.idle{font-size:30px}
.paper{max-width:78ch}.paper h3{font-size:16px;font-weight:600;margin:18px 0 6px}.paper p{margin:0 0 10px;line-height:1.55}.paper .abs{border-left:2px solid var(--line);padding-left:14px;color:var(--fg)}
.paper .fig{margin:12px 0 4px}.paper .cap b{color:var(--fg);font-weight:600}
.story{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(0,1fr);gap:26px;align-items:start}
.refs{font-size:13px;color:var(--mut);line-height:1.5}.refs p{margin:0 0 8px}.refs a{color:var(--fg)}
.story img{width:100%;border-radius:10px;border:1px solid var(--line);display:block}
.story p{margin:0 0 12px;font-size:16px;line-height:1.55;max-width:60ch}
.story .quiet{color:var(--mut);font-size:13.5px}
pre.log{margin:0;font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--mut);white-space:pre-wrap}
@media(max-width:900px){body{grid-template-columns:1fr;overflow:auto}nav{flex-direction:row;flex-wrap:wrap;border-right:0;border-bottom:1px solid var(--line)}nav .foot{display:none}.row,.story{grid-template-columns:1fr}.tiles{grid-template-columns:repeat(2,1fr)}}
</style></head><body>
<nav>
  <h1>Served With a Smile</h1>
  <div class="sub">Meet Skully. He's dead hungry.</div>
  <div class="sub" id="layer" style="color:var(--ok);font-weight:600"></div>
  <button data-tab="all" class="on"><span>Command centre</span><span class="dot" id="d_all"></span></button>
  <button data-tab="robot"><span>What the robot sees</span><span class="dot" id="d_robot"></span></button>
  <button data-tab="face"><span>Your face</span><span class="dot" id="d_face"></span></button>
  <button data-tab="memory"><span>Episodic memory</span><span class="dot" id="d_memory"></span></button>
  <button data-tab="story"><span>Why</span><span class="dot"></span></button>
  <div class="runbox">
    <div class="k" style="margin:10px 8px 6px">Run</div>
    <div class="runrow"><span id="p_smile_dot" class="dot"></span><span>Face reader</span><button class="rb" data-run="smile" data-act="start">Start</button><button class="rb" data-run="smile" data-act="stop">Stop</button></div>
    <div class="runrow"><span id="p_serve_dot" class="dot"></span><span>Arm loop</span><button class="rb" data-run="serve" data-act="start">Start</button><button class="rb" data-run="serve" data-act="stop">Stop</button></div>
    <div class="runrow"><span id="p_view_dot" class="dot"></span><span>Wrist camera</span><button class="rb" data-run="view" data-act="start">Start</button><button class="rb" data-run="view" data-act="stop">Stop</button></div>
    <div class="runrow"><span class="dot" style="background:none"></span><span></span><button class="rb all" data-run="all" data-act="start">Start all</button><button class="rb" data-run="all" data-act="stop">Stop all</button></div>
    <div class="cap" id="run_msg" style="margin:4px 8px 0;font-size:11.5px"></div>
  </div>
  <div class="foot"><b>Serve:</b> left cheek fetch, right cheek feed Skully, smile let go. <b>Two smiles</b> switch to Steer: cheeks turn the arm, smile grabs or lets go. <b>Three smiles</b> home.<br><br><span id="clock"></span></div>
</nav>
<main>
<section id="all" class="on">
  <div class="k"><span>Command centre</span><span class="tag" id="all_layer"></span></div>
  <div class="cc">
    <div>
      <div class="k"><span>Wrist camera</span><span class="tag" id="robot_tag2">-</span></div>
      <div class="pic"><img id="robot2" src="/robot.jpg" alt="robot camera"></div>
      <div class="card checks" style="margin-top:10px"><div class="k"><span>What the feed requires</span><span class="tag" id="checks_tag2"></span></div><table><tbody id="checks2"></tbody></table></div>
    </div>
    <div>
      <div class="k"><span>Laptop camera</span><span class="tag" id="face_tag2">-</span></div>
      <div class="pic"><img id="face2" src="/face.jpg" alt="your face"></div>
      <div class="card heard" style="margin-top:10px"><div class="big idle" id="heard2">...</div><div class="sub" id="heard_sub2"></div></div>
      <div class="card" style="margin-top:10px"><div class="k">This layer</div><table><tbody id="controls2"></tbody></table></div>
    </div>
    <div>
      <div class="k"><span>Episodic memory</span><span class="tag" id="mongo_tag2">-</span></div>
      <div class="tiles" style="grid-template-columns:repeat(2,1fr)">
        <div class="tile"><div class="n" id="t2_total">0</div><div class="l">episodes</div></div>
        <div class="tile"><div class="n" id="t2_today">0</div><div class="l">live today</div></div>
        <div class="tile"><div class="n" id="t2_fetch">0/0</div><div class="l">fetch ok</div></div>
        <div class="tile"><div class="n" id="t2_feed">0/0</div><div class="l">Skully fed</div></div>
      </div>
      <div class="card" style="margin-top:10px"><div class="k">What it did, as it wrote it</div><table><tbody id="rows2"></tbody></table></div>
      <div class="card" style="margin-top:10px"><div class="k">The last episodes</div><table><tbody id="eps2"></tbody></table></div>
    </div>
  </div>
</section>
<section id="robot">
  <div><h2>What the robot sees</h2><p class="lead">The wrist camera, as the arm sees it, and every line the arm wrote as it acted. The feed runs only when five things are true; each is checked in order and shown here.</p></div>
  <div class="row">
    <div>
      <div class="k"><span>Wrist camera</span><span class="tag" id="robot_tag">-</span></div>
      <div class="pic"><img id="robot" src="/robot.jpg" alt="robot camera"></div>
    </div>
    <div style="display:flex;flex-direction:column;gap:14px;min-width:0">
      <div class="card checks"><div class="k"><span>What the feed requires</span><span class="tag" id="checks_tag"></span></div><table><tbody id="checks"></tbody></table></div>
      <div class="card"><div class="k"><span>What it did, as it wrote it</span><span class="tag" id="log_tag"></span></div><table><tbody id="rows"></tbody></table><div class="cap" id="log_empty" style="display:none">nothing yet; serve.py writes a line for every face event it acts on</div></div>
    </div>
  </div>
</section>
<section id="face">
  <div><h2>Your face is the controller</h2><p class="lead">The laptop camera turns a face into 52 numbers thirty times a second. A small model trained on 20 seconds of one person's face turns those into four events, counted in groups. The arm reads the events; it never sees this picture.</p></div>
  <div class="row">
    <div>
      <div class="k"><span>Laptop camera</span><span class="tag" id="face_tag">-</span></div>
      <div class="pic"><img id="face" src="/face.jpg" alt="your face"></div>
      <pre class="log" id="smile_out" style="margin-top:8px"></pre>
    </div>
    <div style="display:flex;flex-direction:column;gap:14px;min-width:0">
      <div class="card heard"><div class="k" style="text-align:left">Hearing</div><div class="big idle" id="heard">...</div><div class="sub" id="heard_sub">listening</div><div class="cap" id="last"></div></div>
      <div class="card"><div class="k"><span>Live, against the calibration</span><span class="tag" id="calib_tag"></span></div>
        <div class="bars">
          <div class="bar"><span class="lab">left</span><span class="tr"><i id="b_left"></i><b id="t_left"></b></span><span class="val" id="v_left">-</span></div>
          <div class="bar"><span class="lab">right</span><span class="tr"><i id="b_right"></i><b id="t_right"></b></span><span class="val" id="v_right">-</span></div>
          <div class="bar"><span class="lab">mouth</span><span class="tr"><i id="b_jaw"></i><b id="t_jaw"></b></span><span class="val" id="v_jaw">-</span></div>
        </div></div>
      <div class="card"><div class="k">Controls</div><table><tbody id="controls"></tbody></table></div>
    </div>
  </div>
</section>
<section id="memory">
  <div class="paper">
    <h2>What a Robot Remembers When It Feeds a Skull</h2>
    <p class="cap">Carl Kho, Somach Systems. Viam x MongoDB Robot Hackathon, San Francisco, 29 to 30 October 2026. Every number below is read from the database at the moment you look at it.</p>
    <h3>Abstract</h3>
    <p class="abs">A robot that acts on a facial gesture has to answer a plain question afterwards: what did it see, what did it decide, and did it work. We store one document per command in MongoDB Atlas: the gesture, what each camera saw, which safety checks passed, the outcome and the duration. Thursday's picks, hand-following runs and handovers were imported with their original timestamps and marked as imported, so the record covers the whole build and not only the demo. Below, the counts, the follower's tracking error run by run, and every episode on a timeline.</p>
  </div>
  <div class="k"><span>MongoDB</span><span class="tag" id="mongo_tag">-</span></div>
  <div class="tiles">
    <div class="tile"><div class="n" id="t_total">0</div><div class="l">episodes</div></div>
    <div class="tile"><div class="n" id="t_fetch">0/0</div><div class="l">fetch, picked and proved</div></div>
    <div class="tile"><div class="n" id="t_feed">0/0</div><div class="l">Skully fed</div></div>
    <div class="tile"><div class="n" id="t_refused">0</div><div class="l">feeds refused</div></div>
    <div class="tile"><div class="n" id="t_today">0</div><div class="l">live today</div></div>
  </div>
  <div class="paper">
    <h3>1. Method</h3>
    <p>The face reader writes a named event to a file; the arm loop reads it and starts one behaviour. The behaviour opens an episode when it starts and closes it when it ends, with a reason in words. The record is written to a local file first and to Atlas on the next successful connection, so a dropped network loses nothing and a record never exists in one place without the other for long. Outcomes are recorded only where the program could measure one: a cube proven in the gripper by the sideways shift, a feed completed, a refusal with its cause. Where the day's logs carried no verdict, the field is left empty rather than guessed.</p>
    <h3>2. Results</h3>
  </div>
  <div class="row">
    <div class="card"><div class="k">Figure 1. The follower's tracking error, run by run</div><div id="c_follow"></div>
      <p class="cap"><b>Figure 1.</b> Forty 30-second runs on 29 October, one panel per kind of hand movement. Each dot is a run; its height is the typical distance of the hand from the centre of the picture (root mean square, as a fraction of the frame; lower is better). Hollow dots are held-out runs the controller was never tuned on. These runs chose the gains in a simulator that replays them: the kept setting scored -0.038 held out, the runner-up -0.066.</p></div>
    <div class="card"><div class="k">Figure 2. Every episode, in time</div><div id="c_time"></div>
      <p class="cap"><b>Figure 2.</b> Each mark is one episode at the minute it started, grouped by what the arm was asked to do. A filled green mark ended as intended; a red diamond did not; grey has no outcome on disk (a recording with no verdict, or a practice round whose grip check was later found untrustworthy). Hover for the reason.</p></div>
  </div>
  <div class="card"><div class="k"><span>Table 1. The last episodes</span><span class="tag" id="eps_tag"></span></div><table><tbody id="eps"></tbody></table><p class="cap" id="mongo"></p></div>
  <div class="paper">
    <h3>3. What this does and does not show</h3>
    <p>It shows that every action the arm took today is accounted for, with its cause, and that Thursday's tuning data is the same kind of record as today's demo. It does not show a learned policy: the behaviours are written by hand. The episodes are what a learned version would be trained on and judged against, and the same records are where a comparison of the camera input against the muscle-signal input would be made, gesture by gesture, on detection rate, latency and false triggers.</p>
  </div>
</section>
<section id="story">
  <div><h2>Why a smile</h2></div>
  <div class="story">
    <img src="/story.jpg" alt="A woman in a wheelchair being helped to eat">
    <div>
      <p>For most of us, bringing food to the mouth is the part of eating we never notice. For someone with a cervical spinal cord injury, the cord is damaged in the neck, the arms and hands are mostly paralysed, and that part belongs to another person.</p>
      <p>The face usually still works. A half-smile is a decision. This robot takes the decision from the face and does the movement itself.</p>
      <p>Skully stands in for the person while the feed is tested. The next input is the electrical signal of the facial muscles, read by two pads on the cheeks, for a face that barely moves. The pads and the board are on the table; the arm side does not change.</p>
      <p class="quiet">Stock photograph; the person shown is not a patient of this project and no diagnosis is implied. Skully is a Halloween skull.</p>
      <div class="refs">
        <p><b>What others have shown.</b> A hands-free feeding interface driven by eye and head movements, with mouth movements as the command, was tested in 2022 with four able-bodied participants and one person with a spinal cord injury: 89.1 percent of bites succeeded, 31.4 s per item, and the participant with the injury rated usability 90 of 100 (Schultz, Slifkin, Yu and Schearer, IEEE ICORR 2022, <a href="https://doi.org/10.1109/ICORR55369.2022.9896535">doi:10.1109/ICORR55369.2022.9896535</a>).</p>
        <p>Taking a feeding robot out of the lab, a 2025 study with two co-designers with motor impairments, five participants eating in a cafeteria, office or meeting room, and one person using it at home for ten meals, found that the unexpected situations of ordinary life are the central problem, and that users get through them when the system gives them control (Nanavati et al., ACM/IEEE HRI 2025, <a href="https://hcrlab.cs.washington.edu/publications/nanavati2025lessons">hcrlab.cs.washington.edu</a>).</p>
        <p>This prototype takes the control point from both: the person issues the command, the robot checks before it acts and says why when it refuses. It stops at the mouth of a skull; feeding a person needs force limits, a breakaway utensil and swallowing safety that are not built here.</p>
      </div>
    </div>
  </div>
</section>
</main>
<script>
const CONTROLS=%CONTROLS%;
const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
document.getElementById('controls').innerHTML=CONTROLS.map(c=>c[1]==null?`<tr><td class="a" colspan="3" style="padding-top:8px;font-size:11.5px;text-transform:uppercase;letter-spacing:.04em">${c[0]}</td></tr>`:`<tr><td class="g">${c[0]}</td><td class="f">${c[1]}</td><td class="a">${c[2]}</td></tr>`).join('');
document.querySelectorAll('nav button[data-tab]').forEach(b=>b.onclick=()=>{document.querySelectorAll('nav button[data-tab]').forEach(x=>x.classList.toggle('on',x===b));document.querySelectorAll('main section').forEach(s=>s.classList.toggle('on',s.id===b.dataset.tab));try{localStorage.setItem('tab',b.dataset.tab)}catch(e){}});
try{const t=localStorage.getItem('tab');if(t)document.querySelector(`nav button[data-tab="${t}"]`)?.click();}catch(e){}
document.querySelectorAll('.rb').forEach(b=>b.onclick=async()=>{b.disabled=true;try{const r=await (await fetch(`/run?what=${b.dataset.run}&action=${b.dataset.act}`,{method:'POST'})).json();$('run_msg').textContent=Object.entries(r).map(([k,v])=>`${k}: ${v}`).join(' · ');}catch(e){$('run_msg').textContent='no answer from the screen server';}b.disabled=false;});
document.addEventListener('keydown',e=>{const m={'0':'all','1':'robot','2':'face','3':'memory','4':'story'};if(m[e.key])document.querySelector(`nav button[data-tab="${m[e.key]}"]`).click();});
function bar(name,val,thr){const b=$('b_'+name),t=$('t_'+name),v=$('v_'+name);const have=typeof val==='number';b.style.width=have?(Math.min(val,1)*100)+'%':'0';b.className=(have&&typeof thr==='number'&&val>=thr)?'on':'';t.style.display=typeof thr==='number'?'':'none';if(typeof thr==='number')t.style.left=(Math.min(thr,1)*100)+'%';v.textContent=have?val.toFixed(2):'-';}
function fmtAge(a){if(a==null)return '?';if(a<60)return a.toFixed(0)+' s';if(a<3600)return (a/60).toFixed(0)+' min';return (a/3600).toFixed(1)+' h';}
function wordsOf(g){const w={L:'left cheek',R:'right cheek',B:'full smile',M:'mouth open'};return g.split('').map(ch=>w[ch]||ch).join(', ');}
function hhmm(t){return new Date(t*1000).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'});}
function followChart(runs){
  const tags=['still','slow','fast','cross','exit','rest'];const W=300,H=120,P={l:30,r:8,t:14,b:22};
  const vals=runs.map(r=>r.rms).filter(v=>typeof v==='number');if(!vals.length)return '<p class="cap">no follower runs</p>';
  const ymax=Math.max(0.2,Math.ceil(Math.max(...vals)*20)/20);
  return `<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px">`+tags.map(tag=>{
    const rs=runs.filter(r=>r.tag===tag);const n=Math.max(rs.length,1);
    const x=i=>P.l+(W-P.l-P.r)*(rs.length>1?i/(rs.length-1):0.5), y=v=>P.t+(H-P.t-P.b)*(1-v/ymax);
    const ticks=[0,ymax/2,ymax];
    return `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="${tag}: ${rs.length} runs">
      <text x="${P.l}" y="10" style="fill:var(--fg);font-weight:600">${tag}</text><text x="${W-P.r}" y="10" text-anchor="end">${rs.length} runs</text>
      ${ticks.map(v=>`<line class="ax" x1="${P.l}" x2="${W-P.r}" y1="${y(v)}" y2="${y(v)}"/><text x="${P.l-4}" y="${y(v)+3.5}" text-anchor="end">${v.toFixed(2)}</text>`).join('')}
      ${rs.map((r,i)=>typeof r.rms==='number'?`<circle class="pt ${r.held_out?'h':''}" cx="${x(i)}" cy="${y(r.rms)}" r="4.5"><title>${tag} run ${i+1} at ${hhmm(r.t)}: RMS ${r.rms}, lost ${Math.round((r.lost||0)*100)}% of frames, reward ${r.reward}${r.held_out?' (held out)':''}</title></circle>`:'').join('')}
      <text x="${(P.l+W-P.r)/2}" y="${H-6}" text-anchor="middle">runs in order</text></svg>`;}).join('')+`</div>`;
}
function timeChart(tl){
  if(!tl.length)return '<p class="cap">no episodes</p>';
  const kinds=[...new Set(tl.map(e=>e.kind))];const W=620,rowH=22,P={l:92,r:10,t:18,b:22};const H=P.t+P.b+rowH*kinds.length;
  const t0=Math.min(...tl.map(e=>e.t)),t1=Math.max(...tl.map(e=>e.t))+60;const x=t=>P.l+(W-P.l-P.r)*((t-t0)/(t1-t0));
  const span=t1-t0;const step=Math.max(1800,Math.ceil(span/7/1800)*1800);const ticks=[];for(let t=Math.ceil(t0/step)*step;t<t1;t+=step)ticks.push(t);
  const lab=t=>span>12*3600?new Date(t*1000).toLocaleString([],{weekday:'short',hour:'2-digit',minute:'2-digit'}):hhmm(t);
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="episodes over time">
    ${ticks.map(t=>`<line class="ax" x1="${x(t)}" x2="${x(t)}" y1="${P.t}" y2="${H-P.b}"/><text x="${x(t)}" y="${H-7}" text-anchor="middle">${lab(t)}</text>`).join('')}
    ${kinds.map((k,i)=>`<text x="${P.l-8}" y="${P.t+rowH*i+rowH/2+4}" text-anchor="end" style="fill:var(--fg)">${esc(k)}</text><line class="ax" x1="${P.l}" x2="${W-P.r}" y1="${P.t+rowH*i+rowH/2}" y2="${P.t+rowH*i+rowH/2}"/>`).join('')}
    ${tl.map(e=>{const i=kinds.indexOf(e.kind);const cy=P.t+rowH*i+rowH/2;const cls=e.ok===true?'ok':e.ok===false?'no':'na';const cx=x(e.t);
      const shape=e.ok===true?`<circle class="${cls}" cx="${cx}" cy="${cy}" r="5"/>`:e.ok===false?`<rect class="${cls}" x="${cx-4.5}" y="${cy-4.5}" width="9" height="9" transform="rotate(45 ${cx} ${cy})"/>`:`<circle class="${cls}" cx="${cx}" cy="${cy}" r="4" fill-opacity=".55"/>`;
      return `<g>${shape}<title>${hhmm(e.t)} ${esc(e.kind)}: ${e.ok===true?'done':e.ok===false?'not done':'no outcome on disk'}. ${esc(e.reason)}${e.duration_s?` (${e.duration_s} s)`:''}${e.source!=='live'?' ['+esc(e.source)+']':''}</title></g>`;}).join('')}
  </svg>`;
}
let lastSig='';
async function tick(){
  let s; try{s=await (await fetch('/state')).json();}catch(e){return;}
  const rows=s.serve||[];
  $('rows').innerHTML=rows.slice(-8).map(r=>`<tr><td class="t">${r.time}</td><td class="f">${esc(r.face)}</td><td class="a">${esc(r.action)}</td></tr>`).join('');
  $('log_empty').style.display=rows.length?'none':'';$('log_tag').textContent=rows.length?`${s.counts.lines} lines`:'';
  const rb=s.robot||{};const robotLive=rb.source==='box'||(rb.source==='file'&&rb.age<3);
  $('robot_tag').textContent=robotLive?'live':rb.source==='file'?`last picture ${fmtAge(rb.age)} old`:'robot offline';$('robot_tag').className='tag '+(robotLive?'ok':'warn');$('d_robot').className='dot '+(robotLive?'ok':'warn');
  const md=s.mode;$('layer').textContent=md?`Layer: ${md.mode.toUpperCase()}`:'';
  if(md){$('controls').innerHTML=md.controls.map(c=>`<tr><td class="g">${c[0]}</td><td class="f">${esc(c[1])}</td><td class="a">${esc(c[2])}</td></tr>`).join('');}
  const sk=s.skully||{};const checks=sk.checks||[];
  $('checks').innerHTML=(checks.length?checks:[{name:'seen',what:'the blue tape on Skully\'s nose is in view from home'},{name:'in reach',what:'530 to 800 mm from the base'},{name:'holding',what:'a cube is between the fingers'},{name:'re-found',what:'seen again from the hover above him'},{name:'path',what:'the planner finds a way in and out'}]).map(c=>`<tr><td class="${c.ok===true?'ok':c.ok===false?'no':'na'}">${c.ok===true?'&#10003;':c.ok===false?'&#10007;':'·'}</td><td class="name">${esc(c.name)}</td><td class="a">${esc(c.what)}</td></tr>`).join('');
  $('checks_tag').textContent=sk.t?(sk.fed?'fed, '+fmtAge(Date.now()/1000-sk.t)+' ago':sk.seen===false?'not seen, '+fmtAge(Date.now()/1000-sk.t)+' ago':(sk.dist?`Skully ${sk.dist.toFixed(0)} mm from the base, `:'')+fmtAge(Date.now()/1000-sk.t)+' ago'):'no attempt yet';
  const f=s.face||{}, c=s.calib||{};const fresh=typeof s.face_age==='number'&&s.face_age<2;
  $('face_tag').textContent=fresh?(f.left==null?'camera on, no face seen':'face seen'):(s.face_age==null?'smile.py not running':`smile.py stopped ${fmtAge(s.face_age)} ago`);
  $('face_tag').className='tag '+(fresh&&f.left!=null?'ok':'warn');$('d_face').className='dot '+(fresh&&f.left!=null?'ok':'warn');
  $('smile_out').textContent=fresh?'':(s.smile_out||[]).join('\n');
  bar('left',fresh?f.left:null,c.up_left);bar('right',fresh?f.right:null,c.up_right);bar('jaw',fresh?f.jaw:null,c.jaw_open);
  $('calib_tag').textContent=c.at?`calibrated ${c.at}`:'no calibration (defaults)';
  const g=fresh?(f.group||''):'',holding=fresh?f.holding:null;const h=$('heard');
  if(g){h.textContent=g;h.className='big';$('heard_sub').textContent=wordsOf(g);}else if(holding){h.textContent=holding;h.className='big';$('heard_sub').textContent='holding '+wordsOf(holding);}else{h.textContent='...';h.className='big idle';$('heard_sub').textContent=fresh?'listening':'';}
  const l=s.last_heard;$('last').textContent=l?`last heard ${l.group} (${l.face}) ${fmtAge(l.age)} ago`:'';
  const ep=s.episodes||{},bk=ep.by_kind||{};const k=n=>bk[n]||{attempts:0,ok:0,refused:0};
  $('t_total').textContent=ep.total||0;$('t_fetch').textContent=`${k('fetch').ok}/${k('fetch').attempts}`;$('t_feed').textContent=`${k('feed_skully').ok}/${k('feed_skully').attempts}`;$('t_refused').textContent=k('feed_skully').refused;$('t_today').textContent=ep.live||0;
  const mg=ep.mongo||{};$('mongo_tag').textContent=mg.ok?`${mg.which==='local'?'local MongoDB':'Atlas'} connected, ${mg.synced} synced`:`MongoDB unreachable; ${mg.queued} queued on disk`;$('mongo_tag').className='tag '+(mg.ok?'ok':'warn');$('d_memory').className='dot '+(mg.ok?'ok':'warn');
  $('mongo').textContent=`${mg.db||''}.${mg.collection||''}. The file data/serve_episodes.jsonl is written first; every unsynced document goes up on the next connection.`;
  const last=ep.last||[];$('eps_tag').textContent=`${ep.live||0} live today, ${ep.backfill||0} backfilled`;
  $('eps').innerHTML=last.map(e=>`<tr><td class="t">${e.t_start?new Date(e.t_start*1000).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit',second:'2-digit'}):''}</td><td class="${e.ok===true?'ok':e.ok===false?'no':'na'}">${e.ok===true?'&#10003;':e.ok===false?'&#10007;':'?'}</td><td class="f">${esc(e.kind)}</td><td class="g">${esc(e.trigger)}</td><td class="a">${esc(e.reason)}${e.duration_s?` (${e.duration_s} s)`:''}${e.source&&e.source!=='live'?` <i>${esc(e.source)}</i>`:''}</td></tr>`).join('');
  const pg=s.programs||{};for(const n of ['smile','serve','view']){const d=$('p_'+n+'_dot');if(d)d.className='dot '+(pg[n]&&pg[n].running?'ok':'warn');}
  // the command centre mirrors
  $('robot_tag2').textContent=$('robot_tag').textContent;$('robot_tag2').className=$('robot_tag').className;
  $('face_tag2').textContent=$('face_tag').textContent;$('face_tag2').className=$('face_tag').className;
  $('checks2').innerHTML=$('checks').innerHTML;$('checks_tag2').textContent=$('checks_tag').textContent;
  $('heard2').textContent=h.textContent;$('heard2').className=h.className;$('heard_sub2').textContent=$('heard_sub').textContent;
  $('controls2').innerHTML=$('controls').innerHTML;$('all_layer').textContent=md?`layer ${md.mode.toUpperCase()}`:'arm loop not running';$('all_layer').className='tag '+(md?'ok':'warn');
  $('mongo_tag2').textContent=$('mongo_tag').textContent;$('mongo_tag2').className=$('mongo_tag').className;
  $('t2_total').textContent=ep.total||0;$('t2_today').textContent=ep.live||0;$('t2_fetch').textContent=$('t_fetch').textContent;$('t2_feed').textContent=$('t_feed').textContent;
  $('rows2').innerHTML=rows.slice(-6).map(r=>`<tr><td class="t">${r.time}</td><td class="f">${esc(r.face)}</td><td class="a">${esc(r.action)}</td></tr>`).join('');
  $('eps2').innerHTML=last.slice(0,5).map(e=>`<tr><td class="t">${e.t_start?new Date(e.t_start*1000).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'}):''}</td><td class="${e.ok===true?'ok':e.ok===false?'no':'na'}">${e.ok===true?'&#10003;':e.ok===false?'&#10007;':'?'}</td><td class="f">${esc(e.kind)}</td><td class="a">${esc(e.reason)}</td></tr>`).join('');
  $('d_all').className='dot '+((robotLive&&fresh&&mg.ok)?'ok':'warn');
  const sig=`${ep.total}|${(ep.follow_runs||[]).length}`;if(sig!==lastSig){lastSig=sig;$('c_follow').innerHTML=followChart(ep.follow_runs||[]);$('c_time').innerHTML=timeChart(ep.timeline||[]);}
}
function swap(id,url){const img=$(id);const n=new Image();n.onload=()=>{img.src=n.src;};n.src=url+'?'+Date.now();}
setInterval(()=>{swap('robot','/robot.jpg');swap('robot2','/robot.jpg')},150);setInterval(()=>{swap('face','/face.jpg');swap('face2','/face.jpg')},120);setInterval(tick,300);tick();
setInterval(()=>{$('clock').textContent=new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'});},1000);
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

    def do_POST(self):
        import urllib.parse
        path, _, qs = self.path.partition("?")
        q = urllib.parse.parse_qs(qs)
        if path == "/run":
            name, action = q.get("what", [""])[0], q.get("action", [""])[0]
            if name == "all":
                names = ["view", "smile", "serve"] if action == "start" else ["serve", "smile"]
            else:
                names = [name]
            res = {n: (start(n) if action == "start" else stop(n)) for n in names if n in PROGRAMS}
            self._send(json.dumps(res))
        else:
            self.send_response(404); self.send_header("Content-Length", "0"); self.end_headers()

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        try:
            if path == "/robot.jpg":
                body, ctype, source, age = robot_picture()
                self._send(body, ctype, {"X-Source": source, "X-Age": f"{age:.1f}" if age is not None else "none"})
            elif path == "/face.jpg":
                body, ctype, age = face_picture()
                self._send(body, ctype, {"X-Age": f"{age:.1f}" if age is not None else "none"})
            elif path == "/story.jpg":
                try:
                    self._send(open(os.path.join(DATA, "story.jpg"), "rb").read(), "image/jpeg")
                except OSError:
                    self.send_response(404); self.send_header("Content-Length", "0"); self.end_headers()
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
