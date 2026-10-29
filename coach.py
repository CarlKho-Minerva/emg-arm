"""The episode coach: a page that tells Carl what to do, runs the episode, scores it, moves on.

    .venv/bin/python coach.py          # then open http://localhost:8787

One screen. It shows which episode is next (of 30), what to do with the hand for 30 seconds, a
Start button, a countdown while the follower runs, and the episode's reward when it ends. Every
episode is recorded by follow.py as usual; the coach only sequences them and keeps the score sheet
in data/episodes.json. Nothing here touches the robot except by launching follow.py.
"""
import glob
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, ".venv", "bin", "python")
SHEET = os.path.join(HERE, "data", "episodes.json")
SECONDS = 30
FOLLOW = ["--live", "--arm", "left", "--hand", "watch", "--lost-go", "greet",
          "--speed", "60", "--accel", "200", "--seconds", str(SECONDS)]

PLAN = [
    ("still", "Hold the left hand still at the centre of the picture for 10 s, then still at the left edge for 10 s, then at the right edge for 10 s.", "Jitter at rest; the dead zone."),
    ("slow", "Sweep the left hand from the left edge to the right edge and back, slowly, about 5 s per sweep, for the whole 30 s.", "Tracking at the hand's usual speed."),
    ("fast", "Same sweep, fast: about 1 s per sweep. Keep it up for 30 s.", "The burst speeds the simulator saw."),
    ("cross", "Both hands up. Cross the left hand over the right and back, three times, slowly.", "Hand swaps."),
    ("exit", "Drift the left hand out of the picture to the left, keep it out 3 s, bring it back. Then the same to the right. Repeat.", "Coasting, search, the elbow guess."),
    ("rest", "Left hand lies on the keyboard and types. Right hand gestures freely.", "Staying on the left hand when the right one is busier."),
]
PER_TAG = 5
ORDER = [tag for tag, _, _ in PLAN for _ in range(PER_TAG)]   # still x5, slow x5, ...

state = {"running": False, "started": 0.0, "last": None, "proc": None, "log": ""}


def sheet():
    if os.path.exists(SHEET):
        return json.load(open(SHEET))
    return []


def save(rows):
    os.makedirs(os.path.dirname(SHEET), exist_ok=True)
    json.dump(rows, open(SHEET, "w"), indent=1)


def next_index():
    return len(sheet())


def score_run(run_dir):
    """The episode's reward from the simulator's scorer, on what actually happened (not a replay)."""
    sys.path.insert(0, HERE)
    import sim
    import csv
    rows = list(csv.DictReader(open(os.path.join(run_dir, "frames.csv"))))
    errs, steps = [], []
    for r in rows:
        errs.append(float(r["err_x"]) if r["err_x"] else None)
        steps.append(float(r["step_x"]) if r["step_x"] else 0.0)
    # on a real run a frame with no hand IS a lost frame (the simulator's "lost" is error > half a width)
    seen = sum(1 for e in errs if e is not None)
    if not seen:
        return {"reward": None, "frames": len(rows), "seen": 0}
    total, prev = 0.0, 0.0
    for e, u in zip(errs, steps):
        if e is None:
            total -= sim.LOST_PENALTY
        else:
            total -= e * e + (sim.LOST_PENALTY if abs(e) > 0.5 else 0.0)
        total -= sim.JERK_PENALTY * abs(u - prev); prev = u
    es = [e for e in errs if e is not None]
    return {"reward": round(total / len(rows), 3), "rms": round((sum(e * e for e in es) / len(es)) ** 0.5, 3),
            "lost": round(1 - seen / len(rows), 2), "frames": len(rows), "seen": seen}


def run_episode(tag):
    state.update(running=True, started=time.time(), log="")
    before = set(glob.glob(os.path.join(HERE, "data", "follow", "*")))
    proc = subprocess.Popen([PY, os.path.join(HERE, "follow.py"), *FOLLOW, "--tag", tag],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=HERE)
    state["proc"] = proc
    tail = []
    for line in proc.stdout:
        if "INFO" in line or line.startswith(("W0000", "I0000")) or "viam_rust" in line:
            continue
        tail.append(line.rstrip()); tail = tail[-6:]
        state["log"] = "\n".join(tail)
    proc.wait()
    new = set(glob.glob(os.path.join(HERE, "data", "follow", "*"))) - before
    run_dir = max(new, key=os.path.getmtime) if new else None
    result = {"i": next_index(), "tag": tag, "run": os.path.basename(run_dir) if run_dir else None,
              "held_out": (run_dir or "x")[-1] in "012", "exit": proc.returncode, "at": time.strftime("%H:%M:%S")}
    if run_dir and os.path.exists(os.path.join(run_dir, "frames.csv")):
        result.update(score_run(run_dir))
    rows = sheet(); rows.append(result); save(rows)
    state.update(running=False, last=result, proc=None)


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Episode coach</title>
<style>
:root{--fg:#111;--mut:#666;--line:#e5e5e5;--bg:#fff;--ok:#0a7d3b;--acc:#000}
@media(prefers-color-scheme:dark){:root{--fg:#ededed;--mut:#888;--line:#333;--bg:#0a0a0a;--acc:#fff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 -apple-system,Inter,system-ui,sans-serif}
main{max-width:720px;margin:48px auto;padding:0 24px}
.k{font-size:13px;color:var(--mut);letter-spacing:.02em;text-transform:uppercase}
h1{font-size:28px;margin:4px 0 24px;font-weight:600;letter-spacing:-.02em}
.card{border:1px solid var(--line);border-radius:8px;padding:20px 24px;margin:16px 0}
.big{font-size:56px;font-weight:600;letter-spacing:-.03em;line-height:1;margin:8px 0}
.do{font-size:20px;line-height:1.4;margin:8px 0 4px}
.why{color:var(--mut)}
button{font:inherit;font-weight:500;background:var(--acc);color:var(--bg);border:0;border-radius:6px;padding:12px 20px;cursor:pointer}
button[disabled]{opacity:.4;cursor:default}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin:12px 0}
.bar i{display:block;height:100%;background:var(--acc);width:0;transition:width .5s linear}
pre{font:13px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--mut);white-space:pre-wrap;margin:8px 0 0}
table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}th{color:var(--mut);font-weight:500}
.ok{color:var(--ok)}
</style></head><body><main>
<div class="k">Episode <span id="n">1</span> of 30 <span id="held"></span></div>
<h1 id="tag">still</h1>
<div class="card">
  <div class="k">For 30 seconds</div>
  <div class="do" id="do"></div>
  <div class="why" id="why"></div>
</div>
<div class="card">
  <div class="k">Before you press Start</div>
  <div>Arm at greet, you in the chair, laptop pulled back, one hand on the red button for the whole 30 s.</div>
  <p><button id="start" onclick="start()">Start episode</button> <span id="count" class="k"></span></p>
  <div class="bar"><i id="bar"></i></div>
  <pre id="log"></pre>
</div>
<div class="card" id="lastcard" style="display:none">
  <div class="k">Last episode</div>
  <div class="big" id="reward"></div>
  <div class="why" id="detail"></div>
</div>
<div class="card">
  <div class="k">Score sheet <span id="sum"></span></div>
  <table><thead><tr><th>#</th><th>tag</th><th>reward</th><th>rms</th><th>lost</th><th>seen</th><th>held out</th></tr></thead><tbody id="rows"></tbody></table>
</div>
</main>
<script>
const PLAN=%PLAN%, ORDER=%ORDER%, SECS=%SECS%;
async function tick(){
  const s=await (await fetch('/state')).json();
  const i=s.count; const tag=ORDER[i]||'done';
  document.getElementById('n').textContent=Math.min(i+1,30);
  document.getElementById('tag').textContent=tag;
  const p=PLAN.find(x=>x[0]===tag)||['','All 30 done. Run sim.py and fit_arm.py.',''];
  document.getElementById('do').textContent=p[1]; document.getElementById('why').textContent=p[2];
  document.getElementById('start').disabled=s.running||i>=30;
  document.getElementById('log').textContent=s.log||'';
  if(s.running){const t=Math.max(0,SECS-(Date.now()/1000-s.started));
    document.getElementById('count').textContent=t.toFixed(0)+' s left';
    document.getElementById('bar').style.width=(100*(1-t/SECS))+'%%';}
  else{document.getElementById('count').textContent='';document.getElementById('bar').style.width='0';}
  if(s.last){document.getElementById('lastcard').style.display='';
    document.getElementById('reward').textContent=s.last.reward==null?'no score':s.last.reward.toFixed(3);
    document.getElementById('detail').textContent=`${s.last.tag}: hand seen in ${s.last.seen} of ${s.last.frames} frames, rms ${s.last.rms??'?'}, lost ${s.last.lost??'?'}${s.last.held_out?' (held out)':''}`;}
  const rows=s.sheet; document.getElementById('rows').innerHTML=rows.map((r,k)=>`<tr><td>${k+1}</td><td>${r.tag}</td><td>${r.reward??''}</td><td>${r.rms??''}</td><td>${r.lost??''}</td><td>${r.seen}/${r.frames}</td><td>${r.held_out?'yes':''}</td></tr>`).join('');
  const tr=rows.filter(r=>!r.held_out&&r.reward!=null), ho=rows.filter(r=>r.held_out&&r.reward!=null);
  const m=a=>a.length?(a.reduce((x,r)=>x+r.reward,0)/a.length).toFixed(3):'none';
  document.getElementById('sum').textContent=`  tuning set mean ${m(tr)} (${tr.length}), held-out mean ${m(ho)} (${ho.length})`;
}
async function start(){document.getElementById('start').disabled=true;await fetch('/start',{method:'POST'});}
setInterval(tick,500);tick();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path == "/state":
            self._send(json.dumps({"running": state["running"], "started": state["started"], "last": state["last"],
                                   "log": state["log"], "count": next_index(), "sheet": sheet()}))
        else:
            page = PAGE.replace("%PLAN%", json.dumps(PLAN)).replace("%ORDER%", json.dumps(ORDER)).replace("%SECS%", str(SECONDS)).replace("%%", "%")
            self._send(page, "text/html; charset=utf-8")

    def do_POST(self):
        if self.path == "/start" and not state["running"] and next_index() < len(ORDER):
            tag = ORDER[next_index()]
            threading.Thread(target=run_episode, args=(tag,), daemon=True).start()
            self._send(json.dumps({"started": tag}))
        else:
            self._send(json.dumps({"started": None}))


if __name__ == "__main__":
    port = int(os.environ.get("COACH_PORT", "8787"))
    print(f"episode coach on http://localhost:{port}  (next: episode {next_index() + 1} of {len(ORDER)})")
    HTTPServer(("127.0.0.1", port), H).serve_forever()
