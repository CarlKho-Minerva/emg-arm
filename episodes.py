"""Episodic memory for Served With a Smile: one document per thing the arm was asked to do.

Every command the face gives (fetch, feed Skully, let go, home) becomes an episode: what was heard, what
the robot did, what it saw, whether it worked, how long it took. Episodes are appended to
data/serve_episodes.jsonl at once (the file is the truth) and pushed to MongoDB when a cluster can be
reached; what could not be pushed is marked synced=false and retried on the next push.

    .venv/bin/python episodes.py stats        # what the dashboard shows
    .venv/bin/python episodes.py backfill     # Thursday's serve.log lines become episodes, source "backfill"
    .venv/bin/python episodes.py push         # push everything unsynced to MongoDB (needs a reachable URI)

The MongoDB URI is read from SERVE_MONGODB_URI, then SANDBOX_MONGODB_URI, in ~/.config/carl-life-os/.env.
At the venue on 30 Oct the network blocked SRV lookups and port 27017, so Atlas was unreachable from the
table; the file kept everything, and the screen says "MongoDB: unreachable, N queued" rather than hiding it.
"""
import json
import os
import sys
import threading
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(HERE, "data", "serve_episodes.jsonl")
SERVE_LOG = os.path.join(HERE, "data", "serve.log")
DB, COLL = "served_with_a_smile", "episodes"
_state = {"uri": None, "ok": None, "error": "", "checked": 0.0, "client": None}


def _uri():
    from dotenv import load_dotenv
    load_dotenv(os.path.expanduser("~/.config/carl-life-os/.env"))
    return os.environ.get("SERVE_MONGODB_URI") or os.environ.get("SANDBOX_MONGODB_URI")


def _append(doc):
    os.makedirs(os.path.dirname(FILE), exist_ok=True)
    with open(FILE, "a") as f:
        f.write(json.dumps(doc) + "\n")


def load():
    try:
        return [json.loads(l) for l in open(FILE) if l.strip()]
    except FileNotFoundError:
        return []


class Episode(dict):
    """A dict with start()/end(). Use: ep = Episode.start("feed_skully", "R"); ... ep.end(True, "fed")."""
    @classmethod
    def start(cls, kind, trigger, source="live", **extra):
        ep = cls(episode_id=f"{kind}_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:4]}", kind=kind, trigger=trigger,
                 t_start=time.time(), source=source, synced=False, **extra)
        return ep

    def end(self, ok, reason, **extra):
        self.update(ok=bool(ok), reason=reason, t_end=time.time(), duration_s=round(time.time() - self["t_start"], 2), **extra)
        _append(dict(self))
        threading.Thread(target=push, daemon=True).start()
        return self


def mongo(timeout_ms=4000):
    """A connected pymongo client or None. Re-tries at most every 20 s so a dead network costs nothing."""
    if _state["client"] is not None and _state["ok"]:
        return _state["client"]
    if time.time() - _state["checked"] < 20:
        return None
    _state["checked"] = time.time()
    uri = _uri()
    _state["uri"] = uri
    if not uri:
        _state.update(ok=False, error="no MongoDB URI in the env"); return None
    try:
        from pymongo import MongoClient
        c = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms, connectTimeoutMS=timeout_ms)
        c.admin.command("ping")
        _state.update(ok=True, error="", client=c)
        return c
    except Exception as e:
        _state.update(ok=False, error=str(e)[:120], client=None)
        return None


_lock = threading.Lock()


def push():
    """Push every unsynced episode. Rewrites the file with synced=true for the ones that went through."""
    with _lock:
        c = mongo()
        if c is None:
            return 0
        rows = load()
        todo = [r for r in rows if not r.get("synced")]
        if not todo:
            return 0
        try:
            coll = c[DB][COLL]
            for r in todo:
                d = dict(r); d.pop("synced", None); d["_id"] = d["episode_id"]
                coll.replace_one({"_id": d["_id"]}, d, upsert=True)
                r["synced"] = True
            tmp = FILE + ".tmp"
            with open(tmp, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            os.replace(tmp, FILE)
            return len(todo)
        except Exception as e:
            _state.update(ok=False, error=str(e)[:120], client=None)
            return 0


def mongo_status():
    rows = load()
    queued = sum(1 for r in rows if not r.get("synced"))
    return {"configured": bool(_uri()), "ok": _state["ok"], "error": _state["error"], "queued": queued,
            "synced": len(rows) - queued, "db": DB, "collection": COLL}


def stats():
    rows = load()
    by = {}
    for r in rows:
        k = r.get("kind", "?")
        d = by.setdefault(k, {"attempts": 0, "ok": 0, "refused": 0, "duration_s": []})
        d["attempts"] += 1
        d["ok"] += 1 if r.get("ok") else 0
        d["refused"] += 1 if (not r.get("ok") and str(r.get("reason", "")).startswith(("skully moving", "skully out", "skully not", "nothing in"))) else 0
        if r.get("duration_s") is not None:
            d["duration_s"].append(r["duration_s"])
    for d in by.values():
        ds = d.pop("duration_s")
        d["avg_s"] = round(sum(ds) / len(ds), 1) if ds else None
    live = [r for r in rows if r.get("source") == "live"]
    return {"total": len(rows), "live": len(live), "backfill": len(rows) - len(live), "by_kind": by,
            "last": rows[-5:][::-1], "mongo": mongo_status()}


def backfill():
    """Thursday's serve.log (time, group, action) as episodes. Outcomes come from the action text only;
    a line that does not say is recorded as ok=None, never guessed."""
    have = {r["episode_id"] for r in load()}
    n = 0
    for ln in open(SERVE_LOG):
        parts = ln.strip().split(" ", 2)
        if len(parts) < 3:
            continue
        t, g, action = float(parts[0]), parts[1], parts[2]
        eid = f"backfill_{int(t * 1000)}"
        if eid in have:
            continue
        kind = ("fetch" if action.startswith("fetch") else "feed_face" if action == "feed" else "handover" if "hand it over" in action
                else "let_go" if action == "let go" else "home" if action == "home" else "nudge" if "nudge" in action or "axis" in action else action)
        ok = True if "picked and proved" in action or action in ("let go", "home") else False if "missed" in action else None
        _append({"episode_id": eid, "kind": kind, "trigger": g, "t_start": t, "t_end": None, "duration_s": None, "ok": ok,
                 "reason": action, "source": "backfill", "synced": False, "note": "from data/serve.log, 29 Oct; outcome only where the line says it"})
        n += 1
    return n


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stats"
    if cmd == "backfill":
        print("backfilled", backfill(), "episodes")
    elif cmd == "push":
        print("pushed", push(), "; status", mongo_status())
    else:
        print(json.dumps(stats(), indent=1))
