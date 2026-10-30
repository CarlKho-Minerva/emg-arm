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


LOCAL_URI = "mongodb://localhost:27017"


def _uris():
    """Candidates in order: SERVE_MONGODB_URI (Atlas, set at the table), SANDBOX_MONGODB_URI, then a local mongod.
    On 30 Oct both Atlas hostnames in the env returned NXDOMAIN from two networks: those clusters are gone."""
    from dotenv import load_dotenv
    load_dotenv(os.path.expanduser("~/.config/carl-life-os/.env"))
    out = []
    for k in ("SERVE_MONGODB_URI", "SANDBOX_MONGODB_URI"):
        if os.environ.get(k):
            out.append((k, os.environ[k]))
    out.append(("local", LOCAL_URI))
    return out


def _uri():
    return _uris()[0][1]


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
    errors = []
    from pymongo import MongoClient
    for name, uri in _uris():
        try:
            c = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms if name != "local" else 1500, connectTimeoutMS=timeout_ms)
            c.admin.command("ping")
            _state.update(ok=True, error="", client=c, uri=uri, which=name)
            return c
        except Exception as e:
            errors.append(f"{name}: {str(e)[:60]}")
    _state.update(ok=False, error="; ".join(errors)[:200], client=None, which=None)
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
    return {"configured": True, "ok": _state["ok"], "error": _state["error"], "queued": queued, "which": _state.get("which"),
            "synced": len(rows) - queued, "db": DB, "collection": COLL}


def stats():
    mongo()            # so the screen reports the connection, not just the file (throttled to one try per 20 s)
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
            "last": rows[-7:][::-1], "mongo": mongo_status()}


def backfill():
    """Thursday's records as episodes, source "backfill", original timestamps kept, nothing invented.
    Four sources: data/serve.log (face commands and what the arm did), data/practice.json (the cube-pick
    practice rounds, whose hold check was later found untrusted and is recorded as such), data/episodes.json
    (the 30-second follower runs scored by sim.py), and data/handover/<run>/ (the handover recordings: frame
    count and duration from the frame timestamps; no outcome is on disk, so ok=None)."""
    import glob
    have = {r["episode_id"] for r in load()}
    new = []
    def add(doc):
        if doc["episode_id"] not in have:
            new.append(dict(doc, source="backfill", synced=False)); have.add(doc["episode_id"])
    # 1. serve.log
    try:
        for ln in open(SERVE_LOG):
            parts = ln.strip().split(" ", 2)
            if len(parts) < 3:
                continue
            t, g, action = float(parts[0]), parts[1], parts[2]
            kind = ("fetch" if action.startswith("fetch") else "feed_face" if action == "feed" else "handover" if "hand it over" in action
                    else "let_go" if action == "let go" else "home" if action == "home" else "nudge" if "nudge" in action or "axis" in action else action)
            ok = True if "picked and proved" in action or action in ("let go", "home") else False if "missed" in action else None
            add({"episode_id": f"backfill_{int(t * 1000)}", "kind": kind, "trigger": g, "t_start": t, "t_end": None, "duration_s": None,
                 "ok": ok, "reason": action, "note": "from data/serve.log, 29 Oct; outcome only where the line says it"})
    except FileNotFoundError:
        pass
    # 2. practice.json: cube-pick practice rounds, 29 Oct 18:00 to 18:51
    pj = os.path.join(HERE, "data", "practice.json")
    if os.path.exists(pj):
        import datetime
        for i, r in enumerate(json.load(open(pj))):
            hh, mm, ss = (int(v) for v in r["at"].split(":"))
            t = datetime.datetime(2026, 10, 29, hh, mm, ss).timestamp()
            trusted = "untrusted" not in r.get("held_by", "") and "held_by" in r
            add({"episode_id": f"backfill_practice_{i:02d}", "kind": "pick_practice", "trigger": "keyboard", "t_start": t,
                 "t_end": t + r["seconds"], "duration_s": r["seconds"], "cube": r["colour"], "round": r["round"],
                 "ok": r["held"] if trusted else None, "reason": r.get("held_by", "held flag only; no finger check recorded"),
                 "note": "practice.py, 29 Oct; 'held' from the early finger read, later found to report 12 false holds, so ok=None where untrusted"})
    # 3. episodes.json: follower runs scored by sim.py
    ej = os.path.join(HERE, "data", "episodes.json")
    if os.path.exists(ej):
        for r in json.load(open(ej)):
            run = r["run"]
            try:
                t = __import__("time").mktime(__import__("time").strptime(run[:15], "%Y%m%d-%H%M%S"))
            except ValueError:
                t = None
            add({"episode_id": f"backfill_follow_{run}", "kind": "follow", "trigger": "keyboard", "t_start": t, "t_end": (t + 30) if t else None,
                 "duration_s": 30.0, "tag": r["tag"], "held_out": r["held_out"], "frames": r["frames"], "seen": r["seen"],
                 "lost_share": r["lost"], "rms_error": r["rms"], "reward": r["reward"], "ok": r["exit"] == 0 and r["seen"] > 0,
                 "reason": f"follower run, tag {r['tag']}, reward {r['reward']}", "note": "follow.py --live 30 s, scored by sim.py, 29 Oct"})
    # 4. handover recordings: frames on disk
    for d in sorted(glob.glob(os.path.join(HERE, "data", "handover", "*"))):
        frames = sorted(f for f in os.listdir(d) if f.endswith(".jpg"))
        if len(frames) < 2:
            continue
        t0, t1 = float(frames[0][:-4]), float(frames[-1][:-4])
        name = os.path.basename(d)
        add({"episode_id": f"backfill_handover_{name}", "kind": "handover_face" if name.endswith("-face") else "handover", "trigger": "keyboard",
             "t_start": t0, "t_end": t1, "duration_s": round(t1 - t0, 1), "frames": len(frames), "ok": None,
             "reason": f"{len(frames)} frames over {t1 - t0:.0f} s; outcome not on disk", "note": "handme.handover recording, 29 Oct"})
    for doc in new:
        _append(doc)
    return len(new)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stats"
    if cmd == "backfill":
        print("backfilled", backfill(), "episodes")
    elif cmd == "push":
        print("pushed", push(), "; status", mongo_status())
    else:
        print(json.dumps(stats(), indent=1))
