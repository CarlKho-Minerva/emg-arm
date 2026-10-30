"""Served With a Smile: the arm driven by smile groups from smile.py (data/face.log).

    .venv/bin/python serve.py --live          # smile.py must be running in Terminal (run_smile.command)
    .venv/bin/python serve.py                 # dry run: hears the groups, moves nothing, logs what it would do

Two layers (Carl, Fri 30 Oct 14:33). The face window and the screen show the one you are in (data/mode.json).
SERVE (start):
  L     left cheek:  fetch: home, find the green cube, pick it, prove the grip, back to greet
  R     right cheek: feed Skully: home, find the blue tape on his nose, hover, re-centre, cube into the mouth, let go
  B     one smile:   let go
  BB    two smiles:  to STEER
STEER:
  L / R turn the waist STEER_DEG toward YOUR left / right (the arm faces you, so it is mirrored for you)
  B     one smile:   grab if empty, let go if holding
  BB    two smiles:  back to SERVE
BBB   three smiles, in either layer: home, gripper open, SERVE (the stop)
M     open mouth: feed ME; off unless --face-feed (never verified live; it moves toward a person's face)
Every command is an episode (episodes.py): data/serve_episodes.jsonl at once, MongoDB when reachable.
Every line the arm does is also in data/serve.log, which the screen (dash.py) shows as it is written.
While idle the wrist camera's picture is written to data/live.jpg four times a second for the screen.
"""
import argparse
import asyncio
import os
import time

from camgrab import grab
from episodes import Episode
from handme import Rig, feed, feed_held, say, tape_point
from viam.components.switch import Switch
from pick import HELD_MAX, HELD_MIN, pick, settled_fingers
from skull import feed_skully
from viamarm import connect

HERE = os.path.dirname(os.path.abspath(__file__))
FACE_LOG = os.path.join(HERE, "data", "face.log")
SERVE_LOG = os.path.join(HERE, "data", "serve.log")
LIVE_JPG = os.path.join(HERE, "data", "live.jpg")
MODE_FILE = os.path.join(HERE, "data", "mode.json")     # the layer the arm is in, for the face window and the screen
STEER_DEG = 8.0        # (table path) one cheek event = this much waist turn; the live STEER layer pans while the cheek is held instead
STEER_RATE = 12.0      # deg/s of waist turn while a cheek is held in STEER (streamed small steps, 10 a second)
FACE_STATUS = os.path.join(HERE, "data", "face_status.json")
# The arm faces Carl, so Carl's left is the arm's right. In follow.py a hand on the picture's right (Carl's left,
# the wrist camera is not mirrored) makes joint 0 DEcrease. So Carl's left cheek = joint 0 minus STEER_DEG.
STEER_SIGN = {"L": -1.0, "R": +1.0}

MODES = {
    "serve": [("L", "left cheek", "fetch the cube"), ("R", "right cheek", "extend to Skully and let go (after a fetch the arm follows him on its own)"), ("B", "one smile", "let go"),
              ("BB", "two smiles", "to STEER"), ("BBB", "three smiles", "home (stop)")],
    "steer": [("L", "hold left cheek", "pan to your left while held"), ("R", "hold right cheek", "pan to your right while held"),
              ("B", "one smile", "grab / let go"), ("BB", "two smiles", "to SERVE"), ("BBB", "three smiles", "home (stop)")],
}


def set_mode(mode):
    import json
    os.makedirs(HERE + "/data", exist_ok=True)
    tmp = MODE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"mode": mode, "since": time.time(), "controls": MODES[mode]}, f)
    os.replace(tmp, MODE_FILE)


def new_groups(after):
    """Smile groups written after time `after`: [(t, group)]."""
    try:
        lines = open(FACE_LOG).read().strip().split("\n")
    except FileNotFoundError:
        return []
    out = []
    for line in lines:
        if not line.strip():
            continue
        t, g = line.split()
        if float(t) > after:
            out.append((float(t), g))
    return out


def log(group, action):
    os.makedirs(HERE + "/data", exist_ok=True)
    with open(SERVE_LOG, "a") as f:
        f.write(f"{time.time():.3f} {group} {action}\n")


async def holding(rig):
    """Is there a cube between the fingers? Settled finger position inside the cube band."""
    if not rig.live:
        return True
    posn = await settled_fingers(rig)
    return HELD_MIN < posn < HELD_MAX


async def publish_view(rig):
    try:
        jpeg, _ = await asyncio.wait_for(grab(rig.cam, want_depth=False), 2)
        tmp = LIVE_JPG + ".serve.tmp"
        with open(tmp, "wb") as f:
            f.write(jpeg)
        os.replace(tmp, LIVE_JPG)
    except Exception:
        pass


def holding_now():
    """Which cheek the face reader says is up right now: "L", "R", "B" or None (from face_status.json, 4 writes a second)."""
    import json
    try:
        d = json.load(open(FACE_STATUS))
    except (OSError, ValueError):
        return None
    if time.time() - d.get("t", 0) > 1.0:
        return None
    return d.get("holding")


async def steer(rig, g):
    """One waist step, streamed like the follower's moves, so steps chain without stopping."""
    from viam.proto.component.arm import JointPositions
    j = list((await rig.arm.get_joint_positions()).values)
    j[0] += STEER_SIGN[g] * STEER_DEG
    await rig.arm.move_to_joint_positions(JointPositions(values=j))
    return j[0]


SKULLY_MODE = "extend"   # "extend": after the fetch the arm follows the tape on its own; a smile sends it to the saved pose "extend" and opens
                         # (Carl, 30 Oct 15:10). "approach": depth-ranged approach to the mouth (feed_held). "table": skull.py's path.
EXTEND_POSE = "extend"   # saved with: .venv/bin/python savepose.py extend   (jog the arm to the hand-off pose first)
EXTEND_S = 5.0           # seconds to give the go-to before opening


async def follow_step(rig, state):
    """One step of keeping the tape on Skully's nose centred in the wrist camera: the follower's pan/tilt
    (waist and elbow), streamed. state holds the smoothed point and the step filter between calls."""
    from follow import DEADBAND, GAIN, SMOOTH, STEP_CAP
    from viam.proto.component.arm import JointPositions
    try:
        jpeg, _ = await asyncio.wait_for(grab(rig.cam, want_depth=False), 2)
    except Exception:
        return
    tmp = LIVE_JPG + ".serve.tmp"
    with open(tmp, "wb") as f:
        f.write(jpeg)
    os.replace(tmp, LIVE_JPG)
    raw = tape_point(jpeg)
    if raw is None:
        state["seen"] = False
        return
    state["seen"] = True
    sm = state.get("sm")
    sm = raw if sm is None else (sm[0] + SMOOTH * (raw[0] - sm[0]), sm[1] + SMOOTH * (raw[1] - sm[1]))
    state["sm"] = sm
    ex, ey = sm[0] - 0.5, sm[1] - 0.5
    if abs(ex) < DEADBAND and abs(ey) < DEADBAND:
        return
    dx = max(-STEP_CAP, min(STEP_CAP, -GAIN * ex)); dy = max(-STEP_CAP, min(STEP_CAP, GAIN * ey))
    state["adx"] = state.get("adx", 0.0) + 0.7 * (dx - state.get("adx", 0.0))
    state["ady"] = state.get("ady", 0.0) + 0.7 * (dy - state.get("ady", 0.0))
    if rig.live:
        j = list((await rig.arm.get_joint_positions()).values)
        j[0] += state["adx"]; j[2] += state["ady"]
        asyncio.ensure_future(rig.arm.move_to_joint_positions(JointPositions(values=j)))


async def extend_and_release(rig):
    """Go to the saved "extend" pose, open the gripper, back to greet. Returns (ok, reason)."""
    if not rig.live:
        return True, "dry: would extend and open"
    try:
        await rig.machine.refresh()
        names = [r.name for r in rig.machine.resource_names]
    except Exception:
        names = []
    if EXTEND_POSE not in names:
        await say(f"no saved pose '{EXTEND_POSE}': jog the arm to the hand-off pose and run savepose.py {EXTEND_POSE}. Opening here instead.")
        await rig.gripper.open(); await asyncio.sleep(1)
        return False, f"no '{EXTEND_POSE}' pose saved; opened at greet"
    await say(f"extending to '{EXTEND_POSE}'")
    await Switch.from_robot(rig.machine, EXTEND_POSE).set_position(2)
    await asyncio.sleep(EXTEND_S)
    await rig.gripper.open(); await say("released")
    await asyncio.sleep(1.5)
    await rig.greet.set_position(2)
    return True, "extended and released"


async def main(live, cube, face_feed):
    async with await connect() as machine:
        rig = await Rig(machine, live).setup()
        rig.last_grasp = None
        mode = "serve"; set_mode(mode)
        steering = False
        following = False; fstate = {}
        since = time.time()
        last_view = 0.0
        await say(f"Served With a Smile. Layer SERVE. Left cheek: fetch the {cube} cube. Right cheek: feed Skully. One smile: let go. Two smiles: steer.")
        while True:
            groups = new_groups(since)
            if mode == "steer" and live:
                held = holding_now()
                if held in ("L", "R"):
                    # pan while the cheek is held: a small streamed step ten times a second
                    from viam.proto.component.arm import JointPositions
                    j = list((await rig.arm.get_joint_positions()).values)
                    j[0] += STEER_SIGN[held] * STEER_RATE * 0.1
                    asyncio.ensure_future(rig.arm.move_to_joint_positions(JointPositions(values=j)))
                    if not steering:
                        steering = True; log(held, "steer: panning while held")
                    await asyncio.sleep(0.1)
                    since = time.time()           # the cheek's own group event is the pan, not a jump
                    continue
                elif steering:
                    steering = False
                    log("-", "steer: stopped")
            if not groups:
                if following and mode == "serve":
                    await follow_step(rig, fstate)          # keeps the tape centred; ~8 to 10 steps a second
                    await asyncio.sleep(0.03)
                    continue
                if time.time() - last_view > 0.25:
                    await publish_view(rig); last_view = time.time()
                await asyncio.sleep(0.05)
                continue
            since = groups[-1][0]
            for _, g in groups:
                await say(f"heard {g} in {mode.upper()}")
                if g == "BBB":
                    ep = Episode.start("home", g, mode=mode)
                    log(g, "home, gripper open")
                    if live:
                        await rig.gripper.open(); await rig.home.set_position(2)
                    mode = "serve"; set_mode(mode); following = False
                    ep.end(True, "home"); await say("home; layer SERVE")
                elif g == "BB":
                    mode = "steer" if mode == "serve" else "serve"; set_mode(mode); following = False
                    log(g, f"layer {mode.upper()}")
                    Episode.start("switch", g, mode=mode).end(True, f"layer {mode}")
                    await say(f"layer {mode.upper()}")
                elif mode == "serve":
                    if g == "L":
                        following = False
                        ep = Episode.start("fetch", g, cube=cube, mode=mode)
                        ok = await pick(rig, cube)
                        log(g, "fetch: picked and proved" if ok else "fetch: missed")
                        ep.end(ok, "picked and proved" if ok else "missed", grasp=rig.last_grasp)
                        if ok and SKULLY_MODE == "extend":
                            following = True; fstate = {}
                            log("-", "following Skully's tape; a smile extends and lets go")
                            await say("fetched; following Skully. Smile when he is ready.")
                        else:
                            await say("fetched; at greet" if ok else "could not fetch")
                    elif g in ("R", "B") and SKULLY_MODE == "extend":
                        ep = Episode.start("feed_skully", g, mode=mode, how="extend")
                        if not await holding(rig):
                            if g == "B":
                                log(g, "let go (nothing held)"); ep.end(True, "opened, nothing held")
                                if live:
                                    await rig.gripper.open()
                                continue
                            log(g, "feed Skully refused: nothing in the gripper (left cheek first)")
                            ep.end(False, "nothing in the gripper")
                            await say("nothing in the gripper; left cheek first")
                            continue
                        following = False
                        ok, why = await extend_and_release(rig)
                        log(g, f"feed Skully: {why}")
                        ep.end(ok, why)
                    elif g == "R":
                        ep = Episode.start("feed_skully", g, mode=mode)
                        if not await holding(rig):
                            log(g, "feed Skully refused: nothing in the gripper (left cheek first)")
                            ep.end(False, "nothing in the gripper")
                            await say("nothing in the gripper to feed him with; left cheek first")
                            continue
                        if SKULLY_MODE == "approach":
                            fed = await feed_held(rig)
                            why = "cube brought to the mouth; smile let go" if fed else "Skully not seen still; cube kept"
                            log(g, f"feed Skully (held): {why}")
                            ep.end(fed, why, mode_skully="held")
                        else:
                            ok, why = await feed_skully(rig, say, episode=ep)
                            log(g, f"feed Skully: {why}")
                            ep.end(ok, why)
                    elif g == "M":
                        ep = Episode.start("feed_face", g, mode=mode)
                        if not face_feed:
                            log(g, "feed me: off (start with --face-feed to allow it)")
                            ep.end(False, "disabled"); continue
                        log(g, "feed me")
                        await feed(rig, put_back=rig.last_grasp)
                        ep.end(None, "ran; outcome not measured")
                    elif g == "B":
                        ep = Episode.start("let_go", g, mode=mode)
                        log(g, "let go")
                        if live:
                            await rig.gripper.open()
                        ep.end(True, "opened")
                    else:
                        log(g, "ignored"); await say(f"{g}: nothing bound to that in SERVE")
                else:   # steer
                    if g in ("L", "R"):
                        if not live:
                            log(g, f"steer {'left' if g == 'L' else 'right'} (dry)")
                        # live: the pan already happened while the cheek was held; the event itself is not a jump
                    elif g == "B":
                        ep = Episode.start("grip", g, mode=mode)
                        if live:
                            if await holding(rig):
                                await rig.gripper.open(); log(g, "let go"); ep.end(True, "opened")
                            else:
                                await rig.gripper.grab(); log(g, "grab"); ep.end(True, "grabbed")
                        else:
                            log(g, "grab/let go (dry)"); ep.end(True, "dry")
                        await say("gripper toggled")
                    else:
                        log(g, "ignored"); await say(f"{g}: nothing bound to that in STEER")
            since = time.time()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--cube", default="green")
    ap.add_argument("--face-feed", action="store_true", help="let the open mouth (M) run the feed-a-person path")
    ap.add_argument("--skully", choices=["extend", "approach", "table"], default="extend", help="extend (default): follow, then a smile goes to the saved 'extend' pose and opens; approach: depth-ranged; table: skull.py")
    a = ap.parse_args()
    SKULLY_MODE = a.skully
    asyncio.run(main(a.live, a.cube, a.face_feed))
