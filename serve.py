"""Served With a Smile: the arm driven by smile groups from smile.py (data/face.log).

    .venv/bin/python serve.py --live          # smile.py must be running in Terminal (run_smile.command)
    .venv/bin/python serve.py                 # dry run: hears the groups, moves nothing, logs what it would do

The controls (final, Fri 30 Oct 14:00; the crane/nudge mode is gone):
  L     half-smile LEFT:  fetch the food: home, find the green cube, pick it, prove the grip, back to greet
  R     half-smile RIGHT: feed Alex: home, find the blue tape on his nose twice a second apart, and only if he
        held still and is in reach, bring the cube level into his open mouth, let go, back out, greet.
        If he moved, the robot says so, refuses, and comes back to greet still holding the cube.
  B     full smile: let go (the gripper opens where it is)
  BB    two smiles: home, gripper open (the universal recovery)
  M     open mouth: feed ME (follow my face, hold the cube at my mouth, take it back). Off unless --face-feed:
        it was never verified live and it moves toward a person's face.
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
from handme import Rig, feed, say
from pick import HELD_MAX, HELD_MIN, pick, settled_fingers
from skull import feed_alex
from viamarm import connect

HERE = os.path.dirname(os.path.abspath(__file__))
FACE_LOG = os.path.join(HERE, "data", "face.log")
SERVE_LOG = os.path.join(HERE, "data", "serve.log")
LIVE_JPG = os.path.join(HERE, "data", "live.jpg")


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
        tmp = LIVE_JPG + ".tmp"
        with open(tmp, "wb") as f:
            f.write(jpeg)
        os.replace(tmp, LIVE_JPG)
    except Exception:
        pass


async def main(live, cube, face_feed):
    async with await connect() as machine:
        rig = await Rig(machine, live).setup()
        rig.last_grasp = None
        since = time.time()
        last_view = 0.0
        await say(f"Served With a Smile. LEFT cheek: fetch the {cube} cube. RIGHT cheek: feed Alex. Smile: let go. Two smiles: home.")
        while True:
            groups = new_groups(since)
            if not groups:
                if time.time() - last_view > 0.25:
                    await publish_view(rig); last_view = time.time()
                await asyncio.sleep(0.05)
                continue
            since = groups[-1][0]
            for _, g in groups:
                await say(f"heard {g}")
                if g == "L":
                    ep = Episode.start("fetch", g, cube=cube)
                    ok = await pick(rig, cube)
                    log(g, "fetch: picked and proved" if ok else "fetch: missed")
                    ep.end(ok, "picked and proved" if ok else "missed", grasp=rig.last_grasp)
                    await say("fetched; at greet" if ok else "could not fetch")
                elif g == "R":
                    ep = Episode.start("feed_alex", g)
                    if not await holding(rig):
                        log(g, "feed Alex refused: nothing in the gripper (smile left first)")
                        ep.end(False, "nothing in the gripper")
                        await say("nothing in the gripper to feed him with; smile left first")
                        continue
                    ok, why = await feed_alex(rig, say, episode=ep)
                    log(g, f"feed Alex: {why}")
                    ep.end(ok, why)
                elif g == "M":
                    ep = Episode.start("feed_face", g)
                    if not face_feed:
                        log(g, "feed me: off (start with --face-feed to allow it)")
                        ep.end(False, "disabled")
                        await say("feeding a person is off in this run")
                        continue
                    log(g, "feed me")
                    await feed(rig, put_back=rig.last_grasp)
                    ep.end(None, "ran; outcome not measured")
                elif g == "B":
                    ep = Episode.start("let_go", g)
                    log(g, "let go")
                    if live:
                        await rig.gripper.open()
                    ep.end(True, "opened")
                elif g == "BB" or g == "BBB":
                    ep = Episode.start("home", g)
                    log(g, "home, gripper open")
                    if live:
                        await rig.gripper.open(); await rig.home.set_position(2)
                    ep.end(True, "home")
                    await say("home")
                else:
                    log(g, "ignored")
                    await say(f"{g}: nothing bound to that")
            since = time.time()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--cube", default="green")
    ap.add_argument("--face-feed", action="store_true", help="let the open mouth (M) run the feed-a-person path")
    a = ap.parse_args()
    asyncio.run(main(a.live, a.cube, a.face_feed))
