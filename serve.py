"""Served With a Smile: the arm driven by smile groups from smile.py (data/face.log).

    .venv/bin/python serve.py --live          # smile.py must be running in Terminal (run_smile.command)

Fetch mode (start, at greet):
  L     grab the food: home, find the green cube, pick, prove the grip, back to greet
  R     feed me: follow the left hand and hand it over (palm under the gripper), tonight's verified path
  B     let go now (gripper opens wherever it is)
  BB    Nudge mode
  BBB   home, gripper open, Fetch mode
Nudge mode (30 mm steps):
  L / R one step along the current axis (left/right, toward/away, up/down)
  BB    next axis
  B     grab if empty, let go if holding
  BBB   greet, Fetch mode
Every group is logged with what it did, to data/serve.log.
"""
import argparse
import asyncio
import os
import time

from viam.proto.common import Pose
from viam.components.switch import Switch

from handme import Rig, handover, say
from pick import pick, settled_fingers, HELD_MIN, HELD_MAX
from viamarm import connect

HERE = os.path.dirname(os.path.abspath(__file__))
FACE_LOG = os.path.join(HERE, "data", "face.log")
SERVE_LOG = os.path.join(HERE, "data", "serve.log")
STEP_MM = 30.0
AXES = (("left/right", (0, 1, 0)), ("toward/away", (1, 0, 0)), ("up/down", (0, 0, 1)))


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


async def nudge(rig, axis, sign):
    name, (ax, ay, az) = AXES[axis]
    p = (await rig.motion.get_pose("gripper", "world")).pose
    tx, ty, tz = p.x + sign * STEP_MM * ax, p.y + sign * STEP_MM * ay, max(p.z + sign * STEP_MM * az, 80.0)
    await say(f"nudge {name} {'+' if sign > 0 else '-'}{STEP_MM:.0f} mm")
    if rig.live:
        try:
            await asyncio.wait_for(rig.arm.move_to_position(Pose(x=tx, y=ty, z=tz, o_x=p.o_x, o_y=p.o_y, o_z=p.o_z, theta=p.theta)), 15)
        except Exception as e:
            await say(f"nudge refused: {str(e)[:80]}")


async def main(live, cube):
    async with await connect() as machine:
        rig = await Rig(machine, live).setup()
        rig.last_grasp = None
        home = Switch.from_robot(machine, "home")
        mode, axis = "fetch", 0
        since = time.time()
        await say(f"Served With a Smile. Mode FETCH. Smile left to fetch the {cube} cube.")
        while True:
            groups = new_groups(since)
            if not groups:
                await asyncio.sleep(0.1)
                continue
            since = groups[-1][0]
            for _, g in groups:
                await say(f"heard {g} in {mode.upper()}")
                if g == "BBB":
                    log(g, "home"); mode = "fetch"
                    if live:
                        await rig.gripper.open(); await home.set_position(2)
                    await say("home; mode FETCH")
                elif mode == "fetch":
                    if g == "L":
                        log(g, "fetch")
                        ok = await pick(rig, cube)
                        await say("fetched; at greet" if ok else "could not fetch")
                    elif g == "R":
                        log(g, "feed")
                        await handover(rig)
                    elif g == "B":
                        log(g, "let go")
                        if live:
                            await rig.gripper.open()
                    elif g == "BB":
                        log(g, "nudge mode"); mode = "nudge"; axis = 0
                        await say(f"mode NUDGE, axis {AXES[axis][0]}")
                elif mode == "nudge":
                    if g in ("L", "R"):
                        log(g, f"nudge {AXES[axis][0]}")
                        await nudge(rig, axis, -1 if g == "L" else 1)
                    elif g == "BB":
                        axis = (axis + 1) % len(AXES); log(g, f"axis {AXES[axis][0]}")
                        await say(f"axis {AXES[axis][0]}")
                    elif g == "B":
                        if live:
                            posn = await settled_fingers(rig)
                            if HELD_MIN < posn < HELD_MAX:
                                await rig.gripper.open(); log(g, "let go")
                            else:
                                await rig.gripper.grab(); log(g, "grab")
                        await say("gripper toggled")
            since = time.time()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--cube", default="green")
    a = ap.parse_args()
    asyncio.run(main(a.live, a.cube))
