"""Find the bottle on the table, pick it up, hand it to Carl.

    .venv/bin/python pick.py            # dry run: finds the bottle, prints the grasp, moves nothing
    .venv/bin/python pick.py --live     # the real thing; a hand on the red button

Steps:
  1. greet pose, gripper open. Find the nearest standing bottle within reach (bottle.py).
  2. planner: gripper to a point ABOVE_MM above the grasp point, fingers pointing down.
  3. planner: straight down to the grasp point. Close. Report whether something is held.
  4. planner: back up ABOVE_MM. Then the handover from handme.py (watch for the hand, go, open, greet).
Nothing learns; the detector finds, the planner moves.
"""
import argparse
import asyncio
import math
import time

from viam.proto.common import Pose, PoseInFrame

from bottle import MAX_REACH, find_bottles
from handme import Rig, handover, say
from viamarm import connect

ABOVE_MM = 120.0


async def plan_to(rig, x, y, z, label):
    dest = PoseInFrame(reference_frame="world", pose=Pose(x=x, y=y, z=z, o_x=0, o_y=0, o_z=-1, theta=0))
    t0 = time.time()
    try:
        ok = await rig.motion.move(component_name="gripper", destination=dest)
    except Exception as e:
        await say(f"planner, {label}: {str(e)[:100]}"); return False
    await say(f"planner, {label}: ok={ok} in {time.time() - t0:.1f} s")
    return ok


async def pick(rig):
    await say("to greet, gripper open")
    if rig.live:
        await rig.greet.set_position(2); await asyncio.sleep(4)
        await rig.gripper.open()
    bottles, _ = await find_bottles(rig.machine, rig.cam, rig.intr)
    bottles = [b for b in bottles if b[0] >= 0.5 and math.hypot(b[2][0], b[2][1]) <= MAX_REACH]
    if not bottles:
        await say("no bottle within reach; put it on the table nearer the arm"); return False
    conf, box, c, top, g = bottles[0]
    await say(f"bottle (conf {conf:.2f}) at x={c[0]:.0f} y={c[1]:.0f}, {math.hypot(c[0], c[1]):.0f} mm from the base; "
              f"grasp at z={g[2]:.0f}")
    if not rig.live:
        await say(f"would go above it (z={g[2] + ABOVE_MM:.0f}), down to z={g[2]:.0f}, close, lift"); return True
    if not await plan_to(rig, g[0], g[1], g[2] + ABOVE_MM, "above the bottle"):
        return False
    if not await plan_to(rig, g[0], g[1], g[2], "down to the grasp"):
        return False
    held = await rig.gripper.grab()
    await say(f"gripper closed, holding something: {held}")
    if not await plan_to(rig, g[0], g[1], g[2] + ABOVE_MM, "lift"):
        return False
    return held


async def main(live):
    async with await connect() as machine:
        rig = await Rig(machine, live).setup()
        ok = await pick(rig)
        if ok or not live:
            await handover(rig)
        else:
            await say("nothing in the gripper; back to greet")
            if live:
                await rig.greet.set_position(2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    asyncio.run(main(ap.parse_args().live))
