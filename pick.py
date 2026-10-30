"""Find the bottle on the table, pick it up, hand it to Carl.

    .venv/bin/python pick.py            # dry run: finds the bottle, prints the grasp, moves nothing
    .venv/bin/python pick.py --live     # the real thing; a hand on the red button

Steps:
  1. home pose (looking down at the table), gripper open. Find the nearest standing bottle within reach.
  2. planner: gripper level at label height, PRE_MM short of the bottle, pointing at it (a side grasp,
     like a hand, not a claw from above).
  3. planner: slide in to the label. Close gently. Report whether something is held.
  4. planner: lift; go to greet (facing Carl). Then the handover from handme.py: follow the left hand,
     re-planning as it moves, and let go only once the hand has stayed at the gripper for a second.
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
LABEL_Z = 100.0        # mm above the table: the blue label, the middle of the bottle's body
PRE_MM = 160.0         # stop this far short of the bottle's axis before sliding in
IN_MM = 35.0           # gripper frame this far short of the axis when closing (fingers reach past it)
LIFT_MM = 100.0
MAX_TRIES = 4
HELD_POS = 60.0        # finger position above this after closing = something is between the fingers
SOFT_SPEED = 800.0     # gripper closing speed while grasping (driver default 2000); gentler on the bottle


async def plan_to(rig, x, y, z, label, o=(0.0, 0.0, -1.0), theta=0.0):
    dest = PoseInFrame(reference_frame="world", pose=Pose(x=x, y=y, z=z, o_x=o[0], o_y=o[1], o_z=o[2], theta=theta))
    t0 = time.time()
    try:
        ok = await rig.motion.move(component_name="gripper", destination=dest)
    except Exception as e:
        await say(f"planner, {label}: {str(e)[:100]}"); return False
    await say(f"planner, {label}: ok={ok} in {time.time() - t0:.1f} s")
    try:
        p = (await rig.motion.get_pose("gripper", "world")).pose
        await say(f"   gripper now at x={p.x:.0f} y={p.y:.0f} z={p.z:.0f} (asked x={x:.0f} y={y:.0f} z={z:.0f}; "
                  f"off by {p.x - x:+.0f},{p.y - y:+.0f},{p.z - z:+.0f} mm)")
    except Exception as e:
        await say(f"   could not read the gripper pose: {str(e)[:60]}")
    return ok


async def pick_once(rig, attempt):
    """Side grasp: come in level at label height from the base's side, slide in, close gently, lift."""
    if attempt == 1:
        await say("to greet first (if a bottle is still held from last time, it drops near Carl, not on the table)")
        if rig.live:
            await rig.greet.set_position(2); await asyncio.sleep(4)
            await rig.gripper.open(); await asyncio.sleep(1)
    await say(f"attempt {attempt}: to home (the view of the table)")
    if rig.live:
        await rig.home.set_position(2); await asyncio.sleep(5)
        await rig.gripper.open()
    bottles, _ = await find_bottles(rig.machine, rig.cam, rig.intr)
    bottles = [b for b in bottles if b[0] >= 0.5 and math.hypot(b[2][0], b[2][1]) <= MAX_REACH]
    if not bottles:
        await say("bottle not seen, or further than the arm can reach; put it nearer the arm"); return False
    conf, box, c, top, g = bottles[0]
    cx, cy = c[0], c[1]
    flat = math.hypot(cx, cy)
    dx, dy = cx / flat, cy / flat                       # from the base toward the bottle, level
    pre = (cx - dx * PRE_MM, cy - dy * PRE_MM, LABEL_Z)
    grasp = (cx - dx * IN_MM, cy - dy * IN_MM, LABEL_Z)
    await say(f"bottle (conf {conf:.2f}) at x={cx:.0f} y={cy:.0f}, {flat:.0f} mm from the base; "
              f"side grasp at the label, z={LABEL_Z:.0f}")
    if not rig.live:
        await say(f"would go to x={pre[0]:.0f} y={pre[1]:.0f} level, facing it; slide to x={grasp[0]:.0f} y={grasp[1]:.0f}; close; lift {LIFT_MM:.0f}")
        return True
    face = (dx, dy, 0.0)
    ok = False
    for theta in (0.0, 90.0):                             # which way the fingers close: try both
        ok = await plan_to(rig, *pre, f"beside the bottle, level (theta {theta:.0f})", o=face, theta=theta)
        if ok:
            break
    if not ok:
        return False
    if not await plan_to(rig, *grasp, "slide in to the label", o=face, theta=theta):
        return False
    try:
        await rig.gripper.do_command({"set_gripper_speed": SOFT_SPEED})
    except Exception as e:
        await say(f"could not soften the gripper: {str(e)[:60]}")
    held = await rig.gripper.grab()
    # the driver's "held" flag misses a soft, empty bottle (reported False on 29 Oct 17:31 with the fingers on it);
    # the finger position is the honest answer: fully closed means nothing, stopped part-way means something
    try:
        posn = (await rig.gripper.do_command({"get": True})).get("pos", -1)
    except Exception:
        posn = -1
    await say(f"gripper closed: driver says held={held}, finger position {posn:.0f} (0 = fully closed, 850 = open)")
    if posn > HELD_POS:
        held = True
    if not held:
        await say("nothing between the fingers; opening and trying again")
        await rig.gripper.open()
        return False
    try:
        await rig.gripper.do_command({"set_gripper_speed": 2000.0})
    except Exception:
        pass
    if not await plan_to(rig, grasp[0], grasp[1], grasp[2] + LIFT_MM, "lift", o=face, theta=theta):
        return False
    await say("bottle confirmed in the gripper; to greet, to find your hand")
    await rig.greet.set_position(2); await asyncio.sleep(5)
    return True


async def pick(rig):
    """Try until the fingers confirm a bottle, up to MAX_TRIES."""
    for attempt in range(1, MAX_TRIES + 1):
        ok = await pick_once(rig, attempt)
        if ok:
            return True
        if not rig.live:
            return False
        await say("retrying in 3 s")
        await asyncio.sleep(3)
    await say(f"no bottle after {MAX_TRIES} attempts")
    return False


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
