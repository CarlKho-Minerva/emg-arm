"""Find the bottle on the table, pick it up, hand it to Carl.

    .venv/bin/python pick.py            # dry run: finds the bottle, prints the grasp, moves nothing
    .venv/bin/python pick.py --live     # the real thing; a hand on the red button
    .venv/bin/python pick.py --live --cube red    # a coloured cube instead (straight-down grasp)

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
LABEL_Z = 135.0        # mm above the table, on the label; the fingers hang below the steered point and
                       # scratched the table at 100 (29 Oct 17:38)
PRE_MM = 160.0         # stop this far short of the bottle's axis before sliding in
MIN_PRE_DIST = 530.0   # mm from the base: the nearest a level pre-grasp pose can be without self-collision
IN_MM = 35.0           # gripper frame this far short of the axis when closing (fingers reach past it)
LIFT_MM = 100.0
MAX_TRIES = 4
CUBE_REACH = 750.0     # mm from the base for a straight-down grasp
CUBE_ABOVE = 160.0     # mm: hover height of the steered point over the cube
CUBE_LOOKS = (240.0, 160.0)   # hover heights for the close looks (coarse to fine), before the descent
CUBE_PASSES = 3        # look-descend passes per attempt before going back to home
CUBE_MAX_SHIFT = 80.0  # mm: a close look that moves the estimate more than this is a different blob; ignore it
CUBE_STAGES = (90.0, 70.0, 52.0)   # steered-point heights to close at, in order; the fingertips hang some
                                   # unknown distance below the steered point, so go down in steps and stop at the first hold
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
    seen, _ = await find_bottles(rig.machine, rig.cam, rig.intr)
    seen = [b for b in seen if b[0] >= 0.5]
    bottles = [b for b in seen if math.hypot(b[2][0], b[2][1]) <= MAX_REACH]
    if not bottles:
        if seen:
            d = math.hypot(seen[0][2][0], seen[0][2][1])
            await say(f"bottle seen but {d:.0f} mm from the base, over the {MAX_REACH:.0f} limit: slide it toward the arm")
        else:
            await say("no bottle seen from home")
        return False
    conf, box, c, top, g = bottles[0]
    cx, cy = c[0], c[1]
    flat = math.hypot(cx, cy)
    dx, dy = cx / flat, cy / flat                       # from the base toward the bottle, level
    # the pre-grasp point must not come nearer the base than MIN_PRE_DIST: at 49 cm a level wrist folds the arm
    # onto itself (planner: self-collision, 29 Oct 17:35); 53 to 55 cm worked
    pre_d = max(flat - PRE_MM, MIN_PRE_DIST)
    pre = (dx * pre_d, dy * pre_d, LABEL_Z)
    grasp = (cx - dx * IN_MM, cy - dy * IN_MM, LABEL_Z)
    await say(f"bottle (conf {conf:.2f}) at x={cx:.0f} y={cy:.0f}, {flat:.0f} mm from the base; "
              f"side grasp at the label, z={LABEL_Z:.0f}")
    if not rig.live:
        await say(f"would go to x={pre[0]:.0f} y={pre[1]:.0f} level, facing it; slide to x={grasp[0]:.0f} y={grasp[1]:.0f}; close; lift {LIFT_MM:.0f}")
        return True
    ok = False
    for tilt, theta in ((0.0, 0.0), (0.0, 90.0), (-0.35, 0.0), (-0.35, 90.0)):   # level first, then nosing down a little
        face = (dx, dy, tilt)
        ok = await plan_to(rig, *pre, f"beside the bottle (tilt {tilt:.2f}, theta {theta:.0f})", o=face, theta=theta)
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


async def pick_cube_once(rig, colour, attempt):
    """Top-down grasp of a coloured cube: above it, then down in stages, closing at each until the fingers hold."""
    from cube import CUBE_MM, find_cubes
    if attempt == 1:
        await say("to greet first; gripper open")
        if rig.live:
            await rig.greet.set_position(2); await asyncio.sleep(4)
            await rig.gripper.open(); await asyncio.sleep(1)
    await say(f"attempt {attempt}: to home (the view of the table)")
    if rig.live:
        await rig.home.set_position(2); await asyncio.sleep(5)
        await rig.gripper.open()
    found = await find_cubes(rig.machine, rig.cam, rig.intr, [colour])
    if colour not in found:
        await say(f"no {colour} cube seen from home"); return False
    (cx, cy, cz), area, box = found[colour]
    flat = math.hypot(cx, cy)
    if flat > CUBE_REACH:
        await say(f"{colour} cube seen but {flat:.0f} mm from the base, over the {CUBE_REACH:.0f} limit: slide the cubes toward the arm")
        return False
    await say(f"{colour} cube at x={cx:.0f} y={cy:.0f}, {flat:.0f} mm from the base (top read z={cz:.0f}; using the known height)")
    if not rig.live:
        await say(f"would go above it at z={CUBE_ABOVE:.0f}, then down through {CUBE_STAGES}, closing at each until held")
        return True
    # coarse to fine: hover high, look again and re-centre; hover lower, look again; then descend.
    # A miss comes back up to the low look height, looks again (the cube may have been nudged), and
    # tries the descent again, up to CUBE_PASSES times, before this attempt counts as failed.
    from cube import find_cube_close
    try:
        await rig.gripper.do_command({"set_gripper_speed": SOFT_SPEED})
    except Exception:
        pass
    held = False
    looks = list(CUBE_LOOKS)
    for pass_n in range(1, CUBE_PASSES + 1):
        for hover in looks:
            if not await plan_to(rig, cx, cy, hover, f"over the cube at z={hover:.0f}"):
                return False
            await asyncio.sleep(0.4)
            r = await find_cube_close(rig.machine, rig.cam, rig.intr, colour)
            if r is None or r[0] is None:
                await say(f"   close look at z={hover:.0f}: cube not seen from here; keeping the last estimate")
                continue
            (nx, ny, _), area = r
            shift = math.hypot(nx - cx, ny - cy)
            await say(f"   close look at z={hover:.0f}: cube at x={nx:.0f} y={ny:.0f} (moved the estimate {shift:.0f} mm, blob {area} px)")
            if shift < CUBE_MAX_SHIFT:
                cx, cy = nx, ny
            else:
                await say(f"   that is a jump of more than {CUBE_MAX_SHIFT:.0f} mm; not trusting it")
        if not await plan_to(rig, cx, cy, CUBE_ABOVE, "above the cube, centred"):
            return False
        for z in CUBE_STAGES:
            if not await plan_to(rig, cx, cy, z, f"down to z={z:.0f}"):
                break
            await rig.gripper.grab()
            try:
                posn = (await rig.gripper.do_command({"get": True})).get("pos", -1)
            except Exception:
                posn = -1
            await say(f"   closed at z={z:.0f}: finger position {posn:.0f}")
            if posn > HELD_POS:
                held = True
                break
            await rig.gripper.open(); await asyncio.sleep(0.5)
        if held:
            break
        await say(f"   pass {pass_n}: fingers closed on nothing; back up to look again")
        looks = [CUBE_LOOKS[-1]]              # the next pass only needs the low look
    try:
        await rig.gripper.do_command({"set_gripper_speed": 2000.0})
    except Exception:
        pass
    if not held:
        await say("no hold after the passes; opening"); await rig.gripper.open()
        await plan_to(rig, cx, cy, CUBE_ABOVE, "back up")
        return False
    await plan_to(rig, cx, cy, CUBE_ABOVE, "lift")
    rig.last_grasp = (cx, cy)
    await say(f"{colour} cube confirmed in the gripper; to greet, to find your hand")
    await rig.greet.set_position(2); await asyncio.sleep(5)
    return True


async def pick(rig, cube=None):
    """Try until the fingers confirm the object, up to MAX_TRIES."""
    for attempt in range(1, MAX_TRIES + 1):
        ok = await (pick_cube_once(rig, cube, attempt) if cube else pick_once(rig, attempt))
        if ok:
            return True
        if not rig.live:
            return False
        await say("retrying in 3 s")
        await asyncio.sleep(3)
    await say(f"nothing held after {MAX_TRIES} attempts")
    return False


async def _unused_pick(rig):
    """(the bottle-only loop, kept for reference)"""
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


async def main(live, cube=None):
    async with await connect() as machine:
        rig = await Rig(machine, live).setup()
        ok = await pick(rig, cube)
        if ok or not live:
            await handover(rig)
        else:
            await say("nothing in the gripper; back to greet")
            if live:
                await rig.greet.set_position(2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--cube", choices=["red", "green", "blue", "yellow"], help="pick this coloured cube instead of the bottle")
    a = ap.parse_args()
    asyncio.run(main(a.live, a.cube))
