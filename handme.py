"""Hand Carl the thing in the gripper.

    .venv/bin/python handme.py            # dry run: prints what it would do, moves nothing
    .venv/bin/python handme.py --live     # the real thing; a hand on the red button

The steps, each printed as it happens:
  1. greet pose; gripper opens; Carl puts the bottle in; after HOLD_S seconds the gripper closes.
  2. watch for Carl's LEFT hand (same chooser as follow.py: body anchor, right-wrist veto, hand net).
     When it has been seen in STEADY_FRAMES frames in a row and moved less than STEADY_MOVE between
     them, measure where it is in the arm's coordinates (pixel + depth, then Viam's frame maths).
  3. ask the motion planner to bring the gripper to STANDOFF mm short of the hand, within REACH of the
     base and above MIN_Z. The planner picks the joints and goes around the walls.
  4. open the gripper; wait TAKE_S seconds for Carl to take the bottle.
  5. back to greet.
Nothing here learns; the planner does the inverse kinematics.
"""
import argparse
import asyncio
import math
import os
import time

from viam.components.camera import Camera
from viam.components.switch import Switch
from viam.proto.common import Pose, PoseInFrame
from viam.services.motion import MotionClient

from camgrab import grab
from follow import MIN_HAND, detector, hand_centre
from handpos import hand_in_world
from tracker import arm_target, pose_detector, wrists
from viamarm import arm_of, connect, gripper_of

HOLD_S = 6.0          # time for Carl to put the bottle in the open gripper
STEADY_FRAMES = 8     # frames in a row the hand must be seen
STEADY_MOVE = 0.05    # fraction of the picture the hand may drift between those frames
STANDOFF = 200.0      # mm short of the hand
REACH = 800.0         # mm, furthest the gripper may be from the base
MIN_Z = 150.0         # mm above the table
TAKE_S = 5.0          # seconds the gripper stays open for Carl to take it
REPLAN_MM = 50.0      # re-plan when the hand has moved more than this
STEADY_MM = 60.0      # the hand counts as still if it moves less than this between frames
DWELL_S = 2.0         # hand still, arm at the standoff, for this long: let go
SAFE_X = 850.0        # mm: direct moves never go further forward than this (the laptop wall is at 950)
FOLLOW_MAX_S = 60.0   # give up following after this


def target_for(hx, hy, hz):
    flat = math.hypot(hx, hy)
    want = min(max(flat - STANDOFF, 150.0), REACH)
    s = want / flat
    return hx * s, hy * s, max(hz, MIN_Z)


async def say(msg):
    print(f"{time.strftime('%H:%M:%S')}  {msg}", flush=True)


class Rig:
    """Everything the steps need, built once per connection."""
    def __init__(self, machine, live):
        self.machine, self.live = machine, live
        self.cam = Camera.from_robot(machine, "cam")
        self.motion = MotionClient.from_robot(machine, "builtin")
        self.arm, self.gripper = arm_of(machine), gripper_of(machine)
        self.greet = Switch.from_robot(machine, "greet")
        self.home = Switch.from_robot(machine, "home")      # the wide view of the table
        self.det, self.pose = detector(), pose_detector()
        self.intr = None

    async def setup(self):
        self.intr = (await self.cam.get_properties()).intrinsic_parameters
        return self


async def place_bottle(rig):
    """Greet, open, give Carl HOLD_S seconds to put the bottle in, close."""
    await say("to greet")
    if rig.live:
        await rig.greet.set_position(2); await asyncio.sleep(4)
        await rig.gripper.open()
    await say(f"gripper open: put the bottle in ({HOLD_S:.0f} s)")
    await asyncio.sleep(HOLD_S if rig.live else 1)
    if rig.live:
        held = await rig.gripper.grab()
        await say(f"gripper closed, holding something: {held}")
    else:
        await say("gripper would close now")


async def handover(rig):
    """Bring the bottle to Carl's steady left hand and let go once he has had DWELL_S to take it.

    The wrist camera is behind the bottle during the carry, so it cannot see fingers at the gripper.
    Rule instead: find the hand, go to STANDOFF short of it (direct move, inside the safe box), then
    if the hand stays put (within STEADY_MM) for DWELL_S, open. Every frame is saved under
    data/handover/<run>/ for review."""
    machine, cam, intr, det, arm, gripper, greet, live = (rig.machine, rig.cam, rig.intr, rig.det,
                                                           rig.arm, rig.gripper, rig.greet, rig.live)
    rec = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "handover", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(rec, exist_ok=True)
    await say(f"following your left hand (recording to {os.path.basename(rec)}); hold your hand still 2 s to take it")
    target = None
    at_target = None      # when the arm reached the hand's standoff
    still_since = None
    last_hand = None
    t_start = time.time()
    while time.time() - t_start < FOLLOW_MAX_S:
        r = await hand_in_world(machine, cam, det, intr)
        jpeg, _ = await grab(cam, want_depth=False)
        open(os.path.join(rec, f"{time.time():.3f}.jpg"), "wb").write(jpeg)
        if r is None:
            still_since = None
            continue
        _, mm, _, (hx, hy, hz) = r
        hand = (hx, hy, hz)
        moved = last_hand is None or math.dist(hand, last_hand) > STEADY_MM
        last_hand = hand
        if moved:
            still_since = None
        else:
            still_since = still_since or time.time()
        tx, ty, tz = target_for(hx, hy, hz)
        tx = min(tx, SAFE_X); tz = max(tz, MIN_Z)
        if target is None or math.dist(target, (tx, ty, tz)) > REPLAN_MM:
            await say(f"hand at x={hx:.0f} y={hy:.0f} z={hz:.0f} ({mm} mm away); gripper to x={tx:.0f} y={ty:.0f} z={tz:.0f}")
            target = (tx, ty, tz); at_target = None
            if live:
                flat = math.hypot(hx, hy)
                try:
                    await arm.move_to_position(Pose(x=tx, y=ty, z=tz, o_x=hx / flat, o_y=hy / flat, o_z=0, theta=0))
                    at_target = time.time()
                except Exception as e:
                    await say(f"direct move refused: {str(e)[:80]}")
            else:
                at_target = time.time()
        if at_target and still_since and time.time() - still_since >= DWELL_S and time.time() - at_target >= DWELL_S:
            await say(f"at your hand and it has been still {DWELL_S:.0f} s: letting go")
            if live:
                await gripper.open()
            await asyncio.sleep(2)
            break
    else:
        await say(f"no handover in {FOLLOW_MAX_S:.0f} s; keeping the bottle")
    if live:
        await greet.set_position(2)
    await say("back to greet; done")


async def main(live):
    async with await connect() as machine:
        rig = await Rig(machine, live).setup()
        await place_bottle(rig)
        await handover(rig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    asyncio.run(main(ap.parse_args().live))
