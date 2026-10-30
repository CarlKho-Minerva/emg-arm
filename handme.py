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
RELEASE_MM = 120.0    # hand this close to the gripper...
RELEASE_S = 1.0       # ...for this long: let go
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
    """Follow Carl's left hand with the bottle: re-plan whenever the hand moves more than REPLAN_MM,
    and let go only when the hand has stayed within RELEASE_MM of the gripper for RELEASE_S."""
    machine, cam, intr, det, pose, motion, gripper, greet, live = (rig.machine, rig.cam, rig.intr, rig.det, rig.pose,
                                                                   rig.motion, rig.gripper, rig.greet, rig.live)
    await say("following your left hand; it lets go when your hand stays at the gripper for a second")
    target = None
    close_since = None
    t_start = time.time()
    while time.time() - t_start < FOLLOW_MAX_S:
        r = await hand_in_world(machine, cam, det, intr)
        if r is None:
            await asyncio.sleep(0.05); continue
        _, mm, _, (hx, hy, hz) = r
        gp = (await motion.get_pose("gripper", "world")).pose
        gap = math.dist((hx, hy, hz), (gp.x, gp.y, gp.z))
        if gap < RELEASE_MM:
            close_since = close_since or time.time()
            if time.time() - close_since >= RELEASE_S:
                await say(f"your hand has been at the gripper ({gap:.0f} mm) for {RELEASE_S:.0f} s: letting go")
                if live:
                    await gripper.open()
                await asyncio.sleep(2)
                break
        else:
            close_since = None
        tx, ty, tz = target_for(hx, hy, hz)
        if target is None or math.dist(target, (tx, ty, tz)) > REPLAN_MM:
            await say(f"hand at x={hx:.0f} y={hy:.0f} z={hz:.0f} ({mm} mm from the camera); going to x={tx:.0f} y={ty:.0f} z={tz:.0f}")
            target = (tx, ty, tz)
            if live:
                flat = math.hypot(hx, hy)
                facing = Pose(x=tx, y=ty, z=tz, o_x=hx / flat, o_y=hy / flat, o_z=0, theta=0)
                try:
                    ok = await motion.move(component_name="gripper", destination=PoseInFrame(reference_frame="world", pose=facing))
                except Exception as e:
                    ok = False; await say(f"planner: {str(e)[:80]}")
                if not ok:
                    await say("planner found no path to that; holding here")
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
