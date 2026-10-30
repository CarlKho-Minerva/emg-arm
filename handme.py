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
    """Watch for Carl's steady left hand, plan to STANDOFF short of it, open, return to greet."""
    machine, cam, intr, det, pose, motion, gripper, greet, live = (rig.machine, rig.cam, rig.intr, rig.det, rig.pose,
                                                                   rig.motion, rig.gripper, rig.greet, rig.live)
    if True:
        # 2. watch for a steady left hand
        await say("watching for your left hand")
        seen, last, sm = 0, None, None
        frame_n = 0
        body_cache = (None, None, None)
        while True:
            jpeg, _ = await grab(cam, want_depth=False)
            frame_n += 1
            if frame_n % 3 == 1:
                body = arm_target(pose, jpeg, "left"); ws = wrists(pose, jpeg)
                body_cache = (body, ws["right"], ws["left"])
            body, other, mine = body_cache
            seen_wrist = bool(body and body[2] == "wrist")
            near = (body[0], body[1]) if seen_wrist else sm
            c = hand_centre(det, jpeg, near=near, want=None if seen_wrist else "watch", avoid=other, keep=mine)
            if c is None:
                seen, last = 0, None
                continue
            sm = c
            if last and math.dist(c, last) < STEADY_MOVE:
                seen += 1
            else:
                seen = 1
            last = c
            if seen >= STEADY_FRAMES:
                break
        await say(f"hand steady at ({c[0]:.2f},{c[1]:.2f}) of the picture")

        # measure it in the arm's coordinates (needs the depth map; a few tries in case the depth is missing)
        world = None
        for _ in range(6):
            r = await hand_in_world(machine, cam, det, intr)
            if r:
                _, mm, _, world = r
                break
        if not world:
            await say("could not get the hand's distance; stopping"); return
        hx, hy, hz = world
        tx, ty, tz = target_for(hx, hy, hz)
        await say(f"hand at x={hx:.0f} y={hy:.0f} z={hz:.0f} mm from the base ({mm} mm from the camera)")
        await say(f"target {STANDOFF:.0f} mm short: x={tx:.0f} y={ty:.0f} z={tz:.0f}")

        # 3. plan and go
        if live:
            # first choice: gripper pointing at Carl, along the line from the base to his hand, level;
            # fallback: pointing down (a pick pose), which the planner can nearly always reach
            flat = math.hypot(hx, hy)
            facing = Pose(x=tx, y=ty, z=tz, o_x=hx / flat, o_y=hy / flat, o_z=0, theta=0)
            down = Pose(x=tx, y=ty, z=tz, o_x=0, o_y=0, o_z=-1, theta=0)
            ok = False
            for label, pose_ in (("facing you", facing), ("pointing down", down)):
                t0 = time.time()
                try:
                    ok = await motion.move(component_name="gripper", destination=PoseInFrame(reference_frame="world", pose=pose_))
                except Exception as e:
                    await say(f"planner, {label}: {str(e)[:90]}"); ok = False
                await say(f"planner, {label}: ok={ok} in {time.time() - t0:.1f} s")
                if ok:
                    break
            if not ok:
                await say("planner found no path; back to greet"); await greet.set_position(2); return
        else:
            await say("would ask the planner to go there now")

        # 4. open, wait, 5. greet
        if live:
            await gripper.open()
        await say(f"gripper open: take it ({TAKE_S:.0f} s)")
        await asyncio.sleep(TAKE_S if live else 1)
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
