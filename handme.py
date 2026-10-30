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
STEADY_PX = 0.03      # smoothed hand moves less than this (fraction of the picture) between frames = still
STEADY_S = 1.0        # still for this long before ranging and moving
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


class StepTimeout(Exception):
    pass


async def step(label, coro, seconds=20.0):
    """Run one robot call with a time limit and a log line before and after. A call that never
    returns (practice hung 4 min on a go-to, 29 Oct 18:03) becomes a StepTimeout the loop can handle."""
    await say(f"> {label}")
    t0 = time.time()
    try:
        r = await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError:
        await say(f"! {label}: no answer in {seconds:.0f} s")
        raise StepTimeout(label)
    if time.time() - t0 > 3:
        await say(f"  {label}: {time.time() - t0:.1f} s")
    return r


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
    """Bring the cube to Carl's left hand using the tuned follower's chooser, then let go on a 2 s dwell.

    Colour frames only (no depth) at the follower's rate: the hand is picked the way follow.py picks it
    (body-wrist anchor every 3rd frame, right-wrist veto, the hand net, smoothing). When the smoothed hand
    has been steady for STEADY_S, ONE depth read gives its distance, the gripper moves to STANDOFF short of
    it (direct move inside the safe box), and after DWELL_S with the hand still there, the gripper opens.
    Every frame is saved under data/handover/<run>/."""
    machine, cam, intr, det, pose, arm, gripper, greet, live = (rig.machine, rig.cam, rig.intr, rig.det, rig.pose,
                                                                rig.arm, rig.gripper, rig.greet, rig.live)
    from follow import SMOOTH, depth_at, hand_centre
    from tracker import arm_target, wrists
    rec = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "handover", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(rec, exist_ok=True)
    await say(f"following your left hand (recording {os.path.basename(rec)}); hold it still and the cube comes to it")
    sm = None; still_since = None; frame_n = 0; body_cache = (None, None, None)
    at_target = None; target = None
    t_start = time.time(); last_beat = 0.0; seen_n = 0
    while time.time() - t_start < FOLLOW_MAX_S:
        if time.time() - last_beat > 3:
            last_beat = time.time()
            await say(f"   ...{frame_n} frames, hand seen in {seen_n}; "
                      + (f"still for {time.time() - still_since:.1f} s" if still_since else "not still")
                      + ("; at target" if at_target else ""))
        jpeg, _ = await grab(cam, want_depth=False)
        open(os.path.join(rec, f"{time.time():.3f}.jpg"), "wb").write(jpeg)
        frame_n += 1
        if frame_n % 3 == 1:
            body = arm_target(pose, jpeg, "left"); ws = wrists(pose, jpeg)
            body_cache = (body, ws["right"], ws["left"])
        body, other, mine = body_cache
        seen_wrist = bool(body and body[2] == "wrist")
        near = (body[0], body[1]) if seen_wrist else sm
        raw = hand_centre(det, jpeg, near=near, want=None if seen_wrist else "watch", avoid=other, keep=mine)
        if raw is None:
            still_since = None
            continue
        seen_n += 1
        prev = sm
        sm = raw if sm is None else (sm[0] + SMOOTH * (raw[0] - sm[0]), sm[1] + SMOOTH * (raw[1] - sm[1]))
        if prev is not None and math.dist(sm, prev) < STEADY_PX:
            still_since = still_since or time.time()
        else:
            still_since = None
            at_target = None
        if still_since and time.time() - still_since >= STEADY_S and at_target is None:
            # one depth read for the distance, then go
            _, depth = await grab(cam, want_depth=True)
            mm = depth_at(depth, *sm)
            if not mm or mm < 280:
                await say(f"hand steady but too near the camera to range ({mm} mm); move it back a little")
                still_since = None
                continue
            u, v = sm[0] * intr.width_px, sm[1] * intr.height_px
            x = (u - intr.center_x_px) * mm / intr.focal_x_px; y = (v - intr.center_y_px) * mm / intr.focal_y_px
            w = (await machine.transform_pose(PoseInFrame(reference_frame="cam", pose=Pose(x=x, y=y, z=mm, o_z=1)), "world")).pose
            hx, hy, hz = w.x, w.y, w.z
            tx, ty, tz = target_for(hx, hy, hz); tx = min(tx, SAFE_X); tz = max(tz, MIN_Z)
            await say(f"hand steady at x={hx:.0f} y={hy:.0f} z={hz:.0f} ({mm} mm away); bringing the cube to x={tx:.0f} y={ty:.0f} z={tz:.0f}")
            target = (tx, ty, tz)
            if live:
                flat = math.hypot(hx, hy)
                try:
                    await arm.move_to_position(Pose(x=tx, y=ty, z=tz, o_x=hx / flat, o_y=hy / flat, o_z=0, theta=0))
                except Exception as e:
                    await say(f"direct move refused: {str(e)[:80]}"); continue
            at_target = time.time()
            still_since = time.time()
            continue
        if at_target and still_since and time.time() - at_target >= DWELL_S and time.time() - still_since >= DWELL_S:
            await say(f"at your hand, still for {DWELL_S:.0f} s: letting go")
            if live:
                await gripper.open()
            await asyncio.sleep(2)
            break
    else:
        await say(f"no handover in {FOLLOW_MAX_S:.0f} s; keeping it")
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
