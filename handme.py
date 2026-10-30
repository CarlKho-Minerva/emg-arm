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
STEADY_PX = 0.06      # smoothed hand moves less than this (fraction of the picture) between frames = still
STEADY_S = 0.6        # still for this long before ranging and moving
TAKE_SIZE = 0.18      # wrist-to-knuckle over this fraction of the picture width = the palm is close under the claw
TAKE_CENTRE = 0.18    # ...and within this of the gripper's point in the picture
TAKE_S = 0.7          # ...for this long: let go
DROP_ABOVE = 120.0    # mm above the palm for the claw pose
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
    """Follow Carl's left hand with the cube the way the coach's follower does (turn to keep it centred,
    colour frames only, the tuned gains), then bring the cube to it and let go.

    Phase 1, follow: the same chooser as follow.py (body-wrist anchor, right-wrist veto, hand net,
    smoothing); the waist (joint 0) pans and the elbow (joint 2) tilts with kp 4, streamed moves, so
    the hand stays in the picture. Phase 2, when the hand is in the dead zone and still for STEADY_S:
    ONE depth read, direct move to STANDOFF short of it. Phase 3: DWELL_S still at the target, open."""
    machine, cam, intr, det, pose, arm, gripper, greet, live = (rig.machine, rig.cam, rig.intr, rig.det, rig.pose,
                                                                rig.arm, rig.gripper, rig.greet, rig.live)
    import follow as F
    from follow import DEADBAND, GAIN, SMOOTH, STEP_CAP, depth_at, hand_centre
    from tracker import arm_target, wrists
    from viam.proto.component.arm import JointPositions
    rec = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "handover", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(rec, exist_ok=True)
    await say(f"following your left hand (recording {os.path.basename(rec)}); hold it still, the cube comes to it")
    if live:
        await arm.do_command({"set_speed": 60.0, "set_acceleration": 200.0})
    sm = None; still_since = None; frame_n = 0; body_cache = (None, None, None)
    at_target = None; last_seen = 0.0; seen_n = 0; last_beat = 0.0; big_since = None
    under = (0.5, 0.5)
    pending = None
    adx = ady = 0.0
    t_start = time.time()
    try:
        while time.time() - t_start < FOLLOW_MAX_S:
            if time.time() - last_beat > 3:
                last_beat = time.time()
                await say(f"   ...{frame_n} frames, hand seen in {seen_n}; "
                          + (f"still {time.time() - still_since:.1f} s" if still_since else "not still")
                          + ("; at target" if at_target else "") + f"; hand size {F.LAST_SIZE:.2f}")
            jpeg, _ = await grab(cam, want_depth=False)
            open(os.path.join(rec, f"{time.time():.3f}.jpg"), "wb").write(jpeg)
            frame_n += 1
            if frame_n % 3 == 1:
                body = arm_target(pose, jpeg, "left"); ws = wrists(pose, jpeg)
                body_cache = (body, ws["right"], ws["left"])
            body, other, mine = body_cache
            seen_wrist = bool(body and body[2] == "wrist")
            near = (body[0], body[1]) if seen_wrist else sm
            if at_target:
                # above the palm: the only hand under the gripper is the one taking the cube. No classifier
                # (it was trained on the back of the hand and rejected a palm-up hand from above, 19:01)
                raw = hand_centre(det, jpeg, use_net=False)
            else:
                raw = hand_centre(det, jpeg, near=near, want=None if seen_wrist else "watch", avoid=other, keep=mine)
            if raw is None:
                if time.time() - last_seen > 0.5:          # a short gap does not reset the clock
                    still_since = None; sm = None
                continue
            seen_n += 1; last_seen = time.time()
            # "take it": only once the arm is above the palm (claw pose), and the palm is centred under it
            # and big in the picture; TAKE_S of that and it opens. Never while still following.
            ex0, ey0 = raw[0] - under[0], raw[1] - under[1]
            if at_target and F.LAST_SIZE >= TAKE_SIZE and abs(ex0) < TAKE_CENTRE and abs(ey0) < TAKE_CENTRE:
                big_since = big_since or time.time()
                if time.time() - big_since >= TAKE_S:
                    await say(f"your hand is at the cube (size {F.LAST_SIZE:.2f} of the picture): letting go")
                    if live:
                        if pending and not pending.done():
                            pending.cancel()
                        await gripper.open()
                    await asyncio.sleep(2)
                    break
            else:
                big_since = None
            prev = sm
            sm = raw if sm is None else (sm[0] + SMOOTH * (raw[0] - sm[0]), sm[1] + SMOOTH * (raw[1] - sm[1]))
            ref = under if at_target else (0.5, 0.5)
            ex, ey = sm[0] - ref[0], sm[1] - ref[1]
            centred = abs(ex) < DEADBAND and abs(ey) < DEADBAND
            moving = prev is not None and math.dist(sm, prev) > STEADY_PX
            if moving:
                still_since = None
            elif still_since is None:
                still_since = time.time()
            # phase 1: follow (only while not yet at the target)
            if live and at_target is None and not centred:
                dx = max(-STEP_CAP, min(STEP_CAP, -GAIN * ex)); dy = max(-STEP_CAP, min(STEP_CAP, GAIN * ey))
                adx += 0.7 * (dx - adx); ady += 0.7 * (dy - ady)
                j = list((await arm.get_joint_positions()).values)
                j[0] += adx; j[2] += ady
                if pending and not pending.done():
                    pending.cancel()
                pending = asyncio.create_task(arm.move_to_joint_positions(JointPositions(values=j)))
                continue
            # phase 2: centred and still: range once, go
            if at_target is None and centred and still_since and time.time() - still_since >= STEADY_S:
                if pending and not pending.done():
                    pending.cancel()
                _, depth = await grab(cam, want_depth=True)
                mm = depth_at(depth, *sm)
                if not mm or mm < 280:
                    await say(f"hand centred but too near to range ({mm} mm); move it back a little"); still_since = None; continue
                u, v = sm[0] * intr.width_px, sm[1] * intr.height_px
                x = (u - intr.center_x_px) * mm / intr.focal_x_px; y = (v - intr.center_y_px) * mm / intr.focal_y_px
                w = (await machine.transform_pose(PoseInFrame(reference_frame="cam", pose=Pose(x=x, y=y, z=mm, o_z=1)), "world")).pose
                hx, hy, hz = w.x, w.y, w.z
                # the claw pose: above the palm, pointing down, the camera looking at the hand
                tx, ty, tz = hx, hy, hz + DROP_ABOVE
                flat = math.hypot(tx, ty)
                if flat > REACH:
                    tx, ty = tx * REACH / flat, ty * REACH / flat
                tx = min(tx, SAFE_X); tz = max(tz, MIN_Z)
                await say(f"hand still at x={hx:.0f} y={hy:.0f} z={hz:.0f} ({mm} mm away); claw to x={tx:.0f} y={ty:.0f} z={tz:.0f}, pointing down")
                if live:
                    try:
                        await asyncio.wait_for(arm.move_to_position(Pose(x=tx, y=ty, z=tz, o_x=0, o_y=0, o_z=-1, theta=0)), 20)
                    except Exception as e:
                        await say(f"direct move refused: {str(e)[:80]}"); still_since = None; continue
                at_target = time.time(); still_since = time.time()
                # where the gripper's axis meets the palm plane, as seen by the camera: the point the palm must
                # sit on (the lens is 7 cm beside the gripper, so it is not the picture's centre)
                try:
                    gp = (await machine.transform_pose(PoseInFrame(reference_frame="world", pose=Pose(x=tx, y=ty, z=hz, o_z=1)), "cam")).pose
                    under = ((intr.focal_x_px * gp.x / gp.z + intr.center_x_px) / intr.width_px,
                             (intr.focal_y_px * gp.y / gp.z + intr.center_y_px) / intr.height_px)
                    await say(f"   the gripper is over picture point ({under[0]:.2f},{under[1]:.2f}); the palm must sit there")
                except Exception as e:
                    await say(f"   could not project the gripper into the picture ({str(e)[:50]}); using the centre")
                continue
            # phase 3: above the palm; keep nudging to stay centred over it (small pan/tilt only)
            if at_target and live and not centred and time.time() - at_target > 1.0:
                dx = max(-2.0, min(2.0, -GAIN * ex)); dy = max(-2.0, min(2.0, GAIN * ey))
                j = list((await arm.get_joint_positions()).values)
                j[0] += dx; j[2] += dy
                if pending and not pending.done():
                    pending.cancel()
                pending = asyncio.create_task(arm.move_to_joint_positions(JointPositions(values=j)))
        else:
            await say(f"no handover in {FOLLOW_MAX_S:.0f} s; keeping it")
    finally:
        if pending and not pending.done():
            pending.cancel()
        if live:
            await arm.do_command({"set_speed": 60.0, "set_acceleration": 500.0})
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
