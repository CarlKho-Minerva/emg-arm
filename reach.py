"""Reach toward your hand, through the planner, so the walls count.

Each round: find the hand in the camera, work out where it is in the robot's coordinates,
pick a spot STANDOFF mm short of it along the line from the arm's base, and ask the motion
planner to bring the gripper there, fingers pointing down. The planner steers around the
table and every wall, including wall-carl. If the hand is out of reach, the target is
clamped to REACH mm from the base so the planner never gets an impossible ask.

    .venv/bin/python reach.py             # dry run: prints targets, moves nothing
    .venv/bin/python reach.py --live      # one hand on the red button

Slow by design: one planned move per round, and no new move while one is in flight.
"""
import argparse
import asyncio
import math
import time

from viam.components.camera import Camera
from viam.proto.common import Pose, PoseInFrame
from viam.services.motion import MotionClient

from follow import detector
from handpos import hand_in_world
from viamarm import arm_of, connect

STANDOFF = 200.0    # mm to stop short of the hand
REACH = 800.0       # mm, furthest the gripper may be from the base (measured full stretch: 851)
MIN_Z = 120.0       # mm above the table, never lower
MOVE_IF = 40.0      # mm: ignore changes smaller than this (the hand jitters)


def target_for(hx, hy, hz):
    """A point on the base-to-hand line, STANDOFF short of the hand, within reach, off the table."""
    flat = math.hypot(hx, hy)
    want = min(max(flat - STANDOFF, 150.0), REACH)
    s = want / flat
    return hx * s, hy * s, max(hz + 100.0, MIN_Z)


async def main(live):
    det = detector()
    async with await connect() as machine:
        cam = Camera.from_robot(machine, "cam")
        intr = (await cam.get_properties()).intrinsic_parameters
        motion = MotionClient.from_robot(machine, "builtin")
        arm = arm_of(machine)
        last = None
        while True:
            r = await hand_in_world(machine, cam, det, intr)
            stamp = time.strftime("%H:%M:%S")
            if r is None:
                print(stamp, "no hand"); continue
            _, mm, _, (hx, hy, hz) = r
            tx, ty, tz = target_for(hx, hy, hz)
            line = f"{stamp} hand world ({hx:.0f},{hy:.0f},{hz:.0f}) -> target ({tx:.0f},{ty:.0f},{tz:.0f})"
            if last and math.dist(last, (tx, ty, tz)) < MOVE_IF:
                print(line, "(same spot)"); continue
            if live:
                if await arm.is_moving():
                    print(line, "(still moving)"); continue
                dest = PoseInFrame(reference_frame="world",
                                   pose=Pose(x=tx, y=ty, z=tz, o_x=0, o_y=0, o_z=-1, theta=0))
                t0 = time.time()
                ok = await motion.move(component_name="gripper", destination=dest)
                line += f"  MOVED ok={ok} in {time.time() - t0:.1f}s"
                last = (tx, ty, tz)
            else:
                last = (tx, ty, tz)
            print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    asyncio.run(main(ap.parse_args().live))
