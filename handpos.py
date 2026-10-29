"""Where is your hand, in the robot's own coordinates? Read-only.

Pixel + depth -> a point in front of the camera -> the same point measured from the
arm's base (the "world" frame), using Viam's own frame maths.

    .venv/bin/python handpos.py
"""
import asyncio
import struct

from viam.components.camera import Camera
from viam.proto.common import Pose, PoseInFrame

from camgrab import grab
from follow import depth_at, detector, hand_centre
from viamarm import connect


async def hand_in_world(machine, cam, det, intr):
    jpeg, depth = await grab(cam)
    c = hand_centre(det, jpeg)
    if c is None:
        return None
    mm = depth_at(depth, *c)
    if not mm:
        return None
    u, v = c[0] * intr.width_px, c[1] * intr.height_px
    # pinhole: a pixel and its depth give a point, z straight out of the lens
    x = (u - intr.center_x_px) * mm / intr.focal_x_px
    y = (v - intr.center_y_px) * mm / intr.focal_y_px
    in_cam = PoseInFrame(reference_frame="cam", pose=Pose(x=x, y=y, z=mm, o_z=1))
    w = (await machine.transform_pose(in_cam, "world")).pose
    return c, mm, (x, y, mm), (w.x, w.y, w.z)


async def main():
    det = detector()
    async with await connect() as machine:
        cam = Camera.from_robot(machine, "cam")
        intr = (await cam.get_properties()).intrinsic_parameters
        for _ in range(8):
            r = await hand_in_world(machine, cam, det, intr)
            if r is None:
                print("no hand"); continue
            c, mm, cp, wp = r
            print(f"pixel ({c[0]:.2f},{c[1]:.2f}) depth {mm} mm | from camera x={cp[0]:.0f} y={cp[1]:.0f} z={cp[2]:.0f}"
                  f" | world x={wp[0]:.0f} y={wp[1]:.0f} z={wp[2]:.0f} mm")

if __name__ == "__main__":
    asyncio.run(main())
