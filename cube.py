"""Find a coloured cube on the table and say where it is in the arm's coordinates. Read-only.

Colour, not a detector: a stock detector does not know "cube", but a saturated red, green, blue or
yellow blob on a wooden table is unmistakable. The blob's centre pixel plus the depth map gives the
cube's top face; the grasp is straight down onto it.

    .venv/bin/python cube.py              # every colour it can see
    .venv/bin/python cube.py green        # one colour
"""
import asyncio
import math
import struct
import sys

import cv2
import numpy as np
from viam.components.camera import Camera
from viam.proto.common import Pose, PoseInFrame

from camgrab import grab
from viamarm import connect

CUBE_MM = 48.0          # side of the foam cubes; the table is z=0 so the top is at CUBE_MM
MIN_AREA = 150          # pixels at 640x480; smaller blobs are noise
HOME_AREA = (250, 3000) # a cube's blob area from the home view (measured 29 Oct: 518 to 786 px)
MAX_REACH = 750.0
# hue ranges (OpenCV hue 0..179), saturation and value floors; red wraps around 0
COLOURS = {
    "red":    [((0, 120, 80), (8, 255, 255)), ((170, 120, 80), (179, 255, 255))],
    "green":  [((40, 80, 60), (85, 255, 255))],
    "blue":   [((95, 120, 60), (130, 255, 255))],
    "yellow": [((20, 120, 100), (35, 255, 255))],
}


def cam_offset():
    """(dx, dy) mm to ADD to a camera estimate to get where the gripper must go. Measured 29 Oct 18:38 by letting
    the gripper place the cube and the camera look: camera minus gripper = (-25, +14), so add (+25, -14).
    Stored in data/cam_offset.json by the calibration; (0, 0) if the file is missing."""
    import json, os
    f = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cam_offset.json")
    if not os.path.exists(f):
        return 0.0, 0.0
    ox, oy = json.load(open(f))["camera_minus_gripper_mm"]
    return -ox, -oy


def depth_mm(depth, u, v, w_img, h_img):
    w, h = struct.unpack(">QQ", depth[8:24])
    x, y = int(u / w_img * w), int(v / h_img * h)
    vals = []
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            xx, yy = min(max(x + dx, 0), w - 1), min(max(y + dy, 0), h - 1)
            d = struct.unpack(">H", depth[24 + 2 * (yy * w + xx):26 + 2 * (yy * w + xx)])[0]
            if d:
                vals.append(d)
    vals.sort()
    return vals[len(vals) // 2] if vals else 0


async def to_world(machine, intr, u, v, mm):
    x = (u - intr.center_x_px) * mm / intr.focal_x_px
    y = (v - intr.center_y_px) * mm / intr.focal_y_px
    p = (await machine.transform_pose(PoseInFrame(reference_frame="cam", pose=Pose(x=x, y=y, z=mm, o_z=1)), "world")).pose
    return p.x, p.y, p.z


def blobs(bgr, colour):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = None
    for lo, hi in COLOURS[colour]:
        m = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA:
            x, y, w, h = stats[i, :4]
            out.append((int(stats[i, cv2.CC_STAT_AREA]), (float(cents[i][0]), float(cents[i][1])), (x, y, w, h)))
    out.sort(reverse=True)
    return out


async def pixel_to_plane(machine, intr, u, v, z_plane):
    """Where the ray through pixel (u, v) meets the horizontal plane z = z_plane, in the arm's frame.
    No depth needed: used for the close looks, where the depth camera is too near to read."""
    x = (u - intr.center_x_px) / intr.focal_x_px
    y = (v - intr.center_y_px) / intr.focal_y_px
    o = (await machine.transform_pose(PoseInFrame(reference_frame="cam", pose=Pose(x=0, y=0, z=0, o_z=1)), "world")).pose
    q = (await machine.transform_pose(PoseInFrame(reference_frame="cam", pose=Pose(x=x * 1000, y=y * 1000, z=1000, o_z=1)), "world")).pose
    dx, dy, dz = q.x - o.x, q.y - o.y, q.z - o.z
    if abs(dz) < 1e-6:
        return None
    t = (z_plane - o.z) / dz
    if t <= 0:
        return None
    cx, cy = cam_offset()
    return o.x + t * dx + cx, o.y + t * dy + cy, z_plane


async def find_cube_close(machine, cam, intr, colour):
    """The cube's centre from a close look: biggest blob of the colour, ray to the known top height."""
    jpeg, _ = await grab(cam, want_depth=False)
    bgr = cv2.imdecode(np.frombuffer(jpeg, "uint8"), cv2.IMREAD_COLOR)
    bl = blobs(bgr, colour)
    if not bl:
        return None
    area, (cu, cv_), box = bl[0]
    return await pixel_to_plane(machine, intr, cu, cv_, CUBE_MM), area


async def find_cubes(machine, cam, intr, colours=None):
    """{colour: (world_top_centre, area, box)} for each colour seen, using the biggest blob of each."""
    jpeg, depth = await grab(cam, want_depth=True)
    bgr = cv2.imdecode(np.frombuffer(jpeg, "uint8"), cv2.IMREAD_COLOR)
    h_img, w_img = bgr.shape[:2]
    found = {}
    for colour in (colours or COLOURS):
        bl = blobs(bgr, colour)
        if not bl:
            continue
        cands = []
        for area, (cu, cv_), box in bl:
            bw, bh = box[2], box[3]
            if not (0.55 < bw / max(bh, 1) < 1.8) or not (HOME_AREA[0] <= area <= HOME_AREA[1]):
                continue                                  # not a cube from here: the orange stand, the chair, tape
            mm = depth_mm(depth, cu, cv_, w_img, h_img)
            if not mm:
                continue
            top = await to_world(machine, intr, cu, cv_, mm)
            cxo, cyo = cam_offset()
            top = (top[0] + cxo, top[1] + cyo, top[2])
            # on the table: z near 0 (the depth frame reads about 4 cm low) and not out in the room (blue chairs, tape)
            if -80 < top[2] < 120 and math.hypot(top[0], top[1]) < 1300:
                cands.append((math.hypot(top[0], top[1]), top, area, box))
        if cands:
            # of the cube-shaped blobs on the table, the nearest to the arm: the cubes sit near it, the orange
            # phone stand and the phone sit by the laptop further out
            cands.sort()
            _, top, area, box = cands[0]
            found[colour] = (top, area, box)
    return found


async def main(colours):
    async with await connect() as machine:
        cam = Camera.from_robot(machine, "cam")
        intr = (await cam.get_properties()).intrinsic_parameters
        found = await find_cubes(machine, cam, intr, colours)
        if not found:
            print("no cube seen"); return
        for colour, (t, area, box) in found.items():
            d = math.hypot(t[0], t[1])
            print(f"{colour:7s} top at x={t[0]:.0f} y={t[1]:.0f} z={t[2]:.0f} mm | {d:.0f} mm from the base | blob {area} px"
                  + ("" if d <= MAX_REACH else "  TOO FAR"))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:] or None))
