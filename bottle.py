"""Find the bottle on the table and say where it is in the arm's coordinates. Read-only.

A stock object detector (YOLOv8n, knows "bottle" out of the box) gives the box around it in the
picture; the depth map gives how far its near surface is; Viam's frame maths turns that into a point
measured from the arm's base. The grasp point is the bottle's axis, a little below its top.

    .venv/bin/python bottle.py            # prints every bottle seen and the grasp point of the nearest
"""
import asyncio
import math
import struct

from viam.components.camera import Camera
from viam.proto.common import Pose, PoseInFrame

from camgrab import grab
from viamarm import connect

BOTTLE_RADIUS = 33.0    # mm, a 500 ml water bottle; the depth map sees the near surface, the axis is one radius further
BOTTLE_HEIGHT = 210.0   # mm, standing; the table is z=0 in the arm's frame (a hand on it read 1 to 2 cm), so the
                        # grasp height comes from the known height, not from the depth of the cap pixel
GRASP_BELOW_TOP = 45.0  # mm below the top of the bottle to close the fingers (on the neck/shoulder, not the cap)
MAX_REACH = 750.0       # mm from the base; further than this the arm cannot grasp from above
_model = None


def detector():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO("yolov8n.pt")
    return _model


def depth_mm(depth, u, v, w_img, h_img):
    """Median depth over a small patch around pixel (u, v); 0 where the camera has no reading."""
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


async def find_bottles(machine, cam, intr):
    """[(conf, box, centre_world, top_world, grasp_world)] for every bottle seen, nearest first."""
    jpeg, depth = await grab(cam, want_depth=True)
    import cv2, numpy as np
    bgr = cv2.imdecode(np.frombuffer(jpeg, "uint8"), cv2.IMREAD_COLOR)
    h_img, w_img = bgr.shape[:2]
    res = detector().predict(bgr, conf=0.3, verbose=False)[0]
    out = []
    for b in res.boxes:
        if detector().names[int(b.cls)] != "bottle":
            continue
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        cu, cv_ = (x1 + x2) / 2, (y1 + y2) / 2
        mm = depth_mm(depth, cu, cv_, w_img, h_img)
        if not mm:
            continue
        # the near surface of the bottle at its centre; its axis is one radius further along the same ray
        centre = await to_world(machine, intr, cu, cv_, mm + BOTTLE_RADIUS)
        top = (centre[0], centre[1], BOTTLE_HEIGHT)
        grasp = (centre[0], centre[1], BOTTLE_HEIGHT - GRASP_BELOW_TOP)
        if not (-50 < centre[2] < BOTTLE_HEIGHT + 50):
            continue                                      # not standing on the table; skip
        out.append((float(b.conf), (x1, y1, x2, y2), centre, top, grasp))
    out.sort(key=lambda t: math.hypot(t[2][0], t[2][1]))
    return out, (w_img, h_img)


async def main():
    async with await connect() as machine:
        cam = Camera.from_robot(machine, "cam")
        intr = (await cam.get_properties()).intrinsic_parameters
        bottles, size = await find_bottles(machine, cam, intr)
        if not bottles:
            print("no bottle seen"); return
        for conf, box, c, t, g in bottles:
            dist = math.hypot(c[0], c[1])
            print(f"bottle conf {conf:.2f}  box ({box[0]:.0f},{box[1]:.0f})-({box[2]:.0f},{box[3]:.0f}) of {size[0]}x{size[1]}")
            print(f"   centre x={c[0]:.0f} y={c[1]:.0f} z={c[2]:.0f} mm | grasp x={g[0]:.0f} y={g[1]:.0f} z={g[2]:.0f}"
                  f" | {dist:.0f} mm from the base" + ("" if dist <= MAX_REACH else f"  TOO FAR (over {MAX_REACH:.0f}); move it closer"))


if __name__ == "__main__":
    asyncio.run(main())
