"""Make the wrist camera keep your hand in the middle of its picture.

The oldest trick there is: measure how far the hand is from the centre of the frame, turn a
little toward it, look again. Two joints do the turning: joint 0 (the waist, left/right) and
joint 4 (the wrist nod, up/down). Nothing here reaches for the hand; the arm only turns to
look at it.

    .venv/bin/python follow.py            # dry run: prints what it would do, moves nothing
    .venv/bin/python follow.py --live     # moves, small steps, capped; you hold the E-stop
    .venv/bin/python follow.py --live --flip-x --flip-y   # if it turns the wrong way

Each step is GAIN degrees per full-frame of error, never more than STEP_CAP degrees, and it
waits until the hand has been seen in two frames in a row before moving at all.
"""
import argparse
import asyncio
import os
import struct
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from viam.components.camera import Camera
from viam.proto.component.arm import JointPositions

from camgrab import grab
from viamarm import arm_of, connect

MODEL = os.path.expanduser("~/.cache/hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
GAIN = 12.0        # degrees of turn per full frame-width of error
STEP_CAP = 4.0     # never more than this per step, degrees
DEADBAND = 0.06    # fraction of frame: close enough, stop fiddling
JOINT_X, JOINT_Y = 0, 4


def detector():
    if not os.path.exists(MODEL):
        os.makedirs(os.path.dirname(MODEL), exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL)
    opts = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=MODEL), num_hands=1)
    return vision.HandLandmarker.create_from_options(opts)


def depth_at(depth_bytes, u, v):
    """Millimetres at a pixel of Viam's depth map (8-byte magic, two uint64 sizes, uint16 big-endian)."""
    if not depth_bytes:
        return None
    w, h = struct.unpack(">QQ", depth_bytes[8:24])
    x, y = int(u * w), int(v * h)
    off = 24 + 2 * (y * w + x)
    return struct.unpack(">H", depth_bytes[off:off + 2])[0]


def hand_centre(det, jpeg):
    bgr = cv2.imdecode(np.frombuffer(jpeg, "uint8"), cv2.IMREAD_COLOR)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    res = det.detect(img)
    if not res.hand_landmarks:
        return None
    pts = res.hand_landmarks[0]
    palm = [pts[i] for i in (0, 5, 9, 13, 17)]       # wrist + knuckles
    return sum(p.x for p in palm) / 5, sum(p.y for p in palm) / 5


async def main(live, flip_x, flip_y):
    det = detector()
    async with await connect() as machine:
        cam = Camera.from_robot(machine, "cam")
        arm = arm_of(machine)
        seen = 0
        while True:
            t0 = time.time()
            jpeg, depth = await grab(cam)
            c = hand_centre(det, jpeg)
            if c is None:
                seen = 0
                print(f"{time.strftime('%H:%M:%S')} no hand")
                continue
            seen += 1
            ex, ey = c[0] - 0.5, c[1] - 0.5          # + means hand is right / below centre
            mm = depth_at(depth, *c)
            dx = max(-STEP_CAP, min(STEP_CAP, -GAIN * ex * (-1 if flip_x else 1)))
            dy = max(-STEP_CAP, min(STEP_CAP, GAIN * ey * (-1 if flip_y else 1)))
            if abs(ex) < DEADBAND: dx = 0.0
            if abs(ey) < DEADBAND: dy = 0.0
            line = (f"{time.strftime('%H:%M:%S')} hand at ({c[0]:.2f},{c[1]:.2f}) "
                    f"{mm or '?'} mm away; step j{JOINT_X} {dx:+.1f} j{JOINT_Y} {dy:+.1f}")
            if live and seen >= 2 and (dx or dy):
                j = list((await arm.get_joint_positions()).values)
                j[JOINT_X] += dx
                j[JOINT_Y] += dy
                await arm.move_to_joint_positions(JointPositions(values=j))
                line += "  MOVED"
            print(line, f"({time.time() - t0:.2f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--flip-x", action="store_true")
    ap.add_argument("--flip-y", action="store_true")
    a = ap.parse_args()
    asyncio.run(main(a.live, a.flip_x, a.flip_y))
