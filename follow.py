"""Make the wrist camera keep your hand in the middle of its picture.

The oldest trick there is: measure how far the hand is from the centre of the frame, turn a
little toward it, look again. Two joints do the turning: joint 0 (the waist, left/right) and
joint 4 (the wrist nod, up/down). Nothing here reaches for the hand; the arm only turns to
look at it.

    .venv/bin/python follow.py            # dry run: prints what it would do, moves nothing
    .venv/bin/python follow.py --live     # moves, small steps, capped; you hold the E-stop
    .venv/bin/python follow.py --live --flip-x --flip-y   # if it turns the wrong way
    .venv/bin/python follow.py --live --kp 8 --kd 0.5      # tune: lower kp if it wobbles, add kd to damp
    .venv/bin/python follow.py --live --arm left     # track the LEFT ARM by the body; keeps going when the hand leaves the frame

Each step is GAIN degrees per full-frame of error, never more than STEP_CAP degrees, and it
waits until the hand has been seen in two frames in a row before moving at all.
"""
import argparse
import asyncio
import csv
import math
import os
import signal
import struct
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from viam.components.camera import Camera
from viam.components.switch import Switch
from viam.proto.component.arm import JointPositions

from camgrab import grab
from tracker import arm_target, pose_detector
from viamarm import arm_of, connect

MODEL = os.path.expanduser("~/.cache/hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
GAIN = 8.0         # Kp: degrees of turn per full frame-width of error
KD = 0.3           # Kd: degrees per (frame-width per second) of error change; damps overshoot
STEP_CAP = 6.0     # never more than this per step, degrees (streamed, so a step is "head this way", not a stop)
DEADBAND = 0.10    # fraction of frame: close enough, stop fiddling
SMOOTH = 0.5       # 0..1: share of the new reading that gets through each frame (lower = smoother, laggier)
ASMOOTH = 0.5      # same filter on the step the arm is told to take (the "low-pass on the action")
COAST = 0.5        # s: after losing the hand, keep moving along its last direction this long
LOOKAHEAD = 0.15   # s: aim where the hand will be, from its speed over the last few frames (state = position + velocity)
REC_EVERY = 0.0    # s between saved pictures; 0 = every frame (about 2 MB/s, stitch with stitch.sh)
REC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "follow")
# Measured 2026-10-29 at the greet pose, +5 deg on each joint (picture shift in px, 1280x720):
#   j0 waist      x+92  y-11   pan, clean
#   j1 shoulder   x+12  y+157  tilt
#   j2 elbow      x+2   y-144  tilt, clean
#   j3 forearm    roll only (-4 deg)
#   j4 wrist nod  x-29  y-69   mixed
#   j5 wrist spin x-42  y+59   mixed, with roll
# So: pan with the waist, tilt with the elbow. Signs: waist + moves the picture right (camera looks
# left), elbow + moves the picture up (camera looks down). -1 for jy = never tilt.
JOINT_X, JOINT_Y = 0, 2
SEARCH_AFTER = 1.0        # s without a hand before the camera starts sweeping to look for it
SEARCH_STEP = 0.5         # degrees per frame while sweeping (streamed at ~10 frames/s = 5 deg/s, slow on purpose)
SEARCH_LIMIT = 35.0
LOST_GO_AFTER = 10.0
# Arm speed and acceleration while following. Measured 2026-10-29 streaming 6-degree targets every
# 100 ms on joint 4: default 60 deg/s + accel 500 stalled on 8 of 15 samples (stutter); 40 + 120 had
# no stalls, 4 to 10 deg/s continuous. The driver's defaults are put back on exit.
ARM_SPEED, ARM_ACCEL = 40.0, 120.0
ARM_SPEED_DEFAULT, ARM_ACCEL_DEFAULT = 60.0, 500.0      # s of fruitless search before returning to the --lost-go pose       # degrees either side of where the sweep started


def detector():
    if not os.path.exists(MODEL):
        os.makedirs(os.path.dirname(MODEL), exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL)
    opts = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=MODEL), num_hands=2)
    return vision.HandLandmarker.create_from_options(opts)


def depth_at(depth_bytes, u, v):
    """Millimetres at a pixel of Viam's depth map (8-byte magic, two uint64 sizes, uint16 big-endian)."""
    if not depth_bytes:
        return None
    w, h = struct.unpack(">QQ", depth_bytes[8:24])
    x, y = int(u * w), int(v * h)
    off = 24 + 2 * (y * w + x)
    return struct.unpack(">H", depth_bytes[off:off + 2])[0]


def watch_score(bgr, pts):
    """How watch-like the wrist looks: share of dark pixels in a patch just past the wrist landmark.
    An Apple Watch face is a black rectangle on the back of the wrist; bare skin is not dark."""
    h, w = bgr.shape[:2]
    wx, wy = pts[0].x * w, pts[0].y * h
    mx, my = pts[9].x * w, pts[9].y * h          # middle knuckle: the wrist-to-knuckle line gives hand size
    size = max(20.0, math.hypot(mx - wx, my - wy))
    # step back from the wrist, away from the fingers, by a third of a hand
    cx, cy = wx - (mx - wx) * 0.35, wy - (my - wy) * 0.35
    r = int(size * 0.3)
    x0, x1 = max(0, int(cx - r)), min(w, int(cx + r)); y0, y1 = max(0, int(cy - r)), min(h, int(cy + r))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    patch = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    return float((patch < 60).mean())


def hand_centre(det, jpeg, near=None, want=None):
    """Palm centre (0..1, 0..1) of the hand to follow, or None.

    near: last known centre; of several hands, the closest to it wins (sticky tracking).
    want: "left" or "right" to keep only hands the detector labels so. The label can come
    out swapped on a camera that is not mirrored; test before trusting it.
    """
    bgr = cv2.imdecode(np.frombuffer(jpeg, "uint8"), cv2.IMREAD_COLOR)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    res = det.detect(img)
    cands = []
    for pts, hd in zip(res.hand_landmarks, res.handedness):
        if want and want != "watch" and hd[0].category_name.lower() != want:
            continue
        palm = [pts[i] for i in (0, 5, 9, 13, 17)]       # wrist + knuckles
        cands.append(((sum(p.x for p in palm) / 5, sum(p.y for p in palm) / 5), watch_score(bgr, pts)))
    if not cands:
        return None
    if near is None and want == "watch" and len(cands) > 1:
        cands.sort(key=lambda c: -c[1])                  # first lock: the darkest wrist
        return cands[0][0]
    if near is None:
        return cands[0][0]
    return min((c[0] for c in cands), key=lambda c: (c[0] - near[0]) ** 2 + (c[1] - near[1]) ** 2)


async def main(live, flip_x, flip_y, kp=GAIN, kd=KD, jx=JOINT_X, jy=JOINT_Y, want=None, smooth=SMOOTH, asmooth=ASMOOTH, arm_side=None, lost_go=None, speed=ARM_SPEED, accel=ARM_ACCEL,
               look=LOOKAHEAD, tag="", seconds=0.0):
    hist = []         # the last few smoothed hand positions with their times: the state
    adx = ady = 0.0   # the low-passed step
    det = detector()
    pose = pose_detector() if arm_side else None
    prev = None   # (ex, ey, time) of the last frame that saw a hand
    sm = None     # smoothed hand centre
    # every run is recorded: one CSV row per frame, a picture every 0.5 s, for learning from later
    run = time.strftime("%Y%m%d-%H%M%S") + ("-live" if live else "-dry") + (f"-{tag}" if tag else "")
    rec_dir = os.path.join(REC_DIR, run)
    os.makedirs(rec_dir, exist_ok=True)
    log_f = open(os.path.join(rec_dir, "frames.csv"), "w", newline="", buffering=1)   # line-buffered: survives a kill
    log = csv.writer(log_f)
    log.writerow(["t", "raw_x", "raw_y", "sm_x", "sm_y", "err_x", "err_y", "step_x", "step_y", "moved", "joints", "frame"])
    last_pic = 0.0
    print("recording to", rec_dir)
    lost_since = None
    search_dir = 1.0
    search_origin = None
    pending = None   # the move in flight; a new one cancels it, so the arm never stops to wait

    async def stream(joints):
        nonlocal pending
        if pending and not pending.done():
            pending.cancel()
        pending = asyncio.create_task(arm.move_to_joint_positions(JointPositions(values=joints)))
    async with await connect() as machine:
        cam = Camera.from_robot(machine, "cam")
        arm = arm_of(machine)
        home_switch = Switch.from_robot(machine, lost_go) if lost_go else None
        if live:
            await arm.do_command({"set_speed": float(speed), "set_acceleration": float(accel)})
            print(f"arm speed {speed:.0f} deg/s, acceleration {accel:.0f}")
        went_home = False
        seen = 0
        stopping = asyncio.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_running_loop().add_signal_handler(sig, stopping.set)
        try:
            t_start = time.time()
            while not stopping.is_set() and (seconds <= 0 or time.time() - t_start < seconds):
                t0 = time.time()
                jpeg, _ = await grab(cam, want_depth=False)
                how = "hand"
                # the hand detector leads (it does not hallucinate); the body model only (a) picks which hand
                # is the chosen arm at first lock and (b) points past the elbow when no hand is in view
                body = arm_target(pose, jpeg, arm_side) if pose else None
                near = sm if sm is not None else ((body[0], body[1]) if body else None)
                raw = hand_centre(det, jpeg, near=near, want=None if body else want)
                if raw is None and body and body[2] == "elbow+":
                    raw, how = (body[0], body[1]), "elbow+"
                pic = ""
                if time.time() - last_pic > REC_EVERY:
                    pic = f"{time.time():.3f}.jpg"
                    open(os.path.join(rec_dir, pic), "wb").write(jpeg)
                    last_pic = time.time()
                if raw is None and prev is not None and time.time() - prev[2] < COAST and live:
                    # coast: the hand was just there and heading somewhere; keep heading that way
                    j = list((await arm.get_joint_positions()).values)
                    j[jx] += adx * 0.7
                    if jy >= 0: j[jy] += ady * 0.7
                    await stream(j)
                    adx *= 0.85; ady *= 0.85
                    log.writerow([f"{time.time():.3f}", "", "", "", "", "", "", f"{adx:.2f}", f"{ady:.2f}", 1, "", pic])
                    print(f"{time.strftime('%H:%M:%S')} no hand  coasting j{jx} {adx:+.1f}")
                    continue
                if raw is None:
                    seen = 0
                    prev = None
                    sm = None
                    lost_since = lost_since or time.time()
                    note = ""
                    if live and home_switch and not went_home and time.time() - lost_since > SEARCH_AFTER + LOST_GO_AFTER:
                        if pending and not pending.done():
                            pending.cancel()
                        await home_switch.set_position(2)     # the block's "go to"
                        went_home = True
                        search_origin = None
                        print(f"{time.strftime('%H:%M:%S')} nobody for {LOST_GO_AFTER:.0f}s: back to {lost_go}")
                        continue
                    if live and time.time() - lost_since > SEARCH_AFTER:
                        # sweep: turn the pan joint toward where the hand was last, bounce at the limits
                        j = list((await arm.get_joint_positions()).values)
                        if search_origin is None:
                            search_origin = j[jx]
                        if abs(j[jx] + search_dir * SEARCH_STEP - search_origin) > SEARCH_LIMIT:
                            search_dir = -search_dir
                        j[jx] += search_dir * SEARCH_STEP
                        await stream(j)
                        note = f"  searching j{jx} {search_dir * SEARCH_STEP:+.0f}"
                    log.writerow([f"{time.time():.3f}", "", "", "", "", "", "", "", "", 0, "", pic])
                    print(f"{time.strftime('%H:%M:%S')} no hand{note}")
                    continue
                lost_since = None
                search_origin = None
                went_home = False
                seen += 1
                # the smoother: blend the new reading into the old one, so one twitchy frame cannot move the arm
                sm = raw if sm is None else (sm[0] + smooth * (raw[0] - sm[0]), sm[1] + smooth * (raw[1] - sm[1]))
                c = sm
                ex, ey = c[0] - 0.5, c[1] - 0.5          # + means hand is right / below centre
                now = time.time()
                vx = vy = 0.0
                if prev:
                    dt = max(now - prev[2], 1e-3)
                    vx, vy = (ex - prev[0]) / dt, (ey - prev[1]) / dt   # how fast the error is changing
                prev = (ex, ey, now)
                ux = kp * ex + kd * vx                   # P + D, in degrees
                uy = kp * ey + kd * vy
                dx = max(-STEP_CAP, min(STEP_CAP, -ux * (-1 if flip_x else 1)))
                dy = max(-STEP_CAP, min(STEP_CAP, uy * (-1 if flip_y else 1)))
                if abs(ex) < DEADBAND: dx = 0.0
                if abs(ey) < DEADBAND or jy < 0: dy = 0.0
                # low-pass on the action: the commanded step eases toward the computed one
                adx += asmooth * (dx - adx); ady += asmooth * (dy - ady)
                dx, dy = (adx if abs(adx) > 0.2 else 0.0), (ady if abs(ady) > 0.2 else 0.0)
                # remember which way the hand went, so a search starts in that direction
                if dx: search_dir = 1.0 if dx > 0 else -1.0
                line = (f"{time.strftime('%H:%M:%S')} hand at ({c[0]:.2f},{c[1]:.2f}) "
                        f"err ({ex:+.2f},{ey:+.2f}) by {how}; step j{jx} {dx:+.1f} j{jy} {dy:+.1f}")
                moved = 0
                j = []
                if live and seen >= 2 and (dx or dy):
                    j = list((await arm.get_joint_positions()).values)
                    j[jx] += dx
                    if jy >= 0:
                        j[jy] += dy
                    await stream(j)
                    line += "  MOVED"
                    moved = 1
                log.writerow([f"{t0:.3f}", f"{raw[0]:.4f}", f"{raw[1]:.4f}", f"{c[0]:.4f}", f"{c[1]:.4f}",
                              f"{ex:.4f}", f"{ey:.4f}", f"{dx:.2f}", f"{dy:.2f}", moved,
                              " ".join(f"{v:.2f}" for v in j), pic])
                print(line, f"({time.time() - t0:.2f}s)")

        finally:
            if live:
                if pending and not pending.done():
                    pending.cancel()
                await arm.do_command({"set_speed": ARM_SPEED_DEFAULT, "set_acceleration": ARM_ACCEL_DEFAULT})
                print("arm speed and acceleration restored to defaults")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--flip-x", action="store_true")
    ap.add_argument("--flip-y", action="store_true")
    ap.add_argument("--kp", type=float, default=GAIN)
    ap.add_argument("--kd", type=float, default=KD)
    ap.add_argument("--jx", type=int, default=JOINT_X, help="joint for left/right (0 waist, 3 forearm twist)")
    ap.add_argument("--jy", type=int, default=JOINT_Y, help="joint for up/down (4 wrist nod)")
    ap.add_argument("--hand", choices=["left", "right", "watch"], help="left/right by label, or watch = the wrist with a dark face")
    ap.add_argument("--smooth", type=float, default=SMOOTH)
    ap.add_argument("--asmooth", type=float, default=ASMOOTH, help="low-pass on the step (lower = smoother)")
    ap.add_argument("--arm", choices=["left", "right"], help="track this arm by the body (shoulder/elbow/wrist); beats --hand")
    ap.add_argument("--lost-go", metavar="BLOCK", help="after 10 s of fruitless search, press this pose block's go-to (e.g. greet)")
    ap.add_argument("--speed", type=float, default=ARM_SPEED, help="arm joint speed deg/s while following")
    ap.add_argument("--accel", type=float, default=ARM_ACCEL, help="arm joint acceleration deg/s^2 while following")
    ap.add_argument("--look", type=float, default=LOOKAHEAD, help="s to aim ahead of the hand from its velocity; 0 = off")
    ap.add_argument("--tag", default="", help="episode label, goes in the recording folder name (see EPISODES.md)")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after this many seconds (0 = until Ctrl-C)")
    a = ap.parse_args()
    asyncio.run(main(a.live, a.flip_x, a.flip_y, a.kp, a.kd, a.jx, a.jy, a.hand, a.smooth, a.asmooth, a.arm, a.lost_go,
                     a.speed, a.accel, a.look, a.tag, a.seconds))
