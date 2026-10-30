"""Skully the skull: find him, decide whether he is holding still, feed him.

Skully is a pewter Halloween skull with a strip of blue painter's tape on the bridge of his nose. The
tape is the marker: from the home view (the arm looking down at the table) it is the only saturated
blue patch of that size. No AprilTag (none to be had at the venue, 30 Oct), so his position comes from
the tape's centre pixel plus the depth map, or, if the depth stream is down, the ray through that pixel
to the plane TAPE_Z above the table. The mouth is a fixed offset below and in front of the tape.

    .venv/bin/python skull.py              # read-only: where is Skully, is he still, where would the cube go
    .venv/bin/python skull.py --live       # the real feed, no smile needed (a hand on the red button)
    .venv/bin/python skull.py --live --probe   # only go to the pre-feed pose in front of his mouth, then greet

The guard (why the robot can refuse): FEED runs only when Skully has been seen twice, STABLE_GAP_S apart,
and moved less than STABLE_MM between the looks. Move him while the robot is looking and it says so and
goes back to greet with the cube still in hand. That refusal is logged as an episode like any other.

Geometry, all in the arm's own frame (x forward from the base, z up from the table, mm). Skully should
face the arm; the gripper comes in level, pointing away from the base, so the cube enters the mouth
from the front. Measure TAPE_Z and MOUTH_BELOW on the real skull with a ruler before trusting a live run.
"""
import argparse
import asyncio
import math
import os
import time

import cv2
import numpy as np

from camgrab import grab
from cube import cam_offset, depth_mm, pixel_to_plane, to_world

HERE = os.path.dirname(os.path.abspath(__file__))
TAPE_HSV = ((95, 150, 60), (125, 255, 255))   # the painter's tape: hue 107, sat 222 on the table photo (30 Oct); the navy banner is sat ~140
TAPE_AREA = (100, 6000)                        # blob area in pixels at 640x480, home view to a close look
TAPE_HUE, TAPE_SAT = 105, 226                  # the tape as the wrist camera sees it (home view, 30 Oct 14:00)
TAPE_MIN_Z, TAPE_MAX_Z = 55.0, 320.0           # the tape sits higher than a cube top (48, read ~40 low) and lower than a head
FALLBACK_WITHOUT_HEIGHT = True                 # no depth: guess by colour alone (weaker; the blue cube must be off the table)
TAPE_Z = 125.0        # mm: height of the tape's centre above the table when Skully stands on it (MEASURE with a ruler; depth read 84 + 40 bias, 30 Oct)
MOUTH_BELOW = 60.0    # mm: the open mouth's centre below the tape centre (MEASURE)
MOUTH_FORWARD = 10.0  # mm: the mouth is this far in front of the tape (toward the arm, since Skully faces it)
CUBE_AHEAD = 60.0     # mm from the gripper's steered point to the centre of a held cube, along the gripper axis
INSERT_MM = 15.0      # how far past the mouth centre the cube's centre goes (half a cube would be 24)
STANDOFF = 150.0      # mm: the pre-feed pose, this far back from the insertion pose
MIN_Z = 60.0          # never steer the gripper point lower than this (fingers hang below it)
MIN_DIST = 530.0      # mm from the base: a level wrist nearer than this folds the arm onto itself (29 Oct)
MAX_DIST = 800.0
STABLE_MM = 20.0      # Skully may drift this much between two looks and still count as still
STABLE_GAP_S = 1.0
STATE = os.path.join(HERE, "data", "skully.json")     # what the robot last believed about Skully, for the screen


def tape_blobs(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(TAPE_HSV[0]), np.array(TAPE_HSV[1]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
    out = []
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if TAPE_AREA[0] <= a <= TAPE_AREA[1]:
            x, y, w, h = (int(v) for v in stats[i, :4])
            if 0.4 < w / max(h, 1) < 2.5:
                out.append((a, (float(cents[i][0]), float(cents[i][1])), (x, y, w, h)))
    out.sort(reverse=True)
    return out


def publish(d):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    import json
    tmp = STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(dict(d, t=time.time()), f)
    os.replace(tmp, STATE)


async def find_skully(machine, cam, intr, tape_z=TAPE_Z):
    """Where the tape is in the arm's frame: (x, y, z, how, area) or None. how = "depth" or "plane".

    The home view is full of blue (chairs, the laptop's glare, the blue cube), so every candidate blob is
    placed in the world first and kept only if it is on the table (under 1300 mm from the base) and
    higher than a cube (TAPE_MIN_Z); of those, the biggest wins. Without depth there is no height to go
    on, so the most tape-like blob (hue and saturation nearest the measured tape) is taken, ray to TAPE_Z."""
    jpeg, depth = await grab(cam, want_depth=True)
    bgr = cv2.imdecode(np.frombuffer(jpeg, "uint8"), cv2.IMREAD_COLOR)
    h_img, w_img = bgr.shape[:2]
    bl = tape_blobs(bgr)
    if not bl:
        return None
    if depth:
        kept = []
        for area, (cu, cv_), box in bl:            # all of them: on 30 Oct the tape was the 11th-biggest blue blob
            mm = depth_mm(depth, cu, cv_, w_img, h_img)
            if not (150 < mm < 2500):
                continue
            x, y, z = await to_world(machine, intr, cu, cv_, mm)
            ox, oy = cam_offset()
            x, y = x + ox, y + oy
            if math.hypot(x, y) < 1300 and TAPE_MIN_Z < z < TAPE_MAX_Z:
                kept.append((area, x, y, z))
        if kept:
            kept.sort(reverse=True)
            area, x, y, z = kept[0]
            # x, y from depth; z from the ruler: the depth map reads about 4 cm low (29 Oct), and the mouth
            # height is what the fingers meet, so TAPE_Z measured on the skull beats a biased reading
            return x, y, tape_z, "depth", area
        if not FALLBACK_WITHOUT_HEIGHT:
            return None
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    def tapeness(b):
        a, (cu, cv_), (x, y, w, h) = b
        patch = hsv[y:y + h, x:x + w].reshape(-1, 3)
        m = patch[:, 1] > 150
        if not m.any():
            return 1e9
        med = np.median(patch[m], axis=0)
        return abs(med[0] - TAPE_HUE) * 3 + abs(med[1] - TAPE_SAT) + (0 if a < 1500 else 200)
    area, (cu, cv_), box = min(bl, key=tapeness)
    p = await pixel_to_plane(machine, intr, cu, cv_, tape_z)
    if p is None:
        return None
    return p[0], p[1], p[2], "plane", area


def feed_poses(tx, ty, tz):
    """From the tape's position: the mouth, the insertion pose, the pre-feed pose, and the level
    orientation pointing from the base toward Skully. Returns dict of (x, y, z) and o."""
    flat = math.hypot(tx, ty)
    dx, dy = tx / flat, ty / flat                        # from the base toward Skully, level
    mx, my, mz = tx - dx * MOUTH_FORWARD, ty - dy * MOUTH_FORWARD, tz - MOUTH_BELOW
    ins = (mx - dx * (CUBE_AHEAD - INSERT_MM), my - dy * (CUBE_AHEAD - INSERT_MM), max(mz, MIN_Z))
    pre = (ins[0] - dx * STANDOFF, ins[1] - dy * STANDOFF, ins[2])
    return {"mouth": (mx, my, mz), "insert": ins, "pre": pre, "o": (dx, dy, 0.0), "dist": flat}


async def skully_stable(rig, say, looks=2, gap=STABLE_GAP_S):
    """Look `looks` times, `gap` seconds apart. Returns (last position or None, moved mm, stable bool, how)."""
    seen = []
    for i in range(looks):
        p = await find_skully(rig.machine, rig.cam, rig.intr)
        if p is None:
            await say(f"   look {i + 1}: no blue tape in view")
            publish({"seen": False, "stable": False})
            return None, None, False, None
        x, y, z, how, area = p
        await say(f"   look {i + 1}: tape at x={x:.0f} y={y:.0f} z={z:.0f} ({how}, blob {area} px)")
        seen.append((x, y, z))
        if i + 1 < looks:
            await asyncio.sleep(gap)
    moved = max(math.dist(a, b) for a in seen for b in seen) if len(seen) > 1 else 0.0
    stable = moved <= STABLE_MM
    publish({"seen": True, "x": seen[-1][0], "y": seen[-1][1], "z": seen[-1][2], "moved_mm": moved, "stable": stable, "how": how})
    return seen[-1], moved, stable, how


async def feed_skully(rig, say, probe=False, episode=None):
    """The guarded feed. Returns (ok, reason). The arm must already hold the cube (caller checks).
    Steps: home (the table view); two looks; refuse if Skully moved or is out of reach; else plan to the
    pre-feed pose, slide in, open, back out, greet."""
    from pick import plan_to
    await say("feed Skully: to home, to look for him")
    if rig.live:
        await rig.home.set_position(2); await asyncio.sleep(4)
    pos, moved, stable, how = await skully_stable(rig, say)
    if pos is None:
        return False, "skully not seen"
    if not stable:
        await say(f"REFUSED: Skully moved {moved:.0f} mm between looks (limit {STABLE_MM:.0f}); he has to hold still")
        if rig.live:
            await rig.greet.set_position(2)
        return False, f"skully moving ({moved:.0f} mm)"
    g = feed_poses(*pos)
    if not (MIN_DIST <= g["dist"] <= MAX_DIST):
        await say(f"REFUSED: Skully is {g['dist']:.0f} mm from the base; the feed works between {MIN_DIST:.0f} and {MAX_DIST:.0f}")
        if rig.live:
            await rig.greet.set_position(2)
        return False, f"skully out of range ({g['dist']:.0f} mm)"
    if episode is not None:
        episode.update(skully={"x": pos[0], "y": pos[1], "z": pos[2], "moved_mm": moved, "ranged_by": how}, mouth=g["mouth"])
    m, ins, pre, o = g["mouth"], g["insert"], g["pre"], g["o"]
    await say(f"Skully still ({moved:.0f} mm). mouth x={m[0]:.0f} y={m[1]:.0f} z={m[2]:.0f}; "
              f"pre-feed x={pre[0]:.0f} y={pre[1]:.0f} z={pre[2]:.0f}; insert x={ins[0]:.0f} y={ins[1]:.0f} z={ins[2]:.0f}")
    if not rig.live:
        await say("dry run: would plan to pre-feed, slide in, open, back out, greet")
        return True, "dry run"
    ok = False
    for tilt, theta in ((0.0, 0.0), (0.0, 90.0), (-0.3, 0.0), (-0.3, 90.0)):
        ok = await plan_to(rig, *pre, f"pre-feed in front of Skully (tilt {tilt:.1f}, theta {theta:.0f})", o=(o[0], o[1], tilt), theta=theta)
        if ok:
            break
    if not ok:
        await say("could not reach the pre-feed pose; back to greet, cube kept")
        await rig.greet.set_position(2)
        return False, "pre-feed unreachable"
    if probe:
        await say("probe: at the pre-feed pose; check the height and the facing by eye. Back to greet in 5 s")
        await asyncio.sleep(5)
        await rig.greet.set_position(2)
        return True, "probe"
    # second stability check from the pre-feed pose: did he move while the arm came over?
    p2 = await find_skully(rig.machine, rig.cam, rig.intr)
    if p2 is not None and math.dist(p2[:3], pos) > STABLE_MM * 2:
        await say(f"ABORT: Skully moved {math.dist(p2[:3], pos):.0f} mm while the arm approached; backing off")
        await rig.greet.set_position(2)
        return False, "skully moved during approach"
    if not await plan_to(rig, *ins, "slide the cube into the mouth", o=(o[0], o[1], tilt), theta=theta):
        await say("insertion refused by the planner; backing out")
        await plan_to(rig, *pre, "back out", o=(o[0], o[1], tilt), theta=theta)
        await rig.greet.set_position(2)
        return False, "insertion unreachable"
    await rig.gripper.open(); await asyncio.sleep(0.6)
    await say("bon appetit, Skully")
    await plan_to(rig, *pre, "back out", o=(o[0], o[1], tilt), theta=theta)
    await rig.greet.set_position(2)
    return True, "fed"


async def main(live, probe):
    from handme import Rig, say
    from viamarm import connect
    async with await connect() as machine:
        rig = await Rig(machine, live).setup()
        if not live:
            p = await find_skully(machine, rig.cam, rig.intr)
            if p is None:
                print("no blue tape in view from where the camera is now"); return
            x, y, z, how, area = p
            g = feed_poses(x, y, z)
            print(f"tape at x={x:.0f} y={y:.0f} z={z:.0f} ({how}, {area} px), {g['dist']:.0f} mm from the base")
            for k in ("mouth", "pre", "insert"):
                print(f"  {k:7s} x={g[k][0]:.0f} y={g[k][1]:.0f} z={g[k][2]:.0f}")
            pos, moved, stable, how = await skully_stable(rig, say)
            print("still" if stable else f"moving ({moved:.0f} mm)")
            return
        ok, why = await feed_skully(rig, say, probe=probe)
        print("result:", ok, why)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--probe", action="store_true", help="live: only go to the pre-feed pose, then greet")
    a = ap.parse_args()
    asyncio.run(main(a.live, a.probe))
