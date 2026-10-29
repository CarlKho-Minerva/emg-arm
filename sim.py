"""Replay recorded runs against a model of the arm and score controller settings offline.

The recordings in data/follow/<run>/frames.csv hold, per frame, where the hand was in the picture
and what the waist joint was. From those two the hand's true bearing (the direction the camera
would have to face to centre it) is recovered: bearing = joint + pixel_offset * degrees_per_pixel.
The joint map measured on 2026-10-29 gives 5 degrees = 92 px on the waist, so 0.054 deg/px, or
about 70 degrees across the whole 1280 px picture.

A controller is then run against that bearing trace, with a simple arm model (speed limit,
acceleration limit, a fixed delay for camera + network), and scored:
  rms      average error in picture widths (lower is better)
  overshoot number of times the error crossed zero coming from wide
  lost     share of frames the hand would have been outside the picture (error > 0.5 width)

    .venv/bin/python sim.py                # grid-search kp, kd, smooth, asmooth over every live run
    .venv/bin/python sim.py --run <dir>    # one run

No ROS, no physics engine: the arm here is a speed- and acceleration-limited integrator with a
delay, which is what the bench measured it to be for this purpose.
"""
import argparse
import csv
import glob
import itertools
import math
import os

DEG_PER_WIDTH = 70.0        # degrees of waist turn across one picture width (5 deg = 92 px of 1280)
DT = 0.05                   # s per simulated frame (the live loop ran at about 20 Hz)
DELAY_S = 0.15              # camera + network + detection before the controller sees a frame
STEP_CAP = 6.0
DEADBAND = 0.10


def load_bearing(run_dir):
    """Hand bearing in degrees over time, from the recording. Frames without a hand are None."""
    rows = list(csv.DictReader(open(os.path.join(run_dir, "frames.csv"))))
    out = []
    joint = None
    for r in rows:
        if r["joints"]:
            joint = float(r["joints"].split()[0])
        if r["raw_x"] and joint is not None:
            err_w = float(r["raw_x"]) - 0.5                    # + = hand right of centre
            out.append((float(r["t"]), joint - err_w * DEG_PER_WIDTH))   # waist + turns the picture right
        else:
            out.append((float(r["t"]), None))
    return out


def resample(trace, dt=DT):
    """Even time steps; a missing hand stays missing."""
    if not trace:
        return []
    t0, t1 = trace[0][0], trace[-1][0]
    out, i = [], 0
    t = t0
    while t <= t1:
        while i + 1 < len(trace) and trace[i + 1][0] <= t:
            i += 1
        out.append(trace[i][1])
        t += dt
    return out


def simulate(bearing, kp, kd, smooth, asmooth, speed=40.0, accel=120.0, delay_s=DELAY_S):
    """Run the controller against the bearing trace. Returns per-frame error in picture widths."""
    delay = max(1, int(round(delay_s / DT)))
    joint = next((b for b in bearing if b is not None), 0.0)
    vel = 0.0
    sm = None
    adx = 0.0
    prev_e = None
    errs = []
    seen = []
    for k, _ in enumerate(bearing):
        true_b = bearing[k]
        obs = bearing[k - delay] if k >= delay else None
        if true_b is None:
            errs.append(None)
            continue
        err_now = (joint - true_b) / DEG_PER_WIDTH        # what the camera shows this frame
        errs.append(err_now)
        if obs is None:
            prev_e = None
            continue
        # the controller sees the delayed picture: error as it was `delay` frames ago, with the joint then
        e = (joint - obs) / DEG_PER_WIDTH
        sm = e if sm is None else sm + smooth * (e - sm)
        d = 0.0 if prev_e is None else (sm - prev_e) / DT
        prev_e = sm
        u = kp * sm + kd * d
        dx = max(-STEP_CAP, min(STEP_CAP, -u))
        if abs(sm) < DEADBAND:
            dx = 0.0
        adx += asmooth * (dx - adx)
        target = joint + (adx if abs(adx) > 0.2 else 0.0)
        # arm model: head for target under speed and acceleration limits
        want_v = max(-speed, min(speed, (target - joint) / DT))
        dv = max(-accel * DT, min(accel * DT, want_v - vel))
        vel += dv
        joint += vel * DT
    return errs


def score(errs):
    e = [x for x in errs if x is not None]
    if not e:
        return None
    rms = math.sqrt(sum(x * x for x in e) / len(e))
    lost = sum(1 for x in e if abs(x) > 0.5) / len(e)
    over = sum(1 for a, b in zip(e, e[1:]) if a * b < 0 and abs(a) > 0.08)
    return rms, over, lost


def main(runs, grid):
    traces = [resample(load_bearing(r)) for r in runs]
    traces = [t for t in traces if sum(1 for b in t if b is not None) > 40]
    print(f"{len(traces)} runs with enough hand frames")
    results = []
    for kp, kd, smooth, asmooth in grid:
        tot_rms = tot_over = tot_lost = 0.0
        for tr in traces:
            s = score(simulate(tr, kp, kd, smooth, asmooth))
            if s:
                tot_rms += s[0]; tot_over += s[1]; tot_lost += s[2]
        n = len(traces)
        results.append((tot_rms / n, tot_over / n, tot_lost / n, kp, kd, smooth, asmooth))
    results.sort()
    print("rms   over  lost  kp   kd   smooth asmooth")
    for r in results[:12]:
        print(f"{r[0]:.3f} {r[1]:4.1f}  {r[2]:.2f}  {r[3]:4.1f} {r[4]:4.2f} {r[5]:5.2f}  {r[6]:5.2f}")
    print("...")
    for r in results[-3:]:
        print(f"{r[0]:.3f} {r[1]:4.1f}  {r[2]:.2f}  {r[3]:4.1f} {r[4]:4.2f} {r[5]:5.2f}  {r[6]:5.2f}   (worst)")
    cur = next(r for r in results if (r[3], r[4], r[5], r[6]) == (8.0, 0.3, 0.4, 0.4))
    print(f"current live setting (kp 8, kd 0.3, smooth 0.4, asmooth 0.4): rms {cur[0]:.3f} over {cur[1]:.1f} lost {cur[2]:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    a = ap.parse_args()
    runs = [a.run] if a.run else sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/follow/*-live")))
    grid = list(itertools.product((4.0, 8.0, 12.0, 16.0), (0.0, 0.3, 0.6), (0.2, 0.4, 0.7), (0.2, 0.4, 0.7)))
    main(runs, grid)
