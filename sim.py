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
  reward   per frame: minus error squared, minus 1 per lost frame, minus 0.02 per degree of jerk;
           the one number to raise (0 = hand dead centre, arm still)

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

DEG_PER_WIDTH = 47.0        # degrees of waist turn across one picture width (5 deg = 135 px of 1280 at the
                            # eye-level greet pose of 29 Oct 15:45; it was 70 at the earlier, lower pose)
DT = 0.05                   # s per simulated frame (the live loop ran at about 20 Hz)
DELAY_S = 0.20              # camera + network + detection + arm reaction; fitted from the logs by fit_arm.py (29 Oct: 0.2 s)
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


def simulate(bearing, kp, kd, smooth, asmooth, speed=40.0, accel=120.0, delay_s=DELAY_S, lookahead=0.0):
    """Run the controller against the bearing trace. Returns (per-frame error in picture widths,
    per-frame commanded step). lookahead > 0 aims at where the hand will be that many seconds on,
    from its velocity over the last few observations (the state is position + velocity)."""
    hist = []           # the last few observed hand bearings, (time, bearing)
    delay = max(1, int(round(delay_s / DT)))
    joint = next((b for b in bearing if b is not None), 0.0)
    vel = 0.0
    sm = None
    adx = 0.0
    prev_e = None
    errs = []
    steps = []
    for k, _ in enumerate(bearing):
        true_b = bearing[k]
        obs = bearing[k - delay] if k >= delay else None
        if true_b is None:
            errs.append(None)
            steps.append(0.0)
            continue
        err_now = (joint - true_b) / DEG_PER_WIDTH        # what the camera shows this frame
        errs.append(err_now)
        if obs is None:
            prev_e = None
            hist = []
            steps.append(0.0)
            continue
        hist.append((k * DT, obs))
        hist = hist[-5:]
        aim = obs
        if lookahead > 0 and len(hist) >= 3:
            v = (hist[-1][1] - hist[0][1]) / max(hist[-1][0] - hist[0][0], DT)   # deg/s over the last few
            aim = obs + v * (lookahead + delay_s)
        # the controller sees the delayed picture: error as it was `delay` frames ago, with the joint then
        e = (joint - aim) / DEG_PER_WIDTH
        sm = e if sm is None else sm + smooth * (e - sm)
        d = 0.0 if prev_e is None else (sm - prev_e) / DT
        prev_e = sm
        u = kp * sm + kd * d
        dx = max(-STEP_CAP, min(STEP_CAP, -u))
        if abs(sm) < DEADBAND:
            dx = 0.0
        adx += asmooth * (dx - adx)
        steps.append(adx)
        target = joint + (adx if abs(adx) > 0.2 else 0.0)
        # arm model: head for target under speed and acceleration limits
        want_v = max(-speed, min(speed, (target - joint) / DT))
        dv = max(-accel * DT, min(accel * DT, want_v - vel))
        vel += dv
        joint += vel * DT
    return errs, steps


LOST_PENALTY = 1.0    # per frame the hand is outside the picture
JERK_PENALTY = 0.02   # per degree of change in the commanded step between frames


def reward(errs, steps):
    """One number per frame, summed: minus the error squared, minus a penalty for a lost frame,
    minus a penalty for jerk (how much the commanded step changed since the last frame).
    Higher is better; 0 is a hand held dead centre with the arm still."""
    total = 0.0
    prev = 0.0
    for e, u in zip(errs, steps):
        if e is None:
            continue
        total -= e * e
        if abs(e) > 0.5:
            total -= LOST_PENALTY
        total -= JERK_PENALTY * abs(u - prev)
        prev = u
    n = sum(1 for e in errs if e is not None)
    return total / max(n, 1)


def score(errs, steps=None):
    e = [x for x in errs if x is not None]
    if not e:
        return None
    rms = math.sqrt(sum(x * x for x in e) / len(e))
    lost = sum(1 for x in e if abs(x) > 0.5) / len(e)
    over = sum(1 for a, b in zip(e, e[1:]) if a * b < 0 and abs(a) > 0.08)
    r = reward(errs, steps) if steps is not None else float("nan")
    return rms, over, lost, r


def main(runs, grid, speed=60.0, accel=200.0):
    traces = [resample(load_bearing(r)) for r in runs]
    traces = [t for t in traces if sum(1 for b in t if b is not None) > 40]
    print(f"{len(traces)} runs with enough hand frames; arm model speed {speed:.0f} accel {accel:.0f} delay {DELAY_S:.2f}s")
    results = []
    for kp, kd, smooth, asmooth, look in grid:
        agg = [0.0, 0.0, 0.0, 0.0]
        for tr in traces:
            errs, steps = simulate(tr, kp, kd, smooth, asmooth, speed=speed, accel=accel, lookahead=look)
            s = score(errs, steps)
            if s:
                for i in range(4): agg[i] += s[i]
        n = len(traces)
        results.append((-agg[3] / n, agg[0] / n, agg[1] / n, agg[2] / n, kp, kd, smooth, asmooth, look))
    results.sort()
    print("reward  rms   over  lost  kp   kd   smooth asmooth look")
    for r in results[:12]:
        print(f"{-r[0]:+.3f} {r[1]:.3f} {r[2]:4.1f}  {r[3]:.2f}  {r[4]:4.1f} {r[5]:4.2f} {r[6]:5.2f}  {r[7]:5.2f}  {r[8]:.2f}")
    print("...")
    for r in results[-2:]:
        print(f"{-r[0]:+.3f} {r[1]:.3f} {r[2]:4.1f}  {r[3]:.2f}  {r[4]:4.1f} {r[5]:4.2f} {r[6]:5.2f}  {r[7]:5.2f}  {r[8]:.2f}   (worst)")
    cur = next((r for r in results if r[4:] == (8.0, 0.3, 0.4, 0.4, 0.0)), None)
    if cur:
        print(f"current live setting (kp 8, kd 0.3, smooth 0.4, asmooth 0.4, no lookahead): reward {-cur[0]:+.3f} rms {cur[1]:.3f} lost {cur[3]:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    a = ap.parse_args()
    runs = [a.run] if a.run else sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/follow/*-live")))
    grid = list(itertools.product((4.0, 8.0, 12.0), (0.0, 0.3), (0.4, 0.7), (0.4, 0.7), (0.0, 0.15, 0.3, 0.5)))
    main(runs, grid)
