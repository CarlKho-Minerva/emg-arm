"""Fit the arm model (speed, acceleration, delay) from what the follower logged, not from a guess.

Each logged row that moved holds the target sent (joints column = measured + step), so the measured
waist angle at that moment is target - step. Replaying the sent targets through the simulator's arm
model and comparing with the measured trace, the (speed, accel, delay) with the smallest error wins.

    .venv/bin/python fit_arm.py            # over every live run
"""
import csv, glob, itertools, math, os
import sim

def load(run_dir):
    rows = list(csv.DictReader(open(os.path.join(run_dir, "frames.csv"))))
    out = []
    for r in rows:
        if r["joints"] and r["step_x"]:
            tgt = float(r["joints"].split()[0]); meas = tgt - float(r["step_x"])
            out.append((float(r["t"]), tgt, meas))
    return out

def replay(trace, speed, accel, delay_s):
    joint = trace[0][2]; vel = 0.0; err = 0.0; n = 0
    targets = [t[1] for t in trace]; times = [t[0] for t in trace]
    for i in range(1, len(trace)):
        dt = max(times[i] - times[i-1], 1e-3)
        # the target the arm is acting on is the one sent delay_s ago
        k = i
        while k > 0 and times[i] - times[k] < delay_s: k -= 1
        want_v = max(-speed, min(speed, (targets[k] - joint) / dt))
        dv = max(-accel * dt, min(accel * dt, want_v - vel))
        vel += dv; joint += vel * dt
        err += (joint - trace[i][2]) ** 2; n += 1
    return math.sqrt(err / max(n, 1))

if __name__ == "__main__":
    runs = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/follow/*-live")))
    traces = [t for t in (load(r) for r in runs) if len(t) > 50]
    print(f"{len(traces)} runs with enough moves")
    best = []
    for sp, ac, dl in itertools.product((20, 40, 60, 80, 120), (60, 120, 200, 350, 500), (0.0, 0.1, 0.2, 0.3)):
        e = sum(replay(t, sp, ac, dl) for t in traces) / len(traces)
        best.append((e, sp, ac, dl))
    best.sort()
    print("rms deg  speed accel delay")
    for e, sp, ac, dl in best[:8]:
        print(f"{e:6.2f}   {sp:4d}  {ac:4d}  {dl:.1f}")
