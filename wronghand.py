"""Score an episode for following the wrong hand, after the fact.

Re-runs the body model on every saved frame, finds the nearest person's left and right wrists, and
checks which one the point the arm was chasing sat on. WRONG: nearer the right wrist than the left.
NOBODY: a body is seen but the point is near neither wrist. Each costs one point, like a lost frame,
and the episode's reward is re-issued with that penalty. Writes back into data/episodes.json.

    .venv/bin/python wronghand.py            # every episode not yet checked
    .venv/bin/python wronghand.py --all      # re-check all
"""
import csv, json, os, sys
from tracker import pose_detector, wrists

HERE = os.path.dirname(os.path.abspath(__file__))
SHEET = os.path.join(HERE, "data", "episodes.json")
NEAR_WRONG = 0.25   # chased point this close to the right wrist, and nearer it than the left: wrong hand
NEAR_NOBODY = 0.35  # further than this from both wrists: chasing someone or something else
# (0.15 for both flagged 85% of frames as "nobody" on 29 Oct: the palm centre sits ~0.1 from the wrist at this range)


def check(run_dir, det):
    rows = list(csv.DictReader(open(os.path.join(run_dir, "frames.csv"))))
    wrong = nobody = checked = 0
    for r in rows:
        if not r["sm_x"] or not r["frame"]:
            continue
        path = os.path.join(run_dir, r["frame"])
        if not os.path.exists(path):
            continue
        ws = wrists(det, open(path, "rb").read())
        if ws["left"] is None and ws["right"] is None:
            continue
        checked += 1
        p = (float(r["sm_x"]), float(r["sm_y"]))
        d = lambda w: float("inf") if w is None else ((p[0] - w[0]) ** 2 + (p[1] - w[1]) ** 2) ** 0.5
        dl, dr = d(ws["left"]), d(ws["right"])
        if dr < dl and dr < NEAR_WRONG:
            wrong += 1
        elif dl >= NEAR_NOBODY and dr >= NEAR_NOBODY:
            nobody += 1
    return wrong, nobody, checked, (wrong + nobody) / max(len(rows), 1)


if __name__ == "__main__":
    det = pose_detector()
    rows = json.load(open(SHEET))
    for r in rows:
        if not r.get("run") or (r.get("wrong") is not None and "--all" not in sys.argv):
            continue
        run_dir = os.path.join(HERE, "data", "follow", r["run"])
        if not os.path.exists(os.path.join(run_dir, "frames.csv")):
            continue
        wrong, nobody, checked, pen = check(run_dir, det)
        r["wrong"], r["nobody"], r["checked"] = wrong, nobody, checked
        if r.get("reward") is not None:
            r["reward_base"] = r.get("reward_base", r["reward"])
            r["reward"] = round(r["reward_base"] - pen, 3)
        print(f"{r['i']+1:2d} {r['tag']:5s} wrong-hand {wrong:3d}  nobody {nobody:3d}  of {checked:3d}  reward {r.get('reward_base')} -> {r['reward']}", flush=True)
        json.dump(rows, open(SHEET, "w"), indent=1)
