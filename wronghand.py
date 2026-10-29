"""Score an episode for following the wrong hand, after the fact.

Re-runs the body model on every saved frame, finds the nearest person's left and right wrists,
and checks which one the point the arm was chasing sat on. A frame counts as WRONG when the
chased point is within 0.12 of the right wrist and not within 0.12 of the left; as NOBODY when a
body is seen but the point is near neither wrist. Each costs one point, like a lost frame, and the
episode's reward is re-issued with that penalty in. Writes the counts back into data/episodes.json.

    .venv/bin/python wronghand.py            # every scored episode not yet checked
    .venv/bin/python wronghand.py --all      # re-check all
"""
import csv, json, os, sys
import sim
from tracker import pose_detector, wrists

HERE = os.path.dirname(os.path.abspath(__file__))
SHEET = os.path.join(HERE, "data", "episodes.json")
NEAR = 0.12


def check(run_dir, det):
    rows = list(csv.DictReader(open(os.path.join(run_dir, "frames.csv"))))
    wrong = nobody = checked = 0
    penalty = 0.0
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
        if dr < NEAR and not dl < NEAR:
            wrong += 1; penalty += 1.0
        elif dl >= NEAR and dr >= NEAR:
            nobody += 1; penalty += 1.0
    return wrong, nobody, checked, penalty / max(len(rows), 1)


if __name__ == "__main__":
    det = pose_detector()
    rows = json.load(open(SHEET))
    for r in rows:
        if r.get("run") is None or (r.get("wrong") is not None and "--all" not in sys.argv):
            continue
        run_dir = os.path.join(HERE, "data", "follow", r["run"])
        wrong, nobody, checked, pen = check(run_dir, det)
        r["wrong"], r["nobody"], r["checked"] = wrong, nobody, checked
        if r.get("reward") is not None:
            r["reward_base"] = r.get("reward_base", r["reward"])
            r["reward"] = round(r["reward_base"] - pen, 3)
        print(f"{r['i']+1:2d} {r['tag']:5s} wrong-hand {wrong:3d}  nobody {nobody:3d}  of {checked:3d} checked  reward {r.get('reward_base')} -> {r['reward']}")
    json.dump(rows, open(SHEET, "w"), indent=1)
