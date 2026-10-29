"""Cut every hand out of every recorded frame and label it LEFT, RIGHT or OTHER.

LEFT and RIGHT mean Carl's own hands, told apart by the body model: a hand whose wrist sits
within 0.15 of the nearest body's left wrist (and nearer it than the right one) is LEFT, and
the mirror of that is RIGHT. OTHER is a hand with no body wrist nearby, or one too small to be
at arm's length (someone behind him). The watch score from follow.py is recorded as a feature.

    .venv/bin/python build_hands.py              # the 30 episodes in data/episodes.json
    .venv/bin/python build_hands.py --all-runs   # every folder under data/follow
    .venv/bin/python build_hands.py --relabel    # apply the label rule again from index.csv, no detection

Writes data/hands/{left,right,other,unsure}/<run>_<frame>_<k>.png (96x96) and data/hands/index.csv.
Held out = the run's seconds digit is 0, 1 or 2 (the same split as data/episodes.json).
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

from follow import MIN_HAND, detector, watch_score
from tracker import pose_detector, wrists

HERE = os.path.dirname(os.path.abspath(__file__))
FOLLOW = os.path.join(HERE, "data", "follow")
OUT = os.path.join(HERE, "data", "hands")
NEAR = 0.15      # wrist-to-wrist distance (frame units) that counts as "that is the same wrist"
PAD = 0.40       # square crop side = longest side of the landmark box * (1 + PAD)
SIZE = 96


def held_out(run):
    return run.split("-")[1][-1] in "012"


def square_box(pts, w, h):
    """Square crop (x0, y0, side) in pixels around the 21 landmarks, padded 40%."""
    xs = [p.x * w for p in pts]; ys = [p.y * h for p in pts]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    side = max(max(xs) - min(xs), max(ys) - min(ys)) * (1 + PAD)
    side = max(side, 24.0)
    return int(round(cx - side / 2)), int(round(cy - side / 2)), int(round(side))


def crop_square(bgr, x0, y0, side):
    """The square crop, zero-padded where it runs off the frame, resized to SIZE."""
    h, w = bgr.shape[:2]
    canvas = np.zeros((side, side, 3), np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x0 + side), min(h, y0 + side)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = bgr[sy0:sy1, sx0:sx1]
    return cv2.resize(canvas, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def label_from(size, dl, dr, mp_handed):
    """The label rule, from numbers only (so it can be re-run from index.csv).

    size: wrist-to-knuckle length in frame units; dl, dr: this hand's wrist to the body's left / right
    wrist (None when that wrist is not in view); mp_handed: the hand model's own left/right guess.
    Returns left, right, other, or unsure (not used for training).

    First pass (29 Oct) called every hand with no body wrist nearby OTHER; on inspection most of those
    were Carl's own left hand (ring in plain view) in frames where the body model saw no body. And the
    body's RIGHT disagreed with the hand model's handedness 18% of the time, LEFT only 1.4%. So now:
    a side label needs the body AND the hand model to agree, and OTHER means a hand too small to be his.
    """
    if size < MIN_HAND:
        return "other"                                   # a far-away person's hand
    best = min((v for v in (dl, dr) if v is not None), default=None)
    if best is None:
        return "unsure"                                  # no body wrist in view: probably Carl, side unknown
    if best < NEAR:
        side = "left" if dl is not None and dl == best else "right"
        return side if side == mp_handed else "unsure"
    # a big hand far from both of Carl's wrists was OTHER in the second pass; of 112 such crops the sampled
    # ones showed his ring, so the body model had the wrists wrong. OTHER is now far-away (small) hands only.
    return "unsure"


def label_hand(pts, body, size, mp_handed):
    wx, wy = pts[0].x, pts[0].y
    d = lambda bw: None if bw is None else math.hypot(wx - bw[0], wy - bw[1])
    dl, dr = d(body["left"]), d(body["right"])
    return label_from(size, dl, dr, mp_handed), dl, dr


def relabel():
    """Re-derive every label from index.csv with the current rule and move the files to match."""
    path = os.path.join(OUT, "index.csv")
    rows = list(csv.DictReader(open(path)))
    os.makedirs(os.path.join(OUT, "unsure"), exist_ok=True)
    counts = {}
    for r in rows:
        f = lambda v: None if v == "" else float(v)
        new = label_from(float(r["size"]), f(r["d_left"]), f(r["d_right"]), r["mp_handed"])
        old_file = r["file"]; name = os.path.basename(old_file)
        new_file = f"{new}/{name}"
        if new_file != old_file:
            os.replace(os.path.join(OUT, old_file), os.path.join(OUT, new_file))
        r["file"], r["label"] = new_file, new
        counts[new] = counts.get(new, 0) + 1
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("relabelled", len(rows), counts)


def main(all_runs):
    if all_runs:
        runs = sorted(d for d in os.listdir(FOLLOW) if os.path.exists(os.path.join(FOLLOW, d, "frames.csv")))
    else:
        runs = [r["run"] for r in json.load(open(os.path.join(HERE, "data", "episodes.json"))) if r.get("run")]
    episodes = {r["run"] for r in json.load(open(os.path.join(HERE, "data", "episodes.json"))) if r.get("run")}
    for c in ("left", "right", "other", "unsure"):
        os.makedirs(os.path.join(OUT, c), exist_ok=True)
    hand_det, pose = detector(), pose_detector()
    index = open(os.path.join(OUT, "index.csv"), "w", newline="")
    w = csv.writer(index)
    w.writerow(["file", "run", "frame", "label", "held_out", "episode", "watch_score", "size", "d_left", "d_right",
                "x0", "y0", "side", "img_w", "img_h", "mp_handed"])
    counts = {"left": 0, "right": 0, "other": 0}
    t0 = time.time(); n_frames = 0
    for run in runs:
        rd = os.path.join(FOLLOW, run)
        frames = sorted(f for f in os.listdir(rd) if f.endswith(".jpg"))
        ho = held_out(run)
        for frame in frames:
            jpeg = open(os.path.join(rd, frame), "rb").read()
            bgr = cv2.imdecode(np.frombuffer(jpeg, "uint8"), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            n_frames += 1
            H, W = bgr.shape[:2]
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            res = hand_det.detect(img)
            if not res.hand_landmarks:
                continue
            body = wrists(pose, jpeg)
            for k, (pts, hd) in enumerate(zip(res.hand_landmarks, res.handedness)):
                size = math.hypot(pts[9].x - pts[0].x, pts[9].y - pts[0].y)
                label, dl, dr = label_hand(pts, body, size, hd[0].category_name.lower())
                x0, y0, side = square_box(pts, W, H)
                crop = crop_square(bgr, x0, y0, side)
                name = f"{run}_{frame[:-4]}_{k}.png"
                cv2.imwrite(os.path.join(OUT, label, name), crop)
                counts[label] += 1
                w.writerow([f"{label}/{name}", run, frame, label, int(ho), int(run in episodes),
                            f"{watch_score(bgr, pts):.3f}", f"{size:.4f}",
                            "" if dl is None else f"{dl:.3f}", "" if dr is None else f"{dr:.3f}",
                            x0, y0, side, W, H, hd[0].category_name.lower()])
        index.flush()
        print(f"{run} {'HELD' if ho else 'train'} frames {len(frames)} totals {counts} ({time.time() - t0:.0f}s)", flush=True)
    print("done", n_frames, "frames", counts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-runs", action="store_true")
    ap.add_argument("--relabel", action="store_true", help="re-derive labels from index.csv, move the files; no detection")
    a = ap.parse_args()
    relabel() if a.relabel else main(a.all_runs)
