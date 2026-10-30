"""Train Carl's own face classifier from the calibration frames, in a second, no GPU.

Input: data/face_frames.jsonl, one line per calibration frame: label (rest, left, right, both, mouth)
and the 52 face numbers the face model gives (how much each part of the face has moved). Model: a
nearest-centroid classifier on the standardised numbers, with the rest class given a margin. It is
tiny on purpose: it must be retrained at the table in one calibration pass and never surprise anyone.

    .venv/bin/python train_face.py          # writes data/face_model.json, prints held-out accuracy
"""
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
FRAMES = os.path.join(HERE, "data", "face_frames.jsonl")
MODEL = os.path.join(HERE, "data", "face_model.json")
LABELS = ("rest", "left", "right", "both", "mouth")


def load():
    rows = [json.loads(l) for l in open(FRAMES) if l.strip()]
    names = sorted({k for r in rows for k in r["shapes"]})
    X = [[r["shapes"].get(n, 0.0) for n in names] for r in rows]
    y = [r["label"] for r in rows]
    return names, X, y


def fit(names, X, y):
    d = len(names)
    mean = [sum(x[i] for x in X) / len(X) for i in range(d)]
    std = [max((sum((x[i] - mean[i]) ** 2 for x in X) / len(X)) ** 0.5, 1e-3) for i in range(d)]
    Z = [[(x[i] - mean[i]) / std[i] for i in range(d)] for x in X]
    cent = {}
    for lab in LABELS:
        rows = [z for z, l in zip(Z, y) if l == lab]
        if rows:
            cent[lab] = [sum(r[i] for r in rows) / len(rows) for i in range(d)]
    return {"names": names, "mean": mean, "std": std, "centroids": cent}


def predict(model, shapes, rest_margin=0.15):
    z = [(shapes.get(n, 0.0) - m) / s for n, m, s in zip(model["names"], model["mean"], model["std"])]
    dists = {lab: sum((a - b) ** 2 for a, b in zip(z, c)) ** 0.5 for lab, c in model["centroids"].items()}
    best = min(dists, key=dists.get)
    # rest wins ties generously: a gesture must be clearly nearer its centroid than to rest
    if best != "rest" and "rest" in dists and dists[best] > dists["rest"] * (1 - rest_margin):
        return "rest", dists
    return best, dists


def main():
    names, X, y = load()
    idx = list(range(len(X))); random.Random(0).shuffle(idx)
    cut = int(len(idx) * 0.7)
    tr, te = idx[:cut], idx[cut:]
    model = fit(names, [X[i] for i in tr], [y[i] for i in tr])
    right = sum(1 for i in te if predict(model, dict(zip(names, X[i])))[0] == y[i])
    conf = {}
    for i in te:
        p = predict(model, dict(zip(names, X[i])))[0]
        conf[(y[i], p)] = conf.get((y[i], p), 0) + 1
    print(f"{len(X)} frames; held-out accuracy {right}/{len(te)} = {100 * right / max(len(te), 1):.1f}%")
    for (t, p), n in sorted(conf.items()):
        if t != p:
            print(f"   confused {t} as {p}: {n}")
    model = fit(names, X, y)
    json.dump(model, open(MODEL, "w"))
    print("saved", MODEL)


if __name__ == "__main__":
    main()
