"""Train Carl's own face classifier from the calibration frames, in a second, no GPU.

Input: data/face_frames.jsonl, one line per calibration frame: label (rest, left, right, both, mouth)
and the 52 face numbers the face model gives (how much each part of the face has moved). Model: a
multinomial logistic regression on the standardised numbers, plain numpy. It is tiny on purpose: it
retrains at the table from one calibration pass in under a second and never surprises anyone.

    .venv/bin/python train_face.py          # writes data/face_model.json, prints held-out accuracy
"""
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
FRAMES = os.path.join(HERE, "data", "face_frames.jsonl")
MODEL = os.path.join(HERE, "data", "face_model.json")
LABELS = ("rest", "left", "right", "both", "mouth")


def load(all_sessions=False):
    """Frames of the newest calibration pass only (the one with head turning, 30 Oct), unless all_sessions.
    Mixing an old pass without the geometry features in would teach the model that missing features mean
    something; they mean the pass was older."""
    rows = [json.loads(l) for l in open(FRAMES) if l.strip()]
    sessions = [r.get("session") for r in rows if r.get("session")]
    if sessions and not all_sessions:
        newest = max(sessions)
        rows = [r for r in rows if r.get("session") == newest]
        print(f"training on calibration {newest} ({len(rows)} frames)")
    names = sorted({k for r in rows for k in r["shapes"]})
    X = [[r["shapes"].get(n, 0.0) for n in names] for r in rows]
    y = [r["label"] for r in rows]
    return names, X, y


def fit(names, X, y, iters=400, lr=0.5, l2=1e-3):
    """Multinomial logistic regression on standardised features, plain numpy. Nearest-centroid failed the
    synthetic check (47 noise dimensions drowned the 5 that matter: 27% accuracy); this weighs each
    feature by how much it separates the classes."""
    import numpy as np
    X = np.array(X, dtype=float); labs = [l for l in LABELS if l in set(y)]
    Y = np.array([labs.index(l) for l in y])
    mean, std = X.mean(0), np.maximum(X.std(0), 1e-3)
    Z = (X - mean) / std
    n, d = Z.shape; k = len(labs)
    W = np.zeros((d, k)); b = np.zeros(k)
    onehot = np.eye(k)[Y]
    # class weights so a short class (few frames) is not ignored
    cw = (n / (k * np.bincount(Y, minlength=k)))[Y][:, None]
    for _ in range(iters):
        logits = Z @ W + b
        logits -= logits.max(1, keepdims=True)
        P = np.exp(logits); P /= P.sum(1, keepdims=True)
        G = (P - onehot) * cw / n
        W -= lr * (Z.T @ G + l2 * W); b -= lr * G.sum(0)
    return {"names": names, "mean": mean.tolist(), "std": std.tolist(), "labels": labs, "W": W.tolist(), "b": b.tolist()}


def predict(model, shapes, rest_margin=0.0):
    """(label, probabilities). rest_margin: a gesture must beat rest by this much probability."""
    import math
    z = [(shapes.get(n, 0.0) - m) / s for n, m, s in zip(model["names"], model["mean"], model["std"])]
    labs = model["labels"]; W = model["W"]; b = model["b"]
    logits = [sum(z[i] * W[i][j] for i in range(len(z))) + b[j] for j in range(len(labs))]
    mx = max(logits); ex = [math.exp(v - mx) for v in logits]; tot = sum(ex)
    probs = {lab: e / tot for lab, e in zip(labs, ex)}
    best = max(probs, key=probs.get)
    if best != "rest" and "rest" in probs and probs[best] - probs["rest"] < rest_margin:
        return "rest", probs
    return best, probs


def main():
    import sys
    names, X, y = load(all_sessions="--all" in sys.argv)
    print("features:", len(names), "including geometry" if any(n.startswith("geo_") for n in names) else "(blendshapes only)")
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
