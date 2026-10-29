"""Teach a tiny net to say whether a hand crop is Carl's LEFT hand (watch, silver ring), his RIGHT, or OTHER.

Reads data/hands/index.csv (from build_hands.py). Trains on every crop whose run is not held out;
reports on the held-out episodes (seconds digit 0, 1, 2). No flips: a flipped left hand looks like
a right hand. Brightness, contrast, small shifts, zooms and rotations only.

    python train_hands.py --data data/hands --out data/hands --epochs 20

Writes <out>/left_hand.onnx (opset 17, batch dynamic, input float32 Nx3x96x96 RGB in 0..1,
output Nx3 logits for [left, right, other]) and <out>/train_report.json.
"""
import argparse
import csv
import json
import math
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = ["left", "right", "other"]
SIZE = 96


class HandNet(nn.Module):
    """Four stride-2 convs, 96 -> 6 pixels, global average, one linear. About 100k weights."""

    def __init__(self, width=24):
        super().__init__()
        w = width
        chans = [3, w, 2 * w, 4 * w, 8 * w]
        layers = []
        for i in range(4):
            layers += [nn.Conv2d(chans[i], chans[i + 1], 3, stride=2, padding=1, bias=False),
                       nn.BatchNorm2d(chans[i + 1]), nn.ReLU(inplace=True)]
            if i >= 1:
                layers += [nn.Conv2d(chans[i + 1], chans[i + 1], 3, padding=1, bias=False),
                           nn.BatchNorm2d(chans[i + 1]), nn.ReLU(inplace=True)]
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(chans[-1], len(CLASSES))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):            # x: N x 3 x 96 x 96, RGB, 0..1
        x = (x - self.mean) / self.std
        x = self.features(x)
        return self.head(x.mean((2, 3)))


def load(data):
    rows = [r for r in csv.DictReader(open(os.path.join(data, "index.csv"))) if r["label"] in CLASSES]   # "unsure" is dropped
    imgs = np.zeros((len(rows), SIZE, SIZE, 3), np.uint8)
    y = np.zeros(len(rows), np.int64)
    held = np.zeros(len(rows), bool); epi = np.zeros(len(rows), bool)
    for i, r in enumerate(rows):
        im = cv2.imread(os.path.join(data, r["file"]))
        imgs[i] = cv2.cvtColor(cv2.resize(im, (SIZE, SIZE)), cv2.COLOR_BGR2RGB)
        y[i] = CLASSES.index(r["label"]); held[i] = r["held_out"] == "1"; epi[i] = r["episode"] == "1"
    x = torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous()   # uint8 N x 3 x H x W
    return x, torch.from_numpy(y), torch.from_numpy(held), torch.from_numpy(epi), rows


def augment(x):
    """x: float N x 3 x 96 x 96 on the GPU. Shift, zoom, rotate a little; brightness, contrast, colour gain, noise. No flip."""
    n = x.shape[0]; dev = x.device
    ang = (torch.rand(n, device=dev) * 2 - 1) * math.radians(12)
    scale = 1 + (torch.rand(n, device=dev) * 2 - 1) * 0.15
    tx = (torch.rand(n, device=dev) * 2 - 1) * 0.12; ty = (torch.rand(n, device=dev) * 2 - 1) * 0.12
    cos, sin = torch.cos(ang) * scale, torch.sin(ang) * scale
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
    bright = 1 + (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * 0.35
    gain = 1 + (torch.rand(n, 3, 1, 1, device=dev) * 2 - 1) * 0.08
    x = x * bright * gain
    contrast = 1 + (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * 0.3
    m = x.mean((1, 2, 3), keepdim=True)
    x = (x - m) * contrast + m
    x = x + torch.randn_like(x) * 0.02
    return x.clamp(0, 1)


@torch.no_grad()
def predict(model, x, dev, bs=1024):
    model.eval(); out = []
    for i in range(0, len(x), bs):
        out.append(model(x[i:i + bs].to(dev).float() / 255).argmax(1).cpu())
    return torch.cat(out) if out else torch.zeros(0, dtype=torch.long)


def report(name, y, p):
    k = len(CLASSES)
    cm = np.zeros((k, k), int)
    for a, b in zip(y.tolist(), p.tolist()):
        cm[a, b] += 1
    acc = float(np.trace(cm) / max(cm.sum(), 1))
    per = {}
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]; prec = tp / max(cm[:, i].sum(), 1); rec = tp / max(cm[i].sum(), 1)
        per[c] = {"precision": round(float(prec), 4), "recall": round(float(rec), 4), "n": int(cm[i].sum())}
    print(f"{name}: n={cm.sum()} acc={acc:.4f}")
    print("  rows=true [left right other], cols=predicted")
    for i, c in enumerate(CLASSES):
        print(f"  {c:5s} {cm[i].tolist()}  prec {per[c]['precision']:.3f} rec {per[c]['recall']:.3f}")
    return {"n": int(cm.sum()), "acc": round(acc, 4), "confusion": cm.tolist(), "per_class": per}


def main(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "")
    x, y, held, epi, rows = load(a.data)
    tr = ~held
    print("crops", len(x), "train", int(tr.sum()), "held-out", int(held.sum()),
          {c: int((y[tr] == i).sum()) for i, c in enumerate(CLASSES)})
    xtr, ytr = x[tr], y[tr]
    counts = torch.bincount(ytr, minlength=3).float()
    weights = (1.0 / counts)[ytr]
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(ytr), replacement=True)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xtr, ytr), batch_size=a.batch,
                                         sampler=sampler, drop_last=True, num_workers=0)
    torch.manual_seed(a.seed)
    model = HandNet(a.width).to(dev)
    print("params", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.15)
    t0 = time.time()
    for ep in range(a.epochs):
        model.train(); tot = 0.0; correct = 0; n = 0
        for xb, yb in loader:
            xb = augment(xb.to(dev, non_blocking=True).float() / 255); yb = yb.to(dev)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(yb); correct += (logits.argmax(1) == yb).sum().item(); n += len(yb)
        ptr = predict(model, xtr, dev); pho = predict(model, x[held], dev)
        print(f"epoch {ep + 1:2d} loss {tot / n:.4f} aug-acc {correct / n:.4f}  train-acc {(ptr == ytr).float().mean():.4f}"
              f"  held-out-acc {(pho == y[held]).float().mean():.4f}  ({time.time() - t0:.0f}s)", flush=True)
    model.eval()
    rep = {"params": sum(p.numel() for p in model.parameters()), "epochs": a.epochs, "width": a.width,
           "train_n": int(tr.sum())}
    rep["train"] = report("train (no augmentation)", ytr, predict(model, xtr, dev))
    he = held & epi
    rep["held_out_episodes"] = report("held-out episodes (data/episodes.json)", y[he], predict(model, x[he], dev))
    if int((held & ~epi).sum()):
        rep["held_out_extra_runs"] = report("held-out extra runs (not in episodes.json)", y[held & ~epi], predict(model, x[held & ~epi], dev))
    rep["held_out_all"] = report("held-out all", y[held], predict(model, x[held], dev))
    # how sure is it on held-out episode crops, so the caller can pick a threshold
    with torch.no_grad():
        prob = F.softmax(model(x[he].to(dev).float() / 255), 1)[:, 0].cpu()
    yl = (y[he] == 0)
    for th in (0.5, 0.7, 0.9):
        pred = prob > th
        tp = int((pred & yl).sum()); fp = int((pred & ~yl).sum()); fn = int((~pred & yl).sum())
        print(f"  P(left) > {th}: precision {tp / max(tp + fp, 1):.3f} recall {tp / max(tp + fn, 1):.3f}")
        rep[f"left_thresh_{th}"] = {"precision": round(tp / max(tp + fp, 1), 4), "recall": round(tp / max(tp + fn, 1), 4)}
    # which held-out crops it got wrong, for looking at
    pho = predict(model, x[he], dev)
    idx = torch.nonzero(he).flatten()
    wrong = [{"file": rows[int(idx[i])]["file"], "pred": CLASSES[int(pho[i])], "p_left": round(float(prob[i]), 3)}
             for i in range(len(pho)) if pho[i] != y[he][i]]
    rep["wrong_held_out"] = wrong[:200]
    torch.save(model.state_dict(), os.path.join(a.out, "left_hand.pt"))
    model_cpu = model.cpu().eval()
    dummy = torch.zeros(1, 3, SIZE, SIZE)
    onnx_path = os.path.join(a.out, "left_hand.onnx")
    torch.onnx.export(model_cpu, dummy, onnx_path, opset_version=17, input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, dynamo=False)
    # check the ONNX file says the same as torch
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    xs = x[he][:256].float() / 255
    o = sess.run(None, {"image": xs.numpy()})[0]
    with torch.no_grad():
        t = model_cpu(xs).numpy()
    rep["onnx_max_abs_diff"] = float(np.abs(o - t).max())
    print("onnx vs torch max abs diff", rep["onnx_max_abs_diff"], "file", os.path.getsize(onnx_path), "bytes")
    json.dump(rep, open(os.path.join(a.out, "train_report.json"), "w"), indent=1)
    print("wrote", onnx_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join("data", "hands"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out = a.out or a.data
    main(a)
