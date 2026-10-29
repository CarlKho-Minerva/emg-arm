"""Is this hand crop Carl's LEFT hand (Apple Watch, silver ring)? A tiny net answers in about a millisecond.

    from hand_id import is_left, crop_for
    p = is_left(crop_for(bgr, pts))          # pts: the 21 mediapipe hand landmarks; p in 0..1

The net (data/hands/left_hand.onnx, trained by train_hands.py on the PC) says left / right / other
for a 96x96 RGB crop; is_left returns the probability of "left". crop_for cuts the same square the
training set used: the landmarks' box, padded 40%, zero-filled past the frame edge.

    .venv/bin/python hand_id.py data/follow/<run>/<frame>.jpg      # scores every hand in a picture
"""
import os
import sys
import time

import cv2
import numpy as np

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "hands", "left_hand.onnx")
SIZE = 96
PAD = 0.40
CLASSES = ["left", "right", "other"]
_sess = None


def _session():
    global _sess
    if _sess is None:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1          # one thread is fastest for a net this small; no thread pool wake-up
        so.inter_op_num_threads = 1
        _sess = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
    return _sess


def crop_for(bgr, pts):
    """Square crop around the 21 landmarks (objects with .x .y in 0..1), padded 40%, resized to 96x96 BGR."""
    h, w = bgr.shape[:2]
    xs = [p.x * w for p in pts]; ys = [p.y * h for p in pts]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    side = max(max(max(xs) - min(xs), max(ys) - min(ys)) * (1 + PAD), 24.0)
    x0, y0, side = int(round(cx - side / 2)), int(round(cy - side / 2)), int(round(side))
    canvas = np.zeros((side, side, 3), np.uint8)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x0 + side), min(h, y0 + side)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = bgr[sy0:sy1, sx0:sx1]
    return cv2.resize(canvas, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def probs(bgr_crop):
    """[p_left, p_right, p_other] for one BGR crop (any size; resized to 96x96)."""
    im = bgr_crop if bgr_crop.shape[:2] == (SIZE, SIZE) else cv2.resize(bgr_crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    x = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
    logits = _session().run(None, {"image": x})[0][0]
    e = np.exp(logits - logits.max())
    return e / e.sum()


def is_left(bgr_crop):
    """Probability (0..1) that the crop is Carl's left hand."""
    return float(probs(bgr_crop)[0])


if __name__ == "__main__":
    import mediapipe as mp
    from follow import detector
    det = detector()
    for path in sys.argv[1:]:
        bgr = cv2.imread(path)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        res = det.detect(img)
        for pts, hd in zip(res.hand_landmarks, res.handedness):
            crop = crop_for(bgr, pts)
            t0 = time.perf_counter(); p = probs(crop); ms = (time.perf_counter() - t0) * 1000
            print(f"{os.path.basename(path)} hand at ({pts[0].x:.2f},{pts[0].y:.2f}) mp says {hd[0].category_name.lower():5s} "
                  f"net: left {p[0]:.2f} right {p[1]:.2f} other {p[2]:.2f}  ({ms:.2f} ms)")
