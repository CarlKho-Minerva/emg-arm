"""Served With a Smile: the face as the controller, from the MacBook camera.

The face model gives, every frame, how much the left and right corners of the mouth are raised
(blendshapes mouthSmileLeft and mouthSmileRight, 0 to 1). From those, three events:
  L  half-smile left   (left corner up, right not)
  R  half-smile right
  B  both corners up   (a full smile)
An event fires when a corner crosses UP and ends when it drops below DOWN. Events closer than
GROUP_S are one group (L, LL, B B B); a group is acted on after GROUP_S of quiet. Each group is
written to data/face.log as "<time> <group>" for the arm side, and printed.

    .venv/bin/python smile.py            # run; press q in the window to stop
    .venv/bin/python smile.py --no-window
"""
import argparse
import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL = os.path.expanduser("~/.cache/face_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/"
             "float16/1/face_landmarker.task")
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "face.log")
UP, DOWN = 0.18, 0.10      # a side counts as pulled above UP, released below DOWN (Carl's cheek pull reads 0.20, rest 0.08)
SIDE_GAP = 0.10            # one side must beat the other by this to be one-sided; closer = both
GROUP_S = 0.8              # events within this are one group; quiet this long ends the group
MIN_HOLD_S = 0.08          # a corner must stay up this long (one blink of noise is not a smile)


def detector():
    if not os.path.exists(MODEL):
        os.makedirs(os.path.dirname(MODEL), exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL)
    opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=MODEL), num_faces=1,
                                        output_face_blendshapes=True, running_mode=vision.RunningMode.VIDEO)
    return vision.FaceLandmarker.create_from_options(opts)


SIDE_SHAPES = ("mouthSmile{}", "cheekSquint{}", "mouth{}", "mouthPress{}", "mouthDimple{}")   # the cheek thing: any one-sided pull


def smile_scores(result):
    """(left, right): the strongest one-sided signal on each side. A smirk, a cheek squint, the mouth
    pulled to one side, a pressed corner, a dimple all count; B needs both sides up together."""
    if not result.face_blendshapes:
        return None
    d = {b.category_name: b.score for b in result.face_blendshapes[0]}
    left = max(d.get(n.format("Left"), 0.0) for n in SIDE_SHAPES)
    right = max(d.get(n.format("Right"), 0.0) for n in SIDE_SHAPES)
    return right, left      # mirrored frame: the model's "Left" is Carl's right (checked on his face, 29 Oct 19:50)


def write_group(group):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as f:
        f.write(f"{time.time():.3f} {group}\n")


def main(window=True, cam=0):
    det = detector()
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        raise SystemExit("no camera; macOS may need to grant camera access to the terminal")
    up = None            # the event currently being held: "L", "R", "B" or None
    up_since = 0.0
    counted = False
    group = []
    last_event = 0.0
    t0 = time.time()
    print("Served With a Smile. Smile left, right, or both. Groups print when you stop.", flush=True)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)           # mirror, so left means your left
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int((time.time() - t0) * 1000))
        sc = smile_scores(res)
        now = time.time()
        if sc:
            l, r = sc
            # after the mirror, the picture's left is YOUR left, and the model's "Left" is the picture's left
            if up is None:
                if l > UP or r > UP:
                    if abs(l - r) < SIDE_GAP or (l > UP and r > UP):
                        up = "B"
                    else:
                        up = "L" if l > r else "R"
                    up_since = now; counted = False
            else:
                if not counted and now - up_since >= MIN_HOLD_S:
                    group.append(up); last_event = now; counted = True
                    print(f"  {''.join(group)}", flush=True)
                if l < DOWN and r < DOWN:
                    up = None
        if group and up is None and now - last_event > GROUP_S:
            g = "".join(group)
            write_group(g)
            print(f"{time.strftime('%H:%M:%S')} GROUP {g}", flush=True)
            group = []
        if int(now * 4) != int((now - 0.05) * 4):           # four times a second: what it sees, for the other thread
            import json
            with open(LOG.replace("face.log", "face_status.json"), "w") as f:
                json.dump({"t": now, "left": sc[0] if sc else None, "right": sc[1] if sc else None,
                           "holding": up, "group": "".join(group)}, f)
        if window:
            txt = f"L {sc[0]:.2f}  R {sc[1]:.2f}" if sc else "no face"
            cv2.putText(frame, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "".join(group) or ("holding " + up if up else ""), (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 90), 2)
            cv2.imshow("Served With a Smile", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--cam", type=int, default=0)
    a = ap.parse_args()
    main(window=not a.no_window, cam=a.cam)
