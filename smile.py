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
REFRACTORY_S = 0.3         # after a release, nothing new counts for this long (a relaxing cheek crosses the other side's line)


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


CALIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "face_calib.json")
FRAMES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "face_frames.jsonl")
FACE_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "face_model.json")
GESTURES = (("rest", "relax your face"), ("left", "pull your LEFT cheek up (a smirk)"), ("right", "pull your RIGHT cheek up"),
            ("both", "full smile, both sides"), ("mouth", "open your mouth"))


def card(frame, title, lines, colour=(255, 255, 255)):
    """A dark card over the picture with a title and lines of text."""
    h, w = frame.shape[:2]
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (w, h), (10, 10, 10), -1)
    frame = cv2.addWeighted(panel, 0.78, frame, 0.22, 0)
    cv2.putText(frame, title, (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.4, colour, 3)
    y = 130
    for line in lines:
        cv2.putText(frame, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (230, 230, 230), 2)
        y += 40
    return frame


def calibrate(cam=0, seconds=3.0):
    """Measure Carl's own rest and gestures in the window, coach-style: one instruction at a time, a
    countdown, a bar, then a card with the numbers. Thresholds halfway between rest and the gesture."""
    import json, statistics
    det = detector()
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        raise SystemExit("no camera")
    win = "Served With a Smile: calibration"
    t0 = time.time(); results = {}; frames = []
    steps = [("rest", "RELAX. Straight face."), ("left", "Pull your LEFT cheek up. A smirk."), ("right", "Pull your RIGHT cheek up."),
             ("both", "Full smile, both sides."), ("mouth", "Open your mouth.")]
    def show(frame, title, lines, colour=(255, 255, 255)):
        cv2.imshow(win, card(cv2.flip(frame, 1), title, lines, colour)); cv2.waitKey(1)
    ok, frame = cap.read()
    for i in range(3, 0, -1):
        ok, frame = cap.read(); show(frame, "Calibration", [f"Five short steps, {seconds:.0f} s each. Face the camera.", f"Starting in {i}"]); time.sleep(1)
    for n, (name, instruction) in enumerate(steps, 1):
        for i in range(2, 0, -1):
            ok, frame = cap.read(); show(frame, f"Step {n} of {len(steps)}", [instruction, f"in {i}..."], (0, 200, 90)); time.sleep(1)
        ls, rs, js = [], [], []
        t_start = time.time()
        while time.time() - t_start < seconds:
            ok, frame = cap.read()
            if not ok:
                continue
            fr = cv2.flip(frame, 1)
            res = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)), int((time.time() - t0) * 1000))
            sc = smile_scores(res)
            if sc:
                ls.append(sc[0]); rs.append(sc[1])
                d = {b.category_name: b.score for b in res.face_blendshapes[0]}
                js.append(d.get("jawOpen", 0.0))
                frames.append({"label": name, "t": time.time(), "shapes": d})
            left = seconds - (time.time() - t_start)
            img = card(fr, f"Step {n} of {len(steps)}: {instruction}", [f"hold it... {left:.1f} s", "" if sc else "NO FACE SEEN"], (0, 200, 90))
            hh, ww = img.shape[:2]
            cv2.rectangle(img, (40, hh - 60), (ww - 40, hh - 30), (60, 60, 60), -1)
            cv2.rectangle(img, (40, hh - 60), (40 + int((ww - 80) * (1 - left / seconds)), hh - 30), (0, 200, 90), -1)
            cv2.imshow(win, img); cv2.waitKey(1)
        if ls:
            results[name] = {"left": statistics.median(ls), "right": statistics.median(rs), "jaw": statistics.median(js), "n": len(ls)}
    cap.release()
    r = results.get("rest", {"left": 0, "right": 0, "jaw": 0})
    def mid(a, b): return a + 0.5 * (b - a)
    calib = {
        "up_left": mid(r["left"], results.get("left", {}).get("left", r["left"] + 0.4)),
        "up_right": mid(r["right"], results.get("right", {}).get("right", r["right"] + 0.4)),
        "down": max(r["left"], r["right"]) + 0.03,
        "jaw_open": mid(r["jaw"], results.get("mouth", {}).get("jaw", r["jaw"] + 0.4)),
        "measured": results, "at": time.strftime("%Y-%m-%d %H:%M"),
    }
    os.makedirs(os.path.dirname(CALIB), exist_ok=True)
    json.dump(calib, open(CALIB, "w"), indent=1)
    with open(FRAMES, "a") as f:                       # every frame of every calibration, for train_face.py
        for fr in frames:
            f.write(json.dumps(fr) + "\n")
    lines = [f"{k:6s}  left {v['left']:.2f}   right {v['right']:.2f}   mouth {v['jaw']:.2f}" for k, v in results.items()]
    lines += ["", f"triggers: left {calib['up_left']:.2f}  right {calib['up_right']:.2f}  mouth {calib['jaw_open']:.2f}",
              "saved. Press any key to close."]
    sep_ok = results.get("left", {}).get("left", 0) - results.get("left", {}).get("right", 0) > 0.10 and \
             results.get("right", {}).get("right", 0) - results.get("right", {}).get("left", 0) > 0.10
    img = card(cv2.flip(frame, 1), "Done" if sep_ok else "Done, but the sides are close; try again", lines, (0, 200, 90) if sep_ok else (0, 160, 255))
    cv2.imshow(win, img); cv2.waitKey(0); cv2.destroyAllWindows()
    print(json.dumps({k: round(v, 3) for k, v in calib.items() if isinstance(v, float)}))


INSTRUCTIONS = (
    ("SERVED WITH A SMILE", None),
    ("", None),
    ("LEFT cheek up", "fetch the cube"),
    ("RIGHT cheek up", "hand it over"),
    ("OPEN your mouth", "feed me"),
    ("SMILE once", "let go"),
    ("SMILE twice", "nudge mode"),
    ("SMILE 3 times", "home"),
    ("", None),
    ("in nudge mode:", None),
    ("LEFT / RIGHT", "move a step"),
    ("SMILE twice", "next axis"),
    ("SMILE once", "grab / release"),
)


def draw_panel(frame, sc, up, group, thresholds):
    """The instructions down the right side, the live scores as bars, the group being heard.
    (Dropped by mistake in the calibration commit of 19:50; restored 23:10 with the mouth line added.)"""
    h, w = frame.shape[:2]
    pw = 360
    panel = frame.copy()
    cv2.rectangle(panel, (w - pw, 0), (w, h), (18, 18, 18), -1)
    frame = cv2.addWeighted(panel, 0.82, frame, 0.18, 0)
    y = 44
    for left, right in INSTRUCTIONS:
        if left == "SERVED WITH A SMILE":
            cv2.putText(frame, left, (w - pw + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        elif left:
            cv2.putText(frame, left, (w - pw + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 200, 90) if right else (170, 170, 170), 1)
            if right:
                cv2.putText(frame, right, (w - pw + 190, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 1)
        y += 30
    # live bars: how far each side is toward its trigger
    y += 10
    for label, val, thr in (("left", sc[0] if sc else 0.0, thresholds[0]), ("right", sc[1] if sc else 0.0, thresholds[1])):
        cv2.putText(frame, f"{label} {val:.2f}", (w - pw + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        x0, x1 = w - pw + 120, w - 30
        cv2.rectangle(frame, (x0, y - 14), (x1, y - 2), (60, 60, 60), -1)
        fill = int((x1 - x0) * min(val / max(thr * 2, 0.01), 1.0))
        cv2.rectangle(frame, (x0, y - 14), (x0 + fill, y - 2), (0, 200, 90) if val > thr else (120, 120, 120), -1)
        tx = x0 + int((x1 - x0) * 0.5)
        cv2.line(frame, (tx, y - 17), (tx, y + 1), (255, 255, 255), 1)
        y += 28
    heard = "".join(group) or ("holding " + up if up else "")
    cv2.putText(frame, heard, (w - pw + 20, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 200, 90), 2)
    if not sc:
        cv2.putText(frame, "no face", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return frame


FACE_JPG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "face_live.jpg")


def publish_face(frame):
    """The frame the window shows, for the demo screen (dash.py). Atomic: a reader never sees half a file."""
    tmp = FACE_JPG + ".tmp"
    with open(tmp, "wb") as f:
        f.write(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes())
    os.replace(tmp, FACE_JPG)


def load_calib():
    import json
    if os.path.exists(CALIB):
        c = json.load(open(CALIB))
        return c["up_left"], c["up_right"], c["down"], c.get("jaw_open", 0.5)
    return UP, UP, DOWN, 0.5


def main(window=True, cam=0):
    up_l, up_r, down, jaw_open = load_calib()
    mouth_up = False; mouth_at = 0.0; jaw = 0.0
    published_at = 0.0
    print(f"thresholds: left {up_l:.2f} right {up_r:.2f} release {down:.2f}" + ("" if os.path.exists(CALIB) else " (defaults; run --calibrate)"), flush=True)
    det = detector()
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        raise SystemExit("no camera; macOS may need to grant camera access to the terminal")
    up = None            # the event currently being held: "L", "R", "B" or None
    released_at = 0.0
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
        if sc and res.face_blendshapes:
            jaw = {b.category_name: b.score for b in res.face_blendshapes[0]}.get("jawOpen", 0.0)
            if jaw > jaw_open and not mouth_up and now - mouth_at > 2.0:
                mouth_up = True; mouth_at = now
                write_group("M"); print(f"{time.strftime('%H:%M:%S')} GROUP M (mouth open)", flush=True)
            elif jaw < jaw_open * 0.6:
                mouth_up = False
        if sc:
            l, r = sc
            # after the mirror, the picture's left is YOUR left, and the model's "Left" is the picture's left
            if up is None:
                if (l > up_l or r > up_r) and now - released_at > REFRACTORY_S:
                    if l > up_l and r > up_r:           # both sides up together: B
                        up = "B"
                    else:
                        up = "L" if (l - up_l) > (r - up_r) else "R"
                    up_since = now; counted = False
            else:
                if not counted and now - up_since >= MIN_HOLD_S:
                    group.append(up); last_event = now; counted = True
                    print(f"  {''.join(group)}", flush=True)
                if l < down and r < down:
                    up = None; released_at = now
        if group and up is None and now - last_event > GROUP_S:
            g = "".join(group)
            write_group(g)
            print(f"{time.strftime('%H:%M:%S')} GROUP {g}", flush=True)
            group = []
        if int(now * 4) != int((now - 0.05) * 4):           # four times a second: what it sees, for the other thread
            import json
            with open(LOG.replace("face.log", "face_status.json"), "w") as f:
                json.dump({"t": now, "left": sc[0] if sc else None, "right": sc[1] if sc else None,
                           "jaw": jaw if sc else None, "holding": up, "group": "".join(group)}, f)
        publish = now - published_at >= 0.1                  # at most ten times a second, for the demo screen
        if window or publish:
            frame = draw_panel(frame, sc, up, group, (up_l, up_r))
            if publish:
                publish_face(frame); published_at = now
        if window:
            cv2.imshow("Served With a Smile", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--calibrate", action="store_true", help="measure rest and each gesture, save the thresholds")
    a = ap.parse_args()
    if a.calibrate:
        calibrate(cam=a.cam)
    else:
        main(window=not a.no_window, cam=a.cam)
