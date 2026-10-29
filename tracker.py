"""Find one of Carl's arms in a picture by his body, not by his hand.

The pose model marks shoulder, elbow and wrist on each side, and which side is which comes
from the body, so "left" means his left whatever way he faces the camera. When the wrist is
out of the picture but the elbow is in, the target is where the wrist must be: a bit past the
elbow along the upper arm's line. That is what keeps the follower from giving up at the edge.

    .venv/bin/python tracker.py           # one frame: prints both arms and the target
"""
import os
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL = os.path.expanduser("~/.cache/pose_landmarker_heavy.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_lite/float16/1/pose_landmarker_heavy.task")
# landmark numbers in the pose model
SHOULDER = {"left": 11, "right": 12}
ELBOW = {"left": 13, "right": 14}
WRIST = {"left": 15, "right": 16}
VISIBLE = 0.5          # the model's confidence that a point is actually in view


def pose_detector():
    if not os.path.exists(MODEL):
        os.makedirs(os.path.dirname(MODEL), exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL)
    # heavy at a low threshold: the only one that found a half-body at the frame edge (2026-10-29, 32 ms);
    # lite and full saw nothing there at any threshold
    opts = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=MODEL), num_poses=1,
                                        min_pose_detection_confidence=0.2, min_pose_presence_confidence=0.2)
    return vision.PoseLandmarker.create_from_options(opts)


def arm_target(det, jpeg, side="left"):
    """(x, y, how) for the chosen arm, each 0..1 of the frame, or None.

    how is "wrist" when the wrist is in view, "elbow+" when it is guessed past the elbow,
    "shoulder+" when only the shoulder is, and the guess can be outside 0..1, which is useful:
    it says which way to turn.
    """
    bgr = cv2.imdecode(np.frombuffer(jpeg, "uint8"), cv2.IMREAD_COLOR)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    res = det.detect(img)
    if not res.pose_landmarks:
        return None
    p = res.pose_landmarks[0]
    s, e, w = p[SHOULDER[side]], p[ELBOW[side]], p[WRIST[side]]
    if w.visibility > VISIBLE:
        return w.x, w.y, "wrist"
    if e.visibility > VISIBLE and s.visibility > VISIBLE:
        # forearm roughly continues the upper arm; put the wrist 0.9 of an upper-arm past the elbow
        return e.x + 0.9 * (e.x - s.x), e.y + 0.9 * (e.y - s.y), "elbow+"
    if s.visibility > VISIBLE:
        return s.x + (0.25 if side == "right" else -0.25), s.y + 0.3, "shoulder+"
    return None


def both_arms(det, jpeg):
    return {side: arm_target(det, jpeg, side) for side in ("left", "right")}


if __name__ == "__main__":
    import asyncio
    from viam.components.camera import Camera
    from camgrab import grab
    from viamarm import connect

    async def main():
        det = pose_detector()
        async with await connect() as machine:
            jpeg, _ = await grab(Camera.from_robot(machine, "cam"), want_depth=False)
        for side, t in both_arms(det, jpeg).items():
            print(side, "arm:", None if t is None else f"({t[0]:.2f}, {t[1]:.2f}) by {t[2]}")

    asyncio.run(main())
