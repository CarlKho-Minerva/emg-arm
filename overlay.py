"""Draw what the follower sees on the camera picture, and publish it for the coach page.

The picture gets: the dead zone (a box at the centre; inside it the arm holds still), the raw
hand position (thin ring), the smoothed one (filled dot), the aim point when looking ahead (small
cross), a line from the centre to the aim point (the error the controller acts on), and a status
line. Written as data/live.jpg plus data/live.json every frame, atomically, so the page can poll.
"""
import json
import os
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIVE_JPG = os.path.join(HERE, "data", "live.jpg")
LIVE_JSON = os.path.join(HERE, "data", "live.json")
_last = {"t": 0.0, "n": 0, "fps": 0.0}


def draw(jpeg, raw=None, sm=None, aim=None, deadband=0.10, status="", how="", step=(0.0, 0.0), joints=None):
    bgr = cv2.imdecode(np.frombuffer(jpeg, "uint8"), cv2.IMREAD_COLOR)
    h, w = bgr.shape[:2]
    cx, cy = w // 2, h // 2
    px = lambda p: (int(p[0] * w), int(p[1] * h))
    # dead zone
    dx, dy = int(deadband * w), int(deadband * h)
    cv2.rectangle(bgr, (cx - dx, cy - dy), (cx + dx, cy + dy), (200, 200, 200), 1)
    cv2.drawMarker(bgr, (cx, cy), (200, 200, 200), cv2.MARKER_CROSS, 24, 1)
    if raw:
        cv2.circle(bgr, px(raw), 18, (255, 255, 255), 2)
    if sm:
        cv2.circle(bgr, px(sm), 9, (0, 200, 90), -1)
    if aim:
        cv2.drawMarker(bgr, px(aim), (0, 120, 255), cv2.MARKER_TILTED_CROSS, 22, 2)
        cv2.line(bgr, (cx, cy), px(aim), (0, 120, 255), 2)
    elif sm:
        cv2.line(bgr, (cx, cy), px(sm), (0, 200, 90), 2)
    # status strip
    cv2.rectangle(bgr, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.putText(bgr, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()


def publish(jpeg_annotated, **info):
    """Write the picture and the numbers for the page. Atomic so a reader never sees half a file."""
    os.makedirs(os.path.dirname(LIVE_JPG), exist_ok=True)
    now = time.time()
    _last["n"] += 1
    if now - _last["t"] >= 1.0:
        _last["fps"] = _last["n"] / (now - _last["t"]) if _last["t"] else 0.0
        _last["t"], _last["n"] = now, 0
    info["fps"] = round(_last["fps"], 1)
    info["t"] = now
    tmp = LIVE_JPG + ".tmp"
    with open(tmp, "wb") as f:
        f.write(jpeg_annotated)
    os.replace(tmp, LIVE_JPG)
    tmp = LIVE_JSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump(info, f)
    os.replace(tmp, LIVE_JSON)
