"""The jaw-clench channel between the EMG process and the arm.

The EMG side (ssc live, in silent-speech-cerelog) writes one line per event to CLENCH_FILE; the
arm side waits for a fresh CLENCH line. One bit, by file, so either program can be restarted alone.

    .venv/bin/python clench.py fake          # pretend a clench happened now (for testing the arm side)
    .venv/bin/python clench.py bridge --port /dev/cu.usbserial-110   # run the real detector and feed the file
"""
import os
import sys
import time

CLENCH_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "clench.log")


def write(state):
    os.makedirs(os.path.dirname(CLENCH_FILE), exist_ok=True)
    with open(CLENCH_FILE, "a") as f:
        f.write(f"{time.time():.3f} {state}\n")


def last_clench_time():
    """Time of the newest CLENCH line, or 0."""
    try:
        lines = open(CLENCH_FILE).read().strip().split("\n")
    except FileNotFoundError:
        return 0.0
    for line in reversed(lines):
        t, state = line.split()
        if state == "CLENCH":
            return float(t)
    return 0.0


async def wait_for_clench(since, timeout, say=None, fallback=None):
    """Return True when a CLENCH newer than `since` arrives, False at timeout. `fallback` is an
    optional async function polled every loop; if it returns True the wait ends as if clenched."""
    import asyncio
    t0 = time.time()
    while time.time() - t0 < timeout:
        if last_clench_time() > since:
            return True
        if fallback and await fallback():
            return True
        await asyncio.sleep(0.05)
    return False


def bridge(port, channels=("1", "2")):
    """Run the real detector from silent-speech-cerelog and copy its events to the file."""
    sys.path.insert(0, os.path.expanduser("~/CODELocalProjects/silent-speech-cerelog"))
    import argparse
    from ssc import live
    args = argparse.Namespace(port=port, channels=[int(c) for c in channels], fake=False, seconds=0,
                              board_version="auto", off_body=False)
    for ev in live.events(args):
        state = ev.get("state")
        if state in ("CLENCH", "RELEASE", "ARMED"):
            write(state)
            print(f"{time.strftime('%H:%M:%S')} {state}", flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["fake"]:
        write("CLENCH"); print("wrote a CLENCH")
    elif sys.argv[1:2] == ["bridge"]:
        port = sys.argv[sys.argv.index("--port") + 1] if "--port" in sys.argv else "/dev/cu.usbserial-110"
        bridge(port)
    else:
        print(__doc__)
