"""Keep data/live.jpg fresh from the wrist camera, read-only, for the demo screen (dash.py).

    .venv/bin/python view.py        # four colour frames a second; Ctrl-C to stop

serve.py publishes the same picture while it idles; this runs when serve.py is not up (Carl testing pick.py by
hand, or the arm off) so the screen never shows a stale frame. Never touches the arm or the gripper.
"""
import asyncio
import os
import time

from viam.components.camera import Camera

from camgrab import grab
from viamarm import connect

HERE = os.path.dirname(os.path.abspath(__file__))
LIVE_JPG = os.path.join(HERE, "data", "live.jpg")


async def main():
    async with await connect() as machine:
        cam = Camera.from_robot(machine, "cam")
        n = 0
        while True:
            try:
                jpeg, _ = await asyncio.wait_for(grab(cam, want_depth=False), 3)
                tmp = LIVE_JPG + ".tmp"
                with open(tmp, "wb") as f:
                    f.write(jpeg)
                os.replace(tmp, LIVE_JPG)
                n += 1
                if n % 200 == 1:
                    print(f"{time.strftime('%H:%M:%S')} frame {n}", flush=True)
            except Exception as e:
                print(f"{time.strftime('%H:%M:%S')} no frame: {str(e)[:80]}", flush=True)
                await asyncio.sleep(1)
            await asyncio.sleep(0.25)


if __name__ == "__main__":
    asyncio.run(main())
