"""Grab one colour frame (and the depth map) from the wrist camera. Read-only.

    .venv/bin/python camgrab.py [out.jpg]
"""
import asyncio, sys
from viam.components.camera import Camera
from viamarm import connect

async def grab(cam, want_depth=True):
    """Return (jpeg bytes, depth bytes or None) from one get_images call.

    The depth map is 1.9 MB and costs about 350 ms a frame over the cloud (measured
    2026-10-29: 452 ms with it, 100 ms without). Ask for it only when you need it.
    """
    imgs, _ = await cam.get_images(**({} if want_depth else {"filter_source_names": ["color"]}))
    jpeg = depth = None
    for im in imgs:
        if im.mime_type == "image/jpeg":
            jpeg = im.data
        elif "dep" in im.mime_type:
            depth = im.data
    return jpeg, depth

async def main(out):
    async with await connect() as machine:
        cam = Camera.from_robot(machine, "cam")
        jpeg, depth = await grab(cam)
        open(out, "wb").write(jpeg)
        print("saved", out, len(jpeg), "bytes jpeg;", len(depth) if depth else 0, "bytes depth")

if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/cam.jpg"))
