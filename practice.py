"""Practice the cube pick on its own: pick each colour, lift, put it back, log the result, repeat.

    .venv/bin/python practice.py --live --rounds 3

Someone at the red button. No handover: the cube goes back down where it was. Results append to
data/practice.json (colour, held, finger position, how many passes, the estimates) so the misses can
be studied.
"""
import argparse
import asyncio
import json
import os
import time

from pick import CUBE_ABOVE, pick_cube_once, plan_to, say
from handme import Rig, StepTimeout, step
from viamarm import connect

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "data", "practice.json")
PUT_Z = 75.0      # steered-point height to open at when putting the cube back


async def put_back(rig, x, y):
    await plan_to(rig, x, y, CUBE_ABOVE, "over the spot")
    await plan_to(rig, x, y, PUT_Z, "down to put it back")
    await step("gripper open", rig.gripper.open()); await asyncio.sleep(0.5)
    await plan_to(rig, x, y, CUBE_ABOVE, "up")


async def main(live, rounds, colours):
    results = json.load(open(LOG)) if os.path.exists(LOG) else []
    async with await connect() as machine:
        rig = await Rig(machine, live).setup()
        for rnd in range(1, rounds + 1):
            for colour in colours:
                t0 = time.time()
                await say(f"=== round {rnd}, {colour}")
                try:
                    ok = await pick_cube_once(rig, colour, attempt=2)      # attempt=2: skip the greet-first step
                except StepTimeout as e:
                    await say(f"{colour}: a step hung ({e}); arm stop, counting it as a miss")
                    try:
                        await asyncio.wait_for(rig.arm.stop(), 5)
                    except Exception:
                        pass
                    ok = False
                res = {"at": time.strftime("%H:%M:%S"), "round": rnd, "colour": colour, "held": bool(ok),
                       "seconds": round(time.time() - t0, 1)}
                if ok and live:
                    # pick_cube_once ends at greet; the cube's spot is in its last log line, so re-find it is
                    # not possible from greet; put it down at the last known grasp instead
                    gx, gy = rig.last_grasp
                    try:
                        await step("home go-to", rig.home.set_position(2)); await asyncio.sleep(4)
                        await put_back(rig, gx, gy)
                    except StepTimeout as e:
                        await say(f"put-back hung at {e}; opening the gripper where it is")
                        try:
                            await asyncio.wait_for(rig.gripper.open(), 5)
                        except Exception:
                            pass
                results.append(res)
                json.dump(results, open(LOG, "w"), indent=1)
                await say(f"=== {colour}: {'HELD' if ok else 'missed'} in {res['seconds']} s")
        await say(f"practice done: {sum(r['held'] for r in results[-rounds*len(colours):])} of {rounds*len(colours)} held")
        if live:
            await rig.home.set_position(2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--colours", nargs="+", default=["red", "green", "blue", "yellow"])
    a = ap.parse_args()
    asyncio.run(main(a.live, a.rounds, a.colours))
