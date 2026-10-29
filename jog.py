"""Nudge one joint, or open/close the gripper. The smallest possible way to feel the arm.

    .venv/bin/python jog.py joint 5 +5      # joint 5, five degrees more
    .venv/bin/python jog.py joint 0 -10     # joint 0, ten degrees less (cap: 20 per call)
    .venv/bin/python jog.py grip close      # or: grip open
    .venv/bin/python jog.py stop

A joint move goes straight to the angle (no obstacle planning), so stand clear and know
where the E-stop is. One call moves one joint by at most MAX_STEP degrees.
"""
import asyncio
import sys

from viam.proto.component.arm import JointPositions

from viamarm import arm_of, connect, gripper_of

MAX_STEP = 20.0


async def jog_joint(index: int, delta: float):
    if abs(delta) > MAX_STEP:
        sys.exit(f"refusing: {delta} deg is more than the {MAX_STEP} deg cap")
    async with await connect() as machine:
        arm = arm_of(machine)
        before = list((await arm.get_joint_positions()).values)
        if not 0 <= index < len(before):
            sys.exit(f"no joint {index}; the arm has {len(before)} (0 to {len(before) - 1})")
        target = before[:]
        target[index] += delta
        print(f"joint {index}: {before[index]:.2f} -> {target[index]:.2f} deg")
        await arm.move_to_joint_positions(JointPositions(values=target))
        after = list((await arm.get_joint_positions()).values)
        print(f"now at {after[index]:.2f} deg (moved {after[index] - before[index]:+.2f})")


async def grip(action: str):
    async with await connect() as machine:
        g = gripper_of(machine)
        if action == "open":
            await g.open()
            print("gripper: open")
        elif action == "close":
            held = await g.grab()
            print("gripper: closed, holding something" if held else "gripper: closed on nothing")
        else:
            sys.exit("grip takes open or close")


async def stop():
    async with await connect() as machine:
        await arm_of(machine).stop()
        print("arm: stop sent")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["joint"] and len(a) == 3:
        asyncio.run(jog_joint(int(a[1]), float(a[2])))
    elif a[:1] == ["grip"] and len(a) == 2:
        asyncio.run(grip(a[1]))
    elif a == ["stop"]:
        asyncio.run(stop())
    else:
        sys.exit(__doc__)
