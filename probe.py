"""Read-only look at the arm: resources, the six joint angles, where the hand is.

    .venv/bin/python probe.py
"""
import asyncio

from viamarm import arm_of, connect, gripper_of


async def main():
    async with await connect() as machine:
        names = sorted(f"{r.subtype}/{r.name}" for r in machine.resource_names)
        print("resources:", ", ".join(names))

        arm = arm_of(machine)
        joints = await arm.get_joint_positions()
        for i, deg in enumerate(joints.values):
            print(f"joint {i}: {deg:8.2f} deg")

        pose = await arm.get_end_position()
        print(f"hand at x={pose.x:.1f} y={pose.y:.1f} z={pose.z:.1f} mm, "
              f"facing o=({pose.o_x:.2f}, {pose.o_y:.2f}, {pose.o_z:.2f}) theta={pose.theta:.1f}")
        print("arm moving:", await arm.is_moving())

        gripper = gripper_of(machine)
        print("gripper moving:", await gripper.is_moving())


if __name__ == "__main__":
    asyncio.run(main())
