"""One connection to the hackathon arm (sfhack9), shared by every script here.

Secrets come from ~/.config/carl-life-os/.env (VIAM_API_KEY_ID, VIAM_API_KEY,
VIAM_MACHINE_ADDRESS). Nothing secret lives in this repo.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.robot.client import RobotClient

load_dotenv(Path.home() / ".config/carl-life-os/.env")
load_dotenv(Path(__file__).resolve().parent / ".env")   # on the robot's own box the secrets sit next to the code

# On the box itself, talk to viam-server on localhost: no network hop at all.
ADDRESS = os.environ["VIAM_MACHINE_ADDRESS"]


CONNECT_S = 40.0   # a dial through Viam's cloud took 2 to 5 s on 29 and 30 Oct; longer than this is a fault, not patience


async def connect() -> RobotClient:
    """Dial the robot, saying so, with a time limit. On 30 Oct 14:24 a silent dial hung long enough that Carl
    pressed Ctrl-C believing the program was stuck; now it prints while it waits and fails loudly instead."""
    import asyncio, sys, time
    opts = RobotClient.Options.with_api_key(
        api_key=os.environ["VIAM_API_KEY"],
        api_key_id=os.environ["VIAM_API_KEY_ID"],
    )
    print(f"connecting to the robot ({ADDRESS.split('.')[0]})...", flush=True)
    t0 = time.time()
    try:
        machine = await asyncio.wait_for(RobotClient.at_address(ADDRESS, opts), CONNECT_S)
    except asyncio.TimeoutError:
        sys.exit(f"no connection to the robot after {CONNECT_S:.0f} s. Is the box on and on the same network? Try: .venv/bin/python probe.py")
    print(f"connected in {time.time() - t0:.1f} s", flush=True)
    return machine


def arm_of(machine: RobotClient) -> Arm:
    return Arm.from_robot(machine, "arm")


def gripper_of(machine: RobotClient) -> Gripper:
    return Gripper.from_robot(machine, "gripper")
