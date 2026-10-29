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

ADDRESS = os.environ["VIAM_MACHINE_ADDRESS"]


async def connect() -> RobotClient:
    opts = RobotClient.Options.with_api_key(
        api_key=os.environ["VIAM_API_KEY"],
        api_key_id=os.environ["VIAM_API_KEY_ID"],
    )
    return await RobotClient.at_address(ADDRESS, opts)


def arm_of(machine: RobotClient) -> Arm:
    return Arm.from_robot(machine, "arm")


def gripper_of(machine: RobotClient) -> Gripper:
    return Gripper.from_robot(machine, "gripper")
