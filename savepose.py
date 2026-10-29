"""Save the arm's current pose as a named one-button block (like "home").

    .venv/bin/python savepose.py greet      # creates or overwrites the block "greet"
    .venv/bin/python savepose.py greet --go # later: send the arm back to it

Same program as the organisers' "home" block (vmodutils/arm-position-saver); joints are
stored in radians because that is what the block expects.
"""
import asyncio, math, os, sys
from viam.app.viam_client import ViamClient
from viam.components.switch import Switch
from viam.rpc.dial import DialOptions, Credentials
from viamarm import arm_of, connect

MACHINE = "72648a81-5cbf-4d14-9446-a7ec4a8d51a8"

async def save(name):
    async with await connect() as machine:
        deg = list((await arm_of(machine).get_joint_positions()).values)
    rad = [math.radians(d) for d in deg]
    vc = await ViamClient.create_from_dial_options(DialOptions(
        credentials=Credentials(type="api-key", payload=os.environ["VIAM_API_KEY"]),
        auth_entity=os.environ["VIAM_API_KEY_ID"]))
    try:
        part = (await vc.app_client.get_robot_parts(MACHINE))[0]
        cfg = dict(part.robot_config)
        comps = [c for c in cfg.get("components", []) if c["name"] != name]
        home = next(c for c in cfg["components"] if c["name"] == "home")
        block = {k: home[k] for k in ("api", "model") if k in home}
        block.update(name=name, attributes={"arm": "arm", "constraints": {}, "extra": {}, "joints": rad,
                                            "motion": "", "vision_services": []})
        comps.append(block)
        cfg["components"] = comps
        await vc.app_client.update_robot_part(part.id, part.name, cfg)
    finally:
        vc.close()
    print(f"saved '{name}':", ", ".join(f"{d:.1f}" for d in deg), "deg")

async def go(name):
    async with await connect() as machine:
        sw = Switch.from_robot(machine, name)
        await sw.set_position(2)
        print(f"'{name}': go-to sent")

if __name__ == "__main__":
    n = sys.argv[1]
    asyncio.run(go(n) if "--go" in sys.argv else save(n))
