"""Add an invisible wall between the arm and Carl's side of the table (x = 950 mm and beyond).

Same model as the walls the organisers put in (erh:vmodutils:obstacle). The planner refuses
to cross it; direct joint moves ignore it, so follow.py's joint nudges are not covered.

    .venv/bin/python addwall.py          # adds "wall-carl" if missing; prints the obstacle list
"""
import asyncio, os
from viam.app.viam_client import ViamClient
from viam.rpc.dial import DialOptions, Credentials
from viamarm import connect  # loads .env

MACHINE = "72648a81-5cbf-4d14-9446-a7ec4a8d51a8"
WALL = {
    "name": "wall-carl", "api": "rdk:component:gripper", "model": "erh:vmodutils:obstacle",
    "attributes": {"geometries": [{"type": "box", "x": 50.0, "y": 1500.0, "z": 1024.0, "label": "wall-carl"}]},
    "frame": {"parent": "world", "translation": {"x": 950.0, "y": 0.0, "z": 488.0},
              "orientation": {"type": "ov_degrees", "value": {"x": 0.0, "y": 0.0, "z": 1.0, "th": 0.0}}},
}

async def main():
    opts = DialOptions(credentials=Credentials(type="api-key", payload=os.environ["VIAM_API_KEY"]),
                       auth_entity=os.environ["VIAM_API_KEY_ID"])
    vc = await ViamClient.create_from_dial_options(opts)
    try:
        part = (await vc.app_client.get_robot_parts(MACHINE))[0]
        cfg = dict(part.robot_config)
        comps = list(cfg.get("components", []))
        tmpl = next(c for c in comps if c["name"] == "wall-back")
        WALL["api"] = tmpl.get("api", WALL["api"])
        if not any(c["name"] == "wall-carl" for c in comps):
            comps.append(WALL)
            cfg["components"] = comps
            await vc.app_client.update_robot_part(part.id, part.name, cfg)
            print("added wall-carl")
        else:
            print("wall-carl already there")
        part = (await vc.app_client.get_robot_parts(MACHINE))[0]
        for c in part.robot_config["components"]:
            if c["model"].endswith("obstacle"):
                t = c["frame"]["translation"]; print(f"  {c['name']}: at x={t['x']} y={t['y']} z={t['z']}")
    finally:
        vc.close()

asyncio.run(main())
