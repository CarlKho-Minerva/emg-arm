# Served With a Smile

Viam x MongoDB Robot Hackathon, San Francisco, 29 to 30 October 2026. Solo entry by Carl Kho (Somach Systems).

A laptop camera reads a face. A robot arm does the handling. Pull your left cheek up and the arm fetches a cube
from the table and proves it has it. Pull your right cheek up and it feeds Skully, a skeleton skull with blue tape
on his nose. A smile lets go; two smiles send it home. Every command is one record in MongoDB Atlas.

## What has been shown, and what has not

Measured on the real arm, with the time it was seen:

- Cube pick, by colour from the wide table view, then a hover, a close look and a servo until the cube sits under
  the gripper to within 5 mm, then a staged descent closing at each step: 3 of 3 for green, blue and yellow
  (29 Oct 18:58). The grip is proven by a 100 mm sideways move and a second look at the spot.
- The follower that keeps a hand in the middle of the wrist camera's picture runs at about 20 Hz over the LAN;
  the gains were chosen on 40 recorded 30-second runs replayed in a simulator, held-out score -0.038 for the
  kept setting against -0.066 for the runner-up (29 Oct).
- Face control live: left cheek fetched the cube twice, a full smile let go, right cheek started a handover
  (29 Oct 19:46 to 19:49). Calibration on Carl's face: left cheek 0.33 against 0.01 at rest, right 0.21 against
  0.02, full smile 0.80, open mouth 0.55.
- MongoDB Atlas reachable from the venue by host name, ping 1.6 s; 98 episodes stored (30 Oct 14:11).

Not yet shown live: the feed into Skully's mouth (the geometry needs two ruler measurements; see below). Three
smiles and the open mouth do nothing in this build.

## How to run

Secrets (`VIAM_API_KEY_ID`, `VIAM_API_KEY`, `VIAM_MACHINE_ADDRESS`, `SERVE_MONGODB_URI`) live in
`~/.config/carl-life-os/.env`, never here. `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt pymongo`.

1. `open run_calibrate.command` (20 s, five faces), then `.venv/bin/python train_face.py` (prints held-out accuracy).
2. `open run_smile.command`: the face window; writes events to `data/face.log`, log in `data/smile.out`.
3. `.venv/bin/python serve.py --live --cube green`: the arm loop. Dry run without `--live`.
4. `open run_dash.command`: the screen, http://localhost:8790. `.venv/bin/python view.py` keeps its robot picture live.

| face | the arm |
|---|---|
| left cheek (L) | fetch: home, find the green cube, pick, prove the grip, greet |
| right cheek (R) | feed Skully: look from home, hover above him, re-centre, cube level into the mouth, let go, back, greet |
| full smile (B) | let go |
| two smiles (BB) | home, gripper open |

Before the first live feed: Skully faces the robot, 530 to 800 mm from the base (he was at 1227 mm on the first try
and the arm refused), blue cube off the table (same hue as the tape). Measure `TAPE_Z` (table to tape centre,
default 125 mm) and `MOUTH_BELOW` (tape to open-mouth centre, default 60 mm) in `skull.py`, then
`.venv/bin/python skull.py --live --probe` goes only to the pre-feed pose and comes back.

## Files

`smile.py` face events; `train_face.py` the personal classifier; `serve.py` events to arm behaviour; `pick.py` the
cube; `skull.py` Skully; `handme.py` hand and face following; `episodes.py` MongoDB; `dash.py` the screen;
`FORM_ANSWERS.md` the submission text; `DEMO.md` the run of show; `HISTORY.md` the day-by-day record.

Disclosure: the Cerelog EMG driver (`silent-speech-cerelog`) predates the event and is the next input; it does not
drive the arm today.
