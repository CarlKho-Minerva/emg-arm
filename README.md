# emg-arm

## Fri 30 Oct, 14:15: submission form answers (form closes 3:00 PM) and the final controls

Form: https://forms.gle/T4Z5ah4M55uS7KDr8 (Google account needed; the email is recorded). Copy these in.

**Team Roster**
Carl Kho (solo team). Somach Systems, Inc.

**Project Title**
Served With a Smile

**Project Abstract**
A face-controlled robot arm that fetches food and feeds Skully, a skeleton skull, using four facial gestures read
from a laptop camera: left cheek fetches a cube (vision pick with a self-check), right cheek feeds Skully, a smile
releases, two smiles stop. The robot feeds only after two looks a second apart agree the skull is still and in
reach, so it knows when not to act. Every command is stored as an episode (gesture, what the camera saw, the
safety checks, outcome, duration) in MongoDB, with a local mirror so nothing is lost when the network is down.

**Who will use this solution? What is the expected business value?**
People with little or no use of their arms who still control their face: cervical spinal cord injury, late-stage
muscular dystrophy, some stroke and cerebral palsy cases. Eating is the daily task most often handed to another
person; a feeding assistant that takes its orders from a half-smile gives the decision back to the person while
the arm does the movement. The same gesture-event interface takes a muscle-signal (EMG) input next, for faces
that barely move. Business value: a caregiver-hour per meal returned, in a form a clinic or home can run on a
stock arm; the episode log is the audit trail and the training data for safer, learned feeding behaviour.

**What tech stack did you use?**
UFactory xArm 6 with a two-finger gripper and Intel RealSense D435 on Viam (viam-server, motion planner with
collision walls, saved poses, Python SDK). MediaPipe face blendshapes on the MacBook camera plus a 20-second
personal logistic-regression classifier (numpy). OpenCV colour targets for the cube and the blue tape on the
skull, depth map plus Viam frame transforms for 3D. MongoDB 8.0 (pymongo) as the robot's episodic memory: one
document per command, 98 episodes already (Friday's live runs plus Thursday's picks, follower runs and handovers
backfilled with their original timestamps); a local mongod at the table because the Atlas clusters did not
resolve from the venue network, with a JSONL mirror written first so nothing is lost; a stdlib HTTP dashboard. Disclosure: the Cerelog EMG driver (silent-speech-cerelog) predates the
event and is the next input, not driving the arm today.

### The controls (final)

| face | the robot |
|---|---|
| left cheek up (L) | fetch: home, find the green cube, pick, prove the grip, back to greet |
| right cheek up (R) | feed Skully: look twice; if he held still (under 20 mm) and is 530 to 800 mm from the base, cube into his mouth, let go, back out, greet; else refuse and keep the cube |
| full smile (B) | let go |
| two smiles (BB) | home, gripper open (stop) |

Skully faces the robot, between two phones, blue cube off the table (same hue as the tape). Measure two numbers in
`skull.py` with a ruler before the first live feed: `TAPE_Z` (table to the tape's centre, default 125) and
`MOUTH_BELOW` (tape centre to the open mouth's centre, default 60). Then `skull.py --live --probe` (pre-feed pose
only), then `skull.py --live` once, then `serve.py --live`. Run of show, 3-minute script: DEMO.md.

Viam x MongoDB Robot Hackathon, SF, 29 to 30 Oct 2026. Solo entry by Carl Kho (Somach Systems).

A UFactory xArm 6 that finds Carl by camera, follows his arm, and (next) takes orders from his
face muscles through a Cerelog EMG board. No hands, no voice.

## Served With a Smile (Thu 29 Oct, 23:00)

The demo is face-controlled from the MacBook camera. `smile.py` reads the face model's numbers (left cheek,
right cheek, full smile, open mouth), calibrates to the person in 20 s (`--calibrate`), optionally trains a
personal classifier (`train_face.py`, logistic regression on the 52 face numbers), and writes counted groups
(L, R, B, BB, BBB, M) to `data/face.log`. `serve.py --live` turns them into arm behaviour: L fetch the cube,
M feed (follow the face, hold the cube in front of the mouth, take it back), R palm handover, B let go, BBB
home, BB nudge mode. `dash.py` is the screen beside the arm. Verified live Thursday: L fetch (twice), B let go,
R handover; not yet verified live: M feed. Run-of-show in DEMO.md.

## Status, end of day 1 (29 Oct, 18:58)

**The demo works, verified:** `.venv/bin/python pick.py --live --cube green` goes to greet, looks at the
table from home, finds the green cube by colour (cube-shaped blob, on the table, nearest the arm), hovers,
servos until the cube is centred under the gripper, descends in stages (floor 70 mm) closing at each until
the fingers settle on a cube (reads 120 to 700), then proves the lift by shifting 10 cm sideways and
looking (the cube moved with the gripper: 19 mm from it, 110 mm from its spot), goes to greet, follows
Carl's left hand with the tuned follower (colour frames only), and opens when a hand fills the picture
(wrist-to-knuckle over 0.25 of the width) for 0.7 s. Last run 18:58: picked at fingers 290, released
2 s after the hand came to the cube. Red is not a cube (two triangles); green, blue, yellow pick 3/3.

What made it work, in order: measuring the camera-to-gripper offset (+25, -14 mm) by letting the gripper
place the cube and the camera look; a truthful hold check (settled fingers plus the sideways-shift test;
the first check read the fingers too early and reported 12 false holds); the 70 mm floor set by eye.

## What was measured today

Working, measured on the real arm:

- `probe.py`: read-only; joints, hand position, resources.
- `jog.py`: nudge one joint, open/close the gripper, stop.
- `savepose.py <name>`: save the arm's pose as a one-button block (`home` = wide view of the
  table, `greet` = reach position); `--go` sends it there, `--deg a,b,c,d,e,f` sets it from numbers.
- `follow.py --live`: the camera keeps Carl's hand (or, with `--arm left`, his whole arm by the
  body) in the middle of its picture. P + D control with low-pass filters on the hand position
  and on the commanded step, streamed joint targets (a new target cancels the one in flight, so
  the arm does not stop between steps), coasting along the hand's last direction when it drops
  out, a slow search sweep when it is gone. Loop runs at about 20 Hz on a Mac over the LAN.
  Every run is recorded to `data/follow/<run>/` (one CSV row per frame, a picture per frame);
  `./stitch.sh data/follow/<run>` makes an mp4.
- `handpos.py`: the hand's position in the arm's own coordinates (pixel + depth, then Viam's
  frame maths). Checked: a hand on the table reads 1 to 2 cm above it.
- `reach.py --live`: lean the gripper toward the hand through the motion planner, so the walls
  count. Dry-run verified; live run not yet done.
- `addwall.py`: an invisible wall (`wall-carl`) between the arm and Carl's laptop, 95 cm from
  the base; planned moves go around it, direct joint moves do not.
- `camgrab.py`: colour and depth frames. The depth map is 1.9 MB and costs 350 ms a frame;
  ask for it only when needed.

Joint map from the greet pose, measured by wiggling each joint 5 degrees and watching the
picture: waist (0) pans cleanly, elbow (2) tilts cleanly, forearm (3) only rolls, wrist (4, 5)
mix the two. The follower pans with the waist and tilts with the elbow.

## Next

1. Body tracking (`tracker.py`, MediaPipe pose) so left/right comes from the body and the
   target survives the hand leaving the frame.
2. The demo loop: home, wake, greet, follow, intent, act, home.
3. EMG intent from the Cerelog board (`silent-speech-cerelog`, predates the event): jaw
   clench = grab, lip purse = come, mouth open = release.
4. MongoDB Atlas: store EMG windows with labels, classify by nearest neighbours, log every
   move and outcome.

## Setup

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Secrets (`VIAM_API_KEY_ID`, `VIAM_API_KEY`, `VIAM_MACHINE_ADDRESS`) live in
`~/.config/carl-life-os/.env`, never in this repo.

- `HANDOFF.md`: the full plan, hardware facts, pad placement
- `lead-figure.html`: how to make the four EMG leads
