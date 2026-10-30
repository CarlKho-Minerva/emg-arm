# emg-arm

## Fri 30 Oct, 14:15: submission form answers (form closes 3:00 PM) and the final controls

Form: https://forms.gle/T4Z5ah4M55uS7KDr8 (Google account needed; the email is recorded). Copy these in.

**Team Roster**
Carl Kho, solo. Somach Systems, Inc.

**Project Title**
Served With a Smile

**Project Abstract**
A robot arm on a table waits for a face. Pull your left cheek up and it finds a cube, picks it, and checks its own grip by moving sideways and looking at where the cube was. Pull your right cheek up and it feeds Skully, a skeleton skull, but only after two looks a second apart agree that the skull has not moved and is within reach. The problem it works on is the ordinary act of bringing food to a mouth when the arms cannot do it, and the part most feeding robots skip: knowing when not to act. Every command becomes one record in MongoDB, with what the camera saw, which checks passed, and how it ended.

**Who will use this solution? What is the expected business value?**
The person this is for has little or no use of their arms and still has their face. The clearest case is a cervical spinal cord injury: the cord is damaged in the neck, so the arms and hands are mostly paralysed while the muscles of the face keep working. Late muscular dystrophy, some strokes and some cerebral palsy look the same from the robot's side. For all of them, eating is the daily task most often handed to another person. Here the person keeps the decision (a half-smile) and the arm does the movement. The next input is the electrical signal of the facial muscles read from the skin, for a face that barely moves; the arm side does not change. In plain business terms: one caregiver hour returned per meal, on a stock arm a clinic or a home can run, with a record of every attempt that doubles as the audit trail and the training data for a safer version.

**What tech stack did you use?**
A UFactory xArm 6 with a two-finger gripper and a RealSense depth camera on the wrist, run through Viam (the motion planner with invisible walls, saved poses, the Python SDK). The face is read by the MacBook camera with MediaPipe, which turns a face into 52 numbers; a logistic regression trained on 20 seconds of Carl's own face turns those into five labels. The cube and the blue tape on Skully's nose are found by colour in OpenCV and placed in the room with the depth map and Viam's frame maths. MongoDB 8.0 through pymongo holds the episodes, 98 so far, including Thursday's picks, follower runs and handovers backfilled with their real timestamps; the database runs on the laptop because the Atlas clusters did not resolve from the venue network, and a JSONL file is written first so nothing is lost. The screen beside the arm is a plain Python HTTP page. Disclosure: the EMG board driver (silent-speech-cerelog) was written before the event and is not driving the arm today.

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
