# emg-arm

Viam x MongoDB Robot Hackathon, SF, 29 to 30 Oct 2026. Solo entry by Carl Kho (Somach Systems).

A UFactory xArm 6 that finds Carl by camera, follows his arm, and (next) takes orders from his
face muscles through a Cerelog EMG board. No hands, no voice.

## Status, end of day 1 (29 Oct)

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
