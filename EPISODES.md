# Episode collection: 30 recorded runs of the follower

The simulator (`sim.py`) ranks controller settings by replaying recorded hand movements. It is only
as good as the recordings, and the four from the afternoon of 29 Oct were all "Carl waves a bit".
This is the plan for thirty that cover what actually goes wrong. Each episode is 30 seconds, one
command, one label. Twenty for fitting and tuning, ten held out for scoring, chosen by the last
digit of the folder name (0 to 2 held out) so nobody picks.

## Setup, once

- Arm at `greet`, Carl in the chair, laptop pulled back, a hand on the red button for every live run.
- Same lighting and seat for all thirty; change one thing per episode, named by the tag.
- Start each with the arm looking at the chair: `.venv/bin/python savepose.py greet --go`.

## The command

```
.venv/bin/python follow.py --live --arm left --hand watch --lost-go greet --speed 60 --accel 200 --seconds 30 --tag <TAG>
```

It stops itself after 30 s and restores the arm's defaults. The recording lands in
`data/follow/<time>-live-<TAG>/` with a CSV row per frame and a picture per frame.

## The thirty, five of each

| Tag | What Carl does for 30 s | What it tests |
|---|---|---|
| `still` | left hand held still at centre, then still at the left edge, then the right edge, 10 s each | jitter at rest; the dead zone |
| `slow` | left hand sweeps left to right and back, slowly, about 5 s per sweep | tracking at the hand's usual speed |
| `fast` | same sweep, fast, about 1 s per sweep | the burst speeds the sim saw (54 deg/s) |
| `cross` | both hands up; left hand crosses over the right and back, three times | hand swaps (the 1229 deg/s jumps) |
| `exit` | left hand drifts out of the picture to the left, stays out 3 s, comes back; then the same to the right | coasting, search, the elbow guess |
| `rest` | left hand lies on the keyboard and types; right hand gestures | does it stay on the left hand when the right one is busier |

Five episodes per tag gives 30. Do the `still` ones first (they calibrate the dead zone) and
`cross` last (they are the hardest on the arm).

## After each five

```
.venv/bin/python sim.py          # the reward column: higher is better, 0 is perfect
.venv/bin/python fit_arm.py      # the arm model the sim should use: speed, accel, delay
```

If `fit_arm.py` gives a different speed/accel/delay from what `sim.py` assumes, change the
constants at the top of `sim.py` to the fitted ones before trusting its rankings.

## What counts as progress

One number: the held-out reward from `sim.py` on the ten held-out episodes. A change to the
controller stays if that number goes up; it is reverted if it does not, whatever it felt like.
Say the number with its spread across the ten.

## What not to do

- Do not tune on the held-out ten. Ever.
- Do not change lighting, seat or camera pose between episodes of the same tag.
- Do not retake an episode because it "looked bad": a bad episode is data.
