# Viam x MongoDB Robot Hackathon: handoff (written 2026-10-29 ~12:10 PT from the mac128 thread)

## Task

Help Carl build and demo his hackathon entry, working solo (no team). Carl will add MCP servers and the
event's slides to this thread himself; wait for him to say what he has added before assuming a tool exists.
Answer in his house voice (the `research-writing` skill, Experimental History, no em dashes).

## The event (source: https://luma.com/ta40197s and the Viam emails in kho@uni.minerva.edu)

- Robot Hackathon: Harnessing Physical AI & Data. Viam with MongoDB. Robotics Center of Silicon Valley,
  90 Welsh St, SF. Carl is confirmed.
- Thu 29 Oct: lunch 1:00, dinner 6:00, **hard close 9:00 pm** (robots off, no overnight).
- Fri 30 Oct: doors 9:00, hacking 9:30, lunch 12:30, **demos 3:30**, awards 5:00, ends 6:00.
- Provided: UFactory xArm or Universal Robots arm with a two-finger gripper, a Linux compute box,
  a RealSense D435 or Orbbec Astra 2 on the arm, webcams, a free Viam account, a MongoDB Atlas cluster.
- Judges weigh use of MongoDB Atlas (optional). Prizes for three teams: Atlas + Viam credits
  ($5k + $5k gold) and an SO-101 arm kit per winning member. No cash.
- The page says teams of 2 to 3; Carl is solo. Whether code written before the event is allowed is not
  stated: disclose at the demo that the EMG driver predates the event.
- Open in Dia (Fun profile) on this Mac: the Viam machine page `sfhack9`
  (https://app.viam.com/machine/72648a81-5cbf-4d14-9446-a7ec4a8d51a8/control), the slides
  "[Code That Moves] Viam Overview" (Google Slides), and "Welcome to Viam SF Hackathon" in Gmail.

## What Carl wants to build

Somach's demo: **control the robot arm with his face, no hands and no voice**, using surface EMG (the
electrical signal muscles give off) from his Cerelog board.

1. First, one switch: **jaw clench closes the gripper.** A clench is a large signal; a threshold on the
   signal's envelope is enough, no trained classifier.
2. Then the face gestures from his runbook: lip purse, mouth open, jaw clench.
3. Stretch, Friday morning: silently mouthed UP, DOWN, LEFT, RIGHT moving the arm. Honest risk: these may
   not separate (see the board limits below).
4. MongoDB, for the judges: store each short window of EMG with its label in Atlas; classify a new window
   by vector search (nearest stored windows, take their label), so a one-minute calibration becomes the
   classifier; log every arm move and whether it worked; show a live accuracy chart.

Backup if EMG fails: "a robot that checks its own work" (after each grasp, camera + gripper width decide
pass/fail, logged to Atlas; before the next grasp, look up past tries on similar objects and reuse what
worked; live success-rate chart). Needs none of Carl's hardware.

## Hardware facts (verified, with sources)

- Cerelog ESP-EEG **V1** (CH340 USB serial). On a Mac it shows as `usbserial-*` or `wchusbserial-*`
  (vendor 1A86). It has only ever been linked on the Zenbook (COM5, 25 Oct); **never yet on a Mac with the
  real board**. Driver: `~/CODELocalProjects/silent-speech-cerelog` (copied here today, venv built,
  `python -m ssc link --fake` passes on this Mac).
  - `.venv/bin/python -m ssc ports` then `.venv/bin/python -m ssc link --port /dev/cu.usbserial-XXXX`
    must print LINK OK. Ignore a Samsung phone line in `ports`: the tool mislabels it V2.
  - Commands: ports, link, schedule, record, qc, bench, analyze. **There is no live streaming classifier
    yet**: the arm control loop has to be written (read packets from `ssc/cerelog.py`, envelope, threshold
    or nearest-neighbour, send to Viam).
  - 250 samples/s; the chip's filter passes only up to about 65 Hz. Bench noise floor about 4 µV, 22 times
    the rating (Study 00, 25 Oct). A clench towers over that; mouthed words may not.
  - Referential montage: every channel is measured against SRB1. On V1, Bias listens to all 8 inputs, so
    an unplugged channel makes every channel noisy. Fix used on the bench (proven): tie unused + pins to
    the Bias node. Untested idea: a jumper cap across each unused channel's + and − pins.
  - `ssc record` refuses to run with the charger plugged in, and aborts if it goes in (safety). Laptop on
    battery whenever pads are on him.
  - On 25 Oct a grounded foil sheet cut the air-borne hum about thirty-fold. Motor noise is magnetic and
    foil does not stop it: keep Carl and the laptop a couple of metres from the arm.
- Header (USB end of the board on the left): rows GND, GND, SRB2/SRB1, Bias/Bias, then channels 1 to 8;
  on each channel row the left pin is +, the right is −. SRB1 is the right pin of its row; SRB2 stays empty.

| Lead | Pad on Carl | Board pin |
|---|---|---|
| 1 | chin, midline, 1 cm below the lower-lip crease | channel 1 + |
| 2 | right side, just behind and below the earlobe | channel 2 + |
| REF | bony bump behind the LEFT ear (mastoid) | SRB1 |
| BIAS | LEFT collarbone, on bone | either Bias pin |

Skin prep: alcohol wipe, rub until slightly pink, dry fully, press the pad 10 s, tape each lead to the
collar. For clench, the cheek muscle bulge (masseter) is the strongest spot.

- **Leads are not made yet.** Carl brought the AD8232 snap cables, Dupont jumpers and (maybe) his Pinecil.
  A lead = cut the snap wire off its plug, strip 1 cm, wind the copper tightly around a female-to-male
  jumper's male pin (that alone connects: metal on metal), solder through it if an iron is at hand, tape
  over. Female end onto the board pin. He knows how to solder.
- Interactive figure of exactly this (pad, snap, joint in 4 steps, pins): `lead-figure.html` in this
  folder. Built with the anatomy skill; geometry audit passed.

## This Mac (forge-m5-48) as of ~12:05

- Dia restarted with `--remote-debugging-port=9222 --enable-applescript-javascript` at Carl's request; port
  answers; 12 tabs restored. The app-routes skill (`~/.claude/skills/app-routes`) and its lifeos tools
  (errand, cdp_page, elements, replay, dia_*, proton_fill, submit_check, ascript, dia_port_guard, cdp_dia)
  were synced from mac128 (old versions kept as `*.bak-20261029-pre-approutes`). `node errand.mjs doctor`:
  Dia tab route ok, Dia AppleScript flag ok. `elements.mjs els` read 72 elements off a Viam docs tab.
- Down here and not needed today: peekaboo (not installed), ascript.py (not granted on this Mac), MCP hub.
- `dia_port_guard.py` is copied but NOT installed as a launchd job here, so if Dia is quit and reopened
  from the Dock it comes back without the port. Relaunch it with both flags (quit Dia, then
  `open -a /Applications/Dia.app --args --remote-debugging-port=9222 --enable-applescript-javascript`).

## Constraints

- Carl's global rules apply (`~/.claude/CLAUDE.md`): no messages to people without his approval of the
  exact text, no payments, look at the real result before saying done, absolute paths for files.
- One board on his body at a time; on-body leads go straight from pad to header, no breadboard on the body.
- Do not give him a command you have not run on this Mac.

## sEMG status (29 Oct evening)

Written 29 Oct ~15:45 PT by the sEMG thread. Board only; the arm, Viam config, follow.py and coach.py were not touched.

- **The board links on this Mac.** Port `/dev/cu.usbserial-110` (vendor 1A86, V1). `ssc link --port /dev/cu.usbserial-110`
  printed LINK OK at 22:39 UTC: 2520 packets in 10 s, 249.73 samples/s, 0 dropped, 0 checksum failures. Laptop was on battery.
  Every connect on this Mac goes through the driver's reset pulse ("no data at 9600 ... no data at 115200 ... pulsing reset"),
  which takes about 12 s and is normal here. The port name encodes the USB socket: plug into a different socket and rerun `ssc ports`.
- **Driver fix, committed in silent-speech-cerelog as cbd5b11.** The first three link runs said LINK NOT OK with 236-238 "dropped"
  samples. Cause: 21-28 packets from the 9600-baud phase survive the buffer flush and sit in front of the real stream with a clock
  about 4 s ahead. The drop counter read the backwards clock step as lost samples, every block's QC would have printed FAIL on
  "dropped samples" and "sample rate", and the cue-to-sample search (a binary search that assumes a rising clock) could have put
  an early cue on those packets. `cerelog.trim_stale_prefix()` now discards them (only when the step is inside the first 500
  packets) and the count is printed as "N pre-handshake packets discarded". Tests 14/14, fake 3-trial record passes, real link passes.
  The Zenbook on 25 Oct never had this prefix (0 backwards steps), so it is a macOS CH340 buffering quirk.
- **Noise floor: not measurable yet.** With nothing wired, all 8 channels sit at the bottom of the range (-187.5 mV, railed 100%),
  as an open input does. A floor appears only once an input is tied to SRB1 and Bias. The reference figure stays the 25 Oct bench
  floor, about 4 uV, 22x the chip's rating.
- **Next (Carl, on his body):** 4 leads per the pad table above; channels 3 to 8 "+" pins tied to Bias (the proven bench fix: a
  breadboard rail as a Bias bus is fine, no body lead passes through it); charger out; then
  `cd ~/CODELocalProjects/silent-speech-cerelog && .venv/bin/python -m ssc record --port /dev/cu.usbserial-110 --day sat --block 1 --sites 1=mental 2=post_lobule_R --ref mastoid_L --bias clavicle_L --limit 12`
  (first 12 cues of the schedule: 3 rest, 2 jaw_clench, 2 lip_purse, 5 others; about a minute). The "activation, dB above rest"
  table must show jaw_clench as the largest row on ch1, well above lip_purse, and the QC line "jaw clench" must be ok (3 dB is the
  floor of the check; 10 dB or more is what a demo threshold needs). If it is under 3 dB the pads are off the muscle or SRB1/Bias is open.

## sEMG status (29 Oct evening)

Written by the sEMG child thread on this Mac (forge-m5-48). Times are PT.

- **15:37 Port found.** The V1 board is `/dev/cu.usbserial-110` (CH340, vid 1A86). `ssc ports` lists it as V1.
- **15:44 LINK OK.** First `ssc link` (10 s) failed: the board had just been reset by the connect
  routine and 208 packets looked dropped. Second run (20 s): 5035 packets, 249.71 SPS, 0 dropped,
  0 checksum failures, 24 pre-handshake packets discarded, latency p95 5.8 ms. Saved under
  `silent-speech-cerelog/data/bench/20261029T224405Z_link`. Lesson: if the first link check fails
  right after plugging in, run it once more before touching anything.
- **All 8 channels railed with nothing on the pins.** Expected on V1 (floating inputs). No noise
  floor can be measured until the inputs are tied to something. Plan: six female-female jumpers
  across + and - of channels 3 to 8 (ties each unused + to SRB1 internally, since MISC1 SRB1=1
  routes every - input to SRB1), then all four lead snaps on one coin for a 30 s
  `ssc bench noise --condition short`. The jumper trick is the "untested idea" above; tonight's
  floor test is its test. Not yet run: waiting on Carl's leads.
- **Live clench detector written and tested on the fake board.** `python -m ssc live --port
  /dev/cu.usbserial-110 --channels 1 2` (new file `ssc/live.py`, subcommand in `ssc/__main__.py`).
  Causal 20 to 65 Hz band-pass + 60 Hz notch, 100 ms RMS envelope, 5 s rest calibration,
  threshold = 4 x rest (12 dB) and never under 5 uV, 120 ms hold, release at 0.6 x threshold.
  Refuses to arm on a railed channel. Battery rule inherited from `record` (`--off-body` skips it
  only when no pad is on a person). `--fake --seconds 14` gave 3 simulated clenches, 3 CLENCH and
  3 RELEASE lines, none at rest. For the arm thread: `from ssc import live; for ev in
  live.events(args): ...` yields dicts with `state` in ARMED / CLENCH / RELEASE.
- Not yet done: noise floor on the real board, the 12-trial on-skin block, the activation table,
  the live loop on the real board. Each gets its own dated line below when it happens.
- **16:40 n1 lab page published.** Entry `silent-speech-clench-night` in the n1 lab (OpenHost app
  `n1`, built from `n1-lab/private/build_content.py` on the 128, shipped with `tools/push.sh push`).
  Source page: `silent-speech-cerelog/protocol/study01/CLENCH_NIGHT.md` (commit 306a3ad, pushed to
  GitHub, pulled on the 128). `lead-figure.html` is attached to the entry as a copy
  (`n1-lab/private/attachments/ssc_lead_figure.html`); the hackathon copy is untouched. Results get
  appended to the page as dated lines, then re-pushed the same way.
- **18:59 First lead conducts; floor 0.18 uV.** Channel 2 lead (twisted, taped, unsoldered) pinched
  against the bare SRB1 + Bias jumper ends: 0.18 uV rms 0.5-50 Hz, 0% railed, offset 0.08 mV
  (`data/bench/20261030T015933Z_short`). The 4 uV Study 00 floor was the Zenbook, not the board.
  Lesson for the other three leads: a lead hanging in the air reads full scale or swings, same as
  a bad joint; test joints only by shorting them to SRB1 + Bias. Carl is soldering ch2 and making
  leads 1, REF, BIAS next; then one pinch test of all four.
