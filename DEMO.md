# Demo: a robot arm that takes orders from the face

Viam x MongoDB Robot Hackathon, Robotics Center of Silicon Valley, 90 Welsh St, SF.
Demos Friday 30 October 2026 at 3:30 pm, "demos and drinks". Judges walk from table to
table; there is no pitch slot, so the robot has to be running when they arrive.

Solo entry by Carl Kho (Somach Systems). The event page asks for teams of 2 to 3. Say so
at the table, first sentence, before anyone has to ask.

One disclosure, said unprompted: the EMG driver (`silent-speech-cerelog`) was written
before the event. Everything that touches the arm, the camera and Atlas was written here.

## #rightproblem

On a Mac, people already let the machine watch and listen for them. Voice-to-text takes
dictation. Screen reading describes what is on the display. Autocomplete finishes the
sentence. Nobody calls this a robot, but it is the same shape: a sensor, a guess at what
you meant, an action. This demo takes that shape off the screen and puts it on a table.

The body is the input. A camera finds Carl and keeps his arm in the middle of its picture.
Four sticky pads on his chin and jaw read surface EMG, the small electrical signal a muscle
gives off when it tightens. A jaw clench is a command. No hands, no voice.

Why not a cup-fetching demo. A robot that fetches a cup on a voice command is a solved
party trick, and it fails the people who would need it most. Someone who cannot speak, or
cannot use their hands, gets nothing from "hey robot, grab the cup". A factory floor at
90 decibels gets nothing from it either. A sterile room where gloved hands must not touch
a keyboard gets nothing from it. A face muscle still works in all three places. The claim
of the demo is narrow: a muscle twitch, read off the skin, is enough to tell an arm what
you want, and the arm can already be looking at you when you ask.

## #evidencebased

What has been measured on the real arm, as of the end of Thursday (source: README.md).

- The follow loop runs at about 20 Hz on a Mac over the local network. In words: the
  camera looks, the arm nudges, about twenty times a second.
- The joint map was measured by wiggling each joint 5 degrees and watching the picture.
  Waist (joint 0) pans cleanly, elbow (joint 2) tilts cleanly, forearm (joint 3) only
  rolls, the two wrist joints (4 and 5) mix pan and tilt. The follower uses the waist to
  pan and the elbow to tilt, and leaves the rest alone.
- Hand position in the arm's own coordinates: a hand resting on the table reads 1 to 2 cm
  above it. That is the error of the pixel-plus-depth path before any planning.
- When the hand drops out of the picture the arm coasts along the hand's last direction,
  then sweeps slowly to search. Every run writes one CSV row per frame to
  `data/follow/<run>/`, so the count of frames with no hand is read off the file, not
  guessed. Quote the number from Friday morning's run, not a remembered one.
- Saved poses `home` (wide view of the table) and `greet` (reach position) go where they
  were saved. An invisible wall sits 95 cm from the base between the arm and the laptop;
  planned moves route around it, direct joint moves do not.
- The depth map costs 350 ms a frame at 1.9 MB, so it is fetched only when needed.

What has not been shown yet, in the same plain terms.

- `reach.py --live` (leaning the gripper toward the hand through the planner) has passed a
  dry run only. No live run yet.
- The Cerelog EMG board has only ever been linked on a Windows laptop (25 October). It has
  never been linked to the real board on a Mac. The fake-board link test passes on this
  Mac; the real one has not been tried. The leads were not made as of Thursday noon.
- There is no live EMG classifier. The driver records and benches; the loop that reads
  packets, takes an envelope, thresholds it and sends a command to Viam is still to write.
- Nothing has been written to MongoDB Atlas yet.
- Bench noise on the board is about 4 microvolts, 22 times its rating (Study 00, 25
  October). A clench is far larger than that. Mouthed words may not be, so mouthed
  directions are a stretch goal, not a promise.

If a judge asks for the status of any item, give the line above, not a softer version.

## #breakitdown

Run of show, under 3 minutes. Carl stands about two metres from the arm (motor noise is
magnetic and foil does not stop it). Laptop on battery whenever pads are on him.

1. Home (0:00). The arm sits in the `home` pose, wide view of the table. Screen shows the
   camera feed and the live Atlas panel: stored EMG windows, moves logged, accuracy.
2. Wake (0:10). Carl steps into frame. The tracker finds his body and picks the left or
   right arm from the pose, not from the hand alone.
3. Greet (0:20). The arm moves to the `greet` pose through the planner, so the wall
   between the arm and the laptop counts.
4. Follow (0:30). Carl moves his arm left, right, up, down. The camera keeps it centred:
   waist pans, elbow tilts. He steps out of frame on purpose; the arm coasts, then sweeps.
   He steps back in; it reacquires. The CSV is writing the whole time.
5. Intent (1:10). Carl clenches his jaw. The EMG envelope crosses the threshold, the
   screen shows the window and its nearest stored neighbours, the gripper closes.
   Fallback, said out loud if the board is not linked: a key press on the laptop stands in
   for the clench. Say "this key is the clench; the EMG path is not live today". Do not
   let a judge think the key was the muscle.
6. Act (1:40). With the gripper closed on the object, the arm lifts and places it at a
   second saved pose. Lip purse (come) and mouth open (release) are the second and third
   gestures if they separate by Friday; if not, only clench runs.
7. Home (2:20). The arm returns to `home`. The Atlas panel updates: one more move, its
   outcome (pass or fail, from gripper width and camera), and the running accuracy.

The Atlas piece, in one paragraph. Each short window of EMG is stored in Atlas with its
label (clench, purse, open, rest) during a one-minute calibration. A new window is
classified by finding the nearest stored windows and taking their label, so calibration is
the classifier and there is no training step. Every arm move is logged with its command,
timestamp and outcome. The panel shows live accuracy: moves that did what the label said,
divided by moves attempted. This is the part the judges are told to weigh, and it is also
the part that is not built as of Thursday night. If it is not built by 3:30 on Friday, the
panel is replaced by the raw CSV and that is said plainly.

## Sixty seconds, spoken over the running robot

I am Carl, solo, Somach Systems. The team page says two to three; it is one.

On your Mac you already let the machine watch and listen for you: dictation, screen
reading, autocomplete. I wanted the same assist in the room, with the body as the input.

The camera has found me and is keeping my arm in the middle of its picture. That loop runs
about twenty times a second. If I step out, it coasts, then searches. (Step out, step in.)

The pads on my chin and jaw read the electrical signal my muscles make. A jaw clench is a
command. No hands, no voice. Someone who cannot speak or cannot use their hands can still
do this. So can someone on a loud floor or in a sterile room. (Clench. Gripper closes.)

Every clench window goes into MongoDB Atlas with its label. A new one is matched to its
nearest neighbours, so a one-minute calibration is the whole classifier. Every move is
logged with its outcome, and that number on the screen is the running accuracy.

One disclosure: the EMG driver is mine from before the event. The arm, the camera and the
Atlas side were built here. What has not been shown yet is on the card on the table.

## Five questions judges will ask

1. Are you solo? Yes; the page says 2 to 3, this is one person, said up front.
2. Was any of this written before the event? The EMG board driver, yes, and it is
   disclosed; the arm, camera, follow loop, poses, wall and Atlas code were written here.
3. Why EMG and not voice? Voice fails the people who cannot speak, the loud floor and the
   sterile room; a face muscle works in all three, and a clench is loud enough on the
   board that a threshold reads it without training.
4. What is the latency? The follow loop is about 20 Hz on a Mac over the LAN; the depth
   map alone costs 350 ms, so it is fetched only when needed; EMG-to-gripper latency is
   not yet measured.
5. What breaks? The board has never been linked on a Mac with real hardware; an unplugged
   channel makes every channel noisy; the arm's motor hum is magnetic, so Carl stands two
   metres away; if the board is not live, a keyboard key stands in and that is said aloud.

## Friday morning, first hour (written Thu 29 Oct 20:05)

Decided Thursday night: the demo is face-controlled. Two pads on the cheeks (left and right smile muscles),
reference on the bone behind the left ear, bias on the left collarbone.

1. 9:00 Pads on, laptop on battery. In `silent-speech-cerelog`: `.venv/bin/python -m ssc live --port
   /dev/cu.usbserial-110 --channels 1 2`. Five seconds still for the rest level. Then: half-smile left,
   half-smile right, clench. Each should print as its own event with channel 1 or 2 clearly larger (left,
   right) or both (clench). If the three are not separable on screen in ten minutes, the demo is the palm
   handover (`pick.py --live --cube green`), already verified, and the face is one sentence.
2. 9:30 Arm side (main thread): `clench.py bridge` copies events to `data/clench.log`; a new `face_drive.py`
   maps left/right to the waist, long clench to switch to the elbow, short clench to grab or release, with
   the camera doing the fine centring over the cube and the hand check. Thirty minutes.
3. 10:30 Three full runs with hands behind the back. Count them. Rehearse the 60-second script over a run.
4. 12:00 Decide: face demo or palm demo. Then lunch, then the Atlas log on the screen beside the arm.

What is verified and does not need touching: the pick (cube-shaped blob, servo, staged descent, shift test),
the follower on the box, the palm release. The camera-to-gripper offset is in `data/cam_offset.json`.
Robots-off at 21:00 Thursday; the box's coach and recordings were pulled to the Mac before that.

## Face controls (decided Thu 29 Oct 20:20)

Three face events from two cheek pads, counted within 0.8 s: L (half-smile left), R (half-smile right),
B (both sides: full smile or clench). A group is acted on after 0.8 s of quiet; the screen shows the
count as it is heard (L, LL, B B) before anything moves.

Fetch mode (start state, at greet, looking at Carl):
| face | action |
|---|---|
| L | grab the food: home, find the cube, pick, prove the grip, back to greet |
| R | feed me: follow the nose, bring the cube to the mouth, hold it there |
| B | done: let go into a hand if one is under it, else take it back to where it was found |
| B B | switch to Nudge mode |
| B B B | home: stop, arm to home, gripper open |

Nudge mode (manual, 30 mm per step):
| face | action |
|---|---|
| L / R | one step left / right along the current axis |
| B B | next axis: left/right, toward/away, up/down |
| B | grab if empty, let go if holding |
| B B B | back to Fetch mode at greet |

The camera keeps: centring over the cube, which hand is Carl's, the sideways lift proof.
One sentence: "I fed myself with my face: one smile to fetch, one to bring it to my mouth, one to say I'm done."


## Friday, the real first hour (rewritten Thu 29 Oct 23:05: the face is the Mac camera, not pads)

Thursday night the control channel became the MacBook camera: the face model gives left cheek, right cheek,
full smile and open mouth as numbers; no pads, no soldering. The EMG board stays the next step (a sentence
in the pitch: "the same controls from muscle signals, for a face that barely moves").

1. 9:00 Robot on. Terminal: `open run_calibrate.command` (20 s: rest, left cheek, right cheek, smile,
   open mouth; the window coaches you). Then `.venv/bin/python train_face.py` (one second; prints held-out
   accuracy; want over 95%). Then `open run_smile.command`: the live window with the controls panel.
2. 9:10 `.venv/bin/python serve.py --live --cube green` on the Mac (or on the box). Left cheek: fetch.
   Open mouth: feed (follows your face, holds the cube 22 cm from your mouth 2 s, puts it back). Right
   cheek: palm handover. Full smile: let go. Three smiles: home. Never run yet live: the feed. Test it
   first with a hand on the red button.
3. 9:40 `open run_dash.command`: the demo screen on http://localhost:8790 (robot view + honest log on the
   left, your face + controls on the right). Put the Mac beside the arm facing the judges.
4. 10:30 Three runs hands-behind-the-back, counted. Rehearse the 60 s script over one.
5. 12:00 Decide: face demo, or palm demo if the feed misbehaves.


## Friday 30 Oct, 14:05: the final controls and the run of show

Decided at the table: no modes, no crane. Skully is the pewter skull with blue painter's tape on his nose.
He stands on the table between two phones, facing the robot. Blue cube off the table (same hue as the tape).

| face | the robot |
|---|---|
| left cheek up (L) | fetch: home, find the green cube, pick it, prove the grip (shift test), back to greet |
| right cheek up (R) | feed Skully: home, look for the tape twice a second apart; if he held still (under 20 mm) and is 530 to 800 mm from the base, come in level to his mouth, let go, back out, greet. Else refuse, say why, keep the cube |
| full smile (B) | let go where it is |
| two smiles (BB) | home, gripper open: the stop |

Open mouth (M) does nothing unless `serve.py --live --face-feed`: feeding a person's face was never verified
live. Three smiles do nothing.

The three guards, said out loud at the demo: he has to be seen, he has to hold still, he has to be in reach.
One deliberate refusal in the demo: smile right while moving Skully, the robot says "REFUSED: Skully moved", goes
back to greet with the cube. Then put him down, smile right again, he eats.

### Measure before the first live feed (two numbers, a ruler, one minute)

- `TAPE_Z` in `skull.py`: height of the tape's centre above the table with Skully standing. Default 150. Used
  only when the depth map has no reading on the tape; the live finder uses depth first.
- `MOUTH_BELOW`: tape centre to the centre of the open mouth. Default 70.
- Then `.venv/bin/python skull.py --live --probe` with the cube in the gripper: it goes to the pre-feed pose
  150 mm in front of the mouth and waits 5 s. Check the height by eye. Then `skull.py --live` once without
  the smile. Only then `serve.py --live`.

### Start order

1. `open run_smile.command` (calibrate first with `open run_calibrate.command` if the lighting changed).
2. `.venv/bin/python serve.py --live --cube green`
3. `open run_dash.command`, http://localhost:8790, Mac facing the judges.

### Three minutes, spoken

0:00 "Meet Skully. He's dead hungry." (pause) "Jokes aside: for someone with a cervical spinal cord injury, the
spinal cord is damaged in the neck, so the arms and hands can be mostly paralysed while the face still moves.
Bringing food to your own mouth needs another person. This asks: what if a small facial movement were enough?"

0:25 "This is Served With a Smile. A UFactory xArm 6 on Viam, the MacBook camera reading my face, a tiny
classifier I trained on my own face in 20 seconds. Four gestures, no hands, no voice." (left cheek) "That is
fetch: find the cube, pick it, prove it has it by moving sideways and looking."

1:00 "Now Skully is hungry. But the robot does not feed him because I smiled. It looks for the tape on his nose
twice, a second apart. If he moved, it refuses." (move Skully, right cheek) "Refused; he was moving. He keeps
the cube." (put Skully down) (right cheek) "Still, in reach: it comes in level and lets go. Bon appetit, Skully."

2:00 "Every command is an episode: the gesture, what it saw, the three checks, the outcome, how long. On disk
at once, then MongoDB: 98 documents so far, Thursday's runs backfilled with their real timestamps" (point at the
MongoDB panel on the screen: Atlas connected, 98 synced).
"Modular: the face reader writes events to a file; the arm loop reads events. Swap the face reader for the
EMG board (hold it up) and nothing on the arm side changes. That is the next step: the same four commands
from the muscle signal, for a face that barely moves."

2:40 "Solo entry. The EMG driver predates the event; everything touching the arm, the camera and the log was
written here. Thank you."

### Submission form (https://forms.gle/T4Z5ah4M55uS7KDr8, by 3:00 PM), suggested answers

- Project name: Served With a Smile
- Team: Carl Kho (solo), Somach Systems
- One line: a face-controlled robot arm that fetches food and feeds Skully the skeleton, and knows when not to.
- What it does: half-smile left fetches a cube (vision pick with a self-check); half-smile right feeds Skully
  only after two looks a second apart agree he is still and in reach; a full smile releases; two smiles stop.
  Every command is logged as an episode (gesture, observation, checks, outcome, duration) to MongoDB.
- Built with: UFactory xArm 6 via Viam (arm, gripper, RealSense camera, motion planner with walls), Viam
  Python SDK, MediaPipe face blendshapes plus a 20-second personal classifier, OpenCV colour targets, MongoDB
  (pymongo; Atlas when reachable, local JSONL mirror always).
- Repo: https://github.com/CarlKho-Minerva/emg-arm
- Disclosure: the Cerelog EMG driver (silent-speech-cerelog) predates the event and is the next input; it is
  not driving the arm today.


## The run-through, 30 Oct 15:00 (what is actually built; use this one)

Screen: http://localhost:8790, Command centre tab, Focus on (press f). Dashboard sidebar: Start all. Arm at greet.
Cubes within 70 cm of the base. Skully in your hand, blue tape toward the arm's camera.

| you | the arm | say |
|---|---|---|
| "Meet Skully. He's dead hungry." | at greet, looking at you | "Jokes aside: for someone with a cervical spinal cord injury the arms are paralysed and the face still works. This asks whether a half-smile is enough to be fed." |
| left cheek | home, finds the cube, hovers, turns to the cube's angle, picks, shifts sideways to prove the grip, back to greet | "That is one gesture. The arm found the cube by colour, turned its fingers to the cube's angle, and checked its own grip by moving and looking." |
| hold Skully up, tape to the camera, still | the camera follows the tape; when he is still a second, the cube comes to his mouth and holds four seconds, then back to greet | "The robot follows the tape on his nose with the same controller that tracked my hand yesterday. It only moves in once he holds still." |
| right cheek (when he is still) | the feed above | "Bon appetit, Skully." |
| full smile | lets go | |
| two smiles, then hold a cheek | layer STEER, pans while held | "Two smiles change layer. Now a held cheek steers the arm, mirrored for me, because the arm faces me." |
| three smiles | home, gripper open, layer SERVE | "Three smiles is the stop." |
| point at the screen | episodes count up in Atlas | "Every command is one record: gesture, what each camera saw, the checks, the outcome. Thursday's runs are in there too." |
| hold up the EMG board | | "Next input: the muscle signal from two pads on the cheeks, for a face that barely moves. Same events, same file, the arm does not change." |

If the fetch says the cube is out of reach, slide the cubes toward the arm and left cheek again. If Skully is not followed,
bring him closer to the camera (40 to 60 cm) and keep the tape facing it. Three smiles at any time brings everything home.
