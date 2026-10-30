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
