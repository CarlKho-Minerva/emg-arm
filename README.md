# emg-arm

Viam x MongoDB Robot Hackathon, SF, 29 to 30 Oct 2026. Solo entry by Carl Kho (Somach Systems).

Goal: drive a UFactory xArm with the face, no hands, no voice. Surface EMG from a Cerelog board.
Step one is a single switch: a jaw clench closes the gripper.

- `viamarm.py`: the one connection to the machine (secrets in `~/.config/carl-life-os/.env`)
- `probe.py`: read-only, prints the six joints and the hand's position
- `jog.py`: nudge one joint, open or close the gripper, stop
- `HANDOFF.md`: the full plan, hardware facts, pad placement
- `lead-figure.html`: how to make the four leads

The EMG driver (`silent-speech-cerelog`) predates the event and is disclosed as such.
