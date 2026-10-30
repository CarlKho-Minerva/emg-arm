# Submission form answers

Form: https://forms.gle/T4Z5ah4M55uS7KDr8, closes 3:00 PM Friday 30 October 2026. One person per team; the email is recorded.

**Team Roster**
Carl Kho, solo. Somach Systems, Inc.

**Project Title**
Served With a Smile

**Project Abstract**
A laptop camera watches a face; a robot arm on the table does the handling. Pull your left cheek up and the arm's wrist camera finds a cube, the arm picks it and proves the grip by moving sideways and looking at where the cube was. Pull your right cheek up and the arm feeds Skully, a skeleton skull: the wrist camera finds the blue tape on his nose from above, re-centres from a hover, and the arm brings the cube level into his open mouth and lets go. The face reader and the arm share nothing but a stream of four named events, so the camera can be swapped for surface electromyography (sEMG), the electrical signal a facial muscle gives off when it tightens, read by two sticky pads on the cheeks; the board and pads are at the table and the arm side does not change. Every command is one record in MongoDB Atlas: what each camera saw, which checks passed, how it ended.

**Who will use this solution? What is the expected business value?**
The person this is for has little or no use of their arms and still has their face. The clearest case is a cervical spinal cord injury: the cord is damaged in the neck, so the arms and hands are mostly paralysed while the facial muscles keep working. Late muscular dystrophy, some strokes and some cerebral palsy look the same from the robot's side. For all of them, eating is the daily task most often handed to another person. Here the person keeps the decision, a half-smile, and the arm does the movement.

The camera is the first input, not the last. A face that barely moves still produces a muscle signal, and the pads read that signal before it becomes a visible expression. The same four events (left, right, both, release) come from the pads through the same file the camera writes, so the robot behaves identically and the two inputs can be compared on the same episodes: detection rate, latency from gesture to motion, false triggers. That comparison is the research question this project sets up: can a robot read an intended facial gesture as reliably from muscle activity as from a camera.

The architecture is already the shape of a hierarchical policy. The gesture picks a behaviour (fetch, feed, release, home); each behaviour has its own start conditions (a cube in hand, Skully seen and in reach), its own controller (the visual servo over the cube, the close look over Skully) and its own end (fed, refused, lost). Today those are written by hand; the episodes in Atlas are what a learned version would train and be judged on.

In plain business terms: one caregiver hour returned per meal, on a stock arm a clinic or a home can run, with an audit trail of every attempt.

**What tech stack did you use?**
Two cameras and one arm, with a clean seam between them. The face: the MacBook camera, MediaPipe (a face becomes 52 numbers) and a logistic regression trained on 20 seconds of Carl's own face that turns those into five labels; it writes named events to a file. The arm: a UFactory xArm 6 with a two-finger gripper and a RealSense depth camera on the wrist, run through Viam (motion planner with invisible walls, saved poses, Python SDK); it reads the event file and does the handling. The cube and the blue tape on Skully's nose are found by colour in OpenCV and placed in the room with the depth map and Viam's frame maths. MongoDB Atlas through pymongo holds the episodes, 98 so far, Thursday's picks, follower runs and handovers backfilled with their real timestamps; the venue DNS blocks SRV lookups, so the cluster is reached by host name, with a JSONL mirror written first and a local MongoDB as fallback. The screen beside the arm is a plain Python HTTP page. The sEMG input: a Cerelog ESP-EEG board (8 channels, 250 samples a second), linked on this Mac on 29 October (2,520 packets in 10 s, 0 dropped), with a live clench detector that writes to the same event file; its driver was written before the event and is disclosed as such. It is the next input, not today's.
