# Submission form answers

Form: https://forms.gle/T4Z5ah4M55uS7KDr8, closes 3:00 PM Friday 30 October 2026. One person per team; the email is recorded.

**Team Roster**
Carl Kho, solo. Somach Systems, Inc.

**Project Title**
Served With a Smile

**Project Abstract**
A laptop camera watches a face. A robot arm on the table does the handling. Pull your left cheek up and the arm's own wrist camera finds a cube on the table; the arm picks it and checks its grip by moving sideways and looking at where the cube was. Pull your right cheek up and the arm feeds Skully, a skeleton skull: the wrist camera finds the blue tape on his nose from above, re-centres on it from a hover, and the arm brings the cube level into his open mouth and lets go. The problem it works on is the ordinary act of bringing food to a mouth when the arms cannot do it, with the human's part reduced to a facial gesture. Every command becomes one record in MongoDB Atlas, with what each camera saw, which checks passed, and how it ended.

**Who will use this solution? What is the expected business value?**
The person this is for has little or no use of their arms and still has their face. The clearest case is a cervical spinal cord injury: the cord is damaged in the neck, so the arms and hands are mostly paralysed while the muscles of the face keep working. Late muscular dystrophy, some strokes and some cerebral palsy look the same from the robot's side. For all of them, eating is the daily task most often handed to another person. Here the person keeps the decision (a half-smile) and the arm does the movement. The next input is the electrical signal of the facial muscles read from the skin, for a face that barely moves; the arm side does not change, because the face reader and the arm only share a stream of named events. In plain business terms: one caregiver hour returned per meal, on a stock arm a clinic or a home can run, with a record of every attempt that doubles as the audit trail and as the training data for a safer, learned version.

**What tech stack did you use?**
Two cameras and one arm, with a clean seam between them. The face: the MacBook camera, MediaPipe (a face becomes 52 numbers), and a logistic regression trained on 20 seconds of Carl's own face that turns those into five labels; it writes named events (L, R, B, BB) to a file. The arm: a UFactory xArm 6 with a two-finger gripper and a RealSense depth camera on the wrist, run through Viam (motion planner with invisible walls, saved poses, the Python SDK); it reads the event file and does the handling. The cube and the blue tape on Skully's nose are found by colour in OpenCV and placed in the room with the depth map and Viam's frame maths. MongoDB Atlas through pymongo holds the episodes, 98 so far, including Thursday's picks, follower runs and handovers backfilled with their real timestamps; the venue's DNS blocks SRV lookups, so the cluster is reached by host name, with a JSONL mirror written first and a local MongoDB as the fallback. The screen beside the arm is a plain Python HTTP page. Disclosure: the EMG board driver (silent-speech-cerelog) was written before the event and is not driving the arm today.
