# Submission form answers

Form: https://forms.gle/T4Z5ah4M55uS7KDr8, closes 3:00 PM Friday 30 October 2026. One person per team; the email is recorded.

**Team Roster**
Carl Kho, solo. Somach Systems, Inc.

**Project Title**
Served With a Smile

**Project Abstract**
Picture someone at a table who cannot lift a fork. The food is in front of them. Somebody else has to bring it to their mouth, and that somebody decides when. We asked a narrow question: could the person give that order with their face, and could a robot arm do the rest?

Our answer is a table-top prototype. A laptop camera watches the person's face. A six-joint robot arm, the kind used on factory lines, stands on the table with a small camera on its wrist. When the person pulls up the left corner of their mouth, the arm looks down at the table, finds a cube, picks it up, and checks its own grip by moving sideways and looking at where the cube was. When they pull up the right corner, the arm feeds Skully, a plastic skull that stands in for a person while the feeding move is tested: it finds a strip of blue tape on Skully's nose, hovers above him to look again, brings the cube level into his open mouth, and lets go. A full smile makes the arm let go of whatever it holds; two smiles send it home.

The face reader and the arm share nothing but a short list of named events. That is deliberate. It means the camera can later be replaced by small sensors on the cheeks that read the electrical signal a muscle makes when it tightens, for a face that barely moves, without changing anything on the arm. Every command the arm receives is written as one record in a database, with what each camera saw, which safety checks passed, and how the attempt ended.

**Who will use this solution? What is the expected business value?**
The person we have in mind has little or no use of their arms and still has their face. The clearest case is a cervical spinal cord injury. The spinal cord is the bundle of nerves running down the back; when it is damaged in the neck, the arms and hands are mostly paralysed, while the muscles of the face, which are wired above the injury, keep working. Late muscular dystrophy, some strokes and some forms of cerebral palsy look the same from the robot's point of view: the arm is gone, the face is there.

For all of them, eating is the daily task most often handed to another person, three times a day, for life. The point of this prototype is not to remove the carer. It is to give the person back the small decision of when the next bite comes, with a gesture their body can still make.

The camera is the first way of reading that gesture, not the last. A muscle that tightens gives off a faint electrical signal on the skin, and two sticky pads on the cheeks can read it before anything is visible. We have the board and the pads at the table. Because the pads will write the same four events to the same file the camera writes, the arm will behave identically, and we can compare the two inputs on the same recorded attempts: how often each catches the gesture, how long from gesture to motion, how often each fires by mistake. That comparison is the research question this project opens: can a robot read an intended facial gesture as reliably from the muscle as from a picture of the face?

The business value, in plain terms. A home health aide in the United States costs a median of $34 an hour, $77,792 a year on a 44-hour week (Genworth and CareScout Cost of Care Survey, 2024). A meal with help is about half an hour of that attention, three times a day: roughly $17 a meal, $51 a day, close to $19,000 a year for one person's meals, and it cannot be shared across residents or scheduled away. A robot that takes the mechanical part of the meal and leaves the carer to set up, supervise and talk returns most of that time to the rest of the floor, and gives the person being fed the pace they choose. The arm used here is a stock industrial arm with a stock gripper; nothing in the software is tied to it, so the same code moves to a cheaper arm in a home.

Who pays is the same party who already pays for the carer's hour: a care provider, or in many countries the long-term-care insurer or the public program behind it. What they get beyond the hour is the record. Every attempt is stored with its cause and its outcome, which is the audit trail a provider must keep when a machine touches a resident, and it is the evidence that lets a device like this be argued into a reimbursement list at all.

For other robot builders the value is the seam. The face reader and the arm share only a stream of named events, and every behaviour writes one episode to the same database. Swap the face reader for a muscle-signal board, a sip-and-puff switch or a voice command and the arm code does not change; swap the arm and the face code does not change. That pattern, an event file in front and an episode store behind, is what this project leaves behind for anyone building an assistive robot on Viam and MongoDB.

**What tech stack did you use?**
Two cameras, one arm, and a seam between them.

The face side runs on a MacBook. Its camera feeds MediaPipe, a free face-tracking library that turns a face into 52 numbers describing how far each part has moved, plus the positions of 478 points on the face. From those we compute a few shape measurements of our own, such as how far each mouth corner sits above the lip line, corrected for head tilt, so the reading survives the person turning toward someone. A small statistical classifier (a logistic regression) trained on twenty seconds of the user's own face turns these numbers into five labels. It writes named events to a text file.

The arm side reads that file. The arm is a UFactory xArm 6 with a two-finger gripper and an Intel RealSense depth camera on the wrist, run through Viam, a robotics platform that handles the connection, the motion planning around invisible walls we drew, and saved poses. The cube and the blue tape on Skully's nose are found by colour with OpenCV, and placed in the room using the depth picture and Viam's frame arithmetic.

The records go to MongoDB Atlas, the hosted version of the MongoDB database, through pymongo. Ninety-eight attempts are stored so far: today's live runs, and Thursday's picks, hand-following runs and handovers, imported with their original timestamps and marked as imported. The venue's network blocks the kind of name lookup Atlas normally uses, so the database is reached by its server names directly; a file on the laptop is written first and a local copy of MongoDB stands in if the network drops. The screen beside the arm is a plain Python web page.

The muscle-signal input is a Cerelog ESP-EEG board, eight channels at 250 samples a second. It linked on this Mac on 29 October (2,520 packets in ten seconds, none dropped) and has a live detector for a jaw clench that writes to the same event file. Its driver was written before the event and is disclosed as such. It is the next input, not today's.
