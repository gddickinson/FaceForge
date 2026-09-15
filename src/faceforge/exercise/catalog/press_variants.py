"""Pressing variations away from the bench: overhead, and the triceps work.

The strict overhead press, the dips and the cable pushdown live in
:mod:`faceforge.exercise.catalog.upper_push`; the bench family in
``bench_variants``.  These are the four that change how an overhead or triceps
press is driven.

* A **push press** starts the bar with the legs, which lets a load past the
  point where a strict press stalls -- the dip is shallow and fast, and the
  arms take over at about the forehead.
* An **Arnold press** rotates from a supinated start, so the anterior deltoid
  works through an external-rotation range a straight press never enters.
* A **close-grip push-up** is the bodyweight answer to a close-grip bench:
  hands about shoulder width, elbows tucked, triceps carrying the lockout.
* **Overhead triceps extension** loads the long head in its lengthened
  position, which is the range a pushdown does not reach because the long head
  crosses the shoulder.
"""

from __future__ import annotations

from faceforge.exercise.catalog._helpers import (
    ACE, CALATAYUD, CON, ECC, ECC_CON, EXRX, ISO, KOLBER, NSCA, P, S, ST, SEATED_ON_BENCH,
    arms, eq, flat_palm, grip, merge, mu, ph, pose, squat, stand,
)
from faceforge.exercise.model import Category, ExerciseDefinition

_RACK = arms(flex=15, abduct=35, rotate=20, elbow=130, forearm=-60, wrist=-15)
_OVERHEAD = arms(flex=170, abduct=12, rotate=10, elbow=8, forearm=-30, wrist=-10)

push_press = ExerciseDefinition(
    id="push_press", name="Push press", category=Category.UPPER_PUSH,
    description="An overhead press driven by a short dip and drive of the legs. The legs "
                "start the bar and the arms finish it, which moves more load than a strict "
                "press can.",
    setup=("Bar in the front rack, elbows up, feet hip width",
           "Dip a few inches with the trunk vertical: this is not a squat",
           "Drive, then punch the head through as the bar passes the forehead"),
    phases=(
        ph("Dip", ECC, 0.5, merge(squat(20, 25, 5)[0], _RACK, grip()), pitch=5,
           easing="ease_in",
           cues=("A short dip, straight down, heels flat",)),
        ph("Drive", CON, 0.35, merge(stand()[0], arms(flex=95, abduct=25, rotate=15,
                                                      elbow=80, forearm=-45, wrist=-10),
                                     grip()), easing="ease_out",
           cues=("Stand up violently; the bar leaves the shoulders on the legs",)),
        ph("Press out", CON, 0.5, merge(stand()[0], _OVERHEAD, grip()),
           cues=("Punch the head through as the bar passes the forehead",)),
        ph("Lockout", ISO, 0.6, merge(stand()[0], _OVERHEAD, grip()),
           cues=("Bar over the mid-foot, ribs down, glutes tight",)),
        ph("Return", ECC, 0.9, merge(stand()[0], _RACK, grip()),
           cues=("Bring it back to the rack and absorb with the legs",)),
    ),
    muscles=(mu("deltoid_anterior", P, 0.9), mu("triceps_brachii", P, 0.8),
             mu("quadriceps", P, 0.7, note="the dip and drive"),
             mu("deltoid_lateral", S, 0.65), mu("gluteus_maximus", S, 0.6),
             mu("trapezius_upper", S, 0.6), mu("serratus_anterior", S, 0.5),
             mu("erector_spinae", P, 0.65, note="holds the trunk vertical through the dip"),
             mu("rectus_abdominis", S, 0.5), mu("obliques", S, 0.45),
             mu("gastrocnemius", S, 0.4), mu("rotator_cuff", ST, 0.4),
             mu("forearm_flexors", ST, 0.5)),
    equipment=(eq("barbell", plates=2),),
    errors=("Dipping too deep, which turns it into a thruster.",
            "Leaning back to start the bar instead of driving with the legs.",
            "Pressing before the legs have finished."),
    physio_notes=("The dip is shallow and fast: its job is to load the legs elastically, "
                  "not to squat. The arms take over at about forehead height, which is why "
                  "a push press trains the top half of a strict press.",),
    sources=(NSCA, ACE, EXRX, ECC_CON), camera="three_quarter", default_reps=5,
    tags=("barbell", "power"),
)

arnold_press = ExerciseDefinition(
    id="arnold_press", name="Arnold press", category=Category.UPPER_PUSH,
    description="A seated dumbbell press that starts with the palms facing the lifter and "
                "rotates to face forward on the way up, adding an external-rotation range a "
                "straight press never enters.",
    setup=("Seated, dumbbells at chest height with the palms toward you",
           "Rotate the palms out as the weights rise", "Finish with the palms forward overhead"),
    # Every phase has the hips and knees at 90 and the setup says "seated":
    # authored standing, that is a man sitting on nothing, which the ground
    # lock then folds onto the floor.  Same fault as the overhead triceps
    # extension, found by comparing each definition's words with its
    # orientation.
    orientation="seated", anchor="none", base_position=SEATED_ON_BENCH,
    phases=(
        ph("Start", ISO, 0.4, merge(pose(hip_flex=90, knee_flex=90),
                                    arms(flex=35, abduct=5, rotate=-70, elbow=130,
                                         forearm=60, wrist=-10), grip()),
           cues=("Elbows in, palms facing you, weights at the collarbones",)),
        ph("Rotate and press", CON, 1.6, merge(pose(hip_flex=90, knee_flex=90),
                                               arms(flex=165, abduct=18, rotate=20, elbow=10,
                                                    forearm=-40, wrist=-10), grip()),
           cues=("Turn the palms out as the weights pass the face",
                 "Finish with the arms straight and the palms forward")),
        ph("Lockout", ISO, 0.5, merge(pose(hip_flex=90, knee_flex=90),
                                      arms(flex=165, abduct=18, rotate=20, elbow=10,
                                           forearm=-40, wrist=-10), grip())),
        ph("Reverse", ECC, 2.0, merge(pose(hip_flex=90, knee_flex=90),
                                      arms(flex=35, abduct=5, rotate=-70, elbow=130,
                                           forearm=60, wrist=-10), grip()),
           cues=("Rotate back in on the way down",)),
    ),
    muscles=(mu("deltoid_anterior", P, 0.9), mu("deltoid_lateral", P, 0.7),
             mu("triceps_brachii", S, 0.6),
             mu("rotator_cuff", P, 0.6, note="the rotation is loaded, not incidental"),
             mu("trapezius_upper", S, 0.55), mu("serratus_anterior", S, 0.5),
             mu("pectoralis_upper", S, 0.4, note="from the supinated start"),
             mu("erector_spinae", ST, 0.4), mu("rectus_abdominis", ST, 0.4),
             mu("forearm_flexors", ST, 0.45)),
    equipment=(eq("dumbbell", attach="hand_r"), eq("dumbbell", attach="hand_l"),
               eq("bench", attach="static")),
    errors=("Rotating after the press rather than through it.",
            "Loading it like a straight press: the rotation is the limit.",
            "Flaring the ribs and arching the low back at lockout."),
    physio_notes=("The value is the range, not the load: it takes the shoulder from "
                  "internal rotation at the bottom to neutral overhead under control, "
                  "which a straight dumbbell press does not.",),
    sources=(NSCA, ACE, EXRX, KOLBER), camera="front", tags=("dumbbell",),
)

# The push-up's measured pitches (upper_push): the straight body is inclined
# head-up ~20 deg on locked arms and ~8 at the bottom.  Close grip only narrows
# the elbows -- 20 deg of abduction at the bottom instead of 45.
_CGPU_TOP = merge(pose(ankle_flex=45), arms(flex=70, abduct=10, elbow=0), flat_palm())
_CGPU_BOTTOM = merge(pose(ankle_flex=45), arms(flex=38, abduct=20, elbow=95), flat_palm())

close_grip_push_up = ExerciseDefinition(
    id="close_grip_push_up", name="Close-grip push-up", category=Category.UPPER_PUSH,
    description="A push-up with the hands about shoulder width and the elbows tucked, which "
                "is the bodyweight version of a close-grip bench press.",
    setup=("Hands under the shoulders, not narrower", "Elbows brush the ribs on the way down",
           "Body straight from ear to heel throughout"),
    orientation="prone", anchor="hands", base_position=(-85.0, 30.0, 0.0),
    phases=(
        ph("Lower", ECC, 1.8, _CGPU_BOTTOM, pitch=-8,
           cues=("Chest to the floor with the elbows close to the ribs",)),
        ph("Bottom", ISO, 0.3, _CGPU_BOTTOM, pitch=-8,
           cues=("Stay a plank: no sagging, no piking",)),
        ph("Press", CON, 1.3, _CGPU_TOP, pitch=-18,
           cues=("Push the floor away and finish by straightening the elbows",)),
        ph("Top", ISO, 0.4, _CGPU_TOP, pitch=-18),
    ),
    muscles=(mu("triceps_brachii", P, 0.85), mu("pectoralis_major", P, 0.7),
             mu("deltoid_anterior", S, 0.6),
             mu("serratus_anterior", P, 0.65, note="holds the scapula against the ribs"),
             mu("rectus_abdominis", P, 0.6, note="the plank is half the exercise"),
             mu("obliques", S, 0.5), mu("gluteus_maximus", S, 0.45),
             mu("erector_spinae", ST, 0.4), mu("rotator_cuff", ST, 0.4),
             mu("forearm_extensors", ST, 0.45)),
    equipment=(eq("mat", attach="static"),),
    errors=("Hands so close the wrists take it instead of the triceps.",
            "Elbows flaring out, which makes it an ordinary push-up.",
            "Hips sagging or piking."),
    physio_notes=("At the same tempo a push-up and a bench press produce comparable muscle "
                  "activity for a comparable relative load, so a close-grip push-up is a "
                  "reasonable stand-in for a close-grip bench.",),
    sources=(CALATAYUD, NSCA, ACE, EXRX), camera="front", tags=("bodyweight",),
)

overhead_triceps_extension = ExerciseDefinition(
    id="overhead_triceps_extension", name="Overhead triceps extension",
    category=Category.UPPER_PUSH,
    description="Elbow extension with the arm overhead. The long head of triceps crosses the "
                "shoulder, so only an overhead position loads it at length.",
    setup=("Seated or standing, weight held overhead in both hands",
           "Upper arms stay vertical and close to the head",
           "Lower behind the head until the stretch, then extend"),
    orientation="seated", anchor="none", base_position=SEATED_ON_BENCH,
    phases=(
        ph("Lower", ECC, 2.0, merge(pose(hip_flex=90, knee_flex=90),
                                    arms(flex=160, abduct=12, rotate=10, elbow=135,
                                         forearm=-30, wrist=-10), grip()),
           cues=("Bend only at the elbow; the upper arms do not travel",)),
        ph("Stretch", ISO, 0.4, merge(pose(hip_flex=90, knee_flex=90),
                                      arms(flex=160, abduct=12, rotate=10, elbow=135,
                                           forearm=-30, wrist=-10), grip()),
           cues=("Feel it along the back of the arm, not in the shoulder",)),
        ph("Extend", CON, 1.4, merge(pose(hip_flex=90, knee_flex=90),
                                     arms(flex=168, abduct=10, rotate=10, elbow=8,
                                          forearm=-30, wrist=-10), grip()),
           cues=("Straighten the elbows without letting the ribs flare",)),
        ph("Lockout", ISO, 0.4, merge(pose(hip_flex=90, knee_flex=90),
                                      arms(flex=168, abduct=10, rotate=10, elbow=8,
                                           forearm=-30, wrist=-10), grip())),
    ),
    muscles=(mu("triceps_brachii", P, 0.95, note="the long head at length"),
             mu("deltoid_anterior", S, 0.4, note="holds the arm overhead"),
             mu("serratus_anterior", S, 0.45), mu("trapezius_upper", S, 0.4),
             mu("rotator_cuff", ST, 0.4),
             mu("rectus_abdominis", ST, 0.5, note="stops the ribs flaring"),
             mu("erector_spinae", ST, 0.45), mu("forearm_flexors", ST, 0.45)),
    equipment=(eq("dumbbell", attach="hands"), eq("bench", attach="static")),
    errors=("Letting the elbows drift forward and wide, which shortens the long head again.",
            "Arching the low back to get the arms overhead.",
            "Going heavy enough that the shoulder, not the triceps, sets the range."),
    physio_notes=("The long head is the only triceps head crossing the shoulder, so it is "
                  "lengthened by shoulder flexion; that is why overhead work and pushdowns "
                  "are not interchangeable.",),
    sources=(NSCA, ACE, EXRX, ECC_CON), camera="side", tags=("dumbbell",),
)

EXERCISES = (push_press, arnold_press, close_grip_push_up, overhead_triceps_extension)
