"""Horizontal pulling: the rows, the face pull and the reverse fly.

A row is a pull toward the trunk rather than down past it, so the scapular
retractors -- rhomboids and middle trapezius -- do more of it than they do in
a pull-up, and the trunk has to hold whatever angle the variation asks for.
The vertical pulls are in :mod:`faceforge.exercise.catalog.upper_pull`.
"""

from __future__ import annotations

from faceforge.exercise.catalog._helpers import (
    ACE, CON, ECC, EXRX, ISO, NEUMANN, NSCA, P, REINOLD, S, SCHOENFELD_ROW, ST, arms, eq,
    grip, hinge, merge, mu, ph, pose,
)
from faceforge.exercise.model import Category, ExerciseDefinition

#: The bent-over row's trunk: pitched 50 deg with the knees soft.
_ROW_LEGS, _ROW_PITCH = hinge(50, 15)


barbell_bent_over_row = ExerciseDefinition(
    id="barbell_bent_over_row", name="Barbell bent-over row", category=Category.UPPER_PULL,
    description="From a hip hinge with the trunk ~50 deg from vertical, the bar is pulled to "
                "the lower ribs with the elbows driving back.",
    setup=("Hinge at the hips, knees soft, spine neutral", "Overhand grip just wider than "
           "the shoulders, arms hanging", "Brace the trunk; the hinge angle does not change"),
    phases=(
        ph("Row", CON, 1.2, merge(_ROW_LEGS, arms(flex=-10, abduct=25, elbow=105), grip()),
           pitch=_ROW_PITCH, cues=("Pull the elbows back past the trunk; bar to the lower "
                                   "ribs", "Shoulder blades retract at the end")),
        ph("Squeeze", ISO, 0.4, merge(_ROW_LEGS, arms(flex=-10, abduct=25, elbow=105), grip()),
           pitch=_ROW_PITCH),
        ph("Lower", ECC, 1.8, merge(_ROW_LEGS, arms(flex=50, abduct=5, elbow=5), grip()),
           pitch=_ROW_PITCH, cues=("Arms straight, shoulder blades protract; trunk still",)),
    ),
    muscles=(mu("latissimus_dorsi", P, 0.95), mu("rhomboids", P, 0.8),
             mu("trapezius_middle", P, 0.8), mu("deltoid_posterior", P, 0.75),
             mu("biceps_brachii", S, 0.6), mu("brachialis", S, 0.6),
             mu("erector_spinae", S, 0.75, note="isometric hold of the hinge"),
             mu("infraspinatus_teres_minor", S, 0.45), mu("hamstrings", ST, 0.5),
             mu("gluteus_maximus", ST, 0.5), mu("forearm_flexors", ST, 0.6),
             mu("rectus_abdominis", ST, 0.3)),
    equipment=(eq("barbell", plates=1),),
    errors=("Trunk rising and falling with each rep (using the hips).",
            "Rounding the lumbar spine.", "Shrugging instead of retracting."),
    physio_notes=("High erector spinae and lumbar load; the chest-supported row removes it "
                  "(Fenwick 2009).",),
    sources=(SCHOENFELD_ROW, NSCA, EXRX), camera="three_quarter", tags=("barbell",),
)


single_arm_dumbbell_row = ExerciseDefinition(
    id="single_arm_dumbbell_row", name="Single-arm dumbbell row", category=Category.UPPER_PULL,
    description="Supported on one hand with the trunk near horizontal, the other arm rows a "
                "dumbbell to the hip.",
    setup=("Left hand on the bench, staggered stance, trunk ~60 deg from vertical",
           "Dumbbell hanging straight down, shoulder blade relaxed forward"),
    phases=(
        ph("Row", CON, 1.2,
           merge(hinge(60, 20)[0], arms(flex=60, elbow=0, side="l"),
                 arms(flex=-15, abduct=15, elbow=110, side="r"), grip()),
           pitch=60, cues=("Elbow back past the trunk, dumbbell to the hip",
                           "Shoulders square; do not rotate the trunk")),
        ph("Squeeze", ISO, 0.4,
           merge(hinge(60, 20)[0], arms(flex=60, elbow=0, side="l"),
                 arms(flex=-15, abduct=15, elbow=110, side="r"), grip()), pitch=60),
        ph("Lower", ECC, 1.8,
           merge(hinge(60, 20)[0], arms(flex=60, elbow=0, side="l"),
                 arms(flex=60, abduct=5, elbow=5, side="r"), grip()),
           pitch=60, cues=("Arm straight, let the shoulder blade protract",)),
    ),
    muscles=(mu("latissimus_dorsi", P, 0.95, side="R"), mu("rhomboids", P, 0.75, side="R"),
             mu("trapezius_middle", P, 0.75, side="R"), mu("deltoid_posterior", S, 0.7, side="R"),
             mu("biceps_brachii", S, 0.6, side="R"), mu("brachialis", S, 0.55, side="R"),
             mu("infraspinatus_teres_minor", S, 0.4, side="R"), mu("obliques", ST, 0.45,
                                                                   note="anti-rotation"),
             mu("erector_spinae", ST, 0.4), mu("forearm_flexors", ST, 0.6, side="R"),
             mu("triceps_brachii", ST, 0.3, side="L", note="supporting arm"),
             mu("gluteus_maximus", ST, 0.3)),
    equipment=(eq("dumbbell", attach="hand_r"),
               eq("bench", attach="static", position=(-60.0, 0.0, 30.0), height=70.0, length=110.0)),
    errors=("Rotating the trunk to lift the weight.", "Rowing to the shoulder instead of "
            "the hip.", "Yanking with the biceps."),
    physio_notes=("The supported position reduces lumbar load compared with the bent-over "
                  "barbell row.",),
    sources=(SCHOENFELD_ROW, NSCA, ACE), unilateral=True, camera="three_quarter",
    tags=("dumbbell", "unilateral"),
)


seated_cable_row = ExerciseDefinition(
    id="seated_cable_row", name="Seated cable row", category=Category.UPPER_PULL,
    description="Seated with the legs braced, a handle is pulled to the abdomen with the "
                "trunk upright and the elbows close to the body.",
    setup=("Feet on the platform, knees slightly bent", "Trunk upright, chest up",
           "Arms straight at the start, shoulder blades forward"),
    orientation="seated", anchor="none", base_position=(0.0, 58.0 + 10.0 + 81.0, 0.0),
    phases=(
        ph("Row", CON, 1.2, merge(pose(hip_flex=95, knee_flex=25), arms(flex=5, abduct=10, elbow=115), grip()),
           pitch=-5, cues=("Elbows back, handle to the navel, shoulder blades together",)),
        ph("Squeeze", ISO, 0.4, merge(pose(hip_flex=95, knee_flex=25), arms(flex=5, abduct=10, elbow=115), grip()),
           pitch=-5),
        ph("Return", ECC, 1.8, merge(pose(hip_flex=95, knee_flex=25), arms(flex=85, abduct=5, elbow=5), grip()),
           pitch=5, cues=("Reach forward; let the shoulder blades protract",)),
    ),
    muscles=(mu("latissimus_dorsi", P, 0.9), mu("rhomboids", P, 0.8),
             mu("trapezius_middle", P, 0.8), mu("deltoid_posterior", S, 0.65),
             mu("biceps_brachii", S, 0.6), mu("brachialis", S, 0.55),
             mu("erector_spinae", S, 0.5, note="holds the trunk upright"),
             mu("infraspinatus_teres_minor", S, 0.4), mu("forearm_flexors", ST, 0.55),
             mu("hamstrings", ST, 0.25)),
    equipment=(eq("cable_handle", cable_to=(0.0, 6.0, 150.0)),
               eq("bench", attach="static", height=58.0, length=60.0)),
    errors=("Leaning back and forward with each rep.", "Shrugging the shoulders.",
            "Rounding the back on the return."),
    physio_notes=("Lower lumbar shear than the bent-over row; a good early rowing choice "
                  "for low-back patients.",),
    sources=(SCHOENFELD_ROW, NSCA, EXRX), camera="side", tags=("cable",),
)


face_pull = ExerciseDefinition(
    id="face_pull", name="Face pull", category=Category.UPPER_PULL,
    description="A rope at face height is pulled toward the face while the arms externally "
                "rotate so the hands finish beside the ears.",
    setup=("Rope at eye level, overhand grip, thumbs back", "Step back to tension the cable, "
           "staggered stance", "Chest up, shoulders down"),
    phases=(
        ph("Pull", CON, 1.2, merge(pose(), arms(flex=40, abduct=80, rotate=70, elbow=100, forearm=-40), grip()),
           cues=("Pull the rope apart to the ears; elbows high and wide",
                 "Externally rotate so the knuckles face back")),
        ph("Hold", ISO, 0.5, merge(pose(), arms(flex=40, abduct=80, rotate=70, elbow=100, forearm=-40), grip()),
           cues=("Squeeze the shoulder blades together",)),
        ph("Return", ECC, 1.6, merge(pose(), arms(flex=90, abduct=10, rotate=0, elbow=5, forearm=-40), grip()),
           cues=("Arms straight forward, shoulder blades protract",)),
    ),
    muscles=(mu("deltoid_posterior", P, 0.9), mu("infraspinatus_teres_minor", P, 0.8),
             mu("trapezius_middle", P, 0.7), mu("rhomboids", S, 0.7),
             mu("trapezius_lower", S, 0.6), mu("deltoid_lateral", S, 0.35),
             mu("rotator_cuff", S, 0.5), mu("biceps_brachii", ST, 0.3),
             mu("forearm_flexors", ST, 0.45)),
    equipment=(eq("cable_handle", attach="hand_r", cable_to=(0.0, 40.0, 120.0)),
               eq("cable_handle", attach="hand_l", cable_to=(0.0, 40.0, 120.0))),
    errors=("Pulling to the chest (becomes a row).", "Shrugging.", "No external rotation."),
    physio_notes=("Trains the posterior cuff and scapular retractors that counter the "
                  "forward-shoulder posture; a staple in shoulder impingement programmes.",),
    sources=(REINOLD, NEUMANN, ACE), camera="three_quarter", tags=("cable", "rehab"),
)


reverse_fly = ExerciseDefinition(
    id="reverse_fly", name="Bent-over reverse fly (rear delt raise)", category=Category.UPPER_PULL,
    description="From a hip hinge, light dumbbells are raised out to the sides to shoulder "
                "level with nearly straight arms.",
    setup=("Hinge to ~60 deg, knees soft, spine neutral", "Dumbbells hanging under the "
           "shoulders, palms facing", "Slight elbow bend held throughout"),
    phases=(
        ph("Raise", CON, 1.2, merge(hinge(60, 20)[0], arms(flex=60, abduct=85, elbow=15), grip()),
           pitch=60, cues=("Lead with the elbows out to the sides; thumbs slightly down",)),
        ph("Top", ISO, 0.4, merge(hinge(60, 20)[0], arms(flex=60, abduct=85, elbow=15), grip()),
           pitch=60, cues=("Squeeze the shoulder blades",)),
        ph("Lower", ECC, 1.8, merge(hinge(60, 20)[0], arms(flex=60, abduct=5, elbow=15), grip()),
           pitch=60, cues=("Lower slowly without swinging",)),
    ),
    muscles=(mu("deltoid_posterior", P, 0.9), mu("rhomboids", S, 0.7),
             mu("trapezius_middle", S, 0.7), mu("infraspinatus_teres_minor", S, 0.6),
             mu("trapezius_lower", S, 0.4), mu("erector_spinae", ST, 0.55),
             mu("hamstrings", ST, 0.4), mu("gluteus_maximus", ST, 0.35)),
    equipment=(eq("dumbbell", attach="hand_r", head_radius=5.0),
               eq("dumbbell", attach="hand_l", head_radius=5.0)),
    errors=("Standing up during the raise.", "Swinging the weights.", "Shrugging."),
    physio_notes=("Targets the posterior deltoid and scapular retractors; pairs with "
                  "pressing work for shoulder balance.",),
    sources=(REINOLD, NEUMANN, ACE), camera="front", tags=("dumbbell",),
)


pendlay_row = ExerciseDefinition(
    id="pendlay_row", name="Pendlay row", category=Category.UPPER_PULL,
    description="A barbell row from a dead stop on the floor with the trunk parallel to it. "
                "Every rep starts from rest, so there is no stretch to use and no swing.",
    setup=("Trunk parallel to the floor, bar over the mid-foot",
           "Back flat and held there: the trunk angle does not change",
           "Pull to the lower sternum and return the bar to the floor"),
    orientation="standing", anchor="feet", base_position=(0.0, 0.0, 0.0),
    phases=(
        ph("Pull", CON, 1.0, merge(hinge(85, 20)[0],
                                   arms(flex=80, abduct=25, elbow=110, forearm=-90, wrist=-10),
                                   grip()), pitch=85,
           cues=("Explode the bar to the lower sternum with the trunk still",)),
        ph("Squeeze", ISO, 0.3, merge(hinge(85, 20)[0],
                                      arms(flex=80, abduct=25, elbow=110, forearm=-90,
                                           wrist=-10), grip()), pitch=85,
           cues=("Blades together; do not let the chest drop",)),
        ph("Lower", ECC, 1.0, merge(hinge(85, 20)[0],
                                    arms(flex=88, abduct=8, elbow=8, forearm=-90, wrist=-10),
                                    grip()), pitch=85,
           cues=("Put it back on the floor; do not lower it slowly and hover",)),
        ph("Dead stop", ISO, 0.5, merge(hinge(85, 20)[0],
                                        arms(flex=88, abduct=8, elbow=8, forearm=-90,
                                             wrist=-10), grip()), pitch=85,
           cues=("Let it rest: the next rep starts from nothing",)),
    ),
    muscles=(mu("latissimus_dorsi", P, 0.9), mu("rhomboids", P, 0.8),
             mu("trapezius_middle", P, 0.8), mu("deltoid_posterior", S, 0.65),
             mu("biceps_brachii", S, 0.6), mu("brachialis", S, 0.5),
             mu("trapezius_lower", S, 0.5), mu("infraspinatus_teres_minor", S, 0.45),
             mu("erector_spinae", P, 0.8, note="holds the parallel trunk for every rep"),
             mu("hamstrings", ST, 0.55), mu("gluteus_maximus", ST, 0.5),
             mu("forearm_flexors", ST, 0.6), mu("rectus_abdominis", ST, 0.45)),
    equipment=(eq("barbell", plates=2),),
    errors=("Raising the trunk to help the bar up: the angle is the exercise.",
            "Turning it into a bent-over row by never touching the floor.",
            "Rounding the back to reach the bar at the start."),
    physio_notes=("The dead stop removes the stretch-shortening contribution and the "
                  "cheat, which is why the loads are lower than a bent-over row's and the "
                  "upper back does more of the work.",),
    sources=(SCHOENFELD_ROW, NSCA, EXRX), camera="three_quarter", tags=("barbell", "horizontal-pull"),
)


inverted_row = ExerciseDefinition(
    id="inverted_row", name="Inverted row", category=Category.UPPER_PULL,
    description="A horizontal pull under a fixed bar with the heels on the floor. The load is "
                "set by how horizontal the body is, which makes it the scalable partner to "
                "the pull-up.",
    setup=("Bar at about hip height, heels on the floor, body straight from ear to heel",
           "Shoulders down and back before the pull", "Chest to the bar, elbows about 45 deg"),
    orientation="supine", anchor="feet", base_position=(0.0, 0.0, 0.0),
    phases=(
        ph("Pull", CON, 1.4, merge(pose(hip_flex=-5, knee_flex=5),
                                   arms(flex=95, abduct=30, elbow=115, forearm=-90), grip()),
           cues=("Pull the chest to the bar; keep the hips up",)),
        ph("Top", ISO, 0.5, merge(pose(hip_flex=-5, knee_flex=5),
                                  arms(flex=95, abduct=30, elbow=115, forearm=-90), grip()),
           cues=("Blades together, body still a plank",)),
        ph("Lower", ECC, 1.8, merge(pose(hip_flex=-5, knee_flex=5),
                                    arms(flex=100, abduct=10, elbow=10, forearm=-90), grip()),
           cues=("Lower to straight arms without letting the hips sag",)),
        ph("Hang", ISO, 0.4, merge(pose(hip_flex=-5, knee_flex=5),
                                   arms(flex=100, abduct=10, elbow=10, forearm=-90), grip())),
    ),
    muscles=(mu("latissimus_dorsi", P, 0.8), mu("rhomboids", P, 0.8),
             mu("trapezius_middle", P, 0.8), mu("deltoid_posterior", S, 0.65),
             mu("biceps_brachii", S, 0.65), mu("brachialis", S, 0.55),
             mu("trapezius_lower", S, 0.55), mu("infraspinatus_teres_minor", S, 0.5),
             mu("rectus_abdominis", P, 0.6, note="the body is a plank throughout"),
             mu("gluteus_maximus", S, 0.5), mu("erector_spinae", S, 0.5),
             mu("forearm_flexors", ST, 0.6)),
    # Turned a quarter turn: a supine body's long axis is world X and so is an
    # untouched frame's bar, which put the far upright (x 82.5..87.5) exactly
    # on the athlete's hip at x 83.7 -- the full capsule radius inside him.
    # Rotated, the uprights stand at z +-85, beside him, where a rack's are.
    equipment=(eq("pullup_bar", attach="static", height=70.0,
                  rotation_deg=(0.0, 90.0, 0.0)),),
    errors=("Letting the hips sag so the chest reaches the bar first.",
            "Shrugging rather than retracting.",
            "Setting the bar so high the body is nearly upright, which removes the load."),
    physio_notes=("Lowering the bar or raising the feet makes it harder by moving the body "
                  "toward horizontal; it is the usual way to train a horizontal pull "
                  "without any equipment beyond a bar.",),
    sources=(SCHOENFELD_ROW, NSCA, ACE, EXRX), camera="front",
    tags=("bodyweight", "horizontal-pull"),
)

EXERCISES = (barbell_bent_over_row, pendlay_row, single_arm_dumbbell_row, seated_cable_row,
             inverted_row, face_pull, reverse_fly)
