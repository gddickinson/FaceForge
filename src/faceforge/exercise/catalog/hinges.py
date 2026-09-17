"""Hip hinges: the deadlifts and their variations.

A hinge is hip flexion with the shins near vertical, so the load is carried by
the hip extensors and the spinal erectors rather than the knee.  What separates
the variations is where the range starts and stops -- the floor, a deficit
below it, pins at the knee -- and whether one leg or two is under the lifter.

The squats are in :mod:`faceforge.exercise.catalog.lower_body`, the lunges in
``lunges``.
"""

from __future__ import annotations

from faceforge.exercise.catalog._helpers import (
    ACE, BOREN, CON, CONTRERAS, DEADLIFT_SPM, ECC, ECC_CON, ESCAMILLA_DL, EXRX, ISO, NSCA, P,
    S, ST, arms, eq, grip, merge, mu, only, ph, pose, squat,
)
from faceforge.exercise.model import Category, ExerciseDefinition


def _deadlift_phases(pitch_start, knee_start, pitch_mid, knee_mid, sumo: bool = False):
    hip_rot = only(hip_abduct=35, hip_rotate=25) if sumo else {}
    # Shins ~15 deg forward; hip flexion is capped at the rig's 125 deg.  That
    # cap is NOT what used to leave the bar in the air -- knee flexion was.
    # Measured 2026-09-16, lowest point of the loaded bar at the start against
    # `knee_start`, holding `pitch_start` (and so the trunk angle) fixed:
    #
    #   conventional, pitch 62:  knee  88 -> 17.2   110 -> 7.2   130 -> 2.3
    #   conventional, pitch 58:  knee 120 ->  6.4   125 -> 5.3
    #   sumo,         pitch 45:  knee  85 -> 22.6   115 -> 7.5   125 -> 2.5
    #
    # The trunk angle does not move with the knee at all (59.0 deg at every
    # knee value above), so the depth is free: the old 88/85 simply did not
    # bend the knee enough for a figure whose arms are short for its legs.
    #
    # What DOES cap it is the flat foot.  `flat_foot_ankle` makes the sole flat
    # at ankle = pitch - hip + knee, and `ankle_{s}_flex` runs to 45 deg, so
    # knee <= 45 + hip - pitch.  At hip 125 / pitch 58 that is 112, which is
    # why `knee_start` stops at 110 rather than going further: past it the
    # heels would have to leave the floor, and the constraint solver would
    # silently clamp the pose instead.
    start, _ = squat(min(pitch_start + knee_start - 15, 125), knee_start, pitch_start)
    start = merge(start, arms(flex=pitch_start, abduct=5 if sumo else 0), grip(), hip_rot)
    mid, _ = squat(pitch_mid + knee_mid - 8, knee_mid, pitch_mid)
    mid = merge(mid, arms(flex=pitch_mid, abduct=5 if sumo else 0), grip(), hip_rot)
    top = merge(pose(knee_flex=3), arms(flex=-3), grip(), hip_rot if sumo else {})
    return (
        ph("Pull to the knee", CON, 1.0, mid, pitch=pitch_mid,
           cues=("Push the floor away; hips and shoulders rise at the same rate",
                 "Bar stays in contact with the shins")),
        ph("Lockout", CON, 0.8, top, pitch=0.0,
           cues=("Drive the hips through; stand tall without leaning back",)),
        ph("Top", ISO, 0.5, top, cues=("Squeeze the glutes; do not hyperextend",)),
        ph("Lower to the knee", ECC, 1.0, mid, pitch=pitch_mid,
           cues=("Hinge the hips back first; bar slides down the thighs",)),
        ph("Lower to the floor", ECC, 1.0, start, pitch=pitch_start,
           cues=("Bend the knees once the bar passes them",)),
        ph("Reset", ISO, 0.5, start, pitch=pitch_start,
           cues=("Neutral spine, lats tight, take the slack out of the bar",)),
    )


conventional_deadlift = ExerciseDefinition(
    id="conventional_deadlift", name="Conventional deadlift", category=Category.LOWER_BODY,
    description="A hip-dominant lift from the floor: the trunk starts ~55 deg from vertical "
                "with the hips above the knees, and knees and hips extend together to lockout.",
    setup=("Bar over mid-foot, shins ~2-3 cm from the bar", "Hip-width stance, grip just outside "
           "the legs", "Chest up, lats engaged, neutral spine, hips higher than the knees"),
    phases=_deadlift_phases(pitch_start=58, knee_start=110, pitch_mid=40, knee_mid=30),
    muscles=(mu("gluteus_maximus", P, 0.95), mu("hamstrings", P, 0.8),
             mu("erector_spinae", P, 0.9, note="isometric hold against flexion"),
             mu("quadriceps", S, 0.6), mu("adductors", S, 0.5),
             mu("tibialis_anterior", S, 0.5, note="82 %MVC peak in the first pull"),
             mu("latissimus_dorsi", ST, 0.5, note="keeps the bar close"),
             mu("trapezius_upper", ST, 0.5), mu("trapezius_middle", ST, 0.4),
             mu("forearm_flexors", ST, 0.7, note="grip"), mu("rectus_abdominis", ST, 0.4),
             mu("obliques", ST, 0.4), mu("gluteus_medius", ST, 0.3)),
    equipment=(eq("barbell", plates=2, plate_radius=24.0),),
    errors=("Hips shooting up first, turning the lift into a stiff-leg pull: keep the chest "
            "and hips rising together.", "Rounding the lumbar spine: brace and pull the slack "
            "out before the bar leaves the floor.", "Bar drifting away from the legs.",
            "Hyperextending at lockout."),
    physio_notes=("Biceps femoris peaks at ~78 %MVC and vastus lateralis ~55 % in the first "
                  "pull.", "The more horizontal the trunk, the higher the spinal extensor "
                  "demand; the sumo stance is 5-9 deg more upright."),
    sources=(DEADLIFT_SPM, ESCAMILLA_DL, CONTRERAS, NSCA, ECC_CON),
    camera="three_quarter", tags=("barbell",),
)


sumo_deadlift = ExerciseDefinition(
    id="sumo_deadlift", name="Sumo deadlift", category=Category.LOWER_BODY,
    description="A wide stance with the feet turned out puts the trunk more upright and "
                "shares the work between hips, adductors and quadriceps.",
    setup=("Wide stance, toes out 30-45 deg, shins vertical", "Grip inside the knees",
           "Knees pushed out over the toes, chest up"),
    phases=_deadlift_phases(pitch_start=45, knee_start=122, pitch_mid=30, knee_mid=35,
                            sumo=True),
    muscles=(mu("quadriceps", P, 0.85), mu("gluteus_maximus", P, 0.9), mu("adductors", P, 0.8),
             mu("hamstrings", S, 0.65), mu("erector_spinae", S, 0.75),
             mu("gluteus_medius", S, 0.5), mu("hip_external_rotators", S, 0.5),
             mu("forearm_flexors", ST, 0.7), mu("latissimus_dorsi", ST, 0.4),
             mu("rectus_abdominis", ST, 0.4)),
    equipment=(eq("barbell", plates=2),),
    errors=("Knees caving in: keep them over the toes.", "Hips rising early."),
    physio_notes=("Vastus lateralis peaks higher in sumo (63 %MVC) than conventional (55 %).",),
    sources=(DEADLIFT_SPM, ESCAMILLA_DL, NSCA), camera="front", tags=("barbell",),
)


romanian_deadlift = ExerciseDefinition(
    id="romanian_deadlift", name="Romanian deadlift", category=Category.LOWER_BODY,
    description="A pure hip hinge from standing: the bar slides down the thighs with soft "
                "knees until the hamstrings are loaded, then the hips drive forward.",
    setup=("Stand tall with the bar at the thighs, overhand grip", "Soft knees (~15 deg)",
           "Shoulders back, lats tight"),
    phases=(
        ph("Hinge down", ECC, 2.5, merge(squat(95, 15, 75)[0], arms(flex=75), grip()), pitch=75,
           cues=("Push the hips back; the bar stays against the legs",
                 "Stop when the hamstrings are stretched, spine neutral")),
        ph("Bottom", ISO, 0.5, merge(squat(95, 15, 75)[0], arms(flex=75), grip()), pitch=75,
           cues=("Shins vertical, back flat",)),
        ph("Drive up", CON, 1.5, merge(pose(knee_flex=15), arms(flex=0), grip()),
           cues=("Squeeze the glutes and push the hips forward",)),
        ph("Top", ISO, 0.5, merge(pose(knee_flex=15), arms(flex=0), grip()),
           cues=("Stand tall; no lean back",)),
    ),
    muscles=(mu("hamstrings", P, 0.95), mu("gluteus_maximus", P, 0.9),
             mu("erector_spinae", S, 0.7, note="isometric"), mu("adductors", S, 0.4),
             mu("forearm_flexors", ST, 0.6), mu("trapezius_upper", ST, 0.4),
             mu("latissimus_dorsi", ST, 0.4), mu("rectus_abdominis", ST, 0.3),
             mu("gluteus_medius", ST, 0.3)),
    equipment=(eq("barbell", plates=1),),
    errors=("Bending the knees into a squat.", "Rounding the back to reach lower.",
            "Bar drifting away from the legs."),
    physio_notes=("Eccentric hamstring loading; a staple for hamstring strain rehabilitation "
                  "progression.",),
    sources=(NSCA, ACE, EXRX, ECC_CON), camera="three_quarter", tags=("barbell",),
)


deficit_deadlift = ExerciseDefinition(
    id="deficit_deadlift", name="Deficit deadlift", category=Category.LOWER_BODY,
    description="A conventional deadlift standing on a low platform, so the bar starts below "
                "the usual height. The extra range is all at the hardest part of the lift.",
    setup=("Stand on a 5 to 10 cm platform with the bar at normal height",
           "Set the back before the bar moves: the deeper start punishes a rounded one",
           "Everything else is a conventional deadlift"),
    phases=_deadlift_phases(pitch_start=82, knee_start=75, pitch_mid=55, knee_mid=25),
    muscles=(mu("gluteus_maximus", P, 0.95), mu("hamstrings", P, 0.9),
             mu("erector_spinae", P, 0.9, note="the deficit lengthens its hardest range"),
             mu("quadriceps", P, 0.7, note="more knee flexion at the start than a floor pull"),
             mu("adductors", S, 0.5), mu("trapezius_middle", S, 0.55),
             mu("latissimus_dorsi", S, 0.6), mu("forearm_flexors", S, 0.7),
             mu("rectus_abdominis", ST, 0.5), mu("obliques", ST, 0.45)),
    equipment=(eq("barbell", plates=2),),
    errors=("Adding depth before the back can hold a neutral position at normal height.",
            "A deficit so high the hips shoot up to start the bar.",
            "Keeping the same load as a floor deadlift."),
    physio_notes=("The deficit adds range at the bottom, where the moment arm on the hips "
                  "and low back is longest, so loads are lower than a floor pull's.",),
    sources=(DEADLIFT_SPM, ESCAMILLA_DL, NSCA, EXRX), tags=("barbell", "posterior-chain"),
)


rack_pull = ExerciseDefinition(
    id="rack_pull", name="Rack pull", category=Category.LOWER_BODY,
    description="A deadlift started from pins at about knee height. The bottom third is gone, "
                "which leaves the lockout and lets the load go up.",
    setup=("Pins set so the bar starts at or just below the knee",
           "Shins vertical, bar against the legs, lats set",
           "Push the floor away and finish the hips: do not lean back"),
    phases=(
        ph("Pull", CON, 1.2, merge(pose(knee_flex=3), arms(flex=-3), grip()),
           cues=("Drive the hips through to the bar",)),
        ph("Lockout", ISO, 0.6, merge(pose(knee_flex=3), arms(flex=-3), grip()),
           cues=("Stand tall; glutes locked, ribs down, no lay-back",)),
        ph("Lower", ECC, 1.4, merge(squat(72, 22, 50)[0], arms(flex=50), grip()), pitch=50,
           cues=("Hips back to return the bar to the pins",)),
        ph("Pins", ISO, 0.5, merge(squat(72, 22, 50)[0], arms(flex=50), grip()), pitch=50,
           cues=("Let it settle; the next rep starts from a dead stop",)),
    ),
    muscles=(mu("gluteus_maximus", P, 0.9), mu("erector_spinae", P, 0.85),
             mu("hamstrings", P, 0.7), mu("trapezius_middle", P, 0.7),
             mu("trapezius_upper", S, 0.6), mu("latissimus_dorsi", S, 0.6),
             mu("rhomboids", S, 0.55),
             mu("forearm_flexors", P, 0.85, note="the load is usually grip-limited"),
             mu("quadriceps", S, 0.4), mu("rectus_abdominis", ST, 0.5),
             mu("obliques", ST, 0.4)),
    equipment=(eq("barbell", plates=2),),
    errors=("Leaning back at the top, which loads the lumbar spine and proves nothing.",
            "Bouncing the bar off the pins.",
            "Treating the heavier load as a deadlift number."),
    physio_notes=("Removing the bottom third removes the range where the hips and back are "
                  "at their longest moment arm, so loads well above a full deadlift are "
                  "normal; the upper back and grip usually become the limit.",),
    sources=(DEADLIFT_SPM, NSCA, EXRX), camera="three_quarter", tags=("barbell", "posterior-chain"),
)


single_leg_romanian_deadlift = ExerciseDefinition(
    id="single_leg_romanian_deadlift", name="Single-leg Romanian deadlift",
    category=Category.LOWER_BODY,
    description="A hinge on one leg with the other extending behind as a counterweight. The "
                "balance demand makes it a hip-stability exercise as much as a hamstring one.",
    setup=("Stand on one leg with a soft knee", "Hinge at the hip; the free leg extends behind "
           "in line with the trunk", "Keep the hips square: the free hip must not open up"),
    phases=(
        ph("Hinge", ECC, 2.2, merge(pose(hip_r_flex=80, knee_r_flex=18, hip_l_flex=-27,
                                         knee_l_flex=10),
                                    arms(flex=70, elbow=8), grip()), pitch=80,
           cues=("Hips back, back flat, free leg rising as the trunk falls",)),
        ph("Bottom", ISO, 0.5, merge(pose(hip_r_flex=80, knee_r_flex=18, hip_l_flex=-27,
                                          knee_l_flex=10),
                                     arms(flex=70, elbow=8), grip()), pitch=80,
           cues=("Trunk and free leg in one line; hips level",)),
        ph("Stand", CON, 1.6, merge(pose(hip_r_flex=3, knee_r_flex=8, hip_l_flex=-5,
                                         knee_l_flex=10), arms(flex=3, elbow=8), grip()),
           cues=("Drive the standing hip forward to stand tall",)),
        ph("Top", ISO, 0.4, merge(pose(hip_r_flex=3, knee_r_flex=8, hip_l_flex=-5,
                                       knee_l_flex=10), arms(flex=3, elbow=8), grip())),
    ),
    muscles=(mu("hamstrings", P, 0.9, note="of the standing leg"),
             mu("gluteus_maximus", P, 0.85),
             mu("gluteus_medius", P, 0.8, note="keeps the pelvis level on one leg"),
             mu("erector_spinae", P, 0.75), mu("hip_external_rotators", S, 0.6),
             mu("obliques", S, 0.55, note="resists the pelvis rotating open"),
             mu("adductors", S, 0.45), mu("tibialis_anterior", S, 0.45, note="balance"),
             mu("quadriceps", S, 0.35), mu("forearm_flexors", ST, 0.5)),
    equipment=(eq("kettlebell", attach="hand_l", radius=10.0),),
    errors=("Letting the free hip rotate open, which turns it into a side bend.",
            "Rounding the back to reach lower.",
            "Locking the standing knee."),
    physio_notes=("A frontal- and transverse-plane task as much as a sagittal one: gluteus "
                  "medius and the deep rotators hold the pelvis while the hamstrings do the "
                  "hinge.",),
    sources=(BOREN, CONTRERAS, NSCA, ACE), camera="side",
    tags=("kettlebell", "unilateral", "posterior-chain"),
)

EXERCISES = (conventional_deadlift, sumo_deadlift, romanian_deadlift, deficit_deadlift,
             rack_pull, single_leg_romanian_deadlift)
