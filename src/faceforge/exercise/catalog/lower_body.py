"""Squats, deadlifts, lunges and split squats.

Angles: the deep squat reaches hip and knee relative angles of 51-57 degrees
(about 125 degrees of flexion) in trained lifters (SQUAT_KIN); depth bands by
knee flexion are 0-90 partial, 90-110 parallel, 110-135 deep (SQUAT_REVIEW).
The trunk leans 35-45 degrees at the bottom of a high-bar squat (SCHOENFELD).
The conventional deadlift starts with the hips higher than the knees and
moves through 38 deg hip, 33 deg knee and 13 deg ankle in the pull to the knee
(DEADLIFT_SPM).
"""

from __future__ import annotations

from faceforge.exercise.catalog._helpers import (
    ACE, BOREN, CON, CONTRERAS, DEADLIFT_SPM, DIGIOVINE, DISTEFANO, ECC, ECC_CON, ESCAMILLA_DL,
    EXRX, ISO, NSCA, P, S, ST, SCHOENFELD, SQUAT_KIN, SQUAT_REVIEW, TRN, arms, combine, eq, grip,
    hinge, lunge, merge, mu, only, ph, pose, squat, stand,
)
from faceforge.exercise.model import Category, ExerciseDefinition

_ARMS_FORWARD = arms(flex=70, elbow=10)
# Arm bundles are PARTIAL poses (combine, not merge) so they never reset the legs.
# Measured on the rig (wrist-frame probe, 2026-09-10): elbows out and down,
# forearms up to a bar 16 units behind the neck, finger flexion axis 0.5 deg
# from the bar and the palm 21 deg from vertical -- the bar rests on the palms
# with the fingers over it.  The earlier (abduct 45, elbow 70) pose had the
# forearms nearly parallel to the bar, so the bar passed between the fingers.
_BAR_ON_BACK = combine(arms(flex=0, abduct=60, rotate=91, elbow=120, forearm=90, wrist=-70),
                       grip())
_FRONT_RACK = combine(arms(flex=90, abduct=15, elbow=145), grip())
_GOBLET = combine(arms(flex=35, abduct=15, elbow=135, forearm=60), grip())

_SQUAT_MUSCLES = (
    mu("quadriceps", P, 0.95), mu("gluteus_maximus", P, 0.75),
    mu("adductors", S, 0.6, note="adductor magnus extends the hip out of the hole"),
    mu("hamstrings", S, 0.4), mu("soleus", S, 0.4), mu("gastrocnemius", S, 0.3),
    mu("erector_spinae", ST, 0.55), mu("rectus_abdominis", ST, 0.3), mu("obliques", ST, 0.3),
    mu("gluteus_medius", ST, 0.35, note="keeps the knees tracking over the toes"),
    mu("tibialis_anterior", ST, 0.25),
)
_SQUAT_ERRORS = (
    "Knee valgus (knees caving in): cue 'knees out over the toes'; turning the feet out ~30 deg "
    "halves the valgus moment.",
    "Heels lifting: the ankle needs ~35 deg dorsiflexion for a deep squat; raise the heels or "
    "limit depth.",
    "Butt wink (posterior pelvic tilt) below the available hip flexion: stop at the depth "
    "the hips allow.",
    "Excessive forward lean with lumbar flexion: brace and keep the chest up; a rounded "
    "lumbar spine tolerates compression poorly.",
)
_SQUAT_NOTES = (
    "Patellofemoral stress rises steadily from 0 to 90 deg of knee flexion; shallow depths "
    "for anterior knee pain, full depth is tolerated by healthy knees.",
    "Gluteus maximus activity increases ~65 % from partial to parallel depth.",
)


def _squat_phases(bottom_arms: dict, top_arms: dict, hip: float, knee: float, pitch: float,
                  descent: float = 2.0, ascent: float = 1.5):
    bottom, _ = squat(hip, knee, pitch)
    top, _ = stand()
    return (
        ph("Descent", ECC, descent, merge(bottom, bottom_arms), pitch=pitch,
           cues=("Sit the hips back and down; knees track over the toes",
                 "Chest up, spine neutral, weight over mid-foot")),
        ph("Bottom", ISO, 0.4, merge(bottom, bottom_arms), pitch=pitch,
           cues=("Thighs at or below parallel; heels down",)),
        ph("Ascent", CON, ascent, merge(top, top_arms), pitch=0.0,
           cues=("Drive the floor away; hips and chest rise together",
                 "Push the knees out as you stand")),
        ph("Lockout", ISO, 0.5, merge(top, top_arms),
           cues=("Stand tall, squeeze the glutes, breathe",)),
    )


bodyweight_squat = ExerciseDefinition(
    id="bodyweight_squat", name="Bodyweight squat", category=Category.LOWER_BODY,
    description="The air squat: hips and knees flex together to a deep position with the "
                "trunk inclined ~35 deg, then the hips and knees extend to standing.",
    setup=("Feet shoulder-width, toes turned out 10-30 deg", "Arms forward as a counterbalance",
           "Brace the trunk; look ahead"),
    phases=_squat_phases(_ARMS_FORWARD, _ARMS_FORWARD, hip=120, knee=120, pitch=35),
    muscles=_SQUAT_MUSCLES, errors=_SQUAT_ERRORS, physio_notes=_SQUAT_NOTES,
    sources=(SQUAT_KIN, SQUAT_REVIEW, SCHOENFELD, ECC_CON, DIGIOVINE),
    camera="three_quarter", tags=("beginner", "no equipment"),
)

barbell_back_squat = ExerciseDefinition(
    id="barbell_back_squat", name="Barbell back squat (high bar)", category=Category.LOWER_BODY,
    description="The bar rests on the upper trapezius; the lifter squats to full depth with a "
                "35-40 deg trunk lean and stands with hips and chest rising together.",
    setup=("Bar on the upper traps, hands just outside the shoulders, elbows down",
           "Feet shoulder-width, toes out ~15 deg", "Big breath, brace, unrack and step back"),
    phases=_squat_phases(_BAR_ON_BACK, _BAR_ON_BACK, hip=125, knee=125, pitch=40,
                         descent=2.5, ascent=1.8),
    muscles=_SQUAT_MUSCLES + (mu("trapezius_upper", ST, 0.5), mu("forearm_flexors", ST, 0.4)),
    equipment=(eq("barbell", plates=2),),
    errors=_SQUAT_ERRORS + ("Bar drifting forward of mid-foot: keep the shins and trunk "
                            "moving together (trunk and tibia within ~10 deg = balanced hip/"
                            "knee demand).",),
    physio_notes=_SQUAT_NOTES + ("Load intensity from 60 % to 1RM changes depth only slightly "
                                 "(hip 51-56 deg relative angle).",),
    sources=(SQUAT_KIN, SQUAT_REVIEW, SCHOENFELD, NSCA, ECC_CON),
    camera="three_quarter", tags=("barbell",),
)

goblet_squat = ExerciseDefinition(
    id="goblet_squat", name="Goblet squat", category=Category.LOWER_BODY,
    description="A kettlebell or dumbbell held at the chest keeps the trunk upright and "
                "teaches squat depth.",
    setup=("Hold the bell by the horns against the sternum, elbows under it",
           "Feet slightly wider than the hips, toes out"),
    phases=_squat_phases(_GOBLET, _GOBLET, hip=120, knee=125, pitch=25),
    muscles=_SQUAT_MUSCLES + (mu("deltoid_anterior", ST, 0.35), mu("biceps_brachii", ST, 0.35),
                              mu("forearm_flexors", ST, 0.4)),
    equipment=(eq("kettlebell", radius=10.0),),
    errors=_SQUAT_ERRORS[:3], physio_notes=("A common teaching squat: the anterior load "
                                            "counterbalances and limits forward lean.",),
    sources=(SQUAT_REVIEW, ACE, EXRX), tags=("kettlebell", "beginner"),
)

front_squat = ExerciseDefinition(
    id="front_squat", name="Front squat", category=Category.LOWER_BODY,
    description="The bar sits on the front deltoids with the elbows high; the upright trunk "
                "shifts demand toward the quadriceps.",
    setup=("Bar in the front rack, fingertips under it, elbows high",
           "Feet shoulder-width, toes out; brace hard"),
    phases=_squat_phases(_FRONT_RACK, _FRONT_RACK, hip=125, knee=130, pitch=20),
    muscles=(mu("quadriceps", P, 1.0), mu("gluteus_maximus", P, 0.7), mu("adductors", S, 0.5),
             mu("erector_spinae", S, 0.6, note="isometric, holds the upright trunk"),
             mu("hamstrings", S, 0.3), mu("soleus", S, 0.4), mu("rectus_abdominis", ST, 0.4),
             mu("obliques", ST, 0.35), mu("trapezius_upper", ST, 0.5),
             mu("deltoid_anterior", ST, 0.4), mu("gluteus_medius", ST, 0.35)),
    equipment=(eq("barbell", plates=1),),
    errors=("Elbows dropping so the bar rolls forward: keep the upper arms parallel to the "
            "floor.", "Wrists hyperextended under load: open the grip or use straps."),
    physio_notes=("Lower spinal compressive and shear loads than the back squat at the same "
                  "bar load (Gullett 2009).",),
    sources=(SQUAT_REVIEW, SCHOENFELD, NSCA), tags=("barbell",),
)


def _deadlift_phases(pitch_start, knee_start, pitch_mid, knee_mid, sumo: bool = False):
    hip_rot = only(hip_abduct=35, hip_rotate=25) if sumo else {}
    # Shins ~15 deg forward; hip flexion is capped at the rig's 125 deg, which
    # with its short arms leaves the bar a little above the floor at the start.
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
    phases=_deadlift_phases(pitch_start=62, knee_start=88, pitch_mid=40, knee_mid=30),
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
    camera="side", tags=("barbell",),
)

sumo_deadlift = ExerciseDefinition(
    id="sumo_deadlift", name="Sumo deadlift", category=Category.LOWER_BODY,
    description="A wide stance with the feet turned out puts the trunk more upright and "
                "shares the work between hips, adductors and quadriceps.",
    setup=("Wide stance, toes out 30-45 deg, shins vertical", "Grip inside the knees",
           "Knees pushed out over the toes, chest up"),
    phases=_deadlift_phases(pitch_start=45, knee_start=85, pitch_mid=30, knee_mid=35,
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
    sources=(NSCA, ACE, EXRX, ECC_CON), camera="side", tags=("barbell",),
)


def _lunge_definition(id_, name, description, reverse: bool):
    front, back = "r", "l"
    split_stand = lunge(front, 30, 20, -12, 25, pitch=8)[0]
    bottom = lunge(front, 95, 95, -5, 90, pitch=10)[0]
    up_arms = combine(arms(elbow=10), grip(curl=40))
    step_cue = ("Step back with the left leg" if reverse else "Step forward with the right leg")
    return ExerciseDefinition(
        id=id_, name=name, category=Category.LOWER_BODY, description=description,
        setup=("Stand tall, feet hip-width", "Hands on hips or holding dumbbells at the sides"),
        phases=(
            ph("Step", TRN, 0.7, merge(split_stand, up_arms), pitch=8,
               cues=(step_cue, "Keep the hips square")),
            ph("Lower", ECC, 1.2, merge(bottom, up_arms), pitch=10,
               cues=("Drop the back knee toward the floor",
                     "Front shin near vertical, trunk upright")),
            ph("Drive", CON, 1.0, merge(split_stand, up_arms), pitch=8,
               cues=("Push through the front heel",)),
            ph("Return", TRN, 0.7, merge(pose(), up_arms),
               cues=("Bring the feet together",)),
        ),
        muscles=(mu("quadriceps", P, 0.9, side="R"), mu("gluteus_maximus", P, 0.85, side="R"),
                 mu("hamstrings", S, 0.5, side="R"), mu("adductors", S, 0.4),
                 mu("gluteus_medius", S, 0.55, side="R", note="pelvic control"),
                 mu("hip_flexors", S, 0.4, side="L", note="stretched on the back leg"),
                 mu("gastrocnemius", S, 0.4), mu("erector_spinae", ST, 0.4),
                 mu("rectus_abdominis", ST, 0.3), mu("obliques", ST, 0.3)),
        errors=("Front knee collapsing inward.", "Trunk pitching forward.",
                "Step too short so the heel lifts."),
        physio_notes=("Reverse lunges load the front knee less than forward lunges "
                      "(smaller braking forces).",),
        sources=(NSCA, ACE, DISTEFANO), unilateral=True, camera="three_quarter",
        anchor_side="R",
        tags=("no equipment", "unilateral"),
    )


forward_lunge = _lunge_definition(
    "forward_lunge", "Forward lunge",
    "A step forward into a split stance; both knees flex to ~90 deg and the front leg "
    "drives the body back to standing.", reverse=False)
reverse_lunge = _lunge_definition(
    "reverse_lunge", "Reverse lunge",
    "A step backward into the split stance; the front leg does the work with less "
    "braking load on the knee than the forward lunge.", reverse=True)

bulgarian_split_squat = ExerciseDefinition(
    id="bulgarian_split_squat", name="Bulgarian split squat", category=Category.LOWER_BODY,
    description="A rear-foot-elevated split squat: the front leg squats while the back foot "
                "rests on a bench behind.",
    setup=("Back foot laces-down on the bench, front foot ~2 steps ahead",
           "Dumbbells at the sides; trunk upright to slightly inclined"),
    phases=(
        ph("Lower", ECC, 2.0,
           merge(pose(hip_r_flex=100, knee_r_flex=100, ankle_r_flex=10, hip_l_flex=-15,
                      knee_l_flex=95, ankle_l_flex=-35), arms(elbow=5), grip()), pitch=15,
           cues=("Front knee over mid-foot; back knee drops toward the floor",)),
        ph("Bottom", ISO, 0.3,
           merge(pose(hip_r_flex=100, knee_r_flex=100, ankle_r_flex=10, hip_l_flex=-15,
                      knee_l_flex=95, ankle_l_flex=-35), arms(elbow=5), grip()), pitch=15),
        ph("Drive", CON, 1.4,
           merge(pose(hip_r_flex=25, knee_r_flex=20, ankle_r_flex=5, hip_l_flex=-20,
                      knee_l_flex=70, ankle_l_flex=-35), arms(elbow=5), grip()), pitch=8,
           cues=("Push through the front heel; hips forward",)),
        ph("Top", ISO, 0.3,
           merge(pose(hip_r_flex=25, knee_r_flex=20, ankle_r_flex=5, hip_l_flex=-20,
                      knee_l_flex=70, ankle_l_flex=-35), arms(elbow=5), grip()), pitch=8),
    ),
    muscles=(mu("quadriceps", P, 0.9, side="R"), mu("gluteus_maximus", P, 0.9, side="R"),
             mu("gluteus_medius", S, 0.6, side="R"), mu("hamstrings", S, 0.45, side="R"),
             mu("adductors", S, 0.45, side="R"), mu("hip_flexors", S, 0.35, side="L"),
             mu("erector_spinae", ST, 0.4), mu("rectus_abdominis", ST, 0.3),
             mu("forearm_flexors", ST, 0.4)),
    equipment=(eq("dumbbell", attach="hand_r"), eq("dumbbell", attach="hand_l"),
               eq("bench", attach="static", position=(0.0, 0.0, -118.0), rotation_deg=(0, 90, 0),
                  height=50.0)),
    errors=("Front foot too close to the bench (knee far past the toes).",
            "Hips rotating open."),
    physio_notes=("High gluteus maximus and medius demand on the front leg; useful for "
                  "unilateral strength asymmetries.",),
    sources=(BOREN, NSCA, ACE), unilateral=True, camera="three_quarter", anchor_side="R",
    tags=("dumbbell", "unilateral"),
)

EXERCISES = (bodyweight_squat, barbell_back_squat, goblet_squat, front_squat,
             conventional_deadlift, sumo_deadlift, romanian_deadlift, forward_lunge,
             reverse_lunge, bulgarian_split_squat)
