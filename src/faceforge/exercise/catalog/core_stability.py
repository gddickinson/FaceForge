"""Trunk: planks, sit-up, crunch, dead bug, bird dog, twist, knee raise, Pallof press.

The rig's arms and head hang off the pelvis root rather than the thoracic
spine, so trunk curls are shown as whole-body pitch about a chosen pivot
(the hips for a sit-up, the lower chest for a crunch) with a little true
vertebral flexion on top.
"""

from __future__ import annotations

from faceforge.exercise.catalog._helpers import (
    ACE, CON, ECC, EKSTROM, ESCAMILLA_ABS, EXRX, HIPS, ISO, LOWER_CHEST, MCGILL, NEUMANN, P, S,
    SNARR, ST, TRN, arms, eq, grip, merge, mu, only, ph, pose,
)
from faceforge.exercise.model import Category, ExerciseDefinition

_MAT = (eq("mat", attach="static"),)
_PLANK_ARMS = arms(flex=78, abduct=0, elbow=90)   # upper arm vertical under a 12 deg head-up incline
_PLANK = merge(pose(ankle_flex=45), _PLANK_ARMS)
_PRONE_REST = merge(pose(ankle_flex=45, hip_flex=8, knee_flex=10), arms(flex=40, abduct=15, elbow=100))

front_plank = ExerciseDefinition(
    id="front_plank", name="Front plank", category=Category.CORE,
    description="An isometric hold on the forearms and toes with the body in a straight line; "
                "the abdominals prevent the hips from sagging.",
    setup=("Elbows under the shoulders, forearms parallel", "Toes tucked, legs straight",
           "Squeeze the glutes; tuck the ribs; look at the floor"),
    orientation="prone", anchor="hands", base_position=(-85.0, 30.0, 0.0),
    phases=(
        # Forearms on the floor: the shoulders sit one upper-arm length up, so
        # the straight body is inclined ~11 deg (negative pitch = head end up).
        ph("Lift into plank", CON, 1.5, _PLANK, pitch=-12,
           cues=("Push the floor away; straight line from ears to ankles",)),
        ph("Hold", ISO, 8.0, _PLANK, pitch=-12,
           cues=("Breathe; ribs down, glutes tight, no sag",)),
        ph("Lower", ECC, 1.5, _PRONE_REST, cues=("Knees then hips to the floor",)),
        ph("Rest", TRN, 1.0, _PRONE_REST),
    ),
    muscles=(mu("rectus_abdominis", P, 0.6), mu("obliques", P, 0.6),
             mu("transversus_abdominis", P, 0.6), mu("erector_spinae", S, 0.3),
             mu("gluteus_maximus", S, 0.4), mu("quadriceps", S, 0.4),
             mu("deltoid_anterior", S, 0.5), mu("serratus_anterior", S, 0.5),
             mu("hip_flexors", S, 0.3), mu("pectoralis_major", ST, 0.3)),
    equipment=_MAT,
    errors=("Hips sagging (lumbar extension).", "Hips piked up.", "Holding the breath.",
            "Head hanging."),
    physio_notes=("Rectus abdominis and external oblique ~40-50 %MVIC in the standard plank "
                  "(Snarr 2014); progress by narrowing the base or lifting a limb.",),
    sources=(SNARR, MCGILL, EKSTROM), camera="side", tags=("no equipment", "isometric", "rehab"),
)

side_plank = ExerciseDefinition(
    id="side_plank", name="Side plank (side bridge)", category=Category.CORE,
    description="Supported on one forearm and the side of the feet, the hips lift so the body "
                "forms a straight line; the lateral trunk holds it there.",
    setup=("Lie on the left side, left elbow under the shoulder", "Feet stacked or staggered",
           "Top hand on the hip"),
    orientation="side", anchor="none", base_position=(-85.0, 26.0, 0.0),
    phases=(
        # Lying on the left side, the head end rises (positive roll) about the
        # feet until the left elbow, one upper-arm length below the shoulder,
        # rests on the floor: asin(38 / 175) ~ 12.5 deg.
        ph("Lift", CON, 1.5,
           merge(pose(), arms(flex=0, abduct=90, elbow=90, side="l"), arms(flex=0, abduct=0, elbow=70, side="r")),
           roll=10.0, pivot=(0.0, 0.0, -190.0),
           cues=("Lift the hips until the body is straight from head to feet",)),
        ph("Hold", ISO, 6.0,
           merge(pose(), arms(flex=0, abduct=90, elbow=90, side="l"), arms(flex=0, abduct=0, elbow=70, side="r")),
           roll=10.0, pivot=(0.0, 0.0, -190.0), cues=("Hips forward, ribs stacked over the pelvis",)),
        ph("Lower", ECC, 1.5,
           merge(pose(), arms(flex=90, abduct=0, elbow=90, side="l"), arms(flex=0, abduct=0, elbow=70, side="r")),
           roll=0, pivot=(0.0, 0.0, -190.0), cues=("Lower the hips to the floor",)),
        ph("Rest", TRN, 1.0,
           merge(pose(), arms(flex=90, abduct=0, elbow=90, side="l"), arms(flex=0, abduct=0, elbow=70, side="r"))),
    ),
    muscles=(mu("obliques", P, 0.7, side="L"), mu("quadratus_lumborum", P, 0.6, side="L"),
             mu("gluteus_medius", P, 0.6, side="L"), mu("transversus_abdominis", S, 0.5),
             mu("rectus_abdominis", S, 0.4), mu("deltoid_lateral", S, 0.5, side="L"),
             mu("adductors", S, 0.35), mu("erector_spinae", ST, 0.3)),
    equipment=_MAT,
    errors=("Hips dropping or rotating back.", "Shoulder shrugging toward the ear.",
            "Elbow not under the shoulder."),
    physio_notes=("The side bridge loads the lateral trunk with low spinal compression "
                  "(McGill); gluteus medius ~40 %MVIC (Ekstrom 2007).",),
    sources=(MCGILL, EKSTROM, SNARR), unilateral=True, camera="front", tags=("no equipment", "isometric", "rehab"),
)

_SITUP_LEGS = only(hip_flex=45, knee_flex=90, ankle_flex=-30)
_SITUP_ARMS = arms(flex=110, abduct=10, elbow=140)

sit_up = ExerciseDefinition(
    id="sit_up", name="Sit-up", category=Category.CORE,
    description="From supine with the knees bent, the trunk curls and then rotates about the "
                "hips to sit up; the hip flexors take over once the trunk is off the floor.",
    setup=("Knees bent ~90 deg, feet flat", "Hands across the chest or fingertips at the "
           "temples", "Chin tucked"),
    orientation="supine", anchor="none", base_position=(-85.0, 15.0, 0.0),
    phases=(
        ph("Curl and sit up", CON, 1.5,
           merge(pose(spine_flex=25), only(hip_flex=115, knee_flex=90, ankle_flex=-30), _SITUP_ARMS),
           pitch=70, pivot=HIPS, cues=("Curl the ribs to the pelvis first, then come up",
                                       "Exhale on the way up")),
        ph("Top", ISO, 0.4,
           merge(pose(spine_flex=25), only(hip_flex=115, knee_flex=90, ankle_flex=-30), _SITUP_ARMS),
           pitch=70, pivot=HIPS),
        ph("Lower", ECC, 2.0, merge(pose(), _SITUP_LEGS, _SITUP_ARMS), pitch=0, pivot=HIPS,
           cues=("Lower one vertebra at a time",)),
        ph("Rest", TRN, 0.5, merge(pose(), _SITUP_LEGS, _SITUP_ARMS)),
    ),
    muscles=(mu("rectus_abdominis", P, 0.85), mu("hip_flexors", P, 0.8),
             mu("obliques", S, 0.6), mu("tensor_fasciae_latae", S, 0.35),
             mu("quadriceps", S, 0.3, note="rectus femoris"), mu("transversus_abdominis", S, 0.4)),
    equipment=_MAT,
    errors=("Pulling on the neck.", "Jerking up with momentum.", "Feet anchored so the hip "
            "flexors do all the work."),
    physio_notes=("The sit-up loads the lumbar spine with high compression (McGill); the "
                  "curl-up is preferred for low-back patients.", "Hip flexor (iliopsoas) "
                  "activity is substantial in the second half of the movement (Escamilla 2006)."),
    sources=(ESCAMILLA_ABS, MCGILL, ACE), camera="side", tags=("no equipment",),
)

crunch = ExerciseDefinition(
    id="crunch", name="Crunch (curl-up)", category=Category.CORE,
    description="Only the head and shoulder blades leave the floor: a short thoracic curl "
                "that isolates the abdominals with little hip flexor involvement.",
    setup=("Knees bent, feet flat", "Hands beside the head or across the chest",
           "Lower back stays in contact with the floor"),
    orientation="supine", anchor="none", base_position=(-85.0, 15.0, 0.0),
    phases=(
        ph("Curl up", CON, 1.2, merge(pose(spine_flex=30), only(hip_flex=65, knee_flex=90, ankle_flex=-30), _SITUP_ARMS),
           pitch=20, pivot=HIPS, cues=("Lift the shoulder blades off the floor; ribs "
                                       "toward the pelvis", "Exhale")),
        ph("Hold", ISO, 0.5, merge(pose(spine_flex=30), only(hip_flex=65, knee_flex=90, ankle_flex=-30), _SITUP_ARMS),
           pitch=20, pivot=HIPS),
        ph("Lower", ECC, 1.5, merge(pose(), _SITUP_LEGS, _SITUP_ARMS), pitch=0, pivot=HIPS,
           cues=("Lower slowly; keep tension",)),
    ),
    muscles=(mu("rectus_abdominis", P, 0.8), mu("obliques", S, 0.5),
             mu("transversus_abdominis", S, 0.4), mu("hip_flexors", ST, 0.2)),
    equipment=_MAT,
    errors=("Pulling the head forward.", "Lifting the whole trunk (becomes a sit-up)."),
    physio_notes=("The McGill curl-up keeps lumbar loads low while training the rectus "
                  "abdominis.",),
    sources=(ESCAMILLA_ABS, MCGILL), camera="side", tags=("no equipment", "rehab"),
)

_DB_START = merge(pose(hip_flex=90, knee_flex=90), arms(flex=90, elbow=5))

dead_bug = ExerciseDefinition(
    id="dead_bug", name="Dead bug", category=Category.CORE,
    description="Supine with hips and knees at 90 deg and the arms vertical, one leg and the "
                "opposite arm extend while the lower back stays pressed to the floor.",
    setup=("Ribs down, lower back flat on the floor", "Hips and knees at 90 deg, arms "
           "toward the ceiling"),
    orientation="supine", anchor="none", base_position=(-85.0, 15.0, 0.0),
    phases=(
        ph("Extend right leg, left arm", CON, 1.5,
           merge(pose(hip_r_flex=20, knee_r_flex=10, hip_l_flex=90, knee_l_flex=90),
                 arms(flex=90, elbow=5, side="r"), arms(flex=165, elbow=5, side="l")),
           cues=("Reach the heel and hand away; exhale; lower back stays down",)),
        ph("Return", ECC, 1.5, _DB_START, cues=("Bring them back without arching",)),
        ph("Extend left leg, right arm", CON, 1.5,
           merge(pose(hip_l_flex=20, knee_l_flex=10, hip_r_flex=90, knee_r_flex=90),
                 arms(flex=90, elbow=5, side="l"), arms(flex=165, elbow=5, side="r"))),
        ph("Return", ECC, 1.5, _DB_START),
    ),
    muscles=(mu("rectus_abdominis", P, 0.6), mu("transversus_abdominis", P, 0.6),
             mu("obliques", S, 0.5), mu("hip_flexors", S, 0.5),
             mu("deltoid_anterior", S, 0.3), mu("quadriceps", S, 0.3)),
    equipment=_MAT,
    errors=("Lower back arching as the leg extends.", "Moving fast / holding the breath."),
    physio_notes=("An anti-extension exercise: progress by extending the leg lower.",),
    sources=(MCGILL, ACE), camera="three_quarter", tags=("no equipment", "rehab"),
)

# On all fours the dorsum of each foot lies on the floor: plantarflexed ~25 deg.
_QUAD = merge(pose(hip_flex=90, knee_flex=90, ankle_flex=-40), arms(flex=90, abduct=5, elbow=0))

bird_dog = ExerciseDefinition(
    id="bird_dog", name="Bird dog", category=Category.CORE,
    description="On hands and knees, one arm and the opposite leg extend to horizontal while "
                "the trunk and pelvis stay level.",
    setup=("Hands under the shoulders, knees under the hips", "Neutral spine, gaze at the "
           "floor"),
    orientation="prone", anchor="hands", base_position=(-85.0, 70.0, 0.0),
    phases=(
        ph("Extend right arm, left leg", CON, 1.5,
           merge(pose(hip_r_flex=90, knee_r_flex=90, ankle_r_flex=-40, hip_l_flex=-5, knee_l_flex=5,
                      ankle_l_flex=-10),
                 arms(flex=170, abduct=5, elbow=0, side="r"), arms(flex=90, abduct=5, elbow=0, side="l")),
           cues=("Reach the heel back and the hand forward; hips level",)),
        ph("Hold", ISO, 1.5,
           merge(pose(hip_r_flex=90, knee_r_flex=90, ankle_r_flex=-40, hip_l_flex=-5, knee_l_flex=5,
                      ankle_l_flex=-10),
                 arms(flex=170, abduct=5, elbow=0, side="r"), arms(flex=90, abduct=5, elbow=0, side="l")),
           cues=("Do not let the pelvis rotate",)),
        ph("Return", ECC, 1.5, _QUAD, cues=("Back to all fours without shifting the weight",)),
        ph("Extend left arm, right leg", CON, 1.5,
           merge(pose(hip_l_flex=90, knee_l_flex=90, ankle_l_flex=-40, hip_r_flex=-5, knee_r_flex=5,
                      ankle_r_flex=-10),
                 arms(flex=170, abduct=5, elbow=0, side="l"), arms(flex=90, abduct=5, elbow=0, side="r"))),
        ph("Hold", ISO, 1.5,
           merge(pose(hip_l_flex=90, knee_l_flex=90, ankle_l_flex=-40, hip_r_flex=-5, knee_r_flex=5,
                      ankle_r_flex=-10),
                 arms(flex=170, abduct=5, elbow=0, side="l"), arms(flex=90, abduct=5, elbow=0, side="r"))),
        ph("Return", ECC, 1.5, _QUAD),
    ),
    muscles=(mu("erector_spinae", P, 0.6), mu("multifidus", P, 0.6),
             mu("gluteus_maximus", P, 0.6), mu("rectus_abdominis", S, 0.45),
             mu("obliques", S, 0.5), mu("transversus_abdominis", S, 0.5),
             mu("hamstrings", S, 0.4), mu("deltoid_anterior", S, 0.4),
             mu("trapezius_lower", S, 0.35), mu("gluteus_medius", S, 0.4)),
    equipment=_MAT,
    errors=("Lifting the leg above hip height and arching the back.", "Rotating the pelvis.",
            "Rushing."),
    physio_notes=("Multifidus and erector spinae ~30-45 %MVIC with low compression; part of "
                  "the McGill big three.",),
    sources=(MCGILL, EKSTROM), camera="three_quarter", tags=("no equipment", "rehab"),
)

_TWIST_BASE = only(hip_flex=95, knee_flex=55)

russian_twist = ExerciseDefinition(
    id="russian_twist", name="Russian twist", category=Category.CORE,
    description="Seated and leaning back with the trunk braced, the arms and trunk rotate side "
                "to side.",
    setup=("Sit with the knees bent, heels light on the floor", "Lean back ~35 deg with a "
           "straight spine", "Hands together in front of the chest"),
    orientation="seated", anchor="none", base_position=(0.0, 91.0, 0.0),
    phases=(
        ph("Rotate right", CON, 0.8,
           merge(pose(spine_rotation=-25), _TWIST_BASE,
                 arms(flex=45, abduct=-20, elbow=30, side="r"), arms(flex=95, abduct=35, elbow=30, side="l")),
           pitch=-35, pivot=HIPS, cues=("Rotate the ribcage, not just the arms",)),
        ph("Rotate left", CON, 0.8,
           merge(pose(spine_rotation=25), _TWIST_BASE,
                 arms(flex=95, abduct=35, elbow=30, side="r"), arms(flex=45, abduct=-20, elbow=30, side="l")),
           pitch=-35, pivot=HIPS, cues=("Keep the lean; exhale as you turn",)),
    ),
    muscles=(mu("obliques", P, 0.8), mu("rectus_abdominis", S, 0.6),
             mu("hip_flexors", S, 0.5, note="hold the leaned-back position"),
             mu("transversus_abdominis", S, 0.5), mu("erector_spinae", S, 0.4),
             mu("deltoid_anterior", ST, 0.3)),
    equipment=_MAT,
    errors=("Rounding the lumbar spine.", "Moving only the arms."),
    physio_notes=("Loaded lumbar rotation under flexion is poorly tolerated by disc "
                  "patients; prefer anti-rotation (Pallof press) for them.",),
    sources=(ESCAMILLA_ABS, MCGILL, ACE), camera="three_quarter", tags=("no equipment",),
)

hanging_knee_raise = ExerciseDefinition(
    id="hanging_knee_raise", name="Hanging knee raise", category=Category.CORE,
    description="Hanging from a bar, the knees are drawn up toward the chest with a posterior "
                "pelvic tilt at the top.",
    setup=("Overhand grip, shoulders active", "Legs together, body still"),
    orientation="hanging", anchor="hands", anchor_point=(0.0, 275.0, 0.0),
    phases=(
        ph("Raise", CON, 1.2,
           merge(pose(hip_flex=110, knee_flex=100), arms(flex=10, abduct=160, elbow=5, forearm=-70), grip()),
           cues=("Knees to the chest; curl the pelvis up at the top",)),
        ph("Top", ISO, 0.4,
           merge(pose(hip_flex=110, knee_flex=100), arms(flex=10, abduct=160, elbow=5, forearm=-70), grip())),
        ph("Lower", ECC, 1.8,
           merge(pose(hip_flex=5, knee_flex=5), arms(flex=10, abduct=160, elbow=5, forearm=-70), grip()),
           cues=("Lower slowly without swinging",)),
    ),
    muscles=(mu("hip_flexors", P, 0.9), mu("rectus_abdominis", P, 0.8),
             mu("obliques", S, 0.5), mu("tensor_fasciae_latae", S, 0.4),
             mu("quadriceps", S, 0.35), mu("forearm_flexors", ST, 0.7),
             mu("latissimus_dorsi", ST, 0.4)),
    equipment=(eq("pullup_bar", attach="static", height=275.0),),
    errors=("Swinging.", "Lifting the legs without curling the pelvis (all hip flexor)."),
    physio_notes=("Very high rectus abdominis activation in Escamilla 2006; grip and "
                  "shoulder tolerance are the limits.",),
    sources=(ESCAMILLA_ABS, EXRX), camera="side", camera_target=(0.0, 190.0, 0.0),
    tags=("bodyweight", "bar"),
)

pallof_press = ExerciseDefinition(
    id="pallof_press", name="Pallof press (anti-rotation)", category=Category.CORE,
    description="Standing side-on to a cable, the handle is pressed straight out and held; "
                "the trunk resists being turned toward the machine.",
    setup=("Cable at chest height, stand side-on, feet shoulder-width", "Handle at the "
           "sternum with both hands", "Knees soft, brace"),
    phases=(
        ph("Press out", CON, 1.2, merge(pose(knee_flex=10), arms(flex=90, abduct=0, elbow=5), grip()),
           cues=("Press straight out; do not let the trunk turn",)),
        ph("Hold", ISO, 2.0, merge(pose(knee_flex=10), arms(flex=90, abduct=0, elbow=5), grip()),
           cues=("Breathe; hips and shoulders square",)),
        ph("Return", ECC, 1.2, merge(pose(knee_flex=10), arms(flex=30, abduct=0, elbow=120), grip()),
           cues=("Hands back to the sternum under control",)),
    ),
    muscles=(mu("obliques", P, 0.65), mu("transversus_abdominis", P, 0.6),
             mu("rectus_abdominis", S, 0.5), mu("gluteus_medius", S, 0.4),
             mu("erector_spinae", S, 0.4), mu("multifidus", S, 0.4),
             mu("deltoid_anterior", S, 0.4), mu("triceps_brachii", S, 0.4),
             mu("serratus_anterior", S, 0.4), mu("quadratus_lumborum", ST, 0.35)),
    equipment=(eq("cable_handle",),),
    errors=("Trunk rotating toward the cable.", "Shrugging."),
    physio_notes=("An anti-rotation core exercise with negligible spinal motion; suits "
                  "rotation-intolerant backs.",),
    sources=(MCGILL, ACE), camera="front", tags=("cable", "rehab"),
)

EXERCISES = (front_plank, side_plank, sit_up, crunch, dead_bug, bird_dog, russian_twist,
             hanging_knee_raise, pallof_press)
