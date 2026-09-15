"""Static stretches and mobility drills: eight targets, each a reach / hold / release.

A convention worth stating, because it differs from every other module here:
in a stretch the muscle of interest is the one being *lengthened*, not the one
shortening, so it is the one listed as PRIMARY.  The heatmap then colours what
the stretch is for.  Muscles that actually contract to hold the position are
listed as secondary or stabilisers with a note.

Doses follow the ACSM position stand: static holds of 10-30 s, two to three
days a week, to the point of mild tension and never pain.  Static stretching
immediately before a power effort transiently reduces force (Behm 2016), which
is why these belong after training or in their own session.
"""

from __future__ import annotations

from faceforge.exercise.catalog._helpers import (
    ACE, BEHM, CON, ECC, HIPS, ISO, NEUMANN, NSCA, P, S, ST, STRETCH_ACSM, STRETCH_PAGE,
    TRN,
    arms, eq, flat_foot_ankle, merge, mu, only, ph, pose,
)
from faceforge.exercise.model import Category, ExerciseDefinition

_MAT = (eq("mat", attach="static"),)
_STAND = merge(pose(knee_flex=5), arms(flex=5, abduct=8, elbow=5))

# ── Standing ───────────────────────────────────────────────────────────

_FOLD_PITCH = 85.0
_FOLD = merge(pose(hip_flex=_FOLD_PITCH, knee_flex=8,
                   ankle_flex=flat_foot_ankle(_FOLD_PITCH, _FOLD_PITCH, 8)),
              arms(flex=80, abduct=8, elbow=15))
_FOLD_HALF = merge(pose(hip_flex=45, knee_flex=10,
                        ankle_flex=flat_foot_ankle(45, 45, 10)),
                   arms(flex=40, abduct=8, elbow=20))

standing_forward_fold = ExerciseDefinition(
    id="standing_forward_fold", name="Standing forward fold (hamstring stretch)",
    category=Category.MOBILITY,
    description="A hip hinge to full fold with the knees nearly straight: the hamstrings are "
                "lengthened by the pelvis tipping forward, not by rounding the back.",
    setup=("Feet hip width, a soft bend in the knees",
           "Hinge from the hips: tip the pelvis, keep the spine long",
           "Let the head and arms hang; hold 20-30 s to mild tension, never pain"),
    anchor="feet",
    phases=(
        ph("Fold", ECC, 3.0, _FOLD, pitch=_FOLD_PITCH,
           cues=("Send the sitting bones up and back as the chest goes down",
                 "If the back rounds, bend the knees -- the stretch stays in the hamstrings")),
        ph("Hold", ISO, 20.0, _FOLD, pitch=_FOLD_PITCH,
           cues=("Breathe out slowly and let the fold deepen a little on each exhale",)),
        ph("Half lift", TRN, 2.0, _FOLD_HALF, pitch=45,
           cues=("Flat back halfway up, hands on the shins",)),
        ph("Stand", CON, 2.0, _STAND, cues=("Roll up or hinge up with a braced trunk",)),
    ),
    muscles=(mu("hamstrings", P, 0.9, note="the target: lengthened at the hip and the knee"),
             mu("gluteus_maximus", P, 0.7, note="lengthened with the hamstrings"),
             mu("erector_spinae", S, 0.6, note="lengthened, and works on the way up"),
             mu("gastrocnemius", S, 0.5, note="lengthened at the knee-straight end"),
             mu("quadriceps", S, 0.5, note="holds the knees straight"),
             mu("hip_flexors", S, 0.45, note="pulls the pelvis into the fold"),
             mu("multifidus", ST, 0.4), mu("soleus", ST, 0.4),
             mu("tibialis_anterior", ST, 0.35)),
    equipment=_MAT,
    errors=("Rounding the lumbar spine to reach the floor, which moves the stretch off the "
            "hamstrings and onto the back.",
            "Bouncing.  Static holds, not ballistic ones.",
            "Locking the knees back hard."),
    physio_notes=("Hamstring length limits the pelvic tilt; the visible measure is how far "
                  "the pelvis rotates, not how close the fingers get to the floor.",),
    sources=(STRETCH_ACSM, STRETCH_PAGE, NEUMANN, ACE), camera="side", default_reps=1,
    tags=("stretch", "isometric", "no equipment"),
)

_QUAD_ST = merge(pose(hip_r_flex=-22, knee_r_flex=140, ankle_r_flex=-25,
                      hip_l_flex=4, knee_l_flex=6),
                 only(shoulder_r_flex=-45, shoulder_r_abduct=12, elbow_r_flex=70,
                      shoulder_l_flex=15, shoulder_l_abduct=45, elbow_l_flex=8))
_QUAD_DOWN = merge(pose(hip_r_flex=8, knee_r_flex=12, hip_l_flex=4, knee_l_flex=6),
                   arms(flex=10, abduct=12, elbow=8))

standing_quad_stretch = ExerciseDefinition(
    id="standing_quad_stretch", name="Standing quadriceps stretch", category=Category.MOBILITY,
    description="Standing on one leg, the other heel is drawn toward the buttock with the hip "
                "held in extension, so rectus femoris is lengthened over both its joints.",
    setup=("Stand tall; hold something for balance if needed",
           "Take the ankle, not the toes, and draw the heel to the buttock",
           "Keep the knees together and the pelvis tucked under"),
    anchor="feet", anchor_side="L", unilateral=True,
    phases=(
        ph("Take the ankle", CON, 2.0, _QUAD_ST,
           cues=("Knee pointing down and back; tuck the tailbone under",)),
        ph("Hold", ISO, 20.0, _QUAD_ST,
           cues=("Squeeze the glute on the stretched side -- it deepens the hip extension",
                 "Stand tall; do not let the low back arch")),
        ph("Release", ECC, 1.5, _QUAD_DOWN, cues=("Let the foot down under control",)),
        ph("Rest", TRN, 1.5, _QUAD_DOWN),
    ),
    muscles=(mu("quadriceps", P, 0.9, side="R",
                note="the target; rectus femoris crosses the hip, so hip extension matters"),
             mu("hip_flexors", P, 0.7, side="R", note="lengthened with the hip extended"),
             mu("tibialis_anterior", S, 0.45, side="R", note="lengthened at the ankle"),
             mu("gluteus_maximus", S, 0.6, side="R", note="works to hold the hip extended"),
             mu("hamstrings", S, 0.5, side="R", note="works to bend the knee"),
             mu("gluteus_medius", S, 0.6, side="L", note="holds the standing pelvis level"),
             mu("foot_intrinsics", ST, 0.5, side="L"), mu("soleus", ST, 0.45, side="L"),
             mu("rectus_abdominis", ST, 0.45, note="keeps the pelvis tucked"),
             mu("biceps_brachii", ST, 0.35)),
    equipment=(),
    errors=("Letting the knee drift forward, which loses the hip extension and most of the "
            "stretch on rectus femoris.",
            "Arching the low back instead of tucking the pelvis.",
            "Pulling on the toes rather than the ankle."),
    physio_notes=("Rectus femoris is the only quadriceps head crossing the hip; without hip "
                  "extension the stretch reaches only the three vasti.",),
    sources=(STRETCH_ACSM, NEUMANN, ACE), camera="side", default_reps=1,
    tags=("stretch", "unilateral", "no equipment"),
)

_CALF_PITCH = 12.0
_CALF = merge(pose(hip_r_flex=40, knee_r_flex=55,
                   ankle_r_flex=flat_foot_ankle(_CALF_PITCH, 40, 55),
                   hip_l_flex=-12, knee_l_flex=5, ankle_l_flex=32),
              arms(flex=95, abduct=15, elbow=15))
_CALF_BENT = merge(pose(hip_r_flex=40, knee_r_flex=55,
                        ankle_r_flex=flat_foot_ankle(_CALF_PITCH, 40, 55),
                        hip_l_flex=-8, knee_l_flex=25, ankle_l_flex=40),
                   arms(flex=95, abduct=15, elbow=15))

calf_stretch = ExerciseDefinition(
    id="calf_stretch", name="Standing calf stretch (wall)", category=Category.MOBILITY,
    description="A split stance with the hands on a wall and the back heel pressed down. "
                "Straight back knee lengthens gastrocnemius; bending it shifts the stretch "
                "to soleus.",
    setup=("Hands on the wall, one foot well back, both feet pointing forward",
           "Press the back heel into the floor with the back knee straight",
           "Then bend the back knee slightly, heel still down, for soleus"),
    anchor="feet", unilateral=True,
    phases=(
        ph("Press the heel down", ECC, 2.5, _CALF, pitch=_CALF_PITCH,
           cues=("Back leg straight, heel down, hips forward",)),
        ph("Hold (gastrocnemius)", ISO, 15.0, _CALF, pitch=_CALF_PITCH,
           cues=("Straight back knee keeps the stretch on the two-joint calf",)),
        ph("Bend the back knee", TRN, 2.0, _CALF_BENT, pitch=_CALF_PITCH,
           cues=("Soften the back knee, keep the heel down: now it is soleus",)),
        ph("Hold (soleus)", ISO, 15.0, _CALF_BENT, pitch=_CALF_PITCH),
    ),
    muscles=(mu("gastrocnemius", P, 0.85, side="L",
                note="the target with the knee straight: it crosses the knee too"),
             mu("soleus", P, 0.8, side="L", note="the target once the knee bends"),
             mu("tibialis_posterior", S, 0.5, side="L", note="lengthened with them"),
             mu("peroneals", S, 0.45, side="L"),
             mu("foot_intrinsics", S, 0.45, side="L", note="the plantar fascia pulls too"),
             mu("tibialis_anterior", S, 0.5, side="L", note="works to hold the dorsiflexion"),
             mu("quadriceps", S, 0.5, side="R"), mu("gluteus_maximus", ST, 0.4),
             mu("deltoid_anterior", ST, 0.35), mu("triceps_brachii", ST, 0.35)),
    equipment=(),
    errors=("The back heel lifting, which ends the stretch.",
            "The back foot turning out, which takes the stretch out of the calf and into "
            "the arch.",
            "Only ever doing the straight-knee version and never reaching soleus."),
    physio_notes=("Gastrocnemius crosses the knee and the ankle, soleus only the ankle: the "
                  "two versions are not interchangeable, and restricted dorsiflexion is a "
                  "common finding behind squat and running complaints.",),
    sources=(STRETCH_ACSM, NEUMANN, ACE, NSCA), camera="side", default_reps=1,
    tags=("stretch", "no equipment"),
)

#: Same as the triangle: the lean is a wrapper roll, negative being to the
#: body's left, away from the raised arm.  `spine_lat_bend` alone rendered as
#: a man standing perfectly upright with one arm in the air.
_BEND_ROLL = -28.0
_SIDE_BEND = merge(pose(spine_lat_bend=-12, knee_flex=5, hip_flex=3),
                   only(shoulder_r_flex=15, shoulder_r_abduct=165, elbow_r_flex=8,
                        shoulder_l_flex=5, shoulder_l_abduct=15, elbow_l_flex=10))
_SIDE_UP = merge(pose(knee_flex=5),
                 only(shoulder_r_flex=10, shoulder_r_abduct=150, elbow_r_flex=8,
                      shoulder_l_flex=5, shoulder_l_abduct=12, elbow_l_flex=10))

overhead_side_bend = ExerciseDefinition(
    id="overhead_side_bend", name="Overhead side bend (lateral trunk stretch)",
    category=Category.MOBILITY,
    description="Standing with one arm overhead, the trunk bends away from that side: the "
                "whole lateral chain from the hip to the armpit is lengthened.",
    setup=("Feet hip width, weight even on both feet",
           "Reach one arm overhead and lean away from it",
           "Bend sideways only -- do not let the trunk rotate or fold forward"),
    anchor="feet", unilateral=True,
    phases=(
        ph("Bend over", ECC, 2.5, _SIDE_BEND, roll=_BEND_ROLL, pivot=HIPS,
           cues=("Reach up and over; keep both feet flat",
                 "Lengthen the upper side rather than collapsing into the lower one")),
        ph("Hold", ISO, 18.0, _SIDE_BEND, roll=_BEND_ROLL, pivot=HIPS,
           cues=("Breathe into the upper ribs; the stretch should be a broad pull, not a pinch",)),
        ph("Come up", CON, 2.0, _SIDE_UP, cues=("Return to upright with the arm still up",)),
        ph("Rest", TRN, 1.2, _STAND),
    ),
    muscles=(mu("quadratus_lumborum", P, 0.85, side="R", note="the target"),
             mu("latissimus_dorsi", P, 0.75, side="R",
                note="lengthened over the shoulder and the trunk at once"),
             mu("obliques", P, 0.7, side="R", note="lengthened"),
             mu("erector_spinae", S, 0.5, side="R"),
             mu("obliques", S, 0.55, side="L", note="works to hold the bend"),
             mu("quadratus_lumborum", S, 0.5, side="L"),
             mu("gluteus_medius", S, 0.5, note="keeps the pelvis level"),
             mu("deltoid_anterior", ST, 0.4, side="R"),
             mu("serratus_anterior", ST, 0.4, side="R")),
    equipment=(),
    errors=("Rotating or folding forward instead of bending sideways.",
            "Letting the hip swing out so the pelvis, not the trunk, does the bending.",
            "Bouncing at the end of the range."),
    physio_notes=("Reaching the arm overhead first puts latissimus dorsi on stretch before "
                  "the trunk bends, which is what makes this more than a lumbar movement.",),
    sources=(STRETCH_ACSM, STRETCH_PAGE, ACE), camera="front", default_reps=1,
    tags=("stretch", "unilateral", "no equipment"),
)

_CHEST = merge(pose(knee_flex=5),
               arms(flex=-18, abduct=88, rotate=70, elbow=90, forearm=30))
_CHEST_REST = merge(pose(knee_flex=5),
                    arms(flex=10, abduct=30, rotate=20, elbow=50, forearm=10))

chest_opener_stretch = ExerciseDefinition(
    id="chest_opener_stretch", name="Chest opener (doorway pectoral stretch)",
    category=Category.MOBILITY,
    description="With the upper arms out at shoulder height and the elbows bent, the forearms "
                "are braced and the chest pressed forward, lengthening pectoralis major.",
    setup=("Upper arms at shoulder height, elbows bent to a right angle",
           "Forearms on a doorframe; step through gently with one foot",
           "Chest forward, ribs down -- do not arch the low back"),
    anchor="feet",
    phases=(
        ph("Press forward", ECC, 2.5, _CHEST,
           cues=("Let the chest travel forward between the arms",
                 "Shoulder blades down and together")),
        ph("Hold", ISO, 20.0, _CHEST,
           cues=("Broad pull across the front of the chest; no pinching at the front "
                 "of the shoulder",)),
        ph("Release", CON, 1.5, _CHEST_REST, cues=("Step back and let the arms down",)),
        ph("Rest", TRN, 1.2, _CHEST_REST),
    ),
    muscles=(mu("pectoralis_major", P, 0.85, note="the target"),
             mu("pectoralis_minor", P, 0.7, note="the deeper target: it tips the scapula"),
             mu("deltoid_anterior", S, 0.55, note="lengthened"),
             mu("biceps_brachii", S, 0.45, note="the short head crosses the shoulder"),
             mu("latissimus_dorsi", S, 0.4, note="lengthened with the arm out"),
             mu("rhomboids", S, 0.55, note="works to hold the blades back"),
             mu("trapezius_middle", S, 0.55), mu("trapezius_lower", S, 0.5),
             mu("rectus_abdominis", ST, 0.45, note="stops the low back arching"),
             mu("infraspinatus_teres_minor", ST, 0.4)),
    equipment=(),
    errors=("Arching the low back so the movement comes from the spine, not the shoulder.",
            "Going so far that the front of the shoulder pinches -- that is the capsule, "
            "not the muscle.",
            "Only stretching at one height: the fibres run in several directions."),
    physio_notes=("Pectoralis minor shortness tips the scapula forward and is often the "
                  "structure worth lengthening in a rounded-shoulder posture; varying the "
                  "arm height biases the clavicular, sternal and costal fibres.",),
    sources=(STRETCH_ACSM, STRETCH_PAGE, NEUMANN, ACE), camera="front", default_reps=1,
    tags=("stretch", "no equipment"),
)

# ── Lying ──────────────────────────────────────────────────────────────

_KTC = merge(pose(hip_flex=120, knee_flex=130, spine_flex=12),
             arms(flex=95, abduct=20, elbow=110))
_KTC_REST = merge(pose(hip_flex=40, knee_flex=75),
                  arms(flex=25, abduct=15, elbow=20))

supine_knee_to_chest = ExerciseDefinition(
    id="supine_knee_to_chest", name="Supine double knee-to-chest", category=Category.MOBILITY,
    description="Lying on the back, both knees are drawn to the chest, flexing the lumbar "
                "spine and lengthening the extensors and the gluteals.",
    setup=("Lie on the back with the knees bent",
           "Draw both knees toward the chest with the hands behind the thighs",
           "Let the low back flatten and widen; keep the head and shoulders down"),
    orientation="supine", anchor="none", base_position=(-85.0, 15.0, 0.0),
    phases=(
        ph("Draw the knees in", CON, 2.0, _KTC,
           cues=("Hands behind the thighs, not on top of the shins",
                 "Let the tailbone lift a little; breathe out as it deepens")),
        ph("Hold", ISO, 20.0, _KTC, cues=("A broad stretch across the low back and buttocks",)),
        ph("Release", ECC, 1.8, _KTC_REST, cues=("Put the feet back down one at a time",)),
        ph("Rest", TRN, 1.5, _KTC_REST),
    ),
    muscles=(mu("erector_spinae", P, 0.8, note="the target: the lumbar extensors"),
             mu("gluteus_maximus", P, 0.75, note="lengthened at full hip flexion"),
             mu("multifidus", S, 0.6, note="lengthened"),
             mu("quadratus_lumborum", S, 0.55, note="lengthened"),
             mu("hamstrings", S, 0.5, note="lengthened at the hip, slack at the knee"),
             mu("hip_flexors", S, 0.5, note="works to hold the knees in"),
             mu("rectus_abdominis", S, 0.5), mu("biceps_brachii", ST, 0.4),
             mu("latissimus_dorsi", ST, 0.35)),
    equipment=_MAT,
    errors=("Pulling on the front of the shins, which compresses the knee.",
            "Lifting the head and neck to meet the knees.",
            "Using it during an acute flexion-intolerant episode, when it may aggravate."),
    physio_notes=("A lumbar flexion exercise of the Williams type: useful where extension "
                  "reproduces symptoms, and avoided where flexion does.",),
    sources=(STRETCH_PAGE, STRETCH_ACSM, ACE), camera="front", default_reps=1,
    tags=("stretch", "rehab", "no equipment"),
)

_FIG4 = merge(pose(hip_r_flex=50, hip_r_abduct=40, hip_r_rotate=42, knee_r_flex=90,
                   hip_l_flex=95, knee_l_flex=95),
              arms(flex=100, abduct=25, elbow=105))
_FIG4_REST = merge(pose(hip_flex=45, knee_flex=80), arms(flex=25, abduct=15, elbow=20))

supine_figure_four = ExerciseDefinition(
    id="supine_figure_four", name="Supine figure-four (gluteal / piriformis stretch)",
    category=Category.MOBILITY,
    description="Lying on the back, one ankle crosses the opposite thigh and that thigh is "
                "drawn in: hip flexion with external rotation lengthens the deep rotators "
                "and gluteus medius.",
    setup=("Lie on the back, both knees bent",
           "Cross one ankle over the opposite thigh above the knee",
           "Reach through and draw the supporting thigh toward the chest"),
    orientation="supine", anchor="none", base_position=(-85.0, 15.0, 0.0),
    unilateral=True,
    phases=(
        ph("Cross and draw in", CON, 2.5, _FIG4,
           cues=("Let the crossed knee fall out; keep that foot flexed to protect the knee",
                 "Draw the far thigh in until you feel it in the buttock")),
        ph("Hold", ISO, 20.0, _FIG4,
           cues=("Keep the head and the opposite shoulder down; breathe",)),
        ph("Release", ECC, 1.8, _FIG4_REST, cues=("Uncross and put both feet down",)),
        ph("Rest", TRN, 1.5, _FIG4_REST),
    ),
    muscles=(mu("hip_external_rotators", P, 0.85, side="R",
                note="the target, piriformis among them"),
             mu("gluteus_maximus", P, 0.8, side="R", note="lengthened"),
             mu("gluteus_medius", P, 0.75, side="R", note="the posterior fibres"),
             mu("tensor_fasciae_latae", S, 0.5, side="R", note="lengthened"),
             mu("adductors", S, 0.45, side="R"),
             mu("hip_flexors", S, 0.5, side="L", note="works to hold the thigh in"),
             mu("hamstrings", S, 0.45, side="L", note="lengthened at the hip"),
             mu("biceps_brachii", ST, 0.4), mu("rectus_abdominis", ST, 0.35)),
    equipment=_MAT,
    errors=("Letting the crossed foot go floppy, which puts the strain into the knee.",
            "Lifting the head and the opposite shoulder off the floor.",
            "Forcing the crossed knee down with a hand."),
    physio_notes=("Piriformis is an external rotator with the hip extended and an abductor "
                  "with it flexed past ~60 deg, which is why the stretch position combines "
                  "flexion, adduction and external rotation.",),
    sources=(STRETCH_PAGE, NEUMANN, ACE), camera="three_quarter", default_reps=1,
    tags=("stretch", "unilateral", "rehab", "no equipment"),
)

# Side-lying was tried first (the open-book) and measured badly: on its side
# the body's frontal plane is vertical, so the two arms separate by 100 units
# at 10 degrees of abduction and the underneath hand sits 25 below the mat.
# Supine keeps abduction in the plane of the floor, so the arms stay on it.

_TWIST = merge(pose(spine_rotation=-25, hip_r_flex=85, knee_r_flex=90, hip_r_abduct=-20,
                    hip_r_rotate=-25, hip_l_flex=5, knee_l_flex=8),
               arms(flex=0, abduct=82, elbow=10))
_TWIST_REST = merge(pose(hip_flex=35, knee_flex=70), arms(flex=0, abduct=50, elbow=10))

supine_spinal_twist = ExerciseDefinition(
    id="supine_spinal_twist", name="Supine spinal twist (supta matsyendrasana)",
    category=Category.MOBILITY,
    description="Lying on the back with the arms out to the sides, one knee is drawn across "
                "the body toward the floor: rotation for the trunk with both shoulders "
                "staying down.",
    setup=("Lie on the back, arms out to the sides at shoulder height",
           "Draw one knee up and let it fall across the body",
           "Keep both shoulders on the floor -- that is what limits the twist, not the knee"),
    orientation="supine", anchor="none", base_position=(-85.0, 15.0, 0.0),
    unilateral=True,
    phases=(
        ph("Cross over", ECC, 3.0, _TWIST,
           cues=("Let the knee fall; do not push it down with the hand",
                 "Turn the head the other way if the neck is comfortable")),
        ph("Hold", ISO, 20.0, _TWIST,
           cues=("Breathe out and let the shoulder settle back toward the floor",)),
        ph("Return", CON, 2.0, _TWIST_REST, cues=("Bring the knee back to the middle",)),
        ph("Rest", TRN, 1.5, _TWIST_REST),
    ),
    muscles=(mu("obliques", P, 0.8, note="the target: the external oblique of the turning side"),
             mu("erector_spinae", P, 0.7, note="the thoracolumbar rotators, lengthened"),
             mu("quadratus_lumborum", S, 0.6, note="lengthened"),
             mu("gluteus_medius", P, 0.7, side="R",
                note="lengthened as the hip adducts across the body"),
             mu("tensor_fasciae_latae", S, 0.55, side="R"),
             mu("multifidus", S, 0.5), mu("latissimus_dorsi", S, 0.5),
             mu("pectoralis_major", S, 0.5, note="lengthened with the arms out"),
             mu("rhomboids", ST, 0.45, note="works to keep the shoulder down"),
             mu("hip_flexors", ST, 0.4)),
    equipment=_MAT,
    errors=("Pressing the knee down with the hand until the opposite shoulder lifts.",
            "Holding the breath: the exhale is what lets the ribcage turn.",
            "Twisting hard through an irritable lumbar spine rather than letting it settle."),
    physio_notes=("The shoulder staying down is the criterion; once it lifts, the rotation "
                  "has moved from the trunk into the shoulder girdle and nothing further "
                  "is being stretched.",),
    sources=(STRETCH_PAGE, BEHM, NEUMANN, ACE), camera="three_quarter", default_reps=2,
    tags=("stretch", "mobility", "unilateral", "rehab", "no equipment"),
)

EXERCISES = (standing_forward_fold, standing_quad_stretch, calf_stretch, overhead_side_bend,
             chest_opener_stretch, supine_knee_to_chest, supine_figure_four,
             supine_spinal_twist)
