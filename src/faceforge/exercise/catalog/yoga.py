"""Yoga asanas: eight held postures, each authored as enter / hold / release.

A pose is not a lift.  The interesting muscles are the ones holding the shape
against gravity, so most of the work here is isometric and the roles read
differently: the prime mover is whatever keeps the posture, and the muscle
being *lengthened* is named in the notes rather than given a role.

Salem 2013 measured the joint demands of the standing asanas (warrior II and
chair reach knee-extensor demands comparable with a moderate squat); Ni 2014
measured surface EMG across poses and skill levels.
"""

from __future__ import annotations

from faceforge.exercise.catalog._helpers import (
    ACE, CON, ECC, HIPS, ISO, NEUMANN, NSCA, P, S, ST, STRETCH_ACSM, STRETCH_PAGE, TRN,
    YOGA_EMG, YOGA_KIN, arms, eq, flat_foot_ankle, flat_palm, merge, mu, only, ph, pose,
    toes_on_floor, toes_tucked
)
from faceforge.exercise.model import Category, ExerciseDefinition

#: Toes bent back onto their pads under a tucked foot (`toes_tucked`),
#: so the foot is not one rigid wedge balanced on its longest toe.
_TUCKED = toes_tucked(45.0)

_MAT = (eq("mat", attach="floor"),)
_STAND = merge(pose(knee_flex=5), arms(flex=5, abduct=8, elbow=5))

# ── Standing poses ─────────────────────────────────────────────────────

_CHAIR_PITCH = 25.0
_CHAIR = merge(pose(hip_flex=75, knee_flex=85,
                    ankle_flex=flat_foot_ankle(_CHAIR_PITCH, 75, 85)),
               arms(flex=170, abduct=10, elbow=5))

chair_pose = ExerciseDefinition(
    id="chair_pose", name="Chair pose (utkatasana)", category=Category.MOBILITY,
    description="A held half-squat with the arms overhead: the hips sit back, the shins stay "
                "near vertical and the arms reach up alongside the ears.",
    setup=("Feet together or hip width, weight into the heels",
           "Sit the hips back as if into a chair; keep the knees behind the toes",
           "Reach the arms up; draw the lower ribs down so the low back does not arch"),
    phases=(
        ph("Sit back", ECC, 2.0, _CHAIR, pitch=_CHAIR_PITCH,
           cues=("Hips back and down, chest lifted",)),
        ph("Hold", ISO, 8.0, _CHAIR, pitch=_CHAIR_PITCH,
           cues=("Breathe steadily; knees together, ribs down",
                 "Weight in the heels -- you should be able to lift the toes")),
        ph("Stand", CON, 1.5, _STAND, cues=("Press the floor away and stand tall",)),
        ph("Rest", TRN, 1.0, _STAND),
    ),
    muscles=(mu("quadriceps", P, 0.8), mu("gluteus_maximus", P, 0.7),
             mu("erector_spinae", P, 0.65, note="holds the lifted chest"),
             mu("soleus", S, 0.6), mu("tibialis_anterior", S, 0.5),
             mu("deltoid_anterior", S, 0.5, note="holds the arms overhead"),
             mu("trapezius_lower", S, 0.5), mu("serratus_anterior", S, 0.5),
             mu("rectus_abdominis", ST, 0.45), mu("adductors", ST, 0.4),
             mu("hamstrings", ST, 0.4)),
    equipment=_MAT,
    errors=("Arching the low back to get the arms up (usually tight latissimus).",
            "Knees drifting forward over the toes so the heels lift.",
            "Holding the breath."),
    physio_notes=("Salem 2013 measured knee-extensor demand in chair pose comparable with a "
                  "moderate-depth squat; it is the standing asana most often cited as "
                  "strength work rather than flexibility work.",),
    sources=(YOGA_KIN, YOGA_EMG, ACE), camera="side", default_reps=1,
    tags=("yoga", "isometric", "no equipment"),
)

_W2 = merge(pose(hip_r_flex=30, hip_r_abduct=42, hip_r_rotate=40, knee_r_flex=90,
                 ankle_r_flex=20, hip_l_abduct=42, hip_l_rotate=-15, knee_l_flex=5,
                 ankle_l_flex=8),
            arms(flex=0, abduct=90, elbow=0))
_W2_UP = merge(pose(hip_r_abduct=40, hip_r_rotate=35, knee_r_flex=8, ankle_r_flex=5,
                    hip_l_abduct=40, knee_l_flex=5),
               arms(flex=0, abduct=90, elbow=0))

warrior_two = ExerciseDefinition(
    id="warrior_two", name="Warrior II (virabhadrasana II)", category=Category.MOBILITY,
    description="A long stance with the front knee bent to about a right angle over its ankle, "
                "the back leg straight, and the arms held level front and back.",
    setup=("Step the feet wide; turn the front foot out 90 deg, the back foot in ~15",
           "Bend the front knee until the shin is vertical over the ankle",
           "Arms level with the shoulders, trunk stacked over the pelvis -- not leaning"),
    anchor="feet", unilateral=True,
    phases=(
        ph("Bend the front knee", ECC, 2.0, _W2,
           cues=("Front knee tracks over the second toe",
                 "Sink the hips down, not forward")),
        ph("Hold", ISO, 9.0, _W2,
           cues=("Front thigh toward parallel; back leg straight and strong",
                 "Shoulders down; reach through both hands")),
        ph("Straighten", CON, 1.6, _W2_UP, cues=("Press into the front heel to straighten",)),
        ph("Rest", TRN, 1.0, _W2_UP),
    ),
    muscles=(mu("quadriceps", P, 0.85, side="R", note="holds the bent front knee"),
             mu("gluteus_medius", P, 0.7, note="holds the turned-out front hip"),
             mu("hip_external_rotators", P, 0.65, side="R"),
             mu("adductors", S, 0.6, side="L", note="the long back leg"),
             mu("gluteus_maximus", S, 0.6), mu("hamstrings", S, 0.5),
             mu("deltoid_lateral", P, 0.6, note="holds the arms level"),
             mu("trapezius_middle", S, 0.5), mu("erector_spinae", S, 0.5),
             mu("obliques", ST, 0.45), mu("soleus", ST, 0.45),
             mu("tibialis_posterior", ST, 0.4)),
    equipment=_MAT,
    errors=("The front knee collapsing inward.",
            "Leaning the trunk over the front leg instead of stacking it over the pelvis.",
            "Shrugging the shoulders to keep the arms up."),
    physio_notes=("The deltoid fatigues long before the legs do; dropping the arms is the "
                  "usual regression for a longer hold.",),
    sources=(YOGA_KIN, YOGA_EMG, ACE, NSCA), camera="front", default_reps=1,
    tags=("yoga", "isometric", "unilateral", "no equipment"),
)

#: The trunk's tilt is a wrapper ROLL about the hips, not `spine_lat_bend`:
#: the spine DOFs turn vertebrae, and the shoulders hang off the pelvis root,
#: so 30 degrees of lateral flexion moved nothing the render could see
#: (shoulder_R stayed at x=25.2; a 40-degree roll puts it at 110).
#: 25, and not more, measured 2026-09-16.  A deeper lean is what the pose
#: wants -- the lower hand belongs at the shin -- but rolling the body about
#: the hips lifts the BACK foot and the hip that has to put it down runs out
#: of adduction at about -32 degrees:
#:     roll   25    35    45    55
#:     back toe y   5.7  14.2  26.9  40.2   (front toe stays ~6, floor is 3.1)
#:     lower hand y 156   139   122   103
#: So every extra degree of lean buys ~1.7 units of hand travel and costs
#: ~0.9 units of back foot in the air.  Going to 35 would put the hand at 139
#: -- still nowhere near the shin -- and a foot 11 units off the mat, which is
#: a worse picture than a shallow triangle.  Leave it until the trunk can bend
#: at the spine rather than at the wrapper.
_TRI_ROLL = 25.0
#: Rolling a body whose legs are rigid with its pelvis lifts the far foot:
#: at a 42-degree stance the back toes went to 72 with a 25-degree roll.  The
#: back hip adducts to put them back (toe 6.4 against the front foot's 7.9),
#: which costs stance width -- the price of the lean.
_TRI = merge(pose(spine_lat_bend=12, hip_r_abduct=44, hip_r_rotate=40, hip_r_flex=25,
                  knee_r_flex=5, ankle_r_flex=10, hip_l_abduct=-8, hip_l_rotate=-12,
                  knee_l_flex=5, ankle_l_flex=8),
             only(shoulder_r_flex=0, shoulder_r_abduct=95, elbow_r_flex=3,
                  shoulder_l_flex=0, shoulder_l_abduct=95, elbow_l_flex=3))
_TRI_UP = merge(pose(hip_r_abduct=40, hip_r_rotate=35, knee_r_flex=5, hip_l_abduct=40,
                     knee_l_flex=5),
                arms(flex=0, abduct=92, elbow=3))

triangle_pose = ExerciseDefinition(
    id="triangle_pose", name="Triangle pose (trikonasana)", category=Category.MOBILITY,
    description="A wide stance with both legs straight; the trunk tilts sideways over the "
                "front leg so one hand reaches down the shin and the other points up.",
    setup=("Wide stance, front foot turned out, both legs straight",
           "Reach the front hand forward first, then tip sideways from the hip",
           "Stack the shoulders: the top arm points at the ceiling, not forward"),
    anchor="feet", unilateral=True,
    phases=(
        ph("Tip over", ECC, 2.5, _TRI, roll=_TRI_ROLL, pivot=HIPS,
           cues=("Hinge sideways from the hip, not by folding the waist",
                 "Keep both legs straight; lengthen both sides of the trunk")),
        ph("Hold", ISO, 9.0, _TRI, roll=_TRI_ROLL, pivot=HIPS,
           cues=("Chest open to the side; breathe into the top ribs",)),
        ph("Come up", CON, 2.0, _TRI_UP,
           cues=("Press into the back foot and lift with the side of the trunk",)),
        ph("Rest", TRN, 1.0, _TRI_UP),
    ),
    muscles=(mu("obliques", P, 0.7, note="the upper side holds the trunk out"),
             mu("quadratus_lumborum", P, 0.7), mu("erector_spinae", S, 0.6),
             mu("gluteus_medius", P, 0.65, side="R"),
             mu("quadriceps", S, 0.55, note="holds both knees straight"),
             mu("adductors", S, 0.5, note="lengthened on the front leg"),
             mu("hamstrings", S, 0.5, note="lengthened on the front leg"),
             mu("deltoid_lateral", S, 0.5), mu("trapezius_middle", S, 0.45),
             mu("tibialis_posterior", ST, 0.4)),
    equipment=_MAT,
    errors=("Bending the front knee to reach further down.",
            "Rounding forward so the chest faces the floor.",
            "Hyperextending (locking back) the front knee."),
    physio_notes=("The stretch is felt in the front-leg hamstring and adductors; how far the "
                  "hand goes down is a hip-hinge range question, not a target.",),
    sources=(YOGA_KIN, STRETCH_ACSM, ACE), camera="front", default_reps=1,
    tags=("yoga", "isometric", "unilateral", "no equipment"),
)

_TREE = merge(pose(hip_r_abduct=42, hip_r_rotate=42, hip_r_flex=20, knee_r_flex=120,
                   knee_l_flex=5),
              arms(flex=172, abduct=6, elbow=5))
_TREE_DOWN = merge(pose(hip_r_flex=10, knee_r_flex=10, knee_l_flex=5),
                   arms(flex=20, abduct=10, elbow=5))

tree_pose = ExerciseDefinition(
    id="tree_pose", name="Tree pose (vrksasana)", category=Category.MOBILITY,
    description="A one-legged balance: the lifted foot presses into the inner thigh of the "
                "standing leg, the knee turned out, the arms overhead.",
    setup=("Find the balance on one foot before placing the other",
           "Foot above or below the knee, never against it",
           "Press foot and thigh into each other; hips stay level"),
    anchor="feet", anchor_side="L", unilateral=True,
    phases=(
        ph("Lift the foot", CON, 2.0, _TREE,
           cues=("Turn the lifted knee out; keep the pelvis level",)),
        ph("Hold", ISO, 10.0, _TREE,
           cues=("Fix the eyes on one point; breathe",
                 "Standing hip over the standing ankle -- do not let it swing out")),
        ph("Lower", ECC, 1.5, _TREE_DOWN, cues=("Place the foot down without wobbling",)),
        ph("Rest", TRN, 1.0, _TREE_DOWN),
    ),
    muscles=(mu("gluteus_medius", P, 0.75, side="L", note="keeps the pelvis level"),
             mu("hip_external_rotators", P, 0.7, side="R"),
             mu("quadriceps", S, 0.5, side="L"),
             mu("foot_intrinsics", P, 0.65, side="L", note="the balance happens here"),
             mu("peroneals", S, 0.55, side="L"), mu("tibialis_posterior", S, 0.55, side="L"),
             mu("soleus", S, 0.5, side="L"), mu("adductors", S, 0.5, side="R"),
             mu("erector_spinae", S, 0.5), mu("obliques", ST, 0.45),
             mu("deltoid_anterior", ST, 0.45)),
    equipment=_MAT,
    errors=("Resting the foot against the side of the standing knee.",
            "The standing hip swinging out sideways.",
            "Holding the breath while balancing."),
    physio_notes=("Single-leg balance loads the ankle evertors and invertors and the foot's "
                  "intrinsics continuously; it is a standard proprioception drill after an "
                  "ankle sprain.",),
    sources=(YOGA_EMG, NEUMANN, ACE), camera="front", default_reps=1,
    tags=("yoga", "balance", "isometric", "no equipment"),
)

_HL_PITCH = 5.0
#: The back foot: the heel is lifted by definition ("crescent"), and the ground
#: lock anchors the FRONT foot only, so nothing was holding the back one out of
#: the floor -- its longest toe sat 9.4 below it.  A little dorsiflexion brings
#: the foot up toward the shin and the toes do the rest.
_HL_BACK_ANKLE = 15.0
_HIGH_LUNGE = merge(pose(hip_r_flex=50, knee_r_flex=90,
                         ankle_r_flex=flat_foot_ankle(_HL_PITCH, 50, 90),
                         hip_l_flex=-25, knee_l_flex=35, ankle_l_flex=_HL_BACK_ANKLE,
                         toe_curl_l=toes_on_floor(_HL_PITCH, -25, 35, _HL_BACK_ANKLE)),
                    arms(flex=172, abduct=8, elbow=5))
_HL_UP = merge(pose(hip_r_flex=20, knee_r_flex=15, hip_l_flex=-10, knee_l_flex=8,
                    ankle_l_flex=15),
               arms(flex=45, abduct=10, elbow=5))

high_lunge = ExerciseDefinition(
    id="high_lunge", name="High lunge (crescent)", category=Category.MOBILITY,
    description="A long split stance with the back heel lifted and the arms overhead: the "
                "front leg holds the position while the back hip flexor is lengthened.",
    setup=("Step one foot well back, heel lifted, back leg straight",
           "Front shin vertical; front knee over the ankle",
           "Tuck the tailbone under before reaching the arms up"),
    anchor="feet", anchor_side="R", unilateral=True,
    phases=(
        ph("Sink in", ECC, 2.0, _HIGH_LUNGE, pitch=_HL_PITCH,
           cues=("Sink the back hip toward the floor; keep the front shin vertical",)),
        ph("Hold", ISO, 8.0, _HIGH_LUNGE, pitch=_HL_PITCH,
           cues=("Tail tucked -- the stretch belongs in the front of the back hip",
                 "Ribs down; reach up through the fingers")),
        ph("Come up", CON, 1.6, _HL_UP, cues=("Press the front heel down and shorten the stance",)),
        ph("Rest", TRN, 1.0, _HL_UP),
    ),
    muscles=(mu("quadriceps", P, 0.8, side="R"), mu("gluteus_maximus", P, 0.7, side="R"),
             mu("gluteus_maximus", S, 0.6, side="L", note="tucks the pelvis under"),
             mu("hip_flexors", S, 0.5, side="L", note="lengthened, not working"),
             mu("gastrocnemius", S, 0.55, side="L", note="holds the lifted back heel"),
             mu("gluteus_medius", S, 0.55), mu("erector_spinae", S, 0.5),
             mu("rectus_abdominis", P, 0.6, note="stops the low back arching"),
             mu("deltoid_anterior", S, 0.5), mu("trapezius_lower", ST, 0.45),
             mu("hamstrings", ST, 0.4)),
    equipment=_MAT,
    errors=("Letting the front knee travel past the toes.",
            "Arching the low back instead of tucking the pelvis, which moves the stretch "
            "out of the hip flexor.",
            "Dropping the back knee toward the floor (that is the low lunge)."),
    physio_notes=("Posterior pelvic tilt is what puts the stretch on iliopsoas; without it "
                  "the range comes from lumbar extension instead.",),
    sources=(YOGA_KIN, STRETCH_ACSM, NSCA), camera="side", default_reps=1,
    tags=("yoga", "isometric", "unilateral", "no equipment"),
)

# ── Prone and all-fours poses ──────────────────────────────────────────
# Prone: the body lies face down, +pitch drops the head end (see upper_push).

_DD_PITCH = 38.0
# `toes_tucked` is calibrated for a horizontal shank (a plank).  Down dog's is
# steep, and 60 degrees put the toe TIP 3.7 units above the ball -- the foot
# rolled onto its nails.  22 leaves the pads down and the heel up, which is
# what the pose is.
_DOG_TOES = 22.0
_DOWN_DOG = merge(pose(hip_flex=100, knee_flex=5, ankle_flex=25, toe_curl=_DOG_TOES),
                  arms(flex=90 + _DD_PITCH, abduct=10, elbow=0), flat_palm())
_DD_PLANK = merge(pose(ankle_flex=45, toe_curl=_TUCKED), arms(flex=70, abduct=10, elbow=0), flat_palm())

downward_dog = ExerciseDefinition(
    id="downward_dog", name="Downward-facing dog (adho mukha svanasana)",
    category=Category.MOBILITY,
    description="Hands and feet on the floor with the hips lifted high, so the body makes an "
                "inverted V. A whole-body pose: shoulders overhead, hamstrings and calves long.",
    setup=("Hands shoulder width, feet hip width",
           "Lift the hips up and back; let the heels sink toward the floor",
           "Bend the knees if the low back rounds -- a long spine beats straight legs"),
    orientation="prone", anchor="hands", base_position=(-85.0, 30.0, 0.0),
    phases=(
        ph("Lift the hips", CON, 2.0, _DOWN_DOG, pitch=_DD_PITCH,
           cues=("Push the floor away and send the hips up and back",)),
        ph("Hold", ISO, 9.0, _DOWN_DOG, pitch=_DD_PITCH,
           cues=("Ears between the arms; shoulder blades wide",
                 "Press the heels down without locking the knees back")),
        ph("Plank", TRN, 1.5, _DD_PLANK, pitch=-18,
           cues=("Shift forward to a plank; shoulders over the wrists",)),
        ph("Hold the plank", ISO, 1.5, _DD_PLANK, pitch=-18),
    ),
    muscles=(mu("deltoid_anterior", P, 0.7, note="holds the overhead shoulder position"),
             mu("serratus_anterior", P, 0.7, note="upward rotation of the scapula"),
             mu("trapezius_lower", S, 0.6), mu("triceps_brachii", S, 0.6),
             mu("rectus_abdominis", S, 0.5), mu("hip_flexors", S, 0.5,
                                                note="holds the hips folded"),
             mu("quadriceps", S, 0.5, note="holds the knees straight"),
             mu("hamstrings", S, 0.45, note="lengthened"),
             mu("gastrocnemius", S, 0.45, note="lengthened"),
             mu("soleus", ST, 0.4), mu("forearm_extensors", ST, 0.45)),
    equipment=_MAT,
    errors=("Rounding the low back to force the heels down.",
            "Hands too close to the feet, which shortens the pose into a fold.",
            "Shoulders creeping up around the ears."),
    physio_notes=("The limiting tissue is usually the hamstrings and thoracic extension, not "
                  "the calves; bending the knees moves the pose back into the shoulders "
                  "where the strength work is.",),
    sources=(YOGA_EMG, YOGA_KIN, STRETCH_ACSM, ACE), camera="front", default_reps=1,
    tags=("yoga", "isometric", "no equipment"),
)

#: Cobra proper -- pelvis on the mat, chest lifted by spinal extension alone
#: -- is not a pose this rig can show.  The spine DOFs turn vertebrae and the
#: shoulders hang off the pelvis root, so 40 degrees of extension moved the
#: shoulder 0.0 units; a wrapper pitch about the hips lifts the chest (29 ->
#: 66) but takes the legs to -88 with it, and hip extension stops at -27.
#: Pitching about the KNEES instead is the pose the rig does have: the shins
#: stay down, the thighs and pelvis lift, the chest comes up.  That is an
#: upward-facing dog, so that is what this is called.
_DOG_PITCH = -18.0
KNEES = (0.0, 0.0, -141.0)


def _updog(extension: float, arm_flex: float, elbow: float) -> dict[str, float]:
    # The tops of the feet are the contact.  A foot this plantarflexed is
    # upside down, so it is toe FLEXION that lifts the tips here (measured:
    # at ankle -45, -37 raises the tip 3.4 and +75 drops it); with the toes
    # straight they hung 4.7 below the ball and 5.9 through the mat.
    return merge(pose(spine_flex=-extension, hip_flex=0, knee_flex=5, ankle_flex=-45,
                      toe_curl=-37),
                 arms(flex=arm_flex, abduct=12, elbow=elbow), flat_palm())


_UPDOG = _updog(20.0, 20.0, 30.0)
_UPDOG_DOWN = _updog(5.0, 45.0, 110.0)

upward_dog = ExerciseDefinition(
    id="upward_dog", name="Upward-facing dog (urdhva mukha svanasana)",
    category=Category.MOBILITY,
    description="From lying face down, the arms straighten and the chest, hips and thighs lift "
                "clear of the mat, leaving only the hands and the tops of the feet on it. "
                "Back extension with the shoulders pulled down and open.",
    setup=("Hands under the shoulders, tops of the feet on the mat",
           "Press the floor away until the arms are straight and the thighs lift",
           "Shoulders down away from the ears; lead with the chest, not the chin"),
    orientation="prone", anchor="none", base_position=(-85.0, 22.0, 0.0),
    phases=(
        ph("Lift the chest", CON, 2.0, _UPDOG, pitch=_DOG_PITCH, pivot=KNEES,
           cues=("Press the hands down and draw the chest forward and up",
                 "Thighs leave the mat; only the hands and the feet stay")),
        ph("Hold", ISO, 7.0, _UPDOG, pitch=_DOG_PITCH, pivot=KNEES,
           cues=("Breathe into the front of the ribs; shoulder blades down the back",)),
        ph("Lower", ECC, 1.8, _UPDOG_DOWN, pitch=-5.0, pivot=KNEES,
           cues=("Lower the chest and the hips slowly back to the mat",)),
        ph("Rest", TRN, 1.2, _UPDOG_DOWN, pitch=-5.0, pivot=KNEES),
    ),
    muscles=(mu("erector_spinae", P, 0.75), mu("multifidus", P, 0.65),
             mu("quadratus_lumborum", S, 0.5),
             mu("gluteus_maximus", S, 0.5, note="holds the legs down"),
             mu("trapezius_lower", S, 0.55, note="keeps the shoulders down"),
             mu("triceps_brachii", S, 0.5), mu("rhomboids", S, 0.45),
             mu("rectus_abdominis", S, 0.45, note="lengthened"),
             mu("hip_flexors", S, 0.4, note="lengthened"),
             mu("deltoid_posterior", ST, 0.4)),
    equipment=_MAT,
    errors=("Shrugging up to the ears instead of drawing the shoulder blades down.",
            "Throwing the head back and pinching the low back.",
            "Leaving the thighs on the mat -- that is a cobra, a different pose."),
    physio_notes=("A prone extension exercise of the McKenzie family; commonly prescribed for "
                  "flexion-intolerant low back pain and avoided where extension reproduces "
                  "symptoms. Cobra is the lower-effort version, with the pelvis staying "
                  "down.",),
    sources=(YOGA_EMG, STRETCH_PAGE, ACE), camera="front", default_reps=1,
    tags=("yoga", "isometric", "rehab", "no equipment"),
)

# On all fours the dorsum of each foot lies on the floor: fully plantarflexed.
_QUAD = merge(pose(hip_flex=90, knee_flex=90, ankle_flex=-45),
              arms(flex=90, abduct=5, elbow=0), flat_palm())
_CAT = merge(pose(spine_flex=42, hip_flex=90, knee_flex=90, ankle_flex=-45),
             arms(flex=92, abduct=8, elbow=0), flat_palm())
_COW = merge(pose(spine_flex=-38, hip_flex=92, knee_flex=90, ankle_flex=-45),
             arms(flex=88, abduct=5, elbow=0), flat_palm())

cat_cow = ExerciseDefinition(
    id="cat_cow", name="Cat-cow (marjaryasana-bitilasana)", category=Category.MOBILITY,
    description="On hands and knees, the spine cycles between full flexion (cat) and extension "
                "(cow) in time with the breath. The standard spinal mobility drill.",
    setup=("Hands under the shoulders, knees under the hips",
           "Exhale to round the whole spine; inhale to let it sag and lift the chest",
           "Move the whole spine, not just the low back"),
    orientation="prone", anchor="hands", base_position=(-85.0, 70.0, 0.0),
    phases=(
        ph("Cat (exhale, round)", CON, 2.0, _CAT,
           cues=("Push the floor away and round from the tailbone up",)),
        ph("Neutral", TRN, 0.6, _QUAD),
        ph("Cow (inhale, extend)", ECC, 2.0, _COW,
           cues=("Let the belly drop, lift the chest and the sitting bones",)),
        ph("Neutral", TRN, 0.6, _QUAD),
    ),
    muscles=(mu("erector_spinae", P, 0.55, note="drives the extension half"),
             mu("rectus_abdominis", P, 0.55, note="drives the flexion half"),
             mu("multifidus", S, 0.45), mu("obliques", S, 0.4),
             mu("transversus_abdominis", S, 0.4),
             mu("serratus_anterior", S, 0.45, note="pushes the floor away in the cat"),
             mu("deltoid_anterior", ST, 0.4), mu("triceps_brachii", ST, 0.35),
             mu("hip_flexors", ST, 0.3), mu("quadriceps", ST, 0.3)),
    equipment=_MAT,
    errors=("Moving only the lumbar spine and leaving the thoracic spine stiff.",
            "Rushing: the point is to pair the movement with the breath.",
            "Letting the elbows lock hard and the shoulders shrug."),
    physio_notes=("Low-load, full-range spinal motion of this kind is McGill's standard "
                  "warm-up before the stability exercises, not a strengthening exercise.",),
    sources=(YOGA_EMG, STRETCH_PAGE, ACE, NSCA), camera="front", default_reps=4,
    tags=("yoga", "mobility", "rehab", "no equipment"),
)

EXERCISES = (chair_pose, warrior_two, triangle_pose, tree_pose, high_lunge,
             downward_dog, upward_dog, cat_cow)
