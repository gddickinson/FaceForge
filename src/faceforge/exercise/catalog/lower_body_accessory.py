"""Hip thrust, step-up, calf raise, wall sit, machine knee work and the physio staples."""

from __future__ import annotations

from faceforge.exercise.catalog._helpers import (
    ACE, BOREN, CON, CONTRERAS, DISTEFANO, ECC, EKSTROM, EXRX, HIPS, ISO, NEUMANN, NSCA, P, S,
    ST, SEATED_ON_BENCH, SHOULDERS, TRN, arms, combine, eq, grip, merge, mu, only, ph, pose,
)
from faceforge.exercise.model import Category, ExerciseDefinition

# Supine geometry with the feet fixed: the foot-flat rule with the trunk at
# -90 (+ pitch) gives ankle = -90 + pitch - hip + knee.
_BRIDGE_BOTTOM = merge(pose(hip_flex=46, knee_flex=136, ankle_flex=0), arms(abduct=30, elbow=10))
_BRIDGE_TOP = merge(pose(hip_flex=0, knee_flex=110, ankle_flex=0), arms(abduct=30, elbow=10))
_THRUST_BOTTOM = merge(pose(hip_flex=53, knee_flex=118, ankle_flex=0),
                       arms(abduct=45, elbow=90), grip())
_THRUST_TOP = merge(pose(hip_flex=0, knee_flex=90, ankle_flex=-3), arms(abduct=45, elbow=90),
                    grip())

glute_bridge = ExerciseDefinition(
    id="glute_bridge", name="Glute bridge", category=Category.LOWER_BODY,
    description="Supine with the knees bent, the hips extend until shoulders, hips and knees "
                "are in line; the movement is hip extension, not lumbar extension.",
    setup=("Lie on the back, feet flat hip-width, heels close to the buttocks",
           "Arms at the sides, ribs down"),
    orientation="supine", anchor="none", base_position=(-85.0, 15.0, 0.0),
    phases=(
        ph("Lift", CON, 1.2, _BRIDGE_TOP, pitch=-20, pivot=SHOULDERS,
           cues=("Drive through the heels; squeeze the glutes to lift the hips",
                 "Stop when the trunk and thighs are in line")),
        ph("Hold", ISO, 1.0, _BRIDGE_TOP, pitch=-20, pivot=SHOULDERS,
           cues=("Ribs down; do not arch the lower back",)),
        ph("Lower", ECC, 1.5, _BRIDGE_BOTTOM, pitch=0.0, pivot=SHOULDERS,
           cues=("Lower under control to the floor",)),
        ph("Rest", TRN, 0.4, _BRIDGE_BOTTOM),
    ),
    muscles=(mu("gluteus_maximus", P, 0.85), mu("hamstrings", S, 0.55),
             mu("erector_spinae", S, 0.35), mu("adductors", S, 0.35, note="adductor magnus"),
             mu("quadriceps", S, 0.3), mu("rectus_abdominis", ST, 0.3), mu("obliques", ST, 0.25),
             mu("gluteus_medius", S, 0.4)),
    equipment=(eq("mat", attach="static"),),
    errors=("Hyperextending the lumbar spine at the top.", "Pushing through the toes.",
            "Feet too far away so the hamstrings cramp."),
    physio_notes=("Gluteus maximus ~27-40 %MVIC in the bodyweight bridge; a standard early "
                  "rehabilitation exercise for hip extension.",),
    sources=(BOREN, DISTEFANO, EKSTROM, CONTRERAS), camera="side", tags=("no equipment", "rehab"),
)

barbell_hip_thrust = ExerciseDefinition(
    id="barbell_hip_thrust", name="Barbell hip thrust", category=Category.LOWER_BODY,
    description="Upper back on a bench, a bar across the hips: the hips extend to full "
                "lockout with the shins vertical at the top.",
    setup=("Shoulder blades on the bench edge, bar padded across the hip crease",
           "Feet flat so the shins are vertical at the top", "Chin tucked, ribs down"),
    orientation="supine", anchor="none", base_position=(-85.0, 60.0, 0.0),
    phases=(
        ph("Thrust", CON, 1.0, _THRUST_TOP, pitch=-3, pivot=SHOULDERS,
           cues=("Drive the hips up through the heels; posterior pelvic tilt at the top",)),
        ph("Lockout", ISO, 0.6, _THRUST_TOP, pitch=-3, pivot=SHOULDERS,
           cues=("Full hip extension, glutes squeezed",)),
        ph("Lower", ECC, 1.5, _THRUST_BOTTOM, pitch=25, pivot=SHOULDERS,
           cues=("Lower until the hips are just off the floor",)),
    ),
    muscles=(mu("gluteus_maximus", P, 1.0), mu("hamstrings", S, 0.6),
             mu("quadriceps", S, 0.5), mu("adductors", S, 0.4), mu("erector_spinae", S, 0.4),
             mu("rectus_abdominis", ST, 0.35), mu("gluteus_medius", S, 0.4)),
    equipment=(eq("barbell", plates=2), eq("bench", attach="static", position=(-95.0, 0.0, 0.0),
                                           height=45.0, length=40.0)),
    errors=("Arching the lumbar spine instead of extending the hips.",
            "Feet too close or too far from the bench."),
    physio_notes=("Higher gluteus maximus EMG than the back squat and deadlift at matched "
                  "loads (Contreras 2015).",),
    sources=(CONTRERAS, NSCA, ACE), camera="side", tags=("barbell",),
)

step_up = ExerciseDefinition(
    id="step_up", name="Box step-up", category=Category.LOWER_BODY,
    description="One foot on a knee-high box; that leg extends to lift the body onto the box "
                "and lowers it under control.",
    setup=("Box height so the leading thigh is about parallel", "Whole foot on the box",
           "Hands on hips or dumbbells at the sides"),
    phases=(
        ph("Place foot", TRN, 0.7,
           merge(pose(hip_r_flex=95, knee_r_flex=95, ankle_r_flex=10, knee_l_flex=5), arms(elbow=5)),
           pitch=10, cues=("Right foot flat on the box, knee over the toes",)),
        ph("Step up", CON, 1.2,
           merge(pose(hip_r_flex=5, knee_r_flex=5, hip_l_flex=35, knee_l_flex=60,
                      ankle_l_flex=-10), arms(elbow=5)),
           lift=60.0, cues=("Push through the right heel; do not push off the back foot",)),
        ph("Stand on box", ISO, 0.5,
           merge(pose(knee_flex=3), arms(elbow=5)), lift=60.0, cues=("Stand tall, hips level",)),
        ph("Step down", ECC, 1.4,
           merge(pose(hip_r_flex=95, knee_r_flex=95, ankle_r_flex=10, knee_l_flex=5), arms(elbow=5)),
           pitch=10, lift=0.0, cues=("Lower the left foot to the floor under control",)),
        ph("Return", TRN, 0.6, merge(pose(), arms(elbow=5))),
    ),
    muscles=(mu("quadriceps", P, 0.85, side="R"), mu("gluteus_maximus", P, 0.85, side="R"),
             mu("gluteus_medius", S, 0.6, side="R"), mu("hamstrings", S, 0.4, side="R"),
             mu("gastrocnemius", S, 0.4), mu("soleus", S, 0.4), mu("hip_flexors", S, 0.4, side="L"),
             mu("erector_spinae", ST, 0.35), mu("obliques", ST, 0.3)),
    equipment=(eq("plyo_box", attach="static", position=(0.0, 0.0, 0.0), height=60.0),),
    errors=("Pushing off the trailing foot.", "Leaning forward to use momentum.",
            "Knee collapsing inward on the step."),
    physio_notes=("High gluteus medius and maximus activation (Boren 2011); progress height "
                  "for knee rehabilitation.",),
    sources=(BOREN, DISTEFANO, NSCA), unilateral=True, camera="three_quarter",
    tags=("unilateral", "box"),
)

standing_calf_raise = ExerciseDefinition(
    id="standing_calf_raise", name="Standing calf raise", category=Category.LOWER_BODY,
    description="Plantarflexion from standing: the heels rise as high as possible, pause, "
                "and lower slowly.",
    setup=("Feet hip-width, weight over the balls of the feet", "Knees straight but not locked",
           "Fingertips on a wall for balance"),
    phases=(
        ph("Rise", CON, 1.0, merge(pose(ankle_flex=-40, knee_flex=3), arms(flex=15, elbow=10)),
           cues=("Push through the big toe; heels as high as possible",)),
        ph("Top", ISO, 0.6, merge(pose(ankle_flex=-40, knee_flex=3), arms(flex=15, elbow=10)),
           cues=("Squeeze the calves",)),
        ph("Lower", ECC, 2.0, merge(pose(ankle_flex=0, knee_flex=3), arms(flex=15, elbow=10)),
           cues=("Lower slowly, heels to the floor",)),
        ph("Bottom", ISO, 0.4, merge(pose(ankle_flex=0, knee_flex=3), arms(flex=15, elbow=10))),
    ),
    muscles=(mu("gastrocnemius", P, 0.95), mu("soleus", P, 0.9), mu("tibialis_posterior", S, 0.5),
             mu("peroneals", S, 0.5), mu("tibialis_anterior", ST, 0.15),
             mu("quadriceps", ST, 0.2), mu("gluteus_medius", ST, 0.25)),
    errors=("Bouncing at the bottom.", "Rolling onto the outside of the foot.",
            "Bending the knees (shifts work from gastrocnemius to soleus)."),
    physio_notes=("Straight-knee raises bias gastrocnemius; bent-knee (seated) raises bias "
                  "soleus. The heavy slow-eccentric version is the Alfredson protocol for "
                  "Achilles tendinopathy.",),
    sources=(NEUMANN, ACE, EXRX), camera="side", tags=("no equipment", "rehab"),
)

wall_sit = ExerciseDefinition(
    id="wall_sit", name="Wall sit", category=Category.LOWER_BODY,
    description="An isometric squat hold with the back against a wall, thighs parallel and "
                "knees at 90 deg.",
    setup=("Back flat on the wall, feet ~50 cm out", "Slide down until the knees are at 90 deg"),
    phases=(
        ph("Slide down", ECC, 1.5, merge(pose(hip_flex=90, knee_flex=90), arms(abduct=10, elbow=5)),
           cues=("Knees over the ankles, thighs parallel",)),
        ph("Hold", ISO, 6.0, merge(pose(hip_flex=90, knee_flex=90), arms(abduct=10, elbow=5)),
           cues=("Breathe; keep the whole back on the wall",)),
        ph("Rise", CON, 1.5, merge(pose(), arms(abduct=10, elbow=5)),
           cues=("Push through the heels to stand",)),
        ph("Rest", TRN, 1.0, merge(pose(), arms(abduct=10, elbow=5))),
    ),
    muscles=(mu("quadriceps", P, 0.7), mu("gluteus_maximus", S, 0.4), mu("adductors", S, 0.3),
             mu("hamstrings", S, 0.25), mu("soleus", S, 0.3), mu("rectus_abdominis", ST, 0.3),
             mu("erector_spinae", ST, 0.3)),
    errors=("Knees drifting past the toes.", "Holding the breath."),
    physio_notes=("Isometric quadriceps loading for patellofemoral pain and early ACL "
                  "rehabilitation; adjust the knee angle to tolerance.",),
    sources=(NEUMANN, ACE), camera="side", tags=("no equipment", "isometric", "rehab"),
)

leg_extension = ExerciseDefinition(
    id="leg_extension", name="Seated leg extension (machine)", category=Category.LOWER_BODY,
    description="Seated with the thighs supported, the knees extend against a pad on the shins.",
    setup=("Knee joint in line with the machine axis", "Pad just above the ankles",
           "Hold the handles; back against the seat"),
    orientation="seated", anchor="none", base_position=SEATED_ON_BENCH,
    phases=(
        ph("Extend", CON, 1.2, merge(pose(hip_flex=90, knee_flex=5), arms(abduct=10, elbow=20), grip()),
           cues=("Straighten the knees fully; toes up",)),
        ph("Top", ISO, 0.5, merge(pose(hip_flex=90, knee_flex=5), arms(abduct=10, elbow=20), grip()),
           cues=("Squeeze the quadriceps",)),
        ph("Lower", ECC, 2.0, merge(pose(hip_flex=90, knee_flex=90), arms(abduct=10, elbow=20), grip()),
           cues=("Lower to 90 deg under control",)),
    ),
    muscles=(mu("quadriceps", P, 0.95), mu("tibialis_anterior", ST, 0.2),
             mu("hip_flexors", ST, 0.3, note="rectus femoris also flexes the hip"),
             mu("forearm_flexors", ST, 0.3)),
    equipment=(eq("bench", attach="static", height=58.0, length=60.0),),
    errors=("Using momentum / kicking the pad.", "Hips lifting off the seat."),
    physio_notes=("Open-chain knee extension loads the patellofemoral joint most between "
                  "0 and 30 deg; anterior tibial shear peaks near full extension, so limit the "
                  "range after ACL reconstruction.",),
    sources=(NEUMANN, EXRX), camera="side", tags=("machine", "rehab"),
)

lying_leg_curl = ExerciseDefinition(
    id="lying_leg_curl", name="Lying leg curl (machine)", category=Category.LOWER_BODY,
    description="Prone on the bench, the knees flex against a pad at the heels.",
    setup=("Knees just off the edge of the bench", "Pad on the Achilles tendons", "Hips down"),
    orientation="prone", anchor="none", base_position=(-85.0, 73.0, 0.0),
    phases=(
        ph("Curl", CON, 1.0, merge(pose(knee_flex=120, ankle_flex=15), arms(flex=90, abduct=20, elbow=100), grip()),
           cues=("Pull the heels toward the buttocks",)),
        ph("Top", ISO, 0.4, merge(pose(knee_flex=120, ankle_flex=15), arms(flex=90, abduct=20, elbow=100), grip())),
        ph("Lower", ECC, 2.0, merge(pose(knee_flex=5, ankle_flex=15), arms(flex=90, abduct=20, elbow=100), grip()),
           cues=("Lower slowly; do not let the hips rise",)),
    ),
    muscles=(mu("hamstrings", P, 0.95), mu("gastrocnemius", S, 0.45),
             mu("gluteus_maximus", ST, 0.3), mu("erector_spinae", ST, 0.25)),
    equipment=(eq("bench", attach="static", length=190.0),),
    errors=("Hips lifting off the bench.", "Dropping the weight on the eccentric."),
    physio_notes=("Biases the biceps femoris; the kettlebell swing and Nordic curl bias "
                  "semitendinosus (Zebis 2013).",),
    sources=(NEUMANN, EXRX), camera="side", tags=("machine",),
)

#: Lying on the left side: the lower (left) arm lies forward on the floor, the
#: upper (right) hand rests on the hip.
_SIDE_LYING_ARMS = combine(arms(flex=100, abduct=0, elbow=90, side="l"),
                           arms(flex=0, elbow=70, side="r"))

clamshell = ExerciseDefinition(
    id="clamshell", name="Clamshell", category=Category.LOWER_BODY,
    description="Side-lying with hips and knees bent, the top knee opens like a clam while "
                "the feet stay together and the pelvis stays still.",
    setup=("Lie on the left side, hips flexed ~45 deg, knees ~90 deg", "Heels in line with "
           "the buttocks", "Top hand on the pelvis to feel for rolling"),
    orientation="side", anchor="none", base_position=(-85.0, 30.0, 0.0),
    phases=(
        ph("Open", CON, 1.0,
           merge(pose(hip_flex=45, knee_flex=90, hip_r_abduct=35, hip_r_rotate=30),
                 _SIDE_LYING_ARMS), cues=("Lift the top knee; keep the feet together",)),
        ph("Hold", ISO, 1.0,
           merge(pose(hip_flex=45, knee_flex=90, hip_r_abduct=35, hip_r_rotate=30),
                 _SIDE_LYING_ARMS), cues=("Pelvis does not roll back",)),
        ph("Close", ECC, 1.5,
           merge(pose(hip_flex=45, knee_flex=90), _SIDE_LYING_ARMS),
           cues=("Lower the knee slowly",)),
    ),
    muscles=(mu("gluteus_medius", P, 0.6, side="R"),
             mu("hip_external_rotators", P, 0.6, side="R"),
             mu("gluteus_maximus", S, 0.45, side="R", note="upper fibres"),
             mu("obliques", ST, 0.3), mu("quadratus_lumborum", ST, 0.3)),
    equipment=(eq("mat", attach="static"),),
    errors=("Rolling the pelvis backward to lift the knee higher.", "Feet separating."),
    physio_notes=("Gluteus medius ~40 %MVIC (Boren 2011); a first-line exercise for hip "
                  "abductor weakness and patellofemoral pain.",),
    sources=(BOREN, DISTEFANO), unilateral=True, camera="front", tags=("rehab", "no equipment"),
)

lateral_band_walk = ExerciseDefinition(
    id="lateral_band_walk", name="Lateral band walk", category=Category.LOWER_BODY,
    description="In a quarter squat with a band around the knees or ankles, the athlete "
                "steps sideways, keeping tension on the band.",
    setup=("Band above the knees", "Quarter squat: hips ~30 deg, knees ~30 deg",
           "Toes forward, trunk slightly inclined"),
    phases=(
        ph("Step right", CON, 0.8,
           merge(pose(hip_flex=30, knee_flex=30, hip_r_abduct=30, ankle_flex=0), arms(elbow=90)),
           pitch=10, cues=("Push the right foot out against the band",)),
        ph("Follow left", ISO, 0.8,
           merge(pose(hip_flex=30, knee_flex=30, hip_l_abduct=-5, ankle_flex=0), arms(elbow=90)),
           pitch=10, cues=("Bring the left foot in; keep tension, stay low",)),
        ph("Step right again", CON, 0.8,
           merge(pose(hip_flex=30, knee_flex=30, hip_r_abduct=30, ankle_flex=0), arms(elbow=90)),
           pitch=10),
        ph("Follow", ISO, 0.8,
           merge(pose(hip_flex=30, knee_flex=30, hip_l_abduct=-5, ankle_flex=0), arms(elbow=90)),
           pitch=10),
    ),
    muscles=(mu("gluteus_medius", P, 0.65), mu("tensor_fasciae_latae", S, 0.5),
             mu("gluteus_maximus", S, 0.45, note="upper fibres"), mu("quadriceps", S, 0.4),
             mu("hip_external_rotators", S, 0.4), mu("erector_spinae", ST, 0.3)),
    errors=("Standing up tall (loses band tension).", "Feet turning out.",
            "Dragging the trailing foot."),
    physio_notes=("Band above the knees increases gluteus medius relative to TFL compared "
                  "with a band at the ankles.",),
    sources=(DISTEFANO, BOREN, ACE), camera="front", tags=("band", "rehab"),
)

EXERCISES = (glute_bridge, barbell_hip_thrust, step_up, standing_calf_raise, wall_sit,
             leg_extension, lying_leg_curl, clamshell, lateral_band_walk)
