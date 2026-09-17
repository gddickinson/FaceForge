"""Split-stance work: lunges and the split squat.

One leg in front of the other turns a squat into a balance problem: the
frontal-plane muscles -- gluteus medius and the deep hip rotators -- have to
hold a pelvis that only one foot is under.  The bilateral squats are in
:mod:`faceforge.exercise.catalog.lower_body`.
"""

from __future__ import annotations

from faceforge.exercise.catalog._helpers import (
    ACE, BOREN, CON, DISTEFANO, ECC, ISO, NSCA, P, S, ST, TRN, arms, combine, eq, grip,
    lunge, merge, mu, ph, pose,
)
from faceforge.exercise.model import Category, ExerciseDefinition


#: How far the body travels along its own line of progress during the step.
#: A forward lunge plants the FRONT foot ahead and the body goes with it; a
#: reverse lunge plants the BACK foot behind and the body stays where it was,
#: so the anchor moves the other way relative to the stance.  Without this the
#: `reverse` flag chose nothing but a cue string and the two exercises were
#: the same animation to 0.1 units at every pivot -- two catalogue entries,
#: one demonstration.
_STEP_TRAVEL = 34.0


def _lunge_definition(id_, name, description, reverse: bool):
    front = "r"
    split_stand = lunge(front, 30, 20, -12, 25, pitch=8)[0]
    bottom = lunge(front, 95, 95, -5, 90, pitch=10)[0]
    up_arms = combine(arms(elbow=10), grip(curl=40))
    step_cue = ("Step back with the left leg" if reverse else "Step forward with the right leg")
    travel = (-_STEP_TRAVEL, 0.0) if reverse else (_STEP_TRAVEL, 0.0)
    return ExerciseDefinition(
        id=id_, name=name, category=Category.LOWER_BODY, description=description,
        setup=("Stand tall, feet hip-width", "Hands on hips or holding dumbbells at the sides"),
        phases=(
            ph("Step", TRN, 0.7, merge(split_stand, up_arms), pitch=8, travel=travel,
               cues=(step_cue, "Keep the hips square")),
            ph("Lower", ECC, 1.2, merge(bottom, up_arms), pitch=10, travel=travel,
               cues=("Drop the back knee toward the floor",
                     "Front shin near vertical, trunk upright")),
            ph("Drive", CON, 1.0, merge(split_stand, up_arms), pitch=8, travel=travel,
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
    # The back foot is on a 50-high bench and the body rises 50 units between
    # the bottom and the top, so the back knee has to straighten as it goes or
    # the foot leaves the bench: 104 and 50 hold the ankle at 51.3 and 53.6.
    #
    # The bench is at z = -190 and 45 high, not -118 and 50.  At -118 its pad
    # (150 long, so spanning z -193..-43) reached under the lifter: measured
    # 2026-09-17 at the bottom, the hip sat 17.1 ABOVE the pad and the back
    # knee 27.8 BELOW it, both inside its footprint -- the rear thigh ran
    # straight through the bench.
    #
    # Moving it back alone was not enough.  The shin climbs from a knee near
    # the floor (y 16.2) to an ankle on the pad (52.7), so wherever the pad's
    # near edge falls under that climb the shin crosses its top face -- it read
    # 8.5 inside at every height from 42 to 50.  The edge has to be at the
    # ankle, not under the shin: at -190 the pad ends at z = -115 and the ankle
    # is 2.4 in front of it, so the foot lies back onto the pad from its near
    # end (toe at -143.6) and the shin is clear in front.  45 then leaves the
    # ankle 7.7 above the pad, which is a foot resting laces-down on it rather
    # than an ankle pivot 2.7 clear and the shin through the surface.
    phases=(
        ph("Lower", ECC, 2.0,
           merge(pose(hip_r_flex=100, knee_r_flex=100, ankle_r_flex=10, hip_l_flex=-15,
                      knee_l_flex=104, ankle_l_flex=-35), arms(elbow=5), grip()), pitch=15,
           cues=("Front knee over mid-foot; back knee drops toward the floor",)),
        ph("Bottom", ISO, 0.3,
           merge(pose(hip_r_flex=100, knee_r_flex=100, ankle_r_flex=10, hip_l_flex=-15,
                      knee_l_flex=104, ankle_l_flex=-35), arms(elbow=5), grip()), pitch=15),
        ph("Drive", CON, 1.4,
           merge(pose(hip_r_flex=25, knee_r_flex=20, ankle_r_flex=5, hip_l_flex=-20,
                      knee_l_flex=45, ankle_l_flex=-15), arms(elbow=5), grip()), pitch=8,
           cues=("Push through the front heel; hips forward",)),
        ph("Top", ISO, 0.3,
           merge(pose(hip_r_flex=25, knee_r_flex=20, ankle_r_flex=5, hip_l_flex=-20,
                      knee_l_flex=45, ankle_l_flex=-15), arms(elbow=5), grip()), pitch=8),
    ),
    muscles=(mu("quadriceps", P, 0.9, side="R"), mu("gluteus_maximus", P, 0.9, side="R"),
             mu("gluteus_medius", S, 0.6, side="R"), mu("hamstrings", S, 0.45, side="R"),
             mu("adductors", S, 0.45, side="R"), mu("hip_flexors", S, 0.35, side="L"),
             mu("erector_spinae", ST, 0.4), mu("rectus_abdominis", ST, 0.3),
             mu("forearm_flexors", ST, 0.4)),
    equipment=(eq("dumbbell", attach="hand_r"), eq("dumbbell", attach="hand_l"),
               eq("bench", attach="static", position=(0.0, 0.0, -190.0), rotation_deg=(0, 90, 0),
                  height=45.0)),
    errors=("Front foot too close to the bench (knee far past the toes).",
            "Hips rotating open."),
    physio_notes=("High gluteus maximus and medius demand on the front leg; useful for "
                  "unilateral strength asymmetries.",),
    sources=(BOREN, NSCA, ACE), unilateral=True, camera="three_quarter", anchor_side="R",
    tags=("dumbbell", "unilateral"),
)

EXERCISES = (forward_lunge, reverse_lunge, bulgarian_split_squat)
