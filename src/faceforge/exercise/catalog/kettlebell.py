"""Kettlebell work: the rack, the overhead positions, the carries and the get-up.

The Russian swing lives in :mod:`faceforge.exercise.catalog.athletic` with the
other ballistic lifts, and the goblet squat in ``lower_body`` with the squats.
This is the rest of the family.

Three things separate a kettlebell from a dumbbell and shape every entry here.
The mass hangs *below and behind* the handle, so a bell in the rack or
overhead is a lever the shoulder and trunk have to hold still rather than a
weight sitting over the hand. Loading one side at a time makes a frontal-plane
problem the obliques and gluteus medius have to solve. And the handle is
thick, so grip is a limiting factor in the carries long before the legs are.
"""

from __future__ import annotations

from faceforge.exercise.catalog._helpers import (
    ACE, CON, ECC, EKSTROM, EXRX, HIPS, ISO, LAKE, MCGILL, NSCA, P, S, ST, TRN,
    eq, grip, hinge, lunge, merge, mu, only, ph, pose, squat, stand,
)
from faceforge.exercise.model import Category, ExerciseDefinition

# One bell in the right hand, carried in the "rack": the forearm vertical
# against the ribs with the bell resting on the back of the forearm.
_RACK_R = only(shoulder_r_flex=35, shoulder_r_abduct=8, elbow_r_flex=140,
               forearm_r_rotate=-40, wrist_r_flex=-10)
_RACK_BOTH = only(shoulder_flex=35, shoulder_abduct=8, elbow_flex=140,
                  forearm_rotate=-40, wrist_flex=-10)
# Overhead lockout, one arm: the bell stacked over the shoulder.
_OVERHEAD_R = only(shoulder_r_flex=175, shoulder_r_abduct=10, elbow_r_flex=5,
                   forearm_r_rotate=-20)
# The free arm hangs at the side unless it is doing something.
_FREE_L = only(shoulder_l_flex=5, shoulder_l_abduct=8, elbow_l_flex=10)
_HANG_BOTH = only(shoulder_flex=5, shoulder_abduct=8, elbow_flex=8)

_CARRY_TRUNK = (mu("obliques", P, 0.75, note="stops the trunk folding toward the load"),
                mu("erector_spinae", P, 0.7),
                mu("quadratus_lumborum", P, 0.65, note="holds the pelvis against the load"),
                mu("rectus_abdominis", S, 0.5),
                mu("gluteus_medius", P, 0.7, note="keeps the pelvis level"),
                mu("forearm_flexors", P, 0.8, note="a thick handle ends the set"),
                mu("trapezius_upper", S, 0.5), mu("gluteus_maximus", S, 0.45),
                mu("quadriceps", S, 0.4), mu("gastrocnemius", S, 0.4))

kettlebell_deadlift = ExerciseDefinition(
    id="kettlebell_deadlift", name="Kettlebell deadlift", category=Category.LOWER_BODY,
    description="A hip hinge to a bell standing between the feet. The short range and the "
                "centred load make it the usual first hinge taught.",
    setup=("Bell between the mid-feet, feet hip width", "Hinge to it: hips back, shins near "
           "vertical, flat back", "Grip the handle with both hands, lats tight"),
    phases=(
        ph("Stand", CON, 1.4, merge(stand()[0], _HANG_BOTH, grip()),
           cues=("Push the floor away and drive the hips through",)),
        ph("Lockout", ISO, 0.5, merge(stand()[0], _HANG_BOTH, grip()),
           cues=("Stand tall; ribs down, glutes squeezed",)),
        ph("Lower", ECC, 1.8, merge(hinge(75, 25)[0], _HANG_BOTH, grip()), pitch=75,
           cues=("Hips back first; the bell tracks close to the shins",)),
        ph("Floor", ISO, 0.3, merge(hinge(75, 25)[0], _HANG_BOTH, grip()), pitch=75),
    ),
    muscles=(mu("gluteus_maximus", P, 0.85), mu("hamstrings", P, 0.8),
             mu("erector_spinae", P, 0.75), mu("quadriceps", S, 0.45),
             mu("adductors", S, 0.4), mu("latissimus_dorsi", S, 0.5,
                                         note="holds the bell against the body"),
             mu("forearm_flexors", S, 0.6), mu("trapezius_middle", ST, 0.35),
             mu("rectus_abdominis", ST, 0.4), mu("obliques", ST, 0.35)),
    equipment=(eq("kettlebell", radius=11.0),),
    errors=("Squatting it: knees forward, hips down, back rounded at the bottom.",
            "Letting the bell drift away from the shins.",
            "Yanking it off the floor rather than taking the slack out first."),
    physio_notes=("The load sits between the feet rather than in front of them, so the "
                  "moment arm on the lumbar spine is shorter than a barbell deadlift's at "
                  "the same trunk angle.",),
    sources=(LAKE, NSCA, ACE, EXRX), camera="side", tags=("kettlebell", "beginner"),
)

kettlebell_clean = ExerciseDefinition(
    id="kettlebell_clean", name="Kettlebell clean (single arm)", category=Category.ATHLETIC,
    description="A hinge that delivers the bell from between the thighs into the rack, the "
                "bell rolling round the wrist rather than flipping onto it.",
    setup=("Bell slightly in front, one hand on the handle", "Hike it back between the thighs",
           "Keep the elbow in and guide the bell close to the body"),
    phases=(
        ph("Hike", TRN, 0.3, merge(hinge(70, 20)[0], only(shoulder_r_flex=40, elbow_r_flex=10),
                                   _FREE_L, grip()), pitch=70,
           cues=("Bell high between the thighs, back flat",)),
        ph("Pull", CON, 0.35, merge(stand()[0], only(shoulder_r_flex=20, elbow_r_flex=70),
                                    _FREE_L, grip()), easing="ease_out",
           cues=("Snap the hips; keep the elbow close and the bell against the body",)),
        ph("Rack", ISO, 0.5, merge(stand()[0], _RACK_R, _FREE_L, grip()),
           cues=("Bell settles on the forearm, elbow into the ribs, wrist straight",)),
        ph("Drop", ECC, 0.5, merge(hinge(70, 20)[0], only(shoulder_r_flex=40, elbow_r_flex=10),
                                   _FREE_L, grip()), pitch=70, easing="ease_in",
           cues=("Guide it back down the same path into the next hinge",)),
    ),
    muscles=(mu("gluteus_maximus", P, 0.9), mu("hamstrings", P, 0.8),
             mu("erector_spinae", P, 0.7), mu("trapezius_upper", S, 0.6),
             mu("biceps_brachii", S, 0.45), mu("deltoid_anterior", S, 0.45),
             mu("forearm_flexors", P, 0.75, note="the handle rotates in the hand"),
             mu("obliques", S, 0.55, note="one-sided load"), mu("quadriceps", S, 0.4),
             mu("latissimus_dorsi", S, 0.5)),
    equipment=(eq("kettlebell", attach="hand_r", radius=11.0),),
    errors=("Letting the bell flip over and bang the forearm: it should roll round the wrist.",
            "Curling it up with the arm instead of hinging.",
            "Casting the bell away from the body, which turns it into a swing."),
    physio_notes=("The catch is where the coaching is: a loose grip and a close elbow let "
                  "the handle rotate in the hand, which is what stops the bell landing on "
                  "the wrist.",),
    sources=(LAKE, NSCA, ACE), camera="side", default_reps=5,
    tags=("kettlebell", "power"),
)

kettlebell_press = ExerciseDefinition(
    id="kettlebell_press", name="Kettlebell overhead press (single arm)",
    category=Category.UPPER_PUSH,
    description="A strict press from the rack to overhead. The bell hangs behind the wrist, "
                "so the shoulder holds a lever rather than a weight over the hand.",
    setup=("Start in the rack: elbow into the ribs, wrist straight", "Ribs down, glutes and "
           "abs tight: a one-sided press wants to bend the trunk", "Press slightly out and up"),
    phases=(
        ph("Press", CON, 1.4, merge(stand()[0], _OVERHEAD_R, _FREE_L, grip()),
           cues=("Press around the head, not in front of it",
                 "Finish with the biceps by the ear and the bell stacked over the shoulder")),
        ph("Lockout", ISO, 0.6, merge(stand()[0], _OVERHEAD_R, _FREE_L, grip()),
           cues=("Elbow straight, shoulder packed, ribs down",)),
        ph("Lower", ECC, 1.6, merge(stand()[0], _RACK_R, _FREE_L, grip()),
           cues=("Pull it back down into the rack under control",)),
        ph("Rack", ISO, 0.4, merge(stand()[0], _RACK_R, _FREE_L, grip())),
    ),
    muscles=(mu("deltoid_anterior", P, 0.9), mu("triceps_brachii", P, 0.75),
             mu("deltoid_lateral", P, 0.7), mu("trapezius_upper", S, 0.6),
             mu("serratus_anterior", S, 0.55, note="upward rotation of the scapula"),
             mu("rotator_cuff", S, 0.5, note="holds the lever still"),
             mu("obliques", P, 0.6, note="stops the trunk bending away from the load"),
             mu("erector_spinae", S, 0.5), mu("forearm_flexors", S, 0.55),
             mu("gluteus_maximus", ST, 0.4), mu("rectus_abdominis", ST, 0.45)),
    equipment=(eq("kettlebell", attach="hand_r", radius=11.0),),
    errors=("Leaning away from the bell instead of bracing against it.",
            "A bent wrist: the bell should sit over the forearm, not behind it.",
            "Pressing in front of the face, which is where shoulders get pinched."),
    physio_notes=("Unilateral overhead loading is a frontal-plane trunk task as much as a "
                  "shoulder one: the obliques on the loaded side work to keep the trunk "
                  "upright.",),
    sources=(NSCA, ACE, EXRX, EKSTROM), camera="front",
    tags=("kettlebell", "unilateral"),
)

kettlebell_thruster = ExerciseDefinition(
    id="kettlebell_thruster", name="Kettlebell thruster (single arm)",
    category=Category.ATHLETIC,
    description="A front squat into an overhead press in one movement: the legs start the "
                "bell and the shoulder finishes it.",
    setup=("Bell in the rack, feet shoulder width", "Squat with the trunk upright",
           "Drive out of the bottom and let the bar of the press ride the leg drive"),
    phases=(
        ph("Dip", ECC, 1.0, merge(squat(110, 115, 22)[0], _RACK_R, _FREE_L, grip()), pitch=22,
           cues=("Squat down with the elbow in and the trunk upright",)),
        ph("Drive", CON, 0.6, merge(stand()[0], _OVERHEAD_R, _FREE_L, grip()),
           easing="ease_out",
           cues=("Stand up hard and let the momentum carry the press",)),
        ph("Lockout", ISO, 0.4, merge(stand()[0], _OVERHEAD_R, _FREE_L, grip()),
           cues=("Bell over the shoulder, ribs down",)),
        ph("Return", ECC, 0.7, merge(stand()[0], _RACK_R, _FREE_L, grip()),
           cues=("Bring it back to the rack for the next rep",)),
    ),
    muscles=(mu("quadriceps", P, 0.85), mu("gluteus_maximus", P, 0.8),
             mu("deltoid_anterior", P, 0.75), mu("triceps_brachii", S, 0.6),
             mu("deltoid_lateral", S, 0.55), mu("erector_spinae", S, 0.6),
             mu("obliques", S, 0.55), mu("trapezius_upper", S, 0.5),
             mu("adductors", S, 0.4), mu("gastrocnemius", S, 0.4),
             mu("forearm_flexors", ST, 0.5), mu("rectus_abdominis", ST, 0.45)),
    equipment=(eq("kettlebell", attach="hand_r", radius=11.0),),
    errors=("Pressing after the legs have finished rather than through them.",
            "Losing the upright trunk in the squat, which drops the elbow.",
            "Treating it as conditioning and letting the lockout go soft."),
    physio_notes=("The press is the weak link, so the leg drive has to arrive before the "
                  "bell slows: a thruster done well has no pause at the top of the squat.",),
    sources=(NSCA, ACE, EXRX), camera="side", default_reps=5,
    tags=("kettlebell", "power", "conditioning"),
)

kettlebell_high_pull = ExerciseDefinition(
    id="kettlebell_high_pull", name="Kettlebell high pull", category=Category.ATHLETIC,
    description="A swing that finishes with the elbow pulled high and back, the bell arriving "
                "at chest height with the handle by the ribs.",
    setup=("Set up as for a swing", "Hips do the work; the pull is the finish, not the lift",
           "Elbow leads, high and wide"),
    phases=(
        ph("Hip snap", CON, 0.35, merge(stand()[0],
                                        only(shoulder_r_flex=15, shoulder_r_abduct=55,
                                             elbow_r_flex=100), _FREE_L, grip()),
           easing="ease_out",
           cues=("Stand up fast, then pull the elbow high and back",)),
        ph("Catch", ISO, 0.15, merge(stand()[0],
                                     only(shoulder_r_flex=15, shoulder_r_abduct=55,
                                          elbow_r_flex=100), _FREE_L, grip()),
           cues=("Handle by the ribs, forearm level",)),
        ph("Drop", ECC, 0.45, merge(hinge(70, 20)[0],
                                    only(shoulder_r_flex=40, elbow_r_flex=8), _FREE_L, grip()),
           pitch=70, easing="ease_in",
           cues=("Let it fall back into the hinge",)),
        ph("Backswing", ISO, 0.1, merge(hinge(70, 20)[0],
                                        only(shoulder_r_flex=40, elbow_r_flex=8), _FREE_L,
                                        grip()), pitch=70),
    ),
    muscles=(mu("gluteus_maximus", P, 0.9), mu("hamstrings", P, 0.8),
             mu("trapezius_upper", P, 0.7), mu("deltoid_lateral", S, 0.6),
             mu("rhomboids", S, 0.55), mu("erector_spinae", S, 0.65),
             mu("biceps_brachii", S, 0.4), mu("obliques", S, 0.5),
             mu("forearm_flexors", S, 0.6), mu("quadriceps", S, 0.4)),
    equipment=(eq("kettlebell", attach="hand_r", radius=11.0),),
    errors=("Pulling with the arm before the hips have finished.",
            "Letting the elbow drop below the hand at the top.",
            "Rounding the back in the backswing."),
    physio_notes=("It is the swing's hip pattern with an upper-back finish, which is why it "
                  "is often used as the step between a swing and a snatch.",),
    sources=(LAKE, NSCA, ACE), camera="side", default_reps=6,
    tags=("kettlebell", "power"),
)

kettlebell_snatch = ExerciseDefinition(
    id="kettlebell_snatch", name="Kettlebell snatch", category=Category.ATHLETIC,
    description="Hinge to overhead in one movement: the bell travels from between the thighs "
                "to a locked-out arm, turning over the hand at the top.",
    setup=("Set up as for a one-arm swing", "Pull the bell close and punch the hand through "
           "as it passes the head", "Loose grip: the handle must rotate"),
    phases=(
        ph("Hike", TRN, 0.25, merge(hinge(70, 20)[0],
                                    only(shoulder_r_flex=40, elbow_r_flex=8), _FREE_L, grip()),
           pitch=70, cues=("Bell high between the thighs",)),
        ph("Pull", CON, 0.35, merge(stand()[0],
                                    only(shoulder_r_flex=95, shoulder_r_abduct=25,
                                         elbow_r_flex=60), _FREE_L, grip()), easing="ease_out",
           cues=("Hips snap; the bell comes up close, elbow high",)),
        ph("Punch through", TRN, 0.2, merge(stand()[0], _OVERHEAD_R, _FREE_L, grip()),
           cues=("Punch the hand through as it passes the head; do not let it flip over",)),
        ph("Lockout", ISO, 0.4, merge(stand()[0], _OVERHEAD_R, _FREE_L, grip()),
           cues=("Arm straight, bell settled, ribs down",)),
        ph("Drop", ECC, 0.5, merge(hinge(70, 20)[0],
                                   only(shoulder_r_flex=40, elbow_r_flex=8), _FREE_L, grip()),
           pitch=70, easing="ease_in", cues=("Guide it down in an arc into the next hinge",)),
    ),
    muscles=(mu("gluteus_maximus", P, 1.0), mu("hamstrings", P, 0.85),
             mu("erector_spinae", P, 0.75), mu("deltoid_anterior", P, 0.7),
             mu("trapezius_upper", S, 0.65), mu("deltoid_lateral", S, 0.55),
             mu("triceps_brachii", S, 0.5), mu("obliques", S, 0.6),
             mu("forearm_flexors", P, 0.8, note="the handle turns in the hand at the top"),
             mu("rotator_cuff", ST, 0.5), mu("quadriceps", S, 0.45)),
    equipment=(eq("kettlebell", attach="hand_r", radius=11.0),),
    errors=("Letting the bell flip and bang the forearm instead of punching through it.",
            "Muscling it up with the shoulder rather than the hips.",
            "Gripping hard the whole way, which tears the hand and stops the rotation."),
    physio_notes=("The highest-power kettlebell lift of the common set, and the one most "
                  "often taught last: the overhead catch asks for shoulder stability at "
                  "speed on a load that is still moving.",),
    sources=(LAKE, NSCA, ACE, EXRX), camera="side", default_reps=5,
    tags=("kettlebell", "power"),
)

#: The sideways half of the hinge is a wrapper ROLL about the hips, negative
#: being toward the free hand.  `spine_lat_bend` alone did nothing the render
#: could see -- the spine DOFs turn vertebrae and the shoulders hang off the
#: pelvis root -- so this was a forward hinge pretending to be a windmill.
_WINDMILL_ROLL = -22.0
_WINDMILL_DOWN = merge(pose(hip_flex=55, knee_flex=8, spine_lat_bend=-10, spine_rotation=18),
                       _OVERHEAD_R, only(shoulder_l_flex=-10, elbow_l_flex=5), grip())

kettlebell_windmill = ExerciseDefinition(
    id="kettlebell_windmill", name="Kettlebell windmill", category=Category.CORE,
    description="A hip hinge sideways under a locked-out overhead bell: the eyes stay on the "
                "bell while the opposite hand travels down the inside of the front leg.",
    setup=("Bell locked out overhead in the right hand, feet turned about 45 deg away",
           "Weight into the right hip; the right leg stays straight",
           "Eyes on the bell throughout"),
    phases=(
        ph("Descend", ECC, 2.2, _WINDMILL_DOWN, roll=_WINDMILL_ROLL, pivot=HIPS,
           cues=("Push the loaded hip out and hinge sideways, not forward",
                 "Free hand slides down the inside of the front leg")),
        ph("Bottom", ISO, 0.6, _WINDMILL_DOWN, roll=_WINDMILL_ROLL, pivot=HIPS,
           cues=("Arm vertical, eyes on the bell, both knees straight",)),
        ph("Stand", CON, 2.0, merge(stand()[0], _OVERHEAD_R, _FREE_L, grip()),
           cues=("Drive the loaded hip back under the bell to stand up",)),
        ph("Top", ISO, 0.4, merge(stand()[0], _OVERHEAD_R, _FREE_L, grip())),
    ),
    muscles=(mu("obliques", P, 0.9, note="the frontal-plane work is the exercise"),
             mu("erector_spinae", P, 0.8),
             mu("quadratus_lumborum", P, 0.75, note="the side-bend is its plane"),
             mu("hamstrings", P, 0.7), mu("gluteus_maximus", S, 0.6),
             mu("gluteus_medius", S, 0.6), mu("deltoid_anterior", S, 0.55),
             mu("rotator_cuff", P, 0.6, note="holds the lockout through the whole range"),
             mu("rectus_abdominis", S, 0.5), mu("adductors", S, 0.45),
             mu("trapezius_upper", ST, 0.45), mu("forearm_flexors", ST, 0.5)),
    equipment=(eq("kettlebell", attach="hand_r", radius=11.0),),
    errors=("Bending forward instead of hinging sideways at the hip.",
            "Taking the eyes off the bell.",
            "Loading it before the shoulder can hold a still lockout overhead."),
    physio_notes=("A mobility and stability drill more than a strength one: it asks for "
                  "thoracic rotation, hamstring length and an overhead shoulder at the "
                  "same time, which is why it is usually loaded light.",),
    sources=(EKSTROM, MCGILL, NSCA, ACE), camera="front",
    tags=("kettlebell", "core", "unilateral"),
)

#: The halo is the one exercise in this module the rig cannot show.  A bell
#: held by two hands sits at the midpoint of the two grip points, so circling
#: it round the head means getting both hands to one side of the head --
#: which the shoulders will not do.  Measured over seven arm configurations,
#: the hands never come closer than 75 units apart (the shoulders are 21 out
#: on each side and adduction stops at -31.5 degrees) except with both arms
#: straight overhead, at 37, where the bell is on the midline anyway.  The
#: bell's whole available excursion is about +-8 in x and 11 in z against the
#: +-25 a halo needs.  What plays is a small arc over the head; the muscles,
#: cues and sources are right and the shape is as close as the shoulders get.
kettlebell_halo = ExerciseDefinition(
    id="kettlebell_halo", name="Kettlebell halo", category=Category.UPPER_PUSH,
    description="The bell circles the head close to it, held upside down by the horns. A "
                "shoulder and thoracic mobility drill, not a strength lift.",
    setup=("Hold the bell by the horns, base up, at chest height",
           "Circle it round the head keeping it close", "Ribs down: the trunk does not move"),
    phases=(
        ph("Round the head", TRN, 1.6,
           merge(stand()[0], only(shoulder_flex=40, shoulder_abduct=60, elbow_flex=135,
                                  shoulder_rotate=40), grip()),
           cues=("Elbows lead; keep the bell close to the head",)),
        ph("Behind", ISO, 0.4,
           merge(stand()[0], only(shoulder_flex=10, shoulder_abduct=85, elbow_flex=145,
                                  shoulder_rotate=55), grip()),
           cues=("Bell behind the head, elbows high, ribs still down",)),
        ph("Return", TRN, 1.6, merge(stand()[0], only(shoulder_flex=45, shoulder_abduct=20,
                                                     elbow_flex=130), grip()),
           cues=("Bring it back round to the chest",)),
    ),
    muscles=(mu("deltoid_lateral", P, 0.6), mu("deltoid_posterior", P, 0.55),
             mu("rotator_cuff", P, 0.65, note="the point of the drill"),
             mu("trapezius_middle", S, 0.5), mu("trapezius_upper", S, 0.45),
             mu("serratus_anterior", S, 0.45), mu("rhomboids", S, 0.4),
             mu("biceps_brachii", ST, 0.4), mu("forearm_flexors", ST, 0.5),
             mu("rectus_abdominis", ST, 0.45, note="stops the ribs flaring"),
             mu("obliques", ST, 0.4)),
    equipment=(eq("kettlebell", radius=10.0, hang=-4.0),),
    errors=("Letting the ribs flare and the low back extend as the bell passes behind.",
            "Circling it far from the head, which turns a mobility drill into a lever.",
            "Going heavy: this is a warm-up, not a lift."),
    physio_notes=("Used as a shoulder warm-up because it takes the glenohumeral joint "
                  "through rotation at several abduction angles under a light, controlled "
                  "load.",),
    sources=(ACE, EXRX, NSCA), camera="front", default_reps=4,
    tags=("kettlebell", "mobility", "warm-up"),
)

kettlebell_front_rack_carry = ExerciseDefinition(
    id="kettlebell_front_rack_carry", name="Kettlebell front rack carry",
    category=Category.CORE,
    description="Walking with one or two bells in the rack. The load sits in front of the "
                "trunk, so the anterior core works to stop the ribs flaring.",
    setup=("Clean the bells into the rack, elbows in, wrists straight",
           "Ribs down, tall through the crown", "Walk at a normal stride"),
    phases=(
        ph("Left stance", ISO, 0.6, merge(pose(hip_r_flex=20, knee_r_flex=25, hip_l_flex=-5,
                                               knee_l_flex=5), _RACK_BOTH, grip()),
           cues=("Breathe behind the brace; do not lean back",)),
        ph("Right stance", ISO, 0.6, merge(pose(hip_l_flex=20, knee_l_flex=25, hip_r_flex=-5,
                                                knee_r_flex=5), _RACK_BOTH, grip()),
           cues=("Level hips, quiet trunk",)),
    ),
    muscles=(mu("rectus_abdominis", P, 0.8, note="the racked load pulls the ribs up"),
             mu("obliques", P, 0.7), mu("erector_spinae", P, 0.7),
             mu("deltoid_anterior", S, 0.5), mu("trapezius_upper", S, 0.5),
             mu("gluteus_medius", P, 0.65, note="keeps the pelvis level in single stance"),
             mu("forearm_flexors", S, 0.6), mu("quadriceps", S, 0.4),
             mu("gluteus_maximus", S, 0.4), mu("gastrocnemius", S, 0.35)),
    equipment=(eq("kettlebell", attach="hand_r", radius=11.0),
               eq("kettlebell", attach="hand_l", radius=11.0)),
    errors=("Leaning back under the load and letting the ribs flare.",
            "Dropping an elbow, which puts the bell on the wrist.",
            "Holding the breath rather than breathing behind the brace."),
    physio_notes=("An anterior-loaded carry trains the same trunk position a front squat "
                  "asks for, at a load the shoulders can hold for time.",),
    sources=(MCGILL, EKSTROM, NSCA, ACE), camera="front", default_reps=1,
    tags=("kettlebell", "carry", "core"),
)

farmers_carry = ExerciseDefinition(
    id="farmers_carry", name="Farmer's carry", category=Category.CORE,
    description="Walking with a heavy bell in each hand. The trunk's job is to stay square "
                "while the grip runs out.",
    setup=("A bell either side, deadlift them up", "Shoulders back and down, arms hanging",
           "Walk tall with a normal stride"),
    phases=(
        ph("Left stance", ISO, 0.6, merge(pose(hip_r_flex=20, knee_r_flex=25, hip_l_flex=-5,
                                               knee_l_flex=5), _HANG_BOTH, grip()),
           cues=("Tall, ribs down, eyes ahead",)),
        ph("Right stance", ISO, 0.6, merge(pose(hip_l_flex=20, knee_l_flex=25, hip_r_flex=-5,
                                                knee_r_flex=5), _HANG_BOTH, grip()),
           cues=("Do not let the shoulders round forward",)),
    ),
    muscles=(mu("forearm_flexors", P, 0.9, note="the set usually ends here"),
             mu("trapezius_upper", P, 0.7), mu("erector_spinae", P, 0.7),
             mu("obliques", S, 0.55), mu("gluteus_medius", P, 0.65),
             mu("rectus_abdominis", S, 0.5), mu("rhomboids", S, 0.45),
             mu("gluteus_maximus", S, 0.45), mu("quadriceps", S, 0.4),
             mu("gastrocnemius", S, 0.4)),
    equipment=(eq("kettlebell", attach="hand_r", radius=11.0),
               eq("kettlebell", attach="hand_l", radius=11.0)),
    errors=("Shrugging the bells up rather than letting the arms hang.",
            "Short, shuffling steps.",
            "Setting them down by rounding the back."),
    physio_notes=("A bilateral carry is a grip and posture task; making it single-sided "
                  "(a suitcase carry) is what turns it into a frontal-plane trunk task.",),
    sources=(MCGILL, NSCA, ACE, EXRX), camera="front", default_reps=1,
    tags=("kettlebell", "carry", "grip"),
)

#: The get-up is the one exercise here that changes orientation mid-rep, which
#: `Phase.orientation` exists for: the first four phases are supine, the next
#: three standing, and the last comes back down.  A phase carrying its own
#: orientation is placed by that orientation's own base, so `base_position`
#: below applies to the supine half only.
#:
#: Supine, the loaded arm is vertical at 90 degrees of shoulder flexion, not
#: the 175 that means vertical when standing, and as the trunk comes up under
#: it the flexion has to come off by the same angle the trunk gains.
_TGU_SPREAD = merge(pose(hip_abduct=28, knee_flex=5),
                    only(shoulder_r_flex=-8, shoulder_r_abduct=45, elbow_r_flex=8,
                         shoulder_l_flex=-8, shoulder_l_abduct=45, elbow_l_flex=8),
                    grip())
_TGU_SET = merge(pose(hip_r_flex=45, knee_r_flex=112, ankle_r_flex=45,
                      hip_l_abduct=35, knee_l_flex=5),
                 only(shoulder_r_flex=90, shoulder_r_abduct=5, elbow_r_flex=5,
                      shoulder_l_flex=0, shoulder_l_abduct=45, elbow_l_flex=8),
                 grip())
_TGU_ELBOW = merge(pose(hip_r_flex=75, knee_r_flex=112, ankle_r_flex=45,
                        hip_l_flex=30,
                        hip_l_abduct=30, knee_l_flex=5,
                        spine_rotation=-18),
                   only(shoulder_r_flex=60, shoulder_r_abduct=5, elbow_r_flex=5,
                        shoulder_l_flex=0, shoulder_l_abduct=50, elbow_l_flex=95),
                   grip())
_TGU_HAND = merge(pose(hip_r_flex=90, knee_r_flex=112, ankle_r_flex=45,
                       hip_l_flex=52,
                       hip_l_abduct=30, knee_l_flex=5,
                       spine_rotation=-14),
                  only(shoulder_r_flex=45, shoulder_r_abduct=5, elbow_r_flex=5,
                       shoulder_l_flex=0, shoulder_l_abduct=55, elbow_l_flex=10),
                  grip())
# A 140-degree back knee folds the shank up behind the thigh -- that ankle
# measured 60 units in the air.  At 60 the shin lies along the floor: knee
# 13.5, ankle 16.0, toes 6.2, which is a half-kneel.
_TGU_KNEEL = merge(lunge("r", 90, 95, -20, 60, pitch=6, ankle_back=-40)[0],
                   _OVERHEAD_R, only(shoulder_l_flex=8, shoulder_l_abduct=25, elbow_l_flex=10),
                   grip())
_TGU_STAND = merge(stand()[0], _OVERHEAD_R, _FREE_L, grip())

turkish_get_up = ExerciseDefinition(
    id="turkish_get_up", name="Turkish get-up", category=Category.CORE,
    description="Standing up from flat on the floor with a bell locked out overhead, and "
                "lying back down again. Five positions, each of which has to be owned before "
                "the next.",
    setup=("Start on the back, spread-eagled, the bell on the floor in the right hand",
           "Roll it to the chest, press it, then bend the right knee and put that foot flat",
           "Left arm and leg stay at about 45 deg from the body; eyes on the bell until "
           "standing"),
    orientation="supine", anchor="none", base_position=(-85.0, 15.0, 0.0),
    phases=(
        ph("On the floor", ISO, 1.0, _TGU_SPREAD,
           cues=("Flat on the back, arms and legs at 45 deg, bell on the floor in the hand",
                 "Roll onto the side to take hold of it, never reach across for it")),
        ph("Press the bell", CON, 1.2, _TGU_SET,
           cues=("Press it to a straight arm over the shoulder",
                 "Bend the right knee, that foot flat; the left limbs stay out at 45")),
        ph("Roll to the elbow", CON, 1.6, _TGU_ELBOW, pitch=30, pivot=HIPS,
           cues=("Punch the bell up and roll onto the left elbow",)),
        ph("To the hand", CON, 1.2, _TGU_HAND, pitch=45, pivot=HIPS,
           cues=("Straighten the left arm; chest open, shoulder packed",)),
        ph("Sweep to half-kneel", TRN, 1.6, _TGU_KNEEL, pitch=6, orientation="standing",
           position=(0.0, 148.0, 0.0),
           cues=("Bridge the hips and sweep the left leg through to a half-kneel",)),
        ph("Stand", CON, 1.6, _TGU_STAND, orientation="standing",
           cues=("Windshield-wiper the back foot round, then stand",)),
        ph("Lockout", ISO, 0.6, _TGU_STAND, orientation="standing",
           cues=("Tall, bell stacked over the shoulder",)),
        ph("Reverse to the floor", ECC, 3.0, _TGU_SET,
           cues=("Retrace every step back down to the floor",)),
    ),
    muscles=(mu("obliques", P, 0.85), mu("rectus_abdominis", P, 0.75),
             mu("rotator_cuff", P, 0.8, note="holds the lockout through every position"),
             mu("deltoid_anterior", P, 0.7), mu("gluteus_maximus", P, 0.7),
             mu("serratus_anterior", S, 0.6), mu("erector_spinae", S, 0.6),
             mu("quadriceps", S, 0.6), mu("gluteus_medius", S, 0.6),
             mu("triceps_brachii", S, 0.5), mu("hamstrings", S, 0.5),
             mu("trapezius_upper", S, 0.5), mu("latissimus_dorsi", S, 0.5),
             mu("forearm_flexors", ST, 0.5), mu("adductors", ST, 0.4)),
    equipment=(eq("kettlebell", attach="hand_r", radius=11.0),
               eq("mat", attach="static")),
    errors=("Rushing between positions instead of owning each one.",
            "Letting the bell arm drift out of the vertical.",
            "Looking away from the bell before standing."),
    physio_notes=("Usually taught as a movement screen as much as an exercise: it asks for "
                  "an overhead shoulder, thoracic rotation, hip mobility and a half-kneeling "
                  "position in one sequence, and it exposes whichever is missing.",),
    sources=(EKSTROM, MCGILL, NSCA, ACE, EXRX), camera="three_quarter", default_reps=1,
    tags=("kettlebell", "core", "unilateral"),
)

EXERCISES = (kettlebell_deadlift, kettlebell_clean, kettlebell_press, kettlebell_thruster,
             kettlebell_high_pull, kettlebell_snatch, kettlebell_windmill, kettlebell_halo,
             kettlebell_front_rack_carry, farmers_carry, turkish_get_up)
