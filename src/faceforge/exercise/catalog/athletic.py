"""Power and athletic movements: kettlebell swing, jumps, power clean, slam, burpee."""

from __future__ import annotations

from faceforge.exercise.catalog._helpers import (
    ACE, CON, ECC, EXRX, ISO, LAKE, NEUMANN, NSCA, P, S, ST, TRN, ZEBIS, arms, combine, eq,
    flat_palm, grip, hinge, merge, mu, ph, pose, squat, toes_tucked, toes_on_floor
)
from faceforge.exercise.catalog.olympic import _EXT_C, _KNEE_C, _RACK, _START_C
from faceforge.exercise.model import Category, ExerciseDefinition

#: Toes bent back onto their pads under a tucked foot (`toes_tucked`),
#: so the foot is not one rigid wedge balanced on its longest toe.
_TUCKED = toes_tucked(45.0)

# Two hands on ONE handle at the midline: the equipment rig centres an
# `attach="hands"` item between the grip points, so a pose that leaves the
# arms hanging at the sides puts the bell on the midline and the hands 80 cm
# apart on either side of it.  Measured 2026-09-16, grip width by shoulder
# abduction:  +8 -> 101,  0 -> 81,  -10 -> 54,  -20 -> 28,  -25 -> 14.8.
# -25 is a two-handed grip (12 cm) and brings the worst hand-to-bell gap
# from 40 to 3.  A bell in EACH hand (the carries) keeps the wide pose.
_SWING_BOTTOM = merge(hinge(70, 20)[0], arms(flex=45, abduct=-25, elbow=5), grip())
_SWING_TOP = merge(pose(knee_flex=3), arms(flex=90, abduct=-25, elbow=5), grip())

kettlebell_swing = ExerciseDefinition(
    id="kettlebell_swing", name="Kettlebell swing (Russian)", category=Category.ATHLETIC,
    description="A ballistic hip hinge: the bell is hiked back between the thighs and the "
                "hips snap forward to float it to chest height; the arms only guide it.",
    setup=("Feet slightly wider than the hips, bell a foot in front", "Hinge, grip the "
           "handle, lats tight, flat back", "Hike the bell back to start"),
    phases=(
        ph("Hip snap", CON, 0.35, _SWING_TOP, pitch=0, easing="ease_out",
           cues=("Stand up fast: drive the hips forward, squeeze the glutes",
                 "Arms straight; the bell floats to chest height")),
        ph("Float", ISO, 0.15, _SWING_TOP, cues=("Plank standing: ribs down, knees straight",)),
        ph("Drop", ECC, 0.45, _SWING_BOTTOM, pitch=70, easing="ease_in",
           cues=("Let the bell fall; hinge when the forearms touch the thighs",)),
        ph("Backswing", ISO, 0.1, _SWING_BOTTOM, pitch=70,
           cues=("Shins vertical, back flat, bell high between the thighs",)),
    ),
    muscles=(mu("gluteus_maximus", P, 1.0), mu("hamstrings", P, 0.9, note="semitendinosus bias"),
             mu("erector_spinae", P, 0.8), mu("adductors", S, 0.5), mu("quadriceps", S, 0.4),
             mu("latissimus_dorsi", S, 0.5, note="keeps the bell close"),
             mu("forearm_flexors", S, 0.6), mu("deltoid_anterior", S, 0.4),
             mu("rectus_abdominis", S, 0.5), mu("obliques", S, 0.4), mu("trapezius_upper", S, 0.3)),
    equipment=(eq("kettlebell"),),
    errors=("Squatting the swing (knees forward, hips down).", "Lifting with the arms / "
            "shoulders.", "Hyperextending the lumbar spine at the top.", "Bell dropping "
            "below the knees."),
    physio_notes=("Peak hip extension velocity comparable to jump training (Lake 2012); "
                  "the swing preferentially loads semitendinosus (Zebis 2013).",),
    sources=(LAKE, ZEBIS, NSCA), camera="side", default_reps=5, tags=("kettlebell", "power"),
)

_CMJ_ARMS_BACK = arms(flex=-40, elbow=10)
_CMJ_ARMS_UP = arms(flex=150, elbow=5)

countermovement_jump = ExerciseDefinition(
    id="countermovement_jump", name="Countermovement vertical jump", category=Category.ATHLETIC,
    description="A quick dip to ~70 deg of hip and knee flexion with the arms swinging back, "
                "then triple extension of hips, knees and ankles into flight.",
    setup=("Feet hip-width, arms relaxed", "Look ahead"),
    phases=(
        ph("Countermovement", ECC, 0.4, merge(squat(70, 70, 30)[0], _CMJ_ARMS_BACK), pitch=30,
           easing="ease_in", cues=("Dip fast: hips back, arms swing back",)),
        ph("Take-off", CON, 0.25, merge(pose(knee_flex=2, ankle_flex=-35, toe_curl=toes_on_floor(0, 0, 2, -35)), _CMJ_ARMS_UP),
           easing="ease_out", cues=("Triple extension: hips, knees, ankles; arms drive up",)),
        ph("Flight", TRN, 0.35, merge(pose(hip_flex=15, knee_flex=20, ankle_flex=-20), _CMJ_ARMS_UP),
           lift=35.0, cues=("Tall in the air, feet under the hips",)),
        ph("Landing", ECC, 0.35, merge(squat(60, 60, 25)[0], arms(flex=20, elbow=20)), pitch=25,
           lift=0.0, cues=("Land softly on the balls of the feet, knees over the toes",)),
        ph("Reset", TRN, 0.6, merge(pose(), arms(elbow=10))),
    ),
    muscles=(mu("gluteus_maximus", P, 0.95), mu("quadriceps", P, 0.95), mu("gastrocnemius", P, 0.85),
             mu("soleus", S, 0.7), mu("hamstrings", S, 0.6), mu("hip_flexors", S, 0.4),
             mu("erector_spinae", S, 0.5), mu("deltoid_anterior", S, 0.5),
             mu("rectus_abdominis", ST, 0.4), mu("gluteus_medius", ST, 0.4)),
    errors=("Knees caving on landing.", "Landing stiff-legged.", "Dipping too deep and slow."),
    physio_notes=("The countermovement uses the stretch-shortening cycle; landing mechanics "
                  "(soft, knees out) are the ACL-prevention focus.",),
    sources=(NEUMANN, NSCA), camera="side", default_reps=3, tags=("no equipment", "power", "plyometric"),
)

box_jump = ExerciseDefinition(
    id="box_jump", name="Box jump", category=Category.ATHLETIC,
    description="A countermovement jump onto a box, landing softly in a quarter squat, then a "
                "step down.",
    setup=("Box a comfortable jump height, ~30 cm in front", "Arms ready to swing"),
    lock_horizontal=True,
    phases=(
        ph("Countermovement", ECC, 0.4, merge(squat(75, 75, 30)[0], _CMJ_ARMS_BACK), pitch=30,
           easing="ease_in", cues=("Dip and swing the arms back",)),
        ph("Take-off", CON, 0.25, merge(pose(knee_flex=2, ankle_flex=-35, toe_curl=toes_on_floor(0, 0, 2, -35)), _CMJ_ARMS_UP),
           easing="ease_out", cues=("Explode up and forward",)),
        ph("Flight", TRN, 0.35, merge(pose(hip_flex=70, knee_flex=75, ankle_flex=10), arms(flex=60, elbow=30)),
           lift=80.0, travel=(0.0, 30.0), cues=("Tuck the knees, feet toward the box",)),
        ph("Land on box", ECC, 0.35, merge(squat(60, 60, 20)[0], arms(flex=30, elbow=30)), pitch=20,
           lift=60.0, travel=(0.0, 60.0), cues=("Whole foot on the box, land quietly",)),
        ph("Stand", CON, 0.5, merge(pose(), arms(elbow=10)), lift=60.0, travel=(0.0, 60.0),
           cues=("Stand tall on the box",)),
        ph("Step down", TRN, 1.2, merge(pose(), arms(elbow=10)), lift=0.0, travel=(0.0, 0.0),
           cues=("Step down one foot at a time; never jump down",)),
    ),
    muscles=(mu("gluteus_maximus", P, 0.95), mu("quadriceps", P, 0.95), mu("gastrocnemius", P, 0.85),
             mu("soleus", S, 0.7), mu("hamstrings", S, 0.6), mu("hip_flexors", S, 0.6),
             mu("erector_spinae", S, 0.5), mu("deltoid_anterior", S, 0.5),
             mu("rectus_abdominis", ST, 0.4), mu("gluteus_medius", ST, 0.4)),
    equipment=(eq("plyo_box", attach="static", position=(0.0, 0.0, 60.0), height=60.0),),
    errors=("Landing in a deep squat (box too high).", "Jumping down off the box.",
            "Knees caving in."),
    physio_notes=("Landing on the box removes most of the impact of a vertical jump, which is "
                  "why it is used for power with low landing load.",),
    sources=(NSCA, ACE), camera="side", default_reps=3, tags=("box", "power", "plyometric"),
)

# These were a second, identical copy of the clean positions in `olympic.py`,
# so correcting the grip width and the start depth there left the power clean
# -- the exercise they were copied for -- still sliding 15 units a hand and
# holding its bar 35 above the floor.  One copy now, imported.
_CLEAN_START, _CLEAN_KNEE, _CLEAN_EXT = _START_C, _KNEE_C, _EXT_C
_CLEAN_CATCH = merge(squat(65, 65, 10)[0], _RACK, grip())
_CLEAN_STAND = merge(pose(knee_flex=3), _RACK, grip())

power_clean = ExerciseDefinition(
    id="power_clean", name="Power clean", category=Category.ATHLETIC,
    description="The bar is pulled from the floor past the knees, then the hips, knees and "
                "ankles extend explosively and the lifter drops under to catch it on the "
                "shoulders in a quarter squat.",
    setup=("Bar over mid-foot, hook grip just outside the knees", "Hips above the knees, "
           "shoulders over the bar, back flat", "Arms straight and relaxed"),
    phases=(
        ph("First pull", CON, 0.6, _CLEAN_KNEE, pitch=40,
           cues=("Push the floor away; keep the back angle; bar close",)),
        ph("Second pull", CON, 0.25, _CLEAN_EXT, pitch=0, easing="ease_out",
           cues=("Explode: hips through, shrug, onto the toes; elbows high and outside",)),
        ph("Catch", ECC, 0.3, _CLEAN_CATCH, pitch=10, easing="ease_in",
           cues=("Pull under; elbows whip through to the front rack; absorb in a quarter squat",)),
        ph("Recover", CON, 0.6, _CLEAN_STAND, pitch=0, cues=("Stand tall with the bar racked",)),
        ph("Lower to the floor", TRN, 1.2, _CLEAN_START, pitch=50,
           cues=("Return the bar to the thighs, then hinge and squat it down",)),
    ),
    muscles=(mu("gluteus_maximus", P, 1.0), mu("quadriceps", P, 0.95), mu("hamstrings", P, 0.8),
             mu("erector_spinae", P, 0.85), mu("gastrocnemius", S, 0.7), mu("soleus", S, 0.6),
             mu("trapezius_upper", S, 0.8, note="the shrug"), mu("deltoid_anterior", S, 0.5),
             mu("deltoid_lateral", S, 0.4), mu("biceps_brachii", S, 0.4),
             mu("forearm_flexors", S, 0.7), mu("rectus_abdominis", S, 0.45),
             mu("latissimus_dorsi", ST, 0.4)),
    equipment=(eq("barbell", plates=2),),
    errors=("Pulling early with the arms.", "Bar swinging away from the body.",
            "Catching with the elbows low (wrists take the load).", "Hips rising first "
            "in the first pull."),
    physio_notes=("Triple extension of hips, knees and ankles is the athletic template "
                  "shared with jumping and sprinting.",),
    sources=(NSCA, EXRX), camera="three_quarter", default_reps=3, tags=("barbell", "power", "olympic"),
)

# Both hands on one ball: with the arms at the default abduction the grip is
# 80.8 units wide against a ball 24 across, so it floated 28 units clear of
# them.  Adducting to -25 closes the grip to 39.7; a 36-unit slam ball (a real
# one is 28-35 cm, this figure's scale makes that 36-44) then meets the hands.
# The ball is rigid, so the grip has to be about as wide as it is: measured
# 2026-09-16/17, grip width by abduction --
#     abduct          -25   -20   -15    -5
#     reach overhead 14.3  ~28   40.9  67.7
#     slam           15.6  ~29   41.7  67.9
#     pick up        39.6   --   56.5  72.9   (its elbow is 60, not 5-10)
# -20 on the two overhead-to-slam poses and the adduction limit (-31) on the
# pick-up hold it near 28-33, which is a 14-radius ball -- 23 cm, a real slam
# ball.  18 was a bad guess made before the adduction had closed the grip, and
# it put 5.8 units of ball through the calves at the slam.
_SLAM_TOP = merge(pose(knee_flex=5, ankle_flex=-15),
                  arms(flex=175, abduct=-20, elbow=5), grip())
_SLAM_BOTTOM = merge(squat(100, 70, 45)[0], arms(flex=25, abduct=-20, elbow=10), grip())

medicine_ball_slam = ExerciseDefinition(
    id="medicine_ball_slam", name="Medicine ball slam", category=Category.ATHLETIC,
    description="The ball is lifted overhead onto the toes and thrown down into the floor "
                "with the whole trunk; the lifter squats to pick it up.",
    setup=("Feet shoulder-width, ball at the chest", "Brace; the slam is a whole-body "
           "flexion, not just the arms"),
    phases=(
        ph("Reach overhead", CON, 0.6, _SLAM_TOP, cues=("Rise onto the toes, ball high",)),
        ph("Slam", CON, 0.35, _SLAM_BOTTOM, pitch=45, easing="ease_in",
           cues=("Throw the ball down hard: hips back, trunk folds, arms follow",)),
        ph("Pick up", TRN, 0.8, merge(squat(70, 60, 30)[0],
                                      arms(flex=40, abduct=-31, elbow=60), grip()), pitch=30,
           cues=("Squat to the ball with a flat back",)),
    ),
    muscles=(mu("latissimus_dorsi", P, 0.9), mu("rectus_abdominis", P, 0.85), mu("obliques", S, 0.6),
             mu("triceps_brachii", S, 0.5), mu("deltoid_anterior", S, 0.6),
             mu("pectoralis_major", S, 0.4), mu("quadriceps", S, 0.5), mu("gluteus_maximus", S, 0.6),
             mu("erector_spinae", S, 0.5), mu("gastrocnemius", S, 0.4), mu("hip_flexors", S, 0.4)),
    equipment=(eq("medicine_ball", radius=14.0),),
    errors=("Rounding the back to pick the ball up.", "Slamming with the arms only."),
    physio_notes=("A trunk-flexion power exercise; keep the volume low with disc-related "
                  "back pain.",),
    sources=(ACE, NSCA), camera="side", default_reps=4, tags=("medicine ball", "power"),
)

_BURPEE_ARMS_DOWN = combine(arms(flex=110, abduct=15, elbow=10), flat_palm())

burpee = ExerciseDefinition(
    id="burpee", name="Burpee", category=Category.ATHLETIC,
    description="Squat down and place the hands, kick the feet back to a push-up position, "
                "return the feet and jump up with the arms overhead.",
    setup=("Feet shoulder-width", "Move fast but keep the plank straight"),
    anchor="none",
    phases=(
        ph("Squat down", ECC, 0.4, merge(squat(125, 120, 45)[0], _BURPEE_ARMS_DOWN), pitch=45,
           position=(0.0, 140.0, 0.0), cues=("Hands to the floor just in front of the feet",)),
        # The burpee is `anchor="none"`, so each phase's wrapper height is the
        # only thing holding it off the floor and it has to be measured, not
        # guessed.  At the authored heights the straight-arm plank drove the
        # fingers 14.7 units THROUGH the floor while the bent-arm push-up
        # floated them 16.2 above it -- the two poses reach different
        # distances from the shoulder and were given the same allowance.
        #     Press  (straight arm)  y 62 -> -14.7   70 -> -6.7   77 -> +0.3
        #     Push-up (elbow 95)     y 40 -> +16.2   24 -> +0.2
        ph("Kick back", CON, 0.35, merge(pose(ankle_flex=45, toe_curl=_TUCKED), arms(flex=90, abduct=15, elbow=0)),
           orientation="prone", position=(-85.0, 77.0, 0.0), easing="ease_out",
           cues=("Jump the feet back into a plank; hips level",)),
        ph("Push-up", ECC, 0.4, merge(pose(ankle_flex=45, toe_curl=_TUCKED), arms(flex=55, abduct=45, elbow=95)),
           orientation="prone", position=(-85.0, 24.0, 0.0), cues=("Chest to the floor",)),
        ph("Press", CON, 0.35, merge(pose(ankle_flex=45, toe_curl=_TUCKED), arms(flex=90, abduct=15, elbow=0)),
           orientation="prone", position=(-85.0, 77.0, 0.0)),
        ph("Feet in", CON, 0.35, merge(squat(125, 120, 45)[0], _BURPEE_ARMS_DOWN), pitch=45,
           position=(0.0, 140.0, 0.0), cues=("Jump the feet back under the hips",)),
        ph("Jump", CON, 0.3, merge(pose(knee_flex=5, ankle_flex=-30, toe_curl=toes_on_floor(0, 0, 5, -30)), arms(flex=170, elbow=5)),
           position=(0.0, 225.0, 0.0), easing="ease_out", cues=("Stand and jump, arms overhead",)),
        ph("Land", ECC, 0.35, merge(squat(40, 40, 15)[0], arms(flex=10, elbow=15)), pitch=15,
           position=(0.0, 196.0, 0.0), cues=("Land soft, straight into the next rep",)),
    ),
    muscles=(mu("quadriceps", P, 0.85), mu("gluteus_maximus", P, 0.85), mu("pectoralis_major", P, 0.7),
             mu("triceps_brachii", S, 0.6), mu("deltoid_anterior", S, 0.6), mu("gastrocnemius", S, 0.6),
             mu("hip_flexors", S, 0.6), mu("hamstrings", S, 0.5), mu("rectus_abdominis", S, 0.5),
             mu("obliques", S, 0.4), mu("erector_spinae", S, 0.4), mu("serratus_anterior", S, 0.4)),
    errors=("Sagging hips in the plank.", "Landing the jump stiff-legged.",
            "Rounding the back when the hands go down."),
    physio_notes=("Whole-body conditioning with impact; scale by removing the push-up or the "
                  "jump.",),
    sources=(ACE,), camera="side", default_reps=3, tags=("no equipment", "cardio", "power"),
)

EXERCISES = (kettlebell_swing, countermovement_jump, box_jump, power_clean, medicine_ball_slam,
             burpee)
