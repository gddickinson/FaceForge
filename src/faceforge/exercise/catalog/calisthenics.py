"""Bodyweight strength and skill: the harder relatives of the pull-up, and the holds.

What separates these from the bodyweight work in the push, pull and core
modules is that the limit is a *position* rather than a load.  A muscle-up is
a pull-up and a dip with a transition nobody can fake; a pistol squat is a
squat that also asks for ankle range and single-leg balance; an L-sit is a
hold whose difficulty is entirely the lever.

Progression is by leverage, not by plates, so each entry's notes say what the
easier version is.
"""

from __future__ import annotations

from faceforge.exercise.catalog._helpers import (
    ACE, BENCH_TOP, CALATAYUD, CON, ECC, EKSTROM, EXRX, ISO, KOLBER, MCGILL, NSCA, P,
    PULLUP_YOUDAS, S, ST, TRN, arms, eq, flat_palm, flat_foot_ankle, grip, merge, mu, only, ph, pose,
)
from faceforge.exercise.model import Category, ExerciseDefinition

#: Legs of a body hanging from a bar (shared with the pull-up family).
_HANG_LEGS = only(hip_flex=10, knee_flex=30)

#: The knee, in body coordinates: 60 below the hips (`HIPS` z = -81) plus the
#: femur measured on the rig for the bike crank (`conditioning._THIGH`).
KNEE = (0.0, 0.0, -141.0)


# ── The bar: a pull-up that carries on into a dip ──────────────────────

_MU_HANG = merge(pose(), _HANG_LEGS,
                 arms(flex=10, abduct=165, rotate=90, elbow=5, forearm=-90), grip())
_MU_PULL = merge(pose(), _HANG_LEGS,
                 arms(flex=0, abduct=40, rotate=90, elbow=145, forearm=-90), grip())
_MU_OVER = merge(pose(hip_flex=25, knee_flex=25),
                 arms(flex=-25, abduct=55, rotate=45, elbow=140, forearm=-45), grip())
_MU_PRESS = merge(pose(hip_flex=20, knee_flex=60),
                  arms(flex=-20, abduct=25, rotate=15, elbow=85, forearm=-25), grip())
_MU_TOP = merge(pose(hip_flex=20, knee_flex=70),
                arms(flex=0, abduct=10, rotate=10, elbow=0), grip())

muscle_up = ExerciseDefinition(
    id="muscle_up", name="Muscle-up (bar)", category=Category.CALISTHENICS,
    description="A pull-up that continues into a dip: the chest clears the bar, the wrists "
                "roll over it and the arms press the body up to a straight-arm support. The "
                "transition is the whole exercise.",
    setup=("False grip if you have it: a wrist already over the bar saves the transition",
           "Pull higher than a pull-up -- sternum to the bar, not the chin",
           "Lean the chest forward over the bar as the elbows come through"),
    orientation="hanging", anchor="hands", anchor_point=(0.0, 275.0, 0.0),
    phases=(
        ph("Pull", CON, 0.9, _MU_PULL, easing="ease_out",
           cues=("Pull explosively; sternum to the bar, not the chin",
                 "Elbows down and back, shoulder blades depressed")),
        ph("Transition", TRN, 0.6, _MU_OVER,
           cues=("Lean the chest forward over the bar and whip the elbows through",
                 "This is where the rep is won or lost")),
        ph("Press", CON, 0.8, _MU_PRESS, cues=("Press out of the bottom of the dip",)),
        ph("Support", ISO, 0.5, _MU_TOP,
           cues=("Locked out above the bar, shoulders down away from the ears",)),
        ph("Lower", ECC, 1.8, _MU_HANG,
           cues=("Reverse it under control: back through the transition to the hang",)),
    ),
    muscles=(mu("latissimus_dorsi", P, 1.0), mu("pectoralis_major", P, 0.85),
             mu("triceps_brachii", P, 0.85, note="the dip half"),
             mu("biceps_brachii", P, 0.8), mu("deltoid_anterior", S, 0.7),
             mu("trapezius_lower", S, 0.6), mu("rhomboids", S, 0.6),
             mu("serratus_anterior", S, 0.6), mu("brachialis", S, 0.6),
             mu("rectus_abdominis", P, 0.7, note="holds the body against the swing"),
             mu("obliques", S, 0.55),
             mu("forearm_flexors", P, 0.85, note="the false grip"),
             mu("rotator_cuff", ST, 0.5)),
    equipment=(eq("pullup_bar", attach="static", height=275.0),),
    errors=("Kipping so hard the transition becomes a swing the shoulders have to catch.",
            "Pulling only to the chin: the chest has to clear the bar.",
            "Attempting it before a strict pull-up and a bar dip are both easy."),
    physio_notes=("The usual prerequisite is a strict pull-up to the sternum and a clean bar "
                  "dip; the transition loads the shoulder at end-range internal rotation, "
                  "which is why it is taught after both.",),
    sources=(PULLUP_YOUDAS, KOLBER, NSCA, ACE), camera="three_quarter",
    # The preset keeps its offset from the target, so a high target lifts the
    # camera with it: 230 put it at y=285 and the rep was filmed from above.
    camera_target=(0.0, 195.0, 0.0), default_reps=3, tags=("bodyweight", "bar", "skill"),
)

# ── Parallel bars: the L-sit ───────────────────────────────────────────

_LSIT_TUCK = merge(pose(hip_flex=85, knee_flex=110),
                   arms(flex=0, abduct=10, elbow=0), grip())
_LSIT = merge(pose(hip_flex=88, knee_flex=5), arms(flex=0, abduct=10, elbow=0), grip())

l_sit = ExerciseDefinition(
    id="l_sit", name="L-sit (parallel bars)", category=Category.CALISTHENICS,
    description="Supported on straight arms with the legs held horizontal, so the body makes "
                "an L. A hold whose difficulty is the lever: the straighter the legs, the "
                "harder it is.",
    setup=("Straight arms, shoulders pushed down away from the ears",
           "Legs straight and together, held at or just above horizontal",
           "Tucked or one-leg versions first"),
    orientation="hanging", anchor="hands", anchor_point=(0.0, 125.0, 0.0),
    phases=(
        ph("Lift", CON, 1.0, _LSIT, cues=("Press down hard and lift the legs to horizontal",)),
        ph("Hold", ISO, 5.0, _LSIT,
           cues=("Shoulders down, chest up, legs straight and still",
                 "Breathe: the hold is not a breath-hold")),
        ph("Tuck", ECC, 1.0, _LSIT_TUCK, cues=("Fold the knees in when the hold runs out",)),
        ph("Rest", TRN, 0.8, _LSIT_TUCK),
    ),
    muscles=(mu("rectus_abdominis", P, 0.95), mu("hip_flexors", P, 0.9),
             mu("quadriceps", P, 0.7, note="holds the knees straight"),
             mu("obliques", S, 0.6),
             mu("triceps_brachii", P, 0.7, note="locks the arms"),
             mu("latissimus_dorsi", S, 0.6, note="depresses the shoulder girdle"),
             mu("serratus_anterior", S, 0.6), mu("pectoralis_major", S, 0.5),
             mu("trapezius_lower", S, 0.5), mu("forearm_flexors", ST, 0.5)),
    equipment=(eq("dip_station", attach="static", height=125.0),),
    errors=("Shoulders shrugging up to the ears.",
            "Knees quietly bending to shorten the lever.", "Holding the breath."),
    physio_notes=("A straight-leg L-sit is a hip-flexor endurance task as much as an "
                  "abdominal one; the tuck and one-leg versions shorten the lever and are "
                  "the standard progression.",),
    sources=(EKSTROM, MCGILL, NSCA, ACE), camera="side", camera_target=(0.0, 140.0, 0.0),
    default_reps=1, tags=("bodyweight", "isometric", "skill"),
)

# ── One leg: the pistol squat ──────────────────────────────────────────

_PISTOL_PITCH = 30.0
_PISTOL_BOTTOM = merge(
    pose(hip_r_flex=120, knee_r_flex=135,
         ankle_r_flex=flat_foot_ankle(_PISTOL_PITCH, 120, 135),
         hip_l_flex=115, knee_l_flex=10, ankle_l_flex=15),
    arms(flex=85, abduct=10, elbow=10))
_PISTOL_TOP = merge(
    pose(hip_r_flex=5, knee_r_flex=8, ankle_r_flex=flat_foot_ankle(0, 5, 8),
         hip_l_flex=35, knee_l_flex=12, ankle_l_flex=10),
    arms(flex=45, abduct=10, elbow=10))

pistol_squat = ExerciseDefinition(
    id="pistol_squat", name="Pistol squat", category=Category.CALISTHENICS,
    description="A full squat on one leg with the other held out in front. The limit is "
                "usually ankle range and single-leg balance rather than leg strength.",
    setup=("Stand on one leg, the other extended forward", "Arms forward as a counterweight",
           "Sit all the way down, keeping the standing heel flat"),
    anchor="feet", anchor_side="R", unilateral=True,
    phases=(
        ph("Descend", ECC, 2.4, _PISTOL_BOTTOM, pitch=_PISTOL_PITCH,
           cues=("Sit straight down over the standing foot; keep the heel down",
                 "The free leg stays straight and clear of the floor")),
        ph("Bottom", ISO, 0.6, _PISTOL_BOTTOM, pitch=_PISTOL_PITCH,
           cues=("Hamstring on calf, balanced over the mid-foot",)),
        ph("Stand", CON, 2.0, _PISTOL_TOP,
           cues=("Drive through the whole foot without pitching forward",)),
        ph("Top", ISO, 0.4, _PISTOL_TOP),
    ),
    muscles=(mu("quadriceps", P, 1.0, side="R"), mu("gluteus_maximus", P, 0.85, side="R"),
             mu("gluteus_medius", P, 0.8, side="R",
                note="holds the pelvis level on one leg"),
             mu("adductors", S, 0.6, side="R"), mu("hamstrings", S, 0.5, side="R"),
             mu("hip_flexors", S, 0.6, side="L", note="holds the free leg up"),
             mu("quadriceps", S, 0.5, side="L", note="keeps the free knee straight"),
             mu("tibialis_anterior", P, 0.6, note="balance, and the deeply dorsiflexed ankle"),
             mu("soleus", S, 0.5), mu("foot_intrinsics", ST, 0.5),
             mu("erector_spinae", S, 0.55), mu("rectus_abdominis", S, 0.5),
             mu("hip_external_rotators", S, 0.5)),
    equipment=(),
    errors=("The heel lifting, which is ankle range rather than weakness.",
            "The knee collapsing inward at the bottom.",
            "Rounding the low back to reach depth."),
    physio_notes=("Most people fail a pistol on ankle dorsiflexion or gluteus medius rather "
                  "than on the quadriceps; elevating the heel or holding a counterweight "
                  "are the usual regressions.",),
    sources=(NSCA, ACE, EXRX), camera="side", tags=("bodyweight", "unilateral", "skill"),
)

# ── Prone: the two push-up variations ──────────────────────────────────
# The push-up numbers are the measured ones from ``upper_push.push_up``: hands
# and toes on the floor, the straight body inclined head-up ~20 deg on locked
# arms and ~8 at the bottom, which is why the pitches are negative.

_ARCHER_TOP = merge(pose(ankle_flex=45), arms(flex=65, abduct=40, elbow=0), flat_palm())
_ARCHER_R = merge(pose(ankle_flex=45),
                  only(shoulder_r_flex=35, shoulder_r_abduct=55, elbow_r_flex=95,
                       shoulder_l_flex=45, shoulder_l_abduct=75, elbow_l_flex=10),
                  flat_palm())

archer_push_up = ExerciseDefinition(
    id="archer_push_up", name="Archer push-up", category=Category.CALISTHENICS,
    description="A wide push-up that lowers onto one arm while the other stays straight, so "
                "one side takes most of the load. The step between a push-up and a one-arm "
                "push-up.",
    setup=("Hands wider than a push-up, fingers forward",
           "Lower toward one hand; the far arm stays straight",
           "Hips square: the body does not roll"),
    orientation="prone", anchor="hands", base_position=(-85.0, 30.0, 0.0),
    phases=(
        ph("Lower right", ECC, 1.8, _ARCHER_R, pitch=0,
           cues=("Bend one arm and slide onto it; the far arm only supports",)),
        ph("Bottom", ISO, 0.4, _ARCHER_R, pitch=0,
           cues=("Hips square, body still a plank",)),
        ph("Press", CON, 1.4, _ARCHER_TOP, pitch=-7,
           cues=("Push back to the middle and lock both arms",)),
        ph("Top", ISO, 0.4, _ARCHER_TOP, pitch=-7),
    ),
    muscles=(mu("pectoralis_major", P, 0.9), mu("triceps_brachii", P, 0.8),
             mu("deltoid_anterior", S, 0.65),
             mu("serratus_anterior", P, 0.65, note="holds the scapula on the working side"),
             mu("obliques", P, 0.7, note="stops the body rolling toward the bent arm"),
             mu("rectus_abdominis", S, 0.6), mu("gluteus_maximus", S, 0.5),
             mu("latissimus_dorsi", S, 0.45, note="the straight arm resists"),
             mu("rotator_cuff", ST, 0.45), mu("forearm_extensors", ST, 0.45),
             mu("quadriceps", ST, 0.3)),
    equipment=(eq("mat", attach="static"),),
    errors=("Rolling the hips toward the working arm.",
            "Bending the far arm, which makes it a wide push-up.",
            "Hips sagging."),
    physio_notes=("Shifting the load onto one arm while the other only supports is the "
                  "standard route to a one-arm push-up without its wrist and shoulder "
                  "demands.",),
    sources=(CALATAYUD, NSCA, ACE), camera="front", tags=("bodyweight", "unilateral"),
)

# Piked, the body folds past a right angle at the hips and the head end points
# down: positive pitch when prone.  The straight arm is vertical at shoulder
# flexion 90 + pitch.  35 deg is the measured limit: with the legs straight the
# hip cannot go above ~100 units without the feet leaving the floor, because
# hip-to-toe is shorter than hand-to-hip through a vertical arm.
_PIKE_TOP_PITCH, _PIKE_BOTTOM_PITCH = 35.0, 58.0
_PIKE_TOP = merge(pose(hip_flex=100, knee_flex=5, ankle_flex=45),
                  arms(flex=90 + _PIKE_TOP_PITCH, abduct=12, elbow=0), flat_palm())
# Bending the elbows drops the shoulder 37 units (65.9 -> 29.2 measured), and
# the body hangs from the anchored hands, so the trunk has to steepen by the
# same amount or the feet go through the floor: 58 deg puts them back at 9.8.
_PIKE_BOTTOM = merge(pose(hip_flex=100, knee_flex=5, ankle_flex=45),
                     arms(flex=88, abduct=40, elbow=100), flat_palm())

pike_push_up = ExerciseDefinition(
    id="pike_push_up", name="Pike push-up", category=Category.CALISTHENICS,
    description="A push-up with the hips high, so the press is nearly vertical. The bodyweight "
                "step between a shoulder press and a handstand push-up.",
    setup=("Hands shoulder width, hips high, body in an inverted V",
           "The head travels to the floor between the hands",
           "Elbows about 45 deg out, not flared"),
    orientation="prone", anchor="hands", base_position=(-85.0, 30.0, 0.0),
    phases=(
        ph("Lower", ECC, 1.8, _PIKE_BOTTOM, pitch=_PIKE_BOTTOM_PITCH,
           cues=("Crown of the head toward the floor between the hands",)),
        ph("Bottom", ISO, 0.4, _PIKE_BOTTOM, pitch=_PIKE_BOTTOM_PITCH),
        ph("Press", CON, 1.4, _PIKE_TOP, pitch=_PIKE_TOP_PITCH,
           cues=("Press the floor away and push the hips back over the hands",)),
        ph("Top", ISO, 0.4, _PIKE_TOP, pitch=_PIKE_TOP_PITCH),
    ),
    muscles=(mu("deltoid_anterior", P, 0.9), mu("triceps_brachii", P, 0.8),
             mu("deltoid_lateral", S, 0.6),
             mu("serratus_anterior", P, 0.65, note="upward rotation under load"),
             mu("trapezius_upper", S, 0.55), mu("pectoralis_upper", S, 0.45),
             mu("rectus_abdominis", S, 0.5), mu("hamstrings", ST, 0.45),
             mu("rotator_cuff", ST, 0.45), mu("forearm_extensors", ST, 0.45)),
    equipment=(eq("mat", attach="static"),),
    errors=("Hips dropping, which turns it back into a push-up.",
            "The head landing in front of the hands rather than between them.",
            "Flaring the elbows to 90 deg."),
    physio_notes=("Raising the feet makes the press steadily more vertical and is the "
                  "standard route to a handstand push-up.",),
    sources=(CALATAYUD, NSCA, ACE, EXRX), camera="front", tags=("bodyweight",),
)

# ── Kneeling: the Nordic curl ──────────────────────────────────────────
# Kneeling upright is the prone body pitched 90 deg head-up about the knee.
# As the body lowers by phi the shins stay on the floor, so the knee angle is
# 90 - phi: the curl is resisted knee EXTENSION, not flexion.


#: Measured: at ``knee_flex = -pitch`` the shank still slopes 13 deg, leaving
#: the knee 25 units off the floor.  The offset levels it (knee 11.0, ankle
#: 14.2 -- a knee joint centre resting on a mat).
_SHANK_ON_FLOOR = 13.0


def _nordic(pitch: float) -> dict[str, float]:
    return pose(hip_flex=4, knee_flex=-pitch + _SHANK_ON_FLOOR, ankle_flex=-40)


nordic_hamstring_curl = ExerciseDefinition(
    id="nordic_hamstring_curl", name="Nordic hamstring curl", category=Category.CALISTHENICS,
    description="Kneeling with the ankles held, the body is lowered forward as slowly as the "
                "hamstrings can resist. The knees straighten as the body falls, so it is a "
                "near-maximal eccentric for the hamstrings at long muscle lengths.",
    setup=("Kneel with the ankles anchored and the hips straight",
           "Lower as slowly as possible; the body stays in one line from knees to head",
           "Catch with the hands and push back up"),
    orientation="prone", anchor="feet", base_position=(-85.0, 30.0, 0.0),
    phases=(
        ph("Lower", ECC, 3.5, merge(_nordic(-25.0), arms(flex=70, abduct=15, elbow=25)),
           pitch=-25, pivot=KNEE,
           cues=("Resist all the way; keep the hips straight, do not fold at the waist",
                 "Go slowly: the last few degrees are the point")),
        ph("Catch", TRN, 0.5, merge(_nordic(-15.0), arms(flex=95, abduct=25, elbow=60)),
           pitch=-15, pivot=KNEE, cues=("Catch on the hands when control runs out",)),
        ph("Return", CON, 1.6, merge(_nordic(-55.0), arms(flex=70, abduct=15, elbow=25)),
           pitch=-55, pivot=KNEE,
           cues=("Push off the hands and pull back up with the hamstrings",)),
        ph("Kneeling", ISO, 0.6, merge(_nordic(-88.0), arms(flex=25, abduct=10, elbow=15)),
           pitch=-88, pivot=KNEE, cues=("Tall kneeling, hips straight, glutes tight",)),
    ),
    muscles=(mu("hamstrings", P, 1.0, note="a near-maximal eccentric at long lengths"),
             mu("gluteus_maximus", P, 0.7, note="holds the hips straight"),
             mu("erector_spinae", S, 0.6), mu("gastrocnemius", S, 0.5),
             mu("rectus_abdominis", S, 0.5, note="stops the hips folding"),
             mu("adductors", S, 0.4),
             mu("triceps_brachii", ST, 0.4, note="the catch"),
             mu("pectoralis_major", ST, 0.35)),
    equipment=(eq("mat", attach="static"),),
    errors=("Folding at the hips, which shortens the lever and skips the work.",
            "Dropping rather than lowering.",
            "Adding volume too fast: it is notorious for soreness."),
    physio_notes=("Widely used in hamstring-injury prevention because it loads the muscle "
                  "eccentrically at long lengths, which is where hamstring strains happen.",),
    sources=(NSCA, ACE, EXRX), camera="front", default_reps=3,
    tags=("bodyweight", "eccentric", "rehab"),
)

# ── Supine: the hollow body hold ───────────────────────────────────────

_HOLLOW = merge(pose(hip_flex=22, knee_flex=5, spine_flex=18),
                arms(flex=170, abduct=8, elbow=5))
_HOLLOW_REST = merge(pose(hip_flex=3, knee_flex=5, spine_flex=2),
                     arms(flex=150, abduct=8, elbow=5))

hollow_body_hold = ExerciseDefinition(
    id="hollow_body_hold", name="Hollow body hold", category=Category.CALISTHENICS,
    description="Lying on the back with the shoulders and legs lifted and the low back pressed "
                "flat into the floor. The shape every other calisthenic hold is built on.",
    setup=("Press the low back into the floor before lifting anything",
           "Arms overhead, legs straight, both ends off the floor",
           "Bend the knees or bring the arms down to make it easier"),
    orientation="supine", anchor="none", base_position=(-85.0, 15.0, 0.0),
    phases=(
        ph("Lift", CON, 1.0, _HOLLOW, cues=("Ribs down, low back flat, both ends up",)),
        ph("Hold", ISO, 6.0, _HOLLOW,
           cues=("Do not let the low back arch off the floor",
                 "Breathe shallowly behind the brace")),
        ph("Release", ECC, 1.0, _HOLLOW_REST, cues=("Lower both ends together",)),
        ph("Rest", TRN, 0.8, _HOLLOW_REST),
    ),
    muscles=(mu("rectus_abdominis", P, 0.95), mu("hip_flexors", P, 0.8),
             mu("obliques", P, 0.7), mu("transversus_abdominis", S, 0.6),
             mu("quadriceps", S, 0.5, note="holds the knees straight"),
             mu("serratus_anterior", S, 0.45), mu("latissimus_dorsi", ST, 0.4),
             mu("deltoid_anterior", ST, 0.4)),
    equipment=(eq("mat", attach="static"),),
    errors=("Letting the low back arch, which is the one thing the hold is for.",
            "Holding the breath.",
            "Going to the full lever before the flat back can be kept."),
    physio_notes=("The flat low back is the criterion: if it lifts, the lever is too long, "
                  "and the fix is to shorten it by bending the knees or lowering the arms.",),
    sources=(MCGILL, EKSTROM, NSCA, ACE), camera="front", default_reps=1,
    tags=("bodyweight", "isometric", "core"),
)

# ── A bench behind: the bench dip ──────────────────────────────────────

_BDIP_TOP = merge(pose(hip_flex=63, knee_flex=10, ankle_flex=-5),
                  arms(flex=-20, abduct=8, rotate=-5, elbow=10, forearm=-40), grip(curl=45))
_BDIP_BOTTOM = merge(pose(hip_flex=84, knee_flex=10, ankle_flex=-5),
                     arms(flex=-40, abduct=12, rotate=-10, elbow=90, forearm=-40),
                     grip(curl=45))

bench_dip = ExerciseDefinition(
    id="bench_dip", name="Bench dip", category=Category.CALISTHENICS,
    description="Hands behind on a bench and the heels out in front, the body lowered by "
                "bending the elbows. The accessible triceps dip, at the cost of a more "
                "demanding shoulder position.",
    setup=("Hands on the bench edge behind the hips, fingers forward",
           "Heels on the floor, legs straight, hips close to the bench",
           "Lower until the upper arms are about parallel to the floor"),
    orientation="hanging", anchor="hands", anchor_point=(0.0, BENCH_TOP, 0.0),
    phases=(
        ph("Lower", ECC, 1.8, _BDIP_BOTTOM,
           cues=("Elbows straight back, not flared; keep the hips close to the bench",)),
        ph("Bottom", ISO, 0.4, _BDIP_BOTTOM, cues=("Upper arms about parallel; no deeper",)),
        ph("Press", CON, 1.4, _BDIP_TOP,
           cues=("Push through the heels of the hands to straight arms",)),
        ph("Top", ISO, 0.4, _BDIP_TOP),
    ),
    muscles=(mu("triceps_brachii", P, 0.95), mu("deltoid_anterior", P, 0.7),
             mu("pectoralis_major", S, 0.55), mu("pectoralis_minor", S, 0.4),
             mu("rhomboids", S, 0.4), mu("serratus_anterior", S, 0.4),
             mu("rectus_abdominis", ST, 0.45), mu("rotator_cuff", ST, 0.5),
             mu("forearm_extensors", ST, 0.5)),
    equipment=(eq("bench", attach="static", height=BENCH_TOP, length=60.0),),
    errors=("Going deep enough that the shoulders roll forward under the body.",
            "Letting the hips drift away from the bench, which loads the shoulder further.",
            "Choosing it over a parallel-bar dip when the shoulder is irritable."),
    physio_notes=("The hands-behind position holds the shoulder in extension and internal "
                  "rotation, the position most associated with anterior shoulder pain "
                  "(Kolber 2010); a parallel-bar dip is usually the kinder choice.",),
    sources=(KOLBER, NSCA, ACE, EXRX), camera="side", camera_target=(0.0, 70.0, 0.0),
    tags=("bodyweight",),
)

EXERCISES = (muscle_up, l_sit, pistol_squat, archer_push_up, pike_push_up,
             nordic_hamstring_curl, hollow_body_hold, bench_dip)
