"""The squats.

The hip hinges are in :mod:`faceforge.exercise.catalog.hinges` and the
split-stance work in ``lunges``.

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
# Measured on the rig (2026-09-11, closed-finger ring centres): the front
# rack puts the hands 45 apart, 6 forward of and 20 above the shoulder
# pivots with the elbows level and 40 forward -- the bar is then hung 13
# below the hands so it rests on the front deltoids.  The goblet hold puts
# the hands 14 apart, 19 forward of and 19 above the mid-sternum; the bell
# (its origin 4 above the hands) hangs against the chest without entering it.
_FRONT_RACK = combine(arms(flex=65, abduct=0, rotate=-60, elbow=145), grip())
_GOBLET = combine(arms(flex=35, abduct=10, rotate=-58, elbow=100, forearm=60), grip())

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
    equipment=(eq("kettlebell", radius=10.0, hang=-4.0),),
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
    equipment=(eq("barbell", plates=1, hang=13.0),),
    errors=("Elbows dropping so the bar rolls forward: keep the upper arms parallel to the "
            "floor.", "Wrists hyperextended under load: open the grip or use straps."),
    physio_notes=("Lower spinal compressive and shear loads than the back squat at the same "
                  "bar load (Gullett 2009).",),
    sources=(SQUAT_REVIEW, SCHOENFELD, NSCA), tags=("barbell",),
)














box_squat = ExerciseDefinition(
    id="box_squat", name="Box squat", category=Category.LOWER_BODY,
    description="A back squat to a box, sitting back onto it and pausing before standing. The "
                "box fixes the depth and the pause removes the bounce.",
    setup=("Box set so the crease of the hip finishes at or just below the knee",
           "Sit back rather than down: shins stay near vertical",
           "Pause on the box without relaxing the trunk, then drive up"),
    phases=(
        ph("Sit back", ECC, 2.2, merge(squat(105, 95, 38)[0], _BAR_ON_BACK), pitch=38,
           cues=("Push the hips back to the box; keep the shins vertical",)),
        ph("Pause", ISO, 0.8, merge(squat(105, 95, 38)[0], _BAR_ON_BACK), pitch=38,
           cues=("Sit, do not slump: stay braced and keep the weight on the feet",)),
        ph("Drive", CON, 1.4, merge(stand()[0], _BAR_ON_BACK), pitch=0.0,
           cues=("Chest and hips rise together off the box",)),
        ph("Lockout", ISO, 0.4, merge(stand()[0], _BAR_ON_BACK), pitch=0.0),
    ),
    muscles=_SQUAT_MUSCLES + (mu("erector_spinae", P, 0.75,
                                 note="holds the trunk through the dead stop"),),
    equipment=(eq("barbell", plates=2), eq("plyo_box", attach="static", height=45.0)),
    errors=("Rocking backwards on the box and losing the brace.",
            "Dropping onto it rather than sitting back under control.",
            "Letting the shins travel forward, which turns it into an ordinary squat."),
    physio_notes=("Sitting back with vertical shins increases the hip contribution and "
                  "reduces the knee's, which is why it is a common posterior-chain squat "
                  "variant.",
                  "The pause removes the stretch-shortening contribution, so the drive "
                  "starts from a dead stop."),
    sources=(SQUAT_KIN, SQUAT_REVIEW, NSCA, EXRX), tags=("barbell", "posterior-chain"),
)


pause_squat = ExerciseDefinition(
    id="pause_squat", name="Pause squat", category=Category.LOWER_BODY,
    description="A back squat held for two to three seconds at the bottom. The pause kills "
                "the bounce and exposes any position that was being hidden by speed.",
    setup=("Set up as for a back squat", "Descend under control to full depth",
           "Hold without relaxing, then drive: the brace must not change"),
    phases=(
        ph("Descent", ECC, 2.2, merge(squat(120, 125, 32)[0], _BAR_ON_BACK), pitch=32,
           cues=("Controlled descent to depth",)),
        ph("Pause", ISO, 2.5, merge(squat(120, 125, 32)[0], _BAR_ON_BACK), pitch=32,
           cues=("Stay tight: knees out, chest up, weight over mid-foot",
                 "Do not relax into the bottom")),
        ph("Ascent", CON, 1.6, merge(stand()[0], _BAR_ON_BACK), pitch=0.0,
           cues=("Drive from a dead stop; hips and chest together",)),
        ph("Lockout", ISO, 0.4, merge(stand()[0], _BAR_ON_BACK), pitch=0.0),
    ),
    muscles=_SQUAT_MUSCLES + (mu("erector_spinae", P, 0.8, note="the pause is its test"),),
    equipment=(eq("barbell", plates=2),),
    errors=("Relaxing at the bottom and re-bracing to stand.",
            "Letting the chest drop through the pause.",
            "Using a load that only works with a bounce."),
    physio_notes=("Removing the stretch-shortening contribution lowers the load that can be "
                  "moved and raises the demand on the position, which is why it is used as "
                  "a technique lift rather than a maximal one.",),
    sources=(SQUAT_KIN, SCHOENFELD, NSCA), tags=("barbell", "technique"),
)

EXERCISES = (bodyweight_squat, barbell_back_squat, goblet_squat, front_squat,
             box_squat, pause_squat)
