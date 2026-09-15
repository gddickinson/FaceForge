"""Bench press variations: grip width, incline, decline and the floor press.

The flat barbell bench press and its reverse-grip form live in
:mod:`faceforge.exercise.catalog.upper_push`.  These are the five that change
one thing about it.

What each change does, from the literature the project already cites:

* **Grip width.**  A wider grip abducts the shoulder further, shortens the bar
  path and puts more of the work on pectoralis major; a narrower one tucks the
  elbow, lengthens the path and shifts work to triceps brachii (ANDERSEN).
  The standard grip here measures 110 units between the hands; close is 78 and
  wide 132, which is roughly 0.7x and 1.2x of it.
* **Incline.**  Clavicular pectoralis peaks near 30 deg and anterior deltoid
  keeps rising past it, so 45 deg and above is a shoulder press with a wide
  grip (BENCH_INCLINE).
* **Decline.**  Tips the line of pull below the sternum and takes the anterior
  deltoid out of it, leaving the sternal and abdominal heads of pectoralis
  major to do more of the work over a shorter range.
* **The floor.**  The upper arm lands before the chest does, so the bottom
  third is missing along with the stretch and the leg drive.  What is left is
  the lockout, which is triceps.
"""

from __future__ import annotations

from faceforge.exercise.catalog._helpers import (
    ANDERSEN, BENCH_INCLINE, BENCH_TOP, CALATAYUD, CON, ECC, ECC_CON, EXRX, HIPS, ISO, KOLBER,
    NSCA, P, S, ST, arms, bench_legs, eq, grip, incline_legs, merge, mu, ph, pose,
)
from faceforge.exercise.model import Category, ExerciseDefinition

_LEGS = bench_legs()
_INCLINE_LEGS = incline_legs()
_BENCH = eq("bench", attach="static", height=BENCH_TOP)

#: Shoulder abduction at the bottom decides the grip width: the standard press
#: uses 75 and measures 110 units between the hands.
_CLOSE, _STANDARD, _WIDE = 45.0, 75.0, 92.0


def _bench_pose(abduct_bottom: float, abduct_top: float, legs: dict,
                bottom: bool) -> dict:
    """One bench keyframe at a given grip width.

    The other angles are the flat press's, measured on the rig: the palms face
    the feet (shoulder rotation 30 with the forearm at 60 at the bottom), the
    wrist is extended 70 deg and stacked over the elbow.
    """
    if bottom:
        return merge(pose(), legs,
                     arms(flex=-15, abduct=abduct_bottom, rotate=30, elbow=80,
                          forearm=60, wrist=-70), grip())
    return merge(pose(), legs,
                 arms(flex=75, abduct=abduct_top, rotate=0, elbow=10,
                      forearm=0, wrist=-70), grip())


_COMMON_ERRORS = ("Bouncing the bar off the chest.",
                  "Feet lifting or hips leaving the bench.",
                  "Wrists bent back under the bar.")
_BACK = (mu("latissimus_dorsi", ST, 0.45,
            note="pulls the bar down under control and holds the shoulder tight"),
         mu("rhomboids", ST, 0.4, note="holds the blades retracted"),
         mu("trapezius_middle", ST, 0.35, note="holds the blades retracted"),
         mu("erector_spinae", ST, 0.3, note="braces the trunk against the bench"))
_SUPPORT = (mu("rotator_cuff", ST, 0.35), mu("biceps_brachii", ST, 0.2),
            mu("forearm_flexors", ST, 0.45),
            mu("gluteus_maximus", ST, 0.3, note="leg drive"),
            mu("quadriceps", ST, 0.25))

close_grip_bench_press = ExerciseDefinition(
    id="close_grip_bench_press", name="Close-grip bench press",
    category=Category.UPPER_PUSH,
    description="A bench press taken at about shoulder width with the elbows tucked, which "
                "lengthens the bar path and shifts the work to the triceps.",
    setup=("Grip at about shoulder width, not narrower: a very close grip loads the wrists",
           "Elbows tucked to roughly 30 deg from the trunk",
           "Five points of contact; shoulder blades back and down"),
    orientation="supine", anchor="none", base_position=(-85.0, BENCH_TOP + 15.0, 0.0),
    phases=(
        ph("Lower", ECC, 2.0, _bench_pose(_CLOSE, 8, _LEGS, True),
           cues=("Bar to the lower sternum with the elbows close to the ribs",
                 "Forearms vertical")),
        ph("Touch", ISO, 0.3, _bench_pose(_CLOSE, 8, _LEGS, True),
           cues=("Light touch; stay tight",)),
        ph("Press", CON, 1.5, _bench_pose(_CLOSE, 8, _LEGS, False),
           cues=("Drive the bar straight up; finish by straightening the elbows",)),
        ph("Lockout", ISO, 0.5, _bench_pose(_CLOSE, 8, _LEGS, False),
           cues=("Elbows straight, shoulder blades down",)),
    ),
    muscles=(mu("triceps_brachii", P, 0.9, note="the point of the narrow grip"),
             mu("pectoralis_major", P, 0.7), mu("deltoid_anterior", S, 0.6),
             mu("pectoralis_upper", S, 0.45), mu("serratus_anterior", S, 0.35),
             *_BACK, *_SUPPORT),
    equipment=(eq("barbell", plates=2), _BENCH),
    errors=("A grip inside shoulder width, which loads the wrist rather than the triceps.",
            "Letting the elbows drift out, which gives up the triceps emphasis.",
            *_COMMON_ERRORS),
    physio_notes=("Narrowing the grip lengthens the bar path and increases elbow flexion at "
                  "the chest, so the triceps have more work to do and the pectorals less.",
                  "It is the bench variation that tends to sit easiest on the anterior "
                  "shoulder, because the tucked elbow reduces the abduction angle."),
    sources=(ANDERSEN, BENCH_INCLINE, CALATAYUD, NSCA, ECC_CON),
    camera="front", tags=("barbell",),
)

wide_grip_bench_press = ExerciseDefinition(
    id="wide_grip_bench_press", name="Wide-grip bench press",
    category=Category.UPPER_PUSH,
    description="A bench press taken well outside shoulder width, which shortens the bar "
                "path and puts more of the work on pectoralis major.",
    setup=("Grip roughly 1.5 to 2x biacromial width", "Elbows about 60 to 75 deg from the trunk",
           "Five points of contact; shoulder blades back and down"),
    orientation="supine", anchor="none", base_position=(-85.0, BENCH_TOP + 15.0, 0.0),
    phases=(
        ph("Lower", ECC, 2.0, _bench_pose(_WIDE, 18, _LEGS, True),
           cues=("Bar to the mid sternum; the path is shorter than a standard grip's",)),
        ph("Touch", ISO, 0.3, _bench_pose(_WIDE, 18, _LEGS, True),
           cues=("Light touch; do not let the shoulders roll forward",)),
        ph("Press", CON, 1.5, _bench_pose(_WIDE, 18, _LEGS, False),
           cues=("Press up and slightly back over the shoulders",)),
        ph("Lockout", ISO, 0.5, _bench_pose(_WIDE, 18, _LEGS, False),
           cues=("Elbows straight, shoulder blades down",)),
    ),
    muscles=(mu("pectoralis_major", P, 0.95, note="the widest grip's emphasis"),
             mu("deltoid_anterior", S, 0.65), mu("triceps_brachii", S, 0.55),
             mu("pectoralis_upper", S, 0.4), mu("serratus_anterior", S, 0.4),
             *_BACK, *_SUPPORT),
    equipment=(eq("barbell", plates=2), _BENCH),
    errors=("Going wider still to shorten the range: the gain is small and the "
            "anterior shoulder takes the difference.",
            "Flaring the elbows to 90 deg at the chest.",
            *_COMMON_ERRORS),
    physio_notes=("A wider grip abducts the shoulder further and shortens the bar path, "
                  "which raises pectoral demand and lowers triceps demand.",
                  "It is also the grip with the most anterior shoulder stress, so it "
                  "suits a lifter with no impingement history and a good warm-up."),
    sources=(ANDERSEN, BENCH_INCLINE, KOLBER, NSCA, ECC_CON),
    camera="front", tags=("barbell",),
)

incline_barbell_bench_press = ExerciseDefinition(
    id="incline_barbell_bench_press", name="Incline barbell bench press",
    category=Category.UPPER_PUSH,
    description="A barbell press on a 30 deg incline, which moves the line of pull onto the "
                "clavicular head of pectoralis major.",
    setup=("Bench at about 30 deg: past 45 it becomes a shoulder press",
           "Grip a little inside a flat-bench grip", "Feet flat, blades back and down"),
    orientation="supine", anchor="none", base_position=(-85.0, BENCH_TOP + 15.0, 0.0),
    phases=(
        ph("Lower", ECC, 2.0, _bench_pose(70, 14, _INCLINE_LEGS, True), pitch=30, pivot=HIPS,
           cues=("Bar to the upper chest, just below the collarbones",)),
        ph("Touch", ISO, 0.3, _bench_pose(70, 14, _INCLINE_LEGS, True), pitch=30, pivot=HIPS),
        ph("Press", CON, 1.5, _bench_pose(70, 14, _INCLINE_LEGS, False), pitch=30, pivot=HIPS,
           cues=("Press up and slightly back over the shoulders",)),
        ph("Lockout", ISO, 0.5, _bench_pose(70, 14, _INCLINE_LEGS, False), pitch=30, pivot=HIPS),
    ),
    muscles=(mu("pectoralis_upper", P, 0.9, note="peaks near 30 deg of incline"),
             mu("deltoid_anterior", P, 0.8), mu("triceps_brachii", S, 0.6),
             mu("pectoralis_major", S, 0.5), mu("serratus_anterior", S, 0.4),
             *_BACK, *_SUPPORT),
    equipment=(eq("barbell", plates=2),
               eq("bench", attach="static", height=BENCH_TOP, incline_deg=-30.0)),
    errors=("A bench steeper than 45 deg, which hands the lift to the anterior deltoid.",
            "Sliding down the bench as the set goes on.",
            *_COMMON_ERRORS),
    physio_notes=("Clavicular pectoralis EMG peaks around 30 deg; anterior deltoid keeps "
                  "rising to 60, which is why steeper is not better for the chest.",),
    sources=(BENCH_INCLINE, CALATAYUD, NSCA, ECC_CON),
    camera="front", tags=("barbell",),
)

decline_barbell_bench_press = ExerciseDefinition(
    id="decline_barbell_bench_press", name="Decline barbell bench press",
    category=Category.UPPER_PUSH,
    description="A barbell press on a 20 deg decline, which drops the line of pull below the "
                "sternum and takes the anterior deltoid out of it.",
    setup=("Bench declined about 20 deg with the feet hooked",
           "Bar lowered to the lower sternum", "Blades back and down; do not let the head hang"),
    orientation="supine", anchor="none", base_position=(-85.0, BENCH_TOP + 15.0, 0.0),
    phases=(
        ph("Lower", ECC, 2.0, _bench_pose(_STANDARD, 12, _LEGS, True), pitch=-20, pivot=HIPS,
           cues=("Bar to the lower sternum; the range is shorter than a flat bench's",)),
        ph("Touch", ISO, 0.3, _bench_pose(_STANDARD, 12, _LEGS, True), pitch=-20, pivot=HIPS),
        ph("Press", CON, 1.5, _bench_pose(_STANDARD, 12, _LEGS, False), pitch=-20, pivot=HIPS,
           cues=("Press straight up from the lower chest",)),
        ph("Lockout", ISO, 0.5, _bench_pose(_STANDARD, 12, _LEGS, False), pitch=-20, pivot=HIPS),
    ),
    muscles=(mu("pectoralis_major", P, 0.9, note="sternal and abdominal heads"),
             mu("triceps_brachii", P, 0.7), mu("deltoid_anterior", S, 0.45),
             mu("serratus_anterior", S, 0.35), mu("pectoralis_upper", S, 0.25),
             *_BACK, *_SUPPORT),
    equipment=(eq("barbell", plates=2),
               eq("bench", attach="static", height=BENCH_TOP, incline_deg=20.0)),
    errors=("Steep declines with a heavy bar and no spotter: the bar finishes over the throat.",
            "Pushing the head into the pad.",
            *_COMMON_ERRORS),
    physio_notes=("The decline lowers the shoulder flexion angle, so the anterior deltoid "
                  "contributes less and the sternal fibres of pectoralis major more.",
                  "The range is shorter than a flat bench's, which is part of why loads "
                  "tend to be higher."),
    sources=(BENCH_INCLINE, CALATAYUD, EXRX, NSCA, ECC_CON),
    camera="front", tags=("barbell",),
)

floor_press = ExerciseDefinition(
    id="floor_press", name="Floor press", category=Category.UPPER_PUSH,
    description="A barbell press lying on the floor. The upper arm lands before the chest "
                "does, which removes the bottom third of the range, the stretch and the leg "
                "drive, and leaves the lockout.",
    setup=("Flat on the floor, knees bent or legs straight", "Grip as for a flat bench press",
           "Upper arms rest on the floor at the bottom; the pause is real, not a bounce"),
    orientation="supine", anchor="none", base_position=(-85.0, 14.0, 0.0),
    phases=(
        ph("Lower", ECC, 2.0,
           merge(pose(), arms(flex=-5, abduct=60, rotate=30, elbow=55, forearm=60, wrist=-70),
                 grip()),
           cues=("Lower until the triceps touch the floor",)),
        ph("Pause", ISO, 0.6,
           merge(pose(), arms(flex=-5, abduct=60, rotate=30, elbow=55, forearm=60, wrist=-70),
                 grip()),
           cues=("Rest the upper arms without relaxing the grip or the back",)),
        ph("Press", CON, 1.4,
           merge(pose(), arms(flex=75, abduct=12, rotate=0, elbow=10, forearm=0, wrist=-70),
                 grip()),
           cues=("Drive from a dead stop: there is no stretch to use",)),
        ph("Lockout", ISO, 0.5,
           merge(pose(), arms(flex=75, abduct=12, rotate=0, elbow=10, forearm=0, wrist=-70),
                 grip())),
    ),
    muscles=(mu("triceps_brachii", P, 0.85, note="the lockout is what is left"),
             mu("pectoralis_major", P, 0.7), mu("deltoid_anterior", S, 0.55),
             mu("serratus_anterior", S, 0.3),
             mu("latissimus_dorsi", ST, 0.4), mu("rhomboids", ST, 0.35),
             mu("trapezius_middle", ST, 0.3), mu("rotator_cuff", ST, 0.35),
             mu("forearm_flexors", ST, 0.45), mu("biceps_brachii", ST, 0.2)),
    equipment=(eq("barbell", plates=2), eq("mat", attach="static")),
    errors=("Bouncing the elbows off the floor.",
            "Treating it as a bench press with a shorter range and the same load progression."),
    physio_notes=("Removing the bottom third removes the stretch-shortening contribution, "
                  "so the press starts from a dead stop and the triceps carry more of it.",
                  "It is a common choice when a lifter's shoulder does not tolerate the "
                  "bottom of a full-range bench press."),
    sources=(CALATAYUD, KOLBER, NSCA, EXRX, ECC_CON),
    camera="front", tags=("barbell",),
)

EXERCISES = (close_grip_bench_press, wide_grip_bench_press, incline_barbell_bench_press,
             decline_barbell_bench_press, floor_press)
