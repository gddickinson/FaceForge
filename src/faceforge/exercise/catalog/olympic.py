"""The Olympic lifts and the movements they are built from.

Two lifts are contested -- the snatch and the clean and jerk -- and everything
else here is a piece of one of them.  The catalogue already had the power
clean; what it did not have was any snatch at all, any jerk, the overhead
squat that the snatch catches in, the hang variants that teach the second
pull, or the pulls that train it without the catch.

Three things shape every one of them and are worth stating once:

* The bar travels in a straight vertical line over the mid-foot.  Every cue
  about "keeping it close" exists to stop it looping away and forward.
* **Triple extension** -- hips, knees and ankles at once -- is the engine, and
  it is the same pattern as a vertical jump.  It is why these are taught to
  athletes who never compete in weightlifting.
* The lifter does not lift the bar to the finish position; they pull it as
  high as they can and then *drop underneath it*.  The catch is a fast
  eccentric, which is why the phases below are marked that way.

Grip width is the clearest difference between the two: a clean is gripped just
outside the knees, a snatch wide enough that the bar sits in the hip crease at
full extension, which in this rig is the shoulder abduction each catch pose
uses rather than a number in the equipment spec.
"""

from __future__ import annotations

from faceforge.exercise.catalog._helpers import (
    ACE, CON, ECC, EXRX, ISO, NSCA, P, S, ST, TRN, arms, eq, grip, merge, mu, ph, pose, squat, toes_on_floor
)
from faceforge.exercise.model import Category, ExerciseDefinition

#: The rear foot of a split: heel up, on the ball, so its toes have to bend
#: back through the same angle the forefoot has tipped by.  Without it the
#: foot is one rigid wedge and the lifter lands on the ends of his toes --
#: measured 2026-09-17, the toe tip sat 5.8 below the ball of the same foot.
_SPLIT_BACK = dict(hip_l_flex=-22, knee_l_flex=45, ankle_l_flex=-10)
_SPLIT_BACK_TOES = toes_on_floor(4, -22, 45, -10)


#: The pull, shared by both lifts: off the floor, past the knee, full extension.
#: Trunk angles are the measured foot-flat ones (`squat()` applies the rule).
# Depth measured 2026-09-16 against the lowest point of the loaded bar at the
# start.  `hip` was the lever, not `knee`: at hip 110 the bar bottomed out at
# 16.6 whatever the knee did; hip 125 with a deep knee puts it near the floor.
# The knee stops at 118 because the flat-foot rule ties the ankle to
# pitch - hip + knee and `ankle_flex` runs to 45 deg: past 120 here the
# heels would have to come up and the solver would clamp the pose.
# The clean grip has to match the RACK, because the bar is rigid and the rack
# is the position with the least freedom: measured, a front rack cannot open
# wider than ~83 whatever the abduction (more abduction *narrows* it), while
# the pull at abduct 10 was 106.7 -- so each hand slid 15 units a rep.  At
# abduct 0 the pull is 80.8, within 2 of the rack's 78.7, and the bar also
# reaches the floor (low 1.4, against 6.2 at abduct 10).
_START_C = merge(squat(125, 118, 50)[0], arms(flex=50, abduct=0), grip())
_KNEE_C = merge(squat(65, 30, 40)[0], arms(flex=40, abduct=0), grip())
_EXT_C = merge(pose(knee_flex=5, ankle_flex=-30, toe_curl=toes_on_floor(0, 0, 5, -30)), arms(flex=25, abduct=4, elbow=80), grip())

#: The snatch grip is wide, so the same positions carry more abduction and the
#: bar finishes overhead rather than on the shoulders.  Measured 2026-09-16:
#: abduct 32 gave a 155.8-unit grip -- 123 cm, wider than the bar is useful --
#: and held the bar 51.6 above the floor at the start.  A real snatch grip is
#: 81-90 cm, which on this figure is 103-114 units.  Grip width by abduction:
#:     start/knee   ab 32 -> 155.8   24 -> 139.5   16 -> 121.3   12 -> 111.7
#:     extension    ab 55 -> 126.1   40 -> 120.1   30 -> 113.2  (elbow 70)
#: 12 on the pull and 30 at the extension hold it at 111.7-113.2 throughout,
#: so the hands stop sliding 15 units a rep along a rigid bar as well.
_START_S = merge(squat(125, 118, 48)[0], arms(flex=45, abduct=12), grip())
_KNEE_S = merge(squat(62, 28, 38)[0], arms(flex=38, abduct=12), grip())
_EXT_S = merge(pose(knee_flex=5, ankle_flex=-30, toe_curl=toes_on_floor(0, 0, 5, -30)), arms(flex=20, abduct=30, elbow=70), grip())
_OVERHEAD_S = arms(flex=172, abduct=20, rotate=25, elbow=5, forearm=-40)

_RACK = arms(flex=90, abduct=15, elbow=145)
#: The bar cannot go straight past the face, because the model's head cannot
#: move out of its way: `head_yaw/pitch/roll` exist on `BodyState` but are not
#: pose DOFs, so the catalogue cannot tip the head back the way a lifter does.
#: Traced frame by frame, the authored path took the bar from the rack (z -17)
#: to overhead (z -4) straight through the skull sphere -- 9.0 units inside it
#: at the worst frame, in both directions.  Every arm pose that holds the bar
#: AT head height is inside the skull (the best is -5.1); the bar has to be
#: past it before it comes back to the midline.  This is that waypoint: 21
#: below the head and 20 in front of it, clear by 13.5.  It reads as pushing
#: the bar out and up rather than straight up, which is a technical fault in a
#: real jerk -- and the lesser of the two.
_JERK_PAST_THE_FACE = arms(flex=150, abduct=20, elbow=120)

_SOURCES = (NSCA, EXRX, ACE)
_PULL_MUSCLES = (
    mu("gluteus_maximus", P, 1.0), mu("quadriceps", P, 0.95), mu("hamstrings", P, 0.85),
    mu("erector_spinae", P, 0.85), mu("gastrocnemius", S, 0.7), mu("soleus", S, 0.6),
    mu("trapezius_upper", S, 0.8, note="the shrug"), mu("trapezius_middle", S, 0.5),
    mu("deltoid_anterior", S, 0.5), mu("deltoid_lateral", S, 0.45),
    mu("forearm_flexors", S, 0.7), mu("rectus_abdominis", S, 0.45),
    mu("obliques", S, 0.4), mu("latissimus_dorsi", ST, 0.45),
)

# ── The snatch ─────────────────────────────────────────────────────────

squat_snatch = ExerciseDefinition(
    id="squat_snatch", name="Snatch", category=Category.ATHLETIC,
    description="The bar goes from the floor to arms' length overhead in one movement, caught "
                "in a full overhead squat. The most technical lift in the sport: the widest "
                "grip, the longest bar path, and a catch that asks for overhead mobility in "
                "the bottom of a squat.",
    setup=("Wide grip -- at full extension the bar should meet the hip crease",
           "Bar over the mid-foot, shoulders just in front of it, back flat",
           "Pull to full extension, then pull yourself under and catch overhead"),
    phases=(
        ph("First pull", CON, 0.6, _KNEE_S, pitch=38,
           cues=("Push the floor away; the back angle does not change",
                 "Bar brushes up the thighs")),
        ph("Second pull", CON, 0.22, _EXT_S, pitch=0, easing="ease_out",
           cues=("Hips through the bar, shrug, onto the toes",
                 "This is a jump, not a lift")),
        ph("Catch overhead", ECC, 0.3, merge(squat(120, 125, 22)[0], _OVERHEAD_S, grip()),
           pitch=22, easing="ease_in",
           cues=("Punch under it: the bar goes up, you go down",
                 "Catch with locked elbows, bar over the mid-foot")),
        ph("Stand up", CON, 1.0, merge(pose(knee_flex=5), _OVERHEAD_S, grip()),
           cues=("Stand out of the squat with the bar still locked overhead",)),
        ph("Lower to the floor", TRN, 1.4, _START_S, pitch=48,
           cues=("Bring it to the hips, then hinge and set it down",)),
    ),
    muscles=(*_PULL_MUSCLES, mu("triceps_brachii", S, 0.6, note="locks the catch"),
             mu("rotator_cuff", P, 0.7, note="holds a loaded overhead position"),
             mu("serratus_anterior", S, 0.6), mu("trapezius_lower", S, 0.55),
             mu("adductors", S, 0.5, note="the bottom of the squat")),
    equipment=(eq("barbell", plates=2),),
    errors=("Pulling early with the arms, which stops the hips finishing.",
            "Letting the bar loop away from the body.",
            "Pressing it out instead of catching it locked.",
            "Attempting it without the overhead squat position."),
    physio_notes=("The catch is the limiting position for most people: it needs shoulder "
                  "flexion with external rotation at the bottom of a deep squat, and that "
                  "combination is what the overhead squat is drilled for.",),
    sources=_SOURCES, camera="three_quarter", default_reps=3,
    tags=("barbell", "power", "olympic"),
)

power_snatch = ExerciseDefinition(
    id="power_snatch", name="Power snatch", category=Category.ATHLETIC,
    description="The same pull as a snatch, caught overhead above parallel instead of in a "
                "full squat. Less mobility, a higher pull, and the bar path that teaches the "
                "second pull.",
    setup=("Snatch grip, bar over the mid-foot",
           "Pull to full extension; catch with the hips above parallel",
           "If you have to squat to catch it, it was not pulled high enough"),
    phases=(
        ph("First pull", CON, 0.6, _KNEE_S, pitch=38,
           cues=("Back angle constant; bar close",)),
        ph("Second pull", CON, 0.22, _EXT_S, pitch=0, easing="ease_out",
           cues=("Triple extension: hips, knees, ankles",)),
        ph("Catch high", ECC, 0.26, merge(squat(62, 60, 12)[0], _OVERHEAD_S, grip()),
           pitch=12, easing="ease_in",
           cues=("Punch under and catch it locked, thighs above parallel",)),
        ph("Recover", CON, 0.6, merge(pose(knee_flex=5), _OVERHEAD_S, grip()),
           cues=("Stand tall, bar stacked over the mid-foot",)),
        ph("Lower to the floor", TRN, 1.3, _START_S, pitch=48,
           cues=("Down the front of the body to the hips, then hinge",)),
    ),
    muscles=(*_PULL_MUSCLES, mu("triceps_brachii", S, 0.6),
             mu("rotator_cuff", P, 0.65), mu("serratus_anterior", S, 0.55),
             mu("trapezius_lower", S, 0.5)),
    equipment=(eq("barbell", plates=2),),
    errors=("Catching low because the pull stopped early.",
            "Bending the arms before the hips finish.",
            "Soft elbows at the catch."),
    physio_notes=("Used in preference to the full snatch where overhead or squat mobility "
                  "limits the catch, and as a power measure in its own right.",),
    sources=_SOURCES, camera="three_quarter", default_reps=3,
    tags=("barbell", "power", "olympic"),
)

hang_power_clean = ExerciseDefinition(
    id="hang_power_clean", name="Hang power clean", category=Category.ATHLETIC,
    description="A clean started from the hang at mid-thigh rather than the floor. Removing "
                "the first pull leaves the part that makes the lift: the hips.",
    setup=("Stand with the bar at mid-thigh, clean grip, shoulders slightly in front",
           "Hinge to just above the knee, then drive",
           "Elbows whip through fast to the front rack"),
    phases=(
        ph("Hang", ISO, 0.4, merge(squat(40, 20, 28)[0], arms(flex=30, abduct=10), grip()),
           pitch=28, cues=("Bar at mid-thigh, lats tight, weight over the mid-foot",)),
        ph("Dip to the knee", ECC, 0.5, _KNEE_C, pitch=40,
           cues=("Hinge back to just above the knee; shins stay vertical",)),
        ph("Second pull", CON, 0.22, _EXT_C, pitch=0, easing="ease_out",
           cues=("Violent hip extension; shrug; stay over the bar as long as you can",)),
        ph("Catch", ECC, 0.28, merge(squat(62, 60, 10)[0], _RACK, grip()), pitch=10,
           easing="ease_in", cues=("Elbows around and up; catch in the front rack",)),
        ph("Stand", CON, 0.6, merge(pose(knee_flex=3), _RACK, grip()),
           cues=("Stand tall with the bar racked on the shoulders",)),
    ),
    muscles=(*_PULL_MUSCLES, mu("biceps_brachii", S, 0.4, note="the turnover, not the pull"),
             mu("deltoid_posterior", S, 0.4)),
    equipment=(eq("barbell", plates=2),),
    errors=("Curling the bar up instead of pulling with the hips.",
            "Dipping the knees forward rather than hinging back.",
            "Catching with low elbows so the wrists take the load."),
    physio_notes=("The hang position removes the slowest part of the lift and makes the "
                  "hip extension the whole exercise, which is why it is the common teaching "
                  "and team-sport variant.",),
    sources=_SOURCES, camera="three_quarter", default_reps=3,
    tags=("barbell", "power", "olympic"),
)

# ── The jerk ───────────────────────────────────────────────────────────

_JERK_OVERHEAD = arms(flex=172, abduct=12, rotate=12, elbow=5, forearm=-30)

push_jerk = ExerciseDefinition(
    id="push_jerk", name="Push jerk", category=Category.ATHLETIC,
    description="From the front rack, a dip and drive sends the bar off the shoulders and the "
                "lifter drops under it into a quarter squat to catch it locked overhead. More "
                "than a push press: the feet stay in place but the body moves down to meet "
                "the bar.",
    setup=("Bar in the front rack, elbows up, feet under the hips",
           "Dip a hand's width with the trunk vertical, then drive",
           "Punch the head through and receive it with the knees bent"),
    phases=(
        ph("Dip", ECC, 0.35, merge(squat(20, 25, 4)[0], _RACK, grip()), pitch=4,
           easing="ease_in", cues=("Short, vertical, heels down",)),
        ph("Drive", CON, 0.2, merge(pose(knee_flex=4, ankle_flex=-20),
                                    _JERK_PAST_THE_FACE, grip()),
           easing="ease_out", cues=("Extend hard; the bar leaves the shoulders on the legs",)),
        ph("Drop under", ECC, 0.22, merge(squat(55, 55, 8)[0], _JERK_OVERHEAD, grip()),
           pitch=8, easing="ease_in",
           cues=("Punch under it: arms lock as the hips drop",)),
        ph("Stand", CON, 0.6, merge(pose(knee_flex=4), _JERK_OVERHEAD, grip()),
           cues=("Stand up with the bar over the mid-foot, ribs down",)),
        ph("Lower it past the face", TRN, 0.3,
           merge(pose(knee_flex=4), _JERK_PAST_THE_FACE, grip()),
           cues=("Bend the arms and bring it down in front of the face",)),
        ph("Return to the rack", ECC, 0.4, merge(squat(18, 22, 4)[0], _RACK, grip()), pitch=4,
           cues=("Absorb it back onto the shoulders with the legs",)),
    ),
    muscles=(mu("deltoid_anterior", P, 0.9), mu("triceps_brachii", P, 0.85),
             mu("quadriceps", P, 0.85, note="the dip and drive"),
             mu("gluteus_maximus", P, 0.75), mu("gastrocnemius", S, 0.6),
             mu("deltoid_lateral", S, 0.6), mu("trapezius_upper", S, 0.6),
             mu("serratus_anterior", S, 0.55), mu("erector_spinae", P, 0.7,
                                                  note="holds the trunk vertical"),
             mu("rectus_abdominis", S, 0.5), mu("obliques", S, 0.45),
             mu("rotator_cuff", ST, 0.5), mu("forearm_flexors", ST, 0.5)),
    equipment=(eq("barbell", plates=2),),
    errors=("Dipping forward, which sends the bar out in front.",
            "Pressing it out instead of dropping under it.",
            "Catching with the head still behind the bar."),
    physio_notes=("The dip is a few inches and fast: a deep dip turns it into a squat and "
                  "loses the elastic contribution the lift depends on.",),
    sources=_SOURCES, camera="three_quarter", default_reps=3,
    tags=("barbell", "power", "olympic"),
)

split_jerk = ExerciseDefinition(
    id="split_jerk", name="Split jerk", category=Category.ATHLETIC,
    description="The competition jerk: the same dip and drive, received with one foot forward "
                "and one back. The split gives a longer base and a lower catch than the feet "
                "can reach standing.",
    setup=("Front rack, feet under the hips", "Dip and drive vertically",
           "Split: front shin vertical, back knee soft, torso between the feet"),
    phases=(
        ph("Dip", ECC, 0.35, merge(squat(20, 25, 4)[0], _RACK, grip()), pitch=4,
           easing="ease_in", cues=("Vertical dip, heels down, elbows up",)),
        ph("Drive", CON, 0.2, merge(pose(knee_flex=4, ankle_flex=-20),
                                    _JERK_PAST_THE_FACE, grip()),
           easing="ease_out", cues=("Drive through the whole foot; bar straight up",)),
        ph("Split under", ECC, 0.25,
           merge(pose(hip_r_flex=55, knee_r_flex=60, ankle_r_flex=15,
                      **_SPLIT_BACK, toe_curl_l=_SPLIT_BACK_TOES),
                 _JERK_OVERHEAD, grip()), pitch=4, easing="ease_in",
           cues=("Feet move as the arms lock; land both at once",
                 "Front shin vertical, back knee bent and soft")),
        ph("Recover", CON, 0.8, merge(pose(knee_flex=4), _JERK_OVERHEAD, grip()),
           cues=("Front foot back first, then the back foot, bar still locked",)),
        ph("Lower it past the face", TRN, 0.3,
           merge(pose(knee_flex=4), _JERK_PAST_THE_FACE, grip()),
           cues=("Bend the arms and bring it down in front of the face",)),
        ph("Return to the rack", ECC, 0.4, merge(squat(18, 22, 4)[0], _RACK, grip()), pitch=4,
           cues=("Lower it to the shoulders and absorb with the legs",)),
    ),
    muscles=(mu("deltoid_anterior", P, 0.9), mu("triceps_brachii", P, 0.85),
             mu("quadriceps", P, 0.85), mu("gluteus_maximus", P, 0.8),
             mu("gluteus_medius", P, 0.7, note="the split is a single-leg landing twice over"),
             mu("gastrocnemius", S, 0.6), mu("hip_flexors", S, 0.5, note="the back leg"),
             mu("adductors", S, 0.5), mu("deltoid_lateral", S, 0.55),
             mu("trapezius_upper", S, 0.6), mu("erector_spinae", P, 0.7),
             mu("rectus_abdominis", S, 0.5), mu("obliques", S, 0.5),
             mu("rotator_cuff", ST, 0.5), mu("forearm_flexors", ST, 0.5)),
    equipment=(eq("barbell", plates=2),),
    errors=("Splitting too short, which leaves no room to drop.",
            "Front knee travelling past the toes on the catch.",
            "Landing front foot first instead of both together."),
    physio_notes=("The split lands the whole load on a single-leg base twice a rep, which "
                  "is why gluteus medius and the deep rotators matter as much as the "
                  "shoulders here.",),
    sources=_SOURCES, camera="three_quarter", default_reps=3,
    tags=("barbell", "power", "olympic", "unilateral"),
)

clean_and_jerk = ExerciseDefinition(
    id="clean_and_jerk", name="Clean and jerk", category=Category.ATHLETIC,
    description="The two-part contested lift: the bar is cleaned to the shoulders, the lifter "
                "stands, and then jerks it overhead. More weight is lifted this way than by "
                "any other movement.",
    setup=("Clean grip, bar over the mid-foot", "Clean it to the front rack and stand fully",
           "Reset the breath, then dip, drive and split"),
    phases=(
        ph("Pull to the knee", CON, 0.6, _KNEE_C, pitch=40,
           cues=("Push the floor away; back angle constant",)),
        ph("Second pull", CON, 0.22, _EXT_C, pitch=0, easing="ease_out",
           cues=("Hips through, shrug, onto the toes",)),
        ph("Catch the clean", ECC, 0.3, merge(squat(110, 120, 20)[0], _RACK, grip()),
           pitch=20, easing="ease_in", cues=("Elbows around fast; receive it deep",)),
        ph("Stand", CON, 0.9, merge(pose(knee_flex=3), _RACK, grip()),
           cues=("Stand tall; elbows stay up",)),
        ph("Dip and drive", CON, 0.3, merge(pose(knee_flex=4, ankle_flex=-20),
                                            _JERK_PAST_THE_FACE, grip()),
           easing="ease_out", cues=("Short vertical dip, then drive it off the shoulders",)),
        ph("Jerk under", ECC, 0.25,
           merge(pose(hip_r_flex=55, knee_r_flex=60, ankle_r_flex=15,
                      **_SPLIT_BACK, toe_curl_l=_SPLIT_BACK_TOES),
                 _JERK_OVERHEAD, grip()), pitch=4, easing="ease_in",
           cues=("Split and lock in one movement",)),
        ph("Recover", CON, 0.9, merge(pose(knee_flex=4), _JERK_OVERHEAD, grip()),
           cues=("Feet back under, bar overhead, wait for the signal",)),
        ph("Lower it past the face", TRN, 0.4,
           merge(pose(knee_flex=4), _JERK_PAST_THE_FACE, grip()),
           cues=("Bend the arms and bring it down in front of the face",)),
        # Via the shoulders, which is what the cue below already said: going
        # straight from overhead to the floor position swung the bar and the
        # pitching head into each other, 7.4 units inside the skull.
        ph("Back to the shoulders", TRN, 0.35,
           merge(pose(knee_flex=4), _RACK, grip()),
           cues=("Catch it on the shoulders before it goes anywhere else",)),
        ph("Lower to the floor", TRN, 0.9, _START_C, pitch=50,
           cues=("Down to the shoulders, to the thighs, then set it down",)),
    ),
    muscles=(*_PULL_MUSCLES, mu("triceps_brachii", P, 0.8, note="the jerk lockout"),
             mu("deltoid_anterior", P, 0.85), mu("rotator_cuff", S, 0.6),
             mu("gluteus_medius", S, 0.6, note="the split"),
             mu("serratus_anterior", S, 0.55), mu("adductors", S, 0.5)),
    equipment=(eq("barbell", plates=2),),
    errors=("Rushing the stand between the clean and the jerk.",
            "Jerking from a rack position that has drifted forward.",
            "Treating it as two separate lifts with a long pause under load."),
    physio_notes=("The heaviest lift in the sport, and the reason the jerk is trained "
                  "separately: the clean rarely limits it.",),
    sources=_SOURCES, camera="three_quarter", default_reps=2,
    tags=("barbell", "power", "olympic"),
)

# ── What the lifts are built from ──────────────────────────────────────

overhead_squat = ExerciseDefinition(
    id="overhead_squat", name="Overhead squat", category=Category.ATHLETIC,
    description="A full squat with the bar locked overhead in the snatch grip. The position a "
                "snatch is caught in, held still: it exposes ankle, hip, thoracic and shoulder "
                "restriction in one movement.",
    setup=("Snatch-width grip, bar pressed overhead, elbows locked",
           "Bar over the mid-foot and stays there throughout",
           "Squat between the arms, not under them"),
    phases=(
        ph("Descend", ECC, 2.2, merge(squat(120, 125, 22)[0], _OVERHEAD_S, grip()), pitch=22,
           cues=("Knees out, chest up, bar back over the mid-foot",
                 "Push up into the bar the whole way down")),
        ph("Bottom", ISO, 0.6, merge(squat(120, 125, 22)[0], _OVERHEAD_S, grip()), pitch=22,
           cues=("Hip crease below the knee, arms still locked",)),
        ph("Stand", CON, 1.6, merge(pose(knee_flex=5), _OVERHEAD_S, grip()),
           cues=("Drive the floor away; the bar does not move forward",)),
        ph("Top", ISO, 0.5, merge(pose(knee_flex=5), _OVERHEAD_S, grip())),
    ),
    muscles=(mu("quadriceps", P, 0.9), mu("gluteus_maximus", P, 0.85),
             mu("rotator_cuff", P, 0.8, note="holds the bar back over the mid-foot"),
             mu("deltoid_anterior", P, 0.7), mu("trapezius_lower", P, 0.65),
             mu("serratus_anterior", S, 0.6), mu("erector_spinae", P, 0.75),
             mu("rectus_abdominis", S, 0.6), mu("obliques", S, 0.55),
             mu("adductors", S, 0.6), mu("hamstrings", S, 0.5),
             mu("soleus", S, 0.5), mu("tibialis_anterior", S, 0.5,
                                      note="the deep dorsiflexion the position needs"),
             mu("triceps_brachii", ST, 0.55)),
    equipment=(eq("barbell", plates=1),),
    errors=("Bar drifting forward, which is usually thoracic or shoulder range.",
            "Heels lifting -- ankle range, not weakness.",
            "Loading it before the empty bar position is comfortable."),
    physio_notes=("Used as a screen as much as a lift: the overhead squat test in the "
                  "Functional Movement Screen is this movement with a dowel.",),
    sources=_SOURCES, camera="three_quarter", default_reps=3,
    tags=("barbell", "olympic", "mobility"),
)

clean_pull = ExerciseDefinition(
    id="clean_pull", name="Clean pull", category=Category.ATHLETIC,
    description="The clean without the catch: the bar is pulled from the floor to full "
                "extension and set back down. It trains the pull with more weight than can "
                "be caught.",
    setup=("Clean grip and the same start position as the clean",
           "Pull to full extension and shrug; the arms stay straight",
           "No turnover -- put it back down"),
    phases=(
        ph("First pull", CON, 0.7, _KNEE_C, pitch=40,
           cues=("Same back angle off the floor; bar close to the shins",)),
        ph("Second pull", CON, 0.3, _EXT_C, pitch=0, easing="ease_out",
           cues=("Finish the hips and shrug; stay over the bar",
                 "Arms are straps, not levers")),
        ph("Lower", ECC, 1.2, _START_C, pitch=50,
           cues=("Control it back to the floor along the same line",)),
        ph("Reset", ISO, 0.4, _START_C, pitch=50,
           cues=("Re-set the back and the grip between reps",)),
    ),
    muscles=(*_PULL_MUSCLES,),
    equipment=(eq("barbell", plates=2),),
    errors=("Bending the arms, which is the fault the exercise exists to remove.",
            "Letting the hips shoot up first.",
            "Going so heavy the back angle changes off the floor."),
    physio_notes=("Trained above clean weights because no catch has to be survived; the "
                  "same logic gives the snatch pull.",),
    sources=_SOURCES, camera="three_quarter", default_reps=3,
    tags=("barbell", "power", "olympic", "posterior-chain"),
)

snatch_pull = ExerciseDefinition(
    id="snatch_pull", name="Snatch pull", category=Category.ATHLETIC,
    description="The snatch without the catch: the wide grip makes the pull longer and the "
                "start lower, and the bar finishes at the hip crease.",
    setup=("Snatch grip, hips a little lower than a clean start",
           "Pull to full extension with straight arms and a hard shrug",
           "Set it back down; there is no turnover"),
    phases=(
        ph("First pull", CON, 0.7, _KNEE_S, pitch=38,
           cues=("Back angle constant; the bar brushes the thighs",)),
        ph("Second pull", CON, 0.3, _EXT_S, pitch=0, easing="ease_out",
           cues=("Hips through to the bar; shrug at the top",)),
        ph("Lower", ECC, 1.2, _START_S, pitch=48,
           cues=("Same line back to the floor",)),
        ph("Reset", ISO, 0.4, _START_S, pitch=48, cues=("Re-set before the next rep",)),
    ),
    muscles=(*_PULL_MUSCLES, mu("trapezius_lower", S, 0.5)),
    equipment=(eq("barbell", plates=2),),
    errors=("Early arm bend.", "Hips rising faster than the shoulders.",
            "Bar swinging out from the body at the hip."),
    physio_notes=("The wider grip lowers the start position and lengthens the pull, which "
                  "is why snatch technique is limited by position more often than by "
                  "strength.",),
    sources=_SOURCES, camera="three_quarter", default_reps=3,
    tags=("barbell", "power", "olympic", "posterior-chain"),
)

EXERCISES = (squat_snatch, power_snatch, hang_power_clean, push_jerk, split_jerk,
             clean_and_jerk, overhead_squat, clean_pull, snatch_pull)
