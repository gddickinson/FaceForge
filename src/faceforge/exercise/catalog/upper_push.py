"""Pressing: bench press, incline press, push-up, overhead presses, dips, flyes, triceps.

Bench press bottom: shoulders abducted ~45 deg from the torso with a grip of
150 % biacromial width, elbow ~90 deg, bar ~1 cm from the sternum; at 60 %1RM
pectoralis major ~27 %MVIC, anterior deltoid 26-33 %, triceps ~15 %
(BENCH_INCLINE).  Upper pectoralis peaks at 30 deg incline; anterior deltoid
keeps rising to 60 deg.
"""

from __future__ import annotations

from faceforge.exercise.catalog._helpers import (
    ACE, BENCH_INCLINE, BENCH_TOP, CALATAYUD, CON, ECC, ECC_CON, EXRX, HIPS, ISO, KOLBER, NEUMANN,
    NSCA, P, S, ST, SAETERBAKKEN, SEATED_ON_BENCH, TRN, arms, eq, grip, merge, mu, only, ph,
    pose,
)
from faceforge.exercise.model import Category, ExerciseDefinition

_BENCH_LEGS = only(hip_flex=35, knee_flex=90, ankle_flex=-30)
_BENCH_BOTTOM = merge(pose(), _BENCH_LEGS, arms(flex=-10, abduct=45, elbow=95, forearm=-20), grip())
_BENCH_TOP = merge(pose(), _BENCH_LEGS, arms(flex=80, abduct=12, elbow=5, forearm=-20), grip())
_BENCH_EQUIP = eq("bench", attach="static", height=BENCH_TOP)

barbell_bench_press = ExerciseDefinition(
    id="barbell_bench_press", name="Barbell bench press", category=Category.UPPER_PUSH,
    description="Supine on a bench, the bar is lowered to the lower sternum with the elbows "
                "at ~45 deg from the torso and pressed back over the shoulders.",
    setup=("Eyes under the bar, five points of contact (head, shoulders, hips, both feet)",
           "Grip ~1.5x shoulder width, wrists stacked over the elbows",
           "Shoulder blades pulled back and down; slight arch, ribs up"),
    orientation="supine", anchor="none", base_position=(-85.0, BENCH_TOP + 15.0, 0.0),
    phases=(
        ph("Lower", ECC, 2.0, _BENCH_BOTTOM, cues=("Bar to the lower sternum, elbows ~45 deg "
                                                 "from the trunk", "Forearms vertical")),
        ph("Touch", ISO, 0.3, _BENCH_BOTTOM, cues=("Light touch; stay tight",)),
        ph("Press", CON, 1.5, _BENCH_TOP, cues=("Drive the bar up and slightly back over the "
                                              "shoulders", "Push the feet into the floor")),
        ph("Lockout", ISO, 0.5, _BENCH_TOP, cues=("Elbows straight, shoulder blades down",)),
    ),
    muscles=(mu("pectoralis_major", P, 0.9), mu("deltoid_anterior", P, 0.75),
             mu("triceps_brachii", P, 0.7), mu("serratus_anterior", S, 0.4),
             mu("latissimus_dorsi", ST, 0.3, note="controls the descent"),
             mu("rotator_cuff", ST, 0.35), mu("biceps_brachii", ST, 0.2),
             mu("forearm_flexors", ST, 0.45), mu("gluteus_maximus", ST, 0.3, note="leg drive"),
             mu("quadriceps", ST, 0.25)),
    equipment=(eq("barbell", plates=2), _BENCH_EQUIP),
    errors=("Elbows flared to 90 deg (raises impingement risk and shoulder strain).",
            "Bouncing the bar off the chest.", "Feet lifting or hips leaving the bench.",
            "Wrists bent back under the bar."),
    physio_notes=("Pectoralis major dominates the bottom of the lift; triceps and anterior "
                  "deltoid carry the mid and upper range.", "A narrower grip with the elbows "
                  "tucked shifts work to the clavicular pectoralis, anterior deltoid and "
                  "triceps."),
    sources=(BENCH_INCLINE, CALATAYUD, KOLBER, NSCA, ECC_CON), camera="side", tags=("barbell",),
)

incline_dumbbell_press = ExerciseDefinition(
    id="incline_dumbbell_press", name="Incline dumbbell press", category=Category.UPPER_PUSH,
    description="On a 30 deg incline, dumbbells are pressed from chest level to above the "
                "shoulders; the incline shifts emphasis to the clavicular pectoralis.",
    setup=("Bench at ~30 deg", "Dumbbells at the outer chest, palms forward",
           "Feet flat, shoulder blades back"),
    orientation="supine", anchor="none", base_position=(-85.0, BENCH_TOP + 15.0, 0.0),
    phases=(
        ph("Lower", ECC, 2.0, merge(pose(), _BENCH_LEGS, arms(flex=0, abduct=60, elbow=100, forearm=-30), grip()),
           pitch=30, pivot=HIPS,
           cues=("Dumbbells to the outer chest, elbows below the wrists",)),
        ph("Press", CON, 1.5, merge(pose(), _BENCH_LEGS, arms(flex=75, abduct=20, elbow=10, forearm=-30), grip()),
           pitch=30, pivot=HIPS,
           cues=("Press up and slightly in; do not clash the dumbbells",)),
        ph("Top", ISO, 0.4, merge(pose(), _BENCH_LEGS, arms(flex=75, abduct=20, elbow=10, forearm=-30), grip()),
           pitch=30, pivot=HIPS),
    ),
    muscles=(mu("pectoralis_upper", P, 0.9), mu("deltoid_anterior", P, 0.85),
             mu("triceps_brachii", S, 0.6), mu("pectoralis_major", S, 0.5),
             mu("serratus_anterior", S, 0.4), mu("rotator_cuff", ST, 0.35),
             mu("biceps_brachii", ST, 0.25), mu("forearm_flexors", ST, 0.4)),
    equipment=(eq("dumbbell", attach="hand_r"), eq("dumbbell", attach="hand_l"),
               eq("bench", attach="static", height=BENCH_TOP, incline_deg=-30.0)),
    errors=("Bench too steep (>45 deg) turns it into a shoulder press.",
            "Lowering the elbows far below the bench (anterior capsule strain)."),
    physio_notes=("Upper pectoralis EMG peaks at 30 deg; above 45 deg anterior deltoid "
                  "takes over.",),
    sources=(BENCH_INCLINE, NSCA), camera="side", tags=("dumbbell",),
)

# The body is inclined (head end up) by asin(shoulder height / hand-to-toe
# length): ~20 deg on straight arms, ~12 at the bottom.  Tilting a prone body
# head-up turns its anterior direction toward the head, so the arms stay
# vertical over the hands at 90 deg MINUS the incline (measured on the rig).
_PU_TOP = merge(pose(ankle_flex=45), arms(flex=70, abduct=10, elbow=0))
_PU_BOTTOM = merge(pose(ankle_flex=45), arms(flex=38, abduct=45, elbow=95))

push_up = ExerciseDefinition(
    id="push_up", name="Push-up", category=Category.UPPER_PUSH,
    description="A closed-chain press: hands under the shoulders, the body lowers as a rigid "
                "plank until the chest is near the floor and presses back up.",
    setup=("Hands slightly wider than the shoulders, fingers forward",
           "Body in a straight line from head to heels; glutes and abdominals braced",
           "Elbows track ~45 deg from the trunk"),
    orientation="prone", anchor="hands", base_position=(-85.0, 30.0, 0.0),
    phases=(
        # Hands on the floor, toes on the floor: the straight body is inclined
        # by asin(shoulder height / body length): ~20 deg on straight arms,
        # ~10 deg at the bottom.  Negative pitch raises the head end when prone.
        ph("Lower", ECC, 1.8, _PU_BOTTOM, pitch=-8,
           cues=("Chest to a fist's height from the floor",
                 "Keep the plank; do not sag or pike")),
        ph("Bottom", ISO, 0.3, _PU_BOTTOM, pitch=-8),
        ph("Press", CON, 1.2, _PU_TOP, pitch=-20,
           cues=("Push the floor away; protract the shoulder blades at the top",)),
        ph("Top", ISO, 0.4, _PU_TOP, pitch=-20, cues=("Elbows straight, hips level",)),
    ),
    muscles=(mu("pectoralis_major", P, 0.8), mu("triceps_brachii", P, 0.75),
             mu("deltoid_anterior", P, 0.7), mu("serratus_anterior", S, 0.65,
                                                note="protraction at the top"),
             mu("rectus_abdominis", ST, 0.5), mu("obliques", ST, 0.4),
             mu("erector_spinae", ST, 0.3), mu("gluteus_maximus", ST, 0.3),
             mu("quadriceps", ST, 0.3), mu("rotator_cuff", ST, 0.3)),
    equipment=(eq("mat", attach="static"),),
    errors=("Sagging hips (lumbar extension).", "Head dropping / neck flexion.",
            "Elbows flared to 90 deg.", "Half range of motion."),
    physio_notes=("Push-up and bench press produce comparable pectoralis and triceps EMG "
                  "when load-matched (Calatayud 2015); the push-up adds serratus anterior "
                  "and trunk stabiliser demand.",),
    sources=(CALATAYUD, ACE, NEUMANN), camera="side", tags=("no equipment",),
)

_OHP_START = merge(pose(knee_flex=5), arms(flex=35, abduct=20, elbow=140, wrist=-20), grip())
_OHP_TOP = merge(pose(knee_flex=5), arms(flex=172, abduct=8, elbow=5, wrist=-10), grip())

overhead_press = ExerciseDefinition(
    id="overhead_press", name="Standing barbell overhead press", category=Category.UPPER_PUSH,
    description="From the clavicles the bar travels straight up past the face to lockout "
                "overhead with the head pushed through under the bar.",
    setup=("Bar on the front of the shoulders, grip just outside the shoulders, elbows "
           "slightly in front of the bar", "Feet hip-width, glutes and abdominals braced",
           "Chin tucked so the bar can pass"),
    phases=(
        ph("Press", CON, 1.5, _OHP_TOP, cues=("Drive the bar straight up; move the head "
                                            "through once the bar clears it",
                                            "Finish with the bar over the mid-foot, shrug up")),
        ph("Lockout", ISO, 0.5, _OHP_TOP, cues=("Elbows locked, ribs down, no lumbar arch",)),
        ph("Lower", ECC, 2.0, _OHP_START, cues=("Lower to the clavicles under control",)),
        ph("Rack", ISO, 0.4, _OHP_START, cues=("Re-brace",)),
    ),
    muscles=(mu("deltoid_anterior", P, 0.95), mu("deltoid_lateral", P, 0.8),
             mu("triceps_brachii", P, 0.8), mu("trapezius_upper", S, 0.65, note="upward "
                                                                             "rotation"),
             mu("serratus_anterior", S, 0.6), mu("trapezius_lower", S, 0.4),
             mu("rotator_cuff", S, 0.45), mu("pectoralis_upper", S, 0.4),
             mu("rectus_abdominis", ST, 0.45), mu("obliques", ST, 0.4),
             mu("erector_spinae", ST, 0.4), mu("gluteus_maximus", ST, 0.35),
             mu("forearm_flexors", ST, 0.4)),
    equipment=(eq("barbell", plates=1),),
    errors=("Leaning back and arching the lumbar spine.", "Pressing the bar forward around "
            "the face instead of moving the head.", "Incomplete lockout / no shrug at the top."),
    physio_notes=("Full overhead reach needs ~120 deg glenohumeral elevation plus ~60 deg "
                  "scapular upward rotation (2:1 scapulohumeral rhythm).",
                  "Standing free-weight pressing recruits more deltoid and trunk stabilisers "
                  "than seated or machine variants (Saeterbakken 2013)."),
    sources=(SAETERBAKKEN, NEUMANN, NSCA, KOLBER), camera="three_quarter", tags=("barbell",),
)

seated_dumbbell_shoulder_press = ExerciseDefinition(
    id="seated_dumbbell_shoulder_press", name="Seated dumbbell shoulder press",
    category=Category.UPPER_PUSH,
    description="Seated on a bench, dumbbells are pressed from ear height to overhead in the "
                "scapular plane.",
    setup=("Bench upright, feet flat", "Dumbbells at ear height, elbows just in front of the "
           "shoulders, palms forward"),
    orientation="seated", anchor="none", base_position=SEATED_ON_BENCH,
    phases=(
        ph("Press", CON, 1.4,
           merge(pose(hip_flex=90, knee_flex=90), arms(flex=25, abduct=165, elbow=8, forearm=-30), grip()),
           cues=("Press up and slightly in until the arms are straight",)),
        ph("Top", ISO, 0.4,
           merge(pose(hip_flex=90, knee_flex=90), arms(flex=25, abduct=165, elbow=8, forearm=-30), grip())),
        ph("Lower", ECC, 2.0,
           merge(pose(hip_flex=90, knee_flex=90), arms(flex=25, abduct=85, elbow=95, forearm=-30), grip()),
           cues=("Lower until the upper arms are parallel to the floor",)),
    ),
    muscles=(mu("deltoid_anterior", P, 0.9), mu("deltoid_lateral", P, 0.85),
             mu("triceps_brachii", P, 0.7), mu("trapezius_upper", S, 0.6),
             mu("serratus_anterior", S, 0.55), mu("rotator_cuff", S, 0.45),
             mu("pectoralis_upper", S, 0.35), mu("erector_spinae", ST, 0.3),
             mu("forearm_flexors", ST, 0.4)),
    equipment=(eq("dumbbell", attach="hand_r"), eq("dumbbell", attach="hand_l"),
               eq("bench", attach="static", height=BENCH_TOP, length=60.0)),
    errors=("Elbows dropping far below the shoulders (over-stretching the anterior "
            "capsule).", "Arching away from the backrest."),
    physio_notes=("Pressing in the scapular plane (~30 deg forward of the frontal plane) "
                  "keeps the rotator cuff in its least impinged position.",),
    sources=(SAETERBAKKEN, KOLBER, NSCA), camera="three_quarter", tags=("dumbbell",),
)

_DIP_TOP = merge(pose(hip_flex=25, knee_flex=95), arms(flex=0, abduct=10, elbow=0), grip())
_DIP_BOTTOM = merge(pose(hip_flex=25, knee_flex=95), arms(flex=-30, abduct=20, elbow=95), grip())

parallel_bar_dip = ExerciseDefinition(
    id="parallel_bar_dip", name="Parallel-bar dip", category=Category.UPPER_PUSH,
    description="Supported on straight arms between two bars, the body lowers until the "
                "upper arms are about parallel and presses back up.",
    setup=("Arms locked, shoulders down away from the ears", "Slight forward lean for "
           "chest emphasis, upright for triceps", "Knees bent, feet crossed"),
    orientation="hanging", anchor="hands", anchor_point=(0.0, 125.0, 0.0),
    phases=(
        ph("Lower", ECC, 1.8, _DIP_BOTTOM, pitch=15, cues=("Elbows back, lower to ~90 deg",)),
        ph("Bottom", ISO, 0.3, _DIP_BOTTOM, pitch=15, cues=("No lower than the shoulder "
                                                          "tolerates",)),
        ph("Press", CON, 1.3, _DIP_TOP, pitch=10, cues=("Press to lockout; depress the "
                                                       "shoulder blades",)),
        ph("Top", ISO, 0.4, _DIP_TOP, pitch=10),
    ),
    muscles=(mu("pectoralis_major", P, 0.85, note="lower/sternal fibres"),
             mu("triceps_brachii", P, 0.9), mu("deltoid_anterior", S, 0.6),
             mu("latissimus_dorsi", S, 0.4), mu("trapezius_lower", ST, 0.4),
             mu("rhomboids", ST, 0.3), mu("rectus_abdominis", ST, 0.3),
             mu("forearm_flexors", ST, 0.5)),
    equipment=(eq("dip_station", attach="static", height=125.0),),
    errors=("Going too deep with the shoulders rolled forward.", "Shrugging the shoulders "
            "up at the top.", "Swinging the legs."),
    physio_notes=("Deep dips load the anterior shoulder; a common source of gym shoulder "
                  "pain (Kolber 2010). Limit depth to 90 deg elbow flexion.",),
    sources=(KOLBER, NSCA, EXRX), camera="side", camera_target=(0.0, 130.0, 0.0),
    tags=("bodyweight",),
)

dumbbell_lateral_raise = ExerciseDefinition(
    id="dumbbell_lateral_raise", name="Dumbbell lateral raise", category=Category.UPPER_PUSH,
    description="Light dumbbells are raised sideways to shoulder height in the scapular plane "
                "with a soft elbow.",
    setup=("Stand tall, dumbbells at the sides, slight elbow bend",
           "Lean forward 10-15 deg from the hips"),
    phases=(
        ph("Raise", CON, 1.2, merge(pose(), arms(flex=20, abduct=88, elbow=15, forearm=-20), grip()),
           pitch=10, cues=("Lead with the elbows; little finger level with the thumb",
                           "Stop at shoulder height")),
        ph("Top", ISO, 0.4, merge(pose(), arms(flex=20, abduct=88, elbow=15, forearm=-20), grip()),
           pitch=10),
        ph("Lower", ECC, 2.0, merge(pose(), arms(flex=10, abduct=8, elbow=15, forearm=-20), grip()),
           pitch=10, cues=("Lower slowly; do not let the weights touch the thighs",)),
    ),
    muscles=(mu("deltoid_lateral", P, 0.9), mu("rotator_cuff", S, 0.6, note="supraspinatus "
                                                                             "initiates"),
             mu("deltoid_anterior", S, 0.5), mu("trapezius_upper", S, 0.6),
             mu("serratus_anterior", S, 0.5), mu("deltoid_posterior", S, 0.3),
             mu("erector_spinae", ST, 0.25), mu("forearm_flexors", ST, 0.3)),
    equipment=(eq("dumbbell", attach="hand_r", head_radius=5.0),
               eq("dumbbell", attach="hand_l", head_radius=5.0)),
    errors=("Shrugging (upper trapezius takes over).", "Swinging the trunk.",
            "Raising above shoulder height with internal rotation (impingement)."),
    physio_notes=("Deltoid moment arm is smallest near the side; the rotator cuff must hold "
                  "the humeral head down as the deltoid pulls up.",),
    sources=(NEUMANN, KOLBER, ACE), camera="front", tags=("dumbbell",),
)

triceps_pushdown = ExerciseDefinition(
    id="triceps_pushdown", name="Cable triceps pushdown", category=Category.UPPER_PUSH,
    description="Elbows pinned at the sides, the forearms extend against a cable from ~100 "
                "deg to full extension.",
    setup=("Cable at head height, bar or rope in an overhand grip", "Elbows at the sides, "
           "slight forward lean", "Wrists neutral"),
    phases=(
        ph("Push down", CON, 1.0, merge(pose(), arms(flex=15, abduct=5, elbow=5, forearm=-60), grip()),
           pitch=8, cues=("Extend the elbows fully; upper arms do not move",)),
        ph("Bottom", ISO, 0.4, merge(pose(), arms(flex=15, abduct=5, elbow=5, forearm=-60), grip()),
           pitch=8, cues=("Squeeze the triceps",)),
        ph("Return", ECC, 1.5, merge(pose(), arms(flex=15, abduct=5, elbow=105, forearm=-60), grip()),
           pitch=8, cues=("Let the forearms rise to ~100 deg under control",)),
    ),
    muscles=(mu("triceps_brachii", P, 0.95), mu("forearm_extensors", ST, 0.3),
             mu("deltoid_posterior", ST, 0.3, note="holds the arm back"),
             mu("latissimus_dorsi", ST, 0.3), mu("rectus_abdominis", ST, 0.25)),
    equipment=(eq("cable_handle",),),
    errors=("Elbows drifting forward and back (shoulder joins in).",
            "Leaning over the cable to use body weight."),
    physio_notes=("Isolates elbow extension; the long head is under-loaded with the arm at "
                  "the side (overhead extensions load it more).",),
    sources=(NEUMANN, EXRX, ACE), camera="side", tags=("cable",),
)

lying_triceps_extension = ExerciseDefinition(
    id="lying_triceps_extension", name="Lying triceps extension (skull crusher)",
    category=Category.UPPER_PUSH,
    description="Supine with the arms vertical, the elbows flex to lower the bar toward the "
                "forehead and extend to lockout.",
    setup=("Upper arms vertical or angled slightly back toward the head",
           "Narrow overhand grip", "Feet flat, shoulder blades back"),
    orientation="supine", anchor="none", base_position=(-85.0, BENCH_TOP + 15.0, 0.0),
    phases=(
        ph("Lower", ECC, 2.0, merge(pose(), _BENCH_LEGS, arms(flex=100, abduct=5, elbow=110, forearm=-20), grip()),
           cues=("Bend only the elbows; bar toward the forehead or just behind",)),
        ph("Extend", CON, 1.2, merge(pose(), _BENCH_LEGS, arms(flex=100, abduct=5, elbow=8, forearm=-20), grip()),
           cues=("Extend to lockout with the upper arms still",)),
        ph("Top", ISO, 0.4, merge(pose(), _BENCH_LEGS, arms(flex=100, abduct=5, elbow=8, forearm=-20), grip())),
    ),
    muscles=(mu("triceps_brachii", P, 0.95, note="all three heads, long head lengthened"),
             mu("deltoid_anterior", ST, 0.3), mu("pectoralis_major", ST, 0.25),
             mu("forearm_flexors", ST, 0.45), mu("latissimus_dorsi", ST, 0.2)),
    equipment=(eq("barbell", length=120.0, plates=1, plate_radius=12.0), _BENCH_EQUIP),
    errors=("Upper arms swinging toward the chest.", "Elbows flaring out."),
    physio_notes=("Keeping the upper arms angled back keeps tension on the triceps at "
                  "lockout.",),
    sources=(NEUMANN, EXRX), camera="side", tags=("barbell",),
)

dumbbell_chest_fly = ExerciseDefinition(
    id="dumbbell_chest_fly", name="Dumbbell chest fly", category=Category.UPPER_PUSH,
    description="Supine, the arms arc out and down with a fixed slight elbow bend, then "
                "squeeze back together over the chest.",
    setup=("Dumbbells over the chest, palms facing", "Elbows bent ~15 deg and kept there",
           "Shoulder blades back"),
    orientation="supine", anchor="none", base_position=(-85.0, BENCH_TOP + 15.0, 0.0),
    phases=(
        ph("Open", ECC, 2.0, merge(pose(), _BENCH_LEGS, arms(flex=10, abduct=80, elbow=20), grip()),
           cues=("Lower in a wide arc until the upper arms are level with the bench",)),
        ph("Stretch", ISO, 0.3, merge(pose(), _BENCH_LEGS, arms(flex=10, abduct=80, elbow=20), grip())),
        ph("Close", CON, 1.5, merge(pose(), _BENCH_LEGS, arms(flex=85, abduct=10, elbow=20), grip()),
           cues=("Hug a barrel: bring the dumbbells together over the chest",)),
    ),
    muscles=(mu("pectoralis_major", P, 0.9), mu("deltoid_anterior", S, 0.5),
             mu("biceps_brachii", ST, 0.35, note="long head stabilises the elbow"),
             mu("serratus_anterior", ST, 0.3), mu("rotator_cuff", ST, 0.35),
             mu("forearm_flexors", ST, 0.35)),
    equipment=(eq("dumbbell", attach="hand_r"), eq("dumbbell", attach="hand_l"), _BENCH_EQUIP),
    errors=("Bending and straightening the elbows (turns it into a press).",
            "Lowering far below the bench with heavy weight (anterior shoulder strain)."),
    physio_notes=("Loads the pectoralis at long length; keep the range modest with a "
                  "history of anterior instability.",),
    sources=(NEUMANN, KOLBER, EXRX), camera="side", tags=("dumbbell",),
)

EXERCISES = (barbell_bench_press, incline_dumbbell_press, push_up, overhead_press,
             seated_dumbbell_shoulder_press, parallel_bar_dip, dumbbell_lateral_raise,
             triceps_pushdown, lying_triceps_extension, dumbbell_chest_fly)
