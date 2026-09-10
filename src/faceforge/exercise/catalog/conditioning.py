"""Cyclic conditioning: bike, rower, walking, running, jump rope, jacks, climbers, ropes.

Cyclic movements are authored as one cycle of phases with per-phase, per-side
activation overrides taken from published muscle-timing data: pedalling
(HUG: gluteus maximus and vasti fire from ~340 to ~130 deg of crank angle,
hamstrings 30-220, gastrocnemius 30-270, tibialis anterior in the recovery),
gait (PERRY: vasti in loading response, gluteus maximus late swing to
loading, plantarflexors mid to terminal stance, tibialis anterior at
initial contact and through swing, iliopsoas pre-swing to initial swing) and
the rowing stroke (KLESHNEV: legs, then trunk, then arms in the drive).
"""

from __future__ import annotations

import math

from faceforge.exercise.catalog._helpers import (
    ACE, CON, ECC, HIPS, HUG, ISO, KLESHNEV, NEUMANN, NSCA, P, PERRY, S, ST, TRN, arms, combine,
    eq, grip, merge, mu, only, ph, pose,
)
from faceforge.exercise.model import Category, ExerciseDefinition, Phase


def _in_arc(angle: float, start: float, end: float) -> bool:
    """True if ``angle`` (deg) lies in the circular arc from start to end."""
    angle, start, end = angle % 360, start % 360, end % 360
    return start <= angle <= end if start <= end else (angle >= start or angle <= end)


#: Crank-angle activation ranges (deg, 0 = top dead centre) per HUG.
_PEDAL_TIMING = {
    "gluteus_maximus": (340, 130, 0.8), "quadriceps": (300, 130, 0.9),
    "hamstrings": (30, 220, 0.6), "gastrocnemius": (30, 270, 0.6), "soleus": (340, 180, 0.6),
    "tibialis_anterior": (250, 90, 0.4), "hip_flexors": (200, 340, 0.4),
}


def _pedal_phase(i: int, n: int, seconds: float) -> Phase:
    theta = 360.0 * i / n                       # right crank angle at the END of the phase
    kw = {}
    for side, offset in (("r", 0.0), ("l", 180.0)):
        a = math.radians(theta + offset)
        kw[f"hip_{side}_flex"] = 85 + 25 * math.cos(a)
        kw[f"knee_{side}_flex"] = 75 + 37 * math.cos(a)
        kw[f"ankle_{side}_flex"] = -5 + 10 * math.cos(a)
    act = {}
    for group, (start, end, level) in _PEDAL_TIMING.items():
        for side, offset in (("R", 0.0), ("L", 180.0)):
            act[f"{group}:{side}"] = level if _in_arc(theta + offset, start, end) else 0.1
    return ph(f"Crank {int(theta):d} deg", CON, seconds,
              merge(pose(**kw), arms(flex=60, abduct=10, elbow=25), grip()),
              pitch=25, pivot=HIPS, act=act, easing="linear",
              cues=("Smooth circles: push over the top, scrape the mud off at the bottom",)
              if i % 4 == 0 else ())


stationary_bike = ExerciseDefinition(
    id="stationary_bike", name="Stationary bike (upright)", category=Category.CONDITIONING,
    description="Seated pedalling at ~60 rpm: the hip and knee extensors drive the downstroke, "
                "the plantarflexors finish it and the flexors recover the pedal.",
    setup=("Saddle height: knee ~25-35 deg flexed at the bottom of the stroke",
           "Ball of the foot over the pedal axle", "Light grip, elbows soft, trunk ~25 deg"),
    orientation="seated", anchor="none", base_position=(0.0, 183.0, -12.0),
    phases=tuple(_pedal_phase(i + 1, 8, 0.125) for i in range(8)),
    muscles=(mu("quadriceps", P, 0.9), mu("gluteus_maximus", P, 0.8), mu("hamstrings", S, 0.6),
             mu("gastrocnemius", S, 0.6), mu("soleus", S, 0.6), mu("tibialis_anterior", S, 0.4),
             mu("hip_flexors", S, 0.4), mu("erector_spinae", ST, 0.3), mu("triceps_brachii", ST, 0.25)),
    equipment=(eq("bike", attach="static"),),
    errors=("Saddle too low (knee over-flexed, high patellofemoral load).",
            "Knees flaring out.", "Mashing the pedals instead of spinning circles."),
    physio_notes=("Non-weight-bearing knee range of motion; saddle height controls peak knee "
                  "flexion (~110 deg at the top with a correct fit).",),
    sources=(HUG, NEUMANN), camera="side", default_reps=4, tags=("machine", "cardio", "rehab"),
)


def _row_pose(pitch: float, knee: float, ankle: float, flex: float, elbow: float) -> dict:
    return merge(pose(hip_flex=90 + pitch, knee_flex=knee, ankle_flex=ankle),
                 arms(flex=flex, abduct=5, elbow=elbow), grip())


rowing_machine = ExerciseDefinition(
    id="rowing_machine", name="Rowing machine", category=Category.CONDITIONING,
    description="The stroke: catch with shins vertical and arms long, drive with the legs, "
                "swing the trunk, then pull the arms; recover in the reverse order.",
    setup=("Feet strapped, straps over the ball of the foot", "Catch: shins vertical, trunk "
           "leaning forward ~25 deg, arms straight, shoulders relaxed"),
    orientation="seated", anchor="feet", anchor_point=(60.0, 36.0, 0.0),
    base_position=(-40.0, 108.0, 0.0),
    phases=(
        ph("Drive: legs", CON, 0.4, _row_pose(10, 45, 5, 65, 5), pitch=10, pivot=HIPS, yaw=-90,
           act={"quadriceps": 0.95, "gluteus_maximus": 0.9, "hamstrings": 0.5,
                "erector_spinae": 0.6, "latissimus_dorsi": 0.3},
           cues=("Push with the legs first; arms stay straight, trunk angle unchanged",)),
        ph("Drive: back and arms", CON, 0.4, _row_pose(-20, 10, -10, -10, 110), pitch=-20,
           pivot=HIPS, yaw=-90,
           act={"quadriceps": 0.6, "gluteus_maximus": 0.7, "erector_spinae": 0.8,
                "latissimus_dorsi": 0.85, "rhomboids": 0.7, "trapezius_middle": 0.7,
                "deltoid_posterior": 0.7, "biceps_brachii": 0.75},
           cues=("Swing the trunk through vertical, then pull the handle to the lower ribs",)),
        ph("Finish", ISO, 0.2, _row_pose(-20, 10, -10, -10, 110), pitch=-20, pivot=HIPS, yaw=-90,
           cues=("Legs flat, slight layback, elbows past the body",)),
        ph("Recovery: arms and body", TRN, 0.5, _row_pose(5, 30, 5, 70, 5), pitch=5, pivot=HIPS,
           yaw=-90, act={"hip_flexors": 0.35, "rectus_abdominis": 0.35},
           cues=("Arms away, then hinge forward from the hips",)),
        ph("Recovery: slide to the catch", TRN, 0.9, _row_pose(25, 115, 25, 65, 5), pitch=25,
           pivot=HIPS, yaw=-90, act={"hip_flexors": 0.4, "tibialis_anterior": 0.4},
           cues=("Slide forward slowly until the shins are vertical",)),
    ),
    muscles=(mu("quadriceps", P, 0.95), mu("gluteus_maximus", P, 0.9), mu("erector_spinae", P, 0.8),
             mu("latissimus_dorsi", P, 0.85), mu("hamstrings", S, 0.5), mu("rhomboids", S, 0.7),
             mu("trapezius_middle", S, 0.7), mu("deltoid_posterior", S, 0.7),
             mu("biceps_brachii", S, 0.75), mu("forearm_flexors", ST, 0.6),
             mu("rectus_abdominis", S, 0.4), mu("gastrocnemius", S, 0.4),
             mu("hip_flexors", S, 0.4), mu("tibialis_anterior", S, 0.35)),
    equipment=(eq("rower", attach="static"),),
    errors=("Opening the back before the legs have finished (shooting the slide).",
            "Pulling with the arms early.", "Rounding the lumbar spine at the catch.",
            "Rushing the recovery."),
    physio_notes=("Drive:recovery about 1:2 in time; the trunk should not go past ~25 deg "
                  "of forward lean at the catch.",),
    sources=(KLESHNEV, NSCA), camera="side", default_reps=4, tags=("machine", "cardio"),
)

#: Gait cycle landmarks (% cycle of the RIGHT leg): hip, knee, ankle in degrees (PERRY).
_WALK_R = [(0, 30, 5, 0), (12, 25, 18, -7), (31, 0, 5, 8), (50, -12, 15, 12), (62, -5, 40, -18),
           (75, 20, 60, 2), (87, 30, 30, 0), (100, 30, 5, 0)]
_RUN_R = [(0, 35, 20, 5), (12, 25, 40, 10), (25, 5, 30, 18), (40, -15, 15, -22),
          (55, -5, 60, -5), (70, 25, 95, 5), (85, 45, 50, 0), (100, 35, 20, 5)]

#: Muscle timing by % gait cycle (start, end, level) per PERRY.
_GAIT_TIMING = {
    "gluteus_maximus": (85, 15, 0.6), "quadriceps": (90, 25, 0.6), "hamstrings": (80, 12, 0.6),
    "tibialis_anterior": (55, 12, 0.5), "gastrocnemius": (10, 52, 0.7), "soleus": (10, 52, 0.7),
    "hip_flexors": (48, 72, 0.5), "gluteus_medius": (0, 42, 0.5), "erector_spinae": (95, 8, 0.4),
    "adductors": (55, 75, 0.3),
}


def _interp_cycle(table, pct: float) -> tuple[float, float, float]:
    pct = pct % 100
    for (p0, h0, k0, a0), (p1, h1, k1, a1) in zip(table, table[1:]):
        if p0 <= pct <= p1:
            f = 0.0 if p1 == p0 else (pct - p0) / (p1 - p0)
            return h0 + (h1 - h0) * f, k0 + (k1 - k0) * f, a0 + (a1 - a0) * f
    return table[0][1:]


def _gait_phase(i: int, n: int, seconds: float, table, run: bool) -> Phase:
    pct = 100.0 * i / n
    hr, kr, ar = _interp_cycle(table, pct)
    hl, kl, al = _interp_cycle(table, pct + 50)
    swing = 20 if run else 12
    arm_r = -swing * math.cos(math.radians(3.6 * pct))   # right arm opposes right leg
    kw = dict(hip_r_flex=hr, knee_r_flex=kr, ankle_r_flex=ar,
              hip_l_flex=hl, knee_l_flex=kl, ankle_l_flex=al)
    act = {}
    for group, (start, end, level) in _GAIT_TIMING.items():
        for side, offset in (("R", 0.0), ("L", 50.0)):
            lvl = level * (1.4 if run else 1.0)
            act[f"{group}:{side}"] = min(1.0, lvl) if _in_arc(3.6 * (pct + offset), 3.6 * start,
                                                              3.6 * end) else 0.1
    lift = 0.0
    if run and (20 < pct < 30 or 70 < pct < 80):
        lift = 8.0                                        # flight phases
    return ph(f"{int(pct)} % of stride", CON, seconds,
              merge(pose(**kw), arms(flex=arm_r, elbow=90 if run else 20, side="r"),
                    arms(flex=-arm_r, elbow=90 if run else 20, side="l")),
              pitch=8 if run else 3, act=act, lift=lift, easing="linear",
              cues=(("Land under the hips, drive the knee forward",) if run and pct == 0
                    else ("Heel strike, roll through the foot, push off the big toe",)
                    if pct == 0 else ()))


treadmill_walk = ExerciseDefinition(
    id="treadmill_walk", name="Treadmill walking", category=Category.CONDITIONING,
    description="The gait cycle at ~1.1 s per stride: heel strike, loading, mid and terminal "
                "stance, push-off, then swing.",
    setup=("Look ahead, arms swinging opposite the legs", "Land on the heel, roll to the toe"),
    phases=tuple(_gait_phase(i + 1, 8, 0.1375, _WALK_R, run=False) for i in range(8)),
    muscles=(mu("gluteus_maximus", P, 0.6), mu("quadriceps", P, 0.6), mu("gastrocnemius", P, 0.7),
             mu("soleus", P, 0.7), mu("hamstrings", S, 0.6), mu("tibialis_anterior", S, 0.5),
             mu("hip_flexors", S, 0.5), mu("gluteus_medius", S, 0.5), mu("erector_spinae", ST, 0.4),
             mu("adductors", S, 0.3)),
    errors=("Over-striding (heel far ahead of the hips).", "Holding the handrails."),
    physio_notes=("Perry's timing: vasti in loading response, gluteus maximus from terminal "
                  "swing to loading, plantarflexors from mid-stance to push-off, tibialis "
                  "anterior at heel strike and throughout swing.",),
    sources=(PERRY, NEUMANN), camera="side", default_reps=4, tags=("cardio", "gait", "rehab"),
)

treadmill_run = ExerciseDefinition(
    id="treadmill_run", name="Treadmill running", category=Category.CONDITIONING,
    description="Running gait with a flight phase, greater knee flexion in swing and a "
                "mid-foot landing under the hips.",
    setup=("Slight forward lean from the ankles", "Elbows ~90 deg, hands relaxed",
           "Cadence ~170-180 steps per minute"),
    phases=tuple(_gait_phase(i + 1, 8, 0.0875, _RUN_R, run=True) for i in range(8)),
    muscles=(mu("gluteus_maximus", P, 0.85), mu("quadriceps", P, 0.85), mu("gastrocnemius", P, 0.95),
             mu("soleus", P, 0.95), mu("hamstrings", P, 0.8), mu("tibialis_anterior", S, 0.6),
             mu("hip_flexors", S, 0.7), mu("gluteus_medius", S, 0.7), mu("erector_spinae", ST, 0.5),
             mu("rectus_abdominis", ST, 0.4), mu("deltoid_anterior", ST, 0.3)),
    errors=("Over-striding with a heel strike far ahead.", "Excessive vertical bounce.",
            "Crossing the arms over the midline."),
    physio_notes=("Ground reaction forces of 2-3 body weights; hamstrings peak in late swing "
                  "(the common strain moment).",),
    sources=(PERRY, NEUMANN), camera="side", default_reps=4, tags=("cardio", "gait"),
)

_ROPE_ARMS = combine(arms(flex=10, abduct=15, elbow=95), grip())

jump_rope = ExerciseDefinition(
    id="jump_rope", name="Jump rope (single bounce)", category=Category.CONDITIONING,
    description="Small, fast hops on the balls of the feet with the rope turned from the "
                "wrists; the ankle plantarflexors do most of the work.",
    setup=("Elbows close to the ribs, hands at hip height", "Jump only 2-3 cm; land softly "
           "on the balls of the feet", "Turn the rope from the wrists, not the shoulders"),
    phases=(
        ph("Load", ECC, 0.12, merge(pose(knee_flex=20, ankle_flex=10), _ROPE_ARMS),
           cues=("Soft knees, absorb through the ankles",)),
        ph("Take-off", CON, 0.1, merge(pose(knee_flex=5, ankle_flex=-28), _ROPE_ARMS),
           cues=("Push off the balls of the feet",)),
        ph("Flight", TRN, 0.15, merge(pose(knee_flex=8, ankle_flex=-20), _ROPE_ARMS), lift=8.0,
           cues=("Rope passes under the feet",)),
        ph("Land", ECC, 0.13, merge(pose(knee_flex=15, ankle_flex=5), _ROPE_ARMS), lift=0.0),
    ),
    muscles=(mu("gastrocnemius", P, 0.85), mu("soleus", P, 0.85), mu("quadriceps", S, 0.5),
             mu("tibialis_anterior", S, 0.4), mu("gluteus_maximus", S, 0.3),
             mu("forearm_flexors", S, 0.5), mu("forearm_extensors", S, 0.4),
             mu("deltoid_lateral", S, 0.3), mu("rectus_abdominis", ST, 0.3)),
    equipment=(eq("jump_rope",),),
    errors=("Jumping too high / double bouncing.", "Turning the rope from the shoulders.",
            "Landing on the heels."),
    physio_notes=("Ground contact ~0.2 s with 2-3 body weights; build up gradually for the "
                  "Achilles and plantar fascia.",),
    sources=(NEUMANN, ACE), camera="front", default_reps=6, tags=("cardio", "rope"),
)

jumping_jack = ExerciseDefinition(
    id="jumping_jack", name="Jumping jack", category=Category.CONDITIONING,
    description="A hop that spreads the feet as the arms swing overhead, and a hop back.",
    setup=("Stand tall, feet together, arms at the sides",),
    phases=(
        ph("Out", CON, 0.3, merge(pose(hip_abduct=25, knee_flex=15, ankle_flex=-15),
                                  arms(abduct=170, elbow=10)), lift=3.0,
           cues=("Hop the feet wide as the hands clap overhead",)),
        ph("In", CON, 0.3, merge(pose(hip_abduct=0, knee_flex=15, ankle_flex=-15),
                                 arms(abduct=10, elbow=10)), lift=3.0,
           cues=("Hop the feet together as the arms come down",)),
    ),
    muscles=(mu("deltoid_lateral", P, 0.6), mu("gluteus_medius", P, 0.6), mu("gastrocnemius", P, 0.6),
             mu("soleus", S, 0.5), mu("adductors", S, 0.5), mu("quadriceps", S, 0.4),
             mu("trapezius_upper", S, 0.4), mu("hip_flexors", S, 0.3), mu("rectus_abdominis", ST, 0.3)),
    errors=("Landing flat-footed.", "Arms not reaching overhead."),
    physio_notes=("A whole-body warm-up; halve the impact by stepping instead of hopping.",),
    sources=(ACE,), camera="front", default_reps=6, tags=("cardio", "no equipment"),
)

_CLIMB_ARMS = arms(flex=90, abduct=10, elbow=0)

mountain_climber = ExerciseDefinition(
    id="mountain_climber", name="Mountain climber", category=Category.CONDITIONING,
    description="From a push-up position the knees drive alternately toward the chest while "
                "the shoulders stay over the hands.",
    setup=("Hands under the shoulders, body straight", "Hips level with the shoulders"),
    orientation="prone", anchor="hands", base_position=(-85.0, 30.0, 0.0),
    phases=(
        ph("Right knee in", CON, 0.3, merge(pose(hip_r_flex=110, knee_r_flex=105, ankle_flex=45), _CLIMB_ARMS),
           cues=("Drive the knee to the chest; hips stay down",)),
        ph("Switch", CON, 0.3, merge(pose(hip_l_flex=110, knee_l_flex=105, ankle_flex=45), _CLIMB_ARMS),
           cues=("Switch legs; shoulders over the wrists",)),
    ),
    muscles=(mu("hip_flexors", P, 0.85), mu("rectus_abdominis", P, 0.7), mu("obliques", S, 0.5),
             mu("quadriceps", S, 0.5), mu("deltoid_anterior", S, 0.5), mu("pectoralis_major", S, 0.4),
             mu("serratus_anterior", S, 0.5), mu("triceps_brachii", S, 0.4),
             mu("gluteus_maximus", ST, 0.3)),
    equipment=(eq("mat", attach="static"),),
    errors=("Hips rising into a pike.", "Bouncing the feet without full knee drive."),
    physio_notes=("Combines plank stability with hip flexion; a high-heart-rate core "
                  "exercise.",),
    sources=(ACE, NEUMANN), camera="three_quarter", default_reps=6, tags=("cardio", "no equipment"),
)

battle_ropes = ExerciseDefinition(
    id="battle_ropes", name="Battle ropes (alternating waves)", category=Category.CONDITIONING,
    description="In an athletic stance, the arms whip alternately to send waves down two "
                "heavy ropes.",
    setup=("Quarter squat, feet shoulder-width, chest up", "Rope ends at the hips, elbows "
           "soft"),
    phases=(
        ph("Right up", CON, 0.25, merge(pose(hip_flex=40, knee_flex=40, ankle_flex=0),
                                        arms(flex=75, elbow=40, side="r"), arms(flex=15, elbow=40, side="l"), grip()),
           pitch=15, cues=("Whip from the shoulder; stay low",)),
        ph("Left up", CON, 0.25, merge(pose(hip_flex=40, knee_flex=40, ankle_flex=0),
                                       arms(flex=15, elbow=40, side="r"), arms(flex=75, elbow=40, side="l"), grip()),
           pitch=15),
    ),
    muscles=(mu("deltoid_anterior", P, 0.8), mu("deltoid_lateral", S, 0.5), mu("latissimus_dorsi", S, 0.5),
             mu("triceps_brachii", S, 0.4), mu("biceps_brachii", S, 0.4), mu("forearm_flexors", S, 0.6),
             mu("rectus_abdominis", S, 0.5), mu("obliques", S, 0.5), mu("erector_spinae", ST, 0.4),
             mu("quadriceps", ST, 0.4), mu("gluteus_maximus", ST, 0.35)),
    errors=("Standing up tall.", "Using only the forearms."),
    physio_notes=("Upper-body conditioning with low joint loading; the ropes are not drawn.",),
    sources=(ACE,), camera="three_quarter", default_reps=6, tags=("cardio", "rope"),
)

EXERCISES = (stationary_bike, rowing_machine, treadmill_walk, treadmill_run, jump_rope,
             jumping_jack, mountain_climber, battle_ropes)
