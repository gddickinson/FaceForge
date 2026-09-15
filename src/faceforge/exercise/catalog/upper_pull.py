"""Vertical pulling and the arm work: pull-ups, the pulldown, curls, rotator cuff.

The horizontal pulls -- every row, the face pull and the reverse fly -- are in
:mod:`faceforge.exercise.catalog.rows`.

Pull-ups and chin-ups are initiated by the lower trapezius and pectoralis
major and completed by biceps brachii and latissimus dorsi; chin-up EMG
reaches 117-130 % of the MVIC reference for latissimus, 78-96 % for biceps
and 45-56 % for lower trapezius (PULLUP_YOUDAS).
"""

from __future__ import annotations

from faceforge.exercise.catalog._helpers import (
    ACE, ANDERSEN, BOTTON, CON, ECC, EXRX, ISO, LATPULL_HD, NEUMANN, NSCA, P, PULLUP_YOUDAS,
    REINOLD, S, SCHOENFELD_ROW, SEATED_ON_BENCH, SIGNORILE, ST, TRN, arms, eq, grip, hinge,
    merge, mu, only, ph, pose,
)
from faceforge.exercise.model import Category, ExerciseDefinition

_HANG_LEGS = only(hip_flex=10, knee_flex=30)


def _vertical_pull(id_, name, description, bottom_arms, top_arms, muscles, sources, setup,
                   errors, notes, tags):
    bottom = merge(pose(), _HANG_LEGS, bottom_arms, grip())
    top = merge(pose(), _HANG_LEGS, top_arms, grip())
    return ExerciseDefinition(
        id=id_, name=name, category=Category.UPPER_PULL, description=description, setup=setup,
        orientation="hanging", anchor="hands", anchor_point=(0.0, 275.0, 0.0),
        phases=(
            ph("Pull", CON, 1.5, top, cues=("Depress the shoulder blades first, then drive "
                                            "the elbows down and back",
                                            "Chin over the bar, chest to the bar")),
            ph("Top", ISO, 0.4, top, cues=("Hold; shoulders down away from the ears",)),
            ph("Lower", ECC, 2.0, bottom, cues=("Lower under control to straight arms",)),
            ph("Dead hang", ISO, 0.5, bottom, cues=("Shoulders relaxed up; re-brace",)),
        ),
        muscles=muscles, equipment=(eq("pullup_bar", attach="static", height=275.0),),
        errors=errors, physio_notes=notes, sources=sources, camera="three_quarter",
        camera_target=(0.0, 190.0, 0.0), tags=tags,
    )


pull_up = _vertical_pull(
    "pull_up", "Pull-up (overhand)",
    "Hanging from a bar with a pronated grip wider than the shoulders, the body is pulled "
    "up until the chin clears the bar.",
    # Hand DOFs measured on the rig (wrist-frame probe, 2026-09-10): with the
    # humerus externally rotated and the forearm pronated the finger flexion
    # axis lies along the bar (10 deg off at the hang, 0 at the top) and the
    # palm faces forward (10 / 0 deg), so the fingers hook over the bar.
    bottom_arms=arms(flex=10, abduct=165, rotate=90, elbow=5, forearm=-90),
    top_arms=arms(flex=0, abduct=40, rotate=90, elbow=145, forearm=-90),
    muscles=(mu("latissimus_dorsi", P, 1.0), mu("biceps_brachii", P, 0.8),
             mu("brachialis", S, 0.7), mu("brachioradialis", S, 0.55),
             mu("trapezius_lower", S, 0.55, note="initiates by depressing the scapula"),
             mu("rhomboids", S, 0.5), mu("trapezius_middle", S, 0.5),
             mu("deltoid_posterior", S, 0.5), mu("infraspinatus_teres_minor", S, 0.5),
             mu("pectoralis_major", S, 0.4, note="sternal fibres adduct"),
             mu("forearm_flexors", ST, 0.75, note="grip"), mu("rectus_abdominis", ST, 0.4),
             mu("obliques", ST, 0.3)),
    sources=(PULLUP_YOUDAS, LATPULL_HD, NSCA),
    setup=("Overhand grip 1.25-1.5x shoulder width", "Full hang, shoulders active (slightly "
           "depressed)", "Legs straight or knees bent, ribs down"),
    errors=("Kipping / swinging.", "Half reps that stop before the chin clears the bar.",
            "Shrugging the shoulders to the ears at the top."),
    notes=("Pronated grip recruits more lower trapezius than the supinated grip.",
           "Grip strength often limits the set before the lats do."),
    tags=("bodyweight", "bar"),
)

chin_up = _vertical_pull(
    "chin_up", "Chin-up (underhand)",
    "A shoulder-width supinated grip: the same pull with more biceps and pectoralis "
    "contribution.",
    bottom_arms=arms(flex=30, abduct=150, elbow=5, forearm=70),
    top_arms=arms(flex=40, abduct=30, elbow=135, forearm=70),
    muscles=(mu("biceps_brachii", P, 0.95), mu("latissimus_dorsi", P, 0.95),
             mu("brachialis", S, 0.7), mu("pectoralis_major", S, 0.5),
             mu("trapezius_lower", S, 0.5), mu("rhomboids", S, 0.45),
             mu("deltoid_posterior", S, 0.45), mu("infraspinatus_teres_minor", S, 0.45),
             mu("forearm_flexors", ST, 0.7), mu("rectus_abdominis", ST, 0.4)),
    sources=(PULLUP_YOUDAS, NSCA),
    setup=("Underhand grip at shoulder width", "Full hang, shoulders active"),
    errors=("Swinging the legs.", "Stopping short of the bar."),
    notes=("Supinated grip: biceps 78-96 %MVIC and greater pectoralis major activation than "
           "the pull-up.",),
    tags=("bodyweight", "bar"),
)

_PULLDOWN_SEAT = only(hip_flex=90, knee_flex=90)

lat_pulldown = ExerciseDefinition(
    id="lat_pulldown", name="Lat pulldown (to the front)", category=Category.UPPER_PULL,
    description="Seated, the bar is pulled from overhead to the upper chest with a slight "
                "lean back; the elbows drive down and in.",
    setup=("Thighs under the pads, feet flat", "Wide overhand grip", "Chest up, slight lean "
           "back (~10 deg), shoulders down"),
    orientation="seated", anchor="none", base_position=SEATED_ON_BENCH,
    phases=(
        ph("Pull", CON, 1.3,
           merge(pose(), _PULLDOWN_SEAT, arms(flex=30, abduct=40, elbow=120, forearm=-60), grip()),
           pitch=-10, cues=("Bar to the upper chest; elbows down and slightly back",)),
        ph("Squeeze", ISO, 0.4,
           merge(pose(), _PULLDOWN_SEAT, arms(flex=30, abduct=40, elbow=120, forearm=-60), grip()),
           pitch=-10, cues=("Shoulder blades together and down",)),
        ph("Return", ECC, 2.0,
           merge(pose(), _PULLDOWN_SEAT, arms(flex=30, abduct=150, elbow=5, forearm=-60), grip()),
           pitch=-10, cues=("Let the arms straighten fully; shoulders rise slightly",)),
    ),
    muscles=(mu("latissimus_dorsi", P, 0.95), mu("biceps_brachii", S, 0.7),
             mu("brachialis", S, 0.6), mu("trapezius_middle", S, 0.5),
             mu("trapezius_lower", S, 0.5), mu("rhomboids", S, 0.5),
             mu("deltoid_posterior", S, 0.5), mu("pectoralis_major", S, 0.3),
             mu("infraspinatus_teres_minor", S, 0.4), mu("forearm_flexors", ST, 0.6),
             mu("erector_spinae", ST, 0.3)),
    equipment=(eq("barbell", length=110.0, plates=0, bar_radius=1.6),
               eq("bench", attach="static", height=58.0, length=60.0)),
    errors=("Pulling behind the neck (impingement risk, no advantage).",
            "Leaning far back and rowing with body weight.", "Grip much wider than 1.5x "
            "shoulders (shorter range, more strain)."),
    physio_notes=("Front pulldown: greater latissimus activation in the eccentric than the "
                  "behind-the-neck version; no benefit from a very wide grip (Andersen 2014).",),
    sources=(LATPULL_HD, SIGNORILE, ANDERSEN, NSCA), camera="three_quarter", tags=("cable",),
)






barbell_biceps_curl = ExerciseDefinition(
    id="barbell_biceps_curl", name="Barbell biceps curl", category=Category.UPPER_PULL,
    description="Standing with the upper arms still, the elbows flex to bring the bar to the "
                "shoulders with the forearms supinated.",
    setup=("Shoulder-width underhand grip", "Elbows at the sides, slightly in front",
           "Stand tall, knees soft, ribs down"),
    phases=(
        ph("Curl", CON, 1.2, merge(pose(knee_flex=5), arms(flex=15, elbow=135, forearm=80), grip()),
           cues=("Curl to the shoulders; elbows stay at the sides",)),
        ph("Top", ISO, 0.4, merge(pose(knee_flex=5), arms(flex=15, elbow=135, forearm=80), grip()),
           cues=("Squeeze; a little shoulder flexion is fine",)),
        ph("Lower", ECC, 2.0, merge(pose(knee_flex=5), arms(flex=3, elbow=5, forearm=80), grip()),
           cues=("Lower slowly to full extension",)),
    ),
    muscles=(mu("biceps_brachii", P, 0.95), mu("brachialis", P, 0.85),
             mu("brachioradialis", S, 0.55), mu("forearm_flexors", S, 0.5),
             mu("deltoid_anterior", ST, 0.3), mu("rectus_abdominis", ST, 0.2),
             mu("erector_spinae", ST, 0.25)),
    equipment=(eq("barbell", length=120.0, plates=1, plate_radius=12.0),),
    errors=("Swinging the trunk / hip drive.", "Elbows drifting forward at the top.",
            "Wrists curling (forearm flexors take over)."),
    physio_notes=("Supinated curls maximise biceps brachii; the brachialis works in every "
                  "forearm position (Marcolin 2018).",),
    sources=(BOTTON, NEUMANN, EXRX), camera="side", tags=("barbell",),
)

hammer_curl = ExerciseDefinition(
    id="hammer_curl", name="Dumbbell hammer curl", category=Category.UPPER_PULL,
    description="Curls with a neutral (thumbs-up) grip, biasing brachialis and "
                "brachioradialis.",
    setup=("Dumbbells at the sides, palms facing the thighs", "Elbows pinned at the sides"),
    phases=(
        ph("Curl", CON, 1.2, merge(pose(knee_flex=5), arms(flex=12, elbow=135, forearm=0), grip()),
           cues=("Curl to the shoulders keeping the thumbs up",)),
        ph("Top", ISO, 0.4, merge(pose(knee_flex=5), arms(flex=12, elbow=135, forearm=0), grip())),
        ph("Lower", ECC, 2.0, merge(pose(knee_flex=5), arms(flex=3, elbow=5, forearm=0), grip()),
           cues=("Full extension at the bottom",)),
    ),
    muscles=(mu("brachialis", P, 0.9), mu("brachioradialis", P, 0.85),
             mu("biceps_brachii", S, 0.7), mu("forearm_flexors", S, 0.45),
             mu("forearm_extensors", S, 0.35, note="wrist stabilisers"),
             mu("deltoid_anterior", ST, 0.25)),
    equipment=(eq("dumbbell", attach="hand_r"), eq("dumbbell", attach="hand_l")),
    errors=("Swinging.", "Shoulder creeping forward."),
    physio_notes=("Neutral grip reduces load on the distal biceps tendon compared with "
                  "supinated curls; useful in distal biceps tendinopathy.",),
    sources=(BOTTON, NEUMANN), camera="side", tags=("dumbbell",),
)


band_external_rotation = ExerciseDefinition(
    id="band_external_rotation", name="Band external rotation (elbow at the side)",
    category=Category.UPPER_PULL,
    description="With the elbow bent 90 deg and held at the side, the forearm rotates "
                "outward against a band.",
    setup=("Elbow at 90 deg, tucked to the side (a towel under it)", "Band anchored at "
           "elbow height on the opposite side", "Wrist neutral"),
    phases=(
        ph("Rotate out", CON, 1.2,
           merge(pose(), arms(flex=0, abduct=5, rotate=-40, elbow=90, side="l"),
                 arms(flex=0, abduct=5, rotate=45, elbow=90, forearm=0, side="r"), grip()),
           cues=("Rotate the forearm outward as far as comfortable; elbow stays at the side",)),
        ph("Hold", ISO, 0.6,
           merge(pose(), arms(flex=0, abduct=5, rotate=-40, elbow=90, side="l"),
                 arms(flex=0, abduct=5, rotate=45, elbow=90, forearm=0, side="r"), grip()),
           cues=("Hold; shoulder blade back and down",)),
        ph("Return", ECC, 2.0,
           merge(pose(), arms(flex=0, abduct=5, rotate=-40, elbow=90, side="l"),
                 arms(flex=0, abduct=5, rotate=-40, elbow=90, forearm=0, side="r"), grip()),
           cues=("Return slowly across the abdomen",)),
    ),
    muscles=(mu("infraspinatus_teres_minor", P, 0.7, side="R"),
             mu("rotator_cuff", S, 0.5, side="R", note="supraspinatus/subscapularis co-contract"),
             mu("deltoid_posterior", S, 0.3, side="R"), mu("trapezius_lower", ST, 0.3, side="R"),
             mu("rhomboids", ST, 0.3, side="R")),
    equipment=(eq("band", attach="hand_r", length=50.0),),
    errors=("Elbow drifting away from the side.", "Using the trunk to rotate.",
            "Too much band tension (deltoid takes over)."),
    physio_notes=("Side-lying and elbow-at-side external rotation produce the highest "
                  "infraspinatus and teres minor activation with low deltoid contribution "
                  "(Reinold 2004).",),
    sources=(REINOLD, NEUMANN), unilateral=True, camera="front", tags=("band", "rehab"),
)

wide_grip_pull_up = _vertical_pull(
    "wide_grip_pull_up", "Wide-grip pull-up",
    "A pull-up taken well outside the shoulders. The wider grip shortens the range and "
    "shifts the pull toward scapular adduction, which is why it is felt across the upper "
    "back rather than down the lat.",
    bottom_arms=arms(flex=10, abduct=172, rotate=90, elbow=5, forearm=-90),
    top_arms=arms(flex=0, abduct=60, rotate=90, elbow=125, forearm=-90),
    muscles=(mu("latissimus_dorsi", P, 0.95), mu("trapezius_middle", P, 0.7),
             mu("rhomboids", P, 0.65), mu("biceps_brachii", S, 0.6),
             mu("trapezius_lower", S, 0.6), mu("deltoid_posterior", S, 0.55),
             mu("infraspinatus_teres_minor", S, 0.5), mu("brachialis", S, 0.5),
             mu("pectoralis_major", S, 0.35),
             mu("forearm_flexors", ST, 0.8, note="a wide grip is harder to hold"),
             mu("rectus_abdominis", ST, 0.4), mu("obliques", ST, 0.3)),
    sources=(PULLUP_YOUDAS, ANDERSEN, LATPULL_HD, NSCA),
    setup=("Grip well outside the shoulders, palms forward",
           "Chest up, shoulders down before the first inch",
           "Expect fewer reps than a standard grip: the range is shorter but the leverage is worse"),
    errors=("Going so wide the elbows cannot finish flexing.",
            "Shrugging into the top instead of depressing the blades.",
            "Kipping to make up for the leverage."),
    notes=("A wider grip reduces elbow flexion range and increases the shoulder adduction "
           "component, so the mid-trapezius and rhomboids take more of it than in a "
           "shoulder-width pull-up.",),
    tags=("bodyweight", "vertical-pull"),
)

neutral_grip_pull_up = _vertical_pull(
    "neutral_grip_pull_up", "Neutral-grip pull-up",
    "Palms facing each other on parallel handles. The neutral forearm is the position most "
    "shoulders and elbows tolerate best, and it lets brachialis and brachioradialis "
    "contribute more than a pronated grip does.",
    bottom_arms=arms(flex=10, abduct=150, rotate=45, elbow=5, forearm=0),
    top_arms=arms(flex=0, abduct=30, rotate=45, elbow=145, forearm=0),
    muscles=(mu("latissimus_dorsi", P, 0.95), mu("biceps_brachii", P, 0.75),
             mu("brachialis", P, 0.75, note="the neutral forearm's muscle"),
             mu("brachioradialis", S, 0.7), mu("trapezius_lower", S, 0.55),
             mu("rhomboids", S, 0.5), mu("trapezius_middle", S, 0.5),
             mu("deltoid_posterior", S, 0.45), mu("infraspinatus_teres_minor", S, 0.45),
             mu("forearm_flexors", ST, 0.75), mu("rectus_abdominis", ST, 0.4),
             mu("obliques", ST, 0.3)),
    sources=(PULLUP_YOUDAS, BOTTON, NSCA),
    setup=("Parallel handles, palms facing each other", "Shoulders down and back before pulling",
           "Drive the elbows down toward the ribs"),
    errors=("Letting the elbows flare forward, which turns it into a row.",
            "Half reps: the neutral grip makes the top easier to fake."),
    notes=("A neutral forearm puts brachialis and brachioradialis in their strongest "
           "positions, which is why most people can do more of these than pronated "
           "pull-ups.",),
    tags=("bodyweight", "vertical-pull"),
)




EXERCISES = (pull_up, chin_up, wide_grip_pull_up, neutral_grip_pull_up, lat_pulldown,
             barbell_biceps_curl, hammer_curl, band_external_rotation)
