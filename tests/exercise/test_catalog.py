"""The catalogue is complete, valid and buildable -- without any asset."""

from __future__ import annotations

import pytest

from faceforge.body.body_constraints import BodyConstraints
from faceforge.body.dof_ranges import POSE_DOF_FIELDS
from faceforge.exercise.catalog import exercises_in_category, get_exercise_catalog
from faceforge.exercise.clip_builder import LIFT_KEY, build_exercise_clip
from faceforge.exercise.equipment import known_equipment
from faceforge.exercise.model import Category, ExerciseDefinition, Role, validate_definition
from faceforge.exercise.muscle_groups import all_group_names


@pytest.fixture(scope="module")
def catalog():
    return get_exercise_catalog()


@pytest.fixture(scope="module")
def limits():
    bc = BodyConstraints()
    bc.load()
    return dict(bc._limits)


def test_catalog_has_every_category_and_enough_exercises(catalog):
    assert len(catalog) >= 60
    for cat in Category:
        assert len(exercises_in_category(catalog, cat)) >= 6, cat


def test_every_definition_validates(catalog, limits):
    problems = []
    for defn in catalog.values():
        problems += validate_definition(defn, all_group_names(), limits, known_equipment())
    assert problems == []


def test_every_definition_has_technique_content(catalog):
    for defn in catalog.values():
        assert defn.description and defn.setup, defn.id
        assert defn.errors, f"{defn.id} lists no technique errors"
        assert defn.sources, defn.id
        assert defn.muscles_by_role(Role.PRIMARY), defn.id
        for ph in defn.phases:
            assert ph.duration > 0, (defn.id, ph.name)


def test_every_definition_builds_a_playable_clip(catalog):
    for defn in catalog.values():
        built = build_exercise_clip(defn, reps=2)
        times = [kf.time for kf in built.clip.keyframes]
        assert times == sorted(times) and times[0] == 0.0, defn.id
        assert built.clip.loop
        assert len(built.spans) == 2 * len(defn.phases)
        assert built.duration == pytest.approx(2 * defn.rep_duration / 1.0)
        # Every keyframe carries a full pose plus the lift key.
        for kf in built.clip.keyframes:
            assert set(POSE_DOF_FIELDS) <= set(kf.body_state), defn.id
            assert LIFT_KEY in kf.body_state
        # The activation track is defined over the whole clip and non-trivial:
        # some muscle reaches a high level at some point in the rep.
        peak = max(float(arr.max()) for arr in built.activation.levels.values())
        assert peak > 0.5, defn.id
        assert built.activation.sample(built.duration * 0.4), defn.id


def test_ids_are_unique_and_match_keys(catalog):
    assert all(k == d.id for k, d in catalog.items())
    names = [d.name for d in catalog.values()]
    assert len(names) == len(set(names))


def test_round_trip_through_json_dict(catalog):
    defn = catalog["barbell_back_squat"]
    again = ExerciseDefinition.from_dict(defn.to_dict())
    assert again == defn


def test_squat_bottom_pose_is_deep_and_foot_flat(catalog):
    defn = catalog["bodyweight_squat"]
    bottom = defn.phases[0]
    assert bottom.pitch == 35
    # hip 120, knee 120 -> ankle = 35 - 120 + 120 = 35 deg dorsiflexion = 0.78
    assert bottom.pose["hip_r_flex"] == pytest.approx(120 / 90)
    assert bottom.pose["knee_r_flex"] == pytest.approx(120 / 145)
    assert bottom.pose["ankle_r_flex"] == pytest.approx(35 / 45)


def test_arm_bundles_do_not_reset_the_legs(catalog):
    """A regression guard for merge() used as a partial: the back squat, lunge
    and clamshell bottoms must still flex the hips."""
    squat_bottom = catalog["barbell_back_squat"].phases[0].pose
    assert squat_bottom["hip_r_flex"] > 1.2 and squat_bottom["knee_r_flex"] > 0.8
    lunge_bottom = catalog["forward_lunge"].phases[1].pose
    assert lunge_bottom["hip_r_flex"] > 0.9
    clam = catalog["clamshell"].phases[0].pose
    assert clam["hip_l_flex"] == pytest.approx(45 / 90) and clam["hip_r_abduct"] > 0.5
    rope = catalog["jump_rope"].phases[0].pose
    assert rope["knee_r_flex"] > 0.1


def test_a_lying_body_is_never_filmed_from_the_foot_end(catalog):
    """The gym's ``side`` preset sits at +X and a lying body's long axis IS X.

    A prone or supine exercise with ``camera="side"`` therefore looks straight
    down the body from the feet: the whole catalogue was rendered that way
    before this was noticed.  ``front`` (+Z) is the profile.
    """
    end_on = [d.id for d in catalog.values()
              if d.orientation in ("prone", "supine")
              and d.camera in ("side", "side_left", "low_side")]
    assert end_on == []


def test_a_barbell_is_never_between_the_camera_and_the_lifter(catalog):
    """A loaded bar lies across the lifter, so one axis always looks down it.

    Standing, the bar runs along X and a ``side`` camera frames a plate;
    supine, the lifter turns ninety degrees with it, the bar runs along Z and
    it is ``front`` that looks down the bar (the floor press rendered as two
    black discs over the torso).  Either way the answer is an oblique camera.
    """
    def bad(d):
        if not any(e.kind == "barbell" and e.attach == "hands" for e in d.equipment):
            return False
        blocked = ("front", "front_wide", "low_front", "back") if d.orientation == "supine" \
            else ("side", "side_left", "low_side")
        return d.camera in blocked

    assert [d.id for d in catalog.values() if bad(d)] == []


#: A tag that names an implement, and the equipment kinds that satisfy it.
#: The lat pulldown's bar is modelled as a barbell, which is what it is.
IMPLEMENT_TAGS = {
    "barbell": {"barbell"},
    "dumbbell": {"dumbbell"},
    "kettlebell": {"kettlebell"},
    "band": {"band"},
    "bar": {"pullup_bar", "dip_station", "barbell"},
    "rope": {"jump_rope", "battle_rope"},
    "cable": {"cable_handle", "barbell"},
    "box": {"plyo_box"},
    "medicine ball": {"medicine_ball"},
    "machine": {"bike", "rower", "treadmill", "bench", "cable_handle"},
}

#: Equipment that is a load rather than a surface to lie or stand on.
LOADED_KINDS = {"barbell", "dumbbell", "kettlebell", "medicine_ball", "cable_handle", "band"}


def test_an_exercise_that_names_an_implement_ships_one(catalog):
    """The band walk, the battle ropes and both treadmills shipped nothing.

    Nothing caught it: the definitions validated, built a clip and placed the
    body, and the renders simply showed an athlete miming.
    """
    missing = [(d.id, tag) for d in catalog.values() for tag, kinds in IMPLEMENT_TAGS.items()
               if tag in d.tags and not kinds & set(d.equipment_names)]
    assert missing == []


def test_no_equipment_means_no_equipment(catalog):
    """The other direction: a mat is furniture, a kettlebell is not."""
    carrying = [(d.id, sorted(set(d.equipment_names) & LOADED_KINDS)) for d in catalog.values()
                if "no equipment" in d.tags and set(d.equipment_names) & LOADED_KINDS]
    assert carrying == []


#: How far the heel may lift before the toes have to take over.  Below this it
#: is a rounding difference in the authored angles, not a raised heel.
HEEL_UP_DEGREES = 25.0


def _leg(pose_dict, side):
    from faceforge.body.dof_ranges import dof_to_degrees
    return tuple(dof_to_degrees(f, float(pose_dict.get(f, 0.0))) for f in
                 (f"hip_{side}_flex", f"knee_{side}_flex", f"ankle_{side}_flex",
                  f"toe_curl_{side}"))


def test_a_heel_that_leaves_the_floor_takes_the_toes_with_it(catalog):
    """A rigid foot stands on the point of its longest toe, and goes through it.

    Only symmetric stances on the ground are checked: with the two legs
    authored alike there is no swing leg and no split stance, so any heel that
    is up is a heel the body is standing on -- a calf raise, a jump take-off,
    the second pull of a clean.  A phase with ``lift`` is in the air or on a
    box, where there is no floor under the toes to put down.  The rule itself
    is ``pose_library.toes_on_floor``.
    """
    from faceforge.exercise.pose_library import flat_foot_ankle

    rigid = []
    for defn in catalog.values():
        if defn.orientation != "standing":
            continue
        for phase in defn.phases:
            if phase.lift:
                continue                       # airborne, or standing on a box
            right, left = _leg(phase.pose, "r"), _leg(phase.pose, "l")
            if right != left:
                continue                       # a split stance or a swing leg
            hip, knee, ankle, toe = right
            if flat_foot_ankle(phase.pitch, hip, knee) - ankle > HEEL_UP_DEGREES and toe == 0.0:
                rigid.append((defn.id, phase.name))
    assert rigid == []


def test_a_flat_foot_keeps_its_toes_flat(catalog):
    """The other direction: toes bent under a sole that is on the floor."""
    from faceforge.exercise.pose_library import flat_foot_ankle

    bent = []
    for defn in catalog.values():
        if defn.orientation != "standing":
            continue
        for phase in defn.phases:
            for side in ("r", "l"):
                hip, knee, ankle, toe = _leg(phase.pose, side)
                if abs(flat_foot_ankle(phase.pitch, hip, knee) - ankle) < 5.0 and toe > 5.0:
                    bent.append((defn.id, phase.name, side))
    assert bent == []
