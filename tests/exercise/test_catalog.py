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
