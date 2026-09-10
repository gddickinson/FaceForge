"""Implied stabilisers: grip, carry, brace and stance follow from the definition."""

from __future__ import annotations

import pytest

from faceforge.exercise.catalog import get_exercise_catalog
from faceforge.exercise.clip_builder import build_exercise_clip
from faceforge.exercise.model import Role
from faceforge.exercise.muscle_groups import expand_group, regions_for_groups
from faceforge.exercise.stabilisers import (
    implied_levels, implied_stabilisers, with_implied_stabilisers,
)


@pytest.fixture(scope="module")
def catalog():
    return get_exercise_catalog()


def test_a_deadlift_works_the_hands_arms_and_back(catalog):
    levels = implied_levels(catalog["conventional_deadlift"])
    for group in ("hand_intrinsics", "forearm_flexors", "forearm_extensors", "biceps_brachii",
                  "trapezius_upper", "deltoid_lateral", "rotator_cuff", "erector_spinae",
                  "transversus_abdominis", "foot_intrinsics"):
        assert group in levels, group
    assert levels["forearm_flexors"][0] >= 0.5
    assert levels["erector_spinae"][0] == pytest.approx(0.45)   # loaded brace beats the stance
    assert "holding a barbell" in levels["hand_intrinsics"][1]


def test_a_bodyweight_squat_braces_and_stands_but_does_not_grip(catalog):
    levels = implied_levels(catalog["bodyweight_squat"])
    assert "hand_intrinsics" not in levels and "forearm_flexors" not in levels
    assert levels["erector_spinae"][0] == pytest.approx(0.3)
    assert "foot_intrinsics" in levels and "gluteus_medius" in levels


def test_hanging_from_the_bar_grips_hardest(catalog):
    levels = implied_levels(catalog["pull_up"])
    assert levels["hand_intrinsics"][0] == pytest.approx(0.6)
    assert levels["serratus_anterior"][0] == pytest.approx(0.3)
    assert "erector_spinae" not in levels, "a hanging body is not standing"


def test_a_bar_on_the_back_loads_the_upper_back(catalog):
    levels = implied_levels(catalog["barbell_back_squat"])
    assert levels["erector_spinae"][0] == pytest.approx(0.5)
    assert levels["trapezius_upper"][0] == pytest.approx(0.4)


def test_authored_groups_are_never_overridden(catalog):
    defn = catalog["conventional_deadlift"]
    listed = {u.group: u for u in defn.muscles}
    extra = implied_stabilisers(defn)
    assert all(u.group not in listed for u in extra)
    assert all(u.role is Role.STABILISER and u.note.startswith("implied:") for u in extra)
    assert all(0.2 <= u.level <= 0.6 for u in extra)
    merged = with_implied_stabilisers(defn)
    assert merged.muscles[:len(defn.muscles)] == defn.muscles
    assert merged.id == defn.id and merged is not defn


def test_the_built_clip_colours_the_implied_muscles(catalog):
    built = build_exercise_clip(catalog["conventional_deadlift"], reps=1)
    names = set(built.activation.muscle_names)
    assert "R Lumbricals" in names and "Flex. Dig. Prof. R" in names
    mid = built.activation.sample(built.duration * 0.3)
    assert mid["R Lumbricals"] > 0.4
    groups = built.activation.sample_groups(built.definition, built.duration * 0.3)
    assert groups["hand_intrinsics"] > 0.4


def test_hand_and_foot_groups_expand_with_the_side_first():
    assert expand_group("hand_intrinsics", "R")[:2] == ["R Abductor Pollicis Brevis", "R Opponens Pollicis"]
    assert expand_group("foot_intrinsics")[:2] == ["R Abductor Hallucis", "L Abductor Hallucis"]
    assert regions_for_groups(["hand_intrinsics", "quadriceps"]) == ["leg_muscles", "hand_muscles"]
