"""Every muscle group resolves to mesh names that exist in the muscle configs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from faceforge.constants import MUSCLE_CONFIG_DIR
from faceforge.exercise.muscle_groups import (
    GROUP_LABELS, GROUP_REGION, MUSCLE_GROUPS, all_group_names, expand_group, group_label,
    regions_for_groups,
)


@pytest.fixture(scope="module")
def config_names() -> dict[str, set[str]]:
    """region -> mesh names, read from the shipped configs (no STL needed)."""
    out: dict[str, set[str]] = {}
    for path in Path(MUSCLE_CONFIG_DIR).glob("*_muscles.json"):
        out[path.stem] = {d["name"] for d in json.loads(path.read_text())}
    return out


def test_every_group_name_exists_in_a_config(config_names):
    everything = set().union(*config_names.values())
    missing = [(g, n) for g in MUSCLE_GROUPS for n in expand_group(g) if n not in everything]
    assert missing == [], f"group members not in any muscle config: {missing}"


def test_every_group_has_a_region_and_a_label():
    assert set(GROUP_REGION) == all_group_names()
    assert set(GROUP_LABELS) == all_group_names()
    assert group_label("quadriceps") == "Quadriceps"


def test_region_actually_contains_the_group(config_names):
    for group, region in GROUP_REGION.items():
        names = config_names[region]
        present = [n for n in expand_group(group) if n in names]
        assert present, f"{group} claims region {region} but none of its muscles are there"


def test_expand_group_by_side():
    right = expand_group("quadriceps", "R")
    assert right == ["Rectus Femoris R", "Vastus Lat. R", "Vastus Med. R", "Vastus Inter. R"]
    both = expand_group("quadriceps")
    assert len(both) == 8 and "Vastus Med. L" in both


def test_regions_for_groups_is_in_load_order_and_complete():
    regions = regions_for_groups(["quadriceps", "gluteus_maximus", "biceps_brachii"])
    assert regions == ["arm_muscles", "hip_muscles", "leg_muscles"]
    assert regions_for_groups(["tensor_fasciae_latae"]) == ["hip_muscles", "leg_muscles"]
