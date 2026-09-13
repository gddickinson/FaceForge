"""The measurements, and what they are compared against.

This tests the arithmetic, not the asset set: the model's own numbers need the
STL set and live in the slow tier.  What is locked down here is that the
published references are the ones the morph config quotes, and that the
measurements are taken the way an anthropometrist takes them.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.anthropometry import PUBLISHED, RATIOS, measure, span


def box(centre, size) -> np.ndarray:
    c = np.asarray(centre, dtype=np.float64)
    s = np.asarray(size, dtype=np.float64) / 2.0
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1)
                        for z in (-1, 1)], dtype=np.float64)
    return c + corners * s


@pytest.fixture
def bones():
    return {
        "cranium": box((0, 0, 90), (15, 20, 20)),
        "jaw": box((0, -6, 78), (11, 12, 8)),
        "Right Hip Bone": box((10, 0, 10), (14, 16, 22)),
        "Left Hip Bone": box((-10, 0, 10), (14, 16, 22)),
        "Right Scapula": box((16, 6, 66), (10, 4, 14)),
        "Left Scapula": box((-16, 6, 66), (10, 4, 14)),
        "Right Femur": box((8, 0, -20), (6, 6, 44)),
    }


@pytest.fixture
def joints():
    return {"shoulder_R": np.array([18.0, 0.0, 66.0]),
            "elbow_R": np.array([20.0, 0.0, 36.0]),
            "hip_R": np.array([10.0, 0.0, 4.0]),
            "knee_R": np.array([9.0, 0.0, -40.0]),
            "ankle_R": np.array([9.0, 0.0, -78.0])}


def test_every_published_reference_is_a_pair_of_adult_means():
    for name, entry in PUBLISHED.items():
        male, female, unit = entry
        assert male > 0 and female > 0, name
        assert female < male, f"{name}: the female reference should be smaller"
        assert unit == "cm"


def test_the_stature_reference_is_the_one_the_morph_config_quotes():
    male, female, _ = PUBLISHED["stature"]
    assert (male, female) == (175.6, 162.9)
    assert female / male == pytest.approx(0.928, abs=0.001)


def test_stature_is_the_whole_skeleton_end_to_end(bones, joints):
    m = measure(bones, joints)
    assert m["stature"] == pytest.approx(100.0 - (-42.0))


def test_sitting_height_is_the_crown_to_the_ischium(bones, joints):
    """Not to the floor: a seated body rests on the ischial tuberosities."""
    m = measure(bones, joints)
    assert m["sitting height"] == pytest.approx(100.0 - (-1.0))
    assert m["sitting height"] < m["stature"]


def test_the_breadths_are_measured_across_both_sides(bones, joints):
    m = measure(bones, joints)
    assert m["bi-iliac breadth"] == pytest.approx(34.0)
    assert m["biacromial breadth"] == pytest.approx(42.0)


def test_long_bones_are_measured_joint_to_joint(bones, joints):
    m = measure(bones, joints)
    assert m["humerus length"] == pytest.approx(np.hypot(2.0, 30.0))
    assert m["femur length"] == pytest.approx(np.hypot(1.0, 44.0))
    assert m["tibia length"] == pytest.approx(38.0)


def test_the_face_is_measured_on_the_lower_skull(bones, joints):
    """Bizygomatic breadth is across the cheekbones, not the vault."""
    m = measure(bones, joints)
    assert m["bizygomatic breadth"] <= m["head breadth"]
    assert m["bigonial breadth"] == pytest.approx(11.0)


def test_a_missing_bone_is_simply_not_reported(joints):
    m = measure({"cranium": box((0, 0, 0), (2, 2, 2))}, {})
    assert "femur length" not in m
    assert "bi-iliac breadth" not in m
    assert m["head breadth"] == pytest.approx(2.0)


def test_every_proportion_names_measurements_that_exist():
    for _label, num, den, male, female in RATIOS:
        assert num in PUBLISHED and den in PUBLISHED
        assert 0.0 < male < 2.0 and 0.0 < female < 2.0


def test_span_is_an_extent_not_a_centroid():
    pts = np.array([[-3.0, 0, 0], [5.0, 0, 0]])
    assert span(pts, 0) == pytest.approx(8.0)
