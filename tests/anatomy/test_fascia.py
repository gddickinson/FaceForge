"""Unit tests for the FasciaSystem module."""

from __future__ import annotations

import numpy as np
import pytest

from faceforge.anatomy.fascia import (
    FasciaRegion,
    FasciaSystem,
    build_anatomical_fascia_regions,
)
from faceforge.anatomy.bone_anchors import BoneAnchorRegistry
from faceforge.core.scene_graph import SceneNode


# ── Helpers ───────────────────────────────────────────────────────────

def _make_bone_node(name: str, position: tuple[float, float, float]) -> SceneNode:
    """Create a SceneNode at a fixed world position."""
    node = SceneNode(name=name)
    node.set_position(*position)
    node.update_world_matrix(force=True)
    return node


def _make_registry(
    bone_positions: dict[str, tuple[float, float, float]],
) -> BoneAnchorRegistry:
    """Create a BoneAnchorRegistry with the given bone positions."""
    registry = BoneAnchorRegistry()
    nodes = {}
    for name, pos in bone_positions.items():
        nodes[name] = _make_bone_node(name, pos)
    registry.register_bones(nodes)
    registry.snapshot_rest_positions()
    return registry


# ── FasciaRegion tests ────────────────────────────────────────────────

class TestFasciaRegion:
    """Validate FasciaRegion dataclass construction."""

    def test_basic_construction(self):
        r = FasciaRegion(
            name="test",
            bone_names=["Bone A", "Bone B"],
            bone_weights=[1.0, 2.0],
            side="R",
        )
        assert r.name == "test"
        assert len(r.bone_names) == 2
        assert r.side == "R"

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            FasciaRegion(
                name="bad",
                bone_names=["A", "B"],
                bone_weights=[1.0],
            )

    def test_default_side_is_midline(self):
        r = FasciaRegion(name="mid", bone_names=[], bone_weights=[])
        assert r.side == "M"


# ── FasciaSystem init tests ──────────────────────────────────────────

class TestFasciaSystemInit:
    """FasciaSystem should initialise with mock registry."""

    def test_region_names(self):
        regions = [
            FasciaRegion(name="alpha", bone_names=["A"], bone_weights=[1.0]),
            FasciaRegion(name="beta", bone_names=["B"], bone_weights=[1.0]),
        ]
        registry = _make_registry({"A": (1, 0, 0), "B": (0, 1, 0)})
        fs = FasciaSystem(regions, registry)
        assert set(fs.region_names) == {"alpha", "beta"}


# ── Rest target tests ────────────────────────────────────────────────

class TestFasciaSnapshotRest:
    """Verify rest targets match weighted bone centroids."""

    def test_single_bone_region(self):
        registry = _make_registry({"BoneX": (10.0, 20.0, 30.0)})
        region = FasciaRegion(
            name="single", bone_names=["BoneX"], bone_weights=[1.0],
        )
        fs = FasciaSystem([region], registry)
        fs.snapshot_rest()
        target = fs.get_target_rest("single")
        assert target is not None
        np.testing.assert_allclose(target, [10.0, 20.0, 30.0], atol=1e-6)

    def test_uniform_weights(self):
        registry = _make_registry({
            "A": (0.0, 0.0, 0.0),
            "B": (10.0, 0.0, 0.0),
        })
        region = FasciaRegion(
            name="uni", bone_names=["A", "B"], bone_weights=[1.0, 1.0],
        )
        fs = FasciaSystem([region], registry)
        fs.snapshot_rest()
        target = fs.get_target_rest("uni")
        assert target is not None
        np.testing.assert_allclose(target, [5.0, 0.0, 0.0], atol=1e-6)

    def test_non_uniform_weights(self):
        registry = _make_registry({
            "A": (0.0, 0.0, 0.0),
            "B": (10.0, 0.0, 0.0),
        })
        region = FasciaRegion(
            name="weighted", bone_names=["A", "B"], bone_weights=[3.0, 1.0],
        )
        fs = FasciaSystem([region], registry)
        fs.snapshot_rest()
        target = fs.get_target_rest("weighted")
        assert target is not None
        # 3/4 * 0 + 1/4 * 10 = 2.5
        np.testing.assert_allclose(target, [2.5, 0.0, 0.0], atol=1e-6)


class TestFasciaWeightedCentroid:
    """Further weighted centroid edge cases."""

    def test_three_bones(self):
        registry = _make_registry({
            "X": (0.0, 0.0, 0.0),
            "Y": (6.0, 0.0, 0.0),
            "Z": (0.0, 6.0, 0.0),
        })
        region = FasciaRegion(
            name="tri", bone_names=["X", "Y", "Z"],
            bone_weights=[1.0, 1.0, 1.0],
        )
        fs = FasciaSystem([region], registry)
        fs.snapshot_rest()
        target = fs.get_target_rest("tri")
        assert target is not None
        np.testing.assert_allclose(target, [2.0, 2.0, 0.0], atol=1e-6)


# ── Current-frame target tests ───────────────────────────────────────

class TestFasciaTargetCurrent:
    """Moving bones should update current-frame targets."""

    def test_moved_bone(self):
        node_a = _make_bone_node("A", (0.0, 0.0, 0.0))
        node_b = _make_bone_node("B", (10.0, 0.0, 0.0))
        registry = BoneAnchorRegistry()
        registry.register_bones({"A": node_a, "B": node_b})
        registry.snapshot_rest_positions()

        region = FasciaRegion(
            name="test", bone_names=["A", "B"], bone_weights=[1.0, 1.0],
        )
        fs = FasciaSystem([region], registry)
        fs.snapshot_rest()

        # Rest target should be (5, 0, 0)
        rest = fs.get_target_rest("test")
        np.testing.assert_allclose(rest, [5.0, 0.0, 0.0], atol=1e-6)

        # Move bone A
        node_a.set_position(4.0, 0.0, 0.0)
        node_a.update_world_matrix(force=True)
        current = fs.get_target_current("test")
        assert current is not None
        # (4 + 10) / 2 = 7
        np.testing.assert_allclose(current, [7.0, 0.0, 0.0], atol=1e-6)


# ── Missing bones test ───────────────────────────────────────────────

class TestFasciaTargetMissingBones:
    """Gracefully handle missing bones."""

    def test_no_bones_returns_none(self):
        registry = _make_registry({})
        region = FasciaRegion(
            name="missing", bone_names=["NonExistent"], bone_weights=[1.0],
        )
        fs = FasciaSystem([region], registry)
        fs.snapshot_rest()
        assert fs.get_target_rest("missing") is None
        assert fs.get_target_current("missing") is None

    def test_unknown_region_name_returns_none(self):
        registry = _make_registry({"A": (1, 2, 3)})
        fs = FasciaSystem([], registry)
        fs.snapshot_rest()
        assert fs.get_target_rest("nonexistent") is None
        assert fs.get_target_current("nonexistent") is None

    def test_partial_bones(self):
        """Only some bones exist — centroid uses available ones."""
        registry = _make_registry({"A": (10.0, 0.0, 0.0)})
        region = FasciaRegion(
            name="partial",
            bone_names=["A", "Missing"],
            bone_weights=[1.0, 1.0],
        )
        fs = FasciaSystem([region], registry)
        fs.snapshot_rest()
        target = fs.get_target_rest("partial")
        assert target is not None
        # Only bone A found, so centroid = A's position
        np.testing.assert_allclose(target, [10.0, 0.0, 0.0], atol=1e-6)


# ── Factory test ──────────────────────────────────────────────────────

class TestBuildAnatomicalRegions:
    """Factory should return the expected 16 anatomical regions."""

    def test_region_count(self):
        regions = build_anatomical_fascia_regions()
        assert len(regions) == 16

    def test_region_names(self):
        regions = build_anatomical_fascia_regions()
        names = {r.name for r in regions}
        assert names == {
            "pectoral_R", "pectoral_L",
            "deltoid_R", "deltoid_L",
            "investing_R", "investing_L",
            "supraclavicular_R", "supraclavicular_L",
            "trapezius_R", "trapezius_L",
            "scm_sternal_R", "scm_sternal_L",
            "scm_clavicular_R", "scm_clavicular_L",
            "lev_scap_R", "lev_scap_L",
        }

    def test_all_have_bones(self):
        regions = build_anatomical_fascia_regions()
        for r in regions:
            assert len(r.bone_names) > 0
            assert len(r.bone_names) == len(r.bone_weights)

    def test_sides_correct(self):
        regions = build_anatomical_fascia_regions()
        by_name = {r.name: r for r in regions}
        assert by_name["pectoral_R"].side == "R"
        assert by_name["pectoral_L"].side == "L"
        assert by_name["deltoid_R"].side == "R"
        assert by_name["deltoid_L"].side == "L"
        assert by_name["investing_R"].side == "R"
        assert by_name["investing_L"].side == "L"
        assert by_name["supraclavicular_R"].side == "R"
        assert by_name["supraclavicular_L"].side == "L"
        assert by_name["trapezius_R"].side == "R"
        assert by_name["trapezius_L"].side == "L"

    def test_manubrium_name_correct(self):
        """Bone name should be 'Manubrium' (not 'Manubrium of Sternum')."""
        regions = build_anatomical_fascia_regions()
        for r in regions:
            for bone in r.bone_names:
                assert bone != "Manubrium of Sternum", (
                    f"Region '{r.name}' has wrong bone name 'Manubrium of Sternum' "
                    f"— should be 'Manubrium'"
                )
