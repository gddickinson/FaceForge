"""Unit tests for BoneAnchorRegistry.

These tests use mock SceneNodes and do NOT require STL assets.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceforge.anatomy.bone_anchors import BoneAnchorRegistry
from faceforge.core.scene_graph import SceneNode


# ── Helpers ───────────────────────────────────────────────────────────

def _make_bone_node(name: str, position: tuple[float, float, float]) -> SceneNode:
    """Create a SceneNode at a given position."""
    node = SceneNode(name=name)
    node.set_position(*position)
    node.mark_dirty()
    node.update_world_matrix(force=True)
    return node


# ── Tests ─────────────────────────────────────────────────────────────

class TestRegisterAndSnapshot:
    """Bones are registered and rest positions are stored."""

    def test_register_bones(self):
        reg = BoneAnchorRegistry()
        node_a = _make_bone_node("Bone A", (1.0, 2.0, 3.0))
        node_b = _make_bone_node("Bone B", (4.0, 5.0, 6.0))

        reg.register_bones({"Bone A": node_a, "Bone B": node_b})

        assert reg.has_bone("Bone A")
        assert reg.has_bone("Bone B")
        assert not reg.has_bone("Bone C")

    def test_snapshot_rest_positions(self):
        reg = BoneAnchorRegistry()
        node = _make_bone_node("Right Clavicle", (10.0, -5.0, 3.0))
        reg.register_bones({"Right Clavicle": node})
        reg.snapshot_rest_positions()

        pos = reg.get_muscle_anchor("SCM R", ["Right Clavicle"])
        assert pos is not None
        np.testing.assert_allclose(pos, [10.0, -5.0, 3.0], atol=1e-6)

    def test_bone_names_property(self):
        reg = BoneAnchorRegistry()
        reg.register_bones({
            "A": _make_bone_node("A", (0, 0, 0)),
            "B": _make_bone_node("B", (1, 1, 1)),
        })
        names = reg.bone_names
        assert sorted(names) == ["A", "B"]


class TestGetMuscleAnchor:
    """get_muscle_anchor returns correct averaged position."""

    def test_single_bone(self):
        reg = BoneAnchorRegistry()
        node = _make_bone_node("Right Clavicle", (10.0, -5.0, 3.0))
        reg.register_bones({"Right Clavicle": node})
        reg.snapshot_rest_positions()

        pos = reg.get_muscle_anchor("SCM R", ["Right Clavicle"])
        assert pos is not None
        np.testing.assert_allclose(pos, [10.0, -5.0, 3.0], atol=1e-6)

    def test_multiple_bones_averaged(self):
        reg = BoneAnchorRegistry()
        node_r = _make_bone_node("Right Clavicle", (10.0, 0.0, 0.0))
        node_l = _make_bone_node("Left Clavicle", (-10.0, 0.0, 0.0))
        reg.register_bones({"Right Clavicle": node_r, "Left Clavicle": node_l})
        reg.snapshot_rest_positions()

        pos = reg.get_muscle_anchor("Platysma", ["Right Clavicle", "Left Clavicle"])
        assert pos is not None
        np.testing.assert_allclose(pos, [0.0, 0.0, 0.0], atol=1e-6)

    def test_missing_bone_returns_none(self):
        reg = BoneAnchorRegistry()
        reg.register_bones({"Bone A": _make_bone_node("A", (0, 0, 0))})
        reg.snapshot_rest_positions()

        result = reg.get_muscle_anchor("muscle", ["Nonexistent Bone"])
        assert result is None

    def test_partial_bones_uses_available(self):
        reg = BoneAnchorRegistry()
        node = _make_bone_node("Right Clavicle", (5.0, 3.0, 1.0))
        reg.register_bones({"Right Clavicle": node})
        reg.snapshot_rest_positions()

        # Request two bones but only one exists
        pos = reg.get_muscle_anchor("muscle", ["Right Clavicle", "Missing Bone"])
        assert pos is not None
        np.testing.assert_allclose(pos, [5.0, 3.0, 1.0], atol=1e-6)


class TestGetMuscleAnchorCurrent:
    """get_muscle_anchor_current reflects node movement."""

    def test_current_position_after_movement(self):
        reg = BoneAnchorRegistry()
        node = _make_bone_node("Right Clavicle", (10.0, -5.0, 3.0))
        reg.register_bones({"Right Clavicle": node})
        reg.snapshot_rest_positions()

        # Rest position
        rest = reg.get_muscle_anchor("SCM R", ["Right Clavicle"])
        np.testing.assert_allclose(rest, [10.0, -5.0, 3.0], atol=1e-6)

        # Move the node
        node.set_position(12.0, -4.0, 4.0)
        node.mark_dirty()
        node.update_world_matrix(force=True)

        # Current position should reflect the move
        cur = reg.get_muscle_anchor_current("SCM R", ["Right Clavicle"])
        assert cur is not None
        np.testing.assert_allclose(cur, [12.0, -4.0, 4.0], atol=1e-6)

        # Rest position unchanged
        rest2 = reg.get_muscle_anchor("SCM R", ["Right Clavicle"])
        np.testing.assert_allclose(rest2, [10.0, -5.0, 3.0], atol=1e-6)

    def test_current_missing_bone_returns_none(self):
        reg = BoneAnchorRegistry()
        result = reg.get_muscle_anchor_current("muscle", ["Missing"])
        assert result is None
