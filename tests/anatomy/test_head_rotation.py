"""Unit tests for HeadRotationSystem.

These tests use mock SceneNodes and do NOT require STL assets.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceforge.anatomy.head_rotation import HeadRotationSystem, SOFT_CLAMP_THRESHOLD
from faceforge.core.math_utils import (
    quat_identity, quat_normalize, quat_from_euler,
    quat_rotate_vec3, vec3, deg_to_rad,
)
from faceforge.core.scene_graph import SceneNode
from faceforge.core.state import FaceState, ConstraintState
from faceforge.constants import HEAD_YAW_MAX, HEAD_PITCH_MAX, HEAD_ROLL_MAX


# ── Helpers ───────────────────────────────────────────────────────────

def _make_group(name: str) -> SceneNode:
    """Create a SceneNode with a known identity transform."""
    node = SceneNode(name=name)
    return node


def _is_identity_transform(node: SceneNode, atol: float = 1e-6) -> bool:
    """Check if node's quaternion is identity and position is zero."""
    q_identity = quat_identity()
    pos_zero = vec3(0, 0, 0)
    return (
        np.allclose(node.quaternion, q_identity, atol=atol)
        and np.allclose(node.position, pos_zero, atol=atol)
    )


# ── Tests ─────────────────────────────────────────────────────────────

class TestIdentityRotation:
    """Zero rotation should produce identity transforms."""

    def test_identity_rotation_no_movement(self):
        system = HeadRotationSystem(jaw_pivot=(0, -1.5, 10.4))
        skull = _make_group("skull")
        face = _make_group("face")
        state = FaceState()  # all zeros

        q = system.apply(state, skull, face)

        assert np.allclose(q, quat_identity(), atol=1e-6)
        assert _is_identity_transform(skull)
        assert _is_identity_transform(face)

    def test_identity_with_optional_groups(self):
        system = HeadRotationSystem(jaw_pivot=(0, -1.5, 10.4))
        skull = _make_group("skull")
        face = _make_group("face")
        brain = _make_group("brain")
        stl_muscle = _make_group("stlMuscle")
        expr_muscle = _make_group("exprMuscle")
        face_feature = _make_group("faceFeature")
        state = FaceState()

        q = system.apply(
            state, skull, face,
            brain_group=brain,
            stl_muscle_group=stl_muscle,
            expr_muscle_group=expr_muscle,
            face_feature_group=face_feature,
        )

        assert np.allclose(q, quat_identity(), atol=1e-6)
        assert _is_identity_transform(brain)
        assert _is_identity_transform(stl_muscle)
        assert _is_identity_transform(expr_muscle)
        assert _is_identity_transform(face_feature)


class TestAllGroupsReceiveRotation:
    """All 6 groups should receive the same quaternion when rotated."""

    def test_all_groups_get_quaternion(self):
        system = HeadRotationSystem(jaw_pivot=(0, -1.5, 10.4))
        skull = _make_group("skull")
        face = _make_group("face")
        brain = _make_group("brain")
        stl_muscle = _make_group("stlMuscle")
        expr_muscle = _make_group("exprMuscle")
        face_feature = _make_group("faceFeature")

        state = FaceState()
        state.head_yaw = 0.5

        q = system.apply(
            state, skull, face,
            brain_group=brain,
            stl_muscle_group=stl_muscle,
            expr_muscle_group=expr_muscle,
            face_feature_group=face_feature,
        )

        # All groups should have the same quaternion
        for group in [skull, face, brain, stl_muscle, expr_muscle, face_feature]:
            assert np.allclose(group.quaternion, q, atol=1e-6), (
                f"{group.name} quaternion mismatch: {group.quaternion} != {q}"
            )

    def test_rotation_moves_groups_from_identity(self):
        system = HeadRotationSystem(jaw_pivot=(0, -1.5, 10.4))
        skull = _make_group("skull")
        face = _make_group("face")
        stl_muscle = _make_group("stlMuscle")

        state = FaceState()
        state.head_yaw = 0.8

        q = system.apply(
            state, skull, face,
            stl_muscle_group=stl_muscle,
        )

        # Quaternion should NOT be identity
        assert not np.allclose(q, quat_identity(), atol=1e-3)
        # All provided groups should have non-identity quaternion
        assert not _is_identity_transform(skull)
        assert not _is_identity_transform(face)
        assert not _is_identity_transform(stl_muscle)

    def test_optional_groups_skip_when_none(self):
        """Passing None for optional groups should not raise."""
        system = HeadRotationSystem(jaw_pivot=(0, -1.5, 10.4))
        skull = _make_group("skull")
        face = _make_group("face")

        state = FaceState()
        state.head_pitch = -0.5

        # Should not raise
        q = system.apply(
            state, skull, face,
            brain_group=None,
            stl_muscle_group=None,
            expr_muscle_group=None,
            face_feature_group=None,
        )
        assert not np.allclose(q, quat_identity(), atol=1e-3)


class TestVertebraeDistribution:
    """Vertebrae should receive fractional rotation: C1 at 100%, T1 at 0%."""

    def _make_vertebrae_pivots(self) -> list[dict]:
        """Create mock vertebrae pivot list."""
        pivots = []
        for level in range(8):  # C1 through T1
            frac = 1.0 - level / 7.0
            pivots.append({
                "group": _make_group(f"vert_{level}"),
                "level": level,
                "fractions": {"yaw": frac, "pitch": frac, "roll": frac},
            })
        return pivots

    def test_c1_gets_full_rotation(self):
        system = HeadRotationSystem(jaw_pivot=(0, -1.5, 10.4))
        skull = _make_group("skull")
        face = _make_group("face")
        vertebrae = self._make_vertebrae_pivots()

        state = FaceState()
        state.head_yaw = 0.6

        system.apply(state, skull, face, vertebrae_pivots=vertebrae)

        # C1 (level 0) should have a rotation close to full head rotation
        c1 = vertebrae[0]["group"]
        assert not np.allclose(c1.quaternion, quat_identity(), atol=1e-3)

    def test_t1_gets_no_rotation(self):
        system = HeadRotationSystem(jaw_pivot=(0, -1.5, 10.4))
        skull = _make_group("skull")
        face = _make_group("face")
        vertebrae = self._make_vertebrae_pivots()

        state = FaceState()
        state.head_yaw = 0.6

        system.apply(state, skull, face, vertebrae_pivots=vertebrae)

        # T1 (level 7) with frac=0 should have identity rotation
        t1 = vertebrae[7]["group"]
        assert np.allclose(t1.quaternion, quat_identity(), atol=1e-6)

    def test_vertebrae_gradual_decrease(self):
        system = HeadRotationSystem(jaw_pivot=(0, -1.5, 10.4))
        skull = _make_group("skull")
        face = _make_group("face")
        vertebrae = self._make_vertebrae_pivots()

        state = FaceState()
        state.head_yaw = 0.5

        system.apply(state, skull, face, vertebrae_pivots=vertebrae)

        # Quaternion w component should approach 1.0 (identity) as level increases
        w_values = [v["group"].quaternion[3] for v in vertebrae]
        for i in range(len(w_values) - 1):
            assert w_values[i] <= w_values[i + 1] + 1e-6, (
                f"Level {i} w={w_values[i]} > level {i+1} w={w_values[i+1]}"
            )


class TestSoftClamping:
    """Constraint excess should reduce rotation magnitude."""

    def test_no_clamping_without_excess(self):
        system = HeadRotationSystem(jaw_pivot=(0, -1.5, 10.4))
        skull = _make_group("skull")
        face = _make_group("face")

        state = FaceState()
        state.head_yaw = 0.5

        constraint = ConstraintState()
        constraint.total_excess = 0.0

        q = system.apply(state, skull, face, constraint_state=constraint)
        q_no_constraint = system.apply(state, skull, face)

        assert np.allclose(q, q_no_constraint, atol=1e-6)

    def test_clamping_reduces_rotation(self):
        system1 = HeadRotationSystem(jaw_pivot=(0, -1.5, 10.4))
        system2 = HeadRotationSystem(jaw_pivot=(0, -1.5, 10.4))

        skull1 = _make_group("skull1")
        face1 = _make_group("face1")
        skull2 = _make_group("skull2")
        face2 = _make_group("face2")

        state = FaceState()
        state.head_yaw = 0.8

        # Without constraint
        q_free = system1.apply(state, skull1, face1)

        # With significant excess (set both raw and smoothed)
        constraint = ConstraintState()
        constraint.total_excess = 1.0  # well above threshold
        constraint.smoothed_total_excess = 1.0  # smoothed value used by soft-clamp
        q_clamped = system2.apply(state, skull2, face2, constraint_state=constraint)

        # Clamped should be closer to identity (smaller rotation)
        # The w component of the clamped quat should be closer to 1.0
        assert abs(q_clamped[3]) > abs(q_free[3])


class TestReset:
    """Reset should clear rotation state."""

    def test_reset_clears_quaternion(self):
        system = HeadRotationSystem(jaw_pivot=(0, -1.5, 10.4))
        skull = _make_group("skull")
        face = _make_group("face")

        state = FaceState()
        state.head_yaw = 0.5
        system.apply(state, skull, face)

        assert not np.allclose(system.head_quaternion, quat_identity(), atol=1e-3)

        system.reset()
        assert np.allclose(system.head_quaternion, quat_identity(), atol=1e-6)


class TestPivotRotation:
    """The pivot rotation should produce correct world-space results."""

    def test_pivot_rotation_preserves_pivot_position(self):
        """After pivot rotation, the pivot point should map to itself."""
        system = HeadRotationSystem(jaw_pivot=(0, -1.5, 10.4))
        skull = _make_group("skull")
        face = _make_group("face")

        state = FaceState()
        state.head_yaw = 0.5

        q = system.apply(state, skull, face)

        # The pivot point transformed by the group's rotation should still
        # be at the same world position. This is the purpose of the offset.
        pivot = system._head_pivot
        rotated = quat_rotate_vec3(q, pivot)
        offset = pivot - rotated

        expected_pos = offset
        assert np.allclose(skull.position, expected_pos, atol=1e-6)
