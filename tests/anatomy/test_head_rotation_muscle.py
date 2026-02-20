"""Tests for muscle-aware head-to-skin blending (Layer 1)."""

import numpy as np
import pytest

from faceforge.anatomy.head_rotation import HeadRotationSystem
from faceforge.core.math_utils import quat_from_euler, deg_to_rad


class TestApplyToSkinMuscle:
    """Tests for HeadRotationSystem.apply_to_skin_muscle()."""

    def _make_system(self):
        return HeadRotationSystem(jaw_pivot=(0.0, -1.5, 10.4))

    def test_identity_quaternion_noop(self):
        """Identity rotation should not modify positions."""
        sys = self._make_system()
        pos = np.array([0.0, 10.0, 5.0, 0.0, -10.0, -5.0], dtype=np.float32)
        rest = pos.copy()
        original = pos.copy()

        sys.apply_to_skin_muscle(pos, rest, 0.8, 0.1)
        np.testing.assert_array_almost_equal(pos, original)

    def test_zero_fractions_noop(self):
        """Zero fractions should not modify positions regardless of rotation."""
        sys = self._make_system()
        # Set a non-trivial head quaternion
        head_q = quat_from_euler(deg_to_rad(30.0), 0.0, 0.0, "YXZ")
        sys._head_quat = head_q

        pos = np.array([0.0, 10.0, 5.0, 0.0, -10.0, -5.0], dtype=np.float32)
        rest = pos.copy()
        original = pos.copy()

        sys.apply_to_skin_muscle(pos, rest, 0.0, 0.0)
        np.testing.assert_array_almost_equal(pos, original)

    def test_full_fraction_follows_head(self):
        """Fractions of 1.0 should fully follow head rotation."""
        sys = self._make_system()
        head_q = quat_from_euler(deg_to_rad(30.0), 0.0, 0.0, "YXZ")
        sys._head_quat = head_q

        # Single vertex near the head
        rest = np.array([0.0, 5.0, 15.0], dtype=np.float32)
        pos = rest.copy()

        sys.apply_to_skin_muscle(pos, rest, 1.0, 1.0, head_q)

        # Position should have changed (rotated)
        assert not np.allclose(pos, rest, atol=0.01)

    def test_upper_frac_stronger_than_lower(self):
        """Top vertices should move more than bottom vertices."""
        sys = self._make_system()
        head_q = quat_from_euler(deg_to_rad(30.0), 0.0, 0.0, "YXZ")
        sys._head_quat = head_q

        # Two vertices: one at top (Y=20), one at bottom (Y=-20)
        rest = np.array([
            [0.0, 20.0, 5.0],
            [0.0, -20.0, 5.0],
        ], dtype=np.float32)
        pos = rest.copy().ravel()
        rest_flat = rest.ravel().copy()

        sys.apply_to_skin_muscle(pos, rest_flat, 0.9, 0.1, head_q)

        pos_2d = pos.reshape(-1, 3)
        rest_2d = rest
        top_displacement = np.linalg.norm(pos_2d[0] - rest_2d[0])
        bot_displacement = np.linalg.norm(pos_2d[1] - rest_2d[1])

        assert top_displacement > bot_displacement

    def test_explicit_head_q_parameter(self):
        """Passing explicit head_q should override stored quaternion."""
        sys = self._make_system()
        # Store identity
        sys._head_quat = np.array([0.0, 0.0, 0.0, 1.0])

        head_q = quat_from_euler(deg_to_rad(45.0), 0.0, 0.0, "YXZ")

        rest = np.array([0.0, 5.0, 15.0], dtype=np.float32)
        pos = rest.copy()

        sys.apply_to_skin_muscle(pos, rest, 1.0, 1.0, head_q)
        # Should have moved (used explicit head_q, not identity)
        assert not np.allclose(pos, rest, atol=0.01)


class TestSkinBindingHeadFollowConfig:
    """Test that SkinBinding stores head_follow_config correctly."""

    def test_default_none(self):
        from faceforge.body.soft_tissue import SkinBinding
        from faceforge.core.mesh import BufferGeometry, MeshInstance
        from faceforge.core.material import Material

        pos = np.zeros(9, dtype=np.float32)
        nrm = np.zeros(9, dtype=np.float32)
        geom = BufferGeometry(positions=pos, normals=nrm, vertex_count=3)
        mesh = MeshInstance(name="test", geometry=geom, material=Material())

        binding = SkinBinding(
            mesh=mesh,
            joint_indices=np.zeros(3, dtype=np.int32),
            weights=np.ones(3, dtype=np.float32),
            secondary_indices=np.zeros(3, dtype=np.int32),
        )
        assert binding.head_follow_config is None
        assert binding.muscle_name is None

    def test_with_config(self):
        from faceforge.body.soft_tissue import SkinBinding
        from faceforge.core.mesh import BufferGeometry, MeshInstance
        from faceforge.core.material import Material

        pos = np.zeros(9, dtype=np.float32)
        nrm = np.zeros(9, dtype=np.float32)
        geom = BufferGeometry(positions=pos, normals=nrm, vertex_count=3)
        mesh = MeshInstance(name="test", geometry=geom, material=Material())

        config = {"upperFrac": 0.7, "lowerFrac": 0.03}
        binding = SkinBinding(
            mesh=mesh,
            joint_indices=np.zeros(3, dtype=np.int32),
            weights=np.ones(3, dtype=np.float32),
            secondary_indices=np.zeros(3, dtype=np.int32),
            head_follow_config=config,
            muscle_name="Desc. Trapezius R",
        )
        assert binding.head_follow_config == config
        assert binding.muscle_name == "Desc. Trapezius R"
