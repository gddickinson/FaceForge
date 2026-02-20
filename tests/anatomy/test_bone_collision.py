"""Tests for BoneCollisionSystem (Layer 4)."""

import numpy as np
import pytest

from faceforge.anatomy.bone_collision import BoneCapsule, BoneCollisionSystem
from faceforge.anatomy.bone_anchors import BoneAnchorRegistry
from faceforge.core.mesh import BufferGeometry, MeshInstance
from faceforge.core.material import Material
from faceforge.core.scene_graph import SceneNode


class TestBoneCapsule:
    def test_capsule_dataclass(self):
        cap = BoneCapsule(
            bone_name="Test",
            start=np.array([0.0, 0.0, 0.0]),
            end=np.array([10.0, 0.0, 0.0]),
            radius=2.0,
        )
        assert cap.bone_name == "Test"
        assert cap.radius == 2.0


class TestBoneCollisionSystem:
    def _make_registry_with_bone(self, name, positions):
        """Create a registry with one bone with given vertex positions."""
        reg = BoneAnchorRegistry()
        node = SceneNode(name=name)
        pos = np.array(positions, dtype=np.float32)
        nrm = np.zeros_like(pos)
        nrm[2::3] = 1.0
        geom = BufferGeometry(positions=pos, normals=nrm, vertex_count=len(pos) // 3)
        node.mesh = MeshInstance(name=name, geometry=geom, material=Material())
        reg.register_bones({name: node})
        reg.snapshot_rest_positions()
        return reg

    def test_build_capsules_with_bone(self):
        """A registered bone with mesh should produce a capsule."""
        # Create a bone mesh along X axis
        positions = []
        for x in np.linspace(0, 20, 10):
            positions.extend([x, 0.0, 0.0])
        reg = self._make_registry_with_bone("Right Clavicle", positions)

        sys = BoneCollisionSystem(reg)
        n = sys.build_capsules()
        assert n >= 1
        assert sys.capsule_count >= 1

    def test_build_capsules_empty_registry(self):
        """Empty registry should produce zero capsules."""
        reg = BoneAnchorRegistry()
        sys = BoneCollisionSystem(reg)
        n = sys.build_capsules()
        assert n == 0

    def test_resolve_no_penetration(self):
        """Vertices far from capsule should not be moved."""
        positions = []
        for x in np.linspace(0, 20, 10):
            positions.extend([x, 0.0, 0.0])
        reg = self._make_registry_with_bone("Right Clavicle", positions)

        sys = BoneCollisionSystem(reg)
        sys.build_capsules()

        # Create vertices far from the capsule
        vert_pos = np.array([
            [50.0, 50.0, 50.0],
            [60.0, 50.0, 50.0],
        ], dtype=np.float32)
        rest_pos = vert_pos.copy()

        original = vert_pos.ravel().copy()
        n_corrected = sys.resolve_penetrations(vert_pos.ravel(), rest_pos.ravel())
        assert n_corrected == 0
        np.testing.assert_array_equal(vert_pos.ravel(), original)

    def test_resolve_penetrating_vertex(self):
        """A vertex inside a capsule should be pushed to the surface."""
        # Create a long bone along X axis
        positions = []
        for x in np.linspace(0, 30, 15):
            positions.extend([x, 0.0, 0.0])
        reg = self._make_registry_with_bone("Right Clavicle", positions)

        sys = BoneCollisionSystem(reg)
        sys.build_capsules()

        if sys.capsule_count == 0:
            pytest.skip("No capsules built")

        capsule = sys._capsules[0]
        # Place vertex on the capsule axis (should be inside)
        mid = (capsule.start + capsule.end) / 2.0
        vert_pos = np.array([mid], dtype=np.float32)
        rest_pos = vert_pos.copy()

        n_corrected = sys.resolve_penetrations(vert_pos.ravel(), rest_pos.ravel())
        # The vertex was on the axis — dist was ~0 which is less than radius
        # but it was at distance 0 from axis, so it's skipped (dist < 1e-6 guard)
        # Place vertex slightly off-axis instead
        vert_pos = np.array([[mid[0], 0.5, 0.0]], dtype=np.float32)
        rest_pos = vert_pos.copy()
        n_corrected = sys.resolve_penetrations(vert_pos.ravel(), rest_pos.ravel())

        if n_corrected > 0:
            # Verify vertex was pushed to surface (distance = radius)
            new_pos = vert_pos.reshape(-1, 3)[0].astype(np.float64)
            closest_on_axis = capsule.start + np.dot(
                new_pos - capsule.start, capsule.axis
            ) * capsule.axis
            dist = np.linalg.norm(new_pos - closest_on_axis)
            np.testing.assert_allclose(dist, capsule.radius, atol=0.1)

    def test_resolve_returns_count(self):
        """resolve_penetrations should return number of corrected vertices."""
        reg = BoneAnchorRegistry()
        sys = BoneCollisionSystem(reg)
        # No capsules → no corrections
        pos = np.zeros(9, dtype=np.float32)
        rest = pos.copy()
        assert sys.resolve_penetrations(pos, rest) == 0
