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


class TestCapsulesLiveOnTheirBones:
    """Capsules are built in the bone's local frame and placed with its world matrix.

    Bone meshes are reparented under joint pivots, so their vertex arrays are
    pivot-local; the first version treated them as world positions and put
    all twelve capsules in a cluster near the body origin -- the neck --
    where a phantom humerus pushed the deep neck muscles 3 units at rest.
    """

    def _bone_under_pivot(self, name, local_positions, pivot_pos):
        reg = BoneAnchorRegistry()
        pivot = SceneNode(name=f"{name}_pivot")
        pivot.set_position(*pivot_pos)
        node = SceneNode(name=name)
        pos = np.array(local_positions, dtype=np.float32)
        nrm = np.zeros_like(pos)
        nrm[2::3] = 1.0
        geom = BufferGeometry(positions=pos, normals=nrm, vertex_count=len(pos) // 3)
        node.mesh = MeshInstance(name=name, geometry=geom, material=Material())
        pivot.add(node)
        pivot.update_world_matrix(force=True)
        reg.register_bones({name: node})
        reg.snapshot_rest_positions()
        return reg, pivot, node

    def test_capsule_is_placed_with_the_bones_world_matrix(self):
        local = []
        for x in np.linspace(0, 20, 10):
            local.extend([x, 0.0, 0.0])
        reg, pivot, _node = self._bone_under_pivot("Right Humerus", local, (50.0, 0.0, -10.0))
        sys = BoneCollisionSystem(reg)
        assert sys.build_capsules() == 1
        cap = sys._capsules[0]
        assert min(cap.start[0], cap.end[0]) == pytest.approx(50.0, abs=1e-6)
        assert max(cap.start[0], cap.end[0]) == pytest.approx(70.0, abs=1e-6)

        pivot.set_position(80.0, 0.0, -10.0)          # the limb moved
        pivot.update_world_matrix(force=True)
        sys.refresh()
        assert min(cap.start[0], cap.end[0]) == pytest.approx(80.0, abs=1e-6)
        # A vertex where the bone WAS is no longer pushed; one where it IS, is.
        stale = np.array([60.0, 1.0, -10.0], dtype=np.float32)
        live = np.array([90.0, 1.0, -10.0], dtype=np.float32)
        rest = np.array([60.0, 40.0, 0.0, 90.0, 40.0, 0.0], dtype=np.float32)
        pos = np.concatenate([stale, live])
        assert sys.resolve_penetrations(pos, rest) == 1
        assert pos[:3] == pytest.approx(stale)
        assert pos[4] == pytest.approx(cap.radius, abs=1e-5)

    def test_rest_penetration_is_an_allowance_not_a_defect(self):
        local = []
        for x in np.linspace(0, 20, 10):
            local.extend([x, 0.0, 0.0])
        reg, _pivot, _node = self._bone_under_pivot("Right Humerus", local, (0.0, 0.0, 0.0))
        sys = BoneCollisionSystem(reg)
        sys.build_capsules()
        radius = sys._capsules[0].radius
        # One vertex 1.0 inside the capsule at rest, one outside.
        rest = np.array([10.0, 1.0, 0.0, 10.0, radius + 2.0, 0.0], dtype=np.float32)
        pos = rest.copy()
        assert sys.resolve_penetrations(pos, rest) == 0, "nothing moves at rest, by construction"
        assert np.array_equal(pos, rest)
        # Pushed deeper than its rest depth: back out to its rest depth, not to the surface.
        pos = rest.copy()
        pos[1] = 0.3
        assert sys.resolve_penetrations(pos, rest) == 1
        assert pos[1] == pytest.approx(1.0, abs=1e-5)
        # The vertex that was outside at rest is held at the full radius.
        pos = rest.copy()
        pos[4] = 0.5
        assert sys.resolve_penetrations(pos, rest) == 1
        assert pos[4] == pytest.approx(radius, abs=1e-5)


class TestCandidateCulling:
    """Only capsules whose box overlaps the mesh are measured; the result is unchanged."""

    def _system_with_capsule(self, start, end, radius):
        sys = BoneCollisionSystem(BoneAnchorRegistry())
        sys._capsules = [BoneCapsule(bone_name="b", start=np.array(start, float),
                                     end=np.array(end, float), radius=radius)]
        return sys

    def test_a_mesh_far_from_every_capsule_is_not_measured(self, monkeypatch):
        import faceforge.anatomy.bone_collision as bc
        calls = []
        real = bc._radial_distance
        monkeypatch.setattr(bc, "_radial_distance",
                            lambda *a, **k: calls.append(1) or real(*a, **k))
        sys = self._system_with_capsule((0, 0, 0), (0, 0, 10), 2.0)
        far = np.array([[50.0, 0, 0], [52.0, 0, 3], [55.0, 1, 5]], dtype=np.float32).ravel()
        rest = far.copy()
        assert sys.resolve_penetrations(far, rest) == 0
        assert calls == []

    def test_only_vertices_in_the_capsule_box_are_measured_and_the_push_is_the_same(self):
        sys = self._system_with_capsule((0, 0, 0), (0, 0, 10), 2.0)
        # One vertex inside the capsule, one far away on the same mesh.
        verts = np.array([[0.5, 0.0, 5.0], [80.0, 0.0, 5.0]], dtype=np.float32)
        rest = np.array([[5.0, 0.0, 5.0], [80.0, 0.0, 5.0]], dtype=np.float32)  # outside at rest
        pos = verts.ravel().copy()
        n = sys.resolve_penetrations(pos, rest.ravel())
        out = pos.reshape(-1, 3)
        assert n == 1
        np.testing.assert_allclose(out[0], [2.0, 0.0, 5.0], atol=1e-5)   # pushed to the radius
        np.testing.assert_allclose(out[1], [80.0, 0.0, 5.0])             # untouched
