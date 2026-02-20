"""Tests for MuscleAttachmentSystem (Layers 2-3)."""

import numpy as np
import pytest

from faceforge.anatomy.bone_anchors import BoneAnchorRegistry
from faceforge.anatomy.muscle_attachments import (
    MuscleAttachmentData,
    MuscleAttachmentSystem,
    MAX_STRETCH,
    PIN_STRENGTH,
)
from faceforge.body.soft_tissue import SkinBinding
from faceforge.core.mesh import BufferGeometry, MeshInstance
from faceforge.core.material import Material
from faceforge.core.scene_graph import SceneNode


def _make_capsule_mesh(n_verts=100, y_range=(0.0, 50.0)):
    """Create a simple mesh with vertices spanning a Y range."""
    y_min, y_max = y_range
    positions = np.zeros((n_verts, 3), dtype=np.float32)
    positions[:, 1] = np.linspace(y_min, y_max, n_verts)
    positions[:, 0] = np.sin(np.linspace(0, 2 * np.pi, n_verts)) * 3.0
    normals = np.tile([0.0, 0.0, 1.0], (n_verts, 1)).astype(np.float32)
    geom = BufferGeometry(
        positions=positions.ravel(),
        normals=normals.ravel(),
        vertex_count=n_verts,
    )
    mat = Material(color=(0.8, 0.5, 0.5))
    mesh = MeshInstance(name="test_muscle", geometry=geom, material=mat)
    mesh.rest_positions = positions.ravel().copy()
    return mesh


def _make_bone_registry_with_bones():
    """Create a BoneAnchorRegistry with fake bone nodes."""
    reg = BoneAnchorRegistry()

    bones = {}
    for name in ["Right Clavicle", "Right Humerus", "Left Clavicle", "Left Humerus",
                  "Right Scapula", "Left Scapula"]:
        node = SceneNode(name=name)
        # Create tiny mesh so centroid is computed
        pos = np.array([10.0, 10.0, 10.0, 11.0, 10.0, 10.0, 10.0, 11.0, 10.0],
                       dtype=np.float32)
        nrm = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1], dtype=np.float32)
        geom = BufferGeometry(positions=pos, normals=nrm, vertex_count=3)
        node.mesh = MeshInstance(name=name, geometry=geom, material=Material())
        bones[name] = node

    reg.register_bones(bones)
    reg.snapshot_rest_positions()
    return reg


class TestMuscleAttachmentData:
    def test_dataclass_fields(self):
        data = MuscleAttachmentData(
            muscle_name="Deltoid R",
            origin_bones=["Right Clavicle"],
            insertion_bones=["Right Humerus"],
        )
        assert data.muscle_name == "Deltoid R"
        assert data.origin_bones == ["Right Clavicle"]
        assert data.insertion_bones == ["Right Humerus"]
        assert data.origin_frac_threshold == 0.8
        assert data.insertion_frac_threshold == 0.2
        assert data.rest_length == 1.0
        assert data.current_stretch == 1.0
        assert data.fascia_regions == []


class TestMuscleAttachmentSystem:
    def test_register_muscle(self):
        reg = _make_bone_registry_with_bones()
        sys = MuscleAttachmentSystem(reg)
        mesh = _make_capsule_mesh()

        binding = SkinBinding(
            mesh=mesh,
            joint_indices=np.zeros(100, dtype=np.int32),
            weights=np.ones(100, dtype=np.float32),
            secondary_indices=np.zeros(100, dtype=np.int32),
            is_muscle=True,
            muscle_name="Deltoid Clav. R",
        )

        sys.register_muscle(
            binding,
            origin_bones=["Right Clavicle"],
            insertion_bones=["Right Humerus"],
        )
        assert sys.attachment_count == 1

    def test_register_with_fascia(self):
        reg = _make_bone_registry_with_bones()
        sys = MuscleAttachmentSystem(reg)
        mesh = _make_capsule_mesh()

        binding = SkinBinding(
            mesh=mesh,
            joint_indices=np.zeros(100, dtype=np.int32),
            weights=np.ones(100, dtype=np.float32),
            secondary_indices=np.zeros(100, dtype=np.int32),
            is_muscle=True,
            muscle_name="Deltoid R",
        )

        sys.register_muscle(
            binding, ["Right Clavicle"], ["Right Humerus"],
            fascia_regions=["deltoid_R"],
        )
        assert sys.attachment_count == 1

    def test_rest_length_computed(self):
        reg = _make_bone_registry_with_bones()
        sys = MuscleAttachmentSystem(reg)
        mesh = _make_capsule_mesh(n_verts=100, y_range=(0.0, 50.0))

        binding = SkinBinding(
            mesh=mesh,
            joint_indices=np.zeros(100, dtype=np.int32),
            weights=np.ones(100, dtype=np.float32),
            secondary_indices=np.zeros(100, dtype=np.int32),
            is_muscle=True,
            muscle_name="Test",
        )

        sys.register_muscle(binding, ["Right Clavicle"], ["Right Humerus"])
        data = sys._attachments[id(binding)]
        # Rest length should be roughly the Y extent (50 units)
        assert data.rest_length > 30.0  # generous lower bound

    def test_bone_pinning_no_crash(self):
        """Bone pinning should not crash even with simple geometry."""
        reg = _make_bone_registry_with_bones()
        sys = MuscleAttachmentSystem(reg)
        mesh = _make_capsule_mesh()

        binding = SkinBinding(
            mesh=mesh,
            joint_indices=np.zeros(100, dtype=np.int32),
            weights=np.ones(100, dtype=np.float32),
            secondary_indices=np.zeros(100, dtype=np.int32),
            is_muscle=True,
            muscle_name="Test",
        )

        sys.register_muscle(binding, ["Right Clavicle"], ["Right Humerus"])
        # Should not raise
        sys.apply_bone_pinning(binding)

    def test_stretch_clamp_returns_zero_at_rest(self):
        reg = _make_bone_registry_with_bones()
        sys = MuscleAttachmentSystem(reg)
        mesh = _make_capsule_mesh()

        binding = SkinBinding(
            mesh=mesh,
            joint_indices=np.zeros(100, dtype=np.int32),
            weights=np.ones(100, dtype=np.float32),
            secondary_indices=np.zeros(100, dtype=np.int32),
            is_muscle=True,
            muscle_name="Test",
        )

        sys.register_muscle(binding, ["Right Clavicle"], ["Right Humerus"])
        excess = sys.apply_stretch_clamp(binding)
        assert excess == 0.0

    def test_stretch_clamp_detects_elongation(self):
        reg = _make_bone_registry_with_bones()
        sys = MuscleAttachmentSystem(reg)
        mesh = _make_capsule_mesh(n_verts=100, y_range=(0.0, 50.0))

        binding = SkinBinding(
            mesh=mesh,
            joint_indices=np.zeros(100, dtype=np.int32),
            weights=np.ones(100, dtype=np.float32),
            secondary_indices=np.zeros(100, dtype=np.int32),
            is_muscle=True,
            muscle_name="Test",
        )

        sys.register_muscle(binding, ["Right Clavicle"], ["Right Humerus"])

        # Stretch the mesh by 2x in Y
        pos = mesh.geometry.positions.reshape(-1, 3).copy()
        pos[:, 1] *= 2.0
        mesh.geometry.positions = pos.ravel()

        excess = sys.apply_stretch_clamp(binding)
        # 2.0 stretch ratio > MAX_STRETCH (1.35), so excess should be > 0
        assert excess > 0.0

    def test_total_tension_excess(self):
        reg = _make_bone_registry_with_bones()
        sys = MuscleAttachmentSystem(reg)
        mesh = _make_capsule_mesh(n_verts=100, y_range=(0.0, 50.0))

        binding = SkinBinding(
            mesh=mesh,
            joint_indices=np.zeros(100, dtype=np.int32),
            weights=np.ones(100, dtype=np.float32),
            secondary_indices=np.zeros(100, dtype=np.int32),
            is_muscle=True,
            muscle_name="Test",
        )

        sys.register_muscle(binding, ["Right Clavicle"], ["Right Humerus"])
        assert sys.get_total_tension_excess() == 0.0

    def test_unregistered_binding_noop(self):
        """Operations on unregistered bindings should be no-ops."""
        reg = _make_bone_registry_with_bones()
        sys = MuscleAttachmentSystem(reg)
        mesh = _make_capsule_mesh()

        binding = SkinBinding(
            mesh=mesh,
            joint_indices=np.zeros(100, dtype=np.int32),
            weights=np.ones(100, dtype=np.float32),
            secondary_indices=np.zeros(100, dtype=np.int32),
            is_muscle=True,
            muscle_name="Unregistered",
        )

        # Should not crash
        sys.apply_bone_pinning(binding)
        excess = sys.apply_stretch_clamp(binding)
        assert excess == 0.0

    def test_attachment_frac_range(self):
        """Attachment fractions should be in [0, 1]."""
        reg = _make_bone_registry_with_bones()
        sys = MuscleAttachmentSystem(reg)
        mesh = _make_capsule_mesh()

        binding = SkinBinding(
            mesh=mesh,
            joint_indices=np.zeros(100, dtype=np.int32),
            weights=np.ones(100, dtype=np.float32),
            secondary_indices=np.zeros(100, dtype=np.int32),
            is_muscle=True,
            muscle_name="Test",
        )

        sys.register_muscle(binding, ["Right Clavicle"], ["Right Humerus"])
        data = sys._attachments[id(binding)]
        assert data.attachment_frac.min() >= 0.0
        assert data.attachment_frac.max() <= 1.0
        assert data.origin_mask.any()
        assert data.insertion_mask.any()
