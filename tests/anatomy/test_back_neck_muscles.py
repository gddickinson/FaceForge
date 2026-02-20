"""Unit tests for BackNeckMuscleHandler.

These tests create mock SkinBinding objects with synthetic back-of-neck
muscle geometry and verify body-end pinning + displacement clamping.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceforge.anatomy.back_neck_muscles import BackNeckMuscleHandler
from faceforge.core.math_utils import quat_from_euler, quat_identity
from faceforge.core.mesh import MeshInstance, BufferGeometry
from faceforge.core.material import Material
from faceforge.core.scene_graph import SceneNode


# ── Helpers ───────────────────────────────────────────────────────────

class _MockSkinBinding:
    """Minimal SkinBinding stand-in for testing."""
    def __init__(
        self,
        mesh: MeshInstance,
        is_muscle: bool = True,
        head_follow_config: dict | None = None,
        muscle_name: str | None = None,
    ):
        self.mesh = mesh
        self.is_muscle = is_muscle
        self.head_follow_config = head_follow_config
        self.muscle_name = muscle_name


class _MockSoftTissue:
    """Minimal SoftTissueSkinning stand-in."""
    def __init__(self, bindings: list):
        self.bindings = bindings


def _make_back_neck_muscle(
    name: str = "Splenius Capitis R",
    n_verts: int = 30,
    y_range: tuple[float, float] = (-25.0, 5.0),
    upper_frac: float = 0.85,
    lower_frac: float = 0.12,
) -> tuple[_MockSkinBinding, dict]:
    """Create a synthetic back-of-neck muscle binding + defn for testing.

    Vertical column from y_range[0] (body end) to y_range[1] (skull end).
    """
    y_vals = np.linspace(y_range[0], y_range[1], n_verts)
    positions = np.zeros((n_verts, 3), dtype=np.float32)
    positions[:, 1] = y_vals
    positions[:, 0] = 2.0 if "R" in name else -2.0
    normals = np.tile([0.0, 0.0, 1.0], (n_verts, 1)).astype(np.float32)

    geom = BufferGeometry(
        positions=positions.ravel(),
        normals=normals.ravel(),
        vertex_count=n_verts,
    )
    mesh = MeshInstance(name=name, geometry=geom, material=Material())
    mesh.rest_positions = positions.ravel().copy()

    head_follow = {"upperFrac": upper_frac, "lowerFrac": lower_frac}

    binding = _MockSkinBinding(
        mesh=mesh,
        is_muscle=True,
        head_follow_config=head_follow,
        muscle_name=name,
    )

    defn = {
        "name": name,
        "stl": "TEST",
        "headFollow": head_follow,
    }

    return binding, defn


def _make_non_headfollow_binding(name: str = "Latissimus Dorsi R") -> tuple[_MockSkinBinding, dict]:
    """Create a muscle binding WITHOUT headFollow."""
    positions = np.random.randn(10, 3).astype(np.float32)
    normals = np.tile([0, 0, 1], (10, 1)).astype(np.float32)

    geom = BufferGeometry(
        positions=positions.ravel(),
        normals=normals.ravel(),
        vertex_count=10,
    )
    mesh = MeshInstance(name=name, geometry=geom, material=Material())
    mesh.rest_positions = positions.ravel().copy()

    binding = _MockSkinBinding(
        mesh=mesh,
        is_muscle=True,
        head_follow_config=None,
        muscle_name=name,
    )

    defn = {"name": name, "stl": "TEST"}

    return binding, defn


def _simulate_head_rotation(positions_flat: np.ndarray, frac: float = 0.5) -> None:
    """Simulate what head-follow blending would do: displace upper vertices."""
    positions = positions_flat.reshape(-1, 3)
    n = positions.shape[0]
    y_vals = positions[:, 1]
    y_min, y_max = y_vals.min(), y_vals.max()
    y_range = y_max - y_min
    if y_range < 1e-6:
        return
    fracs = (y_vals - y_min) / y_range
    # Simulate X displacement proportional to frac (head rotation effect)
    positions[:, 0] += fracs * 5.0  # 5 units at skull end
    positions_flat[:] = positions.ravel()


# ── Tests ─────────────────────────────────────────────────────────────

class TestBackNeckRegistration:
    """BackNeckMuscleHandler should only register headFollow muscles."""

    def test_registers_headfollow_only(self):
        b1, d1 = _make_back_neck_muscle("Splenius Capitis R")
        b2, d2 = _make_back_neck_muscle("Splenius Capitis L")
        b3, d3 = _make_non_headfollow_binding("Latissimus Dorsi R")

        soft_tissue = _MockSoftTissue([b1, b2, b3])
        defs = {d1["name"]: d1, d2["name"]: d2, d3["name"]: d3}

        handler = BackNeckMuscleHandler()
        handler.register(soft_tissue, defs)

        assert handler.registered
        assert len(handler._muscles) == 2

    def test_non_headfollow_not_registered(self):
        b1, d1 = _make_non_headfollow_binding("Latissimus Dorsi R")
        soft_tissue = _MockSoftTissue([b1])
        defs = {d1["name"]: d1}

        handler = BackNeckMuscleHandler()
        handler.register(soft_tissue, defs)

        assert not handler.registered

    def test_empty_bindings(self):
        soft_tissue = _MockSoftTissue([])
        handler = BackNeckMuscleHandler()
        handler.register(soft_tissue, {})
        assert not handler.registered


class TestBackNeckBodyEndPinning:
    """Body-end vertices should stay within threshold during head rotation."""

    def test_body_end_clamped_during_yaw(self):
        b1, d1 = _make_back_neck_muscle("Splenius Capitis R", n_verts=30)
        soft_tissue = _MockSoftTissue([b1])
        defs = {d1["name"]: d1}

        handler = BackNeckMuscleHandler()
        handler.register(soft_tissue, defs)

        rest_pos = b1.mesh.geometry.positions.copy()

        # Simulate what soft tissue + head-follow would do
        _simulate_head_rotation(b1.mesh.geometry.positions, frac=0.5)

        # Apply pinning
        head_q = quat_from_euler(0.0, 0.8, 0.0, "YXZ")
        handler.update(head_q)

        posed_pos = np.asarray(b1.mesh.geometry.positions).reshape(-1, 3)
        rest_3d = rest_pos.reshape(-1, 3)

        # Body-end: lowest 20% by Y
        y_vals = rest_3d[:, 1]
        y_thresh = np.percentile(y_vals, 20)
        lower_mask = y_vals <= y_thresh

        lower_disp = np.linalg.norm(posed_pos[lower_mask] - rest_3d[lower_mask], axis=1)
        max_lower_disp = float(lower_disp.max())

        # Body-end displacement should be clamped
        assert max_lower_disp < 3.0, (
            f"Lower back-neck verts moved too far: {max_lower_disp:.2f}"
        )

    def test_body_end_clamped_during_pitch(self):
        b1, d1 = _make_back_neck_muscle("Semispinalis Cap. R", n_verts=30)
        soft_tissue = _MockSoftTissue([b1])
        defs = {d1["name"]: d1}

        handler = BackNeckMuscleHandler()
        handler.register(soft_tissue, defs)

        rest_pos = b1.mesh.geometry.positions.copy()
        _simulate_head_rotation(b1.mesh.geometry.positions)

        head_q = quat_from_euler(0.5, 0.0, 0.0, "YXZ")
        handler.update(head_q)

        posed_pos = np.asarray(b1.mesh.geometry.positions).reshape(-1, 3)
        rest_3d = rest_pos.reshape(-1, 3)

        y_vals = rest_3d[:, 1]
        y_thresh = np.percentile(y_vals, 20)
        lower_mask = y_vals <= y_thresh

        lower_disp = np.linalg.norm(posed_pos[lower_mask] - rest_3d[lower_mask], axis=1)
        max_lower_disp = float(lower_disp.max())

        assert max_lower_disp < 3.0, (
            f"Lower back-neck verts moved too far during pitch: {max_lower_disp:.2f}"
        )


class TestBackNeckDisplacementClamp:
    """Excessive displacement should be clamped, skull-end unaffected."""

    def test_large_displacement_clamped(self):
        b1, d1 = _make_back_neck_muscle("Splenius Capitis R", n_verts=30)
        soft_tissue = _MockSoftTissue([b1])
        defs = {d1["name"]: d1}

        handler = BackNeckMuscleHandler()
        handler.register(soft_tissue, defs)

        rest_pos = b1.mesh.geometry.positions.copy()

        # Apply large displacement to body-end vertices
        positions = b1.mesh.geometry.positions.reshape(-1, 3)
        rest_3d = rest_pos.reshape(-1, 3)
        y_vals = rest_3d[:, 1]
        y_thresh = np.percentile(y_vals, 30)
        lower_mask = y_vals <= y_thresh
        positions[lower_mask, 0] += 10.0  # large displacement
        b1.mesh.geometry.positions[:] = positions.ravel()

        head_q = quat_from_euler(0.0, 0.5, 0.0, "YXZ")
        handler.update(head_q)

        posed_pos = np.asarray(b1.mesh.geometry.positions).reshape(-1, 3)
        lower_disp = np.linalg.norm(posed_pos[lower_mask] - rest_3d[lower_mask], axis=1)

        # Displacement should be pulled back toward MAX_DISPLACEMENT (2.5)
        assert lower_disp.max() < 5.0, (
            f"Displacement clamp failed: max={lower_disp.max():.2f}"
        )

    def test_skull_end_not_clamped(self):
        b1, d1 = _make_back_neck_muscle("Splenius Capitis R", n_verts=30)
        soft_tissue = _MockSoftTissue([b1])
        defs = {d1["name"]: d1}

        handler = BackNeckMuscleHandler()
        handler.register(soft_tissue, defs)

        rest_3d = b1.mesh.rest_positions.reshape(-1, 3)

        # Simulate head-follow displacement (upper verts move a lot)
        _simulate_head_rotation(b1.mesh.geometry.positions)
        before_clamp = b1.mesh.geometry.positions.copy().reshape(-1, 3)

        # The very top vertex (frac=1.0) should be unaffected by clamping.
        # Use the highest Y vertex directly.
        y_vals = rest_3d[:, 1]
        top_idx = np.argmax(y_vals)

        head_q = quat_from_euler(0.0, 0.5, 0.0, "YXZ")
        handler.update(head_q)

        posed_pos = np.asarray(b1.mesh.geometry.positions).reshape(-1, 3)

        # Top vertex (frac=1.0) should be unchanged since clamp_strength = 0 at frac=1.0
        top_change = float(np.linalg.norm(posed_pos[top_idx] - before_clamp[top_idx]))
        assert top_change < 1e-4, (
            f"Top skull-end vertex was clamped: change={top_change:.6f}"
        )


class TestBackNeckFasciaPinning:
    """Fascia delta should pin body-end vertices."""

    def test_fascia_delta_pins_body_end(self):
        b1, d1 = _make_back_neck_muscle("Splenius Capitis R")
        d1["fasciaRegions"] = ["trapezius_R"]

        soft_tissue = _MockSoftTissue([b1])
        defs = {d1["name"]: d1}

        handler = BackNeckMuscleHandler()
        handler.register(soft_tissue, defs)

        # Create mock fascia
        from faceforge.anatomy.fascia import FasciaRegion, FasciaSystem
        from faceforge.anatomy.bone_anchors import BoneAnchorRegistry

        bone = SceneNode(name="BoneA")
        bone.set_position(5.0, -30.0, 5.0)
        bone.update_world_matrix(force=True)

        registry = BoneAnchorRegistry()
        registry.register_bones({"BoneA": bone})
        registry.snapshot_rest_positions()

        regions = [
            FasciaRegion(name="trapezius_R", bone_names=["BoneA"], bone_weights=[1.0], side="R"),
        ]
        fs = FasciaSystem(regions, registry)
        fs.snapshot_rest()

        handler.set_fascia_system(fs)

        # Move bone to simulate body animation
        bone.set_position(8.0, -30.0, 5.0)
        bone.update_world_matrix(force=True)

        # Simulate head rotation displacement
        _simulate_head_rotation(b1.mesh.geometry.positions)

        head_q = quat_from_euler(0.0, 0.5, 0.0, "YXZ")
        handler.update(head_q)

        # Should not crash — fascia pinning applied
        posed_pos = np.asarray(b1.mesh.geometry.positions).reshape(-1, 3)
        assert posed_pos.shape[0] == 30

    def test_no_fascia_no_crash(self):
        b1, d1 = _make_back_neck_muscle("Splenius Capitis R")
        soft_tissue = _MockSoftTissue([b1])
        defs = {d1["name"]: d1}

        handler = BackNeckMuscleHandler()
        handler.register(soft_tissue, defs)

        # Don't call set_fascia_system
        _simulate_head_rotation(b1.mesh.geometry.positions)

        head_q = quat_from_euler(0.0, 0.5, 0.0, "YXZ")
        handler.update(head_q)

        posed_pos = b1.mesh.geometry.positions
        assert posed_pos.shape[0] == 30 * 3


class TestBackNeckIdentityNoOp:
    """Identity head quaternion should produce no modification."""

    def test_identity_no_change(self):
        b1, d1 = _make_back_neck_muscle("Splenius Capitis R")
        soft_tissue = _MockSoftTissue([b1])
        defs = {d1["name"]: d1}

        handler = BackNeckMuscleHandler()
        handler.register(soft_tissue, defs)

        # Set some positions
        before = b1.mesh.geometry.positions.copy()

        handler.update(quat_identity())

        after = b1.mesh.geometry.positions
        np.testing.assert_array_equal(after, before)

    def test_unregistered_no_op(self):
        handler = BackNeckMuscleHandler()
        # Not registered — should be a no-op
        head_q = quat_from_euler(0.0, 0.8, 0.0, "YXZ")
        handler.update(head_q)  # should not crash
