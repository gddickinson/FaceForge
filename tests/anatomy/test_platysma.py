"""Unit tests for PlatysmaHandler.

These tests create mock ExprMuscleData with synthetic Platysma geometry
and verify the body-spanning deformation correction.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceforge.anatomy.platysma import PlatysmaHandler
from faceforge.anatomy.expression_muscles import ExprMuscleData
from faceforge.core.math_utils import quat_from_euler, quat_identity, vec3
from faceforge.core.mesh import MeshInstance, BufferGeometry
from faceforge.core.material import Material
from faceforge.core.scene_graph import SceneNode


# ── Helpers ───────────────────────────────────────────────────────────

def _make_platysma_muscle(
    name: str = "Platysma R",
    n_verts: int = 30,
    y_range: tuple[float, float] = (-25.0, 5.0),
) -> ExprMuscleData:
    """Create a synthetic Platysma muscle for testing.

    Vertical column from y_range[0] (clavicle/body end) to y_range[1]
    (mandible/skull end).
    """
    y_vals = np.linspace(y_range[0], y_range[1], n_verts)
    positions = np.zeros((n_verts, 3), dtype=np.float32)
    positions[:, 1] = y_vals
    positions[:, 0] = 2.0 if "R" in name else -2.0  # slight X offset for side
    normals = np.tile([0.0, 0.0, 1.0], (n_verts, 1)).astype(np.float32)

    centroid = positions.mean(axis=0).astype(np.float64)
    diff = centroid.astype(np.float32) - positions
    dists = np.linalg.norm(diff, axis=1).astype(np.float32)
    safe_dists = np.maximum(dists, 1e-6)
    fiber_dirs = diff / safe_dists[:, np.newaxis]

    geom = BufferGeometry(
        positions=positions.ravel(),
        normals=normals.ravel(),
        vertex_count=n_verts,
    )
    mesh = MeshInstance(name=name, geometry=geom, material=Material())

    return ExprMuscleData(
        mesh=mesh,
        node=SceneNode(name=name),
        defn={"name": name, "stl": "TEST", "auMap": {"AU20": 0.5}},
        au_map={"AU20": 0.5},
        rest_positions=positions.ravel().copy(),
        rest_normals=normals.ravel().copy(),
        centroid=centroid,
        fiber_dirs=fiber_dirs,
        fiber_dists=dists,
        vert_count=n_verts,
        base_color=(0.8, 0.5, 0.5),
    )


def _make_non_platysma_muscle(name: str = "Orbicularis Oris R") -> ExprMuscleData:
    """Create a non-Platysma expression muscle."""
    positions = np.random.randn(10, 3).astype(np.float32)
    normals = np.tile([0, 0, 1], (10, 1)).astype(np.float32)
    centroid = positions.mean(axis=0).astype(np.float64)
    diff = centroid.astype(np.float32) - positions
    dists = np.linalg.norm(diff, axis=1).astype(np.float32)
    safe_dists = np.maximum(dists, 1e-6)
    fiber_dirs = diff / safe_dists[:, np.newaxis]

    geom = BufferGeometry(
        positions=positions.ravel(),
        normals=normals.ravel(),
        vertex_count=10,
    )
    mesh = MeshInstance(name=name, geometry=geom, material=Material())

    return ExprMuscleData(
        mesh=mesh,
        node=SceneNode(name=name),
        defn={"name": name, "stl": "TEST", "auMap": {"AU12": 0.8}},
        au_map={"AU12": 0.8},
        rest_positions=positions.ravel().copy(),
        rest_normals=normals.ravel().copy(),
        centroid=centroid,
        fiber_dirs=fiber_dirs,
        fiber_dists=dists,
        vert_count=10,
        base_color=(0.8, 0.5, 0.5),
    )


# ── Tests ─────────────────────────────────────────────────────────────

class TestPlatysmaRegistration:
    """PlatysmaHandler should correctly identify Platysma muscles."""

    def test_registers_platysma_only(self):
        plat_r = _make_platysma_muscle("Platysma R")
        plat_l = _make_platysma_muscle("Platysma L")
        other = _make_non_platysma_muscle("Orbicularis Oris R")

        handler = PlatysmaHandler(head_pivot=(0, -1.5, 10.4))
        handler.register([plat_r, plat_l, other])

        assert handler.registered
        assert len(handler._platysma) == 2

    def test_no_platysma_not_registered(self):
        other = _make_non_platysma_muscle()
        handler = PlatysmaHandler(head_pivot=(0, -1.5, 10.4))
        handler.register([other])

        assert not handler.registered

    def test_empty_list(self):
        handler = PlatysmaHandler(head_pivot=(0, -1.5, 10.4))
        handler.register([])
        assert not handler.registered


class TestPlatysmaLowerStaysNearClavicle:
    """Body-end vertices should not drift far during head rotation."""

    def test_lower_verts_restrained_during_yaw(self):
        plat_r = _make_platysma_muscle("Platysma R", n_verts=30)
        handler = PlatysmaHandler(head_pivot=(0, -1.5, 10.4))
        handler.register([plat_r])

        rest_pos = plat_r.mesh.geometry.positions.copy()

        # Apply head rotation
        head_q = quat_from_euler(0.0, 0.8, 0.0, "YXZ")  # yaw left
        handler.update(head_q)

        posed_pos = np.asarray(plat_r.mesh.geometry.positions).reshape(-1, 3)
        rest_3d = rest_pos.reshape(-1, 3)

        # Lower 20% by Y (body end)
        y_vals = rest_3d[:, 1]
        y_thresh = np.percentile(y_vals, 20)
        lower_mask = y_vals <= y_thresh

        lower_disp = np.linalg.norm(posed_pos[lower_mask] - rest_3d[lower_mask], axis=1)
        max_lower_disp = float(lower_disp.max())

        # Lower vertices should be partially restrained by the correction
        # (not moving as much as they would with pure group rotation)
        assert max_lower_disp < 15.0, (
            f"Lower Platysma verts moved too far: {max_lower_disp:.2f}"
        )


class TestPlatysmaUpperFollowsHead:
    """Skull-end vertices should follow head rotation."""

    def test_upper_verts_move(self):
        plat_r = _make_platysma_muscle("Platysma R", n_verts=30)
        handler = PlatysmaHandler(head_pivot=(0, -1.5, 10.4))
        handler.register([plat_r])

        rest_pos = plat_r.mesh.geometry.positions.copy()

        # Apply head rotation
        head_q = quat_from_euler(0.0, 0.8, 0.0, "YXZ")
        handler.update(head_q)

        posed_pos = np.asarray(plat_r.mesh.geometry.positions).reshape(-1, 3)
        rest_3d = rest_pos.reshape(-1, 3)

        # Upper 20% by Y (skull end)
        y_vals = rest_3d[:, 1]
        y_thresh = np.percentile(y_vals, 80)
        upper_mask = y_vals >= y_thresh

        upper_disp = np.linalg.norm(posed_pos[upper_mask] - rest_3d[upper_mask], axis=1)
        # Upper vertices should still follow head rotation (not be zero)
        # With 0.8 yaw, they should have some displacement
        assert upper_disp.mean() > 0.01, (
            f"Upper Platysma verts didn't follow head: mean disp={upper_disp.mean():.4f}"
        )


class TestPlatysmaIdentityNoChange:
    """Identity quaternion should produce no correction."""

    def test_identity_no_change(self):
        plat_r = _make_platysma_muscle("Platysma R", n_verts=30)
        handler = PlatysmaHandler(head_pivot=(0, -1.5, 10.4))
        handler.register([plat_r])

        rest_pos = plat_r.mesh.geometry.positions.copy()

        handler.update(quat_identity())

        posed_pos = plat_r.mesh.geometry.positions
        np.testing.assert_allclose(posed_pos, rest_pos, atol=1e-6)


class TestPlatysmaAUContractionPreserved:
    """AU-driven contraction should still work after Platysma correction."""

    def test_au_contraction_then_correction(self):
        plat_r = _make_platysma_muscle("Platysma R", n_verts=30)
        handler = PlatysmaHandler(head_pivot=(0, -1.5, 10.4))
        handler.register([plat_r])

        # Simulate AU-driven contraction (expression muscle update sets positions)
        rest = plat_r.rest_positions.reshape(-1, 3)
        centroid = plat_r.centroid.astype(np.float32)
        contraction = 0.5 * 0.15  # activation * MAX_CONTRACTION
        diff = centroid - rest
        dists = np.linalg.norm(diff, axis=1, keepdims=True)
        safe_dists = np.maximum(dists, 1e-6)
        fiber_dirs = diff / safe_dists
        disp = fiber_dirs * (dists * contraction)
        contracted_pos = rest + disp

        plat_r.mesh.geometry.positions[:] = contracted_pos.ravel()

        contracted_before = plat_r.mesh.geometry.positions.copy()

        # Apply head rotation correction
        head_q = quat_from_euler(0.0, 0.3, 0.0, "YXZ")
        handler.update(head_q)

        corrected_pos = np.asarray(plat_r.mesh.geometry.positions).reshape(-1, 3)
        contracted_3d = contracted_before.reshape(-1, 3)

        # Upper vertices (skull end) should still show contraction effect
        # relative to rest, even after correction
        y_vals = rest[:, 1]
        y_thresh = np.percentile(y_vals, 80)
        upper_mask = y_vals >= y_thresh

        # The upper verts should differ from rest (contraction preserved)
        upper_rest_diff = np.linalg.norm(corrected_pos[upper_mask] - rest[upper_mask], axis=1)
        assert upper_rest_diff.mean() > 0.01, (
            "AU contraction was lost after Platysma correction"
        )


# ── Fascia pinning tests ─────────────────────────────────────────────

def _make_platysma_with_fascia_config(
    name: str = "Platysma R",
    medial: str = "pectoral_R",
    lateral: str = "deltoid_R",
) -> ExprMuscleData:
    """Create a Platysma muscle with fasciaRegions in its defn."""
    md = _make_platysma_muscle(name)
    md.defn["fasciaRegions"] = {"medial": medial, "lateral": lateral}
    return md


def _make_mock_fascia():
    """Create a mock FasciaSystem with deterministic positions."""
    from faceforge.anatomy.fascia import FasciaRegion, FasciaSystem
    from faceforge.anatomy.bone_anchors import BoneAnchorRegistry

    bone_a = SceneNode(name="BoneA")
    bone_a.set_position(5.0, -30.0, 5.0)
    bone_a.update_world_matrix(force=True)
    bone_b = SceneNode(name="BoneB")
    bone_b.set_position(-5.0, -30.0, 5.0)
    bone_b.update_world_matrix(force=True)

    registry = BoneAnchorRegistry()
    registry.register_bones({"BoneA": bone_a, "BoneB": bone_b})
    registry.snapshot_rest_positions()

    regions = [
        FasciaRegion(name="pectoral_R", bone_names=["BoneA"], bone_weights=[1.0], side="R"),
        FasciaRegion(name="deltoid_R", bone_names=["BoneA"], bone_weights=[1.0], side="R"),
        FasciaRegion(name="pectoral_L", bone_names=["BoneB"], bone_weights=[1.0], side="L"),
        FasciaRegion(name="deltoid_L", bone_names=["BoneB"], bone_weights=[1.0], side="L"),
    ]
    fs = FasciaSystem(regions, registry)
    fs.snapshot_rest()
    return fs, bone_a, bone_b


class TestPlatysmaFasciaPinning:
    """Body-end vertices should be pulled toward fascia targets."""

    def test_fascia_delta_pulls_body_end(self):
        plat_r = _make_platysma_with_fascia_config("Platysma R")
        handler = PlatysmaHandler(head_pivot=(0, -1.5, 10.4))
        handler.register([plat_r])

        fascia, bone_a, _ = _make_mock_fascia()
        handler.set_fascia_system(fascia)

        # Record rest positions
        rest_pos = plat_r.mesh.geometry.positions.copy().reshape(-1, 3)

        # Move the bone to simulate body animation
        bone_a.set_position(8.0, -30.0, 5.0)  # moved +3 in X
        bone_a.update_world_matrix(force=True)

        # Apply head rotation to trigger the correction
        head_q = quat_from_euler(0.0, 0.5, 0.0, "YXZ")
        handler.update(head_q)

        posed_pos = np.asarray(plat_r.mesh.geometry.positions).reshape(-1, 3)

        # Body-end vertices (lowest Y) should have been pulled by fascia delta
        y_vals = rest_pos[:, 1]
        y_thresh = np.percentile(y_vals, 15)
        lower_mask = y_vals <= y_thresh

        # At least some body-end verts should differ from a non-fascia scenario
        # We just verify they moved (fascia was applied)
        lower_disp = np.linalg.norm(posed_pos[lower_mask] - rest_pos[lower_mask], axis=1)
        assert lower_disp.max() > 0.01, "Fascia pinning had no effect on body-end verts"


class TestPlatysmaFasciaAssignment:
    """Nearest-target assignment with backward-compatible dict format."""

    def test_legacy_dict_format(self):
        """Legacy dict config still works via nearest-target (2 regions)."""
        plat_r = _make_platysma_with_fascia_config("Platysma R")
        handler = PlatysmaHandler(head_pivot=(0, -1.5, 10.4))
        handler.register([plat_r])

        fascia, _, _ = _make_mock_fascia()
        handler.set_fascia_system(fascia)

        pd = handler._platysma[0]
        assert pd.fascia_assignments is not None
        assert pd.fascia_region_names is not None
        assert len(pd.fascia_region_names) == 2
        assert "pectoral_R" in pd.fascia_region_names
        assert "deltoid_R" in pd.fascia_region_names

        # Body-end vertices should have non-negative assignments
        fracs = pd.spine_fracs
        body_end_mask = fracs < 0.50
        body_assignments = pd.fascia_assignments[body_end_mask]
        assert (body_assignments >= 0).all(), "Body-end verts should be assigned to a region"

        # Skull-end vertices should be unassigned (-1)
        skull_end_mask = fracs >= 0.50
        skull_assignments = pd.fascia_assignments[skull_end_mask]
        assert (skull_assignments == -1).all(), "Skull-end verts should not have fascia assignment"


class TestPlatysmaMultiZoneAssignment:
    """Multi-zone arc distribution assignment with 5 regions."""

    def test_five_region_distribution(self):
        """Vertices should distribute across multiple regions when targets are spread."""
        from faceforge.anatomy.fascia import FasciaRegion, FasciaSystem
        from faceforge.anatomy.bone_anchors import BoneAnchorRegistry

        # Create 5 bones spread in an arc from medial to posterior
        bone_positions = {
            "B_pec": (2.0, -30.0, 8.0),     # medial/anterior
            "B_inv": (4.0, -30.0, 6.0),      # anteromedial
            "B_sup": (8.0, -30.0, 3.0),      # anterolateral
            "B_del": (12.0, -30.0, -1.0),    # lateral
            "B_tra": (10.0, -30.0, -6.0),    # posterolateral
        }
        nodes = {}
        for name, pos in bone_positions.items():
            node = SceneNode(name=name)
            node.set_position(*pos)
            node.update_world_matrix(force=True)
            nodes[name] = node

        registry = BoneAnchorRegistry()
        registry.register_bones(nodes)
        registry.snapshot_rest_positions()

        regions = [
            FasciaRegion(name="pectoral_R", bone_names=["B_pec"], bone_weights=[1.0], side="R"),
            FasciaRegion(name="investing_R", bone_names=["B_inv"], bone_weights=[1.0], side="R"),
            FasciaRegion(name="supraclavicular_R", bone_names=["B_sup"], bone_weights=[1.0], side="R"),
            FasciaRegion(name="deltoid_R", bone_names=["B_del"], bone_weights=[1.0], side="R"),
            FasciaRegion(name="trapezius_R", bone_names=["B_tra"], bone_weights=[1.0], side="R"),
        ]
        fs = FasciaSystem(regions, registry)
        fs.snapshot_rest()

        # Create a wide Platysma with X range matching the arc
        n_verts = 60
        y_vals = np.linspace(-30.0, 5.0, n_verts)
        x_vals = np.linspace(2.0, 12.0, n_verts)  # medial to lateral
        z_vals = np.linspace(8.0, -6.0, n_verts)   # anterior to posterior
        positions = np.zeros((n_verts, 3), dtype=np.float32)
        positions[:, 0] = x_vals
        positions[:, 1] = y_vals
        positions[:, 2] = z_vals

        normals = np.tile([0.0, 0.0, 1.0], (n_verts, 1)).astype(np.float32)
        centroid = positions.mean(axis=0).astype(np.float64)
        diff = centroid.astype(np.float32) - positions
        dists = np.linalg.norm(diff, axis=1).astype(np.float32)
        safe_dists = np.maximum(dists, 1e-6)
        fiber_dirs = diff / safe_dists[:, np.newaxis]

        geom = BufferGeometry(
            positions=positions.ravel(),
            normals=normals.ravel(),
            vertex_count=n_verts,
        )
        mesh = MeshInstance(name="Platysma R", geometry=geom, material=Material())

        md = ExprMuscleData(
            mesh=mesh,
            node=SceneNode(name="Platysma R"),
            defn={
                "name": "Platysma R",
                "stl": "TEST",
                "auMap": {"AU20": 0.5},
                "fasciaRegions": [
                    "pectoral_R", "investing_R", "supraclavicular_R",
                    "deltoid_R", "trapezius_R",
                ],
            },
            au_map={"AU20": 0.5},
            rest_positions=positions.ravel().copy(),
            rest_normals=normals.ravel().copy(),
            centroid=centroid,
            fiber_dirs=fiber_dirs,
            fiber_dists=dists,
            vert_count=n_verts,
            base_color=(0.8, 0.5, 0.5),
        )

        handler = PlatysmaHandler(head_pivot=(0, -1.5, 10.4))
        handler.register([md])
        handler.set_fascia_system(fs)

        pd = handler._platysma[0]
        assert pd.fascia_assignments is not None
        assert pd.fascia_region_names is not None
        assert len(pd.fascia_region_names) == 5

        # Body-end vertices should be distributed across >2 regions
        body_mask = pd.spine_fracs < 0.50
        body_assignments = pd.fascia_assignments[body_mask]
        unique_regions = set(body_assignments[body_assignments >= 0].tolist())
        assert len(unique_regions) >= 3, (
            f"Expected vertices distributed across >=3 regions, got {unique_regions}"
        )


class TestPlatysmaAntiAccumulation:
    """Platysma corrections must not compound across frames."""

    def test_repeated_updates_stable(self):
        """Calling update() multiple times with same head_quat should be stable.

        Simulates the expression-muscle early-exit cache: between calls,
        expression muscles DON'T rewrite positions, so the mesh retains
        the previous Platysma correction.  Without anti-accumulation,
        corrections would compound endlessly.
        """
        plat_r = _make_platysma_muscle("Platysma R")
        handler = PlatysmaHandler(head_pivot=(0, -1.5, 10.4))
        handler.register([plat_r])

        rest_pos = plat_r.mesh.geometry.positions.copy().reshape(-1, 3)

        head_q = quat_from_euler(0.0, 0.5, 0.0, "YXZ")

        # Frame 1: expression muscles wrote fresh positions
        handler.update(head_q)
        frame1_pos = plat_r.mesh.geometry.positions.copy().reshape(-1, 3)

        # Frame 2: expression muscles SKIP (early-exit cache) —
        # mesh still has frame 1's Platysma correction.
        # update() should detect this and start from cached AU positions.
        handler.update(head_q)
        frame2_pos = plat_r.mesh.geometry.positions.copy().reshape(-1, 3)

        # Frame 3: same again
        handler.update(head_q)
        frame3_pos = plat_r.mesh.geometry.positions.copy().reshape(-1, 3)

        # All frames should produce the same result
        np.testing.assert_allclose(
            frame2_pos, frame1_pos, atol=1e-5,
            err_msg="Frame 2 differs from frame 1 — correction is accumulating!",
        )
        np.testing.assert_allclose(
            frame3_pos, frame1_pos, atol=1e-5,
            err_msg="Frame 3 differs from frame 1 — correction is accumulating!",
        )

    def test_identity_restores_au_positions(self):
        """Going to identity head rotation should restore AU-deformed positions."""
        plat_r = _make_platysma_muscle("Platysma R")
        handler = PlatysmaHandler(head_pivot=(0, -1.5, 10.4))
        handler.register([plat_r])

        rest_pos = plat_r.mesh.geometry.positions.copy()

        # Apply rotation
        head_q = quat_from_euler(0.0, 0.5, 0.0, "YXZ")
        handler.update(head_q)

        corrected = plat_r.mesh.geometry.positions.copy()
        assert not np.allclose(corrected, rest_pos, atol=1e-4), "Correction should change positions"

        # Return to identity — should restore AU positions (which are rest for this test)
        handler.update(quat_identity())

        restored = plat_r.mesh.geometry.positions.copy()
        np.testing.assert_allclose(
            restored, rest_pos, atol=1e-5,
            err_msg="Identity did not restore AU-deformed positions",
        )

    def test_expr_muscle_update_detected(self):
        """After expression muscles rewrite positions, Platysma should use fresh data."""
        plat_r = _make_platysma_muscle("Platysma R")
        handler = PlatysmaHandler(head_pivot=(0, -1.5, 10.4))
        handler.register([plat_r])

        head_q = quat_from_euler(0.0, 0.5, 0.0, "YXZ")

        # Frame 1
        handler.update(head_q)
        frame1_pos = plat_r.mesh.geometry.positions.copy()

        # Simulate expression muscles updating with new AU deformation
        new_au_pos = plat_r.mesh.geometry.positions.copy()
        new_au_pos += 0.5  # offset all positions
        plat_r.mesh.geometry.positions[:] = new_au_pos

        # Frame 2: expression muscles wrote new data — Platysma should detect
        handler.update(head_q)
        frame2_pos = plat_r.mesh.geometry.positions.copy()

        # Frame 2 should differ from frame 1 because input changed
        assert not np.allclose(frame2_pos, frame1_pos, atol=0.1), (
            "Platysma should produce different result with different AU input"
        )


class TestPlatysmaNoFasciaFallback:
    """Without fascia wired, behaviour should be unchanged."""

    def test_no_fascia_no_crash(self):
        plat_r = _make_platysma_muscle("Platysma R")
        handler = PlatysmaHandler(head_pivot=(0, -1.5, 10.4))
        handler.register([plat_r])

        # Don't call set_fascia_system
        rest_pos = plat_r.mesh.geometry.positions.copy()

        head_q = quat_from_euler(0.0, 0.5, 0.0, "YXZ")
        handler.update(head_q)

        # Should still work without error
        posed_pos = plat_r.mesh.geometry.positions
        assert posed_pos.shape == rest_pos.shape

    def test_no_fascia_regions_config(self):
        """Platysma without fasciaRegions in defn still works."""
        plat_r = _make_platysma_muscle("Platysma R")
        # defn does NOT have fasciaRegions
        handler = PlatysmaHandler(head_pivot=(0, -1.5, 10.4))
        handler.register([plat_r])

        fascia, _, _ = _make_mock_fascia()
        handler.set_fascia_system(fascia)

        pd = handler._platysma[0]
        assert pd.fascia_assignments is None
        assert pd.fascia_region_names is None
