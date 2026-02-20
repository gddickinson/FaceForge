"""Unit tests for neck muscle bone-pinning constraint.

These tests create minimal NeckMuscleSystem instances with synthetic
muscle geometry and mock bone registries to verify the pinning behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceforge.anatomy.bone_anchors import BoneAnchorRegistry
from faceforge.anatomy.neck_muscles import NeckMuscleSystem, NeckMuscleData, BODY_FOLLOW_BASE
from faceforge.core.math_utils import quat_from_euler, quat_identity, vec3
from faceforge.core.mesh import MeshInstance, BufferGeometry
from faceforge.core.material import Material
from faceforge.core.scene_graph import SceneNode


# ── Helpers ───────────────────────────────────────────────────────────

def _make_synthetic_muscle(
    name: str = "Test Muscle R",
    lower_attach: str = "shoulder",
    lower_bones: list[str] | None = None,
    body_follow_override: float | None = None,
    n_verts: int = 20,
    y_range: tuple[float, float] = (-20.0, 5.0),
) -> tuple[dict, MeshInstance, SceneNode]:
    """Create a synthetic muscle definition, mesh, and node.

    The muscle is a vertical column of vertices from y_range[0] (body end)
    to y_range[1] (skull end).
    """
    defn = {
        "name": name,
        "stl": "TEST",
        "color": 0xc85050,
        "upperLevel": 0,
        "lowerLevel": 10,
        "headAttachFrac": 0.3,
        "lowerAttach": lower_attach,
    }
    if lower_bones is not None:
        defn["lowerBones"] = lower_bones
    if body_follow_override is not None:
        defn["bodyFollowOverride"] = body_follow_override

    # Generate vertices along Y axis
    y_vals = np.linspace(y_range[0], y_range[1], n_verts)
    positions = np.zeros((n_verts, 3), dtype=np.float32)
    positions[:, 1] = y_vals
    positions[:, 0] = np.sin(y_vals * 0.2)  # slight X variation
    normals = np.tile([0.0, 0.0, 1.0], (n_verts, 1)).astype(np.float32)

    geom = BufferGeometry(
        positions=positions.ravel(),
        normals=normals.ravel(),
        vertex_count=n_verts,
    )
    mesh = MeshInstance(name=name, geometry=geom, material=Material())
    node = SceneNode(name=name)

    return defn, mesh, node


def _build_system_with_muscle(
    defn: dict,
    mesh: MeshInstance,
    node: SceneNode,
    bone_registry: BoneAnchorRegistry | None = None,
) -> NeckMuscleSystem:
    """Build a NeckMuscleSystem from a single pre-built muscle.

    Manually constructs the system to avoid STL loading.
    """
    system = NeckMuscleSystem([defn], jaw_pivot=(0.0, -1.5, 10.4))

    # Manually populate the muscle data (bypassing load())
    rest_pos = mesh.geometry.positions.copy()
    rest_nrm = mesh.geometry.normals.copy()
    vert_count = mesh.geometry.vertex_count
    lower_attach = defn.get("lowerAttach", "shoulder")

    spine_fracs, upper_frac, lower_frac = system._compute_spine_fractions(
        rest_pos, vert_count, defn,
    )
    body_follow_frac = system._compute_body_follow(lower_attach, defn)

    frac_range = upper_frac - lower_frac
    if frac_range > 1e-6:
        t = np.clip((spine_fracs - lower_frac) / frac_range, 0.0, 1.0)
        spine_fracs = body_follow_frac + t * (upper_frac - body_follow_frac)

    md = NeckMuscleData(
        mesh=mesh,
        node=node,
        defn=defn,
        rest_positions=rest_pos,
        rest_normals=rest_nrm,
        vert_count=vert_count,
        spine_fracs=spine_fracs,
        upper_frac=upper_frac,
        lower_frac=lower_frac,
        body_follow_frac=body_follow_frac,
        lower_attach=lower_attach,
    )
    system._init_fiber_geometry(md)
    system._muscles.append(md)

    if bone_registry is not None:
        system.set_bone_registry(bone_registry)

    return system


def _make_bone_registry(
    bone_name: str = "Right Clavicle",
    rest_pos: tuple[float, float, float] = (5.0, -25.0, 0.0),
    current_pos: tuple[float, float, float] | None = None,
) -> BoneAnchorRegistry:
    """Create a registry with a single bone."""
    reg = BoneAnchorRegistry()
    node = SceneNode(name=bone_name)
    node.set_position(*rest_pos)
    node.mark_dirty()
    node.update_world_matrix(force=True)
    reg.register_bones({bone_name: node})
    reg.snapshot_rest_positions()

    if current_pos is not None:
        node.set_position(*current_pos)
        node.mark_dirty()
        node.update_world_matrix(force=True)

    return reg


# ── Tests ─────────────────────────────────────────────────────────────

class TestPinKeepsLowerVerticesNearBone:
    """Lower-end vertices should stay within threshold of bone position."""

    def test_pin_keeps_lower_near_bone_during_yaw(self):
        defn, mesh, node = _make_synthetic_muscle(
            lower_bones=["Right Clavicle"],
        )
        reg = _make_bone_registry(rest_pos=(5.0, -25.0, 0.0))
        system = _build_system_with_muscle(defn, mesh, node, bone_registry=reg)

        # Apply head rotation (yaw left)
        head_q = quat_from_euler(0.0, 0.5, 0.0, "YXZ")
        system.update(head_q)

        # Get lower-end vertices (lowest 15% by spine frac)
        md = system.muscle_data[0]
        fracs = md.spine_fracs
        threshold = np.percentile(fracs, 15)
        mask = fracs <= threshold

        posed_pos = md.mesh.geometry.positions.reshape(-1, 3)
        rest_pos = md.rest_positions.reshape(-1, 3)

        lower_posed = posed_pos[mask]
        lower_rest = rest_pos[mask]

        # Lower verts should be closer to rest than without pinning
        max_disp = np.linalg.norm(lower_posed - lower_rest, axis=1).max()
        # With pinning, lower verts should not move far (< 5 units)
        assert max_disp < 5.0, f"Lower vertex displacement too large: {max_disp:.2f}"


class TestPinAllowsUpperVertexFreedom:
    """Upper-end vertices should still follow head rotation."""

    def test_upper_verts_move_with_head(self):
        defn, mesh, node = _make_synthetic_muscle(
            lower_bones=["Right Clavicle"],
        )
        reg = _make_bone_registry(rest_pos=(5.0, -25.0, 0.0))
        system = _build_system_with_muscle(defn, mesh, node, bone_registry=reg)

        # Apply significant yaw rotation
        head_q = quat_from_euler(0.0, 0.8, 0.0, "YXZ")
        system.update(head_q)

        md = system.muscle_data[0]
        fracs = md.spine_fracs
        upper_mask = fracs >= np.percentile(fracs, 85)

        posed_pos = md.mesh.geometry.positions.reshape(-1, 3)
        rest_pos = md.rest_positions.reshape(-1, 3)

        upper_posed = posed_pos[upper_mask]
        upper_rest = rest_pos[upper_mask]

        # Upper vertices should have moved significantly
        max_disp = np.linalg.norm(upper_posed - upper_rest, axis=1).max()
        assert max_disp > 0.1, f"Upper vertices didn't move enough: {max_disp:.4f}"


class TestPinStrengthZero:
    """Setting pin strength to 0 should produce same result as no pinning."""

    def test_zero_pin_strength_no_effect(self):
        defn, mesh, node = _make_synthetic_muscle(
            lower_bones=["Right Clavicle"],
        )

        # Build system WITHOUT bone registry (no pinning)
        system_no_pin = _build_system_with_muscle(defn, mesh, node, bone_registry=None)

        # Apply rotation
        head_q = quat_from_euler(0.0, 0.5, 0.0, "YXZ")
        system_no_pin.update(head_q)
        pos_no_pin = system_no_pin.muscle_data[0].mesh.geometry.positions.copy()

        # Now build system WITH bone registry but _PIN_STRENGTH = 0
        defn2, mesh2, node2 = _make_synthetic_muscle(
            lower_bones=["Right Clavicle"],
        )
        reg = _make_bone_registry(rest_pos=(5.0, -25.0, 0.0))
        system_pin = _build_system_with_muscle(defn2, mesh2, node2, bone_registry=reg)

        original_strength = NeckMuscleSystem._PIN_STRENGTH
        try:
            NeckMuscleSystem._PIN_STRENGTH = 0.0
            system_pin.update(head_q)
            pos_pin = system_pin.muscle_data[0].mesh.geometry.positions.copy()
        finally:
            NeckMuscleSystem._PIN_STRENGTH = original_strength

        # Results should be identical with zero pin strength
        np.testing.assert_allclose(pos_pin, pos_no_pin, atol=1e-4)


class TestPerMuscleBoneAnchor:
    """Per-muscle bone anchor should differ from global average."""

    def test_per_muscle_anchor_uses_specific_bone(self):
        reg = BoneAnchorRegistry()
        node_r = SceneNode(name="Right Clavicle")
        node_r.set_position(10.0, -20.0, 0.0)
        node_r.mark_dirty()
        node_r.update_world_matrix(force=True)
        node_l = SceneNode(name="Left Clavicle")
        node_l.set_position(-10.0, -20.0, 0.0)
        node_l.mark_dirty()
        node_l.update_world_matrix(force=True)

        reg.register_bones({"Right Clavicle": node_r, "Left Clavicle": node_l})
        reg.snapshot_rest_positions()

        # SCM R should use Right Clavicle only
        scm_r = reg.get_muscle_anchor("SCM R", ["Right Clavicle"])
        np.testing.assert_allclose(scm_r, [10.0, -20.0, 0.0], atol=1e-6)

        # SCM L should use Left Clavicle only
        scm_l = reg.get_muscle_anchor("SCM L", ["Left Clavicle"])
        np.testing.assert_allclose(scm_l, [-10.0, -20.0, 0.0], atol=1e-6)

        # Global average would be [0, -20, 0] — different from per-muscle
        global_avg = reg.get_muscle_anchor(
            "global", ["Right Clavicle", "Left Clavicle"]
        )
        np.testing.assert_allclose(global_avg, [0.0, -20.0, 0.0], atol=1e-6)

        assert not np.allclose(scm_r, global_avg)
        assert not np.allclose(scm_l, global_avg)


class TestBodyFollowOverride:
    """bodyFollowOverride in config should take precedence."""

    def test_override_changes_follow_frac(self):
        system = NeckMuscleSystem([], jaw_pivot=(0, -1.5, 10.4))

        # Without override: uses base
        defn_no_override = {"lowerAttach": "shoulder"}
        frac = system._compute_body_follow("shoulder", defn_no_override)
        assert frac == BODY_FOLLOW_BASE["shoulder"]

        # With override: uses override value
        defn_override = {"lowerAttach": "shoulder", "bodyFollowOverride": 0.25}
        frac_override = system._compute_body_follow("shoulder", defn_override)
        assert frac_override == 0.25
        assert frac_override != BODY_FOLLOW_BASE["shoulder"]
