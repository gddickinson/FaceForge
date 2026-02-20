"""Integration tests verifying the fascia constraint system works end-to-end.

These tests use the headless loader to replicate the full loading pipeline
and then verify that Platysma body-end vertices are actually affected by
the fascia pinning system during head rotation.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceforge.core.state import FaceState
from faceforge.core.math_utils import quat_identity


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def headless_scene():
    """Load the full headless scene once for all tests in this module."""
    from tools.headless_loader import load_headless_scene
    return load_headless_scene()


# ── Pipeline wiring tests ────────────────────────────────────────────

class TestFasciaPipelineWiring:
    """Verify fascia system is created and wired by the loading pipeline."""

    def test_pipeline_has_fascia(self, headless_scene):
        """LoadingPipeline should create a FasciaSystem after skeleton load."""
        pipeline = headless_scene.pipeline
        assert pipeline.fascia is not None, (
            "pipeline.fascia is None — FasciaSystem not created in load_body_skeleton()"
        )

    def test_pipeline_has_platysma(self, headless_scene):
        """LoadingPipeline should create a PlatysmaHandler during head load."""
        pipeline = headless_scene.pipeline
        assert pipeline.platysma is not None, (
            "pipeline.platysma is None — PlatysmaHandler not created in load_head()"
        )

    def test_platysma_is_registered(self, headless_scene):
        """PlatysmaHandler should find Platysma R/L in expression muscles."""
        pipeline = headless_scene.pipeline
        if pipeline.platysma is None:
            pytest.skip("No platysma handler")
        assert pipeline.platysma.registered, (
            "Platysma handler registered=False — no Platysma muscles found"
        )

    def test_platysma_has_fascia_wired(self, headless_scene):
        """PlatysmaHandler._fascia should be set by load_body_skeleton()."""
        pipeline = headless_scene.pipeline
        if pipeline.platysma is None:
            pytest.skip("No platysma handler")
        assert pipeline.platysma._fascia is not None, (
            "PlatysmaHandler._fascia is None — set_fascia_system() never called"
        )

    def test_fascia_has_rest_targets(self, headless_scene):
        """FasciaSystem should have cached rest-pose targets after snapshot."""
        pipeline = headless_scene.pipeline
        if pipeline.fascia is None:
            pytest.skip("No fascia system")

        regions_with_targets = 0
        for name in pipeline.fascia.region_names:
            if pipeline.fascia.get_target_rest(name) is not None:
                regions_with_targets += 1

        assert regions_with_targets > 0, (
            f"No fascia regions have rest targets — "
            f"bones may not be registered. Regions: {pipeline.fascia.region_names}"
        )

    def test_fascia_region_names(self, headless_scene):
        """Factory should produce the 16 expected anatomical regions."""
        pipeline = headless_scene.pipeline
        if pipeline.fascia is None:
            pytest.skip("No fascia system")
        names = set(pipeline.fascia.region_names)
        expected = {
            "pectoral_R", "pectoral_L",
            "deltoid_R", "deltoid_L",
            "investing_R", "investing_L",
            "supraclavicular_R", "supraclavicular_L",
            "trapezius_R", "trapezius_L",
            "scm_sternal_R", "scm_sternal_L",
            "scm_clavicular_R", "scm_clavicular_L",
            "lev_scap_R", "lev_scap_L",
        }
        assert expected == names

    def test_platysma_fascia_assignments_populated(self, headless_scene):
        """Each Platysma muscle should have fascia region assignments."""
        pipeline = headless_scene.pipeline
        if pipeline.platysma is None or not pipeline.platysma.registered:
            pytest.skip("No registered platysma")
        if pipeline.platysma._fascia is None:
            pytest.skip("No fascia wired")

        for pd in pipeline.platysma._platysma:
            name = pd.md.defn.get("name", "?")
            assert pd.fascia_assignments is not None, (
                f"{name}: fascia_assignments is None"
            )
            assert pd.fascia_region_names is not None, (
                f"{name}: fascia_region_names is None"
            )
            # Body-end verts should have non-negative assignments
            body_mask = pd.spine_fracs < 0.50
            if body_mask.any():
                body_assigns = pd.fascia_assignments[body_mask]
                assert (body_assigns >= 0).any(), (
                    f"{name}: no body-end vertices assigned to a fascia region"
                )

    def test_bone_anchors_wired(self, headless_scene):
        """BoneAnchorRegistry should be available in the pipeline."""
        pipeline = headless_scene.pipeline
        assert pipeline.bone_anchors is not None, (
            "pipeline.bone_anchors is None"
        )
        assert len(pipeline.bone_anchors.bone_names) > 0, (
            "BoneAnchorRegistry has no registered bones"
        )


# ── Functional deformation tests ─────────────────────────────────────

class TestFasciaDeformationEffect:
    """Verify that fascia pinning actually changes Platysma vertex positions."""

    def test_platysma_body_end_pinned_during_yaw(self, headless_scene):
        """Body-end vertices should differ with fascia vs without fascia."""
        from tools.headless_loader import apply_head_rotation

        pipeline = headless_scene.pipeline
        if pipeline.platysma is None or not pipeline.platysma.registered:
            pytest.skip("No registered platysma")

        pd = pipeline.platysma._platysma[0]
        md = pd.md
        rest_pos = pd.rest_positions.copy()  # (N, 3)

        # ── Test WITH fascia ──
        md.mesh.geometry.positions[:] = md.rest_positions.copy()
        face_rest = FaceState()
        apply_head_rotation(headless_scene, face_rest)

        md.mesh.geometry.positions[:] = md.rest_positions.copy()
        face_yaw = FaceState()
        face_yaw.head_yaw = 0.8
        head_q = apply_head_rotation(headless_scene, face_yaw)

        with_fascia_pos = md.mesh.geometry.positions.copy().reshape(-1, 3)

        # ── Reset ──
        md.mesh.geometry.positions[:] = md.rest_positions.copy()
        apply_head_rotation(headless_scene, face_rest)

        # ── Test WITHOUT fascia ──
        saved_fascia = pipeline.platysma._fascia
        pipeline.platysma._fascia = None

        md.mesh.geometry.positions[:] = md.rest_positions.copy()
        head_q = apply_head_rotation(headless_scene, face_yaw)

        without_fascia_pos = md.mesh.geometry.positions.copy().reshape(-1, 3)

        # Restore
        pipeline.platysma._fascia = saved_fascia
        md.mesh.geometry.positions[:] = md.rest_positions.copy()
        apply_head_rotation(headless_scene, face_rest)

        # ── Compare body-end vertices ──
        body_mask = pd.spine_fracs < 0.50
        if not body_mask.any():
            pytest.skip("No body-end vertices found")

        disp_with = np.linalg.norm(
            with_fascia_pos[body_mask] - rest_pos[body_mask], axis=1
        )
        disp_without = np.linalg.norm(
            without_fascia_pos[body_mask] - rest_pos[body_mask], axis=1
        )
        pos_diff = np.linalg.norm(
            with_fascia_pos[body_mask] - without_fascia_pos[body_mask], axis=1
        )
        max_diff = float(pos_diff.max())
        mean_diff = float(pos_diff.mean())

        print(f"\n  Fascia effect on body-end vertices:")
        print(f"    With fascia    — mean disp from rest: {disp_with.mean():.4f}")
        print(f"    Without fascia — mean disp from rest: {disp_without.mean():.4f}")
        print(f"    Max position difference: {max_diff:.4f}")
        print(f"    Mean position difference: {mean_diff:.4f}")

        assert max_diff > 0.001, (
            f"Fascia pinning had zero effect! max_diff={max_diff:.6f}. "
            f"The fascia system may not be running."
        )

    def test_platysma_update_called_in_headless(self, headless_scene):
        """Verify that platysma.update() is actually called by apply_head_rotation."""
        from tools.headless_loader import apply_head_rotation

        pipeline = headless_scene.pipeline
        if pipeline.platysma is None or not pipeline.platysma.registered:
            pytest.skip("No registered platysma")

        pd = pipeline.platysma._platysma[0]
        md = pd.md

        # Reset
        md.mesh.geometry.positions[:] = md.rest_positions.copy()
        rest_positions = md.mesh.geometry.positions.copy()

        # Apply rotation
        face = FaceState()
        face.head_yaw = 0.6
        apply_head_rotation(headless_scene, face)

        posed_positions = md.mesh.geometry.positions.copy()

        # Reset
        face_rest = FaceState()
        apply_head_rotation(headless_scene, face_rest)

        diff = np.abs(posed_positions - rest_positions).max()
        assert diff > 0.01, (
            f"Platysma positions unchanged after head rotation (max diff={diff:.6f}). "
            f"platysma.update() may not be called."
        )


# ── Fascia target validity tests ──────────────────────────────────────

class TestFasciaTargets:
    """Verify fascia targets are anatomically reasonable."""

    def test_pectoral_targets_exist(self, headless_scene):
        """Pectoral fascia targets should have valid positions."""
        pipeline = headless_scene.pipeline
        if pipeline.fascia is None:
            pytest.skip("No fascia system")

        for name in ("pectoral_R", "pectoral_L"):
            target = pipeline.fascia.get_target_rest(name)
            if target is None:
                pytest.skip(f"No rest target for {name}")
            assert np.isfinite(target).all(), f"{name} target has non-finite values"
            assert np.linalg.norm(target) > 1.0, (
                f"{name} target at origin — bones may not be loaded"
            )

    def test_deltoid_targets_lateral(self, headless_scene):
        """Deltoid fascia targets should be laterally offset from midline."""
        pipeline = headless_scene.pipeline
        if pipeline.fascia is None:
            pytest.skip("No fascia system")

        target_r = pipeline.fascia.get_target_rest("deltoid_R")
        target_l = pipeline.fascia.get_target_rest("deltoid_L")
        if target_r is None or target_l is None:
            pytest.skip("Deltoid targets not available")

        x_diff = abs(target_r[0] - target_l[0])
        assert x_diff > 0.5, (
            f"Deltoid R/L targets too close in X: R={target_r[0]:.2f}, L={target_l[0]:.2f}"
        )

    def test_current_targets_match_rest_at_rest_pose(self, headless_scene):
        """At rest pose, current targets should equal rest targets."""
        pipeline = headless_scene.pipeline
        if pipeline.fascia is None:
            pytest.skip("No fascia system")

        headless_scene.scene.update()

        for name in pipeline.fascia.region_names:
            rest = pipeline.fascia.get_target_rest(name)
            current = pipeline.fascia.get_target_current(name)
            if rest is None or current is None:
                continue
            np.testing.assert_allclose(
                current, rest, atol=0.1,
                err_msg=f"Region '{name}': current != rest at rest pose",
            )
