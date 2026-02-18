"""Tests for the headless skinning diagnostic pipeline.

These tests load real BP3D STL assets and exercise the full headless
loading, registration, pose application, and scoring pipeline.
Tests are skipped if STL assets are not found (CI environments).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from faceforge.constants import STL_DIR

# Skip all tests in this module if STL directory is missing
_has_stl_assets = STL_DIR.is_dir() and any(STL_DIR.glob("*.stl"))
pytestmark = pytest.mark.skipif(
    not _has_stl_assets,
    reason="BP3D STL assets not found — skipping headless diagnostic tests",
)


@pytest.fixture(scope="module")
def headless_scene():
    """Module-scoped fixture: load the headless scene once for all tests."""
    from tools.headless_loader import load_headless_scene
    return load_headless_scene()


class TestHeadlessSceneLoads:
    """Test that the headless scene loads successfully."""

    def test_skeleton_built(self, headless_scene):
        hs = headless_scene
        assert hs.skeleton is not None, "Skeleton should be loaded"
        assert len(hs.joint_chains) > 0, "Should have at least one joint chain"

    def test_joint_chains_have_joints(self, headless_scene):
        hs = headless_scene
        assert len(hs.skinning.joints) > 0, "Skinning should have joints"

    def test_chain_ids_populated(self, headless_scene):
        hs = headless_scene
        assert "spine" in hs.chain_ids, "Should have a spine chain"

    def test_body_animation_available(self, headless_scene):
        hs = headless_scene
        assert hs.body_animation is not None, "Body animation should be set up"

    def test_scene_has_body_root(self, headless_scene):
        hs = headless_scene
        assert "bodyRoot" in hs.named_nodes


class TestSkinRegistration:
    """Test that skin meshes can be loaded and registered."""

    @pytest.fixture(scope="class")
    def skin_scene(self, headless_scene):
        """Load and register skin on a copy of the scene."""
        from tools.headless_loader import load_layer, register_layer, reset_skinning
        hs = headless_scene
        reset_skinning(hs)
        meshes = load_layer(hs, "skin")
        register_layer(hs, meshes, "skin")
        yield hs, meshes
        # Clean up after this class
        reset_skinning(hs)

    def test_skin_meshes_loaded(self, skin_scene):
        hs, meshes = skin_scene
        assert len(meshes) > 0, "Should load at least one skin mesh"

    def test_bindings_created(self, skin_scene):
        hs, meshes = skin_scene
        assert len(hs.skinning.bindings) > 0, "Should have skinning bindings"

    def test_vertices_have_rest_positions(self, skin_scene):
        hs, meshes = skin_scene
        for binding in hs.skinning.bindings:
            assert binding.mesh.rest_positions is not None


class TestPoseApplication:
    """Test that applying a pose changes vertex positions."""

    @pytest.fixture(scope="class")
    def posed_scene(self, headless_scene):
        """Register skin and apply a non-trivial pose."""
        from tools.headless_loader import (
            load_layer, register_layer, reset_skinning, apply_pose,
        )
        from faceforge.core.state import BodyState

        hs = headless_scene
        reset_skinning(hs)
        meshes = load_layer(hs, "skin")
        register_layer(hs, meshes, "skin")

        # Capture rest positions
        rest_positions = {}
        for b in hs.skinning.bindings:
            rest_positions[b.mesh.name] = b.mesh.rest_positions.copy()

        # Apply a pose that moves multiple joints
        state = BodyState()
        state.shoulder_r_abduct = 0.8
        state.hip_l_flex = 0.6
        state.spine_flex = 0.2
        apply_pose(hs, state)

        yield hs, rest_positions

        # Reset
        apply_pose(hs, BodyState())
        reset_skinning(hs)

    def test_positions_changed(self, posed_scene):
        hs, rest_positions = posed_scene
        any_changed = False
        for binding in hs.skinning.bindings:
            current = binding.mesh.geometry.positions
            rest = rest_positions.get(binding.mesh.name)
            if rest is not None:
                if not np.allclose(current, rest, atol=1e-6):
                    any_changed = True
                    break
        assert any_changed, "At least one mesh should have moved from rest pose"


class TestScoring:
    """Test that scoring produces finite numbers."""

    @pytest.fixture(scope="class")
    def scored_result(self, headless_scene):
        from tools.headless_loader import load_layer, register_layer, reset_skinning
        from tools.skinning_scorer import SkinningScorer

        hs = headless_scene
        reset_skinning(hs)
        meshes = load_layer(hs, "skin")
        register_layer(hs, meshes, "skin")

        scorer = SkinningScorer(hs, pose_names=["anatomical", "extreme_arm_raise"])
        result = scorer.evaluate()
        yield result
        reset_skinning(hs)

    def test_composite_is_finite(self, scored_result):
        assert math.isfinite(scored_result.composite)

    def test_per_pose_scores_exist(self, scored_result):
        assert len(scored_result.per_pose) == 2

    def test_per_pose_scores_are_finite(self, scored_result):
        for ps in scored_result.per_pose:
            assert math.isfinite(ps.anomaly_pct)
            assert math.isfinite(ps.max_displacement)

    def test_to_dict_serializable(self, scored_result):
        import json
        d = scored_result.to_dict()
        # Should be JSON-serializable
        json.dumps(d)


class TestResetSkinning:
    """Test that reset_skinning allows re-registration with same results."""

    def test_rebind_produces_same_count(self, headless_scene):
        from tools.headless_loader import load_layer, register_layer, reset_skinning

        hs = headless_scene
        reset_skinning(hs)

        meshes = load_layer(hs, "skin")
        register_layer(hs, meshes, "skin")
        count_1 = len(hs.skinning.bindings)

        reset_skinning(hs)
        register_layer(hs, meshes, "skin")
        count_2 = len(hs.skinning.bindings)

        assert count_1 == count_2, "Re-registration should produce same binding count"
        reset_skinning(hs)
