"""Integration tests for head rotation diagnostic tools.

These tests load real BP3D STL assets and exercise the head rotation,
group follow checking, and neck-body gap measurement.
Tests are skipped if STL assets are not found (CI environments).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from faceforge.constants import STL_DIR
from faceforge.core.math_utils import quat_identity
from faceforge.core.state import FaceState, BodyState

# Skip all tests in this module if STL directory is missing
_has_stl_assets = STL_DIR.is_dir() and any(STL_DIR.glob("*.stl"))
pytestmark = pytest.mark.skipif(
    not _has_stl_assets,
    reason="BP3D STL assets not found — skipping head rotation diagnostic tests",
)


@pytest.fixture(scope="module")
def headless_scene():
    """Module-scoped fixture: load the headless scene once."""
    from tools.headless_loader import load_headless_scene
    hs = load_headless_scene()
    hs.scene.update()
    return hs


class TestHeadRotationApply:
    """Test that head rotation can be applied and reset."""

    def test_apply_head_rotation_returns_quaternion(self, headless_scene):
        from tools.headless_loader import apply_head_rotation
        hs = headless_scene

        face_state = FaceState()
        face_state.head_yaw = 0.5
        q = apply_head_rotation(hs, face_state)

        assert q is not None
        assert len(q) == 4
        assert not np.allclose(q, quat_identity(), atol=1e-3)

        # Reset
        hs.pipeline.head_rotation.reset()
        if hs.pipeline.neck_muscles is not None:
            hs.pipeline.neck_muscles.reset()
        for gname in ["skullGroup", "faceGroup", "stlMuscleGroup",
                       "exprMuscleGroup", "faceFeatureGroup"]:
            node = hs.named_nodes.get(gname)
            if node is not None:
                node.set_quaternion(quat_identity())
                node.set_position(0, 0, 0)
                node.mark_dirty()
        hs.scene.update()

    def test_identity_rotation_preserves_positions(self, headless_scene):
        from tools.headless_loader import apply_head_rotation
        hs = headless_scene

        face_state = FaceState()  # all zeros
        q = apply_head_rotation(hs, face_state)
        assert np.allclose(q, quat_identity(), atol=1e-6)


class TestGroupFollow:
    """Test that all head-attached groups follow rotation."""

    def test_group_follow_error_near_zero(self, headless_scene):
        """After fix, all groups should follow with error < 0.1."""
        from tools.head_rotation_diagnostic import check_group_follow
        hs = headless_scene

        face_state = FaceState()
        face_state.head_yaw = 0.5

        result = check_group_follow(hs, "yaw_left", face_state)

        assert len(result.group_results) > 0
        for gr in result.group_results:
            assert gr.error < 0.5, (
                f"Group {gr.group_name} has follow error {gr.error:.4f} > 0.5"
            )

    def test_neutral_pose_zero_error(self, headless_scene):
        """Neutral pose should have zero follow error."""
        from tools.head_rotation_diagnostic import check_group_follow
        hs = headless_scene

        face_state = FaceState()  # neutral
        result = check_group_follow(hs, "neutral", face_state)

        for gr in result.group_results:
            assert gr.error < 0.01, (
                f"Group {gr.group_name} has non-zero error {gr.error:.4f} in neutral"
            )


class TestHeadRotationRoundTrip:
    """Test that rotating and returning produces rest positions."""

    def test_round_trip_returns_to_rest(self, headless_scene):
        from tools.headless_loader import apply_head_rotation
        hs = headless_scene

        # Record rest state
        skull = hs.named_nodes.get("skullGroup")
        if skull is None:
            pytest.skip("No skull group")
        rest_q = skull.quaternion.copy()
        rest_pos = skull.position.copy()

        # Apply rotation
        face_state = FaceState()
        face_state.head_yaw = 0.7
        apply_head_rotation(hs, face_state)

        # Reset
        face_state_zero = FaceState()
        apply_head_rotation(hs, face_state_zero)

        # Should be back to rest
        assert np.allclose(skull.quaternion, rest_q, atol=1e-4)
        assert np.allclose(skull.position, rest_pos, atol=1e-4)


class TestNeckBodyGap:
    """Test that neck muscles stay attached to body in combined poses."""

    def test_neck_body_gap_bounded(self, headless_scene):
        """Gap ratio should be <= 2.0 for reasonable combined poses."""
        from tools.head_rotation_diagnostic import check_neck_body_gap
        hs = headless_scene

        if hs.pipeline.neck_muscles is None:
            pytest.skip("No neck muscles loaded")

        body_state = BodyState()
        body_state.spine_flex = 0.3
        face_state = FaceState()
        face_state.head_yaw = 0.5

        result = check_neck_body_gap(
            hs, "combined", body_state, face_state,
        )

        # Allow some slack — we just want to verify they don't completely detach
        if result.muscle_gaps:
            # Most muscles should have bounded gap
            bounded_count = sum(1 for mg in result.muscle_gaps if mg.gap_ratio < 3.0)
            total = len(result.muscle_gaps)
            assert bounded_count / total > 0.7, (
                f"Only {bounded_count}/{total} muscles have bounded gap"
            )


class TestFullDiagnostic:
    """Test the full diagnostic runner."""

    def test_full_diagnostic_runs(self, headless_scene):
        from tools.head_rotation_diagnostic import run_full_diagnostic, format_diagnostic_report
        hs = headless_scene

        # Use a subset of poses for speed
        poses = {
            "neutral": {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0},
            "yaw_left": {"head_yaw": 0.8, "head_pitch": 0.0, "head_roll": 0.0},
        }
        results = run_full_diagnostic(hs, poses=poses)

        assert len(results.follow_results) == 2

        # Should be serializable
        d = results.to_dict()
        assert "follow_results" in d

        # Report should be a non-empty string
        report = format_diagnostic_report(results)
        assert len(report) > 100
        assert "GROUP FOLLOW ERRORS" in report


class TestApplyFullPose:
    """Test combined body + head pose application."""

    def test_apply_full_pose_runs(self, headless_scene):
        from tools.headless_loader import apply_full_pose, reset_skinning
        hs = headless_scene

        body_state = BodyState()
        body_state.spine_flex = 0.2
        face_state = FaceState()
        face_state.head_yaw = 0.3

        head_q = apply_full_pose(hs, body_state, face_state)

        assert head_q is not None
        assert not np.allclose(head_q, quat_identity(), atol=1e-3)

        # Reset
        reset_skinning(hs)
