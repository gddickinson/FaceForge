"""Test digit curl with real hand muscle data.

Uses the actual headless loader to load real STL files, then tests
whether hand muscle vertices actually move when finger pivots are rotated.
"""

import logging
import numpy as np
import pytest

from faceforge.core.state import BodyState
from faceforge.core.math_utils import quat_from_euler


def _try_load_headless():
    """Try to load the headless scene. Skip if assets not available."""
    try:
        from tools.headless_loader import (
            load_headless_scene, load_layer, register_layer, apply_pose,
        )
        hs = load_headless_scene()
        return hs
    except Exception as e:
        pytest.skip(f"Cannot load headless scene: {e}")


class TestRealHandCurl:
    """Test with actual STL data."""

    def test_digit_chains_exist(self):
        """Verify that digit chains are built from hand skeleton."""
        hs = _try_load_headless()
        hand_chain_ids = [
            cid for name, cid in hs.chain_ids.items()
            if name.startswith("hand_")
        ]
        foot_chain_ids = [
            cid for name, cid in hs.chain_ids.items()
            if name.startswith("foot_")
        ]
        logging.info("Hand chains: %d, Foot chains: %d",
                     len(hand_chain_ids), len(foot_chain_ids))
        # Should have 10 hand chains (5 digits × 2 sides)
        assert len(hand_chain_ids) > 0, (
            f"No hand digit chains! chain_ids keys: {list(hs.chain_ids.keys())}"
        )

    def test_digit_joints_have_segments(self):
        """Verify digit chain joints have valid bone segments."""
        hs = _try_load_headless()
        hand_chain_ids = {
            cid for name, cid in hs.chain_ids.items()
            if name.startswith("hand_")
        }
        if not hand_chain_ids:
            pytest.skip("No hand digit chains available")

        digit_joints = [
            j for j in hs.skinning.joints if j.chain_id in hand_chain_ids
        ]
        joints_with_segments = [
            j for j in digit_joints
            if j.segment_start is not None and j.segment_end is not None
        ]
        for j in digit_joints:
            has_seg = j.segment_start is not None
            logging.info("  Joint %s (chain %d): segment=%s",
                         j.name, j.chain_id, has_seg)
            if has_seg:
                seg_len = np.linalg.norm(j.segment_end - j.segment_start)
                logging.info("    segment length: %.2f", seg_len)

        assert len(joints_with_segments) > 0, (
            f"No digit joints have segments! {len(digit_joints)} digit joints total"
        )

    def test_hand_muscle_curl_displacement(self):
        """Load hand muscles, register with digit chains, apply curl."""
        hs = _try_load_headless()

        # Check digit chains exist
        hand_chain_ids = {
            cid for name, cid in hs.chain_ids.items()
            if name.startswith("hand_")
        }
        if not hand_chain_ids:
            pytest.skip("No hand digit chains available")

        # Load hand muscle meshes manually (not in headless_loader yet)
        from faceforge.loaders.asset_manager import AssetManager
        assets = AssetManager()  # uses default STL_DIR
        body_root = hs.scene.children[0] if hs.scene.children else hs.scene
        result = assets.load_hand_muscles()
        body_root.add(result.group)
        hand_meshes = result.meshes

        if not hand_meshes:
            pytest.skip("No hand muscle meshes loaded")

        # Register with digit-only chains
        bindings_before = len(hs.skinning.bindings)
        for mesh in hand_meshes:
            hs.skinning.register_skin_mesh(
                mesh, is_muscle=True, allowed_chains=hand_chain_ids,
            )
        bindings_after = len(hs.skinning.bindings)
        new_bindings = bindings_after - bindings_before

        logging.info("Registered %d hand muscles → %d new bindings",
                     len(hand_meshes), new_bindings)

        if new_bindings == 0:
            # Log why: check mesh centroids vs joint positions
            for m in hand_meshes[:3]:
                if m.rest_positions is not None:
                    cent = m.rest_positions.reshape(-1, 3).mean(axis=0)
                    logging.info("  Mesh %s centroid: [%.1f, %.1f, %.1f]",
                                 m.name, *cent)
            for j in hs.skinning.joints:
                if j.chain_id in hand_chain_ids and j.segment_start is not None:
                    pos = j.rest_world[:3, 3]
                    seg_len = np.linalg.norm(j.segment_end - j.segment_start)
                    logging.info("  Joint %s pos=[%.1f,%.1f,%.1f] seg_len=%.1f",
                                 j.name, *pos, seg_len)
            pytest.fail(f"0 bindings from {len(hand_meshes)} hand muscle meshes")

        # Save rest positions
        binding = hs.skinning.bindings[bindings_before]
        rest_pos = binding.mesh.rest_positions.reshape(-1, 3).copy()

        # Show binding joint distribution
        ji_unique, ji_counts = np.unique(binding.joint_indices, return_counts=True)
        for ji, cnt in zip(ji_unique, ji_counts):
            j = hs.skinning.joints[ji]
            logging.info("  Joint %s (chain %d): %d vertices",
                         j.name, j.chain_id, cnt)

        # Apply curl via body animation
        body = BodyState()
        body.finger_curl_r = 0.8

        if hs.body_animation is not None:
            hs.body_animation.apply(body, dt=0.016)

        # Update scene graph
        hs.scene.update()

        # Force skinning update
        hs.skinning._last_signature = ""
        hs.skinning.update(body)

        # Check displacement
        new_pos = binding.mesh.geometry.positions.reshape(-1, 3)
        displacement = np.linalg.norm(new_pos - rest_pos, axis=1)
        mean_disp = displacement.mean()
        max_disp = displacement.max()
        n_moved = np.sum(displacement > 0.01)

        logging.info("Curl displacement: mean=%.3f max=%.3f moved=%d/%d",
                     mean_disp, max_disp, n_moved, len(displacement))

        assert max_disp > 0.1, (
            f"Hand muscles did NOT move with curl! max_disp={max_disp:.4f}. "
            f"This confirms the bug."
        )
