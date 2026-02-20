"""Reproduce hand muscle curl skinning — minimal test.

Verifies that when finger pivots are rotated (curl), soft tissue
skinning produces non-zero displacement on meshes registered with
digit-only allowed_chains.
"""

import numpy as np
import pytest

from faceforge.core.scene_graph import Scene, SceneNode
from faceforge.core.mesh import MeshInstance, BufferGeometry
from faceforge.core.math_utils import quat_from_euler, vec3
from faceforge.core.state import BodyState
from faceforge.body.soft_tissue import SoftTissueSkinning


def _make_mesh(name: str, positions: np.ndarray) -> MeshInstance:
    """Create a MeshInstance with given positions."""
    flat = positions.astype(np.float32).ravel()
    normals = np.zeros_like(flat)
    geom = BufferGeometry(
        positions=flat,
        normals=normals,
        vertex_count=len(positions),
    )
    mesh = MeshInstance(name=name, geometry=geom)
    mesh.store_rest_pose()
    return mesh


def _build_digit_scene():
    """Build a minimal scene with wrist + one finger (mc→prox→mid→dist).

    Returns: scene, wrist_pivot, {seg: pivot}, digit_chain
    """
    scene = Scene()
    body_root = SceneNode(name="bodyRoot")
    scene.add(body_root)

    # Arm/wrist hierarchy: shoulder → elbow → wrist
    shoulder = SceneNode(name="shoulder_R")
    shoulder.set_position(10.0, 0.0, -20.0)
    body_root.add(shoulder)

    elbow = SceneNode(name="elbow_R")
    elbow.set_position(0.0, 0.0, -15.0)  # relative to shoulder
    shoulder.add(elbow)

    wrist = SceneNode(name="wrist_R")
    wrist.set_position(0.0, 0.0, -12.0)  # relative to elbow
    elbow.add(wrist)

    # Finger 2 chain: mc → prox → mid → dist
    mc = SceneNode(name="finger_R_2_mc")
    mc.set_position(0.0, 0.0, -3.0)  # relative to wrist
    wrist.add(mc)

    prox = SceneNode(name="finger_R_2_prox")
    prox.set_position(0.0, 0.0, -2.5)  # relative to mc
    mc.add(prox)

    mid = SceneNode(name="finger_R_2_mid")
    mid.set_position(0.0, 0.0, -2.0)  # relative to prox
    prox.add(mid)

    dist = SceneNode(name="finger_R_2_dist")
    dist.set_position(0.0, 0.0, -1.5)  # relative to mid
    mid.add(dist)

    scene.update()

    # Chain tuples
    digit_chain = [
        ("finger_R_2_mc", mc),
        ("finger_R_2_prox", prox),
        ("finger_R_2_mid", mid),
        ("finger_R_2_dist", dist),
    ]

    arm_chain = [("shoulder_R", shoulder), ("elbow_R", elbow)]
    forearm_chain = [("elbow_R", elbow), ("wrist_R", wrist)]

    pivots = {"mc": mc, "prox": prox, "mid": mid, "dist": dist,
              "wrist": wrist, "elbow": elbow, "shoulder": shoulder}

    return scene, pivots, arm_chain, forearm_chain, digit_chain


class TestHandCurlSkinning:
    """Test digit chain curl produces displacement in soft tissue."""

    def test_digit_only_chains_curl_moves_vertices(self):
        """Core test: vertices registered with digit-only chains
        must move when finger pivots are rotated (curl)."""
        scene, pivots, arm_chain, forearm_chain, digit_chain = _build_digit_scene()

        skinning = SoftTissueSkinning()
        joint_chains = [arm_chain, forearm_chain, digit_chain]
        # chain IDs: 0=arm, 1=forearm, 2=digit
        skinning.build_skin_joints(joint_chains)

        # Create test mesh: vertices along the finger, close to bone segments
        mc_world = pivots["mc"].world_matrix[:3, 3]
        prox_world = pivots["prox"].world_matrix[:3, 3]
        mid_world = pivots["mid"].world_matrix[:3, 3]

        # Place test vertices near each finger segment, slightly offset
        positions = np.array([
            mc_world + [0.5, 0.0, 0.0],
            mc_world + [-0.5, 0.0, 0.0],
            (mc_world + prox_world) / 2 + [0.3, 0.0, 0.0],
            prox_world + [0.4, 0.0, 0.0],
            (prox_world + mid_world) / 2 + [0.2, 0.0, 0.0],
            mid_world + [0.3, 0.0, 0.0],
        ], dtype=np.float64)

        mesh = _make_mesh("test_hand_muscle", positions)

        # Add mesh node to scene so it has a scene graph parent
        mesh_node = SceneNode(name="test_hand_muscle_node")
        mesh_node.mesh = mesh
        scene.children[0].add(mesh_node)  # under bodyRoot

        # Register with digit-only chain
        digit_chain_id = 2  # third chain
        skinning.register_skin_mesh(
            mesh, is_muscle=True, allowed_chains={digit_chain_id},
        )

        assert len(skinning.bindings) == 1, "Mesh should be registered"
        binding = skinning.bindings[0]

        # Verify vertices are bound to digit chain joints
        digit_joint_start = sum(len(c) for c in [arm_chain, forearm_chain])
        digit_joint_end = digit_joint_start + len(digit_chain)
        for vi in range(len(positions)):
            ji = binding.joint_indices[vi]
            assert digit_joint_start <= ji < digit_joint_end, (
                f"Vertex {vi} bound to joint {ji} (name={skinning.joints[ji].name}), "
                f"expected digit chain joints [{digit_joint_start}, {digit_joint_end})"
            )

        # Save rest positions
        rest_pos = mesh.rest_positions.reshape(-1, 3).copy()

        # Apply curl: rotate mc pivot by 36° (40% of 90° max curl)
        curl_angle = np.radians(36.0)
        mc_q = quat_from_euler(-curl_angle, 0.0, 0.0, "XYZ")
        pivots["mc"].set_quaternion(mc_q)

        # Also rotate prox (31.5° = 35% of 90°)
        prox_angle = np.radians(31.5)
        prox_q = quat_from_euler(-prox_angle, 0.0, 0.0, "XYZ")
        pivots["prox"].set_quaternion(prox_q)

        # Update scene graph matrices (matches app step 9.6)
        scene.update()

        # Run skinning update
        body_state = BodyState()
        body_state.finger_curl_r = 0.8  # changed from default 0.0
        skinning.update(body_state)

        # Check that positions have changed
        new_pos = mesh.geometry.positions.reshape(-1, 3)
        displacement = np.linalg.norm(new_pos - rest_pos, axis=1)
        mean_disp = displacement.mean()
        max_disp = displacement.max()

        assert max_disp > 0.1, (
            f"Max displacement {max_disp:.4f} too small — curl should move vertices. "
            f"Mean: {mean_disp:.4f}"
        )

    def test_wrist_flex_also_moves_digit_bound_vertices(self):
        """Vertices on digit chains should also move when wrist rotates."""
        scene, pivots, arm_chain, forearm_chain, digit_chain = _build_digit_scene()

        skinning = SoftTissueSkinning()
        joint_chains = [arm_chain, forearm_chain, digit_chain]
        skinning.build_skin_joints(joint_chains)

        mc_world = pivots["mc"].world_matrix[:3, 3]
        positions = np.array([
            mc_world + [0.5, 0.0, 0.0],
            mc_world + [-0.5, 0.0, 0.0],
        ], dtype=np.float64)

        mesh = _make_mesh("test_wrist_muscle", positions)
        mesh_node = SceneNode(name="test_wrist_node")
        mesh_node.mesh = mesh
        scene.children[0].add(mesh_node)

        digit_chain_id = 2
        skinning.register_skin_mesh(
            mesh, is_muscle=True, allowed_chains={digit_chain_id},
        )

        rest_pos = mesh.rest_positions.reshape(-1, 3).copy()

        # Rotate wrist (flex) — this should propagate to finger pivots
        wrist_q = quat_from_euler(-np.radians(45.0), 0.0, 0.0, "XYZ")
        pivots["wrist"].set_quaternion(wrist_q)
        scene.update()

        body_state = BodyState()
        body_state.wrist_r_flex = 0.7
        skinning.update(body_state)

        new_pos = mesh.geometry.positions.reshape(-1, 3)
        displacement = np.linalg.norm(new_pos - rest_pos, axis=1)
        max_disp = displacement.max()

        assert max_disp > 0.5, (
            f"Max displacement {max_disp:.4f} — wrist flex should move digit-chain vertices"
        )

    def test_all_chains_curl_does_not_move(self):
        """When allowed_chains=None (all chains), wrist segment dominates
        and curl rotation is lost — this is the BUG we're preventing."""
        scene, pivots, arm_chain, forearm_chain, digit_chain = _build_digit_scene()

        skinning = SoftTissueSkinning()
        joint_chains = [arm_chain, forearm_chain, digit_chain]
        skinning.build_skin_joints(joint_chains)

        mc_world = pivots["mc"].world_matrix[:3, 3]
        positions = np.array([
            mc_world + [0.5, 0.0, 0.0],
            mc_world + [-0.5, 0.0, 0.0],
        ], dtype=np.float64)

        mesh = _make_mesh("test_all_chains", positions)
        mesh_node = SceneNode(name="test_all_node")
        mesh_node.mesh = mesh
        scene.children[0].add(mesh_node)

        # Register with ALL chains (no restriction)
        skinning.register_skin_mesh(
            mesh, is_muscle=True, allowed_chains=None,
        )

        binding = skinning.bindings[0]
        # Check which joints the vertices bound to
        for vi in range(len(positions)):
            ji = binding.joint_indices[vi]
            joint_name = skinning.joints[ji].name
            print(f"  All-chains: vertex {vi} → {joint_name} (chain {skinning.joints[ji].chain_id})")

        rest_pos = mesh.rest_positions.reshape(-1, 3).copy()

        # Apply curl only (no wrist)
        curl_angle = np.radians(36.0)
        pivots["mc"].set_quaternion(quat_from_euler(-curl_angle, 0.0, 0.0, "XYZ"))
        pivots["prox"].set_quaternion(quat_from_euler(-np.radians(31.5), 0.0, 0.0, "XYZ"))
        scene.update()

        body_state = BodyState()
        body_state.finger_curl_r = 0.8
        skinning.update(body_state)

        new_pos = mesh.geometry.positions.reshape(-1, 3)
        displacement = np.linalg.norm(new_pos - rest_pos, axis=1)
        mean_disp = displacement.mean()
        # This test documents the problem: with all chains, curl displacement may be minimal
        print(f"  All-chains curl displacement: mean={mean_disp:.4f}")
