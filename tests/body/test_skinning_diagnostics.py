"""Tests for soft tissue skinning diagnostics and cross-region binding prevention.

Uses realistic joint coordinates matching the actual BodyParts3D model
(from the diagnostic report) to verify that:
1. Hand/foot chains can't grab distant vertices (proportional Z margin)
2. Arm chains don't bind to leg-region vertices (spatial limit guard)
3. Cross-binding anomalies are detected by the diagnostic tool
4. Arm raise doesn't displace leg/torso skin
"""

import numpy as np
import pytest

from faceforge.core.scene_graph import SceneNode, Scene
from faceforge.core.mesh import BufferGeometry, MeshInstance
from faceforge.core.material import Material
from faceforge.core.state import BodyState
from faceforge.body.soft_tissue import SoftTissueSkinning


def _make_skin_mesh(name: str, positions_3d: np.ndarray) -> MeshInstance:
    """Create a MeshInstance with given 3D positions and store rest pose."""
    flat = positions_3d.ravel().astype(np.float32)
    normals = np.tile([0, 0, 1], len(positions_3d)).astype(np.float32)
    geom = BufferGeometry(
        positions=flat,
        normals=normals,
        vertex_count=len(positions_3d),
    )
    mesh = MeshInstance(name=name, geometry=geom, material=Material())
    mesh.store_rest_pose()
    return mesh


def _make_joint_node(name: str, x: float, y: float, z: float) -> SceneNode:
    """Create a SceneNode positioned at (x, y, z)."""
    node = SceneNode(name=name)
    node.set_position(x, y, z)
    return node


# ── Realistic skeleton matching actual BP3D model diagnostic output ──
# Z-axis ranges from the diagnostic:
#   spine:  Z=[-76.3, -10.9]
#   arm_R:  Z=[-83.3, -17.4]
#   arm_L:  Z=[-83.3, -16.1]
#   leg_R:  Z=[-187.8, -83.1]
#   leg_L:  Z=[-187.9, -84.1]
#   hand_R: Z=[-104.7, -88.8] (varies by digit)
#   hand_L: Z=[-104.7, -88.7]
#   foot_R: Z=[-196.8, -193.1]
#   foot_L: Z=[-196.8, -193.1]
#   ribs:   Z=[-51.2, -9.8]


def _build_realistic_skeleton(scene: Scene) -> tuple[list[list[tuple[str, SceneNode]]], dict[str, int]]:
    """Build a skeleton using real BP3D coordinates from diagnostic output.

    Returns (joint_chains, chain_id_map) where chain_id_map maps names to IDs.
    """
    root = SceneNode(name="bodyRoot")
    scene.add(root)

    def j(name, x, y, z):
        n = _make_joint_node(name, x, y, z)
        root.add(n)
        return n

    chains: list[list[tuple[str, SceneNode]]] = []
    chain_ids: dict[str, int] = {}

    # Chain 0: Spine
    spine = [
        ("thoracic_1", j("thoracic_1", 0, 5, -10.9)),
        ("thoracic_6", j("thoracic_6", 0, 3, -35)),
        ("thoracic_12", j("thoracic_12", 0, 2, -55)),
        ("lumbar_3", j("lumbar_3", 0, 1, -66)),
        ("lumbar_5", j("lumbar_5", 0, 0, -76.3)),
    ]
    chain_ids["spine"] = len(chains)
    chains.append(spine)

    # Chain 1: Arm R (shoulder→elbow→wrist)
    arm_r = [
        ("shoulder_R", j("shoulder_R", -28, 0, -17.4)),
        ("elbow_R", j("elbow_R", -28, -2, -50.4)),
        ("wrist_R", j("wrist_R", -28, -1, -83.3)),
    ]
    chain_ids["arm_R"] = len(chains)
    chains.append(arm_r)

    # Chain 2: Arm L
    arm_l = [
        ("shoulder_L", j("shoulder_L", 28, 0, -16.1)),
        ("elbow_L", j("elbow_L", 28, -2, -49.7)),
        ("wrist_L", j("wrist_L", 28, -1, -83.3)),
    ]
    chain_ids["arm_L"] = len(chains)
    chains.append(arm_l)

    # Chain 3: Leg R (hip→knee→ankle)
    leg_r = [
        ("hip_R", j("hip_R", -12, 0, -83.1)),
        ("knee_R", j("knee_R", -12, -3, -130)),
        ("ankle_R", j("ankle_R", -12, -2, -187.8)),
    ]
    chain_ids["leg_R"] = len(chains)
    chains.append(leg_r)

    # Chain 4: Leg L
    leg_l = [
        ("hip_L", j("hip_L", 12, 0, -84.1)),
        ("knee_L", j("knee_L", 12, -3, -130)),
        ("ankle_L", j("ankle_L", 12, -2, -187.9)),
    ]
    chain_ids["leg_L"] = len(chains)
    chains.append(leg_l)

    # Chain 5: Hand R digit 3 (representative finger)
    hand_r3 = [
        ("finger_R_3_mc", j("finger_R_3_mc", -28, -1, -92.1)),
        ("finger_R_3_prox", j("finger_R_3_prox", -28, -1, -97)),
        ("finger_R_3_mid", j("finger_R_3_mid", -28, -1, -101)),
        ("finger_R_3_dist", j("finger_R_3_dist", -28, -1, -104.7)),
    ]
    chain_ids["hand_R_3"] = len(chains)
    chains.append(hand_r3)

    # Chain 6: Hand R digit 5 (the problematic one from the diagnostic)
    hand_r5 = [
        ("finger_R_5_mc", j("finger_R_5_mc", -30, -1, -91.6)),
        ("finger_R_5_prox", j("finger_R_5_prox", -30, -1, -96)),
        ("finger_R_5_dist", j("finger_R_5_dist", -30, -1, -101.7)),
    ]
    chain_ids["hand_R_5"] = len(chains)
    chains.append(hand_r5)

    # Chain 7: Ribs
    ribs = [
        ("rib_0", j("rib_0", 0, 8, -9.8)),
        ("rib_5", j("rib_5", 0, 10, -30)),
        ("rib_11", j("rib_11", 0, 8, -51.2)),
    ]
    chain_ids["ribs"] = len(chains)
    chains.append(ribs)

    scene.update()
    return chains, chain_ids


def _make_realistic_skin() -> MeshInstance:
    """Create skin vertices at specific locations that triggered anomalies.

    Includes:
    - The actual worst-case vertex from the diagnostic: (23.2, -8.7, -116.3)
    - Torso vertices near the hip
    - Thigh/leg vertices that should NOT bind to arm/hand chains
    - Arm vertices that SHOULD bind to arm chains
    """
    verts = []

    # ── Torso skin (X in [-20, 20], Z in [-10, -80]) ──
    for z in np.arange(-10, -82, -5):
        for x in np.arange(-20, 22, 5):
            verts.append([x, 8, z])

    # ── Right arm skin (X in [-35, -22], Z in [-17, -83]) ──
    for z in np.arange(-17, -85, -5):
        for x in [-35, -30, -28, -25, -22]:
            verts.append([x, 3, z])

    # ── Left arm skin (X in [22, 35], Z in [-16, -83]) ──
    for z in np.arange(-16, -85, -5):
        for x in [22, 25, 28, 30, 35]:
            verts.append([x, 3, z])

    # ── Right leg skin (X in [-18, -6], Z in [-83, -190]) ──
    for z in np.arange(-83, -192, -5):
        for x in [-18, -15, -12, -9, -6]:
            verts.append([x, 5, z])

    # ── Left leg skin (X in [6, 18], Z in [-84, -190]) ──
    for z in np.arange(-84, -192, -5):
        for x in [6, 9, 12, 15, 18]:
            verts.append([x, 5, z])

    # ── Specific anomalous vertices from the diagnostic report ──
    # The worst vertex: X=23.2, Y=-8.7, Z=-116.3 — left side, deep in leg zone
    verts.append([23.2, -8.7, -116.3])
    # Mid-thigh right side
    verts.append([-15, 5, -100])
    verts.append([-15, 5, -110])
    verts.append([-15, 5, -120])
    # Near hip on right side (borderline zone)
    verts.append([-20, 5, -85])
    verts.append([-25, 3, -85])

    return _make_skin_mesh("Skin", np.array(verts, dtype=np.float64))


class TestRealisticBindingFiltering:
    """Test filtering with realistic BP3D coordinates from the diagnostic output."""

    def setup_method(self):
        self.scene = Scene()
        self.chains, self.chain_ids = _build_realistic_skeleton(self.scene)
        self.skinning = SoftTissueSkinning()
        self.skinning.build_skin_joints(self.chains)

    def _get_vertex_chain(self, binding, vert_idx: int) -> int:
        """Return the chain ID assigned to a specific vertex."""
        ji = binding.joint_indices[vert_idx]
        return self.skinning.joints[ji].chain_id

    def _get_chain_name(self, chain_id: int) -> str:
        for name, cid in self.chain_ids.items():
            if cid == chain_id:
                return name
        return f"chain_{chain_id}"

    def test_hand_chain_proportional_margin(self):
        """Hand chains (Z extent ~12) should get margin ~3, not 15."""
        skin = _make_realistic_skin()
        all_cids = set(self.chain_ids.values())
        self.skinning.register_skin_mesh(
            skin, is_muscle=False, allowed_chains=all_cids,
            chain_z_margin=15.0, spatial_limit=10.0,
        )

        binding = self.skinning.bindings[0]
        rest_pos = skin.rest_positions.reshape(-1, 3)
        hand_chain_ids = {self.chain_ids["hand_R_3"], self.chain_ids["hand_R_5"]}

        # Vertices at Z < -110 should NEVER bind to hand chains
        # (hand Z range ~[-105, -89] + proportional margin ~3 → [-108, -86])
        deep_mask = rest_pos[:, 2] < -110
        for vi in np.where(deep_mask)[0]:
            cid = self._get_vertex_chain(binding, vi)
            assert cid not in hand_chain_ids, (
                f"Vertex {vi} at Z={rest_pos[vi, 2]:.1f} bound to "
                f"{self._get_chain_name(cid)} (hand chain)"
            )

    def test_worst_case_vertex_not_on_hand(self):
        """The diagnostic's worst vertex (23.2, -8.7, -116.3) must NOT bind to hand."""
        skin = _make_realistic_skin()
        all_cids = set(self.chain_ids.values())
        self.skinning.register_skin_mesh(
            skin, is_muscle=False, allowed_chains=all_cids,
            chain_z_margin=15.0, spatial_limit=10.0,
        )

        binding = self.skinning.bindings[0]
        rest_pos = skin.rest_positions.reshape(-1, 3)

        # Find the worst-case vertex
        target = np.array([23.2, -8.7, -116.3])
        dists_to_target = np.linalg.norm(rest_pos - target, axis=1)
        vi = int(np.argmin(dists_to_target))

        cid = self._get_vertex_chain(binding, vi)
        chain_name = self._get_chain_name(cid)

        hand_chain_ids = {self.chain_ids["hand_R_3"], self.chain_ids["hand_R_5"]}
        arm_chain_ids = {self.chain_ids["arm_R"], self.chain_ids["arm_L"]}
        bad_chains = hand_chain_ids | arm_chain_ids

        assert cid not in bad_chains, (
            f"Worst-case vertex at {rest_pos[vi]} bound to {chain_name} "
            f"(should be spine or leg)"
        )

    def test_thigh_vertices_not_on_arm(self):
        """Mid-thigh vertices (Z=-100 to -120, |X|<20) should bind to leg or spine."""
        skin = _make_realistic_skin()
        all_cids = set(self.chain_ids.values())
        self.skinning.register_skin_mesh(
            skin, is_muscle=False, allowed_chains=all_cids,
            chain_z_margin=15.0, spatial_limit=10.0,
        )

        binding = self.skinning.bindings[0]
        rest_pos = skin.rest_positions.reshape(-1, 3)

        arm_chain_ids = {self.chain_ids["arm_R"], self.chain_ids["arm_L"]}
        hand_chain_ids = {self.chain_ids["hand_R_3"], self.chain_ids["hand_R_5"]}
        bad_chains = arm_chain_ids | hand_chain_ids

        # Thigh region: Z in [-100, -120], X close to midline
        thigh_mask = (
            (rest_pos[:, 2] < -100) &
            (rest_pos[:, 2] > -120) &
            (np.abs(rest_pos[:, 0]) < 20)
        )
        bad_count = 0
        for vi in np.where(thigh_mask)[0]:
            cid = self._get_vertex_chain(binding, vi)
            if cid in bad_chains:
                bad_count += 1
        assert bad_count == 0, (
            f"{bad_count} thigh vertices (Z=-100 to -120, |X|<20) bound to arm/hand chains"
        )

    def test_hip_border_zone(self):
        """Vertices at the hip border (Z~-83, X~-12 to -20) should prefer leg/spine over arm."""
        skin = _make_realistic_skin()
        all_cids = set(self.chain_ids.values())
        self.skinning.register_skin_mesh(
            skin, is_muscle=False, allowed_chains=all_cids,
            chain_z_margin=15.0, spatial_limit=10.0,
        )

        binding = self.skinning.bindings[0]
        rest_pos = skin.rest_positions.reshape(-1, 3)

        # Hip border: Z in [-80, -90], X in [-20, -6] (right side inner body)
        border_mask = (
            (rest_pos[:, 2] >= -90) &
            (rest_pos[:, 2] <= -80) &
            (rest_pos[:, 0] > -20) &
            (rest_pos[:, 0] < -6)
        )
        if not np.any(border_mask):
            pytest.skip("No hip-border vertices in test mesh")

        hand_chain_ids = {self.chain_ids["hand_R_3"], self.chain_ids["hand_R_5"]}
        # None of these should bind to hand chains (too far laterally and vertically)
        for vi in np.where(border_mask)[0]:
            cid = self._get_vertex_chain(binding, vi)
            assert cid not in hand_chain_ids, (
                f"Hip border vertex {vi} at {rest_pos[vi]} bound to hand chain"
            )

    def test_arm_raise_displacement(self):
        """After arm raise, check that NO leg/torso vertices are displaced."""
        from faceforge.body.diagnostics import SkinningDiagnostic
        from faceforge.core.math_utils import quat_from_euler

        skin = _make_realistic_skin()
        all_cids = set(self.chain_ids.values())
        self.skinning.register_skin_mesh(
            skin, is_muscle=False, allowed_chains=all_cids,
            chain_z_margin=15.0, spatial_limit=10.0,
        )

        # Rotate shoulder_R 90° (arm raise)
        shoulder_r = self.chains[1][0][1]  # arm_R chain, first joint
        shoulder_r.set_quaternion(quat_from_euler(0, 0, np.radians(90)))
        self.scene.update()

        body = BodyState()
        body.shoulder_r_abduct = 1.0
        self.skinning._last_signature = ""
        self.skinning.update(body)

        rest_pos = skin.rest_positions.reshape(-1, 3)
        curr_pos = skin.geometry.positions.reshape(-1, 3)

        # Deep leg region (Z < -100): should have near-zero displacement
        deep_leg = rest_pos[:, 2] < -100
        if np.any(deep_leg):
            leg_disp = np.linalg.norm(curr_pos[deep_leg] - rest_pos[deep_leg], axis=1)
            max_disp = float(leg_disp.max())
            assert max_disp < 3.0, (
                f"Deep leg vertices displaced by {max_disp:.2f} on arm raise (max allowed: 3.0)"
            )

        # Torso near midline (|X| < 10, Z in [-40, -75]): should be stable
        torso_mid = (np.abs(rest_pos[:, 0]) < 10) & (rest_pos[:, 2] > -75) & (rest_pos[:, 2] < -40)
        if np.any(torso_mid):
            torso_disp = np.linalg.norm(curr_pos[torso_mid] - rest_pos[torso_mid], axis=1)
            max_disp = float(torso_disp.max())
            assert max_disp < 3.0, (
                f"Mid-torso vertices displaced by {max_disp:.2f} on arm raise (max allowed: 3.0)"
            )

    def test_diagnostic_report_clean(self):
        """Run full diagnostic with arm raise — anomalies should be arm-only."""
        from faceforge.body.diagnostics import SkinningDiagnostic
        from faceforge.core.math_utils import quat_from_euler

        skin = _make_realistic_skin()
        all_cids = set(self.chain_ids.values())
        self.skinning.register_skin_mesh(
            skin, is_muscle=False, allowed_chains=all_cids,
            chain_z_margin=15.0, spatial_limit=10.0,
        )

        shoulder_r = self.chains[1][0][1]
        shoulder_r.set_quaternion(quat_from_euler(0, 0, np.radians(90)))
        self.scene.update()

        body = BodyState()
        body.shoulder_r_abduct = 1.0
        self.skinning._last_signature = ""
        self.skinning.update(body)

        diag = SkinningDiagnostic(self.skinning)
        anomalies = diag.check_displacements(max_displacement=5.0, relative=True)

        # If there are anomalies, they should NOT be in leg/spine chains
        leg_chain_ids = {self.chain_ids["leg_R"], self.chain_ids["leg_L"]}
        for a in anomalies:
            for chain_name, count in a.chain_breakdown.items():
                # Map chain name back to ID
                for cn, cid in self.chain_ids.items():
                    if cn == chain_name or chain_name.startswith(cn):
                        assert cid not in leg_chain_ids, (
                            f"Leg chain {chain_name} has {count} anomalous vertices"
                        )


class TestChainZMarginFilter:
    """Test Z-axis AABB filter with simplified skeleton."""

    def setup_method(self):
        self.scene = Scene()
        root = SceneNode(name="bodyRoot")
        self.scene.add(root)

        def j(name, x, y, z):
            n = _make_joint_node(name, x, y, z)
            root.add(n)
            return n

        self.chains = [
            [("thoracic_1", j("thoracic_1", 0, 0, 0)),
             ("lumbar_5", j("lumbar_5", 0, 0, -80))],
            [("shoulder_R", j("shoulder_R", -28, 0, -5)),
             ("wrist_R", j("wrist_R", -28, 0, -65))],
            [("hip_R", j("hip_R", -12, 0, -85)),
             ("ankle_R", j("ankle_R", -12, 0, -175))],
        ]
        self.scene.update()
        self.skinning = SoftTissueSkinning()
        self.skinning.build_skin_joints(self.chains)

    def test_z_margin_prevents_deep_leg_from_arm(self):
        """Deep leg vertices (Z<-100) should NOT bind to arm chain."""
        verts = np.array([
            [-15, 5, -110],  # mid-thigh
            [-15, 5, -130],  # knee level
            [-15, 5, -160],  # shin
        ], dtype=np.float64)
        skin = _make_skin_mesh("Skin", verts)

        self.skinning.register_skin_mesh(
            skin, is_muscle=False, allowed_chains={0, 1, 2},
            chain_z_margin=15.0,
        )

        binding = self.skinning.bindings[0]
        arm_chain = 1
        for vi in range(3):
            cid = self.skinning.joints[binding.joint_indices[vi]].chain_id
            assert cid != arm_chain, (
                f"Vertex at Z={verts[vi, 2]} bound to arm chain"
            )

    def test_spine_always_available(self):
        """Spine chain is never filtered, available everywhere."""
        verts = np.array([
            [0, 5, -200],  # way below everything
        ], dtype=np.float64)
        skin = _make_skin_mesh("Skin", verts)

        self.skinning.register_skin_mesh(
            skin, is_muscle=False, allowed_chains={0, 1, 2},
            chain_z_margin=15.0,
        )

        binding = self.skinning.bindings[0]
        cid = self.skinning.joints[binding.joint_indices[0]].chain_id
        assert cid == 0, f"Vertex below all chains should bind to spine, got chain {cid}"

    def test_proportional_margin_for_small_chains(self):
        """Small chains (extent 10) should get small margin (~2.5), not 15."""
        # Add a tiny chain (like a finger) that spans Z=-90 to Z=-100
        root = self.scene.children[0]
        fn1 = _make_joint_node("finger_mc", -28, 0, -90)
        fn2 = _make_joint_node("finger_dist", -28, 0, -100)
        root.add(fn1)
        root.add(fn2)
        self.scene.update()

        chains = self.chains + [[("finger_mc", fn1), ("finger_dist", fn2)]]
        self.skinning = SoftTissueSkinning()
        self.skinning.build_skin_joints(chains)

        # Vertex at Z=-115: outside small chain range [-100, -90] + margin
        # proportional margin = min(15, max(2, 10*0.25)) = min(15, 2.5) = 2.5
        # So range is [-102.5, -87.5]. Z=-115 is outside.
        verts = np.array([[-28, 0, -115]], dtype=np.float64)
        skin = _make_skin_mesh("Skin", verts)

        self.skinning.register_skin_mesh(
            skin, is_muscle=False, allowed_chains={0, 1, 2, 3},
            chain_z_margin=15.0,
        )

        binding = self.skinning.bindings[0]
        cid = self.skinning.joints[binding.joint_indices[0]].chain_id
        # Should NOT be the finger chain (3)
        assert cid != 3, (
            f"Vertex at Z=-115 bound to finger chain (should be filtered by proportional margin)"
        )


class TestMultipleMovements:
    """Test various joint movements to verify no cross-region displacement."""

    def setup_method(self):
        self.scene = Scene()
        self.chains, self.chain_ids = _build_realistic_skeleton(self.scene)
        self.skinning = SoftTissueSkinning()
        self.skinning.build_skin_joints(self.chains)

    def _register_skin_and_move(self, rotations: dict[int, tuple[float, float, float]]):
        """Register skin, apply rotations, and return (rest_pos, curr_pos)."""
        from faceforge.core.math_utils import quat_from_euler

        skin = _make_realistic_skin()
        all_cids = set(self.chain_ids.values())
        self.skinning.register_skin_mesh(
            skin, is_muscle=False, allowed_chains=all_cids,
            chain_z_margin=15.0, spatial_limit=10.0,
        )

        for chain_idx, (rx, ry, rz) in rotations.items():
            node = self.chains[chain_idx][0][1]
            node.set_quaternion(quat_from_euler(rx, ry, rz))
        self.scene.update()

        body = BodyState()
        body.shoulder_r_abduct = 1.0
        self.skinning._last_signature = ""
        self.skinning.update(body)

        rest_pos = skin.rest_positions.reshape(-1, 3)
        curr_pos = skin.geometry.positions.reshape(-1, 3)
        return rest_pos, curr_pos

    def _check_region_stable(self, rest_pos, curr_pos, mask, region_name, max_disp=3.0):
        """Assert that masked region hasn't moved beyond threshold."""
        if not np.any(mask):
            return
        disp = np.linalg.norm(curr_pos[mask] - rest_pos[mask], axis=1)
        md = float(disp.max())
        assert md < max_disp, (
            f"{region_name} displaced by {md:.2f} (max {max_disp})"
        )

    def test_right_arm_raise_legs_stable(self):
        """Right arm 90° raise: legs should not move."""
        rest, curr = self._register_skin_and_move({
            1: (0, 0, np.radians(90)),  # arm_R raise
        })
        # Both legs stable
        legs = rest[:, 2] < -100
        self._check_region_stable(rest, curr, legs, "legs on R arm raise")

    def test_left_arm_raise_legs_stable(self):
        """Left arm 90° raise: legs should not move."""
        rest, curr = self._register_skin_and_move({
            2: (0, 0, np.radians(-90)),  # arm_L raise
        })
        legs = rest[:, 2] < -100
        self._check_region_stable(rest, curr, legs, "legs on L arm raise")

    def test_both_arms_raise_torso_stable(self):
        """Both arms raised: mid-torso should not move."""
        rest, curr = self._register_skin_and_move({
            1: (0, 0, np.radians(90)),   # arm_R
            2: (0, 0, np.radians(-90)),  # arm_L
        })
        torso = (np.abs(rest[:, 0]) < 10) & (rest[:, 2] > -75) & (rest[:, 2] < -30)
        self._check_region_stable(rest, curr, torso, "mid-torso on both arms raise")

    def test_leg_raise_arms_stable(self):
        """Right leg raised: arms should not move."""
        rest, curr = self._register_skin_and_move({
            3: (np.radians(90), 0, 0),  # leg_R hip flex
        })
        arms = (np.abs(rest[:, 0]) > 22) & (rest[:, 2] > -85)
        self._check_region_stable(rest, curr, arms, "arms on R leg raise")

    def test_arm_raise_opposite_arm_stable(self):
        """Right arm raise: left arm should not move."""
        rest, curr = self._register_skin_and_move({
            1: (0, 0, np.radians(90)),  # arm_R only
        })
        left_arm = (rest[:, 0] > 22) & (rest[:, 2] > -85)
        self._check_region_stable(rest, curr, left_arm, "left arm on R arm raise")

    def test_spine_flex_affects_torso_not_extremities(self):
        """Spine flex should affect torso but not feet."""
        rest, curr = self._register_skin_and_move({
            0: (np.radians(30), 0, 0),  # spine flex (chain 0, first joint)
        })
        feet = rest[:, 2] < -180
        self._check_region_stable(rest, curr, feet, "feet on spine flex", max_disp=5.0)


class TestLateralFiltering:
    """Test X-axis lateral filtering prevents cross-body binding."""

    def setup_method(self):
        self.scene = Scene()
        self.chains, self.chain_ids = _build_realistic_skeleton(self.scene)
        self.skinning = SoftTissueSkinning()
        self.skinning.build_skin_joints(self.chains)

    def _get_vertex_chain(self, binding, vert_idx: int) -> int:
        ji = binding.joint_indices[vert_idx]
        return self.skinning.joints[ji].chain_id

    def _get_chain_name(self, chain_id: int) -> str:
        for name, cid in self.chain_ids.items():
            if cid == chain_id:
                return name
        return f"chain_{chain_id}"

    def test_left_side_vertex_not_on_right_hand(self):
        """Vertex at X=+23 (left body) must NOT bind to right hand (X≈-28 to -30).

        This was the exact worst-case vertex from the diagnostic report:
        (23.2, -8.7, -116.3) bound to finger_R_5_dist.
        """
        verts = np.array([
            [23.2, -8.7, -116.3],  # the worst-case vertex
            [20, 5, -100],          # left-side thigh
            [18, 5, -110],          # left-side mid-thigh
        ], dtype=np.float64)
        skin = _make_skin_mesh("Skin", verts)
        all_cids = set(self.chain_ids.values())
        self.skinning.register_skin_mesh(
            skin, is_muscle=False, allowed_chains=all_cids,
            chain_z_margin=15.0, spatial_limit=10.0,
        )
        binding = self.skinning.bindings[0]

        right_chains = {
            self.chain_ids["arm_R"], self.chain_ids["hand_R_3"],
            self.chain_ids["hand_R_5"],
        }
        for vi in range(len(verts)):
            cid = self._get_vertex_chain(binding, vi)
            assert cid not in right_chains, (
                f"Left-side vertex {vi} at X={verts[vi, 0]:.1f} bound to "
                f"right-side chain {self._get_chain_name(cid)}"
            )

    def test_right_side_vertex_not_on_left_arm(self):
        """Vertex at X=-20 (right body) must NOT bind to left arm (X≈+28)."""
        verts = np.array([
            [-20, 5, -100],  # right-side thigh
            [-18, 5, -110],  # right-side mid-thigh
            [-22, 3, -95],   # right-side near hip
        ], dtype=np.float64)
        skin = _make_skin_mesh("Skin", verts)
        all_cids = set(self.chain_ids.values())
        self.skinning.register_skin_mesh(
            skin, is_muscle=False, allowed_chains=all_cids,
            chain_z_margin=15.0, spatial_limit=10.0,
        )
        binding = self.skinning.bindings[0]

        left_chains = {self.chain_ids["arm_L"]}
        for vi in range(len(verts)):
            cid = self._get_vertex_chain(binding, vi)
            assert cid not in left_chains, (
                f"Right-side vertex {vi} at X={verts[vi, 0]:.1f} bound to "
                f"left-side chain {self._get_chain_name(cid)}"
            )

    def test_midline_vertices_not_laterally_filtered(self):
        """Spine/rib vertices near midline (|X|<5) should not be X-filtered."""
        verts = np.array([
            [0, 8, -40],    # midline torso
            [3, 8, -60],    # slightly off-center
            [-4, 8, -30],   # slightly off-center other side
        ], dtype=np.float64)
        skin = _make_skin_mesh("Skin", verts)
        all_cids = set(self.chain_ids.values())
        self.skinning.register_skin_mesh(
            skin, is_muscle=False, allowed_chains=all_cids,
            chain_z_margin=15.0, spatial_limit=10.0,
        )
        binding = self.skinning.bindings[0]

        # Should bind to spine or ribs (midline chains), not arms
        midline_chains = {self.chain_ids["spine"], self.chain_ids["ribs"]}
        for vi in range(len(verts)):
            cid = self._get_vertex_chain(binding, vi)
            assert cid in midline_chains, (
                f"Midline vertex {vi} at X={verts[vi, 0]:.1f} bound to "
                f"{self._get_chain_name(cid)} instead of spine/ribs"
            )

    def test_arm_skin_stays_on_own_side(self):
        """Arm skin on the right side should bind to right arm chain."""
        verts = np.array([
            [-28, 3, -30],   # right arm mid-section
            [-30, 3, -50],   # right arm lower
            [-25, 3, -20],   # right arm upper
        ], dtype=np.float64)
        skin = _make_skin_mesh("Skin", verts)
        all_cids = set(self.chain_ids.values())
        self.skinning.register_skin_mesh(
            skin, is_muscle=False, allowed_chains=all_cids,
            chain_z_margin=15.0, spatial_limit=10.0,
        )
        binding = self.skinning.bindings[0]

        arm_r = self.chain_ids["arm_R"]
        for vi in range(len(verts)):
            cid = self._get_vertex_chain(binding, vi)
            assert cid == arm_r, (
                f"Right arm skin vertex {vi} at X={verts[vi, 0]:.1f} bound to "
                f"{self._get_chain_name(cid)} instead of arm_R"
            )

    def test_cross_body_arm_raise_no_displacement(self):
        """Right arm raise should not displace left-side thigh vertices."""
        from faceforge.core.math_utils import quat_from_euler

        # Create vertices on the left side in the thigh region
        verts = np.array([
            [20, 5, -100],
            [15, 5, -110],
            [23, -8, -116],
        ], dtype=np.float64)
        skin = _make_skin_mesh("Skin", verts)
        all_cids = set(self.chain_ids.values())
        self.skinning.register_skin_mesh(
            skin, is_muscle=False, allowed_chains=all_cids,
            chain_z_margin=15.0, spatial_limit=10.0,
        )

        # Raise right arm 90°
        shoulder_r = self.chains[self.chain_ids["arm_R"]][0][1]
        shoulder_r.set_quaternion(quat_from_euler(0, 0, np.radians(90)))
        self.scene.update()

        body = BodyState()
        body.shoulder_r_abduct = 1.0
        self.skinning._last_signature = ""
        self.skinning.update(body)

        rest_pos = skin.rest_positions.reshape(-1, 3)
        curr_pos = skin.geometry.positions.reshape(-1, 3)
        disp = np.linalg.norm(curr_pos - rest_pos, axis=1)
        max_disp = float(disp.max())
        assert max_disp < 1.0, (
            f"Left-side thigh vertices displaced by {max_disp:.2f} on right arm raise"
        )

    def test_same_side_hand_not_grabbing_thigh(self):
        """Hand_L chains (X≈+28) should NOT grab left thigh vertices (X≈+20).

        In rest position, the hands hang beside the thighs.  Hand bone
        segments are ~9 units from outer-thigh skin.  The proportional
        spatial limit (extent * 0.35 ≈ 4.4 for hand chains) should filter
        these out even though they're on the same body side.
        """
        verts = np.array([
            [20, 3, -100],   # outer left thigh — 8-9 units from hand_L bones
            [18, 5, -95],    # outer left thigh
            [22, 2, -98],    # between thigh and hand_L
        ], dtype=np.float64)
        skin = _make_skin_mesh("Skin", verts)
        all_cids = set(self.chain_ids.values())
        self.skinning.register_skin_mesh(
            skin, is_muscle=False, allowed_chains=all_cids,
            chain_z_margin=15.0, spatial_limit=10.0,
        )
        binding = self.skinning.bindings[0]

        hand_l_chains = set()
        for name, cid in self.chain_ids.items():
            if name.startswith("hand_"):
                hand_l_chains.add(cid)
        hand_r_chains = {self.chain_ids["hand_R_3"], self.chain_ids["hand_R_5"]}

        for vi in range(len(verts)):
            cid = self._get_vertex_chain(binding, vi)
            assert cid not in hand_l_chains and cid not in hand_r_chains, (
                f"Thigh vertex {vi} at ({verts[vi, 0]:.0f}, {verts[vi, 1]:.0f}, "
                f"{verts[vi, 2]:.0f}) bound to {self._get_chain_name(cid)} "
                f"(should be leg_L or spine)"
            )

    def test_hand_skin_still_binds_to_hand(self):
        """Hand skin close to hand bones (3-4 units) should still bind correctly."""
        verts = np.array([
            [27, 0, -96],   # near finger_L_3_prox (28, -1, -97)
            [29, 0, -100],  # near finger_L_3_mid
            [-29, 0, -96],  # near finger_R_3_prox (-28, -1, -97)
        ], dtype=np.float64)
        skin = _make_skin_mesh("Skin", verts)
        all_cids = set(self.chain_ids.values())
        self.skinning.register_skin_mesh(
            skin, is_muscle=False, allowed_chains=all_cids,
            chain_z_margin=15.0, spatial_limit=10.0,
        )
        binding = self.skinning.bindings[0]

        # First two verts should bind to a hand_L chain (not hand_R_3 since
        # we only have hand_R_3 and hand_R_5 in the test skeleton, but
        # there's no hand_L_3 chain — they might bind to arm_L which is fine)
        for vi in [0, 1]:
            cid = self._get_vertex_chain(binding, vi)
            chain_name = self._get_chain_name(cid)
            # Must NOT be a right-side chain (cross-body filter)
            assert "R" not in chain_name or chain_name == "ribs", (
                f"Left hand skin vertex {vi} bound to right chain {chain_name}"
            )
        # Third vert at X=-29 should bind to right-side chain
        cid = self._get_vertex_chain(binding, 2)
        chain_name = self._get_chain_name(cid)
        assert "L" not in chain_name or chain_name == "leg_L", (
            f"Right hand skin vertex 2 bound to left chain {chain_name}"
        )


class TestMuscleBlending:
    """Test that muscles use full-range blending."""

    def setup_method(self):
        self.scene = Scene()
        root = SceneNode(name="bodyRoot")
        self.scene.add(root)

        def j(name, x, y, z):
            n = _make_joint_node(name, x, y, z)
            root.add(n)
            return n

        self.chains = [
            [("shoulder_R", j("shoulder_R", -28, 0, -5)),
             ("elbow_R", j("elbow_R", -28, 0, -35)),
             ("wrist_R", j("wrist_R", -28, 0, -65))],
        ]
        self.scene.update()
        self.skinning = SoftTissueSkinning()
        self.skinning.build_skin_joints(self.chains)

    def test_muscle_full_range_blending(self):
        """Muscle vertices should blend across the full bone segment."""
        verts = np.array([[-28, 3, z] for z in np.linspace(-5, -65, 20)])
        muscle = _make_skin_mesh("biceps_R", verts)

        self.skinning.register_skin_mesh(muscle, is_muscle=True, allowed_chains={0})

        binding = self.skinning.bindings[0]
        unique_weights = np.unique(np.round(binding.weights, 2))
        assert len(unique_weights) > 2, (
            f"Muscle should have varied weights, got {unique_weights}"
        )

    def test_non_muscle_endpoint_only_blending(self):
        """Non-muscle should only blend at segment endpoints."""
        verts = np.array([[-28, 5, z] for z in np.linspace(-5, -65, 20)])
        skin_strip = _make_skin_mesh("skin_arm_R", verts)

        self.skinning.register_skin_mesh(skin_strip, is_muscle=False, allowed_chains={0})

        binding = self.skinning.bindings[0]
        rigid = (binding.weights > 0.999).sum()
        total = len(binding.weights)
        assert rigid / total > 0.5, f"Non-muscle should be mostly rigid, got {rigid}/{total}"
