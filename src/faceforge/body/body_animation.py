"""Body animation: spine flex/bend/rotation, limb articulation, breathing."""

import math

import numpy as np

from faceforge.core.math_utils import (
    Vec3, Quat, deg_to_rad, quat_from_euler, quat_from_axis_angle,
    quat_identity, vec3,
)
from faceforge.core.state import BodyState
from faceforge.core.scene_graph import SceneNode
from faceforge.core.config_loader import load_skeleton_config
from faceforge.body.joint_pivots import JointPivotSetup


class BodyAnimationSystem:
    """Per-frame body animation: spine, limbs, breathing.

    Mirrors applyBodyAnimation() from the JS version.
    """

    def __init__(self, joint_setup: JointPivotSetup):
        self.joints = joint_setup
        self.thoracic_fracs: list[float] = []
        self.lumbar_fracs: list[float] = []
        self.thoracic_pivots: list[dict] = []
        self.lumbar_pivots: list[dict] = []
        self.rib_nodes: list[SceneNode] = []
        self._rib_pivots: list[SceneNode] = []  # lazily built pivot wrappers

    def load_fractions(self) -> None:
        """Load spine distribution fractions from config."""
        try:
            self.thoracic_fracs = load_skeleton_config("thoracic_fractions.json")
        except FileNotFoundError:
            self.thoracic_fracs = [0.12, 0.12, 0.11, 0.10, 0.10, 0.09, 0.09, 0.08, 0.07, 0.06, 0.06]
        try:
            self.lumbar_fracs = load_skeleton_config("lumbar_fractions.json")
        except FileNotFoundError:
            self.lumbar_fracs = [0.20, 0.22, 0.22, 0.20, 0.16, 0.00]

    def set_thoracic_pivots(self, pivots: list[dict]) -> None:
        self.thoracic_pivots = pivots

    def set_lumbar_pivots(self, pivots: list[dict]) -> None:
        self.lumbar_pivots = pivots

    def set_rib_nodes(self, nodes: list[SceneNode]) -> None:
        """Set rib nodes and eagerly create pivot wrappers.

        Must be called before skinning registration so rib pivots
        can be included as a skinning chain.
        """
        self.rib_nodes = nodes
        self._build_rib_pivots()

    def apply(self, state: BodyState, dt: float) -> None:
        """Apply body animation for one frame."""
        # Update breathing phase
        if state.auto_breath_body:
            state.breath_phase_body += dt * state.breath_rate * 2 * math.pi
            if state.breath_phase_body > 2 * math.pi:
                state.breath_phase_body -= 2 * math.pi

        # Apply spine rotation distributed across thoracic + lumbar vertebrae
        self._apply_spine(state)

        # Apply limb articulation
        self._apply_limbs(state)

        # Apply digit articulation
        self._apply_hands(state)
        self._apply_feet(state)

        # Apply breathing to ribs
        self._apply_breathing(state)

    def _apply_spine(self, state: BodyState) -> None:
        """Distribute spine flex/bend/rotation across vertebral pivots.

        JS convention: rotation.set(X=flex, Y=rotation, Z=latBend)
        """
        flex_rad = state.spine_flex * deg_to_rad(45.0)
        bend_rad = state.spine_lat_bend * deg_to_rad(30.0)
        rot_rad = state.spine_rotation * deg_to_rad(30.0)

        # Thoracic distribution
        for i, pivot_info in enumerate(self.thoracic_pivots):
            if i >= len(self.thoracic_fracs):
                break
            frac = self.thoracic_fracs[i]
            pivot_node = pivot_info["group"]

            # JS: rotation.set(flexRad*f + headPitch, rotRad*f + headYaw, bendRad*f + headRoll)
            # headPitch/Yaw/Roll come from constraint solver — skip for now
            x = flex_rad * frac
            y = rot_rad * frac
            z = bend_rad * frac

            q = quat_from_euler(x, y, z, "XYZ")
            pivot_node.set_quaternion(q)

        # Lumbar distribution
        for i, pivot_info in enumerate(self.lumbar_pivots):
            if i >= len(self.lumbar_fracs):
                break
            frac = self.lumbar_fracs[i]
            pivot_node = pivot_info["group"]

            x = flex_rad * frac
            y = rot_rad * frac
            z = bend_rad * frac

            q = quat_from_euler(x, y, z, "XYZ")
            pivot_node.set_quaternion(q)

    def _apply_limbs(self, state: BodyState) -> None:
        """Apply limb joint rotations using JS rotation conventions.

        JS coordinate system: Y=up, Z=toward camera, X=lateral
        - Flexion (forward swing) = negative X rotation
        - Knee flexion (backward) = positive X rotation
        - Shoulder abduction = Z rotation, mirrored for left side
        """
        pivots = self.joints.pivots  # dict[str, SceneNode]

        for side in ("R", "L"):
            s = side.lower()
            mirror = 1.0 if side == "R" else -1.0

            # ── Shoulder: rotation.set(flRad, rotRad, abRad) ──
            shoulder = pivots.get(f"shoulder_{side}")
            if shoulder is not None:
                ab_val = getattr(state, f"shoulder_{s}_abduct", 0.0)
                fl_val = getattr(state, f"shoulder_{s}_flex", 0.0)
                rot_val = getattr(state, f"shoulder_{s}_rotate", 0.0)
                # Abduction: Z-axis, mirrored for left side (±90°)
                ab_rad = ab_val * deg_to_rad(90.0) * mirror
                # Flexion: negative X-axis (±90°)
                fl_rad = -fl_val * deg_to_rad(90.0)
                # Internal/external rotation: Y-axis (±70°)
                rot_rad = rot_val * deg_to_rad(70.0) * mirror
                q = quat_from_euler(fl_rad, rot_rad, ab_rad, "XYZ")
                shoulder.set_quaternion(q)

            # ── Elbow: rotation.set(-flex*145°, 0, 0) ──
            elbow = pivots.get(f"elbow_{side}")
            if elbow is not None:
                el_val = getattr(state, f"elbow_{s}_flex", 0.0)
                el_rad = -el_val * deg_to_rad(145.0)
                q = quat_from_euler(el_rad, 0.0, 0.0, "XYZ")
                elbow.set_quaternion(q)

            # ── Hip: rotation.set(-flex*90°, rotRad, abRad) ──
            hip = pivots.get(f"hip_{side}")
            if hip is not None:
                hf_val = getattr(state, f"hip_{s}_flex", 0.0)
                hab_val = getattr(state, f"hip_{s}_abduct", 0.0)
                hrot_val = getattr(state, f"hip_{s}_rotate", 0.0)
                hf_rad = -hf_val * deg_to_rad(90.0)
                # Abduction: Z-axis, mirrored (±45°)
                hab_rad = hab_val * deg_to_rad(45.0) * mirror
                # Internal/external rotation: Y-axis (±45°)
                hrot_rad = hrot_val * deg_to_rad(45.0) * mirror
                q = quat_from_euler(hf_rad, hrot_rad, hab_rad, "XYZ")
                hip.set_quaternion(q)

            # ── Knee: rotation.set(+flex*145°, 0, 0) ──
            knee = pivots.get(f"knee_{side}")
            if knee is not None:
                kn_val = getattr(state, f"knee_{s}_flex", 0.0)
                kn_rad = kn_val * deg_to_rad(145.0)
                q = quat_from_euler(kn_rad, 0.0, 0.0, "XYZ")
                knee.set_quaternion(q)

            # ── Ankle: rotation.set(-flex*45°, 0, invertRad) ──
            ankle = pivots.get(f"ankle_{side}")
            if ankle is not None:
                an_val = getattr(state, f"ankle_{s}_flex", 0.0)
                an_inv = getattr(state, f"ankle_{s}_invert", 0.0)
                an_rad = -an_val * deg_to_rad(45.0)
                # Inversion/eversion: Z-axis, mirrored (±30°)
                inv_rad = an_inv * deg_to_rad(30.0) * mirror
                q = quat_from_euler(an_rad, 0.0, inv_rad, "XYZ")
                ankle.set_quaternion(q)

            # ── Wrist: rotation.set(flexRad, forearmRot, deviateRad) ──
            wrist = pivots.get(f"wrist_{side}")
            if wrist is not None:
                wr_flex = getattr(state, f"wrist_{s}_flex", 0.0)
                wr_dev = getattr(state, f"wrist_{s}_deviate", 0.0)
                fa_rot = getattr(state, f"forearm_{s}_rotate", 0.0)
                # Wrist flexion: X-axis (±70°)
                wr_fl_rad = -wr_flex * deg_to_rad(70.0)
                # Forearm pronation/supination: Y-axis (±90°)
                fa_rot_rad = fa_rot * deg_to_rad(90.0) * mirror
                # Ulnar/radial deviation: Z-axis (±30°)
                wr_dev_rad = wr_dev * deg_to_rad(30.0) * mirror
                q = quat_from_euler(wr_fl_rad, fa_rot_rad, wr_dev_rad, "XYZ")
                wrist.set_quaternion(q)

    # ── Digit animation ──────────────────────────────────────────────

    # Finger curl distribution: MCP 40%, PIP 35%, DIP 25%
    _FINGER_CURL_DIST = {"mc": 0.40, "prox": 0.35, "mid": 0.25, "dist": 0.0}
    _FINGER_MAX_CURL = 90.0   # degrees at slider=1.0
    _FINGER_MIN_CURL = -20.0  # slight hyperextension at slider=-1.0 (MCP only)

    # Finger spread: fan pattern at MCP (metacarpal) joints
    _FINGER_SPREAD = {2: 12.0, 3: 3.0, 4: -6.0, 5: -12.0}  # degrees per unit

    # Thumb opposition: combined flexion + pronation + adduction at CMC
    _THUMB_OP_FLEX = 50.0
    _THUMB_OP_PRONATE = 40.0
    _THUMB_OP_ADDUCT = 30.0

    # Toe curl distribution: MTP 47%, PIP 33%, DIP 20%
    _TOE_CURL_DIST = {"mt": 0.47, "prox": 0.33, "mid": 0.20, "dist": 0.0}
    _TOE_MAX_CURL = 75.0   # degrees at slider=1.0
    _TOE_MIN_CURL = -30.0  # dorsiflexion at slider=-1.0 (MTP only)

    # Toe spread: fan at MTP joints
    _TOE_SPREAD = {1: 0.0, 2: 5.0, 3: 0.0, 4: -5.0, 5: -8.0}

    def _apply_hands(self, state: BodyState) -> None:
        """Apply finger curl, spread, and thumb opposition."""
        pivots = self.joints.pivots

        for side in ("R", "L"):
            s = side.lower()
            mirror = 1.0 if side == "R" else -1.0

            curl_val = getattr(state, f"finger_curl_{s}", 0.0)
            spread_val = getattr(state, f"finger_spread_{s}", 0.0)
            thumb_op = getattr(state, f"thumb_op_{s}", 0.0)

            # ── Fingers 2-5: curl + spread ──
            for digit in range(2, 6):
                for seg, frac in self._FINGER_CURL_DIST.items():
                    pivot = pivots.get(f"finger_{side}_{digit}_{seg}")
                    if pivot is None:
                        continue

                    # Curl: X-axis flexion
                    if curl_val >= 0:
                        angle = curl_val * self._FINGER_MAX_CURL * frac
                    else:
                        # Hyperextension only at MCP, minimal at PIP/DIP
                        hyper_frac = 1.0 if seg == "mc" else 0.1
                        angle = curl_val * abs(self._FINGER_MIN_CURL) * hyper_frac

                    x_rad = deg_to_rad(-angle)  # negative X = forward flexion

                    # Spread: Z-axis at MCP only
                    z_rad = 0.0
                    if seg == "mc" and digit in self._FINGER_SPREAD:
                        z_rad = deg_to_rad(self._FINGER_SPREAD[digit] * spread_val * mirror)

                    q = quat_from_euler(x_rad, 0.0, z_rad, "XYZ")
                    pivot.set_quaternion(q)

            # ── Thumb (digit 1): opposition ──
            for seg, frac in self._FINGER_CURL_DIST.items():
                pivot = pivots.get(f"finger_{side}_1_{seg}")
                if pivot is None:
                    continue

                if seg == "mc":
                    # CMC joint: combined opposition motion
                    flex = thumb_op * self._THUMB_OP_FLEX
                    pronate = thumb_op * self._THUMB_OP_PRONATE * mirror
                    adduct = thumb_op * self._THUMB_OP_ADDUCT * mirror
                    # Also apply curl to thumb MC
                    if curl_val >= 0:
                        flex += curl_val * self._FINGER_MAX_CURL * 0.3
                    q = quat_from_euler(
                        deg_to_rad(-flex), deg_to_rad(pronate), deg_to_rad(adduct), "XYZ",
                    )
                else:
                    # Thumb phalanges: just curl
                    if curl_val >= 0:
                        angle = curl_val * self._FINGER_MAX_CURL * frac
                    else:
                        angle = curl_val * abs(self._FINGER_MIN_CURL) * 0.1
                    q = quat_from_euler(deg_to_rad(-angle), 0.0, 0.0, "XYZ")
                pivot.set_quaternion(q)

    def _apply_feet(self, state: BodyState) -> None:
        """Apply toe curl and spread."""
        pivots = self.joints.pivots

        for side in ("R", "L"):
            s = side.lower()
            mirror = 1.0 if side == "R" else -1.0

            curl_val = getattr(state, f"toe_curl_{s}", 0.0)
            spread_val = getattr(state, f"toe_spread_{s}", 0.0)

            for digit in range(1, 6):
                for seg, frac in self._TOE_CURL_DIST.items():
                    pivot = pivots.get(f"toe_{side}_{digit}_{seg}")
                    if pivot is None:
                        continue

                    # Curl: X-axis flexion
                    if curl_val >= 0:
                        angle = curl_val * self._TOE_MAX_CURL * frac
                    else:
                        # Dorsiflexion primarily at MTP
                        hyper_frac = 1.0 if seg == "mt" else 0.1
                        angle = curl_val * abs(self._TOE_MIN_CURL) * hyper_frac

                    x_rad = deg_to_rad(-angle)

                    # Spread: Z-axis at MTP only
                    z_rad = 0.0
                    if seg == "mt" and digit in self._TOE_SPREAD:
                        z_rad = deg_to_rad(self._TOE_SPREAD[digit] * spread_val * mirror)

                    q = quat_from_euler(x_rad, 0.0, z_rad, "XYZ")
                    pivot.set_quaternion(q)

    def _apply_breathing(self, state: BodyState) -> None:
        """Apply breathing animation to ribs.

        Each rib rotates around its own centroid via a pivot wrapper node.
        Upper ribs expand more than lower ones (weighted by index).
        """
        if not self._rib_pivots:
            return

        breath = math.sin(state.breath_phase_body) * state.breath_depth
        for i, pivot in enumerate(self._rib_pivots):
            # Upper ribs expand more than lower
            weight = max(0.0, 1.0 - i * 0.02)
            angle = breath * weight * 3.0  # degrees
            q = quat_from_euler(deg_to_rad(angle), 0.0, 0.0, "XYZ")
            pivot.set_quaternion(q)

    def _build_rib_pivots(self) -> None:
        """Create pivot wrapper nodes at each rib's centroid.

        Reparents each rib mesh node under a pivot so rotation happens
        around the rib's center rather than the world origin.
        """
        from faceforge.body.joint_pivots import reparent_under_pivot

        for node in self.rib_nodes:
            if node.mesh is None:
                continue
            # Compute centroid
            geom = node.mesh.geometry
            pos = geom.positions.reshape(-1, 3)[:geom.vertex_count]
            centroid = pos.mean(axis=0)

            # Create pivot at centroid
            pivot = SceneNode(name=f"{node.name}_breath_pivot")
            pivot.set_position(float(centroid[0]), float(centroid[1]), float(centroid[2]))

            # Reparent: remove from current parent, add pivot in its place
            parent = node.parent
            if parent is not None:
                parent.remove(node)
                parent.add(pivot)

            # Offset rib vertices by -centroid so rotation is local
            reparent_under_pivot(node, pivot, centroid)
            self._rib_pivots.append(pivot)
