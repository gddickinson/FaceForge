"""Body animation: spine flex/bend/rotation, limb articulation, breathing."""

import math

import numpy as np

from faceforge.core.math_utils import (
    Vec3, Quat, deg_to_rad, quat_from_euler, quat_from_axis_angle,
    quat_identity, quat_multiply, quat_rotate_vec3, vec3,
)
from faceforge.core.state import BodyState
from faceforge.core.scene_graph import SceneNode
from faceforge.core.config_loader import load_skeleton_config
from faceforge.body.joint_pivots import JointPivotSetup
from faceforge.body.dof_ranges import dof_range


def _rad(pattern: str, value: float) -> float:
    """A normalised DOF value as radians, via the shared range table."""
    return deg_to_rad(dof_range(pattern).degrees_for(value))


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

    # Thoracic/lumbar share of total spine motion
    _THORACIC_SHARE = 0.4
    _LUMBAR_SHARE = 0.6

    def _apply_spine(self, state: BodyState) -> None:
        """Distribute spine flex/bend/rotation across vertebral pivots.

        JS convention: rotation.set(X=flex, Y=rotation, Z=latBend)
        Total rotation is split: 40% thoracic, 60% lumbar (anatomical ratio).
        """
        flex_rad = _rad("spine_flex", state.spine_flex)
        bend_rad = _rad("spine_lat_bend", state.spine_lat_bend)
        rot_rad = _rad("spine_rotation", state.spine_rotation)

        # Split total angle between thoracic and lumbar regions
        flex_thoracic = flex_rad * self._THORACIC_SHARE
        bend_thoracic = bend_rad * self._THORACIC_SHARE
        rot_thoracic = rot_rad * self._THORACIC_SHARE

        flex_lumbar = flex_rad * self._LUMBAR_SHARE
        bend_lumbar = bend_rad * self._LUMBAR_SHARE
        rot_lumbar = rot_rad * self._LUMBAR_SHARE

        # Thoracic distribution
        for i, pivot_info in enumerate(self.thoracic_pivots):
            if i >= len(self.thoracic_fracs):
                break
            frac = self.thoracic_fracs[i]
            pivot_node = pivot_info["group"]

            x = flex_thoracic * frac
            y = rot_thoracic * frac
            z = bend_thoracic * frac

            q = quat_from_euler(x, y, z, "XYZ")
            pivot_node.set_quaternion(q)

        # Lumbar distribution
        for i, pivot_info in enumerate(self.lumbar_pivots):
            if i >= len(self.lumbar_fracs):
                break
            frac = self.lumbar_fracs[i]
            pivot_node = pivot_info["group"]

            x = flex_lumbar * frac
            y = rot_lumbar * frac
            z = bend_lumbar * frac

            q = quat_from_euler(x, y, z, "XYZ")
            pivot_node.set_quaternion(q)

    def _apply_limbs(self, state: BodyState) -> None:
        """Apply limb joint rotations.

        Body frame (measured, see ``docs/exercise_animation.md``): +Z superior,
        -Y anterior, +X right.  So for a limb hanging along -Z:

        * flexion / extension is a rotation about X, the lateral axis
          (negative X swings the limb anteriorly);
        * abduction / adduction is a rotation about Y, the anterior-posterior
          axis (the sign is mirrored so +1 is lateral on both sides);
        * axial rotation is a rotation about Z, the limb's own long axis
          (+1 = external rotation on both sides).

        The port originally kept the JS Y-up assignments (abduction about Z,
        rotation about Y).  In this Z-up frame that made "abduct" spin the limb
        about its length and "rotate" swing it sideways -- measured on the real
        skeleton: shoulder_r_abduct=1 moved the wrist 9 units, shoulder_r_rotate=1
        moved it 76 units medially.  Every range comes from
        :mod:`faceforge.body.dof_ranges` so authoring tools and this code agree.
        """
        pivots = self.joints.pivots  # dict[str, SceneNode]

        for side in ("R", "L"):
            s = side.lower()
            mirror = 1.0 if side == "R" else -1.0

            # ── Shoulder ──
            shoulder = pivots.get(f"shoulder_{side}")
            ab_val = getattr(state, f"shoulder_{s}_abduct", 0.0)
            ab_rad = -_rad("shoulder_{s}_abduct", ab_val) * mirror
            if shoulder is not None:
                fl_val = getattr(state, f"shoulder_{s}_flex", 0.0)
                rot_val = getattr(state, f"shoulder_{s}_rotate", 0.0)
                fl_rad = -_rad("shoulder_{s}_flex", fl_val)
                rot_rad = _rad("shoulder_{s}_rotate", rot_val) * mirror
                q = quat_from_euler(fl_rad, ab_rad, rot_rad, "XYZ")
                shoulder.set_quaternion(q)

            # ── Scapulohumeral rhythm ──
            scapula = pivots.get(f"scapula_{side}")
            if scapula is not None:
                self._apply_girdle(side, mirror, ab_rad, scapula,
                                   pivots.get(f"clavicle_{side}"))

            # ── Elbow: flexion about X ──
            elbow = pivots.get(f"elbow_{side}")
            if elbow is not None:
                el_val = getattr(state, f"elbow_{s}_flex", 0.0)
                el_rad = -_rad("elbow_{s}_flex", el_val)
                q = quat_from_euler(el_rad, 0.0, 0.0, "XYZ")
                elbow.set_quaternion(q)

            # ── Hip ──
            hip = pivots.get(f"hip_{side}")
            if hip is not None:
                hf_val = getattr(state, f"hip_{s}_flex", 0.0)
                hab_val = getattr(state, f"hip_{s}_abduct", 0.0)
                hrot_val = getattr(state, f"hip_{s}_rotate", 0.0)
                hf_rad = -_rad("hip_{s}_flex", hf_val)
                hab_rad = -_rad("hip_{s}_abduct", hab_val) * mirror
                hrot_rad = _rad("hip_{s}_rotate", hrot_val) * mirror
                q = quat_from_euler(hf_rad, hab_rad, hrot_rad, "XYZ")
                hip.set_quaternion(q)

            # ── Knee: flexion is a positive X rotation (heel toward buttock) ──
            knee = pivots.get(f"knee_{side}")
            if knee is not None:
                kn_val = getattr(state, f"knee_{s}_flex", 0.0)
                kn_rad = _rad("knee_{s}_flex", kn_val)
                q = quat_from_euler(kn_rad, 0.0, 0.0, "XYZ")
                knee.set_quaternion(q)

            # ── Ankle: dorsiflexion about X; inversion about the foot's own
            # anterior-posterior axis (Y), lateral border dropping ──
            ankle = pivots.get(f"ankle_{side}")
            if ankle is not None:
                an_val = getattr(state, f"ankle_{s}_flex", 0.0)
                an_inv = getattr(state, f"ankle_{s}_invert", 0.0)
                an_rad = -_rad("ankle_{s}_flex", an_val)
                inv_rad = _rad("ankle_{s}_invert", an_inv) * mirror
                q = quat_from_euler(an_rad, inv_rad, 0.0, "XYZ")
                ankle.set_quaternion(q)

            # ── Wrist: flexion about X, ulnar deviation about Y, and
            # pronation/supination about the forearm's long axis (Z) ──
            wrist = pivots.get(f"wrist_{side}")
            if wrist is not None:
                wr_flex = getattr(state, f"wrist_{s}_flex", 0.0)
                wr_dev = getattr(state, f"wrist_{s}_deviate", 0.0)
                fa_rot = getattr(state, f"forearm_{s}_rotate", 0.0)
                wr_fl_rad = -_rad("wrist_{s}_flex", wr_flex)
                wr_dev_rad = _rad("wrist_{s}_deviate", wr_dev) * mirror
                fa_rot_rad = _rad("forearm_{s}_rotate", fa_rot) * mirror
                # Pronation happens in the forearm, proximal to the wrist, so
                # it is the OUTERMOST rotation: the wrist's flexion axis (and
                # the fingers') turns with it.  Composed as XYZ it was the
                # innermost, which left the flexion axis fixed in the forearm
                # and made "wrist extension" of a pronated hand act as
                # deviation -- no grip pose could face the palm at a bar.
                q = quat_multiply(quat_from_euler(0.0, 0.0, fa_rot_rad, "XYZ"),
                                  quat_from_euler(wr_fl_rad, wr_dev_rad, 0.0, "XYZ"))
                wrist.set_quaternion(q)

    # ── Digit animation ──────────────────────────────────────────────

    # Finger curl distribution: MCP 40%, PIP 35%, DIP 25%
    # Finger flexion at slider=1.0, per joint.  The digit pivots sit at the
    # proximal end of each bone, so "prox" is the metacarpophalangeal joint,
    # "mid" the proximal and "dist" the distal interphalangeal joint, and
    # "mc" the carpometacarpal joint (a few degrees, ulnar fingers only).
    # A closed fist is roughly 90/100/60 degrees at MCP/PIP/DIP (functional
    # tasks use about 60/60/40: Hume et al., J Hand Surg 1990).  The old
    # model spread 90 degrees in total over four joints, which cannot close
    # a hand round a bar -- the bar passed through the palm in every render.
    _FINGER_CURL_MAX = {"mc": 8.0, "prox": 90.0, "mid": 100.0, "dist": 60.0}
    _FINGER_HYPER = {"prox": 1.0, "mid": 0.1, "dist": 0.1}
    _FINGER_MIN_CURL = -20.0  # slight hyperextension at slider=-1.0 (MCP only)
    _THUMB_CURL_MAX = {"mc": 25.0, "prox": 55.0, "dist": 80.0}

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

    #: Scapulothoracic share of arm elevation: 1 deg per 2 deg of glenohumeral
    #: motion (Inman 1944), i.e. a third of the total.
    _SCAPULA_SHARE = 1.0 / 3.0
    #: Clavicle elevation at the sternoclavicular joint per degree of arm
    #: elevation: about 30 deg at a full 165.
    _CLAVICLE_ELEVATION = 30.0 / 165.0

    def _girdle_rest(self, side: str, scapula: SceneNode, clavicle: SceneNode | None):
        """Rest geometry of one side's girdle, read once from the pivots' own frames."""
        cache = getattr(self, "_girdle_cache", None)
        if cache is None:
            cache = self._girdle_cache = {}
        hit = cache.get(side)
        if hit is not None:
            return hit
        scap_pos = np.asarray(scapula.position, dtype=np.float64).copy()
        clav_pos = None
        ac = None
        if clavicle is not None:
            clav_pos = np.asarray(clavicle.position, dtype=np.float64).copy()
            for child in clavicle.children:
                geo = getattr(getattr(child, "mesh", None), "geometry", None)
                if geo is None or geo.positions is None:
                    continue
                pts = np.asarray(geo.positions, dtype=np.float64).reshape(-1, 3)
                pts = pts[:geo.vertex_count] if getattr(geo, "vertex_count", 0) else pts
                if len(pts):
                    # The acromial end: the clavicle's most lateral vertex.
                    lateral = pts[:, 0].argmax() if scap_pos[0] >= 0 else pts[:, 0].argmin()
                    ac = clav_pos + pts[lateral]
                    break
        # Upward rotation happens about the thorax's local surface normal at
        # the scapula, so the blade glides round the ribcage instead of
        # swinging out of it in the coronal plane.
        radial = np.array([scap_pos[0], scap_pos[1], 0.0])
        n = float(np.linalg.norm(radial))
        axis = radial / n if n > 1e-6 else np.array([0.0, 1.0, 0.0])
        hit = (scap_pos, clav_pos, ac, axis)
        cache[side] = hit
        return hit

    def _apply_girdle(self, side: str, mirror: float, ab_rad: float,
                      scapula: SceneNode, clavicle: SceneNode | None) -> None:
        """Scapular upward rotation on the thorax, and clavicle elevation.

        The first version rotated the scapula about an anterior-posterior
        axis through its centroid.  Measured at 165 deg of abduction that put
        the inferior angle at x = 28.3, seven units outside the ribcage's
        lateral extent (21), and every muscle attached to the blade -- teres
        major, infraspinatus, subscapularis -- stood out from the trunk as a
        wing.  Rotating about the thorax's surface normal at the scapula keeps
        the inferior angle on the ribcage (it moves laterally AND forward
        round the curve), and elevating the clavicle at the sternoclavicular
        joint carries the acromion, and with it the whole blade, upward.
        """
        scap_rest, clav_rest, ac_rest, axis = self._girdle_rest(side, scapula, clavicle)
        upward = ab_rad * self._SCAPULA_SHARE
        if abs(upward) < 1e-9:
            scapula.set_quaternion(np.array([0.0, 0.0, 0.0, 1.0]))
            scapula.set_position(*scap_rest)
            if clavicle is not None:
                clavicle.set_quaternion(np.array([0.0, 0.0, 0.0, 1.0]))
            return
        q_scap = quat_from_axis_angle(axis, upward)
        scapula.set_quaternion(q_scap)
        if clavicle is None or clav_rest is None or ac_rest is None:
            scapula.set_position(*scap_rest)
            return
        # Elevation only while the arm rises above the side; adduction below
        # neutral does not depress the clavicle here.
        elevation = max(0.0, -ab_rad * mirror) * self._CLAVICLE_ELEVATION
        q_clav = quat_from_euler(0.0, -elevation * mirror, 0.0, "XYZ")
        clavicle.set_quaternion(q_clav)
        # Keep the acromioclavicular joint together: the blade goes wherever
        # the clavicle's acromial end went.
        ac_clav = clav_rest + quat_rotate_vec3(q_clav, ac_rest - clav_rest)
        ac_scap = scap_rest + quat_rotate_vec3(q_scap, ac_rest - scap_rest)
        shift = ac_clav - ac_scap
        scapula.set_position(*(scap_rest + shift))

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
                for seg, max_deg in self._FINGER_CURL_MAX.items():
                    pivot = pivots.get(f"finger_{side}_{digit}_{seg}")
                    if pivot is None:
                        continue

                    # Curl: X-axis flexion, each joint to its own maximum
                    if curl_val >= 0:
                        angle = curl_val * max_deg
                    else:
                        # Hyperextension only at the MCP, minimal at PIP/DIP
                        hyper_frac = self._FINGER_HYPER.get(seg, 0.0)
                        angle = curl_val * abs(self._FINGER_MIN_CURL) * hyper_frac

                    x_rad = deg_to_rad(-angle)  # negative X = forward flexion

                    # Spread: fans in the palm plane, about the AP axis (Y),
                    # at the MCP only
                    y_rad = 0.0
                    if seg == "mc" and digit in self._FINGER_SPREAD:
                        y_rad = deg_to_rad(self._FINGER_SPREAD[digit] * spread_val * mirror)

                    q = quat_from_euler(x_rad, y_rad, 0.0, "XYZ")
                    pivot.set_quaternion(q)

            # ── Thumb (digit 1): opposition ──
            for seg, max_deg in self._THUMB_CURL_MAX.items():
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
                        flex += curl_val * max_deg
                    q = quat_from_euler(
                        deg_to_rad(-flex), deg_to_rad(pronate), deg_to_rad(adduct), "XYZ",
                    )
                else:
                    # Thumb phalanges: just curl
                    if curl_val >= 0:
                        angle = curl_val * max_deg
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
        n_ribs = len(self._rib_pivots)
        for i, pivot in enumerate(self._rib_pivots):
            # Exponential decay: upper ribs (pump-handle) expand more than
            # lower ribs (bucket-handle), matching anatomical breathing mechanics
            t = i / max(1, n_ribs - 1)  # 0..1 from top to bottom
            weight = math.exp(-2.5 * t)  # ~1.0 at top, ~0.08 at bottom
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
