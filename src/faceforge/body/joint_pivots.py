"""Joint pivot setup for limb articulation.

Dynamically computes joint positions from loaded bone geometry by finding
the closest vertices between adjacent bones (matching the JS findJointCenter).
Creates chained pivot hierarchies: shoulder→elbow→wrist, hip→knee→ankle.
"""

import numpy as np

from faceforge.core.math_utils import Vec3, vec3
from faceforge.core.scene_graph import SceneNode


def find_joint_center(
    pos_a: np.ndarray, count_a: int,
    pos_b: np.ndarray, count_b: int,
) -> Vec3:
    """Find the midpoint between the two closest vertices of two meshes.

    Mirrors the JS ``findJointCenter(geoA, countA, geoB, countB)`` function.
    Subsamples for performance when vertex counts are large.
    """
    a = pos_a.reshape(-1, 3)[:count_a]
    b = pos_b.reshape(-1, 3)[:count_b]

    # Subsample for performance (match JS: max 500 samples per mesh)
    step_a = max(1, count_a // 500)
    step_b = max(1, count_b // 500)
    a_sub = a[::step_a]
    b_sub = b[::step_b]

    # Find closest pair via broadcasting
    # a_sub: (Na, 3), b_sub: (Nb, 3)
    diff = a_sub[:, np.newaxis, :] - b_sub[np.newaxis, :, :]  # (Na, Nb, 3)
    dists = np.sum(diff ** 2, axis=2)  # (Na, Nb)
    idx = np.unravel_index(np.argmin(dists), dists.shape)
    best_a = a_sub[idx[0]]
    best_b = b_sub[idx[1]]
    mid = (best_a + best_b) * 0.5
    return vec3(float(mid[0]), float(mid[1]), float(mid[2]))


def compute_bone_endpoint(
    positions: np.ndarray, vert_count: int, end: str = "top",
) -> Vec3:
    """Compute centroid of the top or bottom 5% of vertices by Y coordinate.

    Fallback when findJointCenter can't find adjacent bones.
    """
    pts = positions.reshape(-1, 3)[:vert_count]
    n = max(1, int(vert_count * 0.05))
    sorted_idx = np.argsort(pts[:, 1])
    if end == "top":
        subset = pts[sorted_idx[-n:]]
    else:
        subset = pts[sorted_idx[:n]]
    c = subset.mean(axis=0)
    return vec3(float(c[0]), float(c[1]), float(c[2]))


def reparent_under_pivot(
    node: SceneNode, pivot: SceneNode, pivot_world_pos: Vec3,
) -> None:
    """Reparent a mesh node under a pivot, offsetting so world position stays.

    Mirrors JS ``reparentUnderPivot(meshGroup, pivot, pivotWorldPos)``.
    """
    if node.parent is not None:
        node.parent.remove(node)
    # Offset mesh positions by -pivot so rotation is around the pivot point
    if node.mesh is not None:
        geom = node.mesh.geometry
        p = geom.positions.reshape(-1, 3)[:geom.vertex_count]
        p -= pivot_world_pos.astype(p.dtype)
        node.mesh.store_rest_pose()
    pivot.add(node)


def compute_tmj_pivot(
    mandible_node: SceneNode,
    temporal_nodes: list[SceneNode],
) -> Vec3:
    """Compute the TMJ (jaw hinge) pivot from BP3D mandible and temporal bones.

    Finds the joint center between the mandible and each temporal bone,
    then averages L/R for a symmetric pivot.  Falls back to the top
    endpoint of the mandible if no temporal bones are available.
    """
    if mandible_node is None or mandible_node.mesh is None:
        from faceforge.constants import JAW_PIVOT_ORIGINAL
        return vec3(*JAW_PIVOT_ORIGINAL)

    mand_geo = mandible_node.mesh.geometry
    mand_pos = mand_geo.positions
    mand_count = mand_geo.vertex_count

    centers = []
    for t_node in temporal_nodes:
        if t_node is not None and t_node.mesh is not None:
            t_geo = t_node.mesh.geometry
            center = find_joint_center(
                mand_pos, mand_count,
                t_geo.positions, t_geo.vertex_count,
            )
            centers.append(center)

    if centers:
        avg = np.mean(centers, axis=0)
        return vec3(float(avg[0]), float(avg[1]), float(avg[2]))

    # Fallback: top of mandible
    return compute_bone_endpoint(mand_pos, mand_count, "top")


class JointPivotSetup:
    """Creates and manages all body joint pivots from loaded skeleton geometry.

    After calling ``setup_from_skeleton()``, the ``pivots`` dict contains
    SceneNode pivot groups keyed by name (e.g. ``"shoulderR"``).
    """

    def __init__(self):
        self.pivots: dict[str, SceneNode] = {}
        self.joint_positions: dict[str, Vec3] = {}

    def setup(self) -> dict[str, SceneNode]:
        """Legacy setup — creates empty dict. Use setup_from_skeleton instead."""
        return self.pivots

    def setup_from_skeleton(
        self,
        bone_nodes: dict[str, SceneNode],
        body_root: SceneNode,
        upper_limb_group: SceneNode | None,
        lower_limb_group: SceneNode | None,
        hand_group: SceneNode | None,
        foot_group: SceneNode | None,
    ) -> None:
        """Compute joint positions from bone geometry and build pivot chains.

        Parameters
        ----------
        bone_nodes : dict
            Maps bone name (e.g. "Right Humerus") to its SceneNode.
        body_root : SceneNode
            Top-level body group node.
        upper_limb_group, lower_limb_group, hand_group, foot_group : SceneNode
            Skeleton group nodes containing the bone meshes.
        """
        for side_char in ("R", "L"):
            side_label = "Right" if side_char == "R" else "Left"

            # ── Upper limb chain: shoulder → elbow → wrist ──
            self._setup_upper_limb(
                side_char, side_label, bone_nodes, body_root,
                upper_limb_group, hand_group,
            )

            # ── Lower limb chain: hip → knee → ankle ──
            self._setup_lower_limb(
                side_char, side_label, bone_nodes, body_root,
                lower_limb_group, foot_group,
            )

        # ── Digit chains (after limbs so wrist/ankle pivots exist) ──
        for side_char in ("R", "L"):
            wrist = self.pivots.get(f"wrist_{side_char}")
            wrist_pos = self.joint_positions.get(f"wrist_{side_char}")
            if wrist is not None and wrist_pos is not None:
                self._setup_hand_digits(side_char, wrist, wrist_pos)

            ankle = self.pivots.get(f"ankle_{side_char}")
            ankle_pos = self.joint_positions.get(f"ankle_{side_char}")
            if ankle is not None and ankle_pos is not None:
                self._setup_foot_digits(side_char, ankle, ankle_pos)

    def _setup_upper_limb(
        self, side: str, label: str, bones: dict,
        body_root: SceneNode,
        ul_group: SceneNode | None,
        hand_group: SceneNode | None,
    ) -> None:
        scapula = bones.get(f"{label} Scapula")
        humerus = bones.get(f"{label} Humerus")
        radius = bones.get(f"{label} Radius")
        ulna = bones.get(f"{label} Ulna")

        if humerus is None or humerus.mesh is None:
            return

        # Compute shoulder position
        shoulder_pos = None
        if scapula and scapula.mesh:
            shoulder_pos = find_joint_center(
                scapula.mesh.geometry.positions, scapula.mesh.geometry.vertex_count,
                humerus.mesh.geometry.positions, humerus.mesh.geometry.vertex_count,
            )
        if shoulder_pos is None:
            shoulder_pos = compute_bone_endpoint(
                humerus.mesh.geometry.positions, humerus.mesh.geometry.vertex_count, "top",
            )

        # Compute elbow position
        elbow_pos = None
        if radius and radius.mesh:
            elbow_pos = find_joint_center(
                humerus.mesh.geometry.positions, humerus.mesh.geometry.vertex_count,
                radius.mesh.geometry.positions, radius.mesh.geometry.vertex_count,
            )
        elif ulna and ulna.mesh:
            elbow_pos = find_joint_center(
                humerus.mesh.geometry.positions, humerus.mesh.geometry.vertex_count,
                ulna.mesh.geometry.positions, ulna.mesh.geometry.vertex_count,
            )
        if elbow_pos is None:
            elbow_pos = compute_bone_endpoint(
                humerus.mesh.geometry.positions, humerus.mesh.geometry.vertex_count, "bottom",
            )

        # Compute wrist position
        wrist_pos = None
        if radius and radius.mesh:
            wrist_pos = compute_bone_endpoint(
                radius.mesh.geometry.positions, radius.mesh.geometry.vertex_count, "bottom",
            )

        # Create pivot nodes
        shoulder_pivot = SceneNode(name=f"shoulder_{side}_pivot")
        shoulder_pivot.set_position(
            float(shoulder_pos[0]), float(shoulder_pos[1]), float(shoulder_pos[2]),
        )

        elbow_pivot = SceneNode(name=f"elbow_{side}_pivot")
        # Elbow position is relative to shoulder
        rel_elbow = elbow_pos - shoulder_pos
        elbow_pivot.set_position(
            float(rel_elbow[0]), float(rel_elbow[1]), float(rel_elbow[2]),
        )

        wrist_pivot = None
        if wrist_pos is not None:
            wrist_pivot = SceneNode(name=f"wrist_{side}_pivot")
            rel_wrist = wrist_pos - elbow_pos
            wrist_pivot.set_position(
                float(rel_wrist[0]), float(rel_wrist[1]), float(rel_wrist[2]),
            )

        # Build chain: shoulder → elbow → wrist
        shoulder_pivot.add(elbow_pivot)
        if wrist_pivot is not None:
            elbow_pivot.add(wrist_pivot)

        # Reparent humerus under shoulder pivot
        reparent_under_pivot(humerus, shoulder_pivot, shoulder_pos)

        # Scapulohumeral rhythm: create scapula pivot for coupled rotation
        if scapula and scapula.mesh:
            scap_geo = scapula.mesh.geometry
            scap_pts = scap_geo.positions.reshape(-1, 3)[:scap_geo.vertex_count]
            scap_center = scap_pts.mean(axis=0)
            scap_pivot = SceneNode(name=f"scapula_{side}_pivot")
            scap_pivot.set_position(
                float(scap_center[0]), float(scap_center[1]), float(scap_center[2]),
            )
            parent = scapula.parent
            if parent is not None:
                parent.remove(scapula)
                parent.add(scap_pivot)
            reparent_under_pivot(scapula, scap_pivot, scap_center)
            self.pivots[f"scapula_{side}"] = scap_pivot
            self.joint_positions[f"scapula_{side}"] = vec3(
                float(scap_center[0]), float(scap_center[1]), float(scap_center[2]),
            )

        if radius:
            reparent_under_pivot(radius, elbow_pivot, elbow_pos)
        if ulna:
            reparent_under_pivot(ulna, elbow_pivot, elbow_pos)

        # Reparent hand bones
        if hand_group is not None:
            side_prefix = "R " if side == "R" else "L "
            for child in list(hand_group.children):
                if child.name and child.name.startswith(side_prefix):
                    target = wrist_pivot if wrist_pivot else elbow_pivot
                    target_pos = wrist_pos if wrist_pivot else elbow_pos
                    reparent_under_pivot(child, target, target_pos)

        # Add shoulder pivot to body_root (or upper limb group)
        if ul_group is not None:
            # Keep remaining non-reparented children in ul_group
            pass
        body_root.add(shoulder_pivot)

        # Store references
        self.pivots[f"shoulder_{side}"] = shoulder_pivot
        self.pivots[f"elbow_{side}"] = elbow_pivot
        if wrist_pivot:
            self.pivots[f"wrist_{side}"] = wrist_pivot
        self.joint_positions[f"shoulder_{side}"] = shoulder_pos
        self.joint_positions[f"elbow_{side}"] = elbow_pos
        if wrist_pos is not None:
            self.joint_positions[f"wrist_{side}"] = wrist_pos

    def _setup_lower_limb(
        self, side: str, label: str, bones: dict,
        body_root: SceneNode,
        ll_group: SceneNode | None,
        foot_group: SceneNode | None,
    ) -> None:
        femur = bones.get(f"{label} Femur")
        patella = bones.get(f"{label} Patella")
        tibia = bones.get(f"{label} Tibia")
        fibula = bones.get(f"{label} Fibula")

        if femur is None or femur.mesh is None:
            return

        # Compute hip position
        # Try to find hip bone
        hip_bone = bones.get(f"{label} Hip Bone")
        hip_pos = None
        if hip_bone and hip_bone.mesh:
            hip_pos = find_joint_center(
                hip_bone.mesh.geometry.positions, hip_bone.mesh.geometry.vertex_count,
                femur.mesh.geometry.positions, femur.mesh.geometry.vertex_count,
            )
        if hip_pos is None:
            hip_pos = compute_bone_endpoint(
                femur.mesh.geometry.positions, femur.mesh.geometry.vertex_count, "top",
            )

        # Compute knee position
        knee_pos = None
        if tibia and tibia.mesh:
            knee_pos = find_joint_center(
                femur.mesh.geometry.positions, femur.mesh.geometry.vertex_count,
                tibia.mesh.geometry.positions, tibia.mesh.geometry.vertex_count,
            )
        if knee_pos is None:
            knee_pos = compute_bone_endpoint(
                femur.mesh.geometry.positions, femur.mesh.geometry.vertex_count, "bottom",
            )

        # Compute ankle position
        ankle_pos = None
        if tibia and tibia.mesh:
            # Find talus in foot bones
            talus = bones.get(f"R Talus" if side == "R" else "L Talus")
            if talus and talus.mesh:
                ankle_pos = find_joint_center(
                    tibia.mesh.geometry.positions, tibia.mesh.geometry.vertex_count,
                    talus.mesh.geometry.positions, talus.mesh.geometry.vertex_count,
                )
            if ankle_pos is None:
                ankle_pos = compute_bone_endpoint(
                    tibia.mesh.geometry.positions, tibia.mesh.geometry.vertex_count, "bottom",
                )

        # Create pivot nodes
        hip_pivot = SceneNode(name=f"hip_{side}_pivot")
        hip_pivot.set_position(
            float(hip_pos[0]), float(hip_pos[1]), float(hip_pos[2]),
        )

        knee_pivot = SceneNode(name=f"knee_{side}_pivot")
        rel_knee = knee_pos - hip_pos
        knee_pivot.set_position(
            float(rel_knee[0]), float(rel_knee[1]), float(rel_knee[2]),
        )

        ankle_pivot = None
        if ankle_pos is not None:
            ankle_pivot = SceneNode(name=f"ankle_{side}_pivot")
            rel_ankle = ankle_pos - knee_pos
            ankle_pivot.set_position(
                float(rel_ankle[0]), float(rel_ankle[1]), float(rel_ankle[2]),
            )

        # Build chain: hip → knee → ankle
        hip_pivot.add(knee_pivot)
        if ankle_pivot is not None:
            knee_pivot.add(ankle_pivot)

        # Reparent bones under pivots
        reparent_under_pivot(femur, hip_pivot, hip_pos)
        if patella:
            reparent_under_pivot(patella, hip_pivot, hip_pos)

        if tibia:
            reparent_under_pivot(tibia, knee_pivot, knee_pos)
        if fibula:
            reparent_under_pivot(fibula, knee_pivot, knee_pos)

        # Reparent foot bones
        if foot_group is not None and ankle_pos is not None:
            side_prefix = "R " if side == "R" else "L "
            target = ankle_pivot if ankle_pivot else knee_pivot
            target_pos = ankle_pos if ankle_pivot else knee_pos
            for child in list(foot_group.children):
                if child.name and child.name.startswith(side_prefix):
                    reparent_under_pivot(child, target, target_pos)

        # Hip pivots are direct bodyRoot children (NOT inside lowerLimbGroup)
        body_root.add(hip_pivot)

        # Store references
        self.pivots[f"hip_{side}"] = hip_pivot
        self.pivots[f"knee_{side}"] = knee_pivot
        if ankle_pivot:
            self.pivots[f"ankle_{side}"] = ankle_pivot
        self.joint_positions[f"hip_{side}"] = hip_pos
        self.joint_positions[f"knee_{side}"] = knee_pos
        if ankle_pos is not None:
            self.joint_positions[f"ankle_{side}"] = ankle_pos

    # ── Digit pivot chain builders ──────────────────────────────────

    # Hand bone name patterns (from hand.json):
    # Metacarpals: "{S} {N}st/nd/rd/th Metacarpal"
    # Phalanges: "{S} Thumb Prox./Dist.", "{S} Index Prox./Mid./Dist.", etc.
    _FINGER_NAMES = {
        1: ("Thumb", ["prox", "dist"]),       # 2 phalanges
        2: ("Index", ["prox", "mid", "dist"]),  # 3 phalanges
        3: ("Middle", ["prox", "mid", "dist"]),
        4: ("Ring", ["prox", "mid", "dist"]),
        5: ("Little", ["prox", "mid", "dist"]),
    }

    _ORDINAL = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th"}

    # Foot bone name patterns (from foot.json):
    # Metatarsals: "{S} {N}st/nd/rd/th Metatarsal"
    # Phalanges: "{S} Big Toe Prox./Dist.", "{S} 2nd Toe Prox./Mid./Dist.", etc.
    _TOE_NAMES = {
        1: ("Big Toe", ["prox", "dist"]),       # 2 phalanges
        2: ("2nd Toe", ["prox", "mid", "dist"]),  # 3 phalanges
        3: ("3rd Toe", ["prox", "mid", "dist"]),
        4: ("4th Toe", ["prox", "mid", "dist"]),
        5: ("5th Toe", ["prox", "mid", "dist"]),
    }

    def _setup_hand_digits(
        self, side: str, wrist_pivot: SceneNode, wrist_pos: Vec3,
    ) -> None:
        """Build per-digit pivot chains for hand bones under the wrist pivot.

        For each digit 1-5, creates a chain: metacarpal → prox → [mid] → dist.
        Each pivot is placed at the bone centroid; bones are reparented under it.
        """
        side_prefix = f"{side} "

        # Collect children by name for O(1) lookup
        children_by_name: dict[str, SceneNode] = {}
        for child in self._collect_all_descendants(wrist_pivot):
            if child.name:
                children_by_name[child.name] = child

        for digit in range(1, 6):
            finger_label, seg_list = self._FINGER_NAMES[digit]
            ordinal = self._ORDINAL[digit]

            # Metacarpal bone name: e.g. "R 1st Metacarpal"
            mc_name = f"{side_prefix}{ordinal} Metacarpal"
            mc_node = children_by_name.get(mc_name)

            # Build ordered segment list: mc, prox, [mid], dist
            segments = [("mc", mc_name, mc_node)]
            for seg in seg_list:
                seg_label = seg.capitalize()
                # Phalanx name: e.g. "R Thumb Prox." or "R Index Mid."
                ph_name = f"{side_prefix}{finger_label} {seg_label}."
                ph_node = children_by_name.get(ph_name)
                segments.append((seg, ph_name, ph_node))

            # Build chained pivots.
            # Bone vertices are already in wrist-local space (offset by
            # -wrist_pos during _setup_upper_limb reparenting), so centroids
            # are wrist-local.  Start prev_pos at the local origin.
            prev_pivot = wrist_pivot
            prev_pos = vec3(0.0, 0.0, 0.0)
            for seg_id, bone_name, bone_node in segments:
                if bone_node is None or bone_node.mesh is None:
                    continue
                # Compute centroid in parent-local space
                geom = bone_node.mesh.geometry
                pts = geom.positions.reshape(-1, 3)[:geom.vertex_count]
                centroid = pts.mean(axis=0)
                pivot_pos = vec3(float(centroid[0]), float(centroid[1]), float(centroid[2]))

                pivot_name = f"finger_{side}_{digit}_{seg_id}_pivot"
                pivot = SceneNode(name=pivot_name)
                # Position relative to parent pivot (both in same local space)
                rel = pivot_pos - prev_pos
                pivot.set_position(float(rel[0]), float(rel[1]), float(rel[2]))

                prev_pivot.add(pivot)
                reparent_under_pivot(bone_node, pivot, pivot_pos)

                self.pivots[f"finger_{side}_{digit}_{seg_id}"] = pivot
                self.joint_positions[f"finger_{side}_{digit}_{seg_id}"] = pivot_pos

                prev_pivot = pivot
                prev_pos = pivot_pos

    def _setup_foot_digits(
        self, side: str, ankle_pivot: SceneNode, ankle_pos: Vec3,
    ) -> None:
        """Build per-digit pivot chains for foot bones under the ankle pivot.

        For each digit 1-5, creates a chain: metatarsal → prox → [mid] → dist.
        """
        side_prefix = f"{side} "

        children_by_name: dict[str, SceneNode] = {}
        for child in self._collect_all_descendants(ankle_pivot):
            if child.name:
                children_by_name[child.name] = child

        for digit in range(1, 6):
            toe_label, seg_list = self._TOE_NAMES[digit]
            ordinal = self._ORDINAL[digit]

            # Metatarsal bone name: e.g. "R 1st Metatarsal"
            mt_name = f"{side_prefix}{ordinal} Metatarsal"
            mt_node = children_by_name.get(mt_name)

            segments = [("mt", mt_name, mt_node)]
            for seg in seg_list:
                seg_label = seg.capitalize()
                # Phalanx name: e.g. "R Big Toe Prox." or "R 2nd Toe Mid."
                ph_name = f"{side_prefix}{toe_label} {seg_label}."
                ph_node = children_by_name.get(ph_name)
                segments.append((seg, ph_name, ph_node))

            # Build chained pivots (bone vertices are in ankle-local space)
            prev_pivot = ankle_pivot
            prev_pos = vec3(0.0, 0.0, 0.0)
            for seg_id, bone_name, bone_node in segments:
                if bone_node is None or bone_node.mesh is None:
                    continue
                geom = bone_node.mesh.geometry
                pts = geom.positions.reshape(-1, 3)[:geom.vertex_count]
                centroid = pts.mean(axis=0)
                pivot_pos = vec3(float(centroid[0]), float(centroid[1]), float(centroid[2]))

                pivot_name = f"toe_{side}_{digit}_{seg_id}_pivot"
                pivot = SceneNode(name=pivot_name)
                rel = pivot_pos - prev_pos
                pivot.set_position(float(rel[0]), float(rel[1]), float(rel[2]))

                prev_pivot.add(pivot)
                reparent_under_pivot(bone_node, pivot, pivot_pos)

                self.pivots[f"toe_{side}_{digit}_{seg_id}"] = pivot
                self.joint_positions[f"toe_{side}_{digit}_{seg_id}"] = pivot_pos

                prev_pivot = pivot
                prev_pos = pivot_pos

    @staticmethod
    def _collect_all_descendants(node: SceneNode) -> list[SceneNode]:
        """Recursively collect all descendant nodes."""
        result = []
        stack = list(node.children)
        while stack:
            child = stack.pop()
            result.append(child)
            stack.extend(child.children)
        return result
