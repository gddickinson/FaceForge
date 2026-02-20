"""Quantitative head rotation diagnostic — checks that all head groups
follow rotation and that neck muscles stay attached to the body skeleton.

Usage::

    from tools.head_rotation_diagnostic import run_full_diagnostic
    from tools.headless_loader import load_headless_scene

    hs = load_headless_scene()
    results = run_full_diagnostic(hs)
    print(format_diagnostic_report(results))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from faceforge.core.math_utils import Quat, quat_identity, quat_rotate_vec3
from faceforge.core.scene_graph import SceneNode
from faceforge.core.state import FaceState, BodyState

logger = logging.getLogger(__name__)


# ── Head poses ────────────────────────────────────────────────────────

HEAD_POSES: dict[str, dict[str, float]] = {
    "neutral":      {"head_yaw": 0.0, "head_pitch": 0.0,  "head_roll": 0.0},
    "yaw_left":     {"head_yaw": 0.8, "head_pitch": 0.0,  "head_roll": 0.0},
    "yaw_right":    {"head_yaw": -0.8, "head_pitch": 0.0, "head_roll": 0.0},
    "pitch_up":     {"head_yaw": 0.0, "head_pitch": 0.8,  "head_roll": 0.0},
    "pitch_down":   {"head_yaw": 0.0, "head_pitch": -0.8, "head_roll": 0.0},
    "roll_left":    {"head_yaw": 0.0, "head_pitch": 0.0,  "head_roll": 0.8},
    "roll_right":   {"head_yaw": 0.0, "head_pitch": 0.0,  "head_roll": -0.8},
    "combined":     {"head_yaw": 0.5, "head_pitch": 0.3,  "head_roll": 0.2},
}

# Combined body + head poses for neck attachment testing
BODY_HEAD_POSES: dict[str, dict] = {
    "head_yaw_spine_flex": {
        "face": {"head_yaw": 0.8},
        "body": {"spine_flex": 0.4},
    },
    "head_pitch_shoulder": {
        "face": {"head_pitch": 0.6},
        "body": {"shoulder_r_abduct": 0.5},
    },
    "combined_all": {
        "face": {"head_yaw": 0.5, "head_pitch": 0.3},
        "body": {"spine_flex": 0.3, "spine_rotation": 0.2},
    },
}


# ── Result dataclasses ────────────────────────────────────────────────

@dataclass
class GroupFollowResult:
    """Per-group centroid follow error for a single pose."""
    group_name: str
    expected_centroid: NDArray[np.float64]
    actual_centroid: NDArray[np.float64]
    error: float  # distance between expected and actual


@dataclass
class PoseFollowResult:
    """Follow errors for all groups in a single pose."""
    pose_name: str
    group_results: list[GroupFollowResult]
    max_error: float


@dataclass
class NeckGapResult:
    """Per-muscle neck-body gap measurement."""
    muscle_name: str
    rest_gap: float
    posed_gap: float
    gap_ratio: float  # posed_gap / rest_gap


@dataclass
class NeckGapPoseResult:
    """Neck gap results for a single combined pose."""
    pose_name: str
    muscle_gaps: list[NeckGapResult]
    max_gap_ratio: float


@dataclass
class BoneProximityResult:
    """Per-muscle bone proximity measurement."""
    muscle_name: str
    bone_names: list[str]
    distance: float  # distance from lower verts to bone anchor


@dataclass
class BoneProximityPoseResult:
    """Bone proximity results for a single pose."""
    pose_name: str
    muscle_results: list[BoneProximityResult]
    max_distance: float


@dataclass
class StretchRatioResult:
    """Per-muscle stretch ratio measurement."""
    muscle_name: str
    stretch_ratio: float  # current_length / rest_length
    centroid_displacement: float  # how far the whole muscle centroid moved


@dataclass
class StretchRatioPoseResult:
    """Stretch ratio results for a single pose."""
    pose_name: str
    muscle_results: list[StretchRatioResult]
    min_stretch_ratio: float
    max_centroid_displacement: float


@dataclass
class PinEffectivenessResult:
    """Per-muscle pin effectiveness measurement."""
    muscle_name: str
    pin_error: float  # distance from pinned verts to bone target


@dataclass
class PinEffectivenessPoseResult:
    """Pin effectiveness results for a single pose."""
    pose_name: str
    muscle_results: list[PinEffectivenessResult]
    max_pin_error: float


@dataclass
class PlatysmaAttachmentResult:
    """Per-muscle Platysma attachment measurement."""
    muscle_name: str
    lower_distance: float  # distance from lower verts to clavicle


@dataclass
class PlatysmaAttachmentPoseResult:
    """Platysma attachment results for a single pose."""
    pose_name: str
    muscle_results: list[PlatysmaAttachmentResult]
    max_distance: float


@dataclass
class DiagnosticResults:
    """Full diagnostic results."""
    follow_results: list[PoseFollowResult] = field(default_factory=list)
    neck_gap_results: list[NeckGapPoseResult] = field(default_factory=list)
    bone_proximity_results: list[BoneProximityPoseResult] = field(default_factory=list)
    stretch_ratio_results: list[StretchRatioPoseResult] = field(default_factory=list)
    pin_effectiveness_results: list[PinEffectivenessPoseResult] = field(default_factory=list)
    platysma_attachment_results: list[PlatysmaAttachmentPoseResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "follow_results": [
                {
                    "pose": pr.pose_name,
                    "max_error": pr.max_error,
                    "groups": [
                        {
                            "name": gr.group_name,
                            "error": gr.error,
                        }
                        for gr in pr.group_results
                    ],
                }
                for pr in self.follow_results
            ],
            "neck_gap_results": [
                {
                    "pose": nr.pose_name,
                    "max_gap_ratio": nr.max_gap_ratio,
                    "muscles": [
                        {
                            "name": mg.muscle_name,
                            "rest_gap": mg.rest_gap,
                            "posed_gap": mg.posed_gap,
                            "gap_ratio": mg.gap_ratio,
                        }
                        for mg in nr.muscle_gaps
                    ],
                }
                for nr in self.neck_gap_results
            ],
            "bone_proximity_results": [
                {
                    "pose": bp.pose_name,
                    "max_distance": bp.max_distance,
                    "muscles": [
                        {"name": mr.muscle_name, "distance": mr.distance}
                        for mr in bp.muscle_results
                    ],
                }
                for bp in self.bone_proximity_results
            ],
            "stretch_ratio_results": [
                {
                    "pose": sr.pose_name,
                    "min_stretch_ratio": sr.min_stretch_ratio,
                    "muscles": [
                        {
                            "name": mr.muscle_name,
                            "stretch_ratio": mr.stretch_ratio,
                            "centroid_displacement": mr.centroid_displacement,
                        }
                        for mr in sr.muscle_results
                    ],
                }
                for sr in self.stretch_ratio_results
            ],
            "pin_effectiveness_results": [
                {
                    "pose": pe.pose_name,
                    "max_pin_error": pe.max_pin_error,
                    "muscles": [
                        {"name": mr.muscle_name, "pin_error": mr.pin_error}
                        for mr in pe.muscle_results
                    ],
                }
                for pe in self.pin_effectiveness_results
            ],
            "platysma_attachment_results": [
                {
                    "pose": pa.pose_name,
                    "max_distance": pa.max_distance,
                    "muscles": [
                        {"name": mr.muscle_name, "lower_distance": mr.lower_distance}
                        for mr in pa.muscle_results
                    ],
                }
                for pa in self.platysma_attachment_results
            ],
        }


# ── Centroid extraction ──────────────────────────────────────────────

def _get_group_centroid(node: SceneNode) -> NDArray[np.float64] | None:
    """Compute centroid of all mesh vertices under a group node."""
    all_positions = []

    def _collect(n: SceneNode):
        if n.mesh is not None:
            pos = n.mesh.geometry.positions
            if pos is not None and len(pos) > 0:
                all_positions.append(np.asarray(pos, dtype=np.float64).reshape(-1, 3))

    node.traverse(_collect)

    if not all_positions:
        return None
    combined = np.concatenate(all_positions, axis=0)
    return combined.mean(axis=0)


def _get_neck_muscle_lower_positions(neck_muscle_system) -> dict[str, NDArray[np.float64]]:
    """Get the lowest 15% of vertices per neck muscle (body-attachment end)."""
    result = {}
    for md in neck_muscle_system.muscle_data:
        pos = md.mesh.geometry.positions
        if pos is None or len(pos) == 0:
            continue
        pos3 = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
        # Y values — lower end has lower Y (or higher, depends on convention)
        # Use spine fracs: lower frac = body end
        fracs = md.spine_fracs
        threshold = np.percentile(fracs, 15)
        mask = fracs <= threshold
        if mask.any():
            lower_verts = pos3[mask]
            result[md.defn.get("name", "unknown")] = lower_verts.mean(axis=0)
    return result


def _get_body_upper_positions(hs) -> NDArray[np.float64] | None:
    """Get approximate upper body anchor positions (thoracic/shoulder area)."""
    skeleton = hs.skeleton
    if skeleton is None:
        return None

    pivots = skeleton.pivots.get("thoracic", [])
    if not pivots:
        return None

    # Use the highest thoracic pivot (closest to head)
    positions = []
    for pinfo in pivots:
        group: SceneNode = pinfo["group"]
        wp = group.get_world_position()
        positions.append(wp)

    if not positions:
        return None
    return np.array(positions).mean(axis=0)


# ── Diagnostic checks ────────────────────────────────────────────────

def check_group_follow(hs, pose_name: str, face_state: FaceState) -> PoseFollowResult:
    """Check that all head-attached groups follow the head rotation.

    For each group, computes:
    - Expected centroid: rest centroid rotated by the head quaternion around pivot
    - Actual centroid: current mesh centroid after applying rotation
    - Error: Euclidean distance between expected and actual
    """
    from tools.headless_loader import apply_head_rotation

    # Reset to rest pose first so rest centroids are captured cleanly.
    # This prevents stale Platysma corrections from a prior call from
    # contaminating the "rest" reference.
    rest_state = FaceState()
    apply_head_rotation(hs, rest_state)

    # Record rest centroids
    group_names = ["skullGroup", "faceGroup", "stlMuscleGroup",
                   "exprMuscleGroup", "faceFeatureGroup"]
    rest_centroids: dict[str, NDArray[np.float64]] = {}

    for gname in group_names:
        node = hs.named_nodes.get(gname)
        if node is not None:
            c = _get_group_centroid(node)
            if c is not None:
                rest_centroids[gname] = c

    # Apply head rotation
    head_q = apply_head_rotation(hs, face_state)

    # Compute expected vs actual
    head_pivot = hs.pipeline.head_rotation._head_pivot if hs.pipeline.head_rotation else np.zeros(3)
    group_results = []

    for gname, rest_c in rest_centroids.items():
        node = hs.named_nodes.get(gname)
        if node is None:
            continue

        # Expected: rotate rest centroid around head pivot
        rel = rest_c - head_pivot
        rotated = quat_rotate_vec3(head_q, rel)
        expected = rotated + head_pivot

        # Actual: world-space centroid after rotation
        # Since groups have transforms applied, we need to compute
        # the actual position considering the group transform
        actual = _get_group_centroid(node)
        if actual is None:
            continue

        # The group itself has a transform set by _apply_pivot_rotation,
        # so we need the world-space centroid:
        # The mesh positions are in local space, transform by group world matrix
        wm = node.world_matrix
        actual_world = (wm[:3, :3] @ actual) + wm[:3, 3]

        error = float(np.linalg.norm(expected - actual_world))
        group_results.append(GroupFollowResult(
            group_name=gname,
            expected_centroid=expected,
            actual_centroid=actual_world,
            error=error,
        ))

    max_error = max((gr.error for gr in group_results), default=0.0)

    # Reset
    if hs.pipeline.head_rotation is not None:
        hs.pipeline.head_rotation.reset()
    if hs.pipeline.neck_muscles is not None:
        hs.pipeline.neck_muscles.reset()
    # Reset group transforms
    for gname in group_names:
        node = hs.named_nodes.get(gname)
        if node is not None:
            node.set_quaternion(quat_identity())
            node.set_position(0, 0, 0)
            node.mark_dirty()
    hs.scene.update()

    return PoseFollowResult(
        pose_name=pose_name,
        group_results=group_results,
        max_error=max_error,
    )


def check_neck_body_gap(
    hs,
    pose_name: str,
    body_state: BodyState,
    face_state: FaceState,
) -> NeckGapPoseResult:
    """Check neck muscle gap to body skeleton in a combined pose.

    Measures the distance from the lower end of each neck muscle to the
    nearest upper body anchor, comparing rest vs posed values.
    """
    from tools.headless_loader import apply_full_pose, reset_skinning

    neck_sys = hs.pipeline.neck_muscles
    if neck_sys is None:
        return NeckGapPoseResult(pose_name=pose_name, muscle_gaps=[], max_gap_ratio=0.0)

    # Rest-pose lower positions
    rest_lower = _get_neck_muscle_lower_positions(neck_sys)
    rest_body_anchor = _get_body_upper_positions(hs)

    if rest_body_anchor is None:
        return NeckGapPoseResult(pose_name=pose_name, muscle_gaps=[], max_gap_ratio=0.0)

    rest_gaps: dict[str, float] = {}
    for name, pos in rest_lower.items():
        rest_gaps[name] = float(np.linalg.norm(pos - rest_body_anchor))

    # Apply combined pose
    apply_full_pose(hs, body_state, face_state)

    # Posed lower positions
    posed_lower = _get_neck_muscle_lower_positions(neck_sys)
    posed_body_anchor = _get_body_upper_positions(hs)

    if posed_body_anchor is None:
        posed_body_anchor = rest_body_anchor

    muscle_gaps = []
    for name in rest_lower:
        if name not in posed_lower:
            continue
        rest_g = rest_gaps.get(name, 1.0)
        posed_g = float(np.linalg.norm(posed_lower[name] - posed_body_anchor))
        ratio = posed_g / max(rest_g, 0.01)
        muscle_gaps.append(NeckGapResult(
            muscle_name=name,
            rest_gap=rest_g,
            posed_gap=posed_g,
            gap_ratio=ratio,
        ))

    max_ratio = max((mg.gap_ratio for mg in muscle_gaps), default=0.0)

    # Reset
    reset_skinning(hs)

    return NeckGapPoseResult(
        pose_name=pose_name,
        muscle_gaps=muscle_gaps,
        max_gap_ratio=max_ratio,
    )


# ── Bone proximity check ─────────────────────────────────────────────

def check_bone_proximity(
    hs,
    pose_name: str,
    body_state: BodyState,
    face_state: FaceState,
) -> BoneProximityPoseResult:
    """Check distance from each muscle's lower-end vertices to its configured bones.

    For each muscle with ``lowerBones`` configured, measures the Euclidean
    distance from the lower 15% vertices centroid to the bone anchor
    position.  Should be < 5 units in any pose.
    """
    from tools.headless_loader import apply_full_pose, reset_skinning

    neck_sys = hs.pipeline.neck_muscles
    bone_reg = getattr(hs.pipeline, 'bone_anchors', None)
    if neck_sys is None or bone_reg is None:
        return BoneProximityPoseResult(pose_name=pose_name, muscle_results=[], max_distance=0.0)

    # Apply pose
    apply_full_pose(hs, body_state, face_state)

    muscle_results = []
    for md in neck_sys.muscle_data:
        lower_bones = md.defn.get("lowerBones", [])
        if not lower_bones:
            continue

        name = md.defn.get("name", "unknown")
        bone_pos = bone_reg.get_muscle_anchor_current(name, lower_bones)
        if bone_pos is None:
            continue

        # Lower-end vertices (lowest 15% by spine frac)
        pos3 = np.asarray(md.mesh.geometry.positions, dtype=np.float64).reshape(-1, 3)
        fracs = md.spine_fracs
        threshold = np.percentile(fracs, 15)
        mask = fracs <= threshold
        if not mask.any():
            continue

        lower_centroid = pos3[mask].mean(axis=0)
        dist = float(np.linalg.norm(lower_centroid - bone_pos))
        muscle_results.append(BoneProximityResult(
            muscle_name=name, bone_names=lower_bones, distance=dist,
        ))

    max_dist = max((mr.distance for mr in muscle_results), default=0.0)

    reset_skinning(hs)
    return BoneProximityPoseResult(
        pose_name=pose_name, muscle_results=muscle_results, max_distance=max_dist,
    )


# ── Muscle stretch ratio check ──────────────────────────────────────

def check_muscle_stretch_ratio(
    hs,
    pose_name: str,
    body_state: BodyState,
    face_state: FaceState,
) -> StretchRatioPoseResult:
    """Check that muscles actually stretch during head rotation.

    Measures per-muscle length change (upper centroid - lower centroid) vs
    rest length.  Muscles should stretch (ratio > 1.0 when head turns away)
    rather than translating rigidly (ratio ~1.0 with large centroid displacement).
    """
    from tools.headless_loader import apply_full_pose, reset_skinning

    neck_sys = hs.pipeline.neck_muscles
    if neck_sys is None:
        return StretchRatioPoseResult(
            pose_name=pose_name, muscle_results=[], min_stretch_ratio=1.0,
            max_centroid_displacement=0.0,
        )

    # Record rest-pose measurements
    rest_data = {}
    for md in neck_sys.muscle_data:
        name = md.defn.get("name", "unknown")
        fracs = md.spine_fracs.astype(np.float64)
        frac_min, frac_max = float(fracs.min()), float(fracs.max())
        frac_range = frac_max - frac_min
        if frac_range < 0.05 or md.vert_count < 4:
            continue

        pos3 = md.rest_positions.reshape(-1, 3).astype(np.float64)
        upper_mask = fracs >= frac_min + frac_range * 0.85
        lower_mask = fracs <= frac_min + frac_range * 0.15

        if upper_mask.sum() < 2 or lower_mask.sum() < 2:
            continue

        upper_c = pos3[upper_mask].mean(axis=0)
        lower_c = pos3[lower_mask].mean(axis=0)
        rest_length = float(np.linalg.norm(upper_c - lower_c))
        rest_centroid = pos3.mean(axis=0)

        if rest_length < 0.1:
            continue

        rest_data[name] = {
            "rest_length": rest_length,
            "rest_centroid": rest_centroid,
            "fracs": fracs,
            "frac_min": frac_min,
            "frac_range": frac_range,
        }

    # Apply pose
    apply_full_pose(hs, body_state, face_state)

    muscle_results = []
    for md in neck_sys.muscle_data:
        name = md.defn.get("name", "unknown")
        if name not in rest_data:
            continue

        rd = rest_data[name]
        pos3 = np.asarray(md.mesh.geometry.positions, dtype=np.float64).reshape(-1, 3)

        upper_mask = rd["fracs"] >= rd["frac_min"] + rd["frac_range"] * 0.85
        lower_mask = rd["fracs"] <= rd["frac_min"] + rd["frac_range"] * 0.15

        cur_upper = pos3[upper_mask].mean(axis=0)
        cur_lower = pos3[lower_mask].mean(axis=0)
        cur_length = float(np.linalg.norm(cur_upper - cur_lower))
        cur_centroid = pos3.mean(axis=0)

        stretch_ratio = cur_length / rd["rest_length"]
        centroid_disp = float(np.linalg.norm(cur_centroid - rd["rest_centroid"]))

        muscle_results.append(StretchRatioResult(
            muscle_name=name,
            stretch_ratio=stretch_ratio,
            centroid_displacement=centroid_disp,
        ))

    min_stretch = min((mr.stretch_ratio for mr in muscle_results), default=1.0)
    max_disp = max((mr.centroid_displacement for mr in muscle_results), default=0.0)

    reset_skinning(hs)
    return StretchRatioPoseResult(
        pose_name=pose_name,
        muscle_results=muscle_results,
        min_stretch_ratio=min_stretch,
        max_centroid_displacement=max_disp,
    )


# ── Pin effectiveness check ──────────────────────────────────────────

def check_pin_effectiveness(
    hs,
    pose_name: str,
    body_state: BodyState,
    face_state: FaceState,
) -> PinEffectivenessPoseResult:
    """Check how close pinned lower-end vertices are to their bone targets.

    Pin error is the distance from the lowest 10% of vertices (by spine
    frac) to the bone target position (rest + bone_delta).  Should be < 2.
    """
    from tools.headless_loader import apply_full_pose, reset_skinning

    neck_sys = hs.pipeline.neck_muscles
    bone_reg = getattr(hs.pipeline, 'bone_anchors', None)
    if neck_sys is None or bone_reg is None:
        return PinEffectivenessPoseResult(
            pose_name=pose_name, muscle_results=[], max_pin_error=0.0,
        )

    apply_full_pose(hs, body_state, face_state)

    muscle_results = []
    for md in neck_sys.muscle_data:
        lower_bones = md.defn.get("lowerBones", [])
        if not lower_bones:
            continue

        name = md.defn.get("name", "unknown")
        bone_cur = bone_reg.get_muscle_anchor_current(name, lower_bones)
        bone_rest = bone_reg.get_muscle_anchor(name, lower_bones)
        if bone_cur is None or bone_rest is None:
            continue

        bone_delta = bone_cur - bone_rest

        # Lowest 10% of vertices
        pos3 = np.asarray(md.mesh.geometry.positions, dtype=np.float64).reshape(-1, 3)
        fracs = md.spine_fracs
        threshold = np.percentile(fracs, 10)
        mask = fracs <= threshold
        if not mask.any():
            continue

        lower_verts = pos3[mask]

        # Target: rest positions of those verts + bone_delta
        rest3 = md.rest_positions.reshape(-1, 3).astype(np.float64)
        target_verts = rest3[mask] + bone_delta

        # Pin error: average distance from actual to target
        errors = np.linalg.norm(lower_verts - target_verts, axis=1)
        pin_error = float(errors.mean())

        muscle_results.append(PinEffectivenessResult(
            muscle_name=name, pin_error=pin_error,
        ))

    max_err = max((mr.pin_error for mr in muscle_results), default=0.0)

    reset_skinning(hs)
    return PinEffectivenessPoseResult(
        pose_name=pose_name, muscle_results=muscle_results, max_pin_error=max_err,
    )


# ── Platysma attachment check ────────────────────────────────────────

def check_platysma_attachment(
    hs,
    pose_name: str,
    body_state: BodyState,
    face_state: FaceState,
) -> PlatysmaAttachmentPoseResult:
    """Verify Platysma lower vertices stay near clavicle during head rotation.

    Finds Platysma R/L in expression muscles, measures the distance from
    their body-end vertices (lowest 20% by Y) to the clavicle bone position.
    Should be < 5 units.
    """
    from tools.headless_loader import apply_full_pose, reset_skinning

    expr_sys = hs.pipeline.expression_muscles
    bone_reg = getattr(hs.pipeline, 'bone_anchors', None)
    if expr_sys is None:
        return PlatysmaAttachmentPoseResult(
            pose_name=pose_name, muscle_results=[], max_distance=0.0,
        )

    apply_full_pose(hs, body_state, face_state)

    muscle_results = []
    for md in expr_sys.muscle_data:
        name = md.defn.get("name", "")
        if "Platysma" not in name:
            continue

        pos3 = np.asarray(md.mesh.geometry.positions, dtype=np.float64).reshape(-1, 3)
        if len(pos3) < 4:
            continue

        # Lower 20% by Y coordinate (body end has lower Y)
        y_vals = pos3[:, 1]
        y_thresh = np.percentile(y_vals, 20)
        mask = y_vals <= y_thresh
        if not mask.any():
            continue

        lower_centroid = pos3[mask].mean(axis=0)

        # Clavicle position (if bone registry available)
        if bone_reg is not None:
            side = "Right" if "R" in name else "Left"
            bone_pos = bone_reg.get_muscle_anchor_current(
                name, [f"{side} Clavicle"],
            )
            if bone_pos is not None:
                dist = float(np.linalg.norm(lower_centroid - bone_pos))
                muscle_results.append(PlatysmaAttachmentResult(
                    muscle_name=name, lower_distance=dist,
                ))
                continue

        # Fallback: measure distance from rest lower centroid
        rest3 = md.rest_positions.reshape(-1, 3).astype(np.float64)
        rest_lower = rest3[mask].mean(axis=0)
        dist = float(np.linalg.norm(lower_centroid - rest_lower))
        muscle_results.append(PlatysmaAttachmentResult(
            muscle_name=name, lower_distance=dist,
        ))

    max_dist = max((mr.lower_distance for mr in muscle_results), default=0.0)

    reset_skinning(hs)
    return PlatysmaAttachmentPoseResult(
        pose_name=pose_name, muscle_results=muscle_results, max_distance=max_dist,
    )


# ── Full diagnostic ──────────────────────────────────────────────────

def run_full_diagnostic(hs, poses: dict[str, dict] | None = None) -> DiagnosticResults:
    """Run all head rotation diagnostics.

    Parameters
    ----------
    hs : HeadlessScene
    poses : optional dict of head poses to test (defaults to HEAD_POSES)

    Returns
    -------
    DiagnosticResults
    """
    if poses is None:
        poses = HEAD_POSES

    results = DiagnosticResults()

    # 1. Group follow checks for each head pose
    for pose_name, pose_values in poses.items():
        face_state = FaceState()
        for key, val in pose_values.items():
            setattr(face_state, key, val)

        follow_result = check_group_follow(hs, pose_name, face_state)
        results.follow_results.append(follow_result)
        logger.info("Pose %s: max follow error = %.4f", pose_name, follow_result.max_error)

    # 2. Neck-body gap checks for combined poses
    for pose_name, pose_cfg in BODY_HEAD_POSES.items():
        face_state = FaceState()
        body_state = BodyState()
        for key, val in pose_cfg.get("face", {}).items():
            setattr(face_state, key, val)
        for key, val in pose_cfg.get("body", {}).items():
            setattr(body_state, key, val)

        gap_result = check_neck_body_gap(hs, pose_name, body_state, face_state)
        results.neck_gap_results.append(gap_result)
        logger.info("Pose %s: max gap ratio = %.4f", pose_name, gap_result.max_gap_ratio)

    # 3. Bone proximity checks for combined poses
    for pose_name, pose_cfg in BODY_HEAD_POSES.items():
        face_state = FaceState()
        body_state = BodyState()
        for key, val in pose_cfg.get("face", {}).items():
            setattr(face_state, key, val)
        for key, val in pose_cfg.get("body", {}).items():
            setattr(body_state, key, val)

        bp_result = check_bone_proximity(hs, pose_name, body_state, face_state)
        results.bone_proximity_results.append(bp_result)
        logger.info("Pose %s: max bone proximity = %.4f", pose_name, bp_result.max_distance)

    # 4. Muscle stretch ratio checks for combined poses
    for pose_name, pose_cfg in BODY_HEAD_POSES.items():
        face_state = FaceState()
        body_state = BodyState()
        for key, val in pose_cfg.get("face", {}).items():
            setattr(face_state, key, val)
        for key, val in pose_cfg.get("body", {}).items():
            setattr(body_state, key, val)

        sr_result = check_muscle_stretch_ratio(hs, pose_name, body_state, face_state)
        results.stretch_ratio_results.append(sr_result)
        logger.info("Pose %s: min stretch ratio = %.4f", pose_name, sr_result.min_stretch_ratio)

    # 5. Pin effectiveness checks for combined poses
    for pose_name, pose_cfg in BODY_HEAD_POSES.items():
        face_state = FaceState()
        body_state = BodyState()
        for key, val in pose_cfg.get("face", {}).items():
            setattr(face_state, key, val)
        for key, val in pose_cfg.get("body", {}).items():
            setattr(body_state, key, val)

        pe_result = check_pin_effectiveness(hs, pose_name, body_state, face_state)
        results.pin_effectiveness_results.append(pe_result)
        logger.info("Pose %s: max pin error = %.4f", pose_name, pe_result.max_pin_error)

    # 6. Platysma attachment checks for combined poses
    for pose_name, pose_cfg in BODY_HEAD_POSES.items():
        face_state = FaceState()
        body_state = BodyState()
        for key, val in pose_cfg.get("face", {}).items():
            setattr(face_state, key, val)
        for key, val in pose_cfg.get("body", {}).items():
            setattr(body_state, key, val)

        pa_result = check_platysma_attachment(hs, pose_name, body_state, face_state)
        results.platysma_attachment_results.append(pa_result)
        logger.info("Pose %s: max platysma dist = %.4f", pose_name, pa_result.max_distance)

    return results


# ── Report formatting ────────────────────────────────────────────────

def format_diagnostic_report(results: DiagnosticResults) -> str:
    """Format diagnostic results as a human-readable report."""
    lines = ["=" * 60, "HEAD ROTATION DIAGNOSTIC REPORT", "=" * 60, ""]

    # Follow errors
    lines.append("GROUP FOLLOW ERRORS (< 0.1 = good)")
    lines.append("-" * 50)
    for pr in results.follow_results:
        status = "PASS" if pr.max_error < 0.1 else "FAIL"
        lines.append(f"  {pr.pose_name:20s}  max_error={pr.max_error:.4f}  [{status}]")
        for gr in pr.group_results:
            flag = " *" if gr.error > 0.1 else ""
            lines.append(f"    {gr.group_name:25s}  error={gr.error:.4f}{flag}")
    lines.append("")

    # Neck gaps
    lines.append("NECK-BODY GAP RATIOS (< 2.0 = good)")
    lines.append("-" * 50)
    for nr in results.neck_gap_results:
        status = "PASS" if nr.max_gap_ratio < 2.0 else "FAIL"
        lines.append(f"  {nr.pose_name:25s}  max_ratio={nr.max_gap_ratio:.4f}  [{status}]")
        # Show worst 5 muscles
        sorted_gaps = sorted(nr.muscle_gaps, key=lambda x: x.gap_ratio, reverse=True)
        for mg in sorted_gaps[:5]:
            flag = " *" if mg.gap_ratio > 2.0 else ""
            lines.append(f"    {mg.muscle_name:25s}  rest={mg.rest_gap:.2f}  "
                         f"posed={mg.posed_gap:.2f}  ratio={mg.gap_ratio:.2f}{flag}")
    lines.append("")

    # Bone proximity
    if results.bone_proximity_results:
        lines.append("BONE PROXIMITY (< 5.0 = good)")
        lines.append("-" * 50)
        for bp in results.bone_proximity_results:
            status = "PASS" if bp.max_distance < 5.0 else "FAIL"
            lines.append(f"  {bp.pose_name:25s}  max_dist={bp.max_distance:.4f}  [{status}]")
            sorted_muscles = sorted(bp.muscle_results, key=lambda x: x.distance, reverse=True)
            for mr in sorted_muscles[:5]:
                flag = " *" if mr.distance > 5.0 else ""
                bones = ", ".join(mr.bone_names)
                lines.append(f"    {mr.muscle_name:25s}  dist={mr.distance:.2f}  bones=[{bones}]{flag}")
        lines.append("")

    # Stretch ratios
    if results.stretch_ratio_results:
        lines.append("MUSCLE STRETCH RATIOS (> 1.05 during rotation = good)")
        lines.append("-" * 50)
        for sr in results.stretch_ratio_results:
            # For non-neutral poses, muscles should stretch
            lines.append(f"  {sr.pose_name:25s}  min_stretch={sr.min_stretch_ratio:.4f}  "
                         f"max_disp={sr.max_centroid_displacement:.4f}")
            sorted_muscles = sorted(sr.muscle_results, key=lambda x: x.stretch_ratio)
            for mr in sorted_muscles[:5]:
                lines.append(f"    {mr.muscle_name:25s}  stretch={mr.stretch_ratio:.4f}  "
                             f"disp={mr.centroid_displacement:.2f}")
        lines.append("")

    # Pin effectiveness
    if results.pin_effectiveness_results:
        lines.append("PIN EFFECTIVENESS (< 2.0 = good)")
        lines.append("-" * 50)
        for pe in results.pin_effectiveness_results:
            status = "PASS" if pe.max_pin_error < 2.0 else "FAIL"
            lines.append(f"  {pe.pose_name:25s}  max_error={pe.max_pin_error:.4f}  [{status}]")
            sorted_muscles = sorted(pe.muscle_results, key=lambda x: x.pin_error, reverse=True)
            for mr in sorted_muscles[:5]:
                flag = " *" if mr.pin_error > 2.0 else ""
                lines.append(f"    {mr.muscle_name:25s}  pin_err={mr.pin_error:.4f}{flag}")
        lines.append("")

    # Platysma attachment
    if results.platysma_attachment_results:
        lines.append("PLATYSMA ATTACHMENT (< 5.0 = good)")
        lines.append("-" * 50)
        for pa in results.platysma_attachment_results:
            status = "PASS" if pa.max_distance < 5.0 else "FAIL"
            lines.append(f"  {pa.pose_name:25s}  max_dist={pa.max_distance:.4f}  [{status}]")
            for mr in pa.muscle_results:
                flag = " *" if mr.lower_distance > 5.0 else ""
                lines.append(f"    {mr.muscle_name:25s}  dist={mr.lower_distance:.4f}{flag}")
        lines.append("")

    # Summary
    all_follow_pass = all(pr.max_error < 0.1 for pr in results.follow_results)
    all_gap_pass = all(nr.max_gap_ratio < 2.0 for nr in results.neck_gap_results)
    all_bone_pass = all(bp.max_distance < 5.0 for bp in results.bone_proximity_results)
    all_pin_pass = all(pe.max_pin_error < 2.0 for pe in results.pin_effectiveness_results)
    all_platysma_pass = all(pa.max_distance < 5.0 for pa in results.platysma_attachment_results)

    lines.append("SUMMARY")
    lines.append("-" * 50)
    lines.append(f"  Group follow:       {'ALL PASS' if all_follow_pass else 'FAILURES DETECTED'}")
    lines.append(f"  Neck gaps:          {'ALL PASS' if all_gap_pass else 'FAILURES DETECTED'}")
    if results.bone_proximity_results:
        lines.append(f"  Bone proximity:     {'ALL PASS' if all_bone_pass else 'FAILURES DETECTED'}")
    if results.pin_effectiveness_results:
        lines.append(f"  Pin effectiveness:  {'ALL PASS' if all_pin_pass else 'FAILURES DETECTED'}")
    if results.platysma_attachment_results:
        lines.append(f"  Platysma attach:    {'ALL PASS' if all_platysma_pass else 'FAILURES DETECTED'}")
    lines.append("")

    return "\n".join(lines)
