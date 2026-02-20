"""Head yaw/pitch/roll rotation with cervical vertebra distribution.

Applies head rotation to the skull and face scene-graph groups by composing
a quaternion from ``headYaw``, ``headPitch``, and ``headRoll`` face-state
values.  The rotation is distributed across cervical vertebrae pivot groups
using the fraction table from ``vertebra_fractions.json``.

Includes soft-clamping via the neck constraint solver when total muscle
tension excess exceeds a threshold.

This module has ZERO GL imports; all math is done with NumPy.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from faceforge.core.math_utils import (
    Vec3, Quat, vec3,
    quat_identity, quat_from_euler, quat_slerp, quat_normalize,
    quat_rotate_vec3, deg_to_rad,
)
from faceforge.core.scene_graph import SceneNode
from faceforge.core.state import FaceState, ConstraintState
from faceforge.constants import (
    JAW_PIVOT,
    HEAD_YAW_MAX,
    HEAD_PITCH_MAX,
    HEAD_ROLL_MAX,
    get_jaw_pivot,
)


# Soft-clamp threshold on total constraint excess
SOFT_CLAMP_THRESHOLD = 0.1


class HeadRotationSystem:
    """Applies head yaw/pitch/roll rotation with vertebra distribution.

    The rotation is composed in YXZ order (yaw first, then pitch, then roll)
    and applied to the skull and face groups around the head pivot point.

    Cervical vertebrae receive a fractional copy of the head rotation so the
    neck appears to articulate smoothly rather than bending at a single joint.
    """

    def __init__(
        self,
        jaw_pivot: tuple[float, float, float] | None = None,
    ) -> None:
        self._head_pivot = vec3(*(jaw_pivot or get_jaw_pivot()))
        self._last_yaw: float = 0.0
        self._last_pitch: float = 0.0
        self._last_roll: float = 0.0
        self._head_quat: Quat = quat_identity()

    def set_head_pivot(self, x: float, y: float, z: float) -> None:
        """Update the head pivot position (e.g. after BP3D skull load)."""
        self._head_pivot = vec3(x, y, z)

    @property
    def head_quaternion(self) -> Quat:
        """The most recently computed head rotation quaternion."""
        return self._head_quat.copy()

    def apply(
        self,
        face_state: FaceState,
        skull_group: SceneNode,
        face_group: SceneNode,
        vertebrae_pivots: Optional[list[dict]] = None,
        constraint_state: Optional[ConstraintState] = None,
        brain_group: Optional[SceneNode] = None,
        stl_muscle_group: Optional[SceneNode] = None,
        expr_muscle_group: Optional[SceneNode] = None,
        face_feature_group: Optional[SceneNode] = None,
    ) -> Quat:
        """Compute and apply head rotation for this frame.

        Parameters
        ----------
        face_state:
            Current face state with ``head_yaw``, ``head_pitch``, ``head_roll``
            in [-1, 1].
        skull_group:
            The skullGroup SceneNode to rotate.
        face_group:
            The faceGroup SceneNode to rotate (follows skull).
        vertebrae_pivots:
            Optional list of dicts ``{group: SceneNode, level: int,
            fractions: {yaw, pitch, roll}}``.  If provided, rotation is
            distributed to each pivot proportionally.
        constraint_state:
            If provided and ``total_excess > SOFT_CLAMP_THRESHOLD``,
            the rotation is scaled back to reduce overstretching.
        stl_muscle_group:
            Optional jaw muscle group (stlMuscleGroup) — sibling of skull,
            must receive same pivot rotation.
        expr_muscle_group:
            Optional expression muscle group (exprMuscleGroup) — same.
        face_feature_group:
            Optional face feature group (faceFeatureGroup) — same.

        Returns
        -------
        Quat
            The head rotation quaternion (for use by neck muscles etc.).
        """
        # Map normalised state values to radians
        yaw_rad = face_state.head_yaw * deg_to_rad(HEAD_YAW_MAX)
        pitch_rad = face_state.head_pitch * deg_to_rad(HEAD_PITCH_MAX)
        roll_rad = face_state.head_roll * deg_to_rad(HEAD_ROLL_MAX)

        # Soft-clamping from constraint solver (uses smoothed excess to
        # prevent frame-to-frame oscillation / jitter)
        if constraint_state is not None and constraint_state.smoothed_total_excess > SOFT_CLAMP_THRESHOLD:
            scale = 1.0 / (1.0 + constraint_state.smoothed_total_excess)
            yaw_rad *= scale
            pitch_rad *= scale
            roll_rad *= scale

        # Compose quaternion in YXZ order (Yaw, Pitch, Roll)
        head_q = quat_from_euler(pitch_rad, yaw_rad, roll_rad, "YXZ")
        head_q = quat_normalize(head_q)
        self._head_quat = head_q

        # Apply full rotation to skull, face, brain, and head-attached groups
        self._apply_pivot_rotation(skull_group, head_q, self._head_pivot)
        self._apply_pivot_rotation(face_group, head_q, self._head_pivot)
        if brain_group is not None:
            self._apply_pivot_rotation(brain_group, head_q, self._head_pivot)
        if stl_muscle_group is not None:
            self._apply_pivot_rotation(stl_muscle_group, head_q, self._head_pivot)
        if expr_muscle_group is not None:
            self._apply_pivot_rotation(expr_muscle_group, head_q, self._head_pivot)
        if face_feature_group is not None:
            self._apply_pivot_rotation(face_feature_group, head_q, self._head_pivot)

        # Distribute rotation to vertebrae
        if vertebrae_pivots is not None:
            self._distribute_to_vertebrae(vertebrae_pivots, yaw_rad, pitch_rad, roll_rad)

        self._last_yaw = face_state.head_yaw
        self._last_pitch = face_state.head_pitch
        self._last_roll = face_state.head_roll

        return head_q

    def _apply_pivot_rotation(self, group: SceneNode, q: Quat, pivot: Vec3 | None = None) -> None:
        """Set a group's quaternion to rotate around the head pivot.

        Instead of rotating around the group's own origin, we offset the
        position so the rotation appears to be centered on the pivot point.
        """
        if pivot is None:
            pivot = self._head_pivot
        group.set_quaternion(q)
        rotated_pivot = quat_rotate_vec3(q, pivot)
        offset = pivot - rotated_pivot
        group.set_position(float(offset[0]), float(offset[1]), float(offset[2]))
        group.mark_dirty()

    def _distribute_to_vertebrae(
        self,
        vertebrae_pivots: list[dict],
        yaw_rad: float,
        pitch_rad: float,
        roll_rad: float,
    ) -> None:
        """Apply fractional rotation to each vertebra pivot.

        Each vertebra level receives a fraction of the total head rotation.
        C1 (level 0) receives 100%, T1 (level 7) receives 0%.
        """
        for vp in vertebrae_pivots:
            group: SceneNode = vp["group"]
            fracs = vp["fractions"]

            frac_yaw = fracs.get("yaw", 0.0)
            frac_pitch = fracs.get("pitch", 0.0)
            frac_roll = fracs.get("roll", 0.0)

            vert_yaw = yaw_rad * frac_yaw
            vert_pitch = pitch_rad * frac_pitch
            vert_roll = roll_rad * frac_roll

            q = quat_from_euler(vert_pitch, vert_yaw, vert_roll, "YXZ")
            q = quat_normalize(q)
            group.set_quaternion(q)
            group.mark_dirty()

    # ------------------------------------------------------------------
    # Skin mesh head-follow
    # ------------------------------------------------------------------

    # Z thresholds for skin head-follow blending (in BP3D world coords).
    # Z > HEAD_Z_FULL: 100% head follow (mid-skull and above)
    # Z between NECK_Z_START and HEAD_Z_FULL: smooth blend
    # Z < NECK_Z_START: 0% head follow (body)
    HEAD_Z_FULL = 5.0      # mid-head level
    NECK_Z_START = -25.0   # below neck / shoulder level

    def apply_to_skin(
        self,
        mesh_positions: np.ndarray,
        rest_positions: np.ndarray,
        head_q: Quat | None = None,
    ) -> None:
        """Apply head rotation to skin mesh vertices with Z-based blending.

        Computes head-rotated REST positions and blends them with the
        body-skinned positions:

            result = (1 - w) * body_skinned + w * head_rotated_rest

        where ``w`` is a smoothstep weight based on the vertex's rest Z
        coordinate.  Head vertices (w=1) get pure head-rotated rest pose;
        body vertices (w=0) keep their body-skinned positions; neck
        vertices blend smoothly.

        This should be called AFTER soft-tissue skinning so that body
        deformation is already in ``mesh_positions``.

        Parameters
        ----------
        mesh_positions : (V*3,) or (V, 3) float32 array
            Current vertex positions (modified in place).  On entry these
            are the body-skinned positions.
        rest_positions : (V*3,) or (V, 3) float32 array
            Rest-pose vertex positions.
        head_q : Quat, optional
            Head rotation quaternion.  If None, uses the most recently
            computed quaternion from :meth:`apply`.
        """
        if head_q is None:
            head_q = self._head_quat

        identity_q = quat_identity()
        if np.allclose(head_q, identity_q, atol=1e-6):
            return

        rest_pos = np.asarray(rest_positions, dtype=np.float64).reshape(-1, 3)
        body_pos = np.asarray(mesh_positions, dtype=np.float64).reshape(-1, 3)

        # Compute per-vertex head-follow weight from rest Z coordinate
        z_vals = rest_pos[:, 2]
        z_range = self.HEAD_Z_FULL - self.NECK_Z_START
        weights = np.clip((z_vals - self.NECK_Z_START) / z_range, 0.0, 1.0)

        # Smoothstep for a nicer transition
        weights = weights * weights * (3.0 - 2.0 * weights)

        # Only process vertices with non-zero weight
        active = weights > 1e-4
        if not active.any():
            return

        # Build full rotation matrix once (Rodrigues for the full head quat)
        qx, qy, qz, qw = head_q
        sin_half = np.sqrt(qx * qx + qy * qy + qz * qz)
        if sin_half < 1e-8:
            return
        full_angle = 2.0 * np.arctan2(sin_half, qw)
        if abs(full_angle) < 1e-8:
            return

        axis = np.array([qx, qy, qz], dtype=np.float64) / sin_half
        kx, ky, kz = axis
        K = np.array([
            [0.0, -kz, ky],
            [kz, 0.0, -kx],
            [-ky, kx, 0.0],
        ], dtype=np.float64)
        K2 = K @ K
        # Full rotation matrix R = I + sin(a)*K + (1-cos(a))*K²
        sin_f = np.sin(full_angle)
        cos_f = np.cos(full_angle)
        R = np.eye(3, dtype=np.float64) + sin_f * K + (1.0 - cos_f) * K2

        pivot = self._head_pivot.astype(np.float64)

        # Compute head-rotated REST positions for active vertices
        idx = np.where(active)[0]
        rest_active = rest_pos[idx]            # (M, 3)
        body_active = body_pos[idx]            # (M, 3) — body-skinned
        w = weights[idx][:, None]              # (M, 1)

        # Rotate rest positions around head pivot
        rel = rest_active - pivot
        head_rotated = (R @ rel.T).T + pivot   # (M, 3)

        # Blend: result = (1 - w) * body_skinned + w * head_rotated_rest
        blended = (1.0 - w) * body_active + w * head_rotated

        # Write back
        body_pos[idx] = blended

        flat = mesh_positions.reshape(-1)
        flat[:] = body_pos.astype(np.float32).ravel()

    def apply_to_skin_muscle(
        self,
        mesh_positions: np.ndarray,
        rest_positions: np.ndarray,
        upper_frac: float,
        lower_frac: float,
        head_q: Quat | None = None,
    ) -> None:
        """Apply head rotation to a muscle mesh with per-muscle fractions.

        Uses the mesh's Y-extent (not Z) to compute per-vertex head-follow
        weight.  Vertex at top (Y=max) gets ``upper_frac``, vertex at bottom
        (Y=min) gets ``lower_frac``, with linear interpolation between.

        Parameters
        ----------
        mesh_positions : flat float32 array
            Current vertex positions (modified in place).
        rest_positions : flat float32 array
            Rest-pose vertex positions.
        upper_frac, lower_frac : float
            Head-follow fractions for the top and bottom of the muscle.
        head_q : Quat, optional
            Head rotation quaternion.  If None, uses the most recently
            computed quaternion.
        """
        if head_q is None:
            head_q = self._head_quat

        identity_q = quat_identity()
        if np.allclose(head_q, identity_q, atol=1e-6):
            return

        rest_pos = np.asarray(rest_positions, dtype=np.float64).reshape(-1, 3)
        body_pos = np.asarray(mesh_positions, dtype=np.float64).reshape(-1, 3)

        # Compute per-vertex weight from Y-extent (top→upper_frac, bottom→lower_frac)
        y_vals = rest_pos[:, 1]
        y_min = y_vals.min()
        y_max = y_vals.max()
        y_range = y_max - y_min
        if y_range < 1e-6:
            t = np.full(len(y_vals), 0.5)
        else:
            t = (y_vals - y_min) / y_range  # 0 at bottom, 1 at top

        weights = lower_frac + t * (upper_frac - lower_frac)
        weights = np.clip(weights, 0.0, 1.0)

        # Only process vertices with non-zero weight
        active = weights > 1e-4
        if not active.any():
            return

        # Build rotation matrix from quaternion (Rodrigues)
        qx, qy, qz, qw = head_q
        sin_half = np.sqrt(qx * qx + qy * qy + qz * qz)
        if sin_half < 1e-8:
            return
        full_angle = 2.0 * np.arctan2(sin_half, qw)
        if abs(full_angle) < 1e-8:
            return

        axis = np.array([qx, qy, qz], dtype=np.float64) / sin_half
        kx, ky, kz = axis
        K = np.array([
            [0.0, -kz, ky],
            [kz, 0.0, -kx],
            [-ky, kx, 0.0],
        ], dtype=np.float64)
        K2 = K @ K
        sin_f = np.sin(full_angle)
        cos_f = np.cos(full_angle)
        R = np.eye(3, dtype=np.float64) + sin_f * K + (1.0 - cos_f) * K2

        pivot = self._head_pivot.astype(np.float64)

        idx = np.where(active)[0]
        rest_active = rest_pos[idx]
        body_active = body_pos[idx]
        w = weights[idx][:, None]

        # Rotate rest positions around head pivot
        rel = rest_active - pivot
        head_rotated = (R @ rel.T).T + pivot

        # Blend: result = (1 - w) * body_skinned + w * head_rotated_rest
        blended = (1.0 - w) * body_active + w * head_rotated

        body_pos[idx] = blended
        flat = mesh_positions.reshape(-1)
        flat[:] = body_pos.astype(np.float32).ravel()

    def reset(self) -> None:
        """Reset head rotation to identity."""
        self._head_quat = quat_identity()
        self._last_yaw = 0.0
        self._last_pitch = 0.0
        self._last_roll = 0.0
