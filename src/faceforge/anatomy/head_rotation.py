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

        Returns
        -------
        Quat
            The head rotation quaternion (for use by neck muscles etc.).
        """
        # Map normalised state values to radians
        yaw_rad = face_state.head_yaw * deg_to_rad(HEAD_YAW_MAX)
        pitch_rad = face_state.head_pitch * deg_to_rad(HEAD_PITCH_MAX)
        roll_rad = face_state.head_roll * deg_to_rad(HEAD_ROLL_MAX)

        # Soft-clamping from constraint solver
        if constraint_state is not None and constraint_state.total_excess > SOFT_CLAMP_THRESHOLD:
            scale = 1.0 / (1.0 + constraint_state.total_excess)
            yaw_rad *= scale
            pitch_rad *= scale
            roll_rad *= scale

        # Compose quaternion in YXZ order (Yaw, Pitch, Roll)
        head_q = quat_from_euler(pitch_rad, yaw_rad, roll_rad, "YXZ")
        head_q = quat_normalize(head_q)
        self._head_quat = head_q

        # Apply full rotation to skull, face, and brain groups around head pivot
        self._apply_pivot_rotation(skull_group, head_q, self._head_pivot)
        self._apply_pivot_rotation(face_group, head_q, self._head_pivot)
        if brain_group is not None:
            self._apply_pivot_rotation(brain_group, head_q, self._head_pivot)

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

    def reset(self) -> None:
        """Reset head rotation to identity."""
        self._head_quat = quat_identity()
        self._last_yaw = 0.0
        self._last_pitch = 0.0
        self._last_roll = 0.0
