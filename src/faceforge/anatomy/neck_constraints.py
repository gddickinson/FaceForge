"""Neck constraint solver: tension monitoring, soft-clamping, spine compensation.

Computes muscle stretch ratios and tensions for each neck muscle,
applies dynamic thoracic compensation (boosted spine fractions), and
soft-clamps head rotation when the total excess exceeds a threshold.

This module has ZERO GL imports; all math is done with NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from faceforge.core.math_utils import (
    Vec3, Quat, vec3,
    quat_identity, quat_slerp, quat_rotate_vec3,
    deg_to_rad,
)
from faceforge.core.scene_graph import SceneNode
from faceforge.core.state import ConstraintState


# Per-vertebra thoracic head-follow base fractions (T1-T6)
THORACIC_HEAD_FRACTIONS = [0.0, 0.02, 0.015, 0.01, 0.005, 0.003]

# Max boost per thoracic level from constraint compensation
THORACIC_MAX_BOOST = 0.40

# Base shoulder compensation fraction and max
SHOULDER_COMP_BASE = 0.03
SHOULDER_COMP_MAX = 0.12

# Soft-clamp starts when total excess exceeds this
SOFT_CLAMP_THRESHOLD = 0.1

# Attachment point computation: top/bottom percentage of spine fraction
ATTACH_TOP_PCTILE = 0.15
ATTACH_BOTTOM_PCTILE = 0.15


@dataclass
class MuscleAttachment:
    """Upper and lower attachment points for a single neck muscle."""
    upper_point: NDArray[np.float64]  # (3,) centroid of top 15% of verts by spine frac
    lower_point: NDArray[np.float64]  # (3,) centroid of bottom 15% of verts
    rest_length: float               # Euclidean distance between attachment points


@dataclass
class MuscleTension:
    """Tension state for a single neck muscle."""
    stretch_ratio: float = 1.0   # current_length / rest_length
    tension: float = 0.0         # Normalised tension [0, 1+]
    excess: float = 0.0          # Amount beyond max stretch (0 if within limits)


class NeckConstraintSolver:
    """Computes neck muscle tension and spine compensation.

    Parameters
    ----------
    joint_limits:
        Dict loaded from ``joint_limits.json``.  Contains
        ``joint_limits.cervical``, ``joint_limits.headTotal``,
        ``joint_limits.thoracicMax``, and ``muscle_limits``.
    """

    def __init__(self, joint_limits_data: dict) -> None:
        jl = joint_limits_data.get("joint_limits", {})
        ml = joint_limits_data.get("muscle_limits", {})

        self._cervical_limits = jl.get("cervical", [])
        self._head_total = jl.get("headTotal", {"yaw": 80, "pitch": 60, "roll": 40})
        self._thoracic_max = jl.get("thoracicMax", {"yaw": 6, "pitch": 4, "roll": 3})

        self._max_stretch = ml.get("maxStretchRatio", 1.4)
        self._min_compress = ml.get("minCompressRatio", 0.7)
        self._stiffness = ml.get("stiffness", 0.85)

        self._attachments: list[MuscleAttachment] = []
        self._tensions: list[MuscleTension] = []

    # ------------------------------------------------------------------
    # Attachment computation
    # ------------------------------------------------------------------

    def compute_attachments(self, neck_muscle_data: list) -> list[MuscleAttachment]:
        """Compute upper/lower attachment points per muscle from rest positions.

        Uses the top/bottom 15% of vertices by spine fraction to determine
        attachment point centroids, then computes the rest length.

        Parameters
        ----------
        neck_muscle_data:
            List of ``NeckMuscleData`` objects from ``NeckMuscleSystem``.

        Returns
        -------
        list[MuscleAttachment]
            One attachment per muscle, in the same order as ``neck_muscle_data``.
        """
        self._attachments = []

        for md in neck_muscle_data:
            pos = md.rest_positions.reshape(-1, 3)
            fracs = md.spine_fracs
            n = md.vert_count

            if n == 0:
                self._attachments.append(MuscleAttachment(
                    upper_point=np.zeros(3, dtype=np.float64),
                    lower_point=np.zeros(3, dtype=np.float64),
                    rest_length=0.0,
                ))
                continue

            # Sort vertices by spine fraction
            sorted_idx = np.argsort(fracs)

            # Bottom 15% (lowest fracs = lower attachment)
            n_bottom = max(1, int(n * ATTACH_BOTTOM_PCTILE))
            bottom_idx = sorted_idx[:n_bottom]
            lower_point = pos[bottom_idx].mean(axis=0).astype(np.float64)

            # Top 15% (highest fracs = upper attachment, skull end)
            n_top = max(1, int(n * ATTACH_TOP_PCTILE))
            top_idx = sorted_idx[-n_top:]
            upper_point = pos[top_idx].mean(axis=0).astype(np.float64)

            rest_length = float(np.linalg.norm(upper_point - lower_point))

            self._attachments.append(MuscleAttachment(
                upper_point=upper_point,
                lower_point=lower_point,
                rest_length=max(rest_length, 1e-6),
            ))

        return self._attachments

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(
        self,
        constraint_state: ConstraintState,
        neck_muscle_data: list,
        head_quat: Quat,
        vertebrae_pivots: Optional[list[dict]] = None,
    ) -> ConstraintState:
        """Solve neck constraints for the current frame.

        1. Compute current stretch ratios and tensions per muscle.
        2. Compute total excess tension.
        3. Apply dynamic thoracic compensation (boost spine fractions).
        4. Compute soft-clamp scale factor.

        The updated ``ConstraintState`` is returned (and modified in-place).

        Parameters
        ----------
        constraint_state:
            The state object to update with tension values.
        neck_muscle_data:
            List of ``NeckMuscleData`` from ``NeckMuscleSystem``.
        head_quat:
            Current head rotation quaternion.
        vertebrae_pivots:
            Optional vertebra pivot info for thoracic compensation.
        """
        if not self._attachments:
            self.compute_attachments(neck_muscle_data)

        identity_q = quat_identity()
        self._tensions = []
        total_excess = 0.0

        for i, (md, att) in enumerate(zip(neck_muscle_data, self._attachments)):
            if att.rest_length < 1e-6:
                self._tensions.append(MuscleTension())
                continue

            # Compute current upper attachment position (rotated with head)
            frac = md.upper_frac
            vert_q = quat_slerp(identity_q, head_quat, frac)

            from faceforge.constants import JAW_PIVOT
            pivot = vec3(*JAW_PIVOT)

            rel = att.upper_point - pivot
            current_upper = quat_rotate_vec3(vert_q, rel) + pivot

            # Lower attachment stays at rest (body-attached)
            current_lower = att.lower_point

            current_length = float(np.linalg.norm(current_upper - current_lower))
            stretch_ratio = current_length / att.rest_length

            # Compute tension
            if stretch_ratio > self._max_stretch:
                excess = (stretch_ratio - self._max_stretch) * self._stiffness
                tension = 1.0 + excess
            elif stretch_ratio < self._min_compress:
                excess = (self._min_compress - stretch_ratio) * self._stiffness
                tension = 0.5 + excess
            else:
                excess = 0.0
                # Normalise tension within comfortable range
                t_range = self._max_stretch - self._min_compress
                tension = (stretch_ratio - self._min_compress) / max(t_range, 1e-6)

            total_excess += max(0.0, excess)

            self._tensions.append(MuscleTension(
                stretch_ratio=stretch_ratio,
                tension=tension,
                excess=max(0.0, excess),
            ))

        # Update constraint state
        constraint_state.tensions = [
            {"stretch_ratio": t.stretch_ratio, "tension": t.tension, "excess": t.excess}
            for t in self._tensions
        ]
        constraint_state.attachments = [
            {"upper": a.upper_point.tolist(), "lower": a.lower_point.tolist(), "rest_length": a.rest_length}
            for a in self._attachments
        ]
        constraint_state.total_excess = total_excess

        # Dynamic thoracic compensation
        if vertebrae_pivots is not None and total_excess > SOFT_CLAMP_THRESHOLD:
            self._apply_thoracic_compensation(
                constraint_state, vertebrae_pivots, total_excess
            )

        return constraint_state

    # ------------------------------------------------------------------
    # Thoracic compensation
    # ------------------------------------------------------------------

    def _apply_thoracic_compensation(
        self,
        constraint_state: ConstraintState,
        vertebrae_pivots: list[dict],
        total_excess: float,
    ) -> None:
        """Boost thoracic vertebrae fractions to share the load.

        When neck muscles are overstretched, thoracic vertebrae take on
        a fraction of the head rotation to reduce cervical strain.
        Each thoracic level can be boosted up to ``THORACIC_MAX_BOOST``.
        """
        # Compensation magnitude scales with excess
        comp_factor = min(1.0, total_excess / 1.0)  # Saturate at excess=1.0

        # Compute compensation per axis
        # The thoracic spine distributes a small fraction of the total
        comp_yaw = comp_factor * self._thoracic_max.get("yaw", 6) / 80.0
        comp_pitch = comp_factor * self._thoracic_max.get("pitch", 4) / 60.0
        comp_roll = comp_factor * self._thoracic_max.get("roll", 3) / 40.0

        constraint_state.spine_compensation_yaw = comp_yaw
        constraint_state.spine_compensation_pitch = comp_pitch
        constraint_state.spine_compensation_roll = comp_roll
        constraint_state.spine_compensation_magnitude = comp_factor

    # ------------------------------------------------------------------
    # Tension visualisation
    # ------------------------------------------------------------------

    def apply_tension_colors(self, neck_muscle_data: list) -> None:
        """Tint muscle colors toward red based on tension.

        Muscles under high tension shift from their base color toward red.
        """
        for md, tension_info in zip(neck_muscle_data, self._tensions):
            t = min(1.0, max(0.0, tension_info.tension))
            base = md.mesh.material.color

            # Lerp toward red (1, 0.2, 0.2)
            r = base[0] + (1.0 - base[0]) * t * 0.5
            g = base[1] * (1.0 - t * 0.3)
            b = base[2] * (1.0 - t * 0.3)
            md.mesh.material.color = (
                min(1.0, r),
                max(0.0, g),
                max(0.0, b),
            )
