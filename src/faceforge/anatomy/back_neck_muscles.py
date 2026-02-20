"""Body-end pinning for back-of-neck muscles.

Back-of-neck muscles (Desc. Trapezius, Splenius, Semispinalis, Longissimus,
Spinalis, Iliocostalis, Cervical Rotators, Interspinales, Intertransversarii)
span the head-body boundary.  They live in the body back muscle group,
are registered with ``SoftTissueSkinning``, and have ``headFollow`` config.

Unlike PlatysmaHandler which applies its own Rodrigues rotation, this handler
only post-processes after the existing soft tissue + head-follow pipeline.
It pins body-end vertices back toward rest (+ fascia delta) and clamps
excessive displacement to prevent "flapping" during head rotation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray

from faceforge.core.math_utils import Quat, quat_identity

if TYPE_CHECKING:
    from faceforge.anatomy.fascia import FasciaSystem
    from faceforge.body.soft_tissue import SkinBinding, SoftTissueSkinning

_logger = logging.getLogger(__name__)

# Body-end threshold: vertices with spine_frac below this are candidates
# for fascia pinning (region assignment).
_FASCIA_FRAC_THRESHOLD = 0.35

# Overall fascia pin strength (0 = disabled, 1 = hard pin).
# Slightly softer than Platysma's 1.0 for deeper muscles.
_FASCIA_PIN_STRENGTH = 0.8

# Maximum displacement from rest before elastic pullback kicks in.
# Larger than Platysma's 2.0 since these are bigger muscles.
_MAX_DISPLACEMENT = 2.5

# Displacement at which elastic pullback starts (soft threshold).
_SOFT_DISP = 0.8

# Frac range over which displacement clamp fades to zero at skull end.
_CLAMP_FADE_START = 0.85
_CLAMP_FADE_END = 1.0


@dataclass
class _BackNeckMuscleData:
    """Internal per-muscle data for back-of-neck correction."""
    binding: SkinBinding
    defn: dict
    rest_positions: NDArray[np.float64]   # (N, 3) in world space
    spine_fracs: NDArray[np.float32]      # (N,) 0=body end, 1=skull end
    # Fascia assignments (filled by set_fascia_system)
    fascia_assignments: Optional[NDArray[np.int32]] = None   # (N,) region index or -1
    fascia_region_names: Optional[list[str]] = None


class BackNeckMuscleHandler:
    """Post-processing handler that pins body-end vertices of back-of-neck muscles.

    These muscles already receive deformation from:
      1. SoftTissueSkinning (delta-matrix joint transforms)
      2. _apply_head_to_skin → apply_to_skin_muscle (Y-based head-follow blending)

    This handler runs after both and re-pins body-end vertices to prevent
    them from drifting away from the skeleton during head rotation.

    Parameters
    ----------
    head_pivot : tuple, optional
        Head rotation center in world coordinates.
    """

    def __init__(
        self,
        head_pivot: tuple[float, float, float] | None = None,
    ) -> None:
        from faceforge.constants import get_jaw_pivot
        from faceforge.core.math_utils import vec3
        self._head_pivot = vec3(*(head_pivot or get_jaw_pivot()))
        self._muscles: list[_BackNeckMuscleData] = []
        self._registered = False
        self._fascia: Optional[FasciaSystem] = None

    @property
    def registered(self) -> bool:
        return self._registered

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        soft_tissue: SoftTissueSkinning,
        muscle_defs: dict[str, dict],
    ) -> None:
        """Scan soft tissue bindings for headFollow muscles and register them.

        Parameters
        ----------
        soft_tissue : SoftTissueSkinning
            The soft tissue system containing muscle bindings.
        muscle_defs : dict[str, dict]
            Mapping of muscle name → definition dict from back_muscles.json.
        """
        self._muscles.clear()

        for binding in soft_tissue.bindings:
            if not binding.is_muscle:
                continue
            if binding.head_follow_config is None:
                continue

            name = binding.muscle_name or ""
            defn = muscle_defs.get(name)
            if defn is None:
                continue

            # Only process muscles that have headFollow in their definition
            if "headFollow" not in defn:
                continue

            rest = binding.mesh.rest_positions
            if rest is None:
                continue

            rest_3d = rest.reshape(-1, 3).astype(np.float64)
            vert_count = rest_3d.shape[0]
            if vert_count < 4:
                continue

            # Compute spine fracs from Y extent (same logic as neck muscles).
            # Higher Y = closer to skull = higher fraction.
            y_vals = rest_3d[:, 1]
            y_min = float(y_vals.min())
            y_max = float(y_vals.max())
            y_range = y_max - y_min

            if y_range < 1e-6:
                fracs = np.full(vert_count, 0.5, dtype=np.float32)
            else:
                fracs = ((y_vals - y_min) / y_range).astype(np.float32)

            self._muscles.append(_BackNeckMuscleData(
                binding=binding,
                defn=defn,
                rest_positions=rest_3d.copy(),
                spine_fracs=fracs,
            ))

        self._registered = len(self._muscles) > 0
        if self._registered:
            _logger.info("BackNeckMuscleHandler: registered %d muscles", len(self._muscles))

    # ------------------------------------------------------------------
    # Fascia system
    # ------------------------------------------------------------------

    def set_fascia_system(self, fascia: FasciaSystem) -> None:
        """Wire the fascia constraint system and assign body-end vertices."""
        self._fascia = fascia
        for md in self._muscles:
            self._assign_fascia_regions(md)

    def _assign_fascia_regions(self, md: _BackNeckMuscleData) -> None:
        """Assign each body-end vertex to a fascia region via arc distribution.

        Follows the same logic as PlatysmaHandler._assign_fascia_regions.
        """
        if self._fascia is None:
            return

        fascia_cfg = md.defn.get("fasciaRegions")
        if not fascia_cfg:
            md.fascia_assignments = None
            md.fascia_region_names = None
            return

        # Accept both list and dict formats
        if isinstance(fascia_cfg, list):
            region_names = list(fascia_cfg)
        elif isinstance(fascia_cfg, dict):
            medial = fascia_cfg.get("medial")
            lateral = fascia_cfg.get("lateral")
            if not medial or not lateral:
                md.fascia_assignments = None
                md.fascia_region_names = None
                return
            region_names = [medial, lateral]
        else:
            md.fascia_assignments = None
            md.fascia_region_names = None
            return

        muscle_name = md.defn.get("name", "?")
        valid_names: list[str] = []
        valid_targets: list[NDArray[np.float64]] = []
        for name in region_names:
            target = self._fascia.get_target_rest(name)
            if target is not None:
                valid_names.append(name)
                valid_targets.append(target)

        if not valid_names:
            md.fascia_assignments = None
            md.fascia_region_names = None
            return

        targets = np.array(valid_targets, dtype=np.float64)  # (K, 3)
        vert_count = md.rest_positions.shape[0]
        assignments = np.full(vert_count, -1, dtype=np.int32)
        rest = md.rest_positions
        fracs = md.spine_fracs

        candidate_mask = fracs < _FASCIA_FRAC_THRESHOLD
        if not candidate_mask.any():
            md.fascia_assignments = assignments
            md.fascia_region_names = valid_names
            return

        candidate_indices = np.where(candidate_mask)[0]
        candidate_pos = rest[candidate_mask]
        n_regions = len(valid_names)

        if n_regions == 1:
            assignments[candidate_indices] = 0
        else:
            target_abs_x = np.abs(targets[:, 0])
            sort_order = np.argsort(target_abs_x)  # medial → lateral
            candidate_abs_x = np.abs(candidate_pos[:, 0])
            percentiles = np.linspace(0, 100, n_regions + 1)
            boundaries = np.percentile(candidate_abs_x, percentiles)
            bin_indices = np.searchsorted(boundaries[1:], candidate_abs_x, side="right")
            bin_indices = np.clip(bin_indices, 0, n_regions - 1)
            for ci, vi in enumerate(candidate_indices):
                assignments[vi] = int(sort_order[bin_indices[ci]])

        md.fascia_assignments = assignments
        md.fascia_region_names = valid_names

        _logger.debug("[BNECK.fascia] %s: %d regions, %d body-end verts",
                      muscle_name, n_regions, int(candidate_mask.sum()))

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(self, head_quat: Quat) -> None:
        """Post-process back-of-neck muscles: pin body ends + clamp displacement.

        Call after step 12.5 (_apply_head_to_skin) so that soft tissue skinning
        and head-follow blending are already applied.

        Parameters
        ----------
        head_quat : Quat
            Current head quaternion [x, y, z, w].
        """
        if not self._registered:
            return

        identity_q = quat_identity()
        if np.allclose(head_quat, identity_q, atol=1e-6):
            return

        for md in self._muscles:
            mesh = md.binding.mesh
            positions = mesh.geometry.positions.reshape(-1, 3).astype(np.float64)

            self._apply_fascia_pinning(positions, md)
            self._apply_displacement_clamp(positions, md)

            mesh.geometry.positions[:] = positions.astype(np.float32).ravel()
            mesh.needs_update = True

    def _apply_fascia_pinning(
        self,
        out_pos: NDArray[np.float64],
        md: _BackNeckMuscleData,
    ) -> None:
        """Pin body-end vertices toward their fascia constraint targets."""
        if self._fascia is None:
            return
        if md.fascia_assignments is None or md.fascia_region_names is None:
            return

        assignments = md.fascia_assignments
        region_names = md.fascia_region_names
        fracs = md.spine_fracs.astype(np.float64)

        for region_idx, region_name in enumerate(region_names):
            mask = assignments == region_idx
            if not mask.any():
                continue

            target_rest = self._fascia.get_target_rest(region_name)
            target_current = self._fascia.get_target_current(region_name)
            if target_rest is None or target_current is None:
                continue

            fascia_delta = target_current - target_rest
            if np.linalg.norm(fascia_delta) < 1e-6:
                continue

            pin_weight = np.clip(1.0 - fracs[mask], 0.0, 1.0)
            blend = pin_weight[:, None] * _FASCIA_PIN_STRENGTH

            rest_masked = md.rest_positions[mask]
            target_world = rest_masked + fascia_delta

            out_pos[mask] = out_pos[mask] * (1.0 - blend) + target_world * blend

    def _apply_displacement_clamp(
        self,
        out_pos: NDArray[np.float64],
        md: _BackNeckMuscleData,
    ) -> None:
        """Clamp displacement from rest — pull back body-end vertices that drift too far.

        Clamp strength fades to zero at the skull end so upper vertices
        follow the head freely.
        """
        rest = md.rest_positions
        fracs = md.spine_fracs

        disp_vec = out_pos - rest
        disp_mag = np.linalg.norm(disp_vec, axis=1)
        over_soft = disp_mag > _SOFT_DISP

        if not over_soft.any():
            return

        t = (disp_mag[over_soft] - _SOFT_DISP) / (_MAX_DISPLACEMENT - _SOFT_DISP)
        t = np.clip(t, 0.0, 1.0)
        blend = t * t  # quadratic ease-in

        # Fade clamp strength based on frac: full below CLAMP_FADE_START, zero at skull end
        frac_t = (fracs[over_soft] - _CLAMP_FADE_START) / (_CLAMP_FADE_END - _CLAMP_FADE_START)
        frac_t = np.clip(frac_t, 0.0, 1.0)
        clamp_strength = 1.0 - frac_t * frac_t  # quadratic fade-out
        blend *= clamp_strength

        active = blend > 1e-6
        if not active.any():
            return

        idx = np.where(over_soft)[0][active]
        b = blend[active]
        cm = np.minimum(disp_mag[idx], _MAX_DISPLACEMENT)
        safe_mag = np.maximum(disp_mag[idx], 1e-8)
        ud = disp_vec[idx] / safe_mag[:, None]
        cp = rest[idx] + ud * cm[:, None]
        out_pos[idx] = out_pos[idx] * (1.0 - b[:, None]) + cp * b[:, None]
