"""Body-spanning deformation for Platysma muscles.

Platysma R/L span from the mandible to the clavicle.  They are loaded
as expression muscles but reparented into their own ``platysmaGroup``
under ``bodyRoot`` (identity transform — no group rotation).

Per-vertex rotation is applied directly (like neck muscles):
  - ``spine_frac=0`` (body-end) → no rotation → stays at rest position
  - ``spine_frac=1`` (skull-end) → full head rotation → follows head
  - intermediate → fractional rotation via Rodrigues formula

This avoids the complexity of undoing a group rotation and produces
exact per-vertex results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

_logger = logging.getLogger(__name__)

import numpy as np
from numpy.typing import NDArray

from faceforge.core.math_utils import Quat, Vec3, vec3, quat_identity
from faceforge.core.mesh import MeshInstance
from faceforge.core.scene_graph import SceneNode
from faceforge.anatomy.expression_muscles import ExprMuscleData

if TYPE_CHECKING:
    from faceforge.anatomy.fascia import FasciaSystem


# Body-end threshold: vertices with spine_frac below this are candidates
# for fascia pinning (region assignment).
_FASCIA_FRAC_THRESHOLD = 0.50

# Fascia anchor zone: vertices below this frac get strong rotation
# reduction.  Extends beyond _FASCIA_FRAC_THRESHOLD so the transition
# from anchored body-end to freely-rotating mid-section is smooth.
_FASCIA_ANCHOR_ZONE = 0.95

# Overall fascia pin strength (0 = disabled, 1 = hard pin).
# Controls how much body-end rotation is reduced for fascia-pinned vertices.
_FASCIA_PIN_STRENGTH = 1.0

# Maximum displacement from rest (units) before elastic pullback kicks in.
# Vertices exceeding this are blended back toward rest to prevent the
# Platysma from pulling away from the shoulders.
_MAX_DISPLACEMENT = 2.0

# Displacement at which elastic pullback starts (soft threshold).
# Between _SOFT_DISP and _MAX_DISPLACEMENT the pullback is gradual.
_SOFT_DISP = 0.5


@dataclass
class _PlatysmaData:
    """Internal per-muscle data for Platysma correction."""
    md: ExprMuscleData
    rest_positions: NDArray[np.float64]   # (N, 3) in world space (identity group)
    vert_count: int
    spine_fracs: NDArray[np.float32]      # (N,) 0=body end, 1=skull end
    # Fascia assignments (filled by set_fascia_system)
    fascia_assignments: Optional[NDArray[np.int32]] = None   # (N,) region index or -1
    fascia_region_names: Optional[list[str]] = None          # ordered region names
    # Anti-accumulation: cache AU-deformed positions before Platysma correction
    # to prevent corrections compounding when expression muscles skip (early-exit cache).
    _au_positions: Optional[NDArray[np.float64]] = None      # positions before our correction
    _last_written_sentinel: Optional[NDArray[np.float32]] = None  # first vertex of what we wrote


class PlatysmaHandler:
    """Applies body-spanning deformation to Platysma muscles.

    The Platysma meshes live in ``platysmaGroup`` which has identity
    transform (no group rotation).  Per-vertex rotation is applied
    directly via Rodrigues formula, matching neck muscle behaviour.

    Parameters
    ----------
    head_pivot : tuple
        Jaw pivot / head rotation center in world coordinates.
    """

    def __init__(
        self,
        head_pivot: tuple[float, float, float] | None = None,
    ) -> None:
        from faceforge.constants import get_jaw_pivot
        self._head_pivot = vec3(*(head_pivot or get_jaw_pivot()))
        self._platysma: list[_PlatysmaData] = []
        self._registered = False
        self._fascia: Optional[FasciaSystem] = None
        self._tension_excess: float = 0.0

    @property
    def registered(self) -> bool:
        return self._registered

    @property
    def tension_excess(self) -> float:
        """Return the latest platysma stretch excess for constraint feedback."""
        return self._tension_excess

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, expr_muscle_data: list[ExprMuscleData]) -> None:
        """Find Platysma R/L in expression muscle data and compute spine fracs.

        Must be called after expression muscles are loaded.
        """
        self._platysma.clear()

        for md in expr_muscle_data:
            name = md.defn.get("name", "")
            if "Platysma" not in name:
                continue

            vert_count = md.vert_count
            if vert_count < 4:
                continue

            rest = md.rest_positions.reshape(-1, 3).astype(np.float64)

            # Compute spine fracs from Y extent (same logic as neck muscles).
            # Higher Y = closer to skull = higher fraction.
            y_vals = rest[:, 1]
            y_min = float(y_vals.min())
            y_max = float(y_vals.max())
            y_range = y_max - y_min

            if y_range < 1e-6:
                fracs = np.full(vert_count, 0.5, dtype=np.float32)
            else:
                fracs = ((y_vals - y_min) / y_range).astype(np.float32)

            # Lateral body-end extension: the platysma wraps around the
            # clavicle/deltoid, so extreme lateral vertices (high |X|)
            # should be treated as more body-end even if their Y is mid-range.
            # Reduce frac for lateral vertices to bring them into fascia-pin zone.
            abs_x = np.abs(rest[:, 0])
            x_max = float(abs_x.max())
            if x_max > 1e-6:
                lateral_t = abs_x / x_max  # 0=midline, 1=extreme lateral
                # Reduce frac by up to 0.3 at extreme edges (cubic for smooth onset)
                lateral_reduction = (lateral_t ** 2) * 0.3
                fracs = np.maximum(fracs - lateral_reduction, 0.0).astype(np.float32)

            self._platysma.append(_PlatysmaData(
                md=md,
                rest_positions=rest.copy(),
                vert_count=vert_count,
                spine_fracs=fracs,
            ))

        self._registered = len(self._platysma) > 0

    # ------------------------------------------------------------------
    # Reparenting
    # ------------------------------------------------------------------

    @staticmethod
    def reparent_to_group(
        expr_muscle_data: list[ExprMuscleData],
        platysma_group: SceneNode,
    ) -> int:
        """Move Platysma mesh nodes from exprMuscleGroup to platysmaGroup.

        Call after expression muscles are loaded but before registration.
        Returns the number of nodes reparented.
        """
        count = 0
        for md in expr_muscle_data:
            name = md.defn.get("name", "")
            if "Platysma" not in name:
                continue
            node = md.node
            if node.parent is not None:
                node.parent.remove(node)
            platysma_group.add(node)
            count += 1
            _logger.info("Reparented '%s' to platysmaGroup", name)
        return count

    # ------------------------------------------------------------------
    # Fascia system
    # ------------------------------------------------------------------

    def set_fascia_system(self, fascia: FasciaSystem) -> None:
        """Wire the fascia constraint system and assign body-end vertices.

        Call after skeleton loads (Phase 3) so bone positions are available.
        """
        self._fascia = fascia
        for pd in self._platysma:
            self._assign_fascia_regions(pd)

    def _assign_fascia_regions(self, pd: _PlatysmaData) -> None:
        """Assign each body-end vertex to a fascia region via arc distribution.

        Vertices with ``spine_frac < _FASCIA_FRAC_THRESHOLD`` are
        candidates.  Regions are sorted by their lateral position (|X|
        of their rest target), and body-end vertices are distributed
        across them by percentile of their own |X| coordinate.  This
        ensures all regions receive vertices even when the mesh is
        clustered near one target.

        Supports two config formats for ``fasciaRegions``:
          - **list** (new): ``["pectoral_R", "deltoid_R", ...]``
          - **dict** (legacy): ``{"medial": "pectoral_R", "lateral": "deltoid_R"}``
        """
        if self._fascia is None:
            return

        fascia_cfg = pd.md.defn.get("fasciaRegions")
        if not fascia_cfg:
            pd.fascia_assignments = None
            pd.fascia_region_names = None
            return

        # Accept both list and dict formats
        if isinstance(fascia_cfg, list):
            region_names = list(fascia_cfg)
        elif isinstance(fascia_cfg, dict):
            # Legacy dict: {"medial": "pectoral_R", "lateral": "deltoid_R"}
            medial = fascia_cfg.get("medial")
            lateral = fascia_cfg.get("lateral")
            if not medial or not lateral:
                pd.fascia_assignments = None
                pd.fascia_region_names = None
                return
            region_names = [medial, lateral]
        else:
            pd.fascia_assignments = None
            pd.fascia_region_names = None
            return

        # Collect rest-pose target positions for valid regions
        muscle_name = pd.md.defn.get("name", "?")
        valid_names: list[str] = []
        valid_targets: list[NDArray[np.float64]] = []
        for name in region_names:
            target = self._fascia.get_target_rest(name)
            if target is not None:
                valid_names.append(name)
                valid_targets.append(target)
            else:
                _logger.warning("[PLAT.fascia] %s: region '%s' has no rest target",
                                muscle_name, name)

        if not valid_names:
            _logger.warning("[PLAT.fascia] %s: NO valid fascia targets — "
                            "assignments will be None", muscle_name)
            pd.fascia_assignments = None
            pd.fascia_region_names = None
            return

        targets = np.array(valid_targets, dtype=np.float64)  # (K, 3)

        assignments = np.full(pd.vert_count, -1, dtype=np.int32)
        rest = pd.rest_positions  # (N, 3)
        fracs = pd.spine_fracs

        # Find body-end candidate vertices
        candidate_mask = fracs < _FASCIA_FRAC_THRESHOLD
        if not candidate_mask.any():
            pd.fascia_assignments = assignments
            pd.fascia_region_names = valid_names
            return

        candidate_indices = np.where(candidate_mask)[0]
        candidate_pos = rest[candidate_mask]  # (C, 3)
        n_regions = len(valid_names)

        if n_regions == 1:
            # Single region: assign all candidates
            assignments[candidate_indices] = 0
        else:
            # Sort regions by lateral position (|X| of target)
            target_abs_x = np.abs(targets[:, 0])
            sort_order = np.argsort(target_abs_x)  # medial → lateral

            # Project body-end vertices onto |X| axis
            candidate_abs_x = np.abs(candidate_pos[:, 0])

            # Compute percentile boundaries for equal-count bins
            percentiles = np.linspace(0, 100, n_regions + 1)
            boundaries = np.percentile(candidate_abs_x, percentiles)

            # Assign each vertex to a bin via searchsorted
            # Use right edges: vertex X < boundary[k+1] → bin k
            bin_indices = np.searchsorted(boundaries[1:], candidate_abs_x, side="right")
            bin_indices = np.clip(bin_indices, 0, n_regions - 1)

            # Map bin index → original region index via sort_order
            for ci, vi in enumerate(candidate_indices):
                assignments[vi] = int(sort_order[bin_indices[ci]])

        pd.fascia_assignments = assignments
        pd.fascia_region_names = valid_names

        # Log assignment distribution
        _logger.info("[PLAT.fascia] %s: %d regions, %d body-end verts",
                     muscle_name, n_regions, int(candidate_mask.sum()))
        body_assigns = assignments[candidate_mask]
        for ri, rn in enumerate(valid_names):
            count = int((body_assigns == ri).sum())
            _logger.info("[PLAT.fascia] %s: region[%d] '%s' → %d verts, "
                         "target=[%.1f, %.1f, %.1f]",
                         muscle_name, ri, rn, count,
                         targets[ri][0], targets[ri][1], targets[ri][2])

    def _apply_fascia_pinning(
        self,
        out_pos: NDArray[np.float64],
        pd: _PlatysmaData,
    ) -> None:
        """Pin body-end vertices toward their fascia constraint targets.

        Since platysmaGroup has identity transform, ``out_pos`` is in
        world space.  Fascia targets are also in world space.  No
        coordinate transform needed.
        """
        if self._fascia is None:
            return
        if pd.fascia_assignments is None or pd.fascia_region_names is None:
            return

        assignments = pd.fascia_assignments
        region_names = pd.fascia_region_names
        fracs = pd.spine_fracs.astype(np.float64)

        for region_idx, region_name in enumerate(region_names):
            mask = assignments == region_idx
            if not mask.any():
                continue

            target_rest = self._fascia.get_target_rest(region_name)
            target_current = self._fascia.get_target_current(region_name)
            if target_rest is None or target_current is None:
                continue

            fascia_delta = target_current - target_rest

            # During head rotation, fascia_delta ≈ 0 (bones don't move).
            # Head-rotation anchoring is handled by effective_fracs in
            # update(), so skip the blend when delta is negligible.
            if np.linalg.norm(fascia_delta) < 1e-6:
                continue

            # Pin weight: linear falloff from body end (frac~0) to threshold
            pin_weight = np.clip(1.0 - fracs[mask], 0.0, 1.0)
            blend = pin_weight[:, None] * _FASCIA_PIN_STRENGTH

            # Target in world space: rest + fascia displacement
            rest_masked = pd.rest_positions[mask]
            target_world = rest_masked + fascia_delta

            # Direct blend in world space (no coordinate transform needed)
            out_pos[mask] = out_pos[mask] * (1.0 - blend) + target_world * blend

    # ------------------------------------------------------------------
    # Anti-accumulation helpers
    # ------------------------------------------------------------------

    def _restore_au_positions(self) -> None:
        """Restore AU-deformed positions and clear correction tracking.

        Called when head rotation goes to identity/zero so the mesh
        doesn't retain stale Platysma corrections.
        """
        for pd in self._platysma:
            if pd._au_positions is not None:
                pd.md.mesh.geometry.positions[:] = (
                    pd._au_positions.astype(np.float32).ravel()
                )
                pd.md.mesh.needs_update = True
            pd._last_written_sentinel = None

    def _get_au_positions(self, pd: _PlatysmaData) -> NDArray[np.float64]:
        """Get AU-deformed positions, detecting expression muscle cache skips.

        When expression muscles skip their update (early-exit cache),
        the mesh still contains our previous frame's corrected positions.
        We detect this by comparing the first vertex against what we
        wrote last frame, and restore from cached AU positions instead.
        """
        flat_mesh = pd.md.mesh.geometry.positions
        cur_first = flat_mesh[:3]  # first vertex (x, y, z) as float32

        if (pd._last_written_sentinel is not None
                and pd._au_positions is not None
                and np.array_equal(cur_first, pd._last_written_sentinel)):
            # Expression muscles skipped — mesh still has our correction.
            # Use cached AU-deformed positions.
            return pd._au_positions.copy()

        # Expression muscles wrote fresh positions — use current mesh.
        return flat_mesh.reshape(-1, 3).astype(np.float64).copy()

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(
        self,
        head_quat: Quat,
        head_pivot: Optional[Vec3] = None,
        bone_anchor_current: Optional[NDArray[np.float64]] = None,
        bone_anchor_rest: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """Apply per-vertex head rotation to Platysma muscles.

        Since platysmaGroup has identity transform, per-vertex rotation
        is applied directly (no group rotation to undo):

          result = R(frac * theta) * (P - pivot) + pivot + body_delta * (1-frac)

        This produces exact fractional rotation:
          - frac=0 (body-end): stays at AU-deformed rest position
          - frac=1 (skull-end): full head rotation (matches skull)
          - intermediate: smooth gradient

        Parameters
        ----------
        head_quat : Quat
            Current head quaternion [x, y, z, w].
        head_pivot : Vec3, optional
            Head rotation pivot in world coords.
        bone_anchor_current : NDArray, optional
            Current clavicle-area bone position (for body-delta).
        bone_anchor_rest : NDArray, optional
            Rest-pose clavicle-area bone position.
        """
        if not self._registered:
            return

        if head_pivot is None:
            head_pivot = self._head_pivot

        identity_q = quat_identity()
        is_identity = np.allclose(head_quat, identity_q, atol=1e-6)

        # One-time fascia wiring diagnostic (fires on first non-identity update)
        if not is_identity and not getattr(self, '_fascia_diagnosed', False):
            self._fascia_diagnosed = True
            for pd in self._platysma:
                _nm = pd.md.defn.get("name", "?")
                _has_fascia = self._fascia is not None
                _has_assign = pd.fascia_assignments is not None
                _n_pinned = int((pd.fascia_assignments >= 0).sum()) if _has_assign else 0
                _n_regions = len(pd.fascia_region_names) if pd.fascia_region_names else 0
                _logger.debug(
                    "[PLAT.FASCIA_DIAG] %s: fascia=%s assignments=%s "
                    "pinned_verts=%d regions=%d total_verts=%d",
                    _nm, _has_fascia, _has_assign, _n_pinned, _n_regions, pd.vert_count,
                )

        if is_identity:
            self._restore_au_positions()
            return

        # Decompose head quaternion into axis + angle for Rodrigues
        qx, qy, qz, qw = head_quat
        sin_half = np.sqrt(qx * qx + qy * qy + qz * qz)
        full_angle = 2.0 * np.arctan2(sin_half, qw)

        if abs(full_angle) < 1e-8:
            self._restore_au_positions()
            return

        axis = np.array([qx, qy, qz], dtype=np.float64) / sin_half
        kx, ky, kz = axis
        K = np.array([
            [0.0, -kz, ky],
            [kz, 0.0, -kx],
            [-ky, kx, 0.0],
        ], dtype=np.float64)
        K2 = K @ K

        # Compute body-delta if bone anchors provided
        body_delta = np.zeros(3, dtype=np.float64)
        if bone_anchor_current is not None and bone_anchor_rest is not None:
            body_delta = bone_anchor_current - bone_anchor_rest

        pivot = np.asarray(head_pivot, dtype=np.float64)
        plat_excess = 0.0

        for pd in self._platysma:
            md = pd.md
            # Get AU-deformed positions, handling expression muscle cache skips
            cur_pos = self._get_au_positions(pd)
            # Cache AU-deformed positions before we apply our correction
            pd._au_positions = cur_pos.copy()

            rest = pd.rest_positions  # (N, 3)
            fracs = pd.spine_fracs   # (N,)

            # Reduce effective rotation for fascia-anchored body-end vertices.
            # Two zones:
            #   frac < threshold (0.35): hard anchor — near-zero rotation
            #   threshold <= frac < anchor_zone (0.50): smooth cubic ramp
            #     back to original frac (prevents stretch discontinuity)
            #   frac >= anchor_zone: no change
            effective_fracs = fracs.astype(np.float64)
            if (self._fascia is not None
                    and pd.fascia_assignments is not None
                    and (pd.fascia_assignments >= 0).any()):
                T = _FASCIA_FRAC_THRESHOLD
                Z = _FASCIA_ANCHOR_ZONE
                S = _FASCIA_PIN_STRENGTH

                # Zone 1: hard anchor (frac < threshold)
                pinned = pd.fascia_assignments >= 0  # frac < T
                effective_fracs[pinned] *= (1.0 - S)

                # Zone 2: transition (T <= frac < Z) — smooth ramp
                trans = (~pinned) & (effective_fracs < Z)
                if trans.any():
                    t = (effective_fracs[trans] - T) / (Z - T)
                    t = np.clip(t, 0.0, 1.0)
                    # Cubic smoothstep: 3t^2 - 2t^3
                    smooth = t * t * (3.0 - 2.0 * t)
                    # At t=0 (frac=T): boundary_frac = T*(1-S)
                    # At t=1 (frac=Z): original frac
                    boundary = T * (1.0 - S)
                    effective_fracs[trans] = (
                        boundary * (1.0 - smooth)
                        + effective_fracs[trans] * smooth
                    )

            # Per-vertex fractional rotation angles
            per_vert_angles = effective_fracs * full_angle  # (N,)
            sin_a = np.sin(per_vert_angles)
            cos_a = np.cos(per_vert_angles)

            # Apply R(frac * theta) to (cur_pos - pivot) via Rodrigues
            rel = cur_pos - pivot
            Kv = (K @ rel.T).T
            K2v = (K2 @ rel.T).T
            rotated = rel + sin_a[:, None] * Kv + (1.0 - cos_a[:, None]) * K2v

            # Final position: rotated + pivot + body_delta weighted by (1-frac)
            body_weight = (1.0 - fracs.astype(np.float64))[:, None]
            final_pos = rotated + pivot
            if np.linalg.norm(body_delta) > 1e-6:
                final_pos += body_weight * body_delta

            # Apply fascia pinning to body-end vertices (world space)
            self._apply_fascia_pinning(final_pos, pd)

            # Elastic displacement clamping — pull mid-section vertices back
            # toward rest when displacement exceeds _SOFT_DISP.  The clamp
            # strength fades to zero at frac=1.0 so the skull-end (jaw
            # attachment) follows the head freely.
            _CLAMP_FADE_START = 0.92  # full clamp strength below this
            _CLAMP_FADE_END = 1.0     # zero clamp at skull-end
            disp_vec = final_pos - rest
            disp_mag = np.linalg.norm(disp_vec, axis=1)
            over_soft = disp_mag > _SOFT_DISP
            if over_soft.any():
                t = (disp_mag[over_soft] - _SOFT_DISP) / (_MAX_DISPLACEMENT - _SOFT_DISP)
                t = np.clip(t, 0.0, 1.0)
                blend = t * t  # quadratic ease-in
                # Fade clamp strength based on frac: full below 0.75, zero at 1.0
                frac_t = (fracs[over_soft] - _CLAMP_FADE_START) / (_CLAMP_FADE_END - _CLAMP_FADE_START)
                frac_t = np.clip(frac_t, 0.0, 1.0)
                clamp_strength = 1.0 - frac_t * frac_t  # quadratic fade-out
                blend *= clamp_strength
                # Only apply where blend > 0
                active = blend > 1e-6
                if active.any():
                    idx = np.where(over_soft)[0][active]
                    b = blend[active]
                    cm = np.minimum(disp_mag[idx], _MAX_DISPLACEMENT)
                    safe_mag = np.maximum(disp_mag[idx], 1e-8)
                    ud = disp_vec[idx] / safe_mag[:, None]
                    cp = rest[idx] + ud * cm[:, None]
                    final_pos[idx] = (
                        final_pos[idx] * (1.0 - b[:, None])
                        + cp * b[:, None]
                    )

            final_flat = final_pos.astype(np.float32).ravel()
            md.mesh.geometry.positions[:] = final_flat
            pd._last_written_sentinel = final_flat[:3].copy()
            md.mesh.needs_update = True

            # Compute stretch tension for constraint feedback.
            # Measure distance between body-end and skull-end centroids
            # vs rest, then convert excess stretch into tension.
            _body_mask_t = fracs < 0.2
            _skull_mask_t = fracs > 0.8
            if _body_mask_t.any() and _skull_mask_t.any():
                body_centroid = final_pos[_body_mask_t].mean(axis=0)
                skull_centroid = final_pos[_skull_mask_t].mean(axis=0)
                rest_body_centroid = rest[_body_mask_t].mean(axis=0)
                rest_skull_centroid = rest[_skull_mask_t].mean(axis=0)
                cur_len = float(np.linalg.norm(skull_centroid - body_centroid))
                rest_len = float(np.linalg.norm(rest_skull_centroid - rest_body_centroid))
                if rest_len > 1e-6:
                    stretch_ratio = cur_len / rest_len
                    # Threshold: 1.15 (15% stretch = comfortable range)
                    # Stiffness: 2.0 (how quickly tension builds beyond threshold)
                    _PLAT_MAX_STRETCH = 1.15
                    _PLAT_STIFFNESS = 2.0
                    if stretch_ratio > _PLAT_MAX_STRETCH:
                        plat_excess += (stretch_ratio - _PLAT_MAX_STRETCH) * _PLAT_STIFFNESS

        # Store total platysma tension for constraint feedback
        self._tension_excess = plat_excess
