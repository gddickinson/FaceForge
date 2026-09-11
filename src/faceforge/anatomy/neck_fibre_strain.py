"""Volume-preserving fibre strain for the neck muscles.

Each muscle's fibre axis runs from the body attachment region to the skull
one.  When head rotation or body follow lengthens that axis the belly thins,
and when it shortens the belly bulges, radially about the axis -- weighted so
the body end, which has to stay on its bone, is left alone.

Split out of ``neck_muscles`` to keep that module under the project's
file-size limit; the system still exposes ``_init_fiber_geometry`` and
``_apply_fiber_strain`` as thin delegates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - typing only
    from faceforge.anatomy.neck_muscles import NeckMuscleData

#: Blend factor -- 50% of the full volume-preserving effect.  Conservative,
#: so a stretched muscle never thins far enough to detach.
STRAIN_STRENGTH = 0.5

#: Fibre-axis stretch is clamped to this range before it drives any radius.
STRETCH_CLAMP = (0.75, 1.4)


def init_fiber_geometry(md: "NeckMuscleData") -> None:
    """Pre-compute rest-pose fiber axis, centroids, and radial offsets.

    The fiber axis runs from the lower (body) attachment region to the
    upper (skull) attachment region.  Radial offsets measure each vertex's
    perpendicular distance from the fiber axis, used for volume-preserving
    bulging during deformation.
    """
    if md.vert_count < 4:
        return

    pos = md.rest_positions.reshape(-1, 3).astype(np.float64)
    fracs = md.spine_fracs.astype(np.float64)

    frac_min = float(fracs.min())
    frac_max = float(fracs.max())
    frac_range = frac_max - frac_min
    if frac_range < 0.05:
        return  # too small — can't define meaningful fiber axis

    upper_thresh = frac_min + frac_range * 0.85
    lower_thresh = frac_min + frac_range * 0.15

    upper_mask = fracs >= upper_thresh
    lower_mask = fracs <= lower_thresh

    if upper_mask.sum() < 2 or lower_mask.sum() < 2:
        return

    upper_centroid = pos[upper_mask].mean(axis=0)
    lower_centroid = pos[lower_mask].mean(axis=0)
    fiber_vec = upper_centroid - lower_centroid
    fiber_len = float(np.linalg.norm(fiber_vec))

    if fiber_len < 0.1:
        return

    fiber_dir = fiber_vec / fiber_len
    centroid = pos.mean(axis=0)

    # Per-vertex decomposition along the fiber axis
    relative = pos - centroid
    axial_scalar = relative @ fiber_dir          # (N,) signed distance along axis
    axial_proj = axial_scalar[:, None] * fiber_dir  # (N, 3) axial component
    radial = relative - axial_proj                   # (N, 3) perpendicular component

    md.fiber_axis_rest = fiber_dir
    md.fiber_length_rest = fiber_len
    md.centroid_rest = centroid
    md.upper_centroid_rest = upper_centroid
    md.lower_centroid_rest = lower_centroid
    md.radial_offsets_rest = radial
    md.axial_positions_rest = axial_scalar

def apply_fiber_strain(
    out_pos: NDArray[np.float64],
    out_nrm: NDArray[np.float64],
    md: "NeckMuscleData",
    strength: float = STRAIN_STRENGTH,
    clamp: tuple[float, float] = STRETCH_CLAMP,
) -> None:
    """Apply gentle volume-preserving fiber strain after rotation.

    Measures how much the muscle has stretched or compressed along its
    fiber axis (from the rotated attachment centroids) and applies radial
    scaling to suggest volume preservation:
    - Stretched muscles → slight radial contraction (muscle thins)
    - Compressed muscles → slight radial expansion (muscle bulges)

    The effect is weighted by spine_frac so body-end vertices are
    unaffected (preventing detachment from the skeleton).
    """
    if md.fiber_axis_rest is None or md.vert_count < 4:
        return

    fracs = md.spine_fracs.astype(np.float64)
    frac_min = float(fracs.min())
    frac_max = float(fracs.max())
    frac_range = frac_max - frac_min
    if frac_range < 0.05:
        return

    # Identify upper/lower attachment regions (same thresholds as init)
    upper_thresh = frac_min + frac_range * 0.85
    lower_thresh = frac_min + frac_range * 0.15
    upper_mask = fracs >= upper_thresh
    lower_mask = fracs <= lower_thresh

    if upper_mask.sum() < 2 or lower_mask.sum() < 2:
        return

    # Current attachment centroids after rotation
    cur_upper = out_pos[upper_mask].mean(axis=0)
    cur_lower = out_pos[lower_mask].mean(axis=0)
    cur_fiber_vec = cur_upper - cur_lower
    cur_length = float(np.linalg.norm(cur_fiber_vec))

    if cur_length < 0.1:
        return

    cur_fiber_dir = cur_fiber_vec / cur_length

    # Stretch ratio: how much did the fiber axis elongate?
    stretch = cur_length / md.fiber_length_rest
    stretch = np.clip(stretch, clamp[0], clamp[1])

    if abs(stretch - 1.0) < 0.01:
        return

    # Volume-preserving radial scale: r' = r / sqrt(stretch)
    # Blended with identity by ``strength`` for a gentler effect
    full_radial_scale = 1.0 / np.sqrt(stretch)
    radial_scale = 1.0 + (full_radial_scale - 1.0) * strength

    # Decompose relative to the LOWER centroid (body anchor) rather than
    # the overall centroid. This keeps body-end vertices pinned.
    anchor = cur_lower
    relative = out_pos - anchor

    axial_scalar = relative @ cur_fiber_dir        # (N,)
    axial_proj = axial_scalar[:, None] * cur_fiber_dir
    radial = relative - axial_proj

    # Per-vertex blend weight: body-end verts (low frac) get no strain,
    # skull-end verts (high frac) get full strain effect. This prevents
    # lower-end detachment from the skeleton.
    t = np.clip((fracs - frac_min) / frac_range, 0.0, 1.0)
    per_vert_scale = 1.0 + (radial_scale - 1.0) * t  # (N,)

    out_pos[:] = anchor + axial_proj + radial * per_vert_scale[:, None]

    # Update normals: inverse-transpose of radial scaling
    nrm_axial = (out_nrm @ cur_fiber_dir)[:, None] * cur_fiber_dir
    nrm_radial = out_nrm - nrm_axial
    nrm_inv_scale = 1.0 + (1.0 / radial_scale - 1.0) * t  # (N,)
    out_nrm[:] = nrm_axial + nrm_radial * nrm_inv_scale[:, None]

    # Re-normalize
    nrm_lengths = np.linalg.norm(out_nrm, axis=1, keepdims=True)
    nrm_lengths = np.maximum(nrm_lengths, 1e-8)
    out_nrm /= nrm_lengths
