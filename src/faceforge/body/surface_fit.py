"""Fitting the body-surface mesh onto the BP3D skeleton.

The surface mesh and the skeleton are different bodies in different poses, so
the mesh is warped onto the skeleton at load: a piecewise Z-remap, a rotation
of each arm chain, and then a refinement that pulls the result onto the BP3D
skin.  That last phase is the one that used to flatten the hands and feet into
blades; see ``docs/sex_morph.md`` for the measurements and the constraint that
replaced it.

Landmarks are extracted here too: from the skeleton's own bone STLs, and from
the surface mesh by shape (the ankle, for instance, is where the foot widens
past the leg).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from faceforge.body.edge_relaxation import enforce_edge_range
from faceforge.body.surface_projection import (
    build_head_mask, closest_points_on_surface, extract_edges,
    laplacian_smooth_displacements, recompute_normals,
    region_constrained_projection,
)

logger = logging.getLogger(__name__)

# ── Surface-based refinement constants ────────────────────────
#
# The refinement pulls the body-surface mesh onto the BP3D skin.  It used to do
# that in one step -- project every vertex to the closest point, cap the
# displacement, smooth it -- and the fit it reported was excellent precisely
# because it had crushed the mesh onto the target: measured on the shipped
# configuration, 1235 triangles ended below a tenth of their original area,
# 3034 hand edges and 727 forearm edges below half their length, the worst at
# 0.4 %.  The hands and feet, which are furthest from the BP3D pose, came out
# as flat blades.
#
# It is now a constrained solve: a few small steps toward the target, with the
# mesh's own edge lengths held inside a band after every step.  A tube whose
# edges may not shorten past a fraction of their rest length cannot be
# flattened, whatever the projection asks for.
SURFACE_K = 16                 # candidate triangles per query point
SURFACE_ITERATIONS = 6         # projection/constraint sweeps
SURFACE_STEP = 0.5             # fraction of the remaining gap taken per sweep
SURFACE_MAX_STRETCH = 0.25     # an edge may lengthen by this much...
SURFACE_MAX_COMPRESSION = 0.2  # ...and shorten by this much, and no more
SURFACE_EDGE_SWEEPS = 24       # constraint sweeps after each projection step
SURFACE_SMOOTH_ITER = 2        # light smoothing of the final displacement
SURFACE_SMOOTH_STR = 0.25
#: A projection is followed in full up to ``SURFACE_NEAR`` and not at all past
#: ``SURFACE_FAR``.  A target that far away is not a difference of *shape*
#: between the two bodies, it is a difference of *pose* -- the body mesh's hand
#: is 5.2 units from the BP3D hand because the two are posed differently -- and
#: dragging the mesh onto it is what flattened the hands.  Measured on the
#: coarse warp: 5653 vertices lie within 1 unit of the target and 2899 beyond
#: 4, and the far group is almost exactly the hands and the feet.
#:
#: A surface-normal agreement test would be the textbook guard here.  It cannot
#: be used: the triangle winding of both meshes is inconsistent (38 % of the
#: body mesh's face normals and 49 % of the BP3D skin's point outward), so the
#: test is noise and merely freezes an arbitrary third of the vertices.
SURFACE_NEAR = 1.5
SURFACE_FAR = 3.5

# FMA IDs: R Humerus=23130, L=23131; R Radius=23464, L=23465
#          R Femur=24474, L=24475; R Tibia=24477, L=24478


def _project(pos, skin_pos, skin_tris, mh_regions, region_kdtrees):
    """Closest point on the BP3D skin, region-constrained when possible."""
    if mh_regions is not None and region_kdtrees is not None:
        return region_constrained_projection(
            pos, skin_pos, skin_tris, mh_regions, region_kdtrees, k=SURFACE_K)
    return closest_points_on_surface(pos, skin_pos, skin_tris, k=SURFACE_K)


def _pull_weight(sq_dists: NDArray) -> NDArray:
    """How much of the projection to follow, by how far away the target is."""
    d = np.sqrt(np.asarray(sq_dists, dtype=np.float64))
    span = max(SURFACE_FAR - SURFACE_NEAR, 1e-9)
    return np.clip((SURFACE_FAR - d) / span, 0.0, 1.0)


def align_to_bp3d(
    male_geom: Any, female_geom: Any, scale: float, translate_z: float,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Align Blender Z-up meshes to BP3D Z-up coordinates.

    Both meshes are already Z-up so no rotation is needed. We:
    1. Center each mesh at X=0
    2. Negate Y (Blender +Y forward vs BP3D -Y anterior)
    3. Scale both by the config scale factor
    4. Align head tops, then shift so male head is at Z=0, apply Z translate

    Returns (male_positions, female_positions, male_normals, female_normals)
    as (V, 3) float32 arrays.
    """
    male_pos = male_geom.positions.reshape(-1, 3).copy()
    female_pos = female_geom.positions.reshape(-1, 3).copy()
    male_norms = male_geom.normals.reshape(-1, 3).copy()
    female_norms = female_geom.normals.reshape(-1, 3).copy()

    # Center each at X=0
    male_pos[:, 0] -= (male_pos[:, 0].min() + male_pos[:, 0].max()) / 2
    female_pos[:, 0] -= (female_pos[:, 0].min() + female_pos[:, 0].max()) / 2

    # Negate Y: Blender +Y = forward, BP3D -Y = anterior
    male_pos[:, 1] *= -1
    female_pos[:, 1] *= -1
    male_norms[:, 1] *= -1
    female_norms[:, 1] *= -1

    # Scale
    male_pos *= scale
    female_pos *= scale

    # Align head tops (max Z) then shift so male head at Z=0
    male_head_z = male_pos[:, 2].max()
    female_head_z = female_pos[:, 2].max()
    female_pos[:, 2] += (male_head_z - female_head_z)

    z_offset = -male_head_z + translate_z
    male_pos[:, 2] += z_offset
    female_pos[:, 2] += z_offset

    # Write aligned positions back to male geometry (used for MeshInstance)
    male_geom.positions = male_pos.reshape(-1).astype(np.float32)
    male_geom.normals = male_norms.reshape(-1).astype(np.float32)

    return (
        male_pos.astype(np.float32),
        female_pos.astype(np.float32),
        male_norms.astype(np.float32),
        female_norms.astype(np.float32),
    )


def load_bone_endpoints(
    assets: Any, fma_id: str, frac: float = 0.15,
) -> tuple[NDArray, NDArray] | None:
    """Load a bone STL and return (proximal, distal) centroids in skull coords.

    Proximal = top ``frac`` by Z, distal = bottom ``frac`` by Z.
    """
    try:
        geom = assets.get_stl(fma_id)
    except Exception:
        return None
    pos = geom.positions.reshape(-1, 3)[:geom.vertex_count].copy()
    t = assets.transform
    pos[:, 0] = (pos[:, 0] - t.center_x) * t.scale_x + t.skull_center_x
    pos[:, 1] = (pos[:, 1] - t.center_y) * t.scale_y + t.skull_center_y
    pos[:, 2] = (pos[:, 2] - t.center_z) * t.scale_z + t.skull_center_z

    n = max(1, int(len(pos) * frac))
    z_sorted = np.argsort(pos[:, 2])
    proximal = pos[z_sorted[-n:]].mean(axis=0)
    distal = pos[z_sorted[:n]].mean(axis=0)
    return proximal, distal

def extract_skeleton_landmarks(assets: Any) -> dict[str, NDArray] | None:
    """Extract joint landmarks from skeleton bone STLs."""
    result: dict[str, NDArray] = {}

    for side_char, fma_offset in (("R", 0), ("L", 1)):
        hum = load_bone_endpoints(assets, f"FMA{23130 + fma_offset}")
        rad = load_bone_endpoints(assets, f"FMA{23464 + fma_offset}")
        if hum is None:
            logger.warning("Cannot load humerus — skipping warp")
            return None
        result[f"shoulder_{side_char}"] = hum[0]
        result[f"elbow_{side_char}"] = hum[1]
        if rad is not None:
            result[f"wrist_{side_char}"] = rad[1]

        fem = load_bone_endpoints(assets, f"FMA{24474 + fma_offset}")
        tib = load_bone_endpoints(assets, f"FMA{24477 + fma_offset}")
        if fem is None:
            logger.warning("Cannot load femur — skipping warp")
            return None
        result[f"hip_{side_char}"] = fem[0]
        result[f"knee_{side_char}"] = fem[1]
        if tib is not None:
            result[f"ankle_{side_char}"] = tib[1]

    logger.info("Skeleton landmarks: %d entries", len(result))
    return result

# ── BP3D skin mesh loading ──────────────────────────────────

def load_bp3d_skin_mesh(
    assets: Any, cache: dict,
) -> Optional[tuple[NDArray, NDArray, NDArray]]:
    """Load BP3D skin STL (FMA7163) and return full mesh data.

    Returns (positions, normals, tri_indices) in skull coords, or None.
    Caches the result for reuse.
    """
    if "skin" in cache:
        return cache["skin"]

    try:
        geom = assets.get_stl("FMA7163", indexed=True)
    except Exception:
        logger.warning("BP3D skin (FMA7163) not available — skipping surface refinement")
        return None

    pos = geom.positions.reshape(-1, 3)[:geom.vertex_count].copy().astype(np.float64)
    nrm = geom.normals.reshape(-1, 3)[:geom.vertex_count].copy().astype(np.float64)

    if geom.indices is None:
        logger.warning("BP3D skin has no indices — skipping surface refinement")
        return None
    tri_indices = geom.indices.reshape(-1, 3)

    # Transform positions to skull coords
    t = assets.transform
    pos[:, 0] = (pos[:, 0] - t.center_x) * t.scale_x + t.skull_center_x
    pos[:, 1] = (pos[:, 1] - t.center_y) * t.scale_y + t.skull_center_y
    pos[:, 2] = (pos[:, 2] - t.center_z) * t.scale_z + t.skull_center_z

    # Transform normals: flip X, renormalize
    nrm[:, 0] = -nrm[:, 0]
    lengths = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm /= np.maximum(lengths, 1e-12)

    cache["skin"] = (pos, nrm, tri_indices)
    logger.info(
        "Loaded BP3D skin mesh: %d vertices, %d triangles",
        len(pos), len(tri_indices),
    )
    return cache["skin"]

def surface_skin_refinement(
    coarse_pos: NDArray,
    indices: Optional[NDArray],
    skin_pos: NDArray,
    skin_normals: NDArray,
    skin_tris: NDArray,
    skel_lm: dict[str, NDArray],
    *,
    mh_regions: Optional[NDArray] = None,
    region_kdtrees: Optional[dict] = None,
) -> NDArray:
    """Project coarse-warped vertices onto BP3D skin surface.

    Uses point-to-surface projection instead of vertex-to-vertex NN.

    Parameters
    ----------
    coarse_pos : (V, 3) float64
        Vertex positions after Phase 1+2 warp.
    indices : (F*3,) int array or None
        Triangle indices for edge extraction.
    skin_pos : (N, 3) float64
        BP3D skin surface vertex positions.
    skin_normals : (N, 3) float64
        BP3D skin surface normals.
    skin_tris : (T, 3) int
        BP3D skin triangle indices.
    skel_lm : dict
        Skeleton landmarks.
    mh_regions : (V,) int32, optional
        Per-vertex region labels for MH mesh.
    region_kdtrees : dict, optional
        Per-region KDTrees from ``build_region_kdtrees``.

    Returns
    -------
    (V, 3) float64
        Displacement to add to coarse_pos.
    """
    V = len(coarse_pos)
    pos = np.asarray(coarse_pos, dtype=np.float64).copy()

    if indices is None:
        # Without connectivity there is no constraint to enforce, so a
        # single guarded projection is all that can safely be done.
        closest, sq, _tri = _project(pos, skin_pos, skin_tris,
                                          mh_regions, region_kdtrees)
        return (closest - pos) * _pull_weight(sq)[:, None] * SURFACE_STEP

    edges = extract_edges(indices)
    rest_len = np.linalg.norm(pos[edges[:, 0]] - pos[edges[:, 1]], axis=1)

    report = {}
    for _ in range(SURFACE_ITERATIONS):
        closest, sq, _hit = _project(pos, skin_pos, skin_tris,
                                          mh_regions, region_kdtrees)
        pull = _pull_weight(sq)[:, None] * SURFACE_STEP
        pos += pull * (closest - pos)
        report = enforce_edge_range(
            pos, edges, rest_len,
            max_stretch=SURFACE_MAX_STRETCH,
            max_compression=SURFACE_MAX_COMPRESSION,
            iterations=SURFACE_EDGE_SWEEPS,
        )

    disp = pos - np.asarray(coarse_pos, dtype=np.float64)
    if SURFACE_SMOOTH_ITER:
        disp = laplacian_smooth_displacements(
            edges, disp, iterations=SURFACE_SMOOTH_ITER, strength=SURFACE_SMOOTH_STR,
        )
        # Smoothing is not constraint-aware, so the band is re-imposed on
        # the result rather than on an intermediate state.
        final = np.asarray(coarse_pos, dtype=np.float64) + disp
        report = enforce_edge_range(
            final, edges, rest_len,
            max_stretch=SURFACE_MAX_STRETCH,
            max_compression=SURFACE_MAX_COMPRESSION,
            iterations=SURFACE_EDGE_SWEEPS * 4,
        )
        disp = final - np.asarray(coarse_pos, dtype=np.float64)
    logger.info("Surface refinement: edge constraints %s after %d sweeps",
                "met" if report.get("converged") else "not fully met",
                report.get("iterations_run", 0))
    return disp

def refine_onto_skin(morph: Any, pos: NDArray, disp: NDArray,
                     skel_lm: dict, assets: Any) -> NDArray:
    """Pull the coarse-warped surface onto the BP3D skin, region by region.

    Returns the extra displacement, or zeros when the skin or scipy is
    unavailable -- the coarse warp alone is a usable result.
    """
    zero = np.zeros_like(np.asarray(disp, dtype=np.float64))
    skin_data = load_bp3d_skin_mesh(assets, morph._bp3d_skin_mesh_cache)
    if skin_data is None:
        return zero
    try:
        coarse_warped = np.asarray(pos, dtype=np.float64) + disp
        skin_pos, skin_normals, skin_tris = skin_data

        mh_regions = None
        region_trees = None
        try:
            from faceforge.body.region_labels import (
                segment_mh_mesh, segment_bp3d_skin, build_region_kdtrees,
                load_region_overrides, apply_region_overrides,
            )
            if morph._mh_region_labels is None:
                morph._mh_region_labels = segment_mh_mesh(coarse_warped, skel_lm)
                overrides = load_region_overrides()
                if overrides.get("mh_body"):
                    apply_region_overrides(morph._mh_region_labels, overrides["mh_body"])
                logger.info("MH region labels computed: %d vertices",
                            len(morph._mh_region_labels))
            if morph._bp3d_tri_regions is None:
                morph._bp3d_tri_regions = segment_bp3d_skin(skin_pos, skin_tris, skel_lm)
                overrides = load_region_overrides()
                if overrides.get("bp3d_skin"):
                    apply_region_overrides(morph._bp3d_tri_regions, overrides["bp3d_skin"])
                logger.info("BP3D tri regions computed: %d triangles",
                            len(morph._bp3d_tri_regions))
            if morph._region_kdtrees is None:
                morph._region_kdtrees = build_region_kdtrees(
                    skin_pos, skin_tris, morph._bp3d_tri_regions)
            mh_regions = morph._mh_region_labels
            region_trees = morph._region_kdtrees
        except Exception as exc:                              # noqa: BLE001 - logged
            logger.warning("Region segmentation failed, using global KDTree: %s", exc)

        surface_disp = surface_skin_refinement(
            coarse_warped, morph._mesh_indices, skin_pos, skin_normals, skin_tris,
            skel_lm, mh_regions=mh_regions, region_kdtrees=region_trees)
        logger.info("Surface skin refinement: median=%.1f, max=%.1f",
                    float(np.median(np.linalg.norm(surface_disp, axis=1))),
                    float(np.max(np.linalg.norm(surface_disp, axis=1))))
        return surface_disp
    except ImportError:
        logger.warning("scipy not available -- skipping surface refinement")
        return zero


def extract_mesh_landmarks(pos: NDArray) -> dict[str, NDArray]:
    """Extract approximate joint landmarks from the body mesh vertices."""
    result: dict[str, NDArray] = {}
    z = pos[:, 2]
    x = pos[:, 0]

    for side_char, x_sign in (("R", 1), ("L", -1)):
        # ── Arms: lateral vertices above hip level ──
        arm_mask = (x * x_sign > 14) & (z > -95)
        if arm_mask.sum() < 10:
            continue
        arm_pts = pos[arm_mask]
        arm_z = arm_pts[:, 2]

        n_top = max(1, int(arm_mask.sum() * 0.10))
        result[f"shoulder_{side_char}"] = arm_pts[
            np.argsort(arm_z)[-n_top:]
        ].mean(axis=0)

        n_bot = max(1, int(arm_mask.sum() * 0.10))
        result[f"wrist_{side_char}"] = arm_pts[
            np.argsort(arm_z)[:n_bot]
        ].mean(axis=0)

        mid_z = (arm_z.max() + arm_z.min()) / 2
        elbow_band = (arm_z > mid_z - 5) & (arm_z < mid_z + 5)
        if elbow_band.sum() > 0:
            result[f"elbow_{side_char}"] = arm_pts[elbow_band].mean(axis=0)
        else:
            result[f"elbow_{side_char}"] = (
                result[f"shoulder_{side_char}"]
                + result[f"wrist_{side_char}"]
            ) / 2

        # ── Legs: below hip, not too wide ──
        leg_mask = (z < -90) & (x * x_sign > 3) & (np.abs(x) < 28)
        if leg_mask.sum() < 10:
            continue
        leg_pts = pos[leg_mask]
        leg_z = leg_pts[:, 2]

        n_top = max(1, int(leg_mask.sum() * 0.10))
        result[f"hip_{side_char}"] = leg_pts[
            np.argsort(leg_z)[-n_top:]
        ].mean(axis=0)

        # Ankle: detect leg→foot transition via Y-extent widening
        ankle_z = find_ankle_z(pos, x_sign)
        ankle_band = leg_mask & (z > ankle_z - 3) & (z < ankle_z + 3)
        if ankle_band.sum() > 0:
            result[f"ankle_{side_char}"] = pos[ankle_band].mean(axis=0)
        else:
            n_bot = max(1, int(leg_mask.sum() * 0.10))
            result[f"ankle_{side_char}"] = leg_pts[
                np.argsort(leg_z)[:n_bot]
            ].mean(axis=0)

        hi_z = float(result[f"hip_{side_char}"][2])
        an_z = float(result[f"ankle_{side_char}"][2])
        knee_z_val = (hi_z + an_z) / 2
        knee_band = leg_mask & (z > knee_z_val - 5) & (z < knee_z_val + 5)
        if knee_band.sum() > 0:
            result[f"knee_{side_char}"] = pos[knee_band].mean(axis=0)
        else:
            result[f"knee_{side_char}"] = (
                result[f"hip_{side_char}"]
                + result[f"ankle_{side_char}"]
            ) / 2

        # Foot centroid (below ankle)
        foot_mask = (z < ankle_z) & (x * x_sign > 2)
        if foot_mask.sum() > 5:
            result[f"foot_{side_char}"] = pos[foot_mask].mean(axis=0)

    return result

def find_ankle_z(pos: NDArray, x_sign: int) -> float:
    """Find ankle Z where leg transitions to foot (Y-extent increase).

    Scans from leg (top) downward; the ankle is the first Z where the
    cross-section Y-extent widens beyond the leg baseline.
    """
    z = pos[:, 2]
    x = pos[:, 0]
    y = pos[:, 1]
    z_min = float(z.min())

    # Scan Z-bands from bottom to z_min + 50 (covers foot + lower leg)
    z_levels = np.arange(z_min + 2, z_min + 50, 1.0)
    y_extents = []
    for zl in z_levels:
        band = (z > zl - 1.5) & (z < zl + 1.5) & (x * x_sign > 5)
        if band.sum() > 3:
            y_extents.append(float(y[band].max() - y[band].min()))
        else:
            y_extents.append(0.0)

    if not y_extents:
        return z_min + 10  # fallback

    n = len(y_extents)
    # Leg baseline from the upper 50% of z_levels (clearly leg)
    leg_extents = [ye for ye in y_extents[n // 2:] if ye > 0]
    if not leg_extents:
        return z_min + 10
    leg_median = float(np.median(leg_extents))
    threshold = leg_median * 1.3

    # Scan from TOP (leg) downward — ankle is first Z where
    # Y-extent exceeds the leg baseline threshold
    for i in range(n - 1, -1, -1):
        if y_extents[i] > threshold:
            return float(z_levels[i])

    return float(z_levels[0])
