"""Geometry the body-surface fit is built from: projection, edges, normals.

Pure functions on arrays, with no knowledge of the morph that calls them.
They were methods on :class:`~faceforge.body.gender_morph.GenderMorphSystem`
until that module grew past the size a single concern should occupy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def closest_point_on_triangle_batch(
    P: NDArray, A: NDArray, B: NDArray, C: NDArray,
) -> NDArray:
    """Compute closest point on triangle for N point-triangle pairs.

    Vectorized Ericson algorithm (Real-Time Collision Detection, Ch. 5.1.5).

    Parameters
    ----------
    P, A, B, C : (N, 3) float64
        Query points and triangle vertices.

    Returns
    -------
    (N, 3) float64
        Closest point on each triangle.
    """
    N = len(P)
    result = np.empty((N, 3), dtype=np.float64)

    ab = B - A
    ac = C - A
    ap = P - A

    d1 = np.sum(ab * ap, axis=1)
    d2 = np.sum(ac * ap, axis=1)

    bp = P - B
    d3 = np.sum(ab * bp, axis=1)
    d4 = np.sum(ac * bp, axis=1)

    cp = P - C
    d5 = np.sum(ab * cp, axis=1)
    d6 = np.sum(ac * cp, axis=1)

    # Region A: vertex A is closest
    reg_a = (d1 <= 0) & (d2 <= 0)

    # Region B: vertex B is closest
    reg_b = (d3 >= 0) & (d4 <= d3)

    # Region C: vertex C is closest
    reg_c = (d6 >= 0) & (d5 <= d6)

    # Region AB: edge AB is closest
    vc = d1 * d4 - d3 * d2
    denom_ab = d1 - d3
    safe_denom_ab = np.where(np.abs(denom_ab) < 1e-30, 1.0, denom_ab)
    v_ab = d1 / safe_denom_ab
    reg_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)

    # Region AC: edge AC is closest
    vb = d5 * d2 - d1 * d6
    denom_ac = d2 - d6
    safe_denom_ac = np.where(np.abs(denom_ac) < 1e-30, 1.0, denom_ac)
    w_ac = d2 / safe_denom_ac
    reg_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)

    # Region BC: edge BC is closest
    va = d3 * d6 - d5 * d4
    denom_bc = (d4 - d3) + (d5 - d6)
    safe_denom_bc = np.where(np.abs(denom_bc) < 1e-30, 1.0, denom_bc)
    w_bc = (d4 - d3) / safe_denom_bc
    reg_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)

    # Interior region: inside triangle
    denom = va + vb + vc
    # Guard against degenerate triangles
    safe_denom = np.where(np.abs(denom) < 1e-30, 1.0, denom)
    v_int = vb / safe_denom
    w_int = vc / safe_denom

    # Assign results by region (later assignments overwrite earlier)
    # Default: interior
    result[:] = A + v_int[:, np.newaxis] * ab + w_int[:, np.newaxis] * ac

    # Edge regions (np.where returns a new array, no out param)
    m = reg_bc[:, np.newaxis]
    result = np.where(m, B + w_bc[:, np.newaxis] * (C - B), result)
    m = reg_ac[:, np.newaxis]
    result = np.where(m, A + w_ac[:, np.newaxis] * ac, result)
    m = reg_ab[:, np.newaxis]
    result = np.where(m, A + v_ab[:, np.newaxis] * ab, result)

    # Vertex regions
    result = np.where(reg_c[:, np.newaxis], C, result)
    result = np.where(reg_b[:, np.newaxis], B, result)
    result = np.where(reg_a[:, np.newaxis], A, result)

    return result


def closest_points_on_surface(
    query_pts: NDArray,
    mesh_pos: NDArray,
    mesh_tris: NDArray,
    k: int = 16,
) -> tuple[NDArray, NDArray]:
    """KDTree-accelerated point-to-surface projection.

    Parameters
    ----------
    query_pts : (Q, 3) float64
        Points to project onto the surface.
    mesh_pos : (V, 3) float64
        Mesh vertex positions.
    mesh_tris : (F, 3) int
        Triangle index array.
    k : int
        Number of candidate triangles per query point.

    Returns
    -------
    (closest_pts, sq_dists, tri_idx) : (Q, 3) float64, (Q,) float64, (Q,) int
    """
    from scipy.spatial import cKDTree

    Q = len(query_pts)
    tri_verts = mesh_pos[mesh_tris]  # (F, 3, 3)
    centroids = tri_verts.mean(axis=1)  # (F, 3)

    tree = cKDTree(centroids)
    # Clamp k to available triangles
    k_actual = min(k, len(centroids))
    _, cand_idx = tree.query(query_pts, k=k_actual)  # (Q, k)

    if cand_idx.ndim == 1:
        cand_idx = cand_idx[:, np.newaxis]

    best_pts = np.empty((Q, 3), dtype=np.float64)
    best_sq = np.full(Q, np.inf, dtype=np.float64)
    best_tri = np.zeros(Q, dtype=np.int64)

    for ci in range(k_actual):
        tri_idx = cand_idx[:, ci]  # (Q,)
        A = mesh_pos[mesh_tris[tri_idx, 0]]
        B = mesh_pos[mesh_tris[tri_idx, 1]]
        C = mesh_pos[mesh_tris[tri_idx, 2]]

        cp = closest_point_on_triangle_batch(
            query_pts, A, B, C,
        )
        diff = cp - query_pts
        sq = np.sum(diff * diff, axis=1)

        better = sq < best_sq
        if better.any():
            best_pts[better] = cp[better]
            best_sq[better] = sq[better]
            best_tri[better] = tri_idx[better]

    return best_pts, best_sq, best_tri


def region_constrained_projection(
    query_pts: NDArray,
    mesh_pos: NDArray,
    mesh_tris: NDArray,
    vertex_regions: NDArray,
    region_kdtrees: dict,
    k: int = 16,
) -> tuple[NDArray, NDArray]:
    """Region-constrained point-to-surface projection.

    Projects each query point only onto BP3D triangles in its matching
    region. Falls back to global search for regions with no BP3D triangles.

    Parameters
    ----------
    query_pts : (Q, 3) float64
    mesh_pos : (V, 3) float64
    mesh_tris : (F, 3) int
    vertex_regions : (Q,) int32 — per-vertex region labels
    region_kdtrees : dict — region_id → (cKDTree, global_tri_indices)
    k : int — candidate triangles per query

    Returns
    -------
    (closest_pts, sq_dists, tri_idx) : (Q, 3) float64, (Q,) float64, (Q,) int
    """
    Q = len(query_pts)
    best_pts = np.empty((Q, 3), dtype=np.float64)
    best_sq = np.full(Q, np.inf, dtype=np.float64)
    best_tri = np.zeros(Q, dtype=np.int64)

    # Process each region separately
    unique_regions = np.unique(vertex_regions)
    fallback_indices = []

    for region_id in unique_regions:
        vmask = vertex_regions == region_id
        v_idx = np.nonzero(vmask)[0]
        pts = query_pts[v_idx]

        if region_id not in region_kdtrees:
            fallback_indices.append(v_idx)
            continue

        tree, global_tri_idx = region_kdtrees[region_id]
        k_actual = min(k, len(global_tri_idx))
        _, local_cand = tree.query(pts, k=k_actual)

        if local_cand.ndim == 1:
            local_cand = local_cand[:, np.newaxis]

        sub_best = np.empty((len(pts), 3), dtype=np.float64)
        sub_sq = np.full(len(pts), np.inf, dtype=np.float64)
        sub_tri = np.zeros(len(pts), dtype=np.int64)

        for ci in range(k_actual):
            local_idx = local_cand[:, ci]
            tri_idx = global_tri_idx[local_idx]
            A = mesh_pos[mesh_tris[tri_idx, 0]]
            B = mesh_pos[mesh_tris[tri_idx, 1]]
            C = mesh_pos[mesh_tris[tri_idx, 2]]

            cp = closest_point_on_triangle_batch(
                pts, A, B, C,
            )
            diff = cp - pts
            sq = np.sum(diff * diff, axis=1)

            better = sq < sub_sq
            if better.any():
                sub_best[better] = cp[better]
                sub_sq[better] = sq[better]
                sub_tri[better] = tri_idx[better]

        best_pts[v_idx] = sub_best
        best_sq[v_idx] = sub_sq
        best_tri[v_idx] = sub_tri

    # Fallback: use global KDTree for vertices in regions with no BP3D tris
    if fallback_indices:
        fb_idx = np.concatenate(fallback_indices)
        fb_pts = query_pts[fb_idx]
        fb_closest, fb_sq, fb_tri = closest_points_on_surface(
            fb_pts, mesh_pos, mesh_tris, k=k,
        )
        best_pts[fb_idx] = fb_closest
        best_sq[fb_idx] = fb_sq
        best_tri[fb_idx] = fb_tri

    return best_pts, best_sq, best_tri


def build_head_mask(
    coarse_pos: NDArray,
    skel_lm: dict[str, NDArray],
) -> tuple[NDArray, NDArray]:
    """Detect head region and return mask + blend weights.

    Parameters
    ----------
    coarse_pos : (V, 3) float64
        Coarse-warped vertex positions.
    skel_lm : dict
        Skeleton landmarks with shoulder positions.

    Returns
    -------
    (head_mask, head_blend) : (V,) bool, (V,) float64
    """
    V = len(coarse_pos)
    z = coarse_pos[:, 2]
    head_mask = np.zeros(V, dtype=bool)
    head_blend = np.zeros(V, dtype=np.float64)

    sh_r = skel_lm.get("shoulder_R")
    sh_l = skel_lm.get("shoulder_L")
    if sh_r is None or sh_l is None:
        return head_mask, head_blend

    sh_z = (float(sh_r[2]) + float(sh_l[2])) / 2
    head_mask = z > sh_z
    head_blend = np.clip((z - sh_z) / 10.0, 0.0, 1.0)
    head_blend[~head_mask] = 0.0

    return head_mask, head_blend


def extract_edges(indices: NDArray) -> NDArray:
    """Extract unique undirected edges from triangle indices.

    Parameters
    ----------
    indices : (F*3,) int array
        Triangle index buffer.

    Returns
    -------
    (E, 2) int array
        Unique undirected edge pairs.
    """
    tri = indices.reshape(-1, 3)
    e01 = np.column_stack([tri[:, 0], tri[:, 1]])
    e12 = np.column_stack([tri[:, 1], tri[:, 2]])
    e20 = np.column_stack([tri[:, 2], tri[:, 0]])
    all_edges = np.concatenate([e01, e12, e20], axis=0)
    sorted_edges = np.sort(all_edges, axis=1)
    return np.unique(sorted_edges, axis=0)


def laplacian_smooth_displacements(
    edges: NDArray,
    disp: NDArray,
    iterations: int = 4,
    strength: float = 0.3,
) -> NDArray:
    """Laplacian-smooth a displacement field over mesh edges.

    Parameters
    ----------
    edges : (E, 2) int array
        Undirected edge pairs.
    disp : (V, 3) float64
        Per-vertex displacement vectors.
    iterations : int
        Number of smoothing passes.
    strength : float
        Blend factor toward neighbor average (0-1).

    Returns
    -------
    (V, 3) float64
        Smoothed displacement field.
    """
    V = len(disp)
    d = disp.copy()

    # Build bidirectional edges for neighbor lookup
    bi_src = np.concatenate([edges[:, 0], edges[:, 1]])
    bi_dst = np.concatenate([edges[:, 1], edges[:, 0]])

    # Precompute neighbor counts
    counts = np.bincount(bi_src, minlength=V).astype(np.float64)
    has_nbrs = counts > 0

    for _ in range(iterations):
        # Sum neighbor displacements
        nbr_sum = np.zeros((V, 3), dtype=np.float64)
        np.add.at(nbr_sum, bi_src, d[bi_dst])

        # Average
        avg = np.zeros((V, 3), dtype=np.float64)
        avg[has_nbrs] = nbr_sum[has_nbrs] / counts[has_nbrs, np.newaxis]

        # Blend
        d[has_nbrs] = (1.0 - strength) * d[has_nbrs] + strength * avg[has_nbrs]

    return d


def rotation_between(v_from: NDArray, v_to: NDArray) -> NDArray:
    """Compute 3x3 rotation matrix: v_from → v_to (Rodrigues)."""
    a = v_from.astype(np.float64)
    b = v_to.astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return np.eye(3, dtype=np.float64)
    a /= na
    b /= nb

    cross = np.cross(a, b)
    sin_a = np.linalg.norm(cross)
    cos_a = np.dot(a, b)

    if sin_a < 1e-8:
        if cos_a > 0:
            return np.eye(3, dtype=np.float64)
        perp = np.array([1, 0, 0], dtype=np.float64)
        if abs(np.dot(a, perp)) > 0.9:
            perp = np.array([0, 1, 0], dtype=np.float64)
        axis = np.cross(a, perp)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)

    axis = cross / sin_a
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + sin_a * K + (1 - cos_a) * (K @ K)


def rotate_normals(norms: NDArray, per_vert_rot: NDArray) -> NDArray:
    """Apply per-vertex rotation to normals (vectorized)."""
    result = np.einsum(
        "ijk,ik->ij", per_vert_rot, norms.astype(np.float64),
    )
    lengths = np.linalg.norm(result, axis=1, keepdims=True)
    result /= np.maximum(lengths, 1e-8)
    return result.astype(np.float32)


def recompute_normals(pos: NDArray, indices: NDArray) -> NDArray:
    """Recompute per-vertex normals from mesh faces.
    Parameters
    ----------
    pos : (V, 3) float32
        Vertex positions.
    indices : (F*3,) int array
        Triangle index buffer.
    Returns
    -------
    (V, 3) float32
        Normalized per-vertex normals.
    """
    V = len(pos)
    tris = indices.reshape(-1, 3)
    v0 = pos[tris[:, 0]]
    v1 = pos[tris[:, 1]]
    v2 = pos[tris[:, 2]]
    face_normals = np.cross(
        (v1 - v0).astype(np.float64),
        (v2 - v0).astype(np.float64),
    )
    # Accumulate face normals to vertices (area-weighted by cross product magnitude)
    vert_normals = np.zeros((V, 3), dtype=np.float64)
    np.add.at(vert_normals, tris[:, 0], face_normals)
    np.add.at(vert_normals, tris[:, 1], face_normals)
    np.add.at(vert_normals, tris[:, 2], face_normals)
    lengths = np.linalg.norm(vert_normals, axis=1, keepdims=True)
    vert_normals /= np.maximum(lengths, 1e-8)
    return vert_normals.astype(np.float32)
