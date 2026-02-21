"""Scanner engine: casts parallel rays through the scene to produce cross-section images.

Uses vectorized Moller-Trumbore ray-triangle intersection with AABB pre-filtering
for performance. Supports multiple reduction modes (mean, max, min, sum).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from faceforge.scanner.tissue_map import TissueMapper
from faceforge.core.math_utils import Mat4


@dataclass
class _CachedMesh:
    """Pre-transformed triangles for a single mesh."""
    v0: NDArray  # (M, 3) float64
    v1: NDArray  # (M, 3)
    v2: NDArray  # (M, 3)
    tissue: str
    color: tuple[float, float, float]
    aabb_min: NDArray  # (3,)
    aabb_max: NDArray  # (3,)


class ScannerEngine:
    """Casts rays through the scene and produces cross-section images."""

    def __init__(self, tissue_mapper: TissueMapper):
        self.tissue_mapper = tissue_mapper
        self._cache: list[_CachedMesh] = []

    def cache_meshes(self, meshes: list[tuple]) -> None:
        """Pre-transform mesh triangles to world space and cache for scanning.

        Parameters
        ----------
        meshes : list of (MeshInstance, Mat4)
            Visible meshes with their world transforms from scene.collect_meshes().
        """
        self._cache.clear()

        for mesh_inst, world_mat in meshes:
            geom = mesh_inst.geometry
            positions = geom.positions.reshape(-1, 3).astype(np.float64)

            # Transform to world space
            rot = world_mat[:3, :3].astype(np.float64)
            trans = world_mat[:3, 3].astype(np.float64)
            world_pos = (positions @ rot.T) + trans

            # Build triangle arrays
            if geom.has_indices:
                idx = geom.indices.reshape(-1, 3)
                v0 = world_pos[idx[:, 0]]
                v1 = world_pos[idx[:, 1]]
                v2 = world_pos[idx[:, 2]]
            else:
                world_pos = world_pos.reshape(-1, 3, 3)
                v0 = world_pos[:, 0]
                v1 = world_pos[:, 1]
                v2 = world_pos[:, 2]

            if len(v0) == 0:
                continue

            # AABB
            all_verts = np.concatenate([v0, v1, v2], axis=0)
            aabb_min = all_verts.min(axis=0)
            aabb_max = all_verts.max(axis=0)

            # Classify tissue
            color = mesh_inst.material.color
            tissue = self.tissue_mapper.classify(mesh_inst.name, color)

            self._cache.append(_CachedMesh(
                v0=v0, v1=v1, v2=v2,
                tissue=tissue, color=color,
                aabb_min=aabb_min, aabb_max=aabb_max,
            ))

    def scan(
        self,
        origin: NDArray,
        normal: NDArray,
        right: NDArray,
        up: NDArray,
        width: float,
        height: float,
        depth: float,
        resolution: int,
        mode: str,
        reduction: str,
        progress_callback: Callable[[float], None] | None = None,
    ) -> NDArray:
        """Cast rays and produce a cross-section image.

        Parameters
        ----------
        origin : (3,) array -- center of the scan plane
        normal : (3,) array -- ray direction (into the slab)
        right  : (3,) array -- rightward direction on the scan plane
        up     : (3,) array -- upward direction on the scan plane
        width, height : float -- dimensions of the scan region
        depth  : float -- slab thickness along normal
        resolution : int -- output image size (resolution x resolution)
        mode : str -- imaging mode (ct, mri_t1, mri_t2, xray, anatomical)
        reduction : str -- how to combine hits (mean, max, min, sum)
        progress_callback : optional callable(float) -- called with 0..1 progress

        Returns
        -------
        If mode == 'anatomical': (resolution, resolution, 3) float32 RGB image
        Otherwise: (resolution, resolution) float32 grayscale image
        """
        is_anatomical = (mode == "anatomical")

        # Build ray grid
        u = np.linspace(-0.5, 0.5, resolution)
        v = np.linspace(0.5, -0.5, resolution)  # top to bottom
        uu, vv = np.meshgrid(u, v)
        uu_flat = uu.ravel()
        vv_flat = vv.ravel()
        n_rays = resolution * resolution

        # Ray origins: origin + u*right*width + v*up*height
        ray_origins = (
            origin[np.newaxis, :]
            + uu_flat[:, np.newaxis] * right[np.newaxis, :] * width
            + vv_flat[:, np.newaxis] * up[np.newaxis, :] * height
        )  # (N, 3)

        ray_dir = normal.astype(np.float64)
        norm = np.linalg.norm(ray_dir)
        if norm > 1e-10:
            ray_dir = ray_dir / norm

        # Slab frustum AABB for mesh pre-filtering
        slab_corners = []
        for du in (-0.5, 0.5):
            for dv in (-0.5, 0.5):
                for dd in (0.0, depth):
                    corner = origin + du * right * width + dv * up * height + dd * ray_dir
                    slab_corners.append(corner)
        slab_corners = np.array(slab_corners)
        slab_min = slab_corners.min(axis=0) - 1.0
        slab_max = slab_corners.max(axis=0) + 1.0

        # Accumulate hits per ray
        if is_anatomical:
            color_accum = np.zeros((n_rays, 3), dtype=np.float64)
            hit_count = np.zeros(n_rays, dtype=np.float64)
        else:
            value_accum = np.zeros(n_rays, dtype=np.float64)
            value_min = np.full(n_rays, np.inf, dtype=np.float64)
            value_max = np.full(n_rays, -np.inf, dtype=np.float64)
            hit_count = np.zeros(n_rays, dtype=np.float64)

        # Count meshes that pass AABB (for progress reporting)
        active_meshes = []
        for cm in self._cache:
            if not (np.any(cm.aabb_max < slab_min) or np.any(cm.aabb_min > slab_max)):
                active_meshes.append(cm)

        total = len(active_meshes)

        # Process each mesh that overlaps the slab
        for i, cm in enumerate(active_meshes):
            tissue_value = self.tissue_mapper.get_value(cm.tissue, mode)

            self._intersect_mesh(
                ray_origins, ray_dir, depth, cm,
                tissue_value, mode, is_anatomical,
                color_accum if is_anatomical else None,
                None if is_anatomical else value_accum,
                None if is_anatomical else value_min,
                None if is_anatomical else value_max,
                hit_count,
            )

            if progress_callback is not None and total > 0:
                progress_callback((i + 1) / total)

        # Apply reduction
        if is_anatomical:
            mask = hit_count > 0
            result = np.zeros((n_rays, 3), dtype=np.float32)
            if mask.any():
                result[mask] = (color_accum[mask] / hit_count[mask, np.newaxis]).astype(np.float32)
            return result.reshape(resolution, resolution, 3)
        else:
            result = np.zeros(n_rays, dtype=np.float32)
            mask = hit_count > 0

            if reduction == "mean":
                if mask.any():
                    result[mask] = (value_accum[mask] / hit_count[mask]).astype(np.float32)
            elif reduction == "max":
                if mask.any():
                    result[mask] = value_max[mask].astype(np.float32)
            elif reduction == "min":
                if mask.any():
                    result[mask] = value_min[mask].astype(np.float32)
            elif reduction == "sum":
                if mask.any():
                    result[mask] = (1.0 - np.exp(-value_accum[mask])).astype(np.float32)
            else:
                if mask.any():
                    result[mask] = (value_accum[mask] / hit_count[mask]).astype(np.float32)

            return result.reshape(resolution, resolution)

    def _intersect_mesh(
        self,
        ray_origins: NDArray,
        ray_dir: NDArray,
        depth: float,
        cm: _CachedMesh,
        tissue_value: float,
        mode: str,
        is_anatomical: bool,
        color_accum: NDArray | None,
        value_accum: NDArray | None,
        value_min: NDArray | None,
        value_max: NDArray | None,
        hit_count: NDArray,
    ) -> None:
        """Intersect all rays with a mesh's triangles, chunked to limit memory."""
        n_rays = len(ray_origins)
        n_tris = len(cm.v0)
        chunk_size = max(1, 500_000 // max(n_tris, 1))

        for start in range(0, n_rays, chunk_size):
            end = min(start + chunk_size, n_rays)
            chunk_origins = ray_origins[start:end]  # (C, 3)

            hit_mask, t_vals = _moller_trumbore(
                chunk_origins, ray_dir, cm.v0, cm.v1, cm.v2,
            )

            # Filter by slab depth
            valid = hit_mask & (t_vals >= 0) & (t_vals <= depth)

            # Any hit per ray
            any_hit = valid.any(axis=1)  # (C,)
            ray_hits = valid.sum(axis=1).astype(np.float64)  # (C,)

            hit_indices = np.where(any_hit)[0]
            if len(hit_indices) == 0:
                continue

            abs_indices = hit_indices + start
            hit_count[abs_indices] += ray_hits[hit_indices]

            if is_anatomical:
                color_arr = np.array(cm.color)
                color_accum[abs_indices] += color_arr[np.newaxis, :] * ray_hits[hit_indices, np.newaxis]
            else:
                value_accum[abs_indices] += tissue_value * ray_hits[hit_indices]
                value_min[abs_indices] = np.minimum(value_min[abs_indices], tissue_value)
                value_max[abs_indices] = np.maximum(value_max[abs_indices], tissue_value)


def _moller_trumbore(
    ray_origins: NDArray,
    ray_dir: NDArray,
    v0: NDArray,
    v1: NDArray,
    v2: NDArray,
) -> tuple[NDArray, NDArray]:
    """Vectorized Moller-Trumbore ray-triangle intersection.

    Parameters
    ----------
    ray_origins : (N, 3) -- ray start points
    ray_dir : (3,) -- shared ray direction
    v0, v1, v2 : (M, 3) -- triangle vertices

    Returns
    -------
    hit_mask : (N, M) bool -- True where ray i intersects triangle j
    t_values : (N, M) float -- distance along ray (valid only where hit_mask is True)
    """
    N = len(ray_origins)
    M = len(v0)

    edge1 = v1 - v0  # (M, 3)
    edge2 = v2 - v0  # (M, 3)

    # h = ray_dir x edge2
    h = np.cross(ray_dir, edge2)  # (M, 3)

    # a = edge1 . h
    a = np.sum(edge1 * h, axis=1)  # (M,)

    EPSILON = 1e-8
    valid_tri = np.abs(a) > EPSILON

    inv_a = np.zeros(M, dtype=np.float64)
    inv_a[valid_tri] = 1.0 / a[valid_tri]

    # s = ray_origin - v0  -> (N, M, 3)
    s = ray_origins[:, np.newaxis, :] - v0[np.newaxis, :, :]

    # u = inv_a * (s . h)
    u = np.sum(s * h[np.newaxis, :, :], axis=2) * inv_a[np.newaxis, :]

    # q = s x edge1
    q = np.cross(s, edge1[np.newaxis, :, :])

    # v = inv_a * (ray_dir . q)
    v = np.sum(ray_dir[np.newaxis, np.newaxis, :] * q, axis=2) * inv_a[np.newaxis, :]

    # t = inv_a * (edge2 . q)
    t = np.sum(edge2[np.newaxis, :, :] * q, axis=2) * inv_a[np.newaxis, :]

    hit_mask = (
        valid_tri[np.newaxis, :]
        & (u >= 0) & (u <= 1)
        & (v >= 0) & ((u + v) <= 1)
        & (t > EPSILON)
    )

    return hit_mask, t
