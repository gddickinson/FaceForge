"""Scanner engine: cross-section imaging via tiled ray-triangle intersection.

Pipeline:
1. **Filter** — slab depth + 2D area test discards ~95%+ of triangles
2. **Tile** — surviving triangles are assigned to 16×16 pixel tiles by
   their projected 2D bounding box
3. **Intersect** — per-tile vectorised Möller–Trumbore on (K_rays, T_tris)
   with K ≈ 256, T typically 10–200, keeping intermediates small and
   cache-friendly

Complexity is O(sum of triangle × tile-overlap) instead of O(N_rays × M_tris).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from faceforge.scanner.tissue_map import TissueMapper


TILE_SIZE = 16  # pixels per tile edge


@dataclass
class _CachedMesh:
    """Pre-transformed triangles for a single mesh."""
    v0: NDArray  # (M, 3) float32
    v1: NDArray
    v2: NDArray
    tissue: str
    color: tuple[float, float, float]
    aabb_min: NDArray  # (3,)
    aabb_max: NDArray


class ScannerEngine:
    """Casts rays through the scene and produces cross-section images."""

    def __init__(self, tissue_mapper: TissueMapper):
        self.tissue_mapper = tissue_mapper
        self._cache: list[_CachedMesh] = []

    def cache_meshes(self, meshes: list[tuple]) -> None:
        """Pre-transform mesh triangles to world space and cache."""
        self._cache.clear()

        for mesh_inst, world_mat in meshes:
            geom = mesh_inst.geometry
            positions = geom.positions.reshape(-1, 3).astype(np.float32)

            rot = world_mat[:3, :3].astype(np.float32)
            trans = world_mat[:3, 3].astype(np.float32)
            world_pos = (positions @ rot.T) + trans

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

            all_v = np.concatenate([v0, v1, v2], axis=0)
            color = mesh_inst.material.color
            tissue = self.tissue_mapper.classify(mesh_inst.name, color)

            self._cache.append(_CachedMesh(
                v0=v0, v1=v1, v2=v2,
                tissue=tissue, color=color,
                aabb_min=all_v.min(axis=0), aabb_max=all_v.max(axis=0),
            ))

    # ── public entry point ───────────────────────────────────────────

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
        is_anatomical = (mode == "anatomical")
        res = resolution
        n_rays = res * res

        # Float32 vectors
        origin_f = origin.astype(np.float32)
        ray_dir = normal.astype(np.float32)
        n = np.linalg.norm(ray_dir)
        if n > 1e-10:
            ray_dir /= n
        right_f = right.astype(np.float32)
        up_f = up.astype(np.float32)
        w_f, h_f, depth_f = np.float32(width), np.float32(height), np.float32(depth)

        # Ray origin grid  (res, res, 3)
        u_vals = np.linspace(-0.5, 0.5, res, dtype=np.float32)
        v_vals = np.linspace(0.5, -0.5, res, dtype=np.float32)
        ray_grid = (
            origin_f
            + u_vals[np.newaxis, :, np.newaxis] * (right_f * w_f)
            + v_vals[:, np.newaxis, np.newaxis] * (up_f * h_f)
        )  # (res, res, 3)

        # Coarse slab frustum AABB for mesh-level rejection
        corners = np.empty((8, 3), dtype=np.float32)
        ci = 0
        for du in (-0.5, 0.5):
            for dv in (-0.5, 0.5):
                for dd in (0.0, depth):
                    corners[ci] = origin_f + du * right_f * w_f + dv * up_f * h_f + dd * ray_dir
                    ci += 1
        slab_min = corners.min(axis=0) - 1.0
        slab_max = corners.max(axis=0) + 1.0

        # ── Phase 1: filter triangles across all meshes ──────────────
        all_v0, all_v1, all_v2 = [], [], []
        all_tissue_val: list[float] = []
        all_color: list[tuple] = []
        # 2D projections for tile assignment (world units along right / up)
        all_u_min, all_u_max = [], []
        all_w_min, all_w_max = [], []

        hw, hh = w_f * 0.5, h_f * 0.5

        for cm in self._cache:
            if np.any(cm.aabb_max < slab_min) or np.any(cm.aabb_min > slab_max):
                continue

            rel0 = cm.v0 - origin_f
            rel1 = cm.v1 - origin_f
            rel2 = cm.v2 - origin_f

            # Slab depth
            d0 = rel0 @ ray_dir
            d1 = rel1 @ ray_dir
            d2 = rel2 @ ray_dir
            in_slab = (np.maximum(np.maximum(d0, d1), d2) >= 0) & \
                      (np.minimum(np.minimum(d0, d1), d2) <= depth_f)

            # 2D area
            pu0 = rel0 @ right_f; pu1 = rel1 @ right_f; pu2 = rel2 @ right_f
            pw0 = rel0 @ up_f;    pw1 = rel1 @ up_f;    pw2 = rel2 @ up_f
            tu_min = np.minimum(np.minimum(pu0, pu1), pu2)
            tu_max = np.maximum(np.maximum(pu0, pu1), pu2)
            tw_min = np.minimum(np.minimum(pw0, pw1), pw2)
            tw_max = np.maximum(np.maximum(pw0, pw1), pw2)
            in_area = (tu_max >= -hw) & (tu_min <= hw) & \
                      (tw_max >= -hh) & (tw_min <= hh)

            keep = np.where(in_slab & in_area)[0]
            if len(keep) == 0:
                continue

            all_v0.append(cm.v0[keep])
            all_v1.append(cm.v1[keep])
            all_v2.append(cm.v2[keep])

            tv = self.tissue_mapper.get_value(cm.tissue, mode)
            all_tissue_val.extend([tv] * len(keep))
            all_color.extend([cm.color] * len(keep))

            all_u_min.append(tu_min[keep])
            all_u_max.append(tu_max[keep])
            all_w_min.append(tw_min[keep])
            all_w_max.append(tw_max[keep])

        if not all_v0:
            if is_anatomical:
                return np.zeros((res, res, 3), dtype=np.float32)
            return np.zeros((res, res), dtype=np.float32)

        v0_all = np.concatenate(all_v0)  # (M, 3)
        v1_all = np.concatenate(all_v1)
        v2_all = np.concatenate(all_v2)
        tissue_arr = np.array(all_tissue_val, dtype=np.float32)  # (M,)
        color_arr = np.array(all_color, dtype=np.float32)        # (M, 3)
        u_min_all = np.concatenate(all_u_min)
        u_max_all = np.concatenate(all_u_max)
        w_min_all = np.concatenate(all_w_min)
        w_max_all = np.concatenate(all_w_max)
        M = len(v0_all)

        if progress_callback:
            progress_callback(0.3)  # filtering done

        # ── Pre-compute edge vectors (reused across all tiles) ──────
        edge1_all = v1_all - v0_all  # (M, 3)
        edge2_all = v2_all - v0_all  # (M, 3)
        EPSILON = np.float32(1e-6)
        h_all = np.cross(ray_dir, edge2_all)              # (M, 3)
        a_all = np.sum(edge1_all * h_all, axis=1)         # (M,)
        good_all = np.abs(a_all) > EPSILON
        inv_a_all = np.zeros(M, dtype=np.float32)
        inv_a_all[good_all] = np.float32(1.0) / a_all[good_all]

        # ── Phase 2: tile coord ranges per triangle (vectorised) ───
        n_tiles_x = (res + TILE_SIZE - 1) // TILE_SIZE
        n_tiles_y = (res + TILE_SIZE - 1) // TILE_SIZE
        inv_w, inv_h = 1.0 / w_f, 1.0 / h_f
        res_m1 = np.float32(res - 1)

        px_lo = np.clip(((u_min_all * inv_w + 0.5) * res_m1 - 1).astype(np.int32), 0, res - 1)
        px_hi = np.clip(((u_max_all * inv_w + 0.5) * res_m1 + 2).astype(np.int32), 0, res - 1)
        py_lo = np.clip(((0.5 - w_max_all * inv_h) * res_m1 - 1).astype(np.int32), 0, res - 1)
        py_hi = np.clip(((0.5 - w_min_all * inv_h) * res_m1 + 2).astype(np.int32), 0, res - 1)

        tx_lo = px_lo // TILE_SIZE
        tx_hi = px_hi // TILE_SIZE
        ty_lo = py_lo // TILE_SIZE
        ty_hi = py_hi // TILE_SIZE

        if progress_callback:
            progress_callback(0.35)

        # ── Phase 3: tile-centric intersection with chunking ───────
        if is_anatomical:
            color_accum = np.zeros((n_rays, 3), dtype=np.float32)
        else:
            value_accum = np.zeros(n_rays, dtype=np.float32)
            value_min_a = np.full(n_rays, np.inf, dtype=np.float32)
            value_max_a = np.full(n_rays, -np.inf, dtype=np.float32)
        hit_count = np.zeros(n_rays, dtype=np.float32)

        TRI_CHUNK = 256  # max triangles per intersection batch
        total_tile_cells = n_tiles_x * n_tiles_y
        done_tiles = 0

        for tx in range(n_tiles_x):
            # Reuse x-mask across all ty for this column
            x_mask = (tx_lo <= tx) & (tx_hi >= tx)
            if not x_mask.any():
                done_tiles += n_tiles_y
                continue

            ix_lo = tx * TILE_SIZE
            ix_hi = min(res, (tx + 1) * TILE_SIZE)

            for ty in range(n_tiles_y):
                done_tiles += 1
                tri_idx = np.where(x_mask & (ty_lo <= ty) & (ty_hi >= ty))[0]
                if len(tri_idx) == 0:
                    continue

                T = len(tri_idx)
                iy_lo = ty * TILE_SIZE
                iy_hi = min(res, (ty + 1) * TILE_SIZE)

                tile_origins = ray_grid[iy_lo:iy_hi, ix_lo:ix_hi]
                ny, nx = tile_origins.shape[:2]
                K = ny * nx
                origins = tile_origins.reshape(K, 3)

                # Flat index mapping (once per tile)
                iy_local = np.arange(ny, dtype=np.int32)
                ix_local = np.arange(nx, dtype=np.int32)
                iy_g, ix_g = np.meshgrid(iy_local, ix_local, indexing="ij")
                flat_map = ((iy_g + iy_lo) * res + (ix_g + ix_lo)).ravel()

                # Process triangles in cache-friendly chunks
                for c0 in range(0, T, TRI_CHUNK):
                    cidx = tri_idx[c0:min(c0 + TRI_CHUNK, T)]

                    # Pre-computed per-triangle data (indexed, not recomputed)
                    c_v0 = v0_all[cidx]
                    c_h = h_all[cidx]
                    c_edge1 = edge1_all[cidx]
                    c_edge2 = edge2_all[cidx]
                    c_inv_a = inv_a_all[cidx]
                    c_good = good_all[cidx]

                    # Broadcast Möller–Trumbore  (K, Tc)
                    s = origins[:, np.newaxis, :] - c_v0[np.newaxis, :, :]
                    u_par = np.sum(s * c_h[np.newaxis, :, :], axis=2) * c_inv_a
                    q = np.cross(s, c_edge1[np.newaxis, :, :])
                    v_par = np.sum(ray_dir * q, axis=2) * c_inv_a
                    t_par = np.sum(c_edge2[np.newaxis, :, :] * q, axis=2) * c_inv_a

                    hit = (
                        c_good[np.newaxis, :]
                        & (u_par >= 0) & (u_par <= 1)
                        & (v_par >= 0) & ((u_par + v_par) <= 1)
                        & (t_par > EPSILON) & (t_par <= depth_f)
                    )

                    any_hit = hit.any(axis=1)
                    if not any_hit.any():
                        continue

                    ray_hit_count = hit.sum(axis=1).astype(np.float32)
                    hit_rays = np.where(any_hit)[0]
                    flat_idx = flat_map[hit_rays]
                    hit_count[flat_idx] += ray_hit_count[hit_rays]

                    if is_anatomical:
                        hit_sub = hit[hit_rays]
                        ray_color = hit_sub.astype(np.float32) @ color_arr[cidx]
                        color_accum[flat_idx] += ray_color
                    else:
                        c_tissue = tissue_arr[cidx]
                        hit_sub = hit[hit_rays]
                        value_accum[flat_idx] += hit_sub.astype(np.float32) @ c_tissue
                        masked_max = np.where(hit_sub, c_tissue[np.newaxis, :], -np.inf)
                        masked_min = np.where(hit_sub, c_tissue[np.newaxis, :], np.inf)
                        np.maximum.at(value_max_a, flat_idx, masked_max.max(axis=1))
                        np.minimum.at(value_min_a, flat_idx, masked_min.min(axis=1))

                if progress_callback and done_tiles % 8 == 0:
                    progress_callback(0.35 + 0.65 * done_tiles / total_tile_cells)

        if progress_callback:
            progress_callback(1.0)

        # ── Reduction ────────────────────────────────────────────────
        if is_anatomical:
            mask = hit_count > 0
            result = np.zeros((n_rays, 3), dtype=np.float32)
            if mask.any():
                result[mask] = color_accum[mask] / hit_count[mask, np.newaxis]
            return result.reshape(res, res, 3)
        else:
            result = np.zeros(n_rays, dtype=np.float32)
            mask = hit_count > 0
            if not mask.any():
                return result.reshape(res, res)
            if reduction == "mean":
                result[mask] = value_accum[mask] / hit_count[mask]
            elif reduction == "max":
                result[mask] = value_max_a[mask]
            elif reduction == "min":
                result[mask] = value_min_a[mask]
            elif reduction == "sum":
                result[mask] = 1.0 - np.exp(-value_accum[mask])
            else:
                result[mask] = value_accum[mask] / hit_count[mask]
            return result.reshape(res, res)
