"""Scanner engine: cross-section imaging via ray-triangle intersection.

Pipeline:
1. **Filter** — slab depth + 2D area test discards ~95%+ of triangles
2. **Box** — each surviving triangle gets its projected pixel bounding box
3. **Intersect** — triangles are grouped by box *shape* so each group is one
   vectorised Möller–Trumbore batch of (T_tris, P_pixels), and hits are
   accumulated with np.bincount over flat ray indices

Each triangle is therefore tested only against the rays inside its own
bounding box.  The earlier version binned triangles into 16×16-pixel tiles
and tested every triangle against all 256 rays of each tile it touched; a
BodyParts3D triangle projects to roughly 20 pixels, so that was a ~17-21x
ray-triangle overtest.  Complexity is now O(sum of per-triangle pixel-box
area) instead of O(sum of triangle × tile area).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from faceforge.scanner.tissue_map import TissueMapper


# Max (triangles x pixels) elements per Möller–Trumbore batch.  Keeps the
# (T, P, 3) temporaries inside a few MB regardless of footprint size.
BATCH_ELEMS = 1 << 21


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

            # min/max over the three vertex arrays directly: the old
            # np.concatenate materialised a 3x copy of every mesh's triangle
            # data just to take an AABB.
            aabb_min = np.minimum(np.minimum(v0.min(axis=0), v1.min(axis=0)),
                                  v2.min(axis=0))
            aabb_max = np.maximum(np.maximum(v0.max(axis=0), v1.max(axis=0)),
                                  v2.max(axis=0))
            color = mesh_inst.material.color
            tissue = self.tissue_mapper.classify(mesh_inst.name, color)

            self._cache.append(_CachedMesh(
                v0=v0, v1=v1, v2=v2,
                tissue=tissue, color=color,
                aabb_min=aabb_min, aabb_max=aabb_max,
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
        all_tissue_val: list[NDArray] = []
        all_color: list[NDArray] = []
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
            # np.full/broadcast_to instead of building Python lists of up to
            # 850k floats and 850k 3-tuples per scan.
            all_tissue_val.append(np.full(len(keep), tv, dtype=np.float32))
            all_color.append(np.broadcast_to(
                np.asarray(cm.color, dtype=np.float32), (len(keep), 3)))

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
        tissue_arr = np.concatenate(all_tissue_val)  # (M,)
        color_arr = np.concatenate(all_color)        # (M, 3)
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

        # ── Phase 2: per-triangle pixel box (vectorised) ───────────
        # Same padding (-1 / +2) as the tile version, so the candidate ray
        # set per triangle is unchanged — only the *grouping* differs.
        inv_w, inv_h = 1.0 / w_f, 1.0 / h_f
        res_m1 = np.float32(res - 1)

        px_lo = np.clip(((u_min_all * inv_w + 0.5) * res_m1 - 1).astype(np.int32), 0, res - 1)
        px_hi = np.clip(((u_max_all * inv_w + 0.5) * res_m1 + 2).astype(np.int32), 0, res - 1)
        py_lo = np.clip(((0.5 - w_max_all * inv_h) * res_m1 - 1).astype(np.int32), 0, res - 1)
        py_hi = np.clip(((0.5 - w_min_all * inv_h) * res_m1 + 2).astype(np.int32), 0, res - 1)

        box_w = (px_hi - px_lo + 1).astype(np.int64)
        box_h = (py_hi - py_lo + 1).astype(np.int64)

        if progress_callback:
            progress_callback(0.35)

        # ── Phase 3: triangle-centric intersection ─────────────────
        ray_flat = ray_grid.reshape(n_rays, 3)
        if is_anatomical:
            color_accum = np.zeros((n_rays, 3), dtype=np.float32)
        else:
            value_accum = np.zeros(n_rays, dtype=np.float32)
            value_min_a = np.full(n_rays, np.inf, dtype=np.float32)
            value_max_a = np.full(n_rays, -np.inf, dtype=np.float32)
        hit_count = np.zeros(n_rays, dtype=np.float32)

        # Group triangles by pixel-box *shape* so each batch is a
        # rectangular (T, W*H) problem.  Shapes are small integers, so the
        # (h, w) pair packs into one int64 sort key.
        shape_key = box_h * (res + 1) + box_w
        order = np.argsort(shape_key, kind="stable")
        sk = shape_key[order]
        bounds = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1], True])
        n_groups = len(bounds) - 1

        for gi in range(n_groups):
            gidx = order[bounds[gi]:bounds[gi + 1]]
            W = int(box_w[gidx[0]])
            H = int(box_h[gidx[0]])
            P = W * H
            step = max(1, BATCH_ELEMS // max(P, 1))

            # Pixel offsets within a box of this shape, relative to its
            # top-left corner, as flat ray indices.
            dy = np.arange(H, dtype=np.int64)[:, np.newaxis]
            dx = np.arange(W, dtype=np.int64)[np.newaxis, :]
            off = (dy * res + dx).ravel()  # (P,)

            for b0 in range(0, len(gidx), step):
                cidx = gidx[b0:b0 + step]

                base = (py_lo[cidx].astype(np.int64) * res
                        + px_lo[cidx].astype(np.int64))
                flat = base[:, np.newaxis] + off[np.newaxis, :]  # (T, P)
                origins = ray_flat[flat]                         # (T, P, 3)

                # Pre-computed per-triangle data (indexed, not recomputed)
                c_v0 = v0_all[cidx][:, np.newaxis, :]
                c_h = h_all[cidx][:, np.newaxis, :]
                c_edge1 = edge1_all[cidx][:, np.newaxis, :]
                c_edge2 = edge2_all[cidx][:, np.newaxis, :]
                c_inv_a = inv_a_all[cidx][:, np.newaxis]
                c_good = good_all[cidx][:, np.newaxis]

                # Broadcast Möller–Trumbore  (T, P)
                s = origins - c_v0
                u_par = np.sum(s * c_h, axis=2) * c_inv_a
                q = np.cross(s, c_edge1)
                v_par = np.sum(ray_dir * q, axis=2) * c_inv_a
                t_par = np.sum(c_edge2 * q, axis=2) * c_inv_a

                hit = (
                    c_good
                    & (u_par >= 0) & (u_par <= 1)
                    & (v_par >= 0) & ((u_par + v_par) <= 1)
                    & (t_par > EPSILON) & (t_par <= depth_f)
                )
                if not hit.any():
                    continue

                hf = flat[hit]  # flat ray index of every (triangle, hit) pair
                hit_count += np.bincount(hf, minlength=n_rays).astype(np.float32)

                if is_anatomical:
                    hc = np.broadcast_to(
                        color_arr[cidx][:, np.newaxis, :], hit.shape + (3,))[hit]
                    for c in range(3):
                        color_accum[:, c] += np.bincount(
                            hf, weights=hc[:, c], minlength=n_rays,
                        ).astype(np.float32)
                else:
                    ht = np.broadcast_to(
                        tissue_arr[cidx][:, np.newaxis], hit.shape)[hit]
                    value_accum += np.bincount(
                        hf, weights=ht, minlength=n_rays,
                    ).astype(np.float32)
                    np.maximum.at(value_max_a, hf, ht)
                    np.minimum.at(value_min_a, hf, ht)

            if progress_callback and gi % 8 == 0:
                progress_callback(0.35 + 0.65 * gi / max(n_groups, 1))

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
