"""Gender morph system: coordinates body surface morphing and bone scaling."""

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from faceforge.core.mesh import BufferGeometry, MeshInstance, Material
from faceforge.core.scene_graph import SceneNode
from faceforge.core.config_loader import load_config
from faceforge.loaders.asset_manager import AssetManager
from faceforge.body.bone_scaling import BoneScaler

logger = logging.getLogger(__name__)

# ── Surface-based refinement constants ────────────────────────
SURFACE_HEAD_CAP = 25.0        # displacement cap for head region
SURFACE_BODY_CAP = 20.0        # displacement cap for body region (relaxed from 15)
SURFACE_SMOOTH_ITER = 4        # smoothing iterations (reduced from 8)
SURFACE_SMOOTH_STR = 0.3       # smoothing strength (reduced from 0.5)
SURFACE_HEAD_SMOOTH_STR = 0.2  # lighter smoothing for head
SURFACE_K = 16                 # candidate triangles per query point


class GenderMorphSystem:
    """Coordinates body surface morphing and skeletal scaling for gender dimorphism.

    Manages three levels of gender differentiation:

    1. **Body surface mesh** — Matched male/female meshes morphed via direct
       vertex-level lerp.
    2. **Bone scaling** — Per-bone affine scaling of skeleton STL meshes.
    3. **Re-registration signal** — Notifies the caller when bone positions
       changed and soft tissue needs re-registration.

    Usage::

        system = GenderMorphSystem()
        system.load(assets)
        system.set_gender(0.5)  # Halfway between male and female
    """

    def __init__(self):
        self._bone_scaler = BoneScaler()
        self._male_positions: Optional[NDArray[np.float32]] = None
        self._female_positions: Optional[NDArray[np.float32]] = None
        self._male_normals: Optional[NDArray[np.float32]] = None
        self._female_normals: Optional[NDArray[np.float32]] = None
        self._body_mesh: Optional[MeshInstance] = None
        self._body_mesh_node: Optional[SceneNode] = None
        self._mesh_indices: Optional[NDArray] = None
        self._bp3d_skin_mesh_cache: Optional[tuple[NDArray, NDArray, NDArray]] = None
        self._gender: float = 0.0
        self._loaded: bool = False

        # Region-constrained projection caches
        self._mh_region_labels: Optional[NDArray[np.int32]] = None
        self._bp3d_tri_regions: Optional[NDArray[np.int32]] = None
        self._region_kdtrees: Optional[dict] = None
        self._skel_landmarks: Optional[dict[str, NDArray]] = None

        # Alignment transform from config
        self._config = load_config("gender_dimorphism.json")
        align = self._config.get("body_mesh_alignment", {})
        self._scale: float = align.get("scale", 118.3)
        self._translate_z: float = align.get("translate_z", -199.3)

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def body_mesh(self) -> Optional[MeshInstance]:
        return self._body_mesh

    @property
    def body_mesh_node(self) -> Optional[SceneNode]:
        return self._body_mesh_node

    @property
    def gender(self) -> float:
        return self._gender

    @property
    def bone_scaler(self) -> BoneScaler:
        return self._bone_scaler

    @property
    def mh_region_labels(self) -> Optional[NDArray]:
        """Per-vertex region labels for the MH body mesh, or None."""
        return self._mh_region_labels

    @mh_region_labels.setter
    def mh_region_labels(self, value: NDArray) -> None:
        self._mh_region_labels = value
        self._region_kdtrees = None  # Invalidate cache

    @property
    def skel_landmarks(self) -> Optional[dict]:
        """Cached skeleton landmarks from last warp computation, or None."""
        return self._skel_landmarks

    def load(self, assets: AssetManager) -> Optional[SceneNode]:
        """Load male and female body meshes.

        Returns the SceneNode containing the body surface mesh, or None
        if loading fails.
        """
        try:
            male_geom, female_geom = assets.load_body_mesh()
        except Exception as e:
            logger.warning("Failed to load body meshes: %s", e)
            return None

        # Store mesh indices for edge extraction (used by surface refinement)
        self._mesh_indices = male_geom.indices.copy() if male_geom.indices is not None else None

        # Align both meshes to BP3D coordinate space
        male_pos, female_pos, male_norms, female_norms = self._align_to_bp3d(
            male_geom, female_geom
        )

        # Warp mesh to match skeleton pose (arms/legs alignment)
        male_pos, female_pos, male_norms, female_norms = self._warp_to_skeleton(
            male_pos, female_pos, male_norms, female_norms, assets
        )

        # Write warped positions back to male geometry (used for MeshInstance)
        male_geom.positions = male_pos.reshape(-1).astype(np.float32)
        male_geom.normals = male_norms.reshape(-1).astype(np.float32)

        self._male_positions = male_pos
        self._female_positions = female_pos
        self._male_normals = male_norms
        self._female_normals = female_norms

        # Use male geometry as the base MeshInstance (at gender=0 we show male)
        self._body_mesh = MeshInstance(
            name="body_surface",
            geometry=male_geom,
            material=Material(
                color=(0.88, 0.72, 0.60),
                opacity=0.45,
                render_mode="solid",
            ),
        )
        self._body_mesh.store_rest_pose()

        # Create SceneNode
        self._body_mesh_node = SceneNode(name="bodySurfaceMesh")
        self._body_mesh_node.mesh = self._body_mesh

        self._loaded = True
        logger.info(
            "Body surface mesh loaded: %d verts, %d tris",
            male_geom.vertex_count,
            len(male_geom.indices) // 3 if male_geom.indices is not None else 0,
        )
        return self._body_mesh_node

    def _align_to_bp3d(
        self, male_geom: BufferGeometry, female_geom: BufferGeometry
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
        male_pos *= self._scale
        female_pos *= self._scale

        # Align head tops (max Z) then shift so male head at Z=0
        male_head_z = male_pos[:, 2].max()
        female_head_z = female_pos[:, 2].max()
        female_pos[:, 2] += (male_head_z - female_head_z)

        z_offset = -male_head_z + self._translate_z
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

    def set_gender(self, value: float) -> None:
        """Set the gender value and morph the body surface mesh.

        Parameters
        ----------
        value : float
            0.0 = male, 1.0 = female.
        """
        self._gender = max(0.0, min(1.0, value))
        self._morph_body_surface()

    def _morph_body_surface(self) -> None:
        """Lerp body surface mesh between male and female shapes."""
        if not self._loaded or self._body_mesh is None:
            return

        g = self._gender
        morphed = self._male_positions * (1.0 - g) + self._female_positions * g
        self._body_mesh.geometry.positions = morphed.reshape(-1).astype(np.float32)

        # Lerp and renormalize normals
        norms = self._male_normals * (1.0 - g) + self._female_normals * g
        lengths = np.linalg.norm(norms, axis=1, keepdims=True)
        norms /= np.maximum(lengths, 1e-8)
        self._body_mesh.geometry.normals = norms.reshape(-1).astype(np.float32)

        self._body_mesh.needs_update = True

    def scale_skeleton(
        self,
        bone_meshes: list[tuple[str, MeshInstance]],
    ) -> int:
        """Apply gender scaling to all skeleton bone meshes.

        Parameters
        ----------
        bone_meshes : list of (bone_name, mesh) tuples
            All skeleton meshes to scale.

        Returns
        -------
        int
            Number of meshes that were scaled.
        """
        return self._bone_scaler.apply_to_all(bone_meshes, self._gender)

    def needs_reregistration(self, old_gender: float) -> bool:
        """Check if gender change is significant enough to need re-registration.

        Small changes (< 0.01) during slider drag don't need full re-registration.
        """
        return abs(self._gender - old_gender) > 0.01

    # ── Skeleton warp ────────────────────────────────────────────────
    #
    # Multi-phase warp aligning the body mesh to the BP3D skeleton:
    #   Phase 1: Piecewise Z-remap (raises torso/head, adjusts leg height)
    #   Phase 2: Arm rotation (aligns arm direction with skeleton)
    #   Phase 3: Surface skin refinement (projects onto BP3D skin triangle mesh)

    def _warp_to_skeleton(
        self,
        male_pos: NDArray,
        female_pos: NDArray,
        male_norms: NDArray,
        female_norms: NDArray,
        assets: AssetManager,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Warp mesh vertices to match BP3D skeleton pose."""
        skel_lm = self._extract_skeleton_landmarks(assets)
        if skel_lm is None:
            return male_pos, female_pos, male_norms, female_norms
        self._skel_landmarks = skel_lm

        mesh_lm = self._extract_mesh_landmarks(male_pos)

        # Compute per-vertex displacement and rotation from the male mesh
        disp, per_vert_rot = self._compute_warp(
            male_pos, mesh_lm, skel_lm, assets=assets,
        )

        # Apply displacement to both meshes (same topology, same warp)
        male_pos = (male_pos.astype(np.float64) + disp).astype(np.float32)
        female_pos = (female_pos.astype(np.float64) + disp).astype(np.float32)

        # Recompute normals from mesh faces for better accuracy after NN warp
        if self._mesh_indices is not None:
            male_norms = self._recompute_normals(male_pos, self._mesh_indices)
            female_norms = self._recompute_normals(female_pos, self._mesh_indices)
        else:
            male_norms = self._rotate_normals(male_norms, per_vert_rot)
            female_norms = self._rotate_normals(female_norms, per_vert_rot)

        return male_pos, female_pos, male_norms, female_norms

    # ── Skeleton landmark extraction ─────────────────────────────

    def _load_bone_endpoints(
        self, assets: AssetManager, fma_id: str, frac: float = 0.15,
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

    # FMA IDs: R Humerus=23130, L=23131; R Radius=23464, L=23465
    #          R Femur=24474, L=24475; R Tibia=24477, L=24478

    def _extract_skeleton_landmarks(
        self, assets: AssetManager,
    ) -> dict[str, NDArray] | None:
        """Extract joint landmarks from skeleton bone STLs."""
        result: dict[str, NDArray] = {}

        for side_char, fma_offset in (("R", 0), ("L", 1)):
            hum = self._load_bone_endpoints(assets, f"FMA{23130 + fma_offset}")
            rad = self._load_bone_endpoints(assets, f"FMA{23464 + fma_offset}")
            if hum is None:
                logger.warning("Cannot load humerus — skipping warp")
                return None
            result[f"shoulder_{side_char}"] = hum[0]
            result[f"elbow_{side_char}"] = hum[1]
            if rad is not None:
                result[f"wrist_{side_char}"] = rad[1]

            fem = self._load_bone_endpoints(assets, f"FMA{24474 + fma_offset}")
            tib = self._load_bone_endpoints(assets, f"FMA{24477 + fma_offset}")
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

    def _load_bp3d_skin_mesh(
        self, assets: AssetManager,
    ) -> Optional[tuple[NDArray, NDArray, NDArray]]:
        """Load BP3D skin STL (FMA7163) and return full mesh data.

        Returns (positions, normals, tri_indices) in skull coords, or None.
        Caches the result for reuse.
        """
        if self._bp3d_skin_mesh_cache is not None:
            return self._bp3d_skin_mesh_cache

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

        self._bp3d_skin_mesh_cache = (pos, nrm, tri_indices)
        logger.info(
            "Loaded BP3D skin mesh: %d vertices, %d triangles",
            len(pos), len(tri_indices),
        )
        return self._bp3d_skin_mesh_cache

    # ── Surface projection methods ───────────────────────────────

    @staticmethod
    def _closest_point_on_triangle_batch(
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

    @staticmethod
    def _closest_points_on_surface(
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
        (closest_pts, sq_dists) : (Q, 3) float64, (Q,) float64
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

        for ci in range(k_actual):
            tri_idx = cand_idx[:, ci]  # (Q,)
            A = mesh_pos[mesh_tris[tri_idx, 0]]
            B = mesh_pos[mesh_tris[tri_idx, 1]]
            C = mesh_pos[mesh_tris[tri_idx, 2]]

            cp = GenderMorphSystem._closest_point_on_triangle_batch(
                query_pts, A, B, C,
            )
            diff = cp - query_pts
            sq = np.sum(diff * diff, axis=1)

            better = sq < best_sq
            if better.any():
                best_pts[better] = cp[better]
                best_sq[better] = sq[better]

        return best_pts, best_sq

    @staticmethod
    def _region_constrained_projection(
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
        (closest_pts, sq_dists) : (Q, 3) float64, (Q,) float64
        """
        Q = len(query_pts)
        best_pts = np.empty((Q, 3), dtype=np.float64)
        best_sq = np.full(Q, np.inf, dtype=np.float64)

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

            for ci in range(k_actual):
                local_idx = local_cand[:, ci]
                tri_idx = global_tri_idx[local_idx]
                A = mesh_pos[mesh_tris[tri_idx, 0]]
                B = mesh_pos[mesh_tris[tri_idx, 1]]
                C = mesh_pos[mesh_tris[tri_idx, 2]]

                cp = GenderMorphSystem._closest_point_on_triangle_batch(
                    pts, A, B, C,
                )
                diff = cp - pts
                sq = np.sum(diff * diff, axis=1)

                better = sq < sub_sq
                if better.any():
                    sub_best[better] = cp[better]
                    sub_sq[better] = sq[better]

            best_pts[v_idx] = sub_best
            best_sq[v_idx] = sub_sq

        # Fallback: use global KDTree for vertices in regions with no BP3D tris
        if fallback_indices:
            fb_idx = np.concatenate(fallback_indices)
            fb_pts = query_pts[fb_idx]
            fb_closest, fb_sq = GenderMorphSystem._closest_points_on_surface(
                fb_pts, mesh_pos, mesh_tris, k=k,
            )
            best_pts[fb_idx] = fb_closest
            best_sq[fb_idx] = fb_sq

        return best_pts, best_sq

    @staticmethod
    def _build_head_mask(
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

    def _surface_skin_refinement(
        self,
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

        # 1. Surface projection — region-constrained if available
        if mh_regions is not None and region_kdtrees is not None:
            closest_pts, _ = self._region_constrained_projection(
                coarse_pos, skin_pos, skin_tris,
                mh_regions, region_kdtrees, k=SURFACE_K,
            )
        else:
            closest_pts, _ = self._closest_points_on_surface(
                coarse_pos, skin_pos, skin_tris, k=SURFACE_K,
            )
        disp = closest_pts - coarse_pos

        # 2. Detect head region
        head_mask, head_blend = self._build_head_mask(coarse_pos, skel_lm)

        # 3. Region-adaptive soft cap
        per_vertex_cap = np.full(V, SURFACE_BODY_CAP, dtype=np.float64)
        per_vertex_cap += head_blend * (SURFACE_HEAD_CAP - SURFACE_BODY_CAP)

        mags = np.linalg.norm(disp, axis=1)
        too_far = mags > per_vertex_cap
        if too_far.any():
            disp[too_far] *= (per_vertex_cap[too_far] / mags[too_far])[:, np.newaxis]

        # 4. Laplacian smooth with reduced parameters
        if indices is not None:
            edges = self._extract_edges(indices)
            # Blend smoothing strength: head gets lighter smoothing
            strength = SURFACE_SMOOTH_STR - head_blend * (SURFACE_SMOOTH_STR - SURFACE_HEAD_SMOOTH_STR)
            # Use per-vertex strength via multiple scalar passes with blended result
            # For efficiency, do a single pass at the average strength
            avg_strength = float(np.mean(strength))
            disp = self._laplacian_smooth_displacements(
                edges, disp,
                iterations=SURFACE_SMOOTH_ITER,
                strength=avg_strength,
            )

        return disp

    # ── Mesh landmark extraction ─────────────────────────────────

    def _extract_mesh_landmarks(self, pos: NDArray) -> dict[str, NDArray]:
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
            ankle_z = self._find_ankle_z(pos, x_sign)
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

    @staticmethod
    def _find_ankle_z(pos: NDArray, x_sign: int) -> float:
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

    @staticmethod
    def _extract_edges(indices: NDArray) -> NDArray:
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

    @staticmethod
    def _laplacian_smooth_displacements(
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

    # ── Warp computation ─────────────────────────────────────────

    def _compute_warp(
        self,
        pos: NDArray,
        mesh_lm: dict[str, NDArray],
        skel_lm: dict[str, NDArray],
        assets: Optional[AssetManager] = None,
    ) -> tuple[NDArray, NDArray]:
        """Compute per-vertex displacement + rotation matrix.

        Returns (displacements (V,3), per_vert_rot (V,3,3)).
        """
        V = len(pos)
        disp = np.zeros((V, 3), dtype=np.float64)
        per_vert_rot = np.tile(np.eye(3, dtype=np.float64), (V, 1, 1))

        z = pos[:, 2].astype(np.float64)
        x = pos[:, 0].astype(np.float64)

        # Average R/L skeleton landmarks for symmetric Z-keyframes
        def _avg(lm, key, axis=2):
            r = lm.get(f"{key}_R")
            l = lm.get(f"{key}_L")
            if r is not None and l is not None:
                return (float(r[axis]) + float(l[axis])) / 2
            if r is not None:
                return float(r[axis])
            return float(l[axis]) if l is not None else 0.0

        sh_z_m = _avg(mesh_lm, "shoulder")
        sh_z_s = _avg(skel_lm, "shoulder")
        hi_z_m = _avg(mesh_lm, "hip")
        hi_z_s = _avg(skel_lm, "hip")
        kn_z_m = _avg(mesh_lm, "knee")
        kn_z_s = _avg(skel_lm, "knee")
        an_z_m = _avg(mesh_lm, "ankle")
        an_z_s = _avg(skel_lm, "ankle")

        head_top_z = float(z.max())
        foot_bot_z = float(z.min())

        # Head shift: at minimum matches the shoulder shift so the head is
        # not compressed.  Add +5 so the cranium top aligns with the skull
        # (skull cranium top ≈ Z=27.6).
        shoulder_shift = sh_z_s - sh_z_m
        head_shift = shoulder_shift + 5.0

        # ── Phase 1: Piecewise Z-remap ──────────────────────────
        # Keyframes ordered low-Z → high-Z for np.interp
        kf_z = np.array([
            foot_bot_z,
            an_z_m,
            kn_z_m,
            hi_z_m,
            sh_z_m,
            head_top_z,
        ])
        kf_dz = np.array([
            an_z_s - an_z_m,     # foot bottom follows ankle shift
            an_z_s - an_z_m,     # ankle
            kn_z_s - kn_z_m,     # knee
            hi_z_s - hi_z_m,     # hip
            sh_z_s - sh_z_m,     # shoulder
            head_shift,          # head top
        ])

        z_shift = np.interp(z, kf_z, kf_dz)
        disp[:, 2] = z_shift

        # ── Phase 2: Arm rotation ───────────────────────────────
        # For arm vertices, REPLACE the Z-shift with a proper rotation
        # that maps mesh arm direction to skeleton arm direction.
        for side_char, x_sign in (("R", 1), ("L", -1)):
            sh = mesh_lm.get(f"shoulder_{side_char}")
            el = mesh_lm.get(f"elbow_{side_char}")
            wr = mesh_lm.get(f"wrist_{side_char}")
            s_sh = skel_lm.get(f"shoulder_{side_char}")
            s_el = skel_lm.get(f"elbow_{side_char}")
            s_wr = skel_lm.get(f"wrist_{side_char}")
            if any(v is None for v in (sh, el, wr, s_sh, s_el, s_wr)):
                continue

            sh_z_val = float(sh[2])
            wr_z_val = float(wr[2])
            lateral = x * x_sign

            # Identify arm vertices (lateral, between shoulder and wrist Z)
            arm_core = (lateral > 14) & (z <= sh_z_val + 8) & (z >= wr_z_val - 5)
            if not arm_core.any():
                continue

            # Blend weight: 0 = use Z-shift, 1 = use arm rotation
            arm_blend = np.zeros(V, dtype=np.float64)
            arm_blend[arm_core] = np.clip((lateral[arm_core] - 14) / 5.0, 0, 1)

            # Taper at Z boundaries
            above_sh = arm_core & (z > sh_z_val)
            below_wr = arm_core & (z < wr_z_val)
            if above_sh.any():
                arm_blend[above_sh] *= np.clip(
                    (sh_z_val + 8 - z[above_sh]) / 8.0, 0, 1,
                )
            if below_wr.any():
                arm_blend[below_wr] *= np.clip(
                    (z[below_wr] - (wr_z_val - 5)) / 5.0, 0, 1,
                )

            has_blend = arm_blend > 0
            if not has_blend.any():
                continue

            # Compute rotation matrices
            sh64, el64, wr64 = (
                sh.astype(np.float64),
                el.astype(np.float64),
                wr.astype(np.float64),
            )
            R_ua = self._rotation_between(el64 - sh64, s_el - s_sh)
            warped_elbow = R_ua @ (el64 - sh64) + s_sh
            R_fa = self._rotation_between(wr64 - el64, s_wr - warped_elbow)

            el_z_val = float(el[2])
            arm_pts = pos[has_blend].astype(np.float64)
            arm_z_local = arm_pts[:, 2]

            arm_disp = np.zeros_like(arm_pts)
            arm_rot = np.tile(np.eye(3, dtype=np.float64), (len(arm_pts), 1, 1))

            # Upper arm: rotate around mesh shoulder → skeleton shoulder
            ua = arm_z_local > el_z_val
            if ua.any():
                rotated = (arm_pts[ua] - sh64) @ R_ua.T + s_sh
                arm_disp[ua] = rotated - arm_pts[ua]
                arm_rot[ua] = R_ua

            # Forearm + hand: rotate around mesh elbow → warped elbow
            fa = ~ua
            if fa.any():
                rotated = (arm_pts[fa] - el64) @ R_fa.T + warped_elbow
                arm_disp[fa] = rotated - arm_pts[fa]
                arm_rot[fa] = R_fa

            # Blend between spine Z-shift and arm rotation
            blend = arm_blend[has_blend, np.newaxis]
            disp[has_blend] = blend * arm_disp + (1 - blend) * disp[has_blend]

            blend_3d = arm_blend[has_blend, np.newaxis, np.newaxis]
            per_vert_rot[has_blend] = (
                blend_3d * arm_rot
                + (1 - blend_3d) * per_vert_rot[has_blend]
            )

        # ── Phase 3: Surface skin refinement ──────────────────────
        if assets is not None:
            skin_data = self._load_bp3d_skin_mesh(assets)
            if skin_data is not None:
                try:
                    coarse_warped = pos.astype(np.float64) + disp
                    skin_pos, skin_normals, skin_tris = skin_data

                    # Lazy-compute region labels and KDTrees
                    mh_regions = None
                    region_trees = None
                    try:
                        from faceforge.body.region_labels import (
                            segment_mh_mesh, segment_bp3d_skin,
                            build_region_kdtrees,
                            load_region_overrides, apply_region_overrides,
                        )
                        if self._mh_region_labels is None:
                            self._mh_region_labels = segment_mh_mesh(coarse_warped, skel_lm)
                            overrides = load_region_overrides()
                            if overrides.get("mh_body"):
                                apply_region_overrides(self._mh_region_labels, overrides["mh_body"])
                            logger.info("MH region labels computed: %d vertices", len(self._mh_region_labels))
                        if self._bp3d_tri_regions is None:
                            self._bp3d_tri_regions = segment_bp3d_skin(skin_pos, skin_tris, skel_lm)
                            overrides = load_region_overrides()
                            if overrides.get("bp3d_skin"):
                                apply_region_overrides(self._bp3d_tri_regions, overrides["bp3d_skin"])
                            logger.info("BP3D tri regions computed: %d triangles", len(self._bp3d_tri_regions))
                        if self._region_kdtrees is None:
                            self._region_kdtrees = build_region_kdtrees(
                                skin_pos, skin_tris, self._bp3d_tri_regions,
                            )
                        mh_regions = self._mh_region_labels
                        region_trees = self._region_kdtrees
                    except Exception as exc:
                        logger.warning("Region segmentation failed, using global KDTree: %s", exc)

                    surface_disp = self._surface_skin_refinement(
                        coarse_warped, self._mesh_indices,
                        skin_pos, skin_normals, skin_tris, skel_lm,
                        mh_regions=mh_regions,
                        region_kdtrees=region_trees,
                    )
                    disp += surface_disp
                    logger.info(
                        "Surface skin refinement: median=%.1f, max=%.1f",
                        float(np.median(np.linalg.norm(surface_disp, axis=1))),
                        float(np.max(np.linalg.norm(surface_disp, axis=1))),
                    )
                except ImportError:
                    logger.warning("scipy not available — skipping surface refinement")

        logger.info(
            "Warp: head dZ=+%.0f, shoulder dZ=%+.0f, ankle dZ=%+.0f",
            head_shift, shoulder_shift, an_z_s - an_z_m,
        )
        return disp, per_vert_rot

    # ── Utilities ────────────────────────────────────────────────

    @staticmethod
    def _rotation_between(v_from: NDArray, v_to: NDArray) -> NDArray:
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

    @staticmethod
    def _rotate_normals(norms: NDArray, per_vert_rot: NDArray) -> NDArray:
        """Apply per-vertex rotation to normals (vectorized)."""
        result = np.einsum(
            "ijk,ik->ij", per_vert_rot, norms.astype(np.float64),
        )
        lengths = np.linalg.norm(result, axis=1, keepdims=True)
        result /= np.maximum(lengths, 1e-8)
        return result.astype(np.float32)

    @staticmethod
    def _recompute_normals(pos: NDArray, indices: NDArray) -> NDArray:
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
