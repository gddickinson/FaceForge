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
from faceforge.body.edge_relaxation import enforce_edge_range
from faceforge.body.skeleton_morph import SkeletonMorph
from faceforge.body.skin_morph import SkinShapeMorph
from faceforge.body.surface_fit import (
    align_to_bp3d, refine_onto_skin, extract_mesh_landmarks, extract_skeleton_landmarks, load_bp3d_skin_mesh,
    surface_skin_refinement,
)
from faceforge.body.surface_projection import (
    build_head_mask, closest_point_on_triangle_batch, closest_points_on_surface,
    extract_edges, laplacian_smooth_displacements, recompute_normals,
    region_constrained_projection, rotate_normals, rotation_between,
)

logger = logging.getLogger(__name__)



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
        self._skeleton_morph = SkeletonMorph(self._bone_scaler)
        self._skin_shape: Optional[SkinShapeMorph] = None
        self._male_positions: Optional[NDArray[np.float32]] = None
        self._female_positions: Optional[NDArray[np.float32]] = None
        self._male_normals: Optional[NDArray[np.float32]] = None
        self._female_normals: Optional[NDArray[np.float32]] = None
        self._body_mesh: Optional[MeshInstance] = None
        self._body_mesh_node: Optional[SceneNode] = None
        self._mesh_indices: Optional[NDArray] = None
        self._bp3d_skin_mesh_cache: dict = {}
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

    def _align_to_bp3d(self, male_geom, female_geom):
        return align_to_bp3d(male_geom, female_geom, self._scale, self._translate_z)

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

    @property
    def skeleton_morph(self) -> SkeletonMorph:
        return self._skeleton_morph

    @property
    def skin_shape(self) -> Optional[SkinShapeMorph]:
        """The soft-tissue sex field, measured from the male/female pair.

        Built on first use from the two warped surface meshes; see
        :mod:`faceforge.body.skin_morph` for what is kept from the difference
        between them and why.
        """
        if self._skin_shape is None and self._male_positions is not None \
                and self._female_positions is not None:
            self._skin_shape = SkinShapeMorph.from_pair(
                self._male_positions, self._female_positions, self._mesh_indices)
        return self._skin_shape

    def scale_skeleton(self, root: SceneNode,
                       joint_positions: dict | None = None,
                       exclude: set[int] | None = None) -> int:
        """Scale the skeleton under ``root`` as an articulated hierarchy.

        Bones are scaled about the joint they hang from and the joints
        themselves are moved, so the skeleton changes proportion without the
        articulations coming apart -- see
        :mod:`faceforge.body.skeleton_morph` for the measurements that
        motivated it.  The joints moving is also what lets the muscles and the
        skin follow: the delta-matrix skinning reads a scaled skeleton the
        same way it reads a pose.

        Returns the number of bone meshes scaled.
        """
        stats = self._skeleton_morph.apply(root, self._gender, joint_positions, exclude)
        return int(stats.get("bones", 0))

    def needs_reregistration(self, old_gender: float) -> bool:
        """Whether a change is large enough to be worth the expensive path."""
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

    # ── Skeleton landmark extraction (see body/surface_fit.py) ───

    def _load_bp3d_skin_mesh(self, assets):
        return load_bp3d_skin_mesh(assets, self._bp3d_skin_mesh_cache)

    def _extract_skeleton_landmarks(self, assets):
        return extract_skeleton_landmarks(assets)

    def _extract_mesh_landmarks(self, pos):
        return extract_mesh_landmarks(pos)

    def _surface_skin_refinement(self, *args, **kwargs):
        return surface_skin_refinement(*args, **kwargs)


    # ── Surface projection methods ───────────────────────────────
    _closest_point_on_triangle_batch = staticmethod(closest_point_on_triangle_batch)
    _closest_points_on_surface = staticmethod(closest_points_on_surface)
    _region_constrained_projection = staticmethod(region_constrained_projection)
    _build_head_mask = staticmethod(build_head_mask)
    _extract_edges = staticmethod(extract_edges)
    _laplacian_smooth_displacements = staticmethod(laplacian_smooth_displacements)
    _rotation_between = staticmethod(rotation_between)
    _rotate_normals = staticmethod(rotate_normals)
    _recompute_normals = staticmethod(recompute_normals)


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
            disp += refine_onto_skin(self, pos, disp, skel_lm, assets)

        logger.info(
            "Warp: head dZ=+%.0f, shoulder dZ=%+.0f, ankle dZ=%+.0f",
            head_shift, shoulder_shift, an_z_s - an_z_m,
        )
        return disp, per_vert_rot
