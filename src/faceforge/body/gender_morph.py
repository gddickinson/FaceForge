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
from faceforge.body.surface_fit import align_to_bp3d, refine_onto_skin
from faceforge.body.surface_landmarks import (
    extract_mesh_landmarks, extract_skeleton_landmarks, load_bp3d_skin_mesh,
)
from faceforge.body.surface_register import fit_head_to_skull, register_onto
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
        self._bone_points: Optional[NDArray] = None
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

    def load(self, assets: AssetManager,
             bone_points: Optional[NDArray] = None) -> Optional[SceneNode]:
        """Load male and female body meshes.

        ``bone_points`` are vertices sampled from the loaded skeleton, in the
        same coordinates.  Given them, the surface is inflated wherever a bone
        would otherwise poke through it; see
        :func:`faceforge.body.surface_fit.inflate_to_contain`.

        Returns the SceneNode containing the body surface mesh, or None
        if loading fails.
        """
        self._bone_points = None if bone_points is None else np.asarray(
            bone_points, dtype=np.float64)
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
        """Per-vertex displacement taking the surface mesh onto the skeleton.

        Two stages.  The first registers the mesh onto the reference skin --
        the skin of the same cadaver the skeleton came from, so landing on it
        is landing on the skeleton -- with a spline fitted to matched
        landmarks.  The second refines the fit locally, under edge-length
        constraints.

        It replaced a piecewise Z-remap blended against two arm rotations.
        Blending a rotation against a translation is not a rigid motion, and
        it sheared the limbs: the forearm's depth fell from 24.7 to 17.4 while
        its width rose, the foot lost a third of its length, and the occiput
        was sheared flat.

        The rotation array is returned for callers that carry normals through
        the warp; it is the identity here, because the deformation is no
        longer a set of per-vertex rigid motions and the normals are
        recomputed from the warped faces instead.
        """
        V = len(pos)
        per_vert_rot = np.tile(np.eye(3, dtype=np.float64), (V, 1, 1))
        zero = np.zeros((V, 3), dtype=np.float64)
        if assets is None:
            return zero, per_vert_rot
        skin = load_bp3d_skin_mesh(assets, self._bp3d_skin_mesh_cache)
        if skin is None:
            logger.warning("No reference skin: the surface mesh is left where it is")
            return zero, per_vert_rot

        skull = None
        if self._bone_points is not None and len(self._bone_points):
            # The skull: everything the skeleton has above the shoulders.
            sh = skel_lm.get("shoulder_R")
            cut = float(sh[2]) + 12.0 if sh is not None else 0.0
            skull = self._bone_points[self._bone_points[:, 2] > cut]
        disp = register_onto(pos, skel_lm, skin[0], self._mesh_indices)
        disp = disp + refine_onto_skin(self, pos, disp, skel_lm, assets)
        if skull is not None and len(skull):
            sh = skel_lm.get("shoulder_R")
            warped = np.asarray(pos, dtype=np.float64) + disp
            disp = (fit_head_to_skull(warped, skull,
                                      float(sh[2]) if sh is not None else -15.0)
                    - np.asarray(pos, dtype=np.float64))
        logger.info("Surface warp: median %.1f, max %.1f",
                    float(np.median(np.linalg.norm(disp, axis=1))),
                    float(np.max(np.linalg.norm(disp, axis=1))))
        return disp, per_vert_rot
