"""36 STL neck muscles with head-follow deformation.

Neck muscles are parented to ``bodyRoot`` (NOT ``skullGroup``) so they
stay in a stable world reference frame.  Per-frame deformation uses
slerp between identity and the head quaternion, weighted per-vertex by
a spine-fraction that depends on the muscle's upper/lower attachment
levels and the ``lowerAttach`` field (shoulder / ribcage / thoracic).

This module has ZERO GL imports; all vertex math is done with NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from faceforge.core.math_utils import (
    Vec3, Quat, vec3,
    quat_identity,
)
from faceforge.core.mesh import MeshInstance
from faceforge.core.scene_graph import SceneNode
from faceforge.core.state import FaceState, BodyState
from faceforge.loaders.stl_batch_loader import CoordinateTransform, load_stl_batch, STLBatchResult
from faceforge.constants import JAW_PIVOT, get_jaw_pivot

# Body anchor types and their approximate rest-pose world positions.
# These are updated at runtime from actual skeleton pivot world positions.
_ANCHOR_REST: dict[str, NDArray | None] = {
    "shoulder": None,
    "ribcage": None,
    "thoracic": None,
}

# Body follow fractions per lowerAttach type
# These represent how much the lower end of the muscle follows body motion
BODY_FOLLOW_BASE = {
    "shoulder": 0.12,
    "ribcage": 0.30,
    "thoracic": 0.18,
}
BODY_FOLLOW_MAX = {
    "shoulder": 0.25,
    "ribcage": 0.45,
    "thoracic": 0.30,
}


@dataclass
class NeckMuscleData:
    """Per-muscle runtime data for neck deformation."""
    mesh: MeshInstance
    node: SceneNode
    defn: dict
    rest_positions: NDArray[np.float32]
    rest_normals: NDArray[np.float32]
    vert_count: int
    # Per-vertex spine fractions: how much each vertex follows the head
    spine_fracs: NDArray[np.float32]
    upper_frac: float
    lower_frac: float
    body_follow_frac: float
    lower_attach: str  # "shoulder", "ribcage", "thoracic"
    # Fiber geometry for volume-preserving strain (pre-computed at load)
    fiber_axis_rest: NDArray[np.float64] | None = None   # (3,) normalized
    fiber_length_rest: float = 0.0
    centroid_rest: NDArray[np.float64] | None = None      # (3,)
    upper_centroid_rest: NDArray[np.float64] | None = None # (3,)
    lower_centroid_rest: NDArray[np.float64] | None = None # (3,)
    # Per-vertex radial offset from fiber axis (for bulging)
    radial_offsets_rest: NDArray[np.float64] | None = None # (N, 3)
    axial_positions_rest: NDArray[np.float64] | None = None  # (N,) scalar


class NeckMuscleSystem:
    """Manages ~36 STL neck muscles with head-follow deformation.

    The neckMuscleGroup is parented to bodyRoot, NOT skullGroup, so it
    stays in a stable world reference frame.

    Parameters
    ----------
    neck_muscle_defs:
        List of muscle definition dicts loaded from ``neck_muscles.json``.
        Each dict has: ``name``, ``stl``, ``color``, ``upperLevel``,
        ``lowerLevel``, ``headAttachFrac``, ``lowerAttach``.
    transform:
        BP3D-to-skull coordinate transform.
    """

    def __init__(
        self,
        neck_muscle_defs: list[dict],
        transform: Optional[CoordinateTransform] = None,
        jaw_pivot: tuple[float, float, float] | None = None,
    ) -> None:
        self._defs = neck_muscle_defs
        self._transform = transform or CoordinateTransform()
        self._muscles: list[NeckMuscleData] = []
        self._group: Optional[SceneNode] = None
        self._last_head_quat: Optional[Quat] = None
        self._last_body_anchors: Optional[dict[str, NDArray]] = None
        self._head_pivot = vec3(*(jaw_pivot or get_jaw_pivot()))
        # Body anchor rest positions (set by set_body_anchors_rest)
        self._body_anchor_rest: dict[str, NDArray] = {}
        # Current body anchor positions (set each frame)
        self._body_anchor_current: dict[str, NDArray] = {}
        # Per-muscle bone anchor registry (optional, set via set_bone_registry)
        self._bone_registry = None  # BoneAnchorRegistry or None

    @property
    def group(self) -> Optional[SceneNode]:
        return self._group

    @property
    def muscle_data(self) -> list[NeckMuscleData]:
        return self._muscles

    def set_body_anchors_rest(self, anchors: dict[str, NDArray]) -> None:
        """Set rest-pose body anchor positions (shoulder, ribcage, thoracic).

        Called once after skeleton loading to establish the reference frame.
        """
        self._body_anchor_rest = {k: np.asarray(v, dtype=np.float64) for k, v in anchors.items()}

    def set_body_anchors_current(self, anchors: dict[str, NDArray]) -> None:
        """Set current-frame body anchor positions.

        Called each frame after body animation to provide body-delta for
        lower-end vertex tracking.
        """
        self._body_anchor_current = {k: np.asarray(v, dtype=np.float64) for k, v in anchors.items()}

    def set_bone_registry(self, registry) -> None:
        """Attach a :class:`BoneAnchorRegistry` for per-muscle bone pinning.

        When set, ``update()`` will apply bone-pinning constraints that
        keep lower-end vertices near their anatomical attachment bones.
        """
        self._bone_registry = registry

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, stl_dir=None) -> SceneNode:
        """Load all neck muscle STL meshes.

        Returns the SceneNode group (to be parented to bodyRoot).
        """
        from faceforge.constants import STL_DIR
        kwargs = {}
        if stl_dir is not None:
            kwargs["stl_dir"] = stl_dir

        result: STLBatchResult = load_stl_batch(
            self._defs,
            label="neck_muscles",
            transform=self._transform,
            indexed=True,
            **kwargs,
        )

        self._group = result.group

        for mesh, node, defn in zip(result.meshes, result.nodes, self._defs):
            rest_pos = mesh.geometry.positions.copy()
            rest_nrm = mesh.geometry.normals.copy()
            vert_count = mesh.geometry.vertex_count
            lower_attach = defn.get("lowerAttach", "shoulder")

            # Compute spine fractions and body follow
            spine_fracs, upper_frac, lower_frac = self._compute_spine_fractions(
                rest_pos, vert_count, defn
            )
            body_follow_frac = self._compute_body_follow(lower_attach, defn)

            # Remap fractions: [lowerFrac, upperFrac] -> [bodyFollowFrac, upperFrac]
            # This keeps lower ends attached to the skeleton
            frac_range = upper_frac - lower_frac
            if frac_range > 1e-6:
                t = np.clip((spine_fracs - lower_frac) / frac_range, 0.0, 1.0)
                spine_fracs = body_follow_frac + t * (upper_frac - body_follow_frac)

            md = NeckMuscleData(
                mesh=mesh,
                node=node,
                defn=defn,
                rest_positions=rest_pos,
                rest_normals=rest_nrm,
                vert_count=vert_count,
                spine_fracs=spine_fracs,
                upper_frac=upper_frac,
                lower_frac=lower_frac,
                body_follow_frac=body_follow_frac,
                lower_attach=lower_attach,
            )

            # Pre-compute fiber geometry for volume-preserving strain
            self._init_fiber_geometry(md)
            self._muscles.append(md)

        return self._group

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(
        self,
        head_quaternion: Quat,
        head_pivot: Optional[Vec3] = None,
        face_state: Optional[FaceState] = None,
        body_state: Optional[BodyState] = None,
    ) -> None:
        """Deform neck muscles based on head rotation and body motion.

        For each vertex, the rotation is ``slerp(identity, headQ, frac)``
        where ``frac`` is the vertex's spine fraction.  Since
        ``slerp(identity, q, t)`` equals rotation by ``t * angle`` around
        the same axis as ``q``, we use Rodrigues' formula vectorized over
        all vertices at once.

        After head-follow rotation, a body-delta displacement is blended in
        for lower-end vertices to track body skeleton motion (thoracic spine
        flex, shoulder movement).  The blend weight is ``(1 - spine_frac)``
        so that body-end vertices track the skeleton and head-end vertices
        remain unaffected.

        Uses early-exit when head quaternion has not changed.
        """
        if head_pivot is None:
            head_pivot = self._head_pivot

        # Compute body delta per anchor type
        body_deltas = self._compute_body_deltas()
        has_body_delta = any(np.linalg.norm(d) > 1e-6 for d in body_deltas.values())

        # Early-exit only if head quat AND body anchors haven't changed
        head_unchanged = (
            self._last_head_quat is not None
            and np.allclose(head_quaternion, self._last_head_quat, atol=1e-6)
        )
        if head_unchanged and not has_body_delta:
            return
        self._last_head_quat = head_quaternion.copy()

        identity_q = quat_identity()
        is_identity = np.allclose(head_quaternion, identity_q, atol=1e-6)

        if is_identity and not has_body_delta:
            for md in self._muscles:
                md.mesh.geometry.positions[:] = md.rest_positions
                md.mesh.geometry.normals[:] = md.rest_normals
                md.mesh.needs_update = True
            return

        # Decompose head quaternion [x,y,z,w] into axis + angle
        qx, qy, qz, qw = head_quaternion
        sin_half = np.sqrt(qx * qx + qy * qy + qz * qz)
        full_angle = 2.0 * np.arctan2(sin_half, qw)

        no_rotation = abs(full_angle) < 1e-8 or is_identity

        if not no_rotation:
            # Normalized rotation axis
            axis = np.array([qx, qy, qz], dtype=np.float64) / sin_half
            kx, ky, kz = axis
            K = np.array([
                [0.0, -kz, ky],
                [kz, 0.0, -kx],
                [-ky, kx, 0.0],
            ], dtype=np.float64)
            K2 = K @ K

        pivot = head_pivot.astype(np.float64)

        for md in self._muscles:
            rest = md.rest_positions.reshape(-1, 3).astype(np.float64)
            rest_n = md.rest_normals.reshape(-1, 3).astype(np.float64)
            fracs = md.spine_fracs  # (N,) float32

            if no_rotation:
                out_pos = rest.copy()
                out_nrm = rest_n.copy()
            else:
                # Per-vertex angles: theta_i = frac_i * full_angle
                angles = fracs.astype(np.float64) * full_angle
                sin_a = np.sin(angles)
                cos_a = np.cos(angles)

                # Rodrigues rotation around head pivot
                rel = rest - pivot
                Kv = (K @ rel.T).T
                K2v = (K2 @ rel.T).T
                rotated = rel + sin_a[:, None] * Kv + (1.0 - cos_a[:, None]) * K2v
                out_pos = rotated + pivot

                # Normals: same Rodrigues, no pivot offset
                Kn = (K @ rest_n.T).T
                K2n = (K2 @ rest_n.T).T
                out_nrm = rest_n + sin_a[:, None] * Kn + (1.0 - cos_a[:, None]) * K2n

            # Body-delta displacement: blend body motion into lower vertices
            if has_body_delta:
                lower_attach = md.lower_attach
                delta = body_deltas.get(lower_attach)
                if delta is not None and np.linalg.norm(delta) > 1e-6:
                    # Weight: (1 - spine_frac) so body-end verts get full delta
                    body_weight = (1.0 - fracs.astype(np.float64))[:, None]
                    out_pos += body_weight * delta

            # Volume-preserving fiber strain: radial bulging/thinning
            self._apply_fiber_strain(out_pos, out_nrm, md)

            # Bone-pinning: keep lower-end vertices near attachment bones
            self._apply_bone_pinning(out_pos, md)

            md.mesh.geometry.positions[:] = out_pos.astype(np.float32).ravel()
            md.mesh.geometry.normals[:] = out_nrm.astype(np.float32).ravel()
            md.mesh.needs_update = True

    def _compute_body_deltas(self) -> dict[str, NDArray]:
        """Compute body anchor displacement from rest to current position."""
        deltas: dict[str, NDArray] = {}
        for anchor_type in ("shoulder", "ribcage", "thoracic"):
            rest = self._body_anchor_rest.get(anchor_type)
            current = self._body_anchor_current.get(anchor_type)
            if rest is not None and current is not None:
                deltas[anchor_type] = current - rest
            else:
                deltas[anchor_type] = np.zeros(3, dtype=np.float64)
        return deltas

    # ------------------------------------------------------------------
    # Bone-pinning constraint
    # ------------------------------------------------------------------

    # Overall pin strength (0 = disabled, 1 = hard pin to bone position)
    _PIN_STRENGTH = 0.6

    def _apply_bone_pinning(
        self,
        out_pos: NDArray[np.float64],
        md: NeckMuscleData,
    ) -> None:
        """Pin lower-end vertices toward their bone attachment positions.

        Uses per-muscle ``lowerBones`` config to query the bone registry
        for the specific attachment bone positions, rather than using a
        shared global anchor.  Falls back to no-op if no registry or no
        bones are configured.

        Pin strength is strongest at the body end (lowest spine fraction)
        and fades to zero at the skull end, using a quadratic falloff.
        """
        if self._bone_registry is None:
            return

        lower_bones = md.defn.get("lowerBones")
        if not lower_bones:
            return

        muscle_name = md.defn.get("name", "")

        bone_anchor_current = self._bone_registry.get_muscle_anchor_current(
            muscle_name, lower_bones,
        )
        bone_anchor_rest = self._bone_registry.get_muscle_anchor(
            muscle_name, lower_bones,
        )

        if bone_anchor_current is None or bone_anchor_rest is None:
            return

        fracs = md.spine_fracs.astype(np.float64)
        frac_min = float(fracs.min())
        frac_max = float(fracs.max())
        frac_range = frac_max - frac_min

        if frac_range < 1e-6:
            return

        # Pin weight: 1.0 at body end (frac=min), 0.0 at skull end (frac=max)
        pin_weight = 1.0 - np.clip((fracs - frac_min) / frac_range, 0.0, 1.0)
        pin_weight = pin_weight ** 2  # quadratic falloff for smooth transition

        # Bone displacement from rest to current
        bone_delta = bone_anchor_current - bone_anchor_rest

        # Target: each vertex's rest position + bone displacement
        rest = md.rest_positions.reshape(-1, 3).astype(np.float64)
        target = rest + bone_delta

        # Blend toward target: strong at body end, zero at skull end
        blend = pin_weight[:, None] * self._PIN_STRENGTH
        out_pos[:] = out_pos * (1.0 - blend) + target * blend

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_spine_fractions(
        self,
        positions: NDArray[np.float32],
        vert_count: int,
        defn: dict,
    ) -> tuple[NDArray[np.float32], float, float]:
        """Compute per-vertex spine fraction for a neck muscle.

        The fraction is interpolated from ``headAttachFrac`` at the upper
        (skull) end to 0 at the lower (body) end, based on each vertex's
        Y position along the muscle extent.

        Returns
        -------
        tuple
            (spine_fracs array, upper_frac, lower_frac)
        """
        head_attach_frac = defn.get("headAttachFrac", 0.3)
        upper_level = defn.get("upperLevel", 0)
        lower_level = defn.get("lowerLevel", 10)

        # The upper fraction is 1.0 for level 0 (skull), decreasing per level
        # Map levels 0-10 to fractions 1.0-0.0
        upper_frac = max(0.0, 1.0 - upper_level / 10.0)
        lower_frac = max(0.0, 1.0 - lower_level / 10.0)

        fracs = np.zeros(vert_count, dtype=np.float32)
        pos = positions.reshape(-1, 3)

        if vert_count == 0:
            return fracs, upper_frac, lower_frac

        # Y extent
        y_vals = pos[:, 1]
        y_min = float(y_vals.min())
        y_max = float(y_vals.max())
        y_range = y_max - y_min

        if y_range < 1e-6:
            fracs[:] = (upper_frac + lower_frac) / 2.0
            return fracs, upper_frac, lower_frac

        # Vectorized: t in [0,1], interpolate between lower_frac and upper_frac
        t = (y_vals - y_min) / y_range
        fracs[:] = lower_frac + t * (upper_frac - lower_frac)

        return fracs, upper_frac, lower_frac

    def _compute_body_follow(self, lower_attach: str, defn: dict) -> float:
        """Compute body follow fraction for a given lower attachment type.

        If the muscle definition contains a ``bodyFollowOverride`` value,
        that takes precedence over the default for the attachment type.
        """
        override = defn.get("bodyFollowOverride")
        if override is not None:
            return float(override)
        return BODY_FOLLOW_BASE.get(lower_attach, 0.05)

    # ------------------------------------------------------------------
    # Fiber geometry + volume-preserving strain
    # ------------------------------------------------------------------

    def _init_fiber_geometry(self, md: NeckMuscleData) -> None:
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

    # Maximum radial change from fiber strain (conservative to avoid detachment)
    _STRAIN_STRENGTH = 0.5   # blend factor — 50% of full volume-preserving effect
    _STRETCH_CLAMP = (0.75, 1.4)  # tighter clamp range

    def _apply_fiber_strain(
        self,
        out_pos: NDArray[np.float64],
        out_nrm: NDArray[np.float64],
        md: NeckMuscleData,
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
        stretch = np.clip(stretch, self._STRETCH_CLAMP[0], self._STRETCH_CLAMP[1])

        if abs(stretch - 1.0) < 0.01:
            return

        # Volume-preserving radial scale: r' = r / sqrt(stretch)
        # Blended with identity by _STRAIN_STRENGTH for a gentler effect
        full_radial_scale = 1.0 / np.sqrt(stretch)
        radial_scale = 1.0 + (full_radial_scale - 1.0) * self._STRAIN_STRENGTH

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

    # ------------------------------------------------------------------
    # Visibility
    # ------------------------------------------------------------------

    def set_visible(self, visible: bool) -> None:
        """Toggle visibility of all neck muscles."""
        if self._group is not None:
            self._group.visible = visible
        for md in self._muscles:
            md.mesh.visible = visible

    def reset(self) -> None:
        """Reset all muscles to rest pose."""
        for md in self._muscles:
            md.mesh.geometry.positions[:] = md.rest_positions
            md.mesh.geometry.normals[:] = md.rest_normals
            md.mesh.needs_update = True
        self._last_head_quat = None
