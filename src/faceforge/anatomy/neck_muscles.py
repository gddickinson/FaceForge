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
from faceforge.anatomy import neck_body_follow as _follow
from faceforge.anatomy import neck_fibre_strain as _strain

# Body follow fractions per lowerAttach type: how much the lower end of the
# muscle follows body motion rather than the head.
#: Re-exported from :mod:`faceforge.anatomy.neck_body_follow`, which owns them.
BODY_FOLLOW_BASE = _follow.BODY_FOLLOW_BASE
BODY_FOLLOW_MAX = _follow.BODY_FOLLOW_MAX


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
        #: The per-muscle body deltas the current vertex buffers were built
        #: from.  ``update`` may only skip work when these still hold.
        self._last_body_deltas: Optional[list[NDArray]] = None
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

        for mesh, node, defn in zip(result.meshes, result.nodes,
                                    result.defs_loaded, strict=True):
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

        # The displacement each muscle's body end follows: its own attachment
        # bones when it names any, the regional anchor otherwise.  A muscle
        # that sits on a bone the thoracic spine does not carry (longus colli
        # originates on T1, which hangs off the cervical chain) must not be
        # dragged by the thoracic anchor.
        body_deltas = self._body_deltas_per_muscle()
        has_body_delta = any(np.linalg.norm(d) > 1e-6 for d in body_deltas)

        # Early-exit only when nothing this pass reads has changed since the
        # positions currently in the buffers were written.  Testing
        # ``has_body_delta`` alone was wrong in one direction that matters:
        # when the body returns to rest the delta goes to zero, which read as
        # "nothing to do" on the very frame the muscles needed to be put back,
        # so a neck bent by a sit-up stayed bent for the rest of the session.
        head_unchanged = (
            self._last_head_quat is not None
            and np.allclose(head_quaternion, self._last_head_quat, atol=1e-6)
        )
        deltas_unchanged = (
            self._last_body_deltas is not None
            and len(self._last_body_deltas) == len(body_deltas)
            and all(np.allclose(a, b, atol=1e-9)
                    for a, b in zip(self._last_body_deltas, body_deltas))
        )
        if head_unchanged and deltas_unchanged:
            return
        self._last_head_quat = head_quaternion.copy()
        self._last_body_deltas = [np.array(d, dtype=np.float64) for d in body_deltas]

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

        for md, body_delta in zip(self._muscles, body_deltas, strict=True):
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
            if np.linalg.norm(body_delta) > 1e-6:
                # Weight: (1 - spine_frac) so body-end verts get full delta
                body_weight = (1.0 - fracs.astype(np.float64))[:, None]
                out_pos += body_weight * body_delta

            # Volume-preserving fiber strain: radial bulging/thinning
            self._apply_fiber_strain(out_pos, out_nrm, md)

            # Bone-pinning: keep lower-end vertices near attachment bones
            self._apply_bone_pinning(out_pos, md)

            md.mesh.geometry.positions[:] = out_pos.astype(np.float32).ravel()
            md.mesh.geometry.normals[:] = out_nrm.astype(np.float32).ravel()
            md.mesh.needs_update = True

    def _compute_body_deltas(self) -> dict[str, NDArray]:
        """Body anchor displacement from rest to current position, per region."""
        return _follow.regional_deltas(self._body_anchor_rest,
                                       self._body_anchor_current)

    def _body_deltas_per_muscle(self) -> list[NDArray]:
        """The displacement each muscle's body end follows, in muscle order.

        A muscle that names its attachment bones follows *those*; the
        regional average is the fallback for the ones that name none.
        """
        regional = self._compute_body_deltas()
        return [_follow.body_delta_for(md, regional, self._bone_registry)
                for md in self._muscles]

    # Overall pin strength (0 = disabled, 1 = hard pin to bone position)
    _PIN_STRENGTH = _follow.PIN_STRENGTH

    def _apply_bone_pinning(
        self,
        out_pos: NDArray[np.float64],
        md: NeckMuscleData,
    ) -> None:
        """Pin lower-end vertices toward their bone attachment positions."""
        _follow.apply_bone_pinning(out_pos, md, self._bone_registry,
                                   self._PIN_STRENGTH)

    # Fibre strain lives in neck_fibre_strain; these keep the system's API.
    _STRAIN_STRENGTH = _strain.STRAIN_STRENGTH
    _STRETCH_CLAMP = _strain.STRETCH_CLAMP

    def _init_fiber_geometry(self, md: NeckMuscleData) -> None:
        """Pre-compute rest-pose fibre axis, centroids and radial offsets."""
        _strain.init_fiber_geometry(md)

    def _apply_fiber_strain(
        self,
        out_pos: NDArray[np.float64],
        out_nrm: NDArray[np.float64],
        md: NeckMuscleData,
    ) -> None:
        """Apply gentle volume-preserving fibre strain after rotation."""
        _strain.apply_fiber_strain(out_pos, out_nrm, md,
                                   self._STRAIN_STRENGTH, self._STRETCH_CLAMP)

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
        self._last_body_deltas = None
