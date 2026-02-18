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

# Body follow fractions per lowerAttach type
# These represent how much the lower end of the muscle follows body motion
BODY_FOLLOW_BASE = {
    "shoulder": 0.03,
    "ribcage": 0.20,
    "thoracic": 0.10,
}
BODY_FOLLOW_MAX = {
    "shoulder": 0.15,
    "ribcage": 0.30,
    "thoracic": 0.20,
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
        self._head_pivot = vec3(*(jaw_pivot or get_jaw_pivot()))

    @property
    def group(self) -> Optional[SceneNode]:
        return self._group

    @property
    def muscle_data(self) -> list[NeckMuscleData]:
        return self._muscles

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
            body_follow_frac = self._compute_body_follow(lower_attach)

            # Remap fractions: [lowerFrac, upperFrac] -> [bodyFollowFrac, upperFrac]
            # This keeps lower ends attached to the skeleton
            frac_range = upper_frac - lower_frac
            if frac_range > 1e-6:
                t = np.clip((spine_fracs - lower_frac) / frac_range, 0.0, 1.0)
                spine_fracs = body_follow_frac + t * (upper_frac - body_follow_frac)

            self._muscles.append(NeckMuscleData(
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
            ))

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
        """Deform neck muscles based on head rotation.

        For each vertex, the rotation is ``slerp(identity, headQ, frac)``
        where ``frac`` is the vertex's spine fraction.  Since
        ``slerp(identity, q, t)`` equals rotation by ``t * angle`` around
        the same axis as ``q``, we use Rodrigues' formula vectorized over
        all vertices at once.

        Uses early-exit when head quaternion has not changed.
        """
        if head_pivot is None:
            head_pivot = self._head_pivot

        # Early-exit
        if (self._last_head_quat is not None
                and np.allclose(head_quaternion, self._last_head_quat, atol=1e-6)):
            return
        self._last_head_quat = head_quaternion.copy()

        identity_q = quat_identity()
        is_identity = np.allclose(head_quaternion, identity_q, atol=1e-6)

        if is_identity:
            for md in self._muscles:
                md.mesh.geometry.positions[:] = md.rest_positions
                md.mesh.geometry.normals[:] = md.rest_normals
                md.mesh.needs_update = True
            return

        # Decompose head quaternion [x,y,z,w] into axis + angle
        # slerp(identity, q, t) = rotation by t*angle around same axis
        qx, qy, qz, qw = head_quaternion
        sin_half = np.sqrt(qx * qx + qy * qy + qz * qz)
        full_angle = 2.0 * np.arctan2(sin_half, qw)

        if abs(full_angle) < 1e-8:
            for md in self._muscles:
                md.mesh.geometry.positions[:] = md.rest_positions
                md.mesh.geometry.normals[:] = md.rest_normals
                md.mesh.needs_update = True
            return

        # Normalized rotation axis
        axis = np.array([qx, qy, qz], dtype=np.float64) / sin_half

        # Skew-symmetric matrix K for cross product: K @ v = axis x v
        kx, ky, kz = axis
        K = np.array([
            [0.0, -kz, ky],
            [kz, 0.0, -kx],
            [-ky, kx, 0.0],
        ], dtype=np.float64)
        K2 = K @ K  # K squared

        pivot = head_pivot.astype(np.float64)

        for md in self._muscles:
            rest = md.rest_positions.reshape(-1, 3).astype(np.float64)
            rest_n = md.rest_normals.reshape(-1, 3).astype(np.float64)
            fracs = md.spine_fracs  # (N,) float32

            # Per-vertex angles: theta_i = frac_i * full_angle
            angles = fracs.astype(np.float64) * full_angle  # (N,)
            sin_a = np.sin(angles)  # (N,)
            cos_a = np.cos(angles)  # (N,)

            # Rodrigues: rotated = v + sin(θ) * (K @ v.T) + (1-cos(θ)) * (K² @ v.T)
            # Position: relative to pivot
            rel = rest - pivot  # (N, 3)
            Kv = (K @ rel.T).T      # (N, 3)
            K2v = (K2 @ rel.T).T    # (N, 3)
            rotated = rel + sin_a[:, None] * Kv + (1.0 - cos_a[:, None]) * K2v
            out_pos = (rotated + pivot).astype(np.float32)

            # Normals: same Rodrigues, no pivot offset
            Kn = (K @ rest_n.T).T   # (N, 3)
            K2n = (K2 @ rest_n.T).T  # (N, 3)
            out_nrm = (rest_n + sin_a[:, None] * Kn + (1.0 - cos_a[:, None]) * K2n).astype(np.float32)

            md.mesh.geometry.positions[:] = out_pos.ravel()
            md.mesh.geometry.normals[:] = out_nrm.ravel()
            md.mesh.needs_update = True

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

    def _compute_body_follow(self, lower_attach: str) -> float:
        """Compute body follow fraction for a given lower attachment type."""
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
