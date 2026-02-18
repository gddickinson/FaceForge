"""22 STL jaw muscles with jaw-angle deformation.

Each jaw muscle mesh is loaded from BodyParts3D STL files and deformed
per-frame based on the current jaw opening angle.  Vertices are assigned
a per-vertex jaw weight that determines how much they follow the jaw
rotation (vs. staying fixed to the skull).

Muscles are divided into closers (masseters, temporalis, pterygoids) and
openers (suprahyoids).  Closers stretch when the jaw opens; openers contract.

This module has ZERO GL imports; all vertex math is done with NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from faceforge.core.math_utils import (
    Vec3, Quat, vec3,
    quat_from_axis_angle, quat_rotate_vec3, quat_identity,
)
from faceforge.core.mesh import MeshInstance, BufferGeometry
from faceforge.core.material import Material
from faceforge.core.scene_graph import SceneNode
from faceforge.loaders.stl_batch_loader import CoordinateTransform, load_stl_batch, STLBatchResult
from faceforge.constants import JAW_PIVOT, get_jaw_pivot


@dataclass
class JawMuscleData:
    """Per-muscle runtime data for jaw deformation."""
    mesh: MeshInstance
    node: SceneNode
    defn: dict
    rest_positions: NDArray[np.float32]   # Flat copy of rest pose
    rest_normals: NDArray[np.float32]     # Flat copy of rest normals
    jaw_weights: NDArray[np.float32]      # Per-vertex weight [0,1]
    vert_count: int


class JawMuscleSystem:
    """Manages 22 STL jaw muscles with jaw-angle deformation.

    Parameters
    ----------
    jaw_muscle_defs:
        List of muscle definition dicts loaded from ``jaw_muscles.json``.
        Each dict has: ``name``, ``stl``, ``type`` (closer/opener),
        ``color``, ``jawAxis``, ``jawThresh``.
    transform:
        BP3D-to-skull coordinate transform.
    """

    def __init__(
        self,
        jaw_muscle_defs: list[dict],
        transform: Optional[CoordinateTransform] = None,
        jaw_pivot: tuple[float, float, float] | None = None,
    ) -> None:
        self._defs = jaw_muscle_defs
        self._transform = transform or CoordinateTransform()
        self._muscles: list[JawMuscleData] = []
        self._group: Optional[SceneNode] = None
        self._last_jaw_angle: Optional[float] = None
        self._jaw_pivot = vec3(*(jaw_pivot or get_jaw_pivot()))

    def set_jaw_pivot(self, x: float, y: float, z: float) -> None:
        """Update the jaw pivot position (e.g. after BP3D skull load)."""
        self._jaw_pivot = vec3(x, y, z)
        self._last_jaw_angle = None  # Force re-deform

    @property
    def group(self) -> Optional[SceneNode]:
        return self._group

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, stl_dir=None) -> SceneNode:
        """Load all 22 jaw muscle STL meshes.

        Returns the SceneNode group containing all jaw muscle nodes.
        """
        from faceforge.constants import STL_DIR
        kwargs = {}
        if stl_dir is not None:
            kwargs["stl_dir"] = stl_dir

        result: STLBatchResult = load_stl_batch(
            self._defs,
            label="jaw_muscles",
            transform=self._transform,
            indexed=True,
            **kwargs,
        )

        self._group = result.group

        # Build per-muscle runtime data
        for mesh, node, defn in zip(result.meshes, result.nodes, self._defs):
            rest_pos = mesh.geometry.positions.copy()
            rest_nrm = mesh.geometry.normals.copy()
            vert_count = mesh.geometry.vertex_count

            jaw_weights = self._compute_jaw_weights(
                rest_pos, vert_count, defn
            )

            self._muscles.append(JawMuscleData(
                mesh=mesh,
                node=node,
                defn=defn,
                rest_positions=rest_pos,
                rest_normals=rest_nrm,
                jaw_weights=jaw_weights,
                vert_count=vert_count,
            ))

        return self._group

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(self, jaw_angle: float) -> None:
        """Deform jaw muscles based on jaw opening angle (radians).

        Vectorized: rotates all vertices at once using NumPy broadcasting.
        """
        # Early-exit
        if self._last_jaw_angle is not None and abs(jaw_angle - self._last_jaw_angle) < 1e-6:
            return
        self._last_jaw_angle = jaw_angle

        if abs(jaw_angle) < 1e-6:
            for md in self._muscles:
                md.mesh.geometry.positions[:] = md.rest_positions
                md.mesh.needs_update = True
            return

        # Build 3x3 rotation matrix for X-axis rotation (faster than per-vertex quat)
        c, s = float(np.cos(jaw_angle)), float(np.sin(jaw_angle))
        rot = np.array([
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ], dtype=np.float64)

        pivot = self._jaw_pivot.astype(np.float64)

        for md in self._muscles:
            rest = md.rest_positions.reshape(-1, 3).astype(np.float64)
            weights = md.jaw_weights  # (N,)

            # Vectorized rotation: rel @ rot.T + pivot
            rel = rest - pivot  # (N, 3)
            rotated = rel @ rot.T + pivot  # (N, 3)

            # Blend: out = rest + w * (rotated - rest)
            w = weights[:, np.newaxis]  # (N, 1) for broadcasting
            out = rest + w * (rotated - rest)

            md.mesh.geometry.positions[:] = out.ravel().astype(np.float32)
            md.mesh.needs_update = True

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_jaw_weights(
        self,
        positions: NDArray[np.float32],
        vert_count: int,
        defn: dict,
    ) -> NDArray[np.float32]:
        """Compute per-vertex jaw-follow weight for a single muscle.

        Weight is based on the vertex position along the jaw axis relative
        to the jaw threshold.  Vertices below the threshold (closer to the
        jaw insertion) get higher weight.

        For ``jawAxis="y"``, the weight is based on the Y coordinate
        relative to the jaw pivot Y.
        """
        weights = np.zeros(vert_count, dtype=np.float32)
        pos = positions.reshape(-1, 3)

        axis = defn.get("jawAxis", "y")
        thresh = defn.get("jawThresh", 0.45)
        pivot_val = self._jaw_pivot[1]  # Y component of jaw pivot

        if axis == "y":
            axis_idx = 1
        elif axis == "x":
            axis_idx = 0
        else:
            axis_idx = 2

        # Compute bounding range along the axis
        axis_vals = pos[:, axis_idx]
        if vert_count == 0:
            return weights

        vmin = float(axis_vals.min())
        vmax = float(axis_vals.max())
        extent = vmax - vmin
        if extent < 1e-6:
            return weights

        # Vectorized: normalized position along muscle extent
        t = (axis_vals - vmin) / extent
        if axis_idx == 1:
            t = 1.0 - t  # Y axis: lower Y = closer to jaw
        # Weight ramps from 0 at threshold to 1 past it
        mask = t > thresh
        weights[mask] = np.clip((t[mask] - thresh) / (1.0 - thresh), 0.0, 1.0)

        return weights

    @property
    def muscle_data(self) -> list[JawMuscleData]:
        """Access to per-muscle runtime data (for constraint solver etc.)."""
        return self._muscles
