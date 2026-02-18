"""38 STL expression muscles with AU-driven contraction.

Each expression muscle has an ``auMap`` that maps Action Unit IDs to
contraction weights.  Muscle activation is the maximum weighted AU value
across the map.  Active muscles contract their vertices toward the
muscle centroid along the fiber direction.

This module has ZERO GL imports; all vertex math is done with NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from faceforge.core.mesh import MeshInstance
from faceforge.core.scene_graph import SceneNode
from faceforge.core.state import FaceState, AU_IDS
from faceforge.loaders.stl_batch_loader import CoordinateTransform, load_stl_batch, STLBatchResult


# Maximum contraction displacement (fraction of distance to centroid)
MAX_CONTRACTION = 0.15

# Color tint: activated muscles shift toward red
ACTIVATION_COLOR_TINT = np.array([0.3, -0.1, -0.1], dtype=np.float32)


@dataclass
class ExprMuscleData:
    """Per-muscle runtime data for expression muscle deformation."""
    mesh: MeshInstance
    node: SceneNode
    defn: dict
    au_map: dict[str, float]  # {AU_ID: weight}
    rest_positions: NDArray[np.float32]
    rest_normals: NDArray[np.float32]
    centroid: NDArray[np.float64]  # (3,)
    fiber_dirs: NDArray[np.float32]  # (N, 3) per-vertex direction toward centroid
    fiber_dists: NDArray[np.float32]  # (N,) per-vertex distance to centroid
    vert_count: int
    base_color: tuple[float, float, float]


class ExpressionMuscleSystem:
    """Manages ~38 STL expression muscles with AU-driven contraction.

    Parameters
    ----------
    expr_muscle_defs:
        List of muscle definition dicts loaded from ``expression_muscles.json``.
        Each dict has: ``name``, ``stl``, ``color``, ``auMap``.
    transform:
        BP3D-to-skull coordinate transform.
    """

    def __init__(
        self,
        expr_muscle_defs: list[dict],
        transform: Optional[CoordinateTransform] = None,
    ) -> None:
        self._defs = expr_muscle_defs
        self._transform = transform or CoordinateTransform()
        self._muscles: list[ExprMuscleData] = []
        self._group: Optional[SceneNode] = None
        self._last_au_snapshot: Optional[tuple] = None

    @property
    def group(self) -> Optional[SceneNode]:
        return self._group

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, stl_dir=None) -> SceneNode:
        """Load all expression muscle STL meshes.

        Returns the SceneNode group containing all expression muscle nodes.
        """
        from faceforge.constants import STL_DIR
        kwargs = {}
        if stl_dir is not None:
            kwargs["stl_dir"] = stl_dir

        result: STLBatchResult = load_stl_batch(
            self._defs,
            label="expression_muscles",
            transform=self._transform,
            indexed=True,
            **kwargs,
        )

        self._group = result.group

        for mesh, node, defn in zip(result.meshes, result.nodes, self._defs):
            rest_pos = mesh.geometry.positions.copy()
            rest_nrm = mesh.geometry.normals.copy()
            vert_count = mesh.geometry.vertex_count
            au_map = defn.get("auMap", {})

            # Compute muscle centroid and per-vertex fiber directions
            pos_2d = rest_pos.reshape(-1, 3)
            centroid = pos_2d.mean(axis=0).astype(np.float64)

            # Direction from each vertex toward centroid
            diff = centroid.astype(np.float32) - pos_2d
            dists = np.linalg.norm(diff, axis=1).astype(np.float32)
            safe_dists = np.maximum(dists, 1e-6)
            fiber_dirs = diff / safe_dists[:, np.newaxis]

            # Base color from material
            base_color = mesh.material.color

            self._muscles.append(ExprMuscleData(
                mesh=mesh,
                node=node,
                defn=defn,
                au_map=au_map,
                rest_positions=rest_pos,
                rest_normals=rest_nrm,
                centroid=centroid,
                fiber_dirs=fiber_dirs,
                fiber_dists=dists,
                vert_count=vert_count,
                base_color=base_color,
            ))

        return self._group

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(self, face_state: FaceState) -> None:
        """Deform expression muscles based on current AU values.

        Each muscle's activation is ``max(auMap[au] * faceState.get_au(au))``
        across all mapped AUs.  Activated muscles contract their vertices
        toward the centroid by a fraction of their distance.

        Uses early-exit when no AU values have changed.
        """
        # Build AU snapshot for early-exit
        au_snapshot = tuple(face_state.get_au(au) for au in AU_IDS)
        # Also include blink_amount since some muscles map to "blink"
        blink_val = face_state.blink_amount
        snapshot_key = au_snapshot + (blink_val,)

        if self._last_au_snapshot is not None and snapshot_key == self._last_au_snapshot:
            return
        self._last_au_snapshot = snapshot_key

        for md in self._muscles:
            # Compute muscle activation
            activation = 0.0
            for au_id, weight in md.au_map.items():
                if au_id == "blink":
                    au_val = blink_val
                elif au_id in AU_IDS:
                    au_val = face_state.get_au(au_id)
                else:
                    continue
                activation = max(activation, au_val * weight)

            activation = min(1.0, activation)

            if activation < 1e-4:
                # No activation — restore rest pose
                md.mesh.geometry.positions[:] = md.rest_positions
                md.mesh.needs_update = True
                continue

            # Vectorized contraction toward centroid
            rest = md.rest_positions.reshape(-1, 3)
            contraction = activation * MAX_CONTRACTION
            # disp = fiber_dirs * fiber_dists[:, None] * contraction
            disp = md.fiber_dirs * (md.fiber_dists[:, np.newaxis] * contraction)
            out = rest + disp

            md.mesh.geometry.positions[:] = out.ravel()
            md.mesh.needs_update = True

            # Tint muscle color based on activation
            r = min(1.0, md.base_color[0] + ACTIVATION_COLOR_TINT[0] * activation)
            g = max(0.0, md.base_color[1] + ACTIVATION_COLOR_TINT[1] * activation)
            b = max(0.0, md.base_color[2] + ACTIVATION_COLOR_TINT[2] * activation)
            md.mesh.material.color = (r, g, b)

    def reset(self) -> None:
        """Reset all muscles to rest positions and colors."""
        for md in self._muscles:
            md.mesh.geometry.positions[:] = md.rest_positions
            md.mesh.material.color = md.base_color
            md.mesh.needs_update = True
        self._last_au_snapshot = None

    @property
    def muscle_data(self) -> list[ExprMuscleData]:
        return self._muscles
