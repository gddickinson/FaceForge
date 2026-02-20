"""Registry mapping muscle names to bone SceneNode references.

Each neck muscle specifies ``lowerBones`` in its config — a list of bone
names where the muscle's lower end attaches anatomically.  This registry
resolves those names to live SceneNode references and provides rest-pose
and current-frame world positions for bone-pinning constraints.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from faceforge.core.scene_graph import SceneNode

logger = logging.getLogger(__name__)


class BoneAnchorRegistry:
    """Lightweight map from bone names to SceneNodes + cached positions."""

    def __init__(self) -> None:
        self._bone_nodes: dict[str, SceneNode] = {}
        self._rest_positions: dict[str, NDArray[np.float64]] = {}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def register_bones(self, bone_nodes: dict[str, SceneNode]) -> None:
        """Register a mapping of bone name -> SceneNode.

        Called once after skeleton loading with the result of
        ``_collect_bone_nodes()`` plus any additional nodes
        (thoracic pivots, rib nodes, etc.).
        """
        self._bone_nodes.update(bone_nodes)

    def snapshot_rest_positions(self) -> None:
        """Store current world position of every registered bone as rest.

        For bone nodes whose SceneNode transform is at the origin (common
        for STL-loaded bones where the vertex data is already in world
        coordinates), the bone position is computed from the mesh vertex
        centroid instead.

        Call once after the scene is in rest pose and world matrices
        have been updated.
        """
        for name, node in self._bone_nodes.items():
            self._rest_positions[name] = self._get_bone_position(node)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_muscle_anchor(
        self,
        muscle_name: str,
        lower_bones: list[str],
    ) -> Optional[NDArray[np.float64]]:
        """Return averaged rest-pose world position of the named bones.

        If none of the requested bones are registered, returns ``None``.
        """
        positions = []
        for bone_name in lower_bones:
            pos = self._rest_positions.get(bone_name)
            if pos is not None:
                positions.append(pos)
        if not positions:
            return None
        return np.mean(positions, axis=0).astype(np.float64)

    def get_muscle_anchor_current(
        self,
        muscle_name: str,
        lower_bones: list[str],
    ) -> Optional[NDArray[np.float64]]:
        """Return averaged *current*-frame world position of the named bones.

        Reads live position from the SceneNode graph, so call after
        scene matrices have been updated for this frame.
        """
        positions = []
        for bone_name in lower_bones:
            node = self._bone_nodes.get(bone_name)
            if node is not None:
                positions.append(self._get_bone_position(node))
        if not positions:
            return None
        return np.mean(positions, axis=0).astype(np.float64)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _get_bone_position(node: SceneNode) -> NDArray[np.float64]:
        """Get bone world position, preferring mesh centroid when available.

        STL-loaded bone nodes typically have their vertex data in world
        coordinates with the SceneNode transform at identity.  For these
        nodes, ``get_world_position()`` returns (0,0,0).  We detect this
        and fall back to the mesh vertex centroid transformed by the
        node's world matrix.
        """
        world_pos = np.asarray(node.get_world_position(), dtype=np.float64)

        # If node has a mesh and the transform-based position is at origin,
        # compute position from mesh centroid instead.
        if node.mesh is not None and np.linalg.norm(world_pos) < 1e-6:
            mesh_centroid = node.mesh.geometry.get_bounding_center()
            # Transform centroid by the node's world matrix
            wm = node.world_matrix
            local = np.append(mesh_centroid, 1.0)
            world = wm @ local
            return world[:3].astype(np.float64)

        return world_pos

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def bone_names(self) -> list[str]:
        """Return list of registered bone names."""
        return list(self._bone_nodes.keys())

    def has_bone(self, name: str) -> bool:
        return name in self._bone_nodes
