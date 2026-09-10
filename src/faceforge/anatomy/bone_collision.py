"""Capsule-based bone collision for muscle deformation.

Prevents muscles from passing through bones during deformation by
modelling critical bones as capsule primitives (cylinder + hemisphere
caps) and pushing penetrating vertices to the capsule surface.

Two facts, measured on the real skeleton on 2026-09-10, shape this module:

* Bone meshes are reparented under joint pivots, so their vertex arrays are
  in PIVOT-LOCAL coordinates.  The first version built capsules from those
  arrays as if they were world positions, which put all twelve capsules in a
  cluster near the body origin -- the neck -- and none of them on a bone.
  The phantom humerus capsule pushed ~3,000 vertices of every deep neck
  muscle up to 3 units at the NEUTRAL pose (semispinalis stretch p99 9.3
  against 1.0 with the pass off).  Capsules therefore live in the bone's
  local frame and are placed with the bone's current world matrix every
  frame, which also makes them follow a moving limb.

* The BodyParts3D surfaces overlap their bones at rest.  A vertex inside a
  capsule at rest is not a defect to resolve, it is where the artist put it,
  so each vertex keeps its rest depth as an allowance and only penetration
  BEYOND that is pushed out.  Nothing moves at rest, by construction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from faceforge.anatomy.bone_anchors import BoneAnchorRegistry
from faceforge.core.scene_graph import SceneNode

logger = logging.getLogger(__name__)


@dataclass
class BoneCapsule:
    """A capsule primitive approximating a bone for collision."""
    bone_name: str
    start: NDArray[np.float64]  # (3,) capsule start point, body frame, this frame
    end: NDArray[np.float64]    # (3,) capsule end point, body frame, this frame
    radius: float               # capsule radius
    # Cached axis and length for fast collision
    axis: NDArray[np.float64] = None  # normalised axis (end - start)
    length: float = 0.0
    #: The bone node and the endpoints in ITS frame; ``place`` derives the
    #: body-frame endpoints above from the node's current world matrix.
    node: SceneNode | None = None
    local_start: NDArray[np.float64] | None = None
    local_end: NDArray[np.float64] | None = None
    #: Body-frame endpoints at rest, for the per-vertex rest allowance.
    rest_start: NDArray[np.float64] | None = None
    rest_end: NDArray[np.float64] | None = None

    def place(self, matrix: NDArray[np.float64]) -> None:
        """Set the body-frame endpoints from a 4x4 node matrix."""
        if self.local_start is None:
            return
        m = np.asarray(matrix, dtype=np.float64)
        self.start = m[:3, :3] @ self.local_start + m[:3, 3]
        self.end = m[:3, :3] @ self.local_end + m[:3, 3]
        axis = self.end - self.start
        self.length = float(np.linalg.norm(axis))
        self.axis = axis / self.length if self.length > 1e-9 else np.zeros(3)


# Capsule definitions for critical bones.
_CAPSULE_DEFS = [
    # Clavicles
    {"name": "Right Clavicle", "radius": 2.0},
    {"name": "Left Clavicle", "radius": 2.0},
    # Scapulae (approximated as fat capsule)
    {"name": "Right Scapula", "radius": 3.0},
    {"name": "Left Scapula", "radius": 3.0},
    # Humeri
    {"name": "Right Humerus", "radius": 2.5},
    {"name": "Left Humerus", "radius": 2.5},
    # Upper ribs
    {"name": "Right 1st Rib", "radius": 1.5},
    {"name": "Left 1st Rib", "radius": 1.5},
    {"name": "Right 2nd Rib", "radius": 1.5},
    {"name": "Left 2nd Rib", "radius": 1.5},
    {"name": "Right 3rd Rib", "radius": 1.5},
    {"name": "Left 3rd Rib", "radius": 1.5},
]


def _radial_distance(pos: np.ndarray, start: np.ndarray, end: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Distance of each point from the segment, and the vector to it."""
    axis = end - start
    length = float(np.linalg.norm(axis))
    if length < 1e-9:
        diff = pos - start[np.newaxis, :]
        return np.linalg.norm(diff, axis=1), diff
    axis = axis / length
    t = np.clip((pos - start[np.newaxis, :]) @ axis, 0.0, length)
    closest = start[np.newaxis, :] + t[:, np.newaxis] * axis[np.newaxis, :]
    diff = pos - closest
    return np.linalg.norm(diff, axis=1), diff


@dataclass
class _RestAllowance:
    """Per-mesh: for each capsule, the vertices inside it at rest and how deep."""
    rest: np.ndarray                       # keeps the rest array alive (stable id)
    per_capsule: list[tuple[np.ndarray, np.ndarray] | None] = field(default_factory=list)


class BoneCollisionSystem:
    """Capsule collision system for muscle deformation.

    Builds capsule primitives from bone mesh geometry and resolves
    muscle vertex penetrations per-frame.
    """

    def __init__(self, bone_registry: BoneAnchorRegistry) -> None:
        self._bones = bone_registry
        self._capsules: list[BoneCapsule] = []
        self._allowance: dict[int, _RestAllowance] = {}

    def build_capsules(self) -> int:
        """Create capsule primitives from bone mesh geometry.

        Returns the number of capsules built.  Call after
        ``BoneAnchorRegistry.snapshot_rest_positions()``, at the rest pose:
        the endpoints placed here are remembered as the rest endpoints.
        """
        self._capsules.clear()
        self._allowance.clear()

        for cdef in _CAPSULE_DEFS:
            name = cdef["name"]
            radius = cdef["radius"]

            if not self._bones.has_bone(name):
                continue

            node = self._bones._bone_nodes.get(name)
            if node is None or node.mesh is None:
                continue

            mesh = node.mesh
            positions = mesh.geometry.positions
            if positions is None or len(positions) < 9:
                continue

            pos = positions.reshape(-1, 3).astype(np.float64)
            count = getattr(mesh.geometry, "vertex_count", len(pos))
            pos = pos[:count] if count else pos
            # Capsule axis = direction of maximum extent, in the node's frame.
            centroid = pos.mean(axis=0)
            centered = pos - centroid
            cov = centered.T @ centered / len(pos)
            try:
                eigvals, eigvecs = np.linalg.eigh(cov)
            except np.linalg.LinAlgError:
                continue

            principal = eigvecs[:, -1]  # last column = largest eigenvalue
            projections = centered @ principal
            local_start = centroid + principal * projections.min()
            local_end = centroid + principal * projections.max()
            if float(np.linalg.norm(local_end - local_start)) < 1e-6:
                continue

            capsule = BoneCapsule(
                bone_name=name, start=local_start.copy(), end=local_end.copy(),
                radius=radius, node=node,
                local_start=local_start, local_end=local_end,
            )
            capsule.place(self._node_matrix(node))
            capsule.rest_start = capsule.start.copy()
            capsule.rest_end = capsule.end.copy()
            self._capsules.append(capsule)

        logger.info("Built %d bone collision capsules", len(self._capsules))
        return len(self._capsules)

    def _node_matrix(self, node: SceneNode) -> NDArray[np.float64]:
        """The node's body-frame matrix: world matrix with the scene wrapper cancelled."""
        node.update_world_matrix()
        wm = np.asarray(node.world_matrix, dtype=np.float64)
        cancel = self._bones.frame_cancel
        return wm if cancel is None else cancel @ wm

    def refresh(self) -> None:
        """Place every capsule on its bone for the current frame.

        Call once per frame after the scene graph update and after the
        registry's frame cancel is set; ``resolve_penetrations`` then works
        against bones where they are now, not where they were loaded.
        """
        for capsule in self._capsules:
            if capsule.node is not None:
                capsule.place(self._node_matrix(capsule.node))

    def _rest_allowance(self, rest_positions: np.ndarray) -> _RestAllowance:
        key = id(rest_positions)
        hit = self._allowance.get(key)
        if hit is not None and hit.rest is rest_positions:
            return hit
        rest = np.asarray(rest_positions).reshape(-1, 3).astype(np.float64)
        entry = _RestAllowance(rest=rest_positions)
        for capsule in self._capsules:
            if capsule.rest_start is None:
                entry.per_capsule.append(None)
                continue
            dist, _ = _radial_distance(rest, capsule.rest_start, capsule.rest_end)
            inside = dist < capsule.radius
            if inside.any():
                idx = np.where(inside)[0]
                entry.per_capsule.append((idx, dist[idx]))
            else:
                entry.per_capsule.append(None)
        self._allowance[key] = entry
        return entry

    def resolve_penetrations(
        self,
        positions: np.ndarray,
        rest_positions: np.ndarray,
    ) -> int:
        """Push vertices out of bone capsules, beyond their rest depth only.

        Parameters
        ----------
        positions : flat float32 array
            Current vertex positions (modified in place).
        rest_positions : flat float32 array
            Rest-pose positions.  A vertex that sits inside a capsule at rest
            keeps that depth as its allowance.

        Returns
        -------
        int
            Number of vertices corrected.
        """
        if not self._capsules:
            return 0

        pos = np.asarray(positions).reshape(-1, 3).astype(np.float64)
        V = len(pos)
        allowance = self._rest_allowance(rest_positions)
        total_corrected = 0

        for k, capsule in enumerate(self._capsules):
            dist, diff = _radial_distance(pos, capsule.start, capsule.end)
            allowed = np.full(V, capsule.radius, dtype=np.float64)
            rest_hit = allowance.per_capsule[k] if k < len(allowance.per_capsule) else None
            if rest_hit is not None:
                idx_r, depth_r = rest_hit
                ok = idx_r < V
                allowed[idx_r[ok]] = np.minimum(allowed[idx_r[ok]], depth_r[ok])

            inside = dist < allowed
            # Don't correct vertices that are exactly on the axis (ambiguous direction)
            inside &= dist > 1e-6
            if not inside.any():
                continue

            idx = np.where(inside)[0]
            radial_dir = diff[idx] / dist[idx][:, np.newaxis]
            pos[idx] = (pos[idx] - diff[idx]) + radial_dir * allowed[idx][:, np.newaxis]
            total_corrected += len(idx)

        if total_corrected > 0:
            positions.reshape(-1)[:] = pos.astype(np.float32).ravel()

        return total_corrected

    @property
    def capsule_count(self) -> int:
        return len(self._capsules)
