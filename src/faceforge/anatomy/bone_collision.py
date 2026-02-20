"""Capsule-based bone collision for muscle deformation.

Prevents muscles from passing through bones during deformation by
modelling critical bones as capsule primitives (cylinder + hemisphere
caps) and pushing penetrating vertices to the capsule surface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from faceforge.anatomy.bone_anchors import BoneAnchorRegistry

logger = logging.getLogger(__name__)


@dataclass
class BoneCapsule:
    """A capsule primitive approximating a bone for collision."""
    bone_name: str
    start: NDArray[np.float64]  # (3,) capsule start point
    end: NDArray[np.float64]    # (3,) capsule end point
    radius: float               # capsule radius
    # Cached axis and length for fast collision
    axis: NDArray[np.float64] = None  # normalised axis (end - start)
    length: float = 0.0


# Capsule definitions for critical bones.
# start_bone/end_bone are bone names whose centroids define endpoints.
# For single-bone capsules, the endpoints are computed from the mesh AABB.
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


class BoneCollisionSystem:
    """Capsule collision system for muscle deformation.

    Builds capsule primitives from bone mesh extents and resolves
    muscle vertex penetrations per-frame.
    """

    def __init__(self, bone_registry: BoneAnchorRegistry) -> None:
        self._bones = bone_registry
        self._capsules: list[BoneCapsule] = []

    def build_capsules(self) -> int:
        """Create capsule primitives from bone mesh geometry.

        Returns the number of capsules built.  Call after
        ``BoneAnchorRegistry.snapshot_rest_positions()``.
        """
        self._capsules.clear()

        for cdef in _CAPSULE_DEFS:
            name = cdef["name"]
            radius = cdef["radius"]

            if not self._bones.has_bone(name):
                continue

            # Get bone node and compute capsule from mesh AABB
            node = self._bones._bone_nodes.get(name)
            if node is None or node.mesh is None:
                continue

            mesh = node.mesh
            positions = mesh.geometry.positions
            if positions is None or len(positions) < 9:
                continue

            pos = positions.reshape(-1, 3).astype(np.float64)
            # Use PCA-like approach: capsule axis = direction of maximum extent
            centroid = pos.mean(axis=0)
            # Compute covariance and find principal direction
            centered = pos - centroid
            cov = centered.T @ centered / len(pos)
            try:
                eigvals, eigvecs = np.linalg.eigh(cov)
            except np.linalg.LinAlgError:
                continue

            # Principal axis = eigenvector with largest eigenvalue
            principal = eigvecs[:, -1]  # last column = largest eigenvalue

            # Project vertices onto principal axis to find extent
            projections = centered @ principal
            t_min = projections.min()
            t_max = projections.max()

            start = centroid + principal * t_min
            end = centroid + principal * t_max
            axis = end - start
            length = float(np.linalg.norm(axis))
            if length < 1e-6:
                continue

            capsule = BoneCapsule(
                bone_name=name,
                start=start,
                end=end,
                radius=radius,
                axis=axis / length,
                length=length,
            )
            self._capsules.append(capsule)

        logger.info("Built %d bone collision capsules", len(self._capsules))
        return len(self._capsules)

    def resolve_penetrations(
        self,
        positions: np.ndarray,
        rest_positions: np.ndarray,
    ) -> int:
        """Push vertices out of bone capsules.

        Parameters
        ----------
        positions : flat float32 array
            Current vertex positions (modified in place).
        rest_positions : flat float32 array
            Rest-pose positions (used to determine push direction).

        Returns
        -------
        int
            Number of vertices corrected.
        """
        if not self._capsules:
            return 0

        pos = np.asarray(positions).reshape(-1, 3).astype(np.float64)
        V = len(pos)
        total_corrected = 0

        for capsule in self._capsules:
            # Vector from capsule start to each vertex
            sv = pos - capsule.start[np.newaxis, :]  # (V, 3)

            # Project onto capsule axis to find closest point on axis
            t = sv @ capsule.axis  # (V,)
            t_clamped = np.clip(t, 0.0, capsule.length)

            # Closest point on capsule axis
            closest = capsule.start[np.newaxis, :] + t_clamped[:, np.newaxis] * capsule.axis[np.newaxis, :]

            # Radial distance from axis
            diff = pos - closest
            dist = np.linalg.norm(diff, axis=1)

            # Find penetrating vertices (inside capsule)
            inside = dist < capsule.radius
            # Don't correct vertices that are exactly on the axis (ambiguous direction)
            inside &= dist > 1e-6

            if not inside.any():
                continue

            # Push to surface along radial direction
            idx = np.where(inside)[0]
            radial = diff[idx]
            radial_len = dist[idx]
            radial_dir = radial / radial_len[:, np.newaxis]

            # New position on capsule surface
            surface_pos = closest[idx] + radial_dir * capsule.radius
            pos[idx] = surface_pos
            total_corrected += len(idx)

        if total_corrected > 0:
            positions.reshape(-1)[:] = pos.astype(np.float32).ravel()

        return total_corrected

    @property
    def capsule_count(self) -> int:
        return len(self._capsules)
