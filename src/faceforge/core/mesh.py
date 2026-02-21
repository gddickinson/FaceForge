"""Mesh data structures for geometry storage (no GL dependencies)."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from faceforge.core.material import Material


@dataclass
class BufferGeometry:
    """Stores vertex attribute arrays for a mesh.

    All arrays use float32 for GL compatibility.
    positions: Nx3 flat array (x,y,z per vertex)
    normals: Nx3 flat array
    indices: triangle index array (uint32), optional for non-indexed geometry
    vertex_colors: optional Nx3 float32 per-vertex RGB (0..1)
    """
    positions: NDArray[np.float32]
    normals: NDArray[np.float32]
    indices: Optional[NDArray[np.uint32]] = None
    vertex_count: int = 0
    vertex_colors: Optional[NDArray[np.float32]] = None
    colors_dirty: bool = False

    def __post_init__(self):
        if self.vertex_count == 0:
            self.vertex_count = len(self.positions) // 3

    @property
    def triangle_count(self) -> int:
        if self.indices is not None:
            return len(self.indices) // 3
        return self.vertex_count // 3

    @property
    def has_indices(self) -> bool:
        return self.indices is not None and len(self.indices) > 0

    def compute_normals(self) -> None:
        """Compute per-vertex normals from face normals (flat or indexed)."""
        pos = self.positions.reshape(-1, 3)
        norms = np.zeros_like(pos)

        if self.has_indices:
            idx = self.indices.reshape(-1, 3)
            for tri in idx:
                v0, v1, v2 = pos[tri[0]], pos[tri[1]], pos[tri[2]]
                n = np.cross(v1 - v0, v2 - v0)
                norms[tri[0]] += n
                norms[tri[1]] += n
                norms[tri[2]] += n
        else:
            for i in range(0, len(pos), 3):
                v0, v1, v2 = pos[i], pos[i + 1], pos[i + 2]
                n = np.cross(v1 - v0, v2 - v0)
                norms[i] = n
                norms[i + 1] = n
                norms[i + 2] = n

        lengths = np.linalg.norm(norms, axis=1, keepdims=True)
        lengths = np.maximum(lengths, 1e-10)
        norms /= lengths
        self.normals = norms.ravel().astype(np.float32)

    def get_bounding_center(self) -> NDArray[np.float64]:
        """Compute centroid of all vertices."""
        pos = self.positions.reshape(-1, 3)
        return pos.mean(axis=0).astype(np.float64)

    def clone(self) -> "BufferGeometry":
        """Create a deep copy."""
        return BufferGeometry(
            positions=self.positions.copy(),
            normals=self.normals.copy(),
            indices=self.indices.copy() if self.indices is not None else None,
            vertex_count=self.vertex_count,
        )


@dataclass
class MeshInstance:
    """A mesh with material, linking geometry to rendering properties."""
    name: str
    geometry: BufferGeometry
    material: Material = field(default_factory=Material)
    visible: bool = True
    # GL handle (set by renderer)
    gl_handle: object = None
    # Flag for geometry updates
    needs_update: bool = True

    # Optional: rest positions for deformation systems
    rest_positions: Optional[NDArray[np.float32]] = None
    rest_normals: Optional[NDArray[np.float32]] = None

    # Scene mode: if True (default), the renderer applies scene_transform
    # to this mesh's model_view matrix.  Set False for environment meshes
    # (table, walls, lamp) that should stay fixed in world space.
    scene_affected: bool = True

    def store_rest_pose(self) -> None:
        """Save current positions/normals as rest pose for deformation."""
        self.rest_positions = self.positions.copy()
        self.rest_normals = self.normals.copy()

    @property
    def positions(self) -> NDArray[np.float32]:
        return self.geometry.positions

    @positions.setter
    def positions(self, value: NDArray[np.float32]):
        self.geometry.positions = value
        self.needs_update = True

    @property
    def normals(self) -> NDArray[np.float32]:
        return self.geometry.normals

    @normals.setter
    def normals(self, value: NDArray[np.float32]):
        self.geometry.normals = value
        self.needs_update = True
