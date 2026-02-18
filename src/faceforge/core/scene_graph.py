"""Scene graph with hierarchical transforms, mirroring Three.js group structure."""

from typing import Optional

import numpy as np

from faceforge.core.math_utils import (
    Mat4, Vec3, Quat,
    mat4_identity, mat4_compose, quat_identity, vec3,
)
from faceforge.core.mesh import MeshInstance


class SceneNode:
    """A node in the scene graph hierarchy.

    Mirrors Three.js Object3D: position, quaternion, scale → local matrix.
    World matrix = parent.world_matrix @ local_matrix.
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.parent: Optional["SceneNode"] = None
        self.children: list["SceneNode"] = []

        # Transform
        self.position: Vec3 = vec3()
        self.quaternion: Quat = quat_identity()
        self.scale: Vec3 = vec3(1, 1, 1)

        # Matrices
        self.local_matrix: Mat4 = mat4_identity()
        self.world_matrix: Mat4 = mat4_identity()

        # Visibility
        self.visible: bool = True

        # Optional mesh attached to this node
        self.mesh: Optional[MeshInstance] = None

        # Dirty flag for matrix updates
        self._matrix_dirty: bool = True

    def add(self, child: "SceneNode") -> "SceneNode":
        """Add a child node. Removes from previous parent if any."""
        if child.parent is not None:
            child.parent.remove(child)
        child.parent = self
        self.children.append(child)
        child._matrix_dirty = True
        return self

    def remove(self, child: "SceneNode") -> "SceneNode":
        """Remove a child node."""
        if child in self.children:
            self.children.remove(child)
            child.parent = None
        return self

    def set_position(self, x: float, y: float, z: float) -> "SceneNode":
        self.position = vec3(x, y, z)
        self._matrix_dirty = True
        return self

    def set_quaternion(self, q: Quat) -> "SceneNode":
        self.quaternion = q.copy()
        self._matrix_dirty = True
        return self

    def set_scale(self, x: float, y: float, z: float) -> "SceneNode":
        self.scale = vec3(x, y, z)
        self._matrix_dirty = True
        return self

    def update_local_matrix(self) -> None:
        """Recompute local matrix from position, quaternion, scale."""
        self.local_matrix = mat4_compose(self.position, self.quaternion, self.scale)
        self._matrix_dirty = False

    def update_world_matrix(self, force: bool = False) -> None:
        """Recursively update world matrices for this node and all descendants."""
        if self._matrix_dirty or force:
            self.update_local_matrix()

        if self.parent is not None:
            self.world_matrix = self.parent.world_matrix @ self.local_matrix
        else:
            self.world_matrix = self.local_matrix.copy()

        for child in self.children:
            child.update_world_matrix(force=force)

    def traverse(self, callback) -> None:
        """Visit this node and all descendants depth-first."""
        callback(self)
        for child in self.children:
            child.traverse(callback)

    def traverse_visible(self, callback) -> None:
        """Visit only visible nodes depth-first."""
        if not self.visible:
            return
        callback(self)
        for child in self.children:
            child.traverse_visible(callback)

    def find(self, name: str) -> Optional["SceneNode"]:
        """Find first descendant with given name."""
        if self.name == name:
            return self
        for child in self.children:
            found = child.find(name)
            if found is not None:
                return found
        return None

    def find_all(self, name: str) -> list["SceneNode"]:
        """Find all descendants with given name."""
        results = []
        if self.name == name:
            results.append(self)
        for child in self.children:
            results.extend(child.find_all(name))
        return results

    def get_world_position(self) -> Vec3:
        """Extract world position from world matrix."""
        return self.world_matrix[:3, 3].copy()

    def mark_dirty(self) -> None:
        """Mark this node and all descendants as needing matrix update."""
        self._matrix_dirty = True
        for child in self.children:
            child.mark_dirty()


class Scene(SceneNode):
    """Root scene node."""

    def __init__(self):
        super().__init__(name="scene")

    def update(self) -> None:
        """Update all world matrices in the scene."""
        self.update_world_matrix(force=False)

    def collect_meshes(self) -> list[tuple[MeshInstance, Mat4]]:
        """Collect all visible meshes with their world transforms."""
        result = []

        def _collect(node: SceneNode):
            if node.mesh is not None and node.mesh.visible:
                result.append((node.mesh, node.world_matrix))

        self.traverse_visible(_collect)
        return result
