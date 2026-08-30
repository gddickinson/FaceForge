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

        # Visibility.  Exposed as a property so that plain
        # ``node.visible = False`` -- the idiom used throughout the app and the
        # tests -- invalidates the root's cached traversal.
        self._visible: bool = True

        # Optional mesh attached to this node
        self.mesh: Optional[MeshInstance] = None

        # Dirty flags for matrix updates.
        #   _matrix_dirty   : the LOCAL matrix must be recomposed
        #   _world_dirty    : this node's WORLD matrix must be recomputed
        #   _subtree_dirty  : some descendant's world matrix must be recomputed
        # The last two are what let update_world_matrix() skip clean subtrees
        # instead of recomputing every world matrix product, and recursing into
        # every child, on every frame.
        self._matrix_dirty: bool = True
        self._world_dirty: bool = True
        self._subtree_dirty: bool = True

    @property
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, flag: bool) -> None:
        flag = bool(flag)
        if flag != self._visible:
            self._visible = flag
            self._bump_revision()

    def add(self, child: "SceneNode") -> "SceneNode":
        """Add a child node. Removes from previous parent if any."""
        if child.parent is not None:
            child.parent.remove(child)
        child.parent = self
        self.children.append(child)
        child._matrix_dirty = True
        child.mark_world_dirty()
        self._bump_revision()
        # A reparent arrives here as remove-then-add; the remove queued this
        # subtree's meshes for GPU eviction, so un-queue them again.
        root = self._root()
        if isinstance(root, Scene):
            root._adopt_meshes(child)
        return self

    def remove(self, child: "SceneNode") -> "SceneNode":
        """Remove a child node."""
        if child in self.children:
            root = self._root()
            self.children.remove(child)
            child.parent = None
            self._bump_revision()
            # Nothing else in the tree tells the renderer that a mesh left the
            # scene, so record it here and let the renderer drain the queue on
            # its next frame.  Without this the GLMesh (one VAO + 2-3 VBOs per
            # structure) leaks for the lifetime of the process.
            if isinstance(root, Scene):
                root._orphan_meshes(child)
        return self

    def set_position(self, x: float, y: float, z: float) -> "SceneNode":
        self.position = vec3(x, y, z)
        self._matrix_dirty = True
        self.mark_world_dirty()
        return self

    def set_quaternion(self, q: Quat) -> "SceneNode":
        self.quaternion = q.copy()
        self._matrix_dirty = True
        self.mark_world_dirty()
        return self

    def set_scale(self, x: float, y: float, z: float) -> "SceneNode":
        self.scale = vec3(x, y, z)
        self._matrix_dirty = True
        self.mark_world_dirty()
        return self

    def set_visible(self, flag: bool) -> "SceneNode":
        """Chainable alias for the ``visible`` property."""
        self.visible = flag
        return self

    # -- dirty-flag bookkeeping -------------------------------------------

    def _root(self) -> "SceneNode":
        n = self
        while n.parent is not None:
            n = n.parent
        return n

    def _bump_revision(self) -> None:
        """Signal a topology / visibility change to the root's collect cache."""
        r = self._root()
        r._revision = getattr(r, "_revision", 0) + 1

    def subtree_meshes(self) -> list[MeshInstance]:
        """Every mesh attached to this node or any descendant."""
        found: list[MeshInstance] = []
        stack: list[SceneNode] = [self]
        while stack:
            n = stack.pop()
            if n.mesh is not None:
                found.append(n.mesh)
            stack.extend(n.children)
        return found

    def mark_world_dirty(self) -> None:
        """Flag this subtree stale and tell the ancestors to walk into it.

        Both halves matter.  Without the descendant marking, moving a group
        would not move its children.  Without the ancestor marking,
        ``Scene.update()`` would return at a clean root and never reach the
        node that moved.
        """
        stack = [self]
        while stack:
            n = stack.pop()
            n._world_dirty = True
            n._subtree_dirty = True
            stack.extend(n.children)
        p = self.parent
        while p is not None:
            if p._subtree_dirty:
                break                   # everything above is already flagged
            p._subtree_dirty = True
            p = p.parent

    def update_local_matrix(self) -> None:
        """Recompute local matrix from position, quaternion, scale."""
        self.local_matrix = mat4_compose(self.position, self.quaternion, self.scale)
        self._matrix_dirty = False

    def update_world_matrix(self, force: bool = False) -> None:
        """Update world matrices for this node and any stale descendants.

        A wholly clean subtree returns immediately, so a static scene costs one
        flag test per frame instead of one 4x4 matrix product plus a recursive
        call per node.  ``_matrix_dirty`` previously gated only the *local*
        matrix: the world-matrix product and the recursion into every child ran
        unconditionally on every frame, for all ~900 mesh nodes.

        World matrices are written in place, which also keeps any cached
        ``(mesh, world_matrix)`` tuples valid across frames.
        """
        if not (force or self._world_dirty or self._subtree_dirty):
            return

        if force or self._world_dirty:
            if self._matrix_dirty:
                self.update_local_matrix()
            if self.parent is not None:
                np.matmul(self.parent.world_matrix, self.local_matrix,
                          out=self.world_matrix)
            else:
                self.world_matrix[...] = self.local_matrix
            self._world_dirty = False
            # our world matrix moved, so every descendant's must be rebuilt
            for child in self.children:
                child.update_world_matrix(force=True)
        else:
            for child in self.children:
                child.update_world_matrix(force=False)

        self._subtree_dirty = False

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
        self._world_dirty = True
        self._subtree_dirty = True
        for child in self.children:
            child.mark_dirty()
        p = self.parent
        while p is not None and not p._subtree_dirty:
            p._subtree_dirty = True
            p = p.parent


class Scene(SceneNode):
    """Root scene node."""

    def __init__(self):
        super().__init__(name="scene")
        self._revision: int = 0
        self._collect_cache: list[tuple[MeshInstance, Mat4]] | None = None
        self._collect_rev: int = -1
        # Meshes detached from the graph since the renderer last looked.
        self._pending_orphans: list[MeshInstance] = []

    def update(self) -> None:
        """Update all world matrices in the scene."""
        self.update_world_matrix(force=False)

    def collect_meshes(self) -> list[tuple[MeshInstance, Mat4]]:
        """Collect all visible meshes with their world transforms.

        The node *traversal* is cached and only redone when the topology or a
        node's ``visible`` flag changes (tracked by ``_revision``); the previous
        implementation rebuilt the list through a Python closure callback every
        frame, and ``render_split`` called it twice per frame.
        ``mesh.visible`` -- which the anatomy toggles flip constantly -- is
        re-tested on every call, so per-structure visibility stays exact.
        """
        if self._collect_cache is None or self._collect_rev != self._revision:
            pairs: list[tuple[MeshInstance, Mat4]] = []
            stack: list[SceneNode] = [self]
            while stack:
                node = stack.pop()
                if not node.visible:
                    continue
                if node.mesh is not None:
                    pairs.append((node.mesh, node.world_matrix))
                # reversed(): a LIFO stack must push children in reverse to
                # visit them in declaration order, matching traverse_visible()
                stack.extend(reversed(node.children))
            self._collect_cache = pairs
            self._collect_rev = self._revision

        cache = self._collect_cache
        for mesh, _world in cache:
            if not mesh.visible:
                return [p for p in cache if p[0].visible]
        return cache

    def invalidate_collect_cache(self) -> None:
        """Force the next ``collect_meshes()`` to re-walk the graph.

        Only needed if something mutates ``node.children`` or ``node.visible``
        directly instead of going through ``add`` / ``remove`` / ``set_visible``.
        """
        self._collect_rev = -1

    # -- GPU-resource lifetime -------------------------------------------

    def _orphan_meshes(self, detached: SceneNode) -> None:
        """Queue every mesh in the detached subtree for GPU eviction."""
        self._pending_orphans.extend(detached.subtree_meshes())

    def _adopt_meshes(self, attached: SceneNode) -> None:
        """Cancel the eviction queued by a preceding ``remove`` (reparenting)."""
        if not self._pending_orphans:
            return
        readopted = {id(m) for m in attached.subtree_meshes()}
        self._pending_orphans = [
            m for m in self._pending_orphans if id(m) not in readopted
        ]

    def take_orphaned_meshes(self) -> list[MeshInstance]:
        """Return meshes that have left the graph, and clear the queue.

        Reachability is re-verified against the live graph -- including nodes
        hidden by ``visible = False`` -- so a mesh that was detached and
        re-attached somewhere else is never reported.  The walk only happens on
        frames where something was actually removed.
        """
        pending = self._pending_orphans
        if not pending:
            return []
        self._pending_orphans = []
        live = {id(m) for m in self.subtree_meshes()}
        gone: list[MeshInstance] = []
        seen: set[int] = set()
        for mesh in pending:
            key = id(mesh)
            if key in live or key in seen:
                continue
            seen.add(key)
            gone.append(mesh)
        return gone
