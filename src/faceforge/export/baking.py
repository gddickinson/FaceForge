"""World-space baking shared by every geometry exporter.

Every interchange format FaceForge writes (GLB, OBJ, PLY, STL) needs the same
thing first: each :class:`~faceforge.core.mesh.MeshInstance`'s positions and
normals in *world* space, with the scene-graph transform baked in, because none
of those formats can express FaceForge's node hierarchy faithfully.

This was inlined in ``glb_exporter._build_gltf``.  It is factored out here --
unchanged, including the ``inv(M)^T`` normal matrix and the renormalisation
epsilon -- so that the four exporters cannot drift from one another.  A
divergence would mean the same scene exported to two formats had different
geometry, which is the sort of thing nobody notices until a measurement taken
off an OBJ disagrees with the same measurement taken off a GLB.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BakedGeometry:
    """One mesh's geometry in world space, ready for any file format.

    ``indices`` is always present and always triangles: a format-level exporter
    should never have to care whether the source geometry was indexed, and a
    non-indexed source gets ``arange`` indices here instead of in four places.
    """

    positions: np.ndarray          # (V, 3) float32
    normals: np.ndarray            # (V, 3) float32
    indices: np.ndarray            # (F, 3) uint32

    @property
    def vertex_count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def triangle_count(self) -> int:
        return int(self.indices.shape[0])

    def triangle_vertices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(v0, v1, v2)``, each ``(F, 3)`` -- what STL needs."""
        return (
            self.positions[self.indices[:, 0]],
            self.positions[self.indices[:, 1]],
            self.positions[self.indices[:, 2]],
        )

    def face_normals(self) -> np.ndarray:
        """Geometric per-facet normals, ``(F, 3)`` float32.

        STL stores a facet normal, not vertex normals, so it cannot reuse the
        interpolated ones.  Degenerate facets (zero area) get ``(0, 0, 0)``,
        which is what the STL specification says to write when the normal is
        not meaningful -- readers then recompute it from the winding.
        """
        v0, v1, v2 = self.triangle_vertices()
        normals = np.cross(v1 - v0, v2 - v0)
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        safe = lengths > 1e-20
        out = np.zeros_like(normals)
        np.divide(normals, lengths, out=out, where=safe)
        return out.astype(np.float32)


def bake_world_geometry(mesh: object, world_mat: np.ndarray) -> BakedGeometry:
    """Apply *world_mat* to *mesh*'s geometry and return it in world space.

    Positions transform with ``M``; normals with ``inv(M[:3,:3])^T`` and are
    renormalised, so a non-uniform scale in the scene graph does not tilt them.
    A singular upper-left block falls back to the rotation/scale block itself
    rather than raising: an unexportable mesh is worse than a mesh with
    approximate normals, and the fallback is what the GLB exporter has always
    done.
    """
    geom = mesh.geometry                                     # type: ignore[attr-defined]
    positions = np.asarray(geom.positions, dtype=np.float32).reshape(-1, 3)
    normals = np.asarray(geom.normals, dtype=np.float32).reshape(-1, 3)

    world = np.asarray(world_mat, dtype=np.float64)
    rot_scale = world[:3, :3]
    translation = world[:3, 3]

    world_pos = (rot_scale @ positions.T).T + translation

    try:
        normal_mat = np.linalg.inv(rot_scale).T
    except np.linalg.LinAlgError:
        normal_mat = rot_scale
    world_norm = (normal_mat @ normals.T).T
    lengths = np.linalg.norm(world_norm, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-10)
    world_norm = world_norm / lengths

    if getattr(geom, "has_indices", False):
        indices = np.asarray(geom.indices, dtype=np.uint32).reshape(-1, 3)
    else:
        indices = np.arange(len(world_pos), dtype=np.uint32).reshape(-1, 3)

    return BakedGeometry(
        positions=world_pos.astype(np.float32),
        normals=world_norm.astype(np.float32),
        indices=indices,
    )
