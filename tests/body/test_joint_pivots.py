"""Joint pivot helpers: re-basing a bone under a pivot must reach the GPU copy."""

from __future__ import annotations

import numpy as np

from faceforge.body.joint_pivots import reparent_under_pivot
from faceforge.core.mesh import BufferGeometry, MeshInstance
from faceforge.core.scene_graph import SceneNode


def _bone(name: str, points: np.ndarray) -> SceneNode:
    geom = BufferGeometry(positions=points.astype(np.float32).ravel(),
                          normals=np.zeros(points.size, dtype=np.float32))
    node = SceneNode(name=name)
    node.mesh = MeshInstance(name=name, geometry=geom)
    return node


def test_reparent_under_pivot_offsets_the_vertices_and_flags_the_gpu_copy():
    pts = np.array([[10.0, 20.0, 30.0], [12.0, 20.0, 30.0], [10.0, 22.0, 30.0]])
    bone = _bone("Right 5th Rib", pts)
    root = SceneNode(name="root")
    root.add(bone)
    bone.mesh.needs_update = False          # as after the renderer's first upload

    pivot = SceneNode(name="pivot")
    centroid = pts.mean(axis=0)
    pivot.set_position(*centroid)
    root.add(pivot)
    reparent_under_pivot(bone, pivot, centroid)

    assert bone.parent is pivot
    local = bone.mesh.geometry.positions.reshape(-1, 3)
    np.testing.assert_allclose(local, pts - centroid, atol=1e-6)
    np.testing.assert_allclose(bone.mesh.rest_positions.reshape(-1, 3), local)
    # The edit happened in place, so a mesh the renderer has already uploaded
    # must be re-streamed, or it is drawn at pivot + the original vertices.
    assert bone.mesh.needs_update is True
    root.update_world_matrix(force=True)
    world = local @ np.asarray(bone.world_matrix)[:3, :3].T + np.asarray(bone.world_matrix)[:3, 3]
    np.testing.assert_allclose(world, pts, atol=1e-5)
