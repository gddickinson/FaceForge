"""Triangles that are not skin, found by moving the body rather than looking.

The asset welds the arm to the chest and the hand to the hip.  Telling those
welds from the genuine armpit rim by geometry alone needs a dihedral test the
mesh's winding cannot support; moving the body settles it, because real skin
stretches a little and a weld across a gap stretches without limit.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceforge.body import weld_webs
from faceforge.core.mesh import BufferGeometry, MeshInstance


def mesh_with(points, faces) -> MeshInstance:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    geo = BufferGeometry(positions=pts.reshape(-1).copy(),
                         normals=np.zeros(pts.size, dtype=np.float32),
                         indices=np.asarray(faces, dtype=np.uint32).reshape(-1))
    mesh = MeshInstance(name="Skin", geometry=geo)
    mesh.store_rest_pose()
    return mesh


@pytest.fixture
def skin():
    """Two triangles: one real, one welding a far vertex to the first."""
    rest = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1.2, 0.2, 0.0]],
                    dtype=np.float64)
    faces = [[0, 1, 2], [1, 3, 2]]
    return mesh_with(rest, faces), rest


def triangles(mesh) -> int:
    return len(mesh.geometry.indices) // 3


def test_a_body_that_has_not_moved_keeps_every_triangle(skin):
    mesh, rest = skin
    assert weld_webs.cull(mesh, rest) == 0
    assert triangles(mesh) == 2


def test_a_triangle_that_triples_is_a_weld(skin):
    mesh, rest = skin
    moved = rest.copy()
    moved[3] += [6.0, 0.0, 0.0]          # the welded vertex walks away
    mesh.rest_positions = moved.reshape(-1).astype(np.float32)
    assert weld_webs.cull(mesh, rest) == 1
    assert triangles(mesh) == 1


def test_real_skin_stretching_a_little_is_kept(skin):
    mesh, rest = skin
    moved = rest * 1.4
    mesh.rest_positions = moved.reshape(-1).astype(np.float32)
    assert weld_webs.cull(mesh, rest) == 0
    assert triangles(mesh) == 2


def test_the_whole_mesh_comes_back(skin):
    mesh, rest = skin
    moved = rest.copy()
    moved[3] += [6.0, 0.0, 0.0]
    mesh.rest_positions = moved.reshape(-1).astype(np.float32)
    weld_webs.cull(mesh, rest)
    assert weld_webs.restore(mesh) is True
    assert triangles(mesh) == 2


def test_culling_is_recomputed_from_the_whole_mesh_each_time(skin):
    """Not from what is left of it, or a second pass would eat the mesh."""
    mesh, rest = skin
    moved = rest.copy()
    moved[3] += [6.0, 0.0, 0.0]
    mesh.rest_positions = moved.reshape(-1).astype(np.float32)
    weld_webs.cull(mesh, rest)
    mesh.rest_positions = rest.reshape(-1).astype(np.float32)
    assert weld_webs.cull(mesh, rest) == 0
    assert triangles(mesh) == 2


def test_the_vertices_are_never_touched(skin):
    mesh, rest = skin
    before = np.array(mesh.geometry.positions, copy=True)
    moved = rest.copy()
    moved[3] += [6.0, 0.0, 0.0]
    mesh.rest_positions = moved.reshape(-1).astype(np.float32)
    weld_webs.cull(mesh, rest)
    assert np.array_equal(mesh.geometry.positions, before), \
        "anything indexing these vertices must not be disturbed"
