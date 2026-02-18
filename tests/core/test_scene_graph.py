"""Tests for scene graph module."""

import numpy as np
import pytest

from faceforge.core.scene_graph import SceneNode, Scene
from faceforge.core.math_utils import vec3, quat_identity
from faceforge.core.mesh import BufferGeometry, MeshInstance
from faceforge.core.material import Material


def _make_mesh(name="test"):
    geom = BufferGeometry(
        positions=np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32),
        normals=np.array([0, 0, 1, 0, 0, 1, 0, 0, 1], dtype=np.float32),
        vertex_count=3,
    )
    return MeshInstance(name=name, geometry=geom)


def test_node_hierarchy():
    parent = SceneNode(name="parent")
    child = SceneNode(name="child")
    parent.add(child)
    assert child.parent is parent
    assert child in parent.children


def test_node_remove():
    parent = SceneNode(name="parent")
    child = SceneNode(name="child")
    parent.add(child)
    parent.remove(child)
    assert child.parent is None
    assert child not in parent.children


def test_node_reparent():
    p1 = SceneNode(name="p1")
    p2 = SceneNode(name="p2")
    child = SceneNode(name="child")
    p1.add(child)
    p2.add(child)  # Should remove from p1
    assert child.parent is p2
    assert child not in p1.children
    assert child in p2.children


def test_find():
    root = SceneNode(name="root")
    a = SceneNode(name="a")
    b = SceneNode(name="b")
    c = SceneNode(name="target")
    root.add(a)
    a.add(b)
    b.add(c)
    found = root.find("target")
    assert found is c


def test_find_not_found():
    root = SceneNode(name="root")
    assert root.find("nonexistent") is None


def test_world_matrix_propagation():
    root = SceneNode(name="root")
    root.set_position(10, 0, 0)
    child = SceneNode(name="child")
    child.set_position(5, 0, 0)
    root.add(child)
    root.update_world_matrix(force=True)

    # Child world position should be 15, 0, 0
    world_pos = child.get_world_position()
    np.testing.assert_array_almost_equal(world_pos, [15, 0, 0])


def test_scene_collect_meshes():
    scene = Scene()
    node = SceneNode(name="mesh_node")
    node.mesh = _make_mesh()
    scene.add(node)
    scene.update()

    meshes = scene.collect_meshes()
    assert len(meshes) == 1
    assert meshes[0][0] is node.mesh


def test_invisible_node_not_collected():
    scene = Scene()
    node = SceneNode(name="hidden")
    node.mesh = _make_mesh()
    node.visible = False
    scene.add(node)
    scene.update()

    meshes = scene.collect_meshes()
    assert len(meshes) == 0


def test_traverse():
    root = SceneNode(name="root")
    a = SceneNode(name="a")
    b = SceneNode(name="b")
    root.add(a)
    root.add(b)

    visited = []
    root.traverse(lambda n: visited.append(n.name))
    assert visited == ["root", "a", "b"]


def test_scale_propagation():
    root = SceneNode(name="root")
    root.set_scale(2, 2, 2)
    child = SceneNode(name="child")
    child.set_position(1, 0, 0)
    root.add(child)
    root.update_world_matrix(force=True)

    world_pos = child.get_world_position()
    np.testing.assert_array_almost_equal(world_pos, [2, 0, 0])
