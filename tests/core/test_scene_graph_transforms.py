"""Dirty-flag and world-matrix propagation tests for the scene graph.

``Scene.update()`` runs once per frame on the whole graph (~900 mesh nodes plus
pivot nodes), so its dirty-flag contract is load-bearing for both correctness
and frame cost.  The existing ``test_scene_graph.py`` covers basic parent/child
propagation; this module covers the flag lifecycle, reparenting, and the
invariants ``collect_meshes`` depends on.
"""

import numpy as np
import pytest

from faceforge.core.material import Material
from faceforge.core.math_utils import quat_from_euler
from faceforge.core.mesh import BufferGeometry, MeshInstance
from faceforge.core.scene_graph import Scene, SceneNode


def _mesh(name: str) -> MeshInstance:
    return MeshInstance(
        name=name,
        geometry=BufferGeometry(
            positions=np.zeros(9, dtype=np.float32),
            normals=np.zeros(9, dtype=np.float32),
            vertex_count=3,
        ),
        material=Material(),
    )


# ----------------------------------------------------------------------
# Dirty-flag lifecycle
# ----------------------------------------------------------------------

def test_new_node_starts_dirty():
    assert SceneNode(name="n")._matrix_dirty is True


def test_update_clears_the_dirty_flag():
    scene = Scene()
    node = SceneNode(name="n")
    scene.add(node)
    scene.update()
    assert node._matrix_dirty is False
    assert scene._matrix_dirty is False


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda n: n.set_position(1.0, 2.0, 3.0), id="set_position"),
        pytest.param(lambda n: n.set_scale(2.0, 2.0, 2.0), id="set_scale"),
        pytest.param(
            lambda n: n.set_quaternion(quat_from_euler(0.1, 0.0, 0.0)),
            id="set_quaternion",
        ),
    ],
)
def test_every_transform_setter_re_dirties(mutate):
    scene = Scene()
    node = SceneNode(name="n")
    scene.add(node)
    scene.update()
    assert node._matrix_dirty is False

    mutate(node)
    assert node._matrix_dirty is True, "transform setter did not mark node dirty"


def test_add_marks_the_child_dirty():
    parent = SceneNode(name="p")
    child = SceneNode(name="c")
    child._matrix_dirty = False
    parent.add(child)
    assert child._matrix_dirty is True


def test_mark_dirty_reaches_every_descendant():
    scene = Scene()
    a = SceneNode(name="a")
    b = SceneNode(name="b")
    c = SceneNode(name="c")
    a.add(b)
    b.add(c)
    scene.add(a)
    scene.update()
    assert [n._matrix_dirty for n in (a, b, c)] == [False, False, False]

    a.mark_dirty()
    assert [n._matrix_dirty for n in (a, b, c)] == [True, True, True]


# ----------------------------------------------------------------------
# World-matrix propagation
# ----------------------------------------------------------------------

def test_parent_move_updates_child_world_without_child_being_dirty():
    """The child's local matrix is unchanged, but its world matrix must move."""
    scene = Scene()
    parent = SceneNode(name="p")
    child = SceneNode(name="c")
    child.set_position(0.0, 0.0, 5.0)
    parent.add(child)
    scene.add(parent)
    scene.update()
    np.testing.assert_allclose(child.get_world_position(), [0, 0, 5])

    parent.set_position(10.0, 0.0, 0.0)
    assert child._matrix_dirty is False   # only the parent was touched
    scene.update()
    np.testing.assert_allclose(child.get_world_position(), [10, 0, 5])


def test_nested_rotation_composes_through_three_levels():
    scene = Scene()
    a = SceneNode(name="a")
    b = SceneNode(name="b")
    c = SceneNode(name="c")
    a.set_quaternion(quat_from_euler(0.0, 0.0, np.pi / 2))   # +90 deg about Z
    b.set_position(1.0, 0.0, 0.0)
    c.set_position(1.0, 0.0, 0.0)
    a.add(b)
    b.add(c)
    scene.add(a)
    scene.update()

    # b at (1,0,0) rotated 90 deg about Z -> (0,1,0); c one further step -> (0,2,0)
    np.testing.assert_allclose(b.get_world_position(), [0, 1, 0], atol=1e-9)
    np.testing.assert_allclose(c.get_world_position(), [0, 2, 0], atol=1e-9)


def test_scale_and_translation_compose_in_the_right_order():
    """Child offsets must be scaled by the parent, not added before scaling."""
    scene = Scene()
    parent = SceneNode(name="p")
    parent.set_scale(2.0, 2.0, 2.0)
    parent.set_position(5.0, 0.0, 0.0)
    child = SceneNode(name="c")
    child.set_position(3.0, 0.0, 0.0)
    parent.add(child)
    scene.add(parent)
    scene.update()
    np.testing.assert_allclose(child.get_world_position(), [5 + 2 * 3, 0, 0])


def test_reparenting_moves_the_child_under_the_new_parent():
    scene = Scene()
    p1 = SceneNode(name="p1")
    p1.set_position(10.0, 0.0, 0.0)
    p2 = SceneNode(name="p2")
    p2.set_position(-10.0, 0.0, 0.0)
    child = SceneNode(name="c")
    child.set_position(1.0, 0.0, 0.0)
    p1.add(child)
    scene.add(p1)
    scene.add(p2)
    scene.update()
    np.testing.assert_allclose(child.get_world_position(), [11, 0, 0])

    p2.add(child)                      # add() detaches from the old parent
    assert child.parent is p2
    assert child not in p1.children
    scene.update()
    np.testing.assert_allclose(child.get_world_position(), [-9, 0, 0])


def test_get_world_position_returns_a_copy_not_a_view():
    """A caller mutating the returned vector must not corrupt the node."""
    scene = Scene()
    node = SceneNode(name="n")
    node.set_position(1.0, 2.0, 3.0)
    scene.add(node)
    scene.update()

    p = node.get_world_position()
    p[0] = 999.0
    np.testing.assert_allclose(node.get_world_position(), [1, 2, 3])


# ----------------------------------------------------------------------
# collect_meshes invariants
# ----------------------------------------------------------------------

def test_collect_meshes_pairs_each_mesh_with_its_own_world_matrix():
    scene = Scene()
    for i, x in enumerate((1.0, 2.0, 3.0)):
        node = SceneNode(name=f"n{i}")
        node.mesh = _mesh(f"m{i}")
        node.set_position(x, 0.0, 0.0)
        scene.add(node)
    scene.update()

    pairs = scene.collect_meshes()
    assert [m.name for m, _ in pairs] == ["m0", "m1", "m2"]
    assert [w[0, 3] for _, w in pairs] == [1.0, 2.0, 3.0]


def test_collect_meshes_skips_mesh_hidden_via_mesh_visible():
    scene = Scene()
    node = SceneNode(name="n")
    node.mesh = _mesh("m")
    node.mesh.visible = False
    scene.add(node)
    scene.update()
    assert scene.collect_meshes() == []


def test_collect_meshes_prunes_subtree_of_invisible_node():
    scene = Scene()
    parent = SceneNode(name="p")
    parent.mesh = _mesh("parent_mesh")
    child = SceneNode(name="c")
    child.mesh = _mesh("child_mesh")
    parent.add(child)
    scene.add(parent)
    scene.update()
    assert len(scene.collect_meshes()) == 2

    parent.visible = False
    assert scene.collect_meshes() == [], "child of an invisible node was collected"


def test_removed_node_is_no_longer_collected():
    scene = Scene()
    node = SceneNode(name="n")
    node.mesh = _mesh("m")
    scene.add(node)
    scene.update()
    assert len(scene.collect_meshes()) == 1

    scene.remove(node)
    assert node.parent is None
    assert scene.collect_meshes() == []


def test_find_all_returns_every_duplicate_name():
    """Mesh ids repeat across the body; find() only ever returns the first."""
    scene = Scene()
    for _ in range(3):
        scene.add(SceneNode(name="rib"))
    scene.update()

    assert scene.find("rib") is scene.children[0]
    assert len(scene.find_all("rib")) == 3
    assert scene.find("nonexistent") is None
    assert scene.find_all("nonexistent") == []
