"""The head's soft tissue owns its rest pose, so something has to move it.

The brain is the case that shows: it hangs off ``brainGroup`` rather than off
the skull, deliberately, so that it stays visible when the skull is hidden.
The cost is that nothing in the scene graph carries it when the skull moves --
measured with the fit on, the skull came down 24 units and the brain stayed
where it was, a whole head-height above the body.
"""

from __future__ import annotations

import numpy as np

from faceforge.anatomy import head_tissue
from faceforge.core.mesh import BufferGeometry, MeshInstance
from faceforge.core.scene_graph import SceneNode


def _mesh(name: str, points: np.ndarray) -> SceneNode:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    node = SceneNode(name=name)
    node.mesh = MeshInstance(
        name=name,
        geometry=BufferGeometry(positions=pts.reshape(-1).copy(),
                                normals=np.zeros(pts.size, dtype=np.float32)))
    node.mesh.store_rest_pose()
    node.mesh.rest_positions = np.array(node.mesh.geometry.positions, copy=True)
    return node


def _brain() -> tuple[SceneNode, SceneNode]:
    group = SceneNode(name="brainGroup")
    cerebrum = _mesh("Cerebrum", [[0, 0, 20], [2, 1, 24], [-2, -1, 18]])
    group.add(cerebrum)
    return group, cerebrum


DOWN = lambda p: np.tile([0.0, 0.0, -24.0], (len(p), 1))   # noqa: E731


def test_the_brain_is_collected_as_head_tissue():
    group, cerebrum = _brain()
    owned = head_tissue.owned_meshes(None, group)
    assert id(cerebrum.mesh) in owned


def test_nothing_is_collected_without_a_brain_group():
    assert head_tissue.owned_meshes(None, None) == set()
    assert head_tissue.brain_meshes(None) == []


def test_the_brain_comes_down_with_the_skull():
    group, cerebrum = _brain()
    before = np.asarray(cerebrum.mesh.geometry.positions,
                        dtype=np.float64).reshape(-1, 3).copy()
    head_tissue.rebase(None, DOWN, brain_group=group)
    after = np.asarray(cerebrum.mesh.geometry.positions,
                       dtype=np.float64).reshape(-1, 3)
    np.testing.assert_allclose(after - before,
                               np.tile([0.0, 0.0, -24.0], (len(before), 1)),
                               atol=1e-6)


def test_the_rest_pose_moves_too_not_only_the_vertex_buffer():
    """It is the rest pose every per-frame deformer rebuilds from."""
    group, cerebrum = _brain()
    head_tissue.rebase(None, DOWN, brain_group=group)
    np.testing.assert_allclose(
        np.asarray(cerebrum.mesh.rest_positions, dtype=np.float64),
        np.asarray(cerebrum.mesh.geometry.positions, dtype=np.float64),
        atol=1e-6)


def test_moving_twice_does_not_compound():
    group, cerebrum = _brain()
    head_tissue.rebase(None, DOWN, brain_group=group)
    once = np.array(cerebrum.mesh.geometry.positions, copy=True)
    head_tissue.rebase(None, DOWN, brain_group=group)
    np.testing.assert_allclose(cerebrum.mesh.geometry.positions, once, atol=1e-6)


def test_taking_the_fit_off_puts_the_brain_back():
    group, cerebrum = _brain()
    before = np.array(cerebrum.mesh.geometry.positions, copy=True)
    head_tissue.rebase(None, DOWN, brain_group=group)
    head_tissue.rebase(None, None, brain_group=group)
    np.testing.assert_allclose(cerebrum.mesh.geometry.positions, before,
                               atol=1e-6)
