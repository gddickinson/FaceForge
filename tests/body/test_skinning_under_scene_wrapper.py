"""Skinning output is a body-frame quantity: the scene wrapper must not leak in.

The wrapper node (scene mode) rotates and translates the whole body.  The
skinning cancels it from the joint deltas -- but its correction passes read
joint world matrices directly and compared body-frame vertices with
world-frame bones, so skinned muscles flew off the skeleton as soon as the
body stood in a room (measured: a thigh muscle 146 units from its femur).
These tests pin the invariant on a synthetic rig: for the same joint pose the
skinned positions are identical whether or not a wrapper is above the body.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceforge.body.soft_tissue import SoftTissueSkinning
from faceforge.core.math_utils import quat_from_axis_angle, vec3
from faceforge.core.mesh import BufferGeometry, MeshInstance
from faceforge.core.scene_graph import Scene, SceneNode
from faceforge.core.state import BodyState


def _grid_mesh(nx: int = 24, ny: int = 24) -> MeshInstance:
    xs = np.linspace(-12.0, 12.0, nx)
    zs = np.linspace(-45.0, 45.0, ny)
    gx, gz = np.meshgrid(xs, zs, indexing="ij")
    positions = np.stack([gx.ravel(), np.full(gx.size, 3.0), gz.ravel()], axis=1).astype(np.float32)
    tris = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a = i * ny + j
            tris.append((a, a + 1, a + ny))
            tris.append((a + 1, a + ny + 1, a + ny))
    geom = BufferGeometry(
        positions=positions.ravel(),
        normals=np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), nx * ny),
        indices=np.array(tris, dtype=np.uint32).ravel(), vertex_count=nx * ny,
    )
    mesh = MeshInstance(name="Test Muscle R", geometry=geom)
    mesh.store_rest_pose()
    return mesh


def _rig():
    """scene -> wrapper -> bodyRoot -> a chained vertical spine of 4 joints."""
    scene = Scene()
    wrapper = SceneNode("scene_wrapper")
    body = SceneNode("bodyRoot")
    scene.add(wrapper)
    wrapper.add(body)
    nodes = []
    parent = body
    for i, z in enumerate([-45.0, 30.0, 30.0, 30.0]):
        n = SceneNode(f"spine_{i}")
        n.set_position(0.0, 0.0, z)
        parent.add(n)
        nodes.append(n)
        parent = n
    scene.update()
    sk = SoftTissueSkinning()
    sk.build_skin_joints([[(n.name, n) for n in nodes]])
    return scene, wrapper, nodes, sk


def _skin(sk, scene, mesh, pose_node, angle_deg: float) -> np.ndarray:
    pose_node.set_quaternion(quat_from_axis_angle(vec3(1, 0, 0), np.radians(angle_deg)))
    scene.update()
    sk._last_signature = ""
    sk.update(BodyState())
    return np.asarray(mesh.geometry.positions, dtype=np.float64).reshape(-1, 3).copy()


def test_same_pose_gives_the_same_body_frame_positions_with_or_without_a_wrapper():
    scene, wrapper, nodes, sk = _rig()
    mesh = _grid_mesh()
    sk.register_skin_mesh(mesh, is_muscle=True, muscle_name=mesh.name)
    _skin(sk, scene, mesh, nodes[1], 0.0)          # neutral reference frame
    without = _skin(sk, scene, mesh, nodes[1], 35.0)
    assert not np.allclose(without, mesh.rest_positions.reshape(-1, 3)), \
        "the joint rotation must actually deform the mesh"

    wrapper.set_position(0.0, 203.0, 0.0)
    wrapper.set_quaternion(quat_from_axis_angle(vec3(1, 0, 0), -np.pi / 2))
    sk.scene_wrapper = wrapper
    with_wrapper = _skin(sk, scene, mesh, nodes[1], 35.0)
    assert np.allclose(with_wrapper, without, atol=1e-6)


def test_the_fixture_detects_a_leaked_world_frame(monkeypatch):
    """Negative control: read raw world matrices and the invariant breaks."""
    scene, wrapper, nodes, sk = _rig()
    mesh = _grid_mesh()
    sk.register_skin_mesh(mesh, is_muscle=True, muscle_name=mesh.name)
    _skin(sk, scene, mesh, nodes[1], 0.0)
    without = _skin(sk, scene, mesh, nodes[1], 35.0)

    wrapper.set_position(0.0, 203.0, 0.0)
    wrapper.set_quaternion(quat_from_axis_angle(vec3(1, 0, 0), -np.pi / 2))
    sk.scene_wrapper = wrapper

    def raw_world(node):
        node.update_world_matrix()
        return np.asarray(node.world_matrix, dtype=np.float64)

    monkeypatch.setattr(sk, "_joint_world", raw_world)
    leaked = _skin(sk, scene, mesh, nodes[1], 35.0)
    assert not np.allclose(leaked, without, atol=1e-3)


def test_moving_only_the_wrapper_does_not_change_the_skinning_signature():
    """The output is wrapper-independent, so the early-exit signature must be too."""
    from faceforge.core.state import BodyState
    scene, wrapper, nodes, sk = _rig()
    sk.scene_wrapper = wrapper
    state = BodyState()
    before = sk._compute_signature(state)
    wrapper.set_position(12.0, 250.0, -3.0)
    wrapper.set_quaternion(quat_from_axis_angle(vec3(1, 0, 0), -np.pi / 2))
    assert sk._compute_signature(state) == before
    state.spine_flex = 0.3
    assert sk._compute_signature(state) != before


def test_rebuilding_the_joints_under_a_wrapper_does_not_move_the_body():
    """The rest matrices must be captured in the frame ``update`` reads them in.

    ``update`` cancels the scene wrapper from every joint matrix.  Capturing
    the rest matrix in world space instead makes the delta the wrapper's own
    transform, so a rebind while the gym wrapper was active dropped the whole
    soft tissue on the floor, rotated 90 degrees.
    """
    from faceforge.core.math_utils import quat_from_axis_angle, vec3

    scene, wrapper, nodes, sk = _rig()
    mesh = _grid_mesh()
    sk.register_skin_mesh(mesh, is_muscle=True, muscle_name=mesh.name)
    sk._last_signature = ""
    sk.update(BodyState())
    before = np.asarray(mesh.geometry.positions, dtype=np.float64).reshape(-1, 3).copy()

    wrapper.set_position(0.0, 203.0, 0.0)
    wrapper.set_quaternion(quat_from_axis_angle(vec3(1, 0, 0), -np.pi / 2))
    sk.scene_wrapper = wrapper
    scene.update()

    sk.rebuild_skin_joints([[(n.name, n) for n in nodes]])
    sk._last_signature = ""
    sk.update(BodyState())
    after = np.asarray(mesh.geometry.positions, dtype=np.float64).reshape(-1, 3)
    np.testing.assert_allclose(after, before, atol=1e-6)


def test_a_joints_rest_matrix_does_not_follow_its_node():
    """The rest matrix is a snapshot, not a view of the live world matrix.

    ``SceneNode.update_world_matrix`` rewrites world matrices *in place*, and
    ``np.asarray`` on an array that is already float64 hands back the same
    object.  ``_joint_world`` therefore used to return a live view whenever
    there was no wrapper to cancel -- which is exactly the case at load time --
    so every joint's rest transform silently became the current world matrix
    the moment the body stood up in the gym.
    """
    scene, wrapper, nodes, sk = _rig()
    rest_before = [np.asarray(j.rest_world, dtype=np.float64).copy()
                   for j in sk.joints]

    wrapper.set_position(0.0, 203.0, 0.0)
    wrapper.set_quaternion(quat_from_axis_angle(vec3(1, 0, 0), -np.pi / 2))
    scene.update()

    for joint, before in zip(sk.joints, rest_before, strict=True):
        np.testing.assert_allclose(np.asarray(joint.rest_world), before, atol=1e-12)
        assert joint.rest_world is not joint.node.world_matrix


def test_a_mesh_registered_inside_the_scene_lands_where_one_registered_outside_does():
    """The exercise flow enters the gym first and loads the muscles after it.

    Measured before the rest matrices stopped aliasing: all 138 loaded meshes
    came out in a different place depending on the order, by a median of 160.6
    units and up to 203.0 -- the forearm flexors lying on the floor behind the
    skeleton, skinned by a delta of exactly the wrapper's inverse.
    """
    # Registered before the wrapper exists, the way a clinical session loads.
    scene_a, wrapper_a, nodes_a, sk_a = _rig()
    mesh_a = _grid_mesh()
    sk_a.register_skin_mesh(mesh_a, is_muscle=True, muscle_name=mesh_a.name)
    wrapper_a.set_position(0.0, 203.0, 0.0)
    wrapper_a.set_quaternion(quat_from_axis_angle(vec3(1, 0, 0), -np.pi / 2))
    sk_a.scene_wrapper = wrapper_a
    scene_a.update()
    outside = _skin(sk_a, scene_a, mesh_a, nodes_a[1], 20.0)

    # Registered after it, the way the exercise module loads a muscle region.
    scene_b, wrapper_b, nodes_b, sk_b = _rig()
    wrapper_b.set_position(0.0, 203.0, 0.0)
    wrapper_b.set_quaternion(quat_from_axis_angle(vec3(1, 0, 0), -np.pi / 2))
    sk_b.scene_wrapper = wrapper_b
    scene_b.update()
    mesh_b = _grid_mesh()
    sk_b.register_skin_mesh(mesh_b, is_muscle=True, muscle_name=mesh_b.name)
    inside = _skin(sk_b, scene_b, mesh_b, nodes_b[1], 20.0)

    np.testing.assert_allclose(inside, outside, atol=1e-6)


def test_an_unmoved_joints_delta_is_the_identity_after_the_body_enters_a_scene():
    """``_joint_delta`` caches the inverse rest transform the first time it is
    asked, which in the exercise flow is after the gym wrapper is already up.

    With the rest matrices aliasing their nodes, that cache inverted the
    *current world* matrix, so every delta came out as the wrapper's inverse.
    The attachment pinning then carried each muscle bodily to that image of
    its rest pose: measured on Pronator Quadratus R, a centroid of
    (37.7, -3.4, -79.4) became (37.7, 79.4, -206.4).
    """
    scene, wrapper, nodes, sk = _rig()
    wrapper.set_position(0.0, 203.0, 0.0)
    wrapper.set_quaternion(quat_from_axis_angle(vec3(1, 0, 0), -np.pi / 2))
    sk.scene_wrapper = wrapper
    scene.update()
    sk._begin_frame()

    for ji in range(len(sk.joints)):
        np.testing.assert_allclose(sk._joint_delta(ji), np.eye(4), atol=1e-9)
