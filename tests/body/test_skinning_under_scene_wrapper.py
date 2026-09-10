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
