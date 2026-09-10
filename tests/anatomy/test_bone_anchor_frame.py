"""Bone anchors are read in the body frame, and footprint pinning follows the joint rigidly.

Muscle vertices live in the body frame (the skinning cancels the scene wrapper
from its joint deltas), but the bone registry read bone positions in WORLD
space.  In any scene the pin target was therefore off by the wrapper's
transform and the pinning tore the attachment zones apart (measured: a
rotator-cuff p99 edge stretch of 13.6x with pinning, 1.6x without).
"""

from __future__ import annotations

import numpy as np
import pytest

from faceforge.anatomy.bone_anchors import BoneAnchorRegistry
from faceforge.anatomy.muscle_attachments import MuscleAttachmentSystem
from faceforge.core.math_utils import mat4_inverse, quat_from_axis_angle, vec3
from faceforge.core.mesh import BufferGeometry, MeshInstance
from faceforge.core.scene_graph import Scene, SceneNode
from faceforge.core.material import Material


def _bone(name, pos):
    node = SceneNode(name)
    node.set_position(*pos)
    return node


def test_current_positions_are_brought_back_into_the_rest_frame():
    scene = Scene()
    wrapper = SceneNode("scene_wrapper")
    body = SceneNode("bodyRoot")
    scene.add(wrapper)
    wrapper.add(body)
    bone = _bone("Right Humerus", (10.0, 0.0, -40.0))
    body.add(bone)
    scene.update()
    reg = BoneAnchorRegistry()
    reg.register_bones({"Right Humerus": bone})
    reg.snapshot_rest_positions()
    rest = reg.get_muscle_anchor("m", ["Right Humerus"])

    wrapper.set_position(0.0, 203.0, 0.0)
    wrapper.set_quaternion(quat_from_axis_angle(vec3(1, 0, 0), -np.pi / 2))
    scene.update()
    assert not np.allclose(reg.get_muscle_anchor_current("m", ["Right Humerus"]), rest), \
        "the world position moved with the wrapper (the defect)"
    wrapper.update_world_matrix(force=True)
    reg.set_frame_cancel(mat4_inverse(wrapper.world_matrix))
    assert np.allclose(reg.get_muscle_anchor_current("m", ["Right Humerus"]), rest, atol=1e-9)
    reg.set_frame_cancel(None)
    assert not np.allclose(reg.get_muscle_anchor_current("m", ["Right Humerus"]), rest)


class _Binding:
    def __init__(self, n):
        pos = np.random.default_rng(0).normal(size=(n, 3)).astype(np.float32)
        geom = BufferGeometry(positions=pos.ravel().copy(), normals=np.zeros(n * 3, np.float32),
                              vertex_count=n)
        self.mesh = MeshInstance(name="Test Muscle R", geometry=geom, material=Material())
        self.mesh.rest_positions = pos.ravel().copy()
        self.muscle_name = "Test Muscle R"
        self.joint_indices = np.zeros(n, dtype=np.int64)
        self.secondary_indices = np.zeros(n, dtype=np.int64)
        self.weights = np.ones(n, dtype=np.float32)
        self.edge_pairs = None


def test_footprint_pinning_targets_the_rigid_image_under_the_attachment_joint():
    reg = BoneAnchorRegistry()
    sys_ = MuscleAttachmentSystem(reg)
    b = _Binding(20)
    sys_.register_muscle(b, ["Right Scapula"], ["Right Humerus"])
    data = sys_._attachments[id(b)]
    data.footprint_masks = True
    data.origin_joint, data.insertion_joint = 0, 1
    om = np.zeros(20, bool)
    om[:5] = True
    im = np.zeros(20, bool)
    im[15:] = True
    data.origin_mask, data.insertion_mask = om, im
    data.pin_strength = 1.0                     # full pull, to read the target exactly

    rot = np.eye(4)
    rot[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]  # 90 deg about Z
    rot[:3, 3] = [5.0, 0.0, 0.0]
    deltas = {0: np.eye(4), 1: rot}

    # Scramble the current positions, then pin.
    b.mesh.geometry.positions = (b.mesh.rest_positions.reshape(-1, 3) + 3.0).ravel().copy()
    sys_.apply_bone_pinning(b, joint_delta=lambda j: deltas[j])
    pos = b.mesh.geometry.positions.reshape(-1, 3)
    rest = b.mesh.rest_positions.reshape(-1, 3)
    assert np.allclose(pos[:5], rest[:5], atol=1e-5), "origin footprint pinned to its joint's rest image"
    expected = rest[15:] @ rot[:3, :3].T + rot[:3, 3]
    assert np.allclose(pos[15:], expected, atol=1e-5), "insertion pinned to the rotated image"
    assert np.allclose(pos[5:15], rest[5:15] + 3.0), "the belly is left to the skinning"
