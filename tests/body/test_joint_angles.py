"""The two dimorphic features that are angles, not proportions.

Scaling bones cannot produce either of them, and measured on the model before
this existed neither moved by a hundredth of a degree between the sexes.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceforge.body import joint_angles
from faceforge.core.mesh import BufferGeometry, MeshInstance
from faceforge.core.scene_graph import SceneNode


def bone(name: str, points) -> SceneNode:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    geo = BufferGeometry(positions=pts.reshape(-1).copy(),
                         normals=np.zeros(pts.size, dtype=np.float32))
    node = SceneNode(name=name)
    node.mesh = MeshInstance(name=name, geometry=geo)
    node.mesh.store_rest_pose()
    return node


@pytest.fixture
def arm():
    """bodyRoot -> elbow_R_pivot -> wrist_R_pivot, with a forearm between."""
    root = SceneNode(name="bodyRoot")
    elbow = SceneNode(name="elbow_R_pivot")
    elbow.set_position(20.0, 0.0, -40.0)
    elbow.add(bone("Right Radius", [[0, 0, 0], [0, 0, -30]]))
    root.add(elbow)
    wrist = SceneNode(name="wrist_R_pivot")
    wrist.set_position(0.0, 0.0, -30.0)
    elbow.add(wrist)
    wrist.add(bone("R Scaphoid", [[0, 0, 0], [0, 0, -4]]))
    return root, elbow, wrist


def find(root, name):
    node = root.find(name)
    assert node is not None, name
    return node


def world(node, root) -> np.ndarray:
    out = np.zeros(3)
    n = node
    while n is not None and n is not root:
        out = out + np.asarray(n.position, dtype=np.float64)
        n = n.parent
    return out


def test_a_male_skeleton_is_left_exactly_as_it_is(arm):
    root, _elbow, wrist = arm
    before = np.asarray(wrist.position, dtype=np.float64).copy()
    assert joint_angles.apply(root, 0.0) == 0
    assert np.allclose(wrist.position, before)


def test_the_forearm_swings_laterally_on_the_right(arm):
    root, _elbow, wrist = arm
    joint_angles.apply(root, 1.0)
    # The wrist hangs below the elbow, so opening the carrying angle carries
    # it away from the midline: +x on the right.
    assert wrist.position[0] > 0.0
    assert np.linalg.norm(wrist.position) == pytest.approx(30.0), \
        "a rotation may not change the forearm's length"


def test_the_left_side_swings_the_other_way():
    root = SceneNode(name="bodyRoot")
    elbow = SceneNode(name="elbow_L_pivot")
    elbow.set_position(-20.0, 0.0, -40.0)
    root.add(elbow)
    wrist = SceneNode(name="wrist_L_pivot")
    wrist.set_position(0.0, 0.0, -30.0)
    elbow.add(wrist)
    joint_angles.apply(root, 1.0)
    assert wrist.position[0] < 0.0


def test_the_bones_below_the_joint_turn_with_it(arm):
    root, elbow, wrist = arm
    radius = find(root, "Right Radius")
    joint_angles.apply(root, 1.0)
    tip = np.asarray(radius.mesh.geometry.positions,
                     dtype=np.float64).reshape(-1, 3)[-1]
    assert tip[0] > 0.0, "the forearm bone must swing with its joint"
    assert np.linalg.norm(tip) == pytest.approx(30.0)
    hand = find(root, "R Scaphoid")
    hand_tip = np.asarray(hand.mesh.geometry.positions,
                          dtype=np.float64).reshape(-1, 3)[-1]
    assert hand_tip[0] > 0.0, "and so must everything below it"


def test_the_change_is_proportional_to_the_slider(arm):
    root, _elbow, wrist = arm
    joint_angles.apply(root, 0.5)
    half = np.asarray(wrist.position, dtype=np.float64).copy()
    wrist.set_position(0.0, 0.0, -30.0)
    joint_angles.apply(root, 1.0)
    full = np.asarray(wrist.position, dtype=np.float64)
    assert 0.0 < half[0] < full[0]


def test_soft_tissue_the_skinning_owns_is_left_to_the_skinning(arm):
    root, elbow, _wrist = arm
    muscle = bone("Biceps R", [[1, 0, -10], [1, 0, -20]])
    elbow.add(muscle)
    before = np.array(muscle.mesh.geometry.positions, copy=True)
    joint_angles.apply(root, 1.0, exclude={id(muscle.mesh)})
    assert np.array_equal(muscle.mesh.geometry.positions, before)


def test_the_published_differences_are_two_degrees_at_each_joint():
    assert dict(joint_angles.SEX_DELTA) == {"elbow": 2.0, "knee": 2.0}
