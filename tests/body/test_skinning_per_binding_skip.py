"""A muscle whose driving joints did not move is not re-skinned; one whose joints moved is."""

from __future__ import annotations

import numpy as np

from faceforge.body.skinning_ops import used_joints
from faceforge.core.math_utils import quat_from_axis_angle, vec3
from faceforge.core.state import BodyState
from tests.body.test_skinning_under_scene_wrapper import _grid_mesh, _rig


def _shifted(dz: float):
    """The grid mesh moved along the spine so it binds to a different pair of joints."""
    m = _grid_mesh()
    pos = m.geometry.positions.reshape(-1, 3)
    pos[:, 2] += dz
    m.store_rest_pose()
    m.name = f"Shifted {dz:+.0f} R"
    return m


def _frame(sk, scene):
    scene.update()
    sk._last_signature = "stale"     # a new pose, not a reset: the per-binding skip may apply
    sk.update(BodyState())


def _drivers(sk, binding) -> set[int]:
    return set(used_joints(binding.joint_indices, binding.secondary_indices, len(sk.joints)).tolist())


def test_only_the_bindings_whose_joints_moved_are_recomputed():
    scene, wrapper, nodes, sk = _rig()      # a chain of four joints, 0 at the root
    low = _shifted(-60.0)
    high = _shifted(+60.0)
    sk.register_skin_mesh(low, is_muscle=True, muscle_name=low.name)
    sk.register_skin_mesh(high, is_muscle=True, muscle_name=high.name)
    b_low, b_high = sk.bindings[-2], sk.bindings[-1]
    assert _drivers(sk, b_low) <= {0, 1} and 3 in _drivers(sk, b_high), (
        _drivers(sk, b_low), _drivers(sk, b_high))

    _frame(sk, scene)                        # first frame: everything computed
    low.needs_update = high.needs_update = False

    # Rotate the top joint only: it is below joints 0 and 1 in the chain, so
    # the low mesh's drivers are exactly where they were.
    nodes[3].set_quaternion(quat_from_axis_angle(vec3(1, 0, 0), np.radians(30.0)))
    _frame(sk, scene)
    assert high.needs_update is True
    assert low.needs_update is False, "a muscle none of whose joints moved was re-skinned"

    # A reset recomputes everything although nothing moved.
    low.needs_update = high.needs_update = False
    sk._last_signature = ()
    scene.update()
    sk.update(BodyState())
    assert low.needs_update is True and high.needs_update is True

    # Rotating joint 1 moves every joint below it: both meshes recompute.
    low.needs_update = high.needs_update = False
    nodes[1].set_quaternion(quat_from_axis_angle(vec3(1, 0, 0), np.radians(20.0)))
    _frame(sk, scene)
    assert low.needs_update is True and high.needs_update is True
