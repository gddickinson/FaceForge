"""Neck body anchors are a body-frame quantity: the scene wrapper must not leak in.

The neck muscles blend a *body delta* -- current anchor minus rest anchor --
into their lower vertices.  The rest anchors are snapshotted at load time,
before any scene wrapper exists, so a current anchor read in world space
inside the gym carried the wrapper's Rx(-90 deg) at Y = 203 with it.  Measured
one frame into a bodyweight squat before the fix: the thoracic anchor read
(0.225, 192.088, -5.661) against a rest of (0.225, 5.661, -10.912) -- a
186-unit delta that stretched Sternohyoid L's 99th-percentile edge by 55x.

These tests pin the invariant on a synthetic rig: for the same joint pose the
anchors are identical whether or not a wrapper is above the body, and a joint
that really moves still shows up.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceforge.coordination.body_anchors import (
    body_anchor_positions,
    to_body_frame,
    wrapper_cancel,
)
from faceforge.core.math_utils import quat_from_axis_angle
from faceforge.core.scene_graph import Scene, SceneNode


class _Joints:
    def __init__(self, pivots):
        self.pivots = pivots


class _BodyAnimation:
    """The three attributes :func:`body_anchor_positions` reads."""

    def __init__(self, thoracic_pivots, joints, rib_pivots):
        self.thoracic_pivots = thoracic_pivots
        self.joints = joints
        self._rib_pivots = rib_pivots


def _rig(with_wrapper: bool):
    """scene -> [wrapper ->] bodyRoot -> the pivots the anchors are read from."""
    scene = Scene()
    body = SceneNode("bodyRoot")
    wrapper = None
    if with_wrapper:
        wrapper = SceneNode("scene_wrapper")
        # The gym placement: stand the body up and lift it onto the floor.
        wrapper.set_position(0.0, 203.0, 0.0)
        wrapper.set_quaternion(quat_from_axis_angle(np.array([1.0, 0.0, 0.0]),
                                                    -np.pi / 2))
        scene.add(wrapper)
        wrapper.add(body)
    else:
        scene.add(body)

    t1 = SceneNode("T1")
    t1.set_position(0.0, 5.0, -11.0)
    body.add(t1)

    pivots = {}
    for side, x in (("R", 16.0), ("L", -16.0)):
        n = SceneNode(f"shoulder_{side}")
        n.set_position(x, 4.0, -15.0)
        body.add(n)
        pivots[f"shoulder_{side}"] = n

    ribs = []
    for i in range(4):
        n = SceneNode(f"rib_{i}")
        n.set_position(9.0, 0.0, -14.0 - i)
        body.add(n)
        ribs.append(n)

    scene.update()
    anim = _BodyAnimation([{"group": t1}], _Joints(pivots), ribs)
    return scene, wrapper, body, anim, t1


def test_anchors_are_the_same_with_and_without_a_wrapper():
    _, _, _, plain, _ = _rig(with_wrapper=False)
    scene, wrapper, _, wrapped, _ = _rig(with_wrapper=True)

    rest = body_anchor_positions(plain, None)
    current = body_anchor_positions(wrapped, wrapper_cancel(wrapper))

    assert set(rest) == {"thoracic", "shoulder", "ribcage"}
    for name, value in rest.items():
        assert current[name] == pytest.approx(value, abs=1e-9), name


def test_reading_a_wrapped_rig_in_world_space_is_what_went_wrong():
    """The control: without the cancel the delta is the wrapper, ~186 units."""
    _, _, _, plain, _ = _rig(with_wrapper=False)
    _, _, _, wrapped, _ = _rig(with_wrapper=True)

    rest = body_anchor_positions(plain, None)
    uncancelled = body_anchor_positions(wrapped, None)

    worst = max(float(np.linalg.norm(uncancelled[k] - rest[k])) for k in rest)
    assert worst > 150.0


def test_a_joint_that_really_moves_still_shows_up_under_a_wrapper():
    scene, wrapper, _, wrapped, t1 = _rig(with_wrapper=True)
    before = body_anchor_positions(wrapped, wrapper_cancel(wrapper))

    t1.set_position(0.0, 4.0, -11.0)          # 1 unit of thoracic flexion
    scene.update()
    after = body_anchor_positions(wrapped, wrapper_cancel(wrapper))

    assert np.linalg.norm(after["thoracic"] - before["thoracic"]) == pytest.approx(1.0, abs=1e-9)
    assert np.linalg.norm(after["shoulder"] - before["shoulder"]) == pytest.approx(0.0, abs=1e-9)


def test_moving_the_wrapper_alone_moves_no_anchor():
    """A squat's trunk lean is a wrapper pitch; the neck rides it through the
    scene graph, so it must not also be blended in as a body delta."""
    scene, wrapper, _, wrapped, _ = _rig(with_wrapper=True)
    before = body_anchor_positions(wrapped, wrapper_cancel(wrapper))

    wrapper.set_quaternion(quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), -1.2))
    wrapper.set_position(0.0, 180.0, 40.0)
    scene.update()
    after = body_anchor_positions(wrapped, wrapper_cancel(wrapper))

    for name in before:
        assert after[name] == pytest.approx(before[name], abs=1e-9), name


def test_a_wrapper_outside_the_scene_graph_cancels_nothing():
    orphan = SceneNode("scene_wrapper")
    orphan.set_position(0.0, 203.0, 0.0)
    assert wrapper_cancel(orphan) is None
    assert wrapper_cancel(None) is None


def test_to_body_frame_is_the_identity_without_a_cancel():
    p = np.array([1.0, 2.0, 3.0])
    assert to_body_frame(p, None) == pytest.approx(p)


def test_missing_pivots_yield_an_empty_dict_not_a_partial_one():
    assert body_anchor_positions(None, None) == {}
    empty = _BodyAnimation([], _Joints({}), [])
    assert body_anchor_positions(empty, None) == {}
