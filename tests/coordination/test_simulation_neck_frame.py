"""The simulation hands the neck path a body-frame view of the skeleton.

Step 9 reads live pivots; steps 10 and 10.5 subtract rest snapshots taken at
load time, before any scene wrapper existed.  These tests drive a real
:class:`Simulation` with stub systems and assert the ordering and the frame:
the bone registry's cancel is set *before* the neck muscles update, and the
anchors the neck muscles receive do not move when only the wrapper does.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceforge.coordination.simulation import Simulation
from faceforge.core.math_utils import quat_from_axis_angle
from faceforge.core.scene_graph import Scene, SceneNode
from faceforge.core.state import StateManager


class _Joints:
    def __init__(self, pivots):
        self.pivots = pivots


class _BodyAnimation:
    def __init__(self, t1, pivots, ribs):
        self.thoracic_pivots = [{"group": t1}]
        self.joints = _Joints(pivots)
        self._rib_pivots = ribs

    def apply(self, body, dt):
        pass


class _NeckMuscles:
    """Records the order of what it was told, and what it was told."""

    def __init__(self, log):
        self._log = log
        self.anchors = None

    def set_body_anchors_current(self, anchors):
        self.anchors = {k: np.asarray(v, dtype=np.float64) for k, v in anchors.items()}
        self._log.append("anchors")

    def update(self, head_quat, face_state=None, body_state=None):
        self._log.append("neck-update")


class _BoneAnchors:
    def __init__(self, log):
        self._log = log
        self.cancel = "unset"

    def set_frame_cancel(self, cancel):
        self.cancel = cancel
        self._log.append("cancel")


def _sim(with_wrapper: bool):
    scene = Scene()
    body = SceneNode("bodyRoot")
    wrapper = None
    if with_wrapper:
        wrapper = SceneNode("scene_wrapper")
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

    group = SceneNode("neckMuscleGroup")
    body.add(group)
    scene.update()

    log: list[str] = []
    sim = Simulation(StateManager(), scene)
    sim.body_animation = _BodyAnimation(t1, pivots, ribs)
    sim.neck_muscles = _NeckMuscles(log)
    sim.neck_muscle_group = group
    sim.bone_anchors = _BoneAnchors(log)
    sim.scene_wrapper = wrapper
    return sim, scene, wrapper, t1, log


def test_the_bone_frame_is_set_before_the_neck_muscles_update():
    sim, _, _, _, log = _sim(with_wrapper=True)
    sim.step(1 / 60)
    assert log.index("cancel") < log.index("neck-update")
    assert log.index("anchors") < log.index("neck-update")


def test_the_neck_anchors_do_not_change_when_only_the_wrapper_moves():
    sim, scene, wrapper, _, _ = _sim(with_wrapper=True)
    sim.step(1 / 60)
    before = dict(sim.neck_muscles.anchors)

    # A squat's trunk lean is a wrapper pitch: the neck rides it through the
    # scene graph and must not also receive it as a body delta.
    wrapper.set_quaternion(quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), -1.2))
    wrapper.set_position(0.0, 180.0, 40.0)
    scene.update()
    sim.step(1 / 60)

    for name, value in before.items():
        assert sim.neck_muscles.anchors[name] == pytest.approx(value, abs=1e-9), name


def test_the_anchors_match_the_unwrapped_body():
    wrapped, _, _, _, _ = _sim(with_wrapper=True)
    plain, _, _, _, _ = _sim(with_wrapper=False)
    wrapped.step(1 / 60)
    plain.step(1 / 60)
    for name, value in plain.neck_muscles.anchors.items():
        assert wrapped.neck_muscles.anchors[name] == pytest.approx(value, abs=1e-9), name


def test_real_thoracic_flexion_still_reaches_the_neck():
    sim, scene, _, t1, _ = _sim(with_wrapper=True)
    sim.step(1 / 60)
    before = dict(sim.neck_muscles.anchors)
    t1.set_position(0.0, 4.0, -11.0)
    scene.update()
    sim.step(1 / 60)
    moved = np.linalg.norm(sim.neck_muscles.anchors["thoracic"] - before["thoracic"])
    assert moved == pytest.approx(1.0, abs=1e-9)


def test_outside_scene_mode_there_is_nothing_to_cancel():
    sim, _, _, _, _ = _sim(with_wrapper=False)
    sim.step(1 / 60)
    assert sim.bone_anchors.cancel is None


def test_the_wrapper_is_read_from_the_skinning_when_not_set_directly():
    sim, _, wrapper, _, _ = _sim(with_wrapper=True)
    sim.scene_wrapper = None

    class _Skinning:
        scene_wrapper = wrapper

    sim.soft_tissue = _Skinning()
    assert sim.scene_wrapper is wrapper
    assert sim.frame_cancel() is not None
