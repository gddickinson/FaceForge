"""The ground lock: rest-derived targets, one-step convergence, lift and travel."""

from __future__ import annotations

import numpy as np
import pytest

from faceforge.body.ground_contact import GroundLock, rest_body_position
from faceforge.core.math_utils import quat_from_axis_angle, vec3
from faceforge.core.scene_graph import Scene, SceneNode

STANDING = quat_from_axis_angle(vec3(1, 0, 0), -np.pi / 2)


def _rig():
    """bodyRoot under a wrapper, with hip -> knee -> ankle -> toe chains per side."""
    scene = Scene()
    wrapper = SceneNode("scene_wrapper")
    body = SceneNode("bodyRoot")
    scene.add(wrapper)
    wrapper.add(body)
    pivots = {}
    for side, x in (("R", 10.0), ("L", -10.0)):
        hip = SceneNode(f"hip_{side}_pivot")
        hip.set_position(x, 0.0, -80.0)
        knee = SceneNode(f"knee_{side}_pivot")
        knee.set_position(0.0, 0.0, -60.0)
        ankle = SceneNode(f"ankle_{side}_pivot")
        ankle.set_position(0.0, 0.0, -48.0)
        toe = SceneNode(f"toe_{side}_1_mt_pivot")
        toe.set_position(0.0, -10.0, -6.0)
        body.add(hip)
        hip.add(knee)
        knee.add(ankle)
        ankle.add(toe)
        pivots[f"hip_{side}"] = hip
        pivots[f"knee_{side}"] = knee
        pivots[f"ankle_{side}"] = ankle
        pivots[f"toe_{side}_1_mt"] = toe
        wrist = SceneNode(f"wrist_{side}_pivot")
        wrist.set_position(x * 4, -5.0, -83.0)
        body.add(wrist)
        pivots[f"wrist_{side}"] = wrist
    wrapper.set_position(0.0, 203.0, 0.0)
    wrapper.set_quaternion(STANDING.copy())
    scene.update()
    return scene, wrapper, pivots


def test_rest_body_position_sums_the_chain():
    _scene, _wrapper, pivots = _rig()
    toe = rest_body_position(pivots["toe_R_1_mt"])
    assert toe == pytest.approx([10.0, -10.0, -194.0])


def test_calibrated_targets_put_the_lowest_foot_pivot_at_its_standing_height():
    scene, wrapper, pivots = _rig()
    lock = GroundLock("feet")
    lock.calibrate(pivots, (0.0, 203.0, 0.0), STANDING)
    assert lock._target_y == pytest.approx(203.0 - 194.0)
    assert lock._target_xz == pytest.approx([0.0, 0.0])
    delta = lock.update(wrapper, pivots)
    assert np.allclose(delta, 0.0, atol=1e-9), "a rest pose needs no correction"


def test_knee_flexion_is_corrected_in_one_step():
    scene, wrapper, pivots = _rig()
    lock = GroundLock("feet")
    lock.calibrate(pivots, (0.0, 203.0, 0.0), STANDING)
    # Flex both knees 90 deg: the feet swing up and back.
    for side in "RL":
        pivots[f"knee_{side}"].set_quaternion(quat_from_axis_angle(vec3(1, 0, 0), np.pi / 2))
    scene.update()
    before = min(pivots[f"toe_{s}_1_mt"].get_world_position()[1] for s in "RL")
    assert before > 20.0
    delta = lock.update(wrapper, pivots)
    scene.update()
    lowest = min(n.get_world_position()[1] for k, n in pivots.items()
                 if k.startswith(("toe", "ankle")))
    assert lowest == pytest.approx(lock._target_y, abs=1e-6)
    assert delta[1] < 0
    # A second update changes nothing.
    assert np.allclose(lock.update(wrapper, pivots), 0.0, atol=1e-9)


def test_lift_and_travel_offset_the_target():
    scene, wrapper, pivots = _rig()
    lock = GroundLock("feet")
    lock.calibrate(pivots, (0.0, 203.0, 0.0), STANDING)
    lock.update(wrapper, pivots, lift=30.0, travel=(5.0, -7.0))
    scene.update()
    y, xz = lock.measure(pivots)
    assert y == pytest.approx(lock._target_y + 30.0)
    assert xz == pytest.approx([5.0, -7.0])


def test_horizontal_lock_can_be_disabled():
    scene, wrapper, pivots = _rig()
    lock = GroundLock("feet", lock_horizontal=False)
    lock.calibrate(pivots, (0.0, 203.0, 0.0), STANDING)
    wrapper.set_position(40.0, 203.0, 0.0)
    scene.update()
    delta = lock.update(wrapper, pivots)
    assert delta[0] == 0.0 and delta[2] == 0.0


def test_hands_anchor_targets_the_floor_and_an_explicit_point():
    scene, wrapper, pivots = _rig()
    lock = GroundLock("hands")
    lock.calibrate(pivots, (0.0, 203.0, 0.0), STANDING, floor_y=0.0)
    assert lock._target_y == pytest.approx(GroundLock.HAND_CONTACT_HEIGHT)
    lock.set_target((0.0, 275.0, 0.0))
    lock.update(wrapper, pivots)
    scene.update()
    y, xz = lock.measure(pivots)
    assert y == pytest.approx(275.0)
    assert xz == pytest.approx([0.0, 0.0], abs=1e-6)


def test_none_anchor_is_inert():
    scene, wrapper, pivots = _rig()
    lock = GroundLock("none")
    assert lock.calibrated
    assert np.allclose(lock.update(wrapper, pivots), 0.0)


def test_unknown_anchor_is_rejected():
    with pytest.raises(ValueError):
        GroundLock("knees")


def _add_fingers(pivots):
    """Curled-finger pivots 13 units beyond each wrist (body -Z), so the ring
    centre is not the wrist."""
    for side in ("R", "L"):
        wrist = pivots[f"wrist_{side}"]
        for digit in (2, 3, 4, 5):
            for seg, dz in (("prox", -12.0), ("mid", -14.0), ("dist", -13.0)):
                node = SceneNode(f"finger_{side}_{digit}_{seg}_pivot")
                node.set_position(float(digit - 3.5), 0.0, dz)
                wrist.add(node)
                pivots[f"finger_{side}_{digit}_{seg}"] = node


def test_a_hanging_body_hangs_by_its_closed_fingers_not_its_wrists():
    scene, wrapper, pivots = _rig()
    _add_fingers(pivots)
    scene.update()
    lock = GroundLock("hands")
    lock.calibrate(pivots, (0.0, 203.0, 0.0), STANDING, floor_y=0.0)
    lock.set_target((0.0, 275.0, 0.0))
    lock.update(wrapper, pivots)
    scene.update()
    y, _xz = lock.measure(pivots)
    assert y == pytest.approx(275.0), "the finger ring is on the bar"
    wrist_y = pivots["wrist_R"].get_world_position()[1]
    assert wrist_y > 275.0 + 10.0, "the wrist is above the bar, the hand closed round it"
    # Hands on the floor (no explicit target) still measure the wrists.
    floor = GroundLock("hands")
    floor.calibrate(pivots, (0.0, 203.0, 0.0), STANDING, floor_y=0.0)
    assert not getattr(floor, "_grip", False)
