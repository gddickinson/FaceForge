"""A hand that can close round a bar: joint-placed digit pivots and per-joint flexion.

Every render had the bar passing through the palm.  Two causes, both here:
the digit pivots sat at bone centroids (a phalanx rotating about its middle
opens the joint) and 90 degrees of curl was shared across four joints, so a
full curl barely bent the fingers.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from faceforge.body.body_animation import BodyAnimationSystem
from faceforge.body.joint_pivots import proximal_end
from faceforge.core.scene_graph import SceneNode
from faceforge.core.state import BodyState


def _phalanx(length=10.0, n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, length, n)
    ang = rng.uniform(0.0, 2 * np.pi, n)
    return np.stack([x, 1.2 * np.cos(ang), 1.2 * np.sin(ang)], axis=1)


def test_proximal_end_is_the_end_nearest_the_parent():
    pts = _phalanx()
    base = proximal_end(pts, np.array([-5.0, 0.0, 0.0]))
    tip = proximal_end(pts, np.array([15.0, 0.0, 0.0]))
    assert base[0] < 1.5 and tip[0] > 8.5
    assert abs(base[1]) < 0.6 and abs(base[2]) < 0.6, "on the bone axis, not a surface point"


def _angle_deg(node: SceneNode) -> float:
    q = np.asarray(node.quaternion, dtype=np.float64)
    return math.degrees(2.0 * math.atan2(float(np.linalg.norm(q[:3])), float(q[3])))


class _Joints:
    def __init__(self):
        self.pivots = {}
        for side in ("R", "L"):
            for digit, segs in ((1, ("mc", "prox", "dist")), (2, ("mc", "prox", "mid", "dist"))):
                for seg in segs:
                    name = f"finger_{side}_{digit}_{seg}"
                    self.pivots[name] = SceneNode(f"{name}_pivot")


def test_each_finger_joint_flexes_to_its_own_maximum():
    joints = _Joints()
    anim = BodyAnimationSystem(joints)
    anim._apply_hands(BodyState(finger_curl_r=1.0))
    p = joints.pivots
    assert _angle_deg(p["finger_R_2_prox"]) == pytest.approx(90.0, abs=0.01), "MCP"
    assert _angle_deg(p["finger_R_2_mid"]) == pytest.approx(100.0, abs=0.01), "PIP"
    assert _angle_deg(p["finger_R_2_dist"]) == pytest.approx(60.0, abs=0.01), "DIP"
    assert _angle_deg(p["finger_R_2_mc"]) < 10.0, "the carpometacarpal joint barely moves"
    total = sum(_angle_deg(p[f"finger_R_2_{s}"]) for s in ("prox", "mid", "dist"))
    assert total > 240.0, "a closed fist, not the old 90 degrees shared over four joints"
    assert _angle_deg(p["finger_L_2_mid"]) == pytest.approx(0.0, abs=1e-9), "left hand untouched"


def test_a_bar_grip_pose_closes_the_hand():
    from faceforge.exercise.pose_library import grip

    pose = grip()
    joints = _Joints()
    anim = BodyAnimationSystem(joints)
    anim._apply_hands(BodyState(**pose))
    assert _angle_deg(joints.pivots["finger_R_2_prox"]) == pytest.approx(85.0, abs=0.5)
    assert _angle_deg(joints.pivots["finger_R_2_mid"]) > 90.0


def test_hyperextension_is_confined_to_the_knuckle():
    joints = _Joints()
    anim = BodyAnimationSystem(joints)
    anim._apply_hands(BodyState(finger_curl_r=-1.0))
    p = joints.pivots
    assert _angle_deg(p["finger_R_2_prox"]) == pytest.approx(20.0, abs=0.01)
    assert _angle_deg(p["finger_R_2_mid"]) == pytest.approx(2.0, abs=0.01)
    assert _angle_deg(p["finger_R_2_mc"]) == pytest.approx(0.0, abs=1e-9)
