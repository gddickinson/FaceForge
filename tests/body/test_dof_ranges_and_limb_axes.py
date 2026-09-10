"""The DOF table, and the limb axes it drives in BodyAnimationSystem.

The body frame is Z-up with -Y anterior.  These tests pin the axis each DOF
rotates about and the direction a positive value moves the limb, measured on
a stub pivot rig, so a future port cannot quietly swap abduction and axial
rotation again.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceforge.body.body_animation import BodyAnimationSystem
from faceforge.body.dof_ranges import (
    DOF_RANGES, POSE_DOF_FIELDS, degrees_to_dof, dof_range, dof_side, dof_term,
    dof_to_degrees, is_pose_dof,
)
from faceforge.core.math_utils import quat_rotate_vec3, vec3
from faceforge.core.scene_graph import SceneNode
from faceforge.core.state import BodyState


def test_every_pose_field_on_body_state_is_in_the_table_and_vice_versa():
    assert set(POSE_DOF_FIELDS) == set(BodyState.POSE_FIELDS)


def test_conversions_round_trip():
    assert dof_to_degrees("hip_r_flex", 1.0) == 90.0
    assert degrees_to_dof("knee_l_flex", 145.0) == pytest.approx(1.0)
    assert dof_side("hip_r_flex") == "R" and dof_side("finger_curl_l") == "L"
    assert dof_side("spine_flex") is None
    assert is_pose_dof("ankle_l_invert") and not is_pose_dof("breath_depth")
    assert dof_term("ankle_r_flex", 0.5) == "dorsiflexion"
    assert dof_term("ankle_r_flex", -0.5) == "plantarflexion"
    assert dof_range("shoulder_l_abduct").joint == "shoulder"


def test_ranges_match_the_documented_degrees():
    by_pattern = {r.pattern: r.degrees for r in DOF_RANGES}
    assert by_pattern["shoulder_{s}_flex"] == 90 and by_pattern["elbow_{s}_flex"] == 145
    assert by_pattern["hip_{s}_flex"] == 90 and by_pattern["knee_{s}_flex"] == 145
    assert by_pattern["ankle_{s}_flex"] == 45 and by_pattern["spine_flex"] == 45


class _Joints:
    def __init__(self):
        self.pivots = {}
        for side in "RL":
            for name in ("shoulder", "scapula", "elbow", "hip", "knee", "ankle", "wrist"):
                self.pivots[f"{name}_{side}"] = SceneNode(f"{name}_{side}_pivot")


@pytest.fixture
def system():
    return BodyAnimationSystem(_Joints())


def _limb_direction(system, pivot: str) -> np.ndarray:
    """Where a limb hanging along -Z points after the pivot's rotation."""
    q = system.joints.pivots[pivot].quaternion
    return quat_rotate_vec3(q, vec3(0.0, 0.0, -1.0))


def _apply(system, **dofs):
    state = BodyState()
    for k, v in dofs.items():
        setattr(state, k, v)
    system._apply_limbs(state)


def test_shoulder_flexion_swings_the_arm_anteriorly(system):
    _apply(system, shoulder_r_flex=1.0)
    d = _limb_direction(system, "shoulder_R")
    assert d[1] < -0.99, "flexion 90 deg points the arm anterior (-Y)"


def test_shoulder_abduction_swings_laterally_on_both_sides(system):
    _apply(system, shoulder_r_abduct=1.0, shoulder_l_abduct=1.0)
    r = _limb_direction(system, "shoulder_R")
    l_ = _limb_direction(system, "shoulder_L")
    assert r[0] > 0.99 and abs(r[1]) < 1e-6, "right arm abducts to +X"
    assert l_[0] < -0.99, "left arm abducts to -X"


def test_shoulder_rotation_is_axial_and_does_not_move_the_hanging_arm(system):
    _apply(system, shoulder_r_rotate=1.0)
    d = _limb_direction(system, "shoulder_R")
    assert d == pytest.approx([0.0, 0.0, -1.0], abs=1e-9)
    # ... but it does turn the lateral surface posteriorly (external rotation).
    q = system.joints.pivots["shoulder_R"].quaternion
    lateral = quat_rotate_vec3(q, vec3(1.0, 0.0, 0.0))
    assert lateral[1] > 0.5


def test_scapula_follows_abduction_about_the_same_axis(system):
    _apply(system, shoulder_r_abduct=1.0)
    q = system.joints.pivots["scapula_R"].quaternion
    assert abs(q[1]) > 0.2 and abs(q[0]) < 1e-9 and abs(q[2]) < 1e-9


def test_hip_abduction_and_knee_flexion_directions(system):
    _apply(system, hip_r_abduct=1.0, hip_l_abduct=1.0, knee_r_flex=0.5)
    assert _limb_direction(system, "hip_R")[0] > 0.5
    assert _limb_direction(system, "hip_L")[0] < -0.5
    knee = _limb_direction(system, "knee_R")
    assert knee[1] > 0.5, "knee flexion carries the shank posteriorly (+Y)"


def test_ankle_dorsiflexion_lifts_the_toes(system):
    _apply(system, ankle_r_flex=1.0)
    q = system.joints.pivots["ankle_R"].quaternion
    toes = quat_rotate_vec3(q, vec3(0.0, -1.0, 0.0))    # the foot points anterior
    assert toes[2] > 0.5, "dorsiflexion raises the toes (+Z)"


def test_overhead_reach_uses_the_extended_range(system):
    _apply(system, shoulder_r_flex=2.0)
    d = _limb_direction(system, "shoulder_R")
    assert d[2] > 0.99, "flexion 180 deg points the arm straight up"
