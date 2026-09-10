"""Scapular upward rotation glides on the thorax; the clavicle elevates and carries the blade.

Measured on the real skeleton (2026-09-10): rotating the scapula about an
anterior-posterior axis through its centroid put the inferior angle 7 units
outside the ribcage at 165 deg of abduction, and teres major, infraspinatus
and subscapularis stood out from the trunk as a wing.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceforge.body.body_animation import BodyAnimationSystem
from faceforge.core.mesh import BufferGeometry, MeshInstance
from faceforge.core.material import Material
from faceforge.core.scene_graph import Scene, SceneNode
from faceforge.core.state import BodyState

SCAP = np.array([18.1, 9.1, -16.7])          # scapula pivot (its centroid), body frame
INFERIOR_ANGLE = np.array([13.9, 15.9, -32.0])
CLAV = np.array([1.7, -6.3, -13.1])          # sternoclavicular end
ACROMIAL_END = np.array([23.0, 4.7, -10.0])  # the clavicle's lateral vertex


class _Joints:
    def __init__(self, with_clavicle=True):
        self.scene = Scene()
        root = SceneNode("bodyRoot")
        self.scene.add(root)
        self.pivots = {}
        scap = SceneNode("scapula_R_pivot")
        scap.set_position(*SCAP)
        root.add(scap)
        self.pivots["scapula_R"] = scap
        # A marker child at the inferior angle, in the pivot's frame.
        marker = SceneNode("inferior_angle")
        marker.set_position(*(INFERIOR_ANGLE - SCAP))
        scap.add(marker)
        self.marker = marker
        if with_clavicle:
            clav = SceneNode("clavicle_R_pivot")
            clav.set_position(*CLAV)
            root.add(clav)
            bone = SceneNode("Right Clavicle")
            pts = np.array([[0.0, 0.0, 0.0], ACROMIAL_END - CLAV, [10.0, 2.0, 1.0]], dtype=np.float32)
            geom = BufferGeometry(positions=pts.ravel().copy(), normals=np.zeros(9, np.float32), vertex_count=3)
            bone.mesh = MeshInstance(name="Right Clavicle", geometry=geom, material=Material())
            clav.add(bone)
            self.pivots["clavicle_R"] = clav
            self.clav_bone = bone
        self.scene.update()


def _world(node):
    node.update_world_matrix(force=True)
    return np.asarray(node.world_matrix)[:3, 3].copy()


def _radius(p):
    return float(np.hypot(p[0], p[1]))


def test_inferior_angle_stays_on_the_ribcage_at_full_elevation():
    joints = _Joints()
    anim = BodyAnimationSystem(joints)
    anim._apply_limbs(BodyState(shoulder_r_abduct=165.0 / 90.0))
    joints.scene.update()
    ia = _world(joints.marker)
    # It glides round the thorax: further lateral and further forward, at
    # about the same distance from the thorax axis -- not 7 units outside it.
    assert ia[0] > INFERIOR_ANGLE[0] + 4.0, "moves laterally"
    assert ia[1] < INFERIOR_ANGLE[1] - 4.0, "and forward round the curve"
    assert abs(_radius(ia) - _radius(INFERIOR_ANGLE)) < 3.0, "stays on the ribcage"
    assert ia[2] > INFERIOR_ANGLE[2] + 4.0, "and rises with the clavicle"


def test_clavicle_elevates_and_the_blade_follows_its_acromial_end():
    joints = _Joints()
    anim = BodyAnimationSystem(joints)
    anim._apply_limbs(BodyState(shoulder_r_abduct=165.0 / 90.0))
    joints.scene.update()
    clav = joints.pivots["clavicle_R"]
    clav.update_world_matrix(force=True)
    m = np.asarray(clav.world_matrix)
    ac_clav = m[:3, :3] @ (ACROMIAL_END - CLAV) + m[:3, 3]
    assert ac_clav[2] > ACROMIAL_END[2] + 6.0, "the acromial end rises about 30 deg"
    scap = joints.pivots["scapula_R"]
    scap.update_world_matrix(force=True)
    ms = np.asarray(scap.world_matrix)
    ac_scap = ms[:3, :3] @ (ACROMIAL_END - SCAP) + ms[:3, 3]
    assert np.allclose(ac_scap, ac_clav, atol=1e-6), "the acromioclavicular joint stays together"


def test_neutral_leaves_the_girdle_at_rest_and_no_clavicle_still_rotates():
    joints = _Joints()
    anim = BodyAnimationSystem(joints)
    anim._apply_limbs(BodyState(shoulder_r_abduct=165.0 / 90.0))
    anim._apply_limbs(BodyState())
    joints.scene.update()
    assert np.allclose(_world(joints.marker), INFERIOR_ANGLE, atol=1e-6)
    bare = _Joints(with_clavicle=False)
    anim2 = BodyAnimationSystem(bare)
    anim2._apply_limbs(BodyState(shoulder_r_abduct=1.0))
    bare.scene.update()
    assert not np.allclose(_world(bare.marker), INFERIOR_ANGLE)
