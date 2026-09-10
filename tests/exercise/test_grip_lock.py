"""The grip-width lock keeps hands on a bar from sliding as the arms flex."""

from __future__ import annotations

import numpy as np
import pytest

from faceforge.core.scene_graph import SceneNode
from faceforge.exercise.grip_lock import GripWidthLock


class _Rig:
    """A fake body animation: each hand's finger ring sits at x = sign * (20 + 30 * abduct)
    -- like a straight arm swinging in the coronal plane -- plus an elbow term
    that widens the grip as the elbow flexes (the pull-up's drift)."""

    def __init__(self):
        self.pivots = {}
        for side, sign in (("R", 1.0), ("L", -1.0)):
            shoulder = SceneNode(f"shoulder_{side}_pivot")
            shoulder.set_position(sign * 20.0, 0.0, 0.0)
            self.pivots[f"shoulder_{side}"] = shoulder
            for digit in (2, 3, 4, 5):
                for seg in ("prox", "mid", "dist"):
                    node = SceneNode(f"finger_{side}_{digit}_{seg}_pivot")
                    self.pivots[f"finger_{side}_{digit}_{seg}"] = node

    def apply(self, state, dt):
        for side, sign in (("R", 1.0), ("L", -1.0)):
            ab = getattr(state, f"shoulder_{side.lower()}_abduct")
            el = getattr(state, f"elbow_{side.lower()}_flex")
            x = sign * (20.0 + 30.0 * ab + 12.0 * el)
            for digit in (2, 3, 4, 5):
                for seg, dz in (("prox", 0.0), ("mid", -3.0), ("dist", -5.0)):
                    self.pivots[f"finger_{side}_{digit}_{seg}"].set_position(x + digit * 0.5, 0.0, dz)

    def update(self):
        for node in self.pivots.values():
            node.update_world_matrix(force=True)


def test_hands_stay_at_their_calibrated_offsets_as_the_elbows_flex():
    rig = _Rig()
    lock = GripWidthLock(rig, rig, rig.pivots)
    hang = {"shoulder_r_abduct": 1.0, "shoulder_l_abduct": 1.0, "elbow_r_flex": 0.0, "elbow_l_flex": 0.0}
    assert lock.calibrate(hang)
    # Ring x = 20 + 30 + mean digit offset 1.75 = 51.75 on the right, measured
    # from the shoulder midpoint at x = 0.
    assert lock.targets["R"] == pytest.approx(51.75, abs=1e-6)
    assert lock.targets["L"] == pytest.approx(-48.25, abs=1e-6)
    top = {"shoulder_r_abduct": 1.0, "shoulder_l_abduct": 1.0, "elbow_r_flex": 0.9, "elbow_l_flex": 0.9}
    lock.apply(top)
    # The elbow term added 10.8 units per side; abduction gave it back: 30 * d = -10.8.
    assert top["shoulder_r_abduct"] == pytest.approx(1.0 - 10.8 / 30.0, abs=1e-4)
    assert top["shoulder_l_abduct"] == pytest.approx(1.0 - 10.8 / 30.0, abs=1e-4)
    rig.apply(_state(top), 0.0)
    rig.update()
    offsets = lock._offsets()
    assert offsets["R"] == pytest.approx(lock.targets["R"], abs=1e-3)


def _state(d):
    from faceforge.core.state import BodyState
    s = BodyState()
    s.set_from_js_dict(d)
    return s


def test_without_finger_pivots_the_lock_is_inert():
    class _Bare:
        pivots = {}

        def apply(self, state, dt):
            pass

        def update(self):
            pass

    bare = _Bare()
    lock = GripWidthLock(bare, bare, bare.pivots)
    d = {"shoulder_r_abduct": 0.5, "shoulder_l_abduct": 0.5}
    assert lock.apply(d) == {}
    assert d["shoulder_r_abduct"] == 0.5 and not lock.calibrated


def test_an_abduction_beyond_the_old_bound_is_left_alone_when_the_hands_are_on_target():
    """A dead hang is 165 deg = 1.83 normalised; a zero-error frame must not be clipped."""
    rig = _Rig()
    lock = GripWidthLock(rig, rig, rig.pivots)
    hang = {"shoulder_r_abduct": 1.83, "shoulder_l_abduct": 1.83, "elbow_r_flex": 0.0, "elbow_l_flex": 0.0}
    assert lock.calibrate(hang)
    same = dict(hang)
    lock.apply(same)
    assert same["shoulder_r_abduct"] == pytest.approx(1.83, abs=1e-6)
    assert same["shoulder_l_abduct"] == pytest.approx(1.83, abs=1e-6)
