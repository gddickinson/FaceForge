"""Both feet on the floor: the higher leg's knee is bent until it reaches down.

The ground lock translates the whole body, which is one correction for two
feet.  It is enough only while the feet end a pose at the same height, and the
donor's legs are not identical -- measured at the bottom of a bodyweight squat,
the two ankles ended 6.8 units apart.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceforge.body.foot_level import (
    ANKLE_PER, ANKLE_PER_KNEE, JOINT_COST, KNEE_RANGE, TOLERANCE, FootLevelLock,
)


class FakeNode:
    def __init__(self, y: float) -> None:
        self._y = y

    def get_world_position(self):
        return np.array([0.0, self._y, 0.0])


class Rig:
    """A leg per side whose foot height is a known function of the knee.

    The right leg is the straight one; the left starts 6.8 units high, the gap
    the real skeleton shows, and each unit of knee flexion lowers a foot by
    eight -- so the gap is reachable inside the knee's range, as it is on the
    real leg.
    """

    def __init__(self, left_high: float = 6.8, gain: float = -8.0,
                 hip_gain: float = 0.0) -> None:
        self.left_high = left_high
        self.gain = gain
        self.hip_gain = hip_gain
        self.pivots = {"ankle_R": FakeNode(0.0), "ankle_L": FakeNode(left_high)}
        self.poses = 0

    # the lock drives these two like a BodyAnimationSystem and a Scene
    def apply(self, state, dt):
        self.poses += 1
        if "ankle_R" in self.pivots:
            self.pivots["ankle_R"] = FakeNode(
                self.gain * state.knee_r_flex + self.hip_gain * state.hip_r_flex)
        if "ankle_L" in self.pivots:
            self.pivots["ankle_L"] = FakeNode(
                self.left_high + self.gain * state.knee_l_flex
                + self.hip_gain * state.hip_l_flex)

    def update(self):
        pass


def _lock(rig, **kw):
    return FootLevelLock(rig, rig, rig.pivots, **kw)


def test_the_higher_foot_comes_down_to_the_lower():
    rig = Rig()
    lock = _lock(rig)
    state = {"knee_r_flex": 0.0, "knee_l_flex": 0.0}
    errors = lock.apply(state)
    assert max(errors.values()) <= TOLERANCE, errors
    assert state["knee_l_flex"] == pytest.approx(6.8 / 8.0, abs=1e-3)


def test_the_planted_leg_is_left_exactly_as_authored():
    rig = Rig()
    state = {"knee_r_flex": 0.31, "knee_l_flex": 0.0, "ankle_r_flex": 0.07}
    _lock(rig).apply(state)
    assert state["knee_r_flex"] == 0.31
    assert state["ankle_r_flex"] == 0.07


def test_the_ankle_follows_the_knee_so_the_sole_stays_flat():
    """Changing the knee alone tilted the sole 43 degrees and drove the toes
    2.2 units through the floor."""
    rig = Rig()
    state = {"knee_r_flex": 0.0, "knee_l_flex": 0.0, "ankle_l_flex": 0.0}
    _lock(rig).apply(state)
    assert state["ankle_l_flex"] == pytest.approx(
        state["knee_l_flex"] * ANKLE_PER_KNEE, rel=1e-6)


def test_the_ankle_ratio_is_the_two_dofs_degrees_not_one():
    """145 degrees of knee against 45 of ankle; the identity is between angles."""
    assert ANKLE_PER_KNEE == pytest.approx(145.0 / 45.0)


def test_level_feet_are_left_alone():
    rig = Rig(left_high=0.0)
    state = {"knee_r_flex": 0.2, "knee_l_flex": 0.2}
    before = dict(state)
    lock = _lock(rig)
    assert max(lock.apply(state).values()) <= TOLERANCE
    assert state == before


def test_a_knee_cannot_hyperextend_to_reach():
    """The foot is unreachable downward; the lock stops at the joint's limit."""
    rig = Rig(left_high=-12.0)         # the LEFT is the low one; R cannot reach
    state = {"knee_r_flex": 0.0, "knee_l_flex": 0.0}
    _lock(rig).apply(state)
    assert KNEE_RANGE[0] <= state["knee_r_flex"] <= KNEE_RANGE[1]


def test_a_rig_without_ankles_is_inert():
    rig = Rig()
    rig.pivots.clear()
    lock = FootLevelLock(rig, rig, rig.pivots)
    state = {"knee_l_flex": 0.0}
    assert lock.apply(state) == {}
    assert state == {"knee_l_flex": 0.0}


# -- the hip joins when the knee cannot ---------------------------------------


def test_the_hip_takes_it_when_the_knee_does_nothing():
    """A deadlift's knees are near straight; the hip is the joint with range."""
    rig = Rig(gain=0.0, hip_gain=-8.0)
    lock = _lock(rig)
    state = {"knee_l_flex": 0.0, "hip_l_flex": 0.0, "ankle_l_flex": 0.0}
    errors = lock.apply(state)
    assert max(errors.values()) <= TOLERANCE, errors
    assert state["hip_l_flex"] == pytest.approx(6.8 / 8.0, abs=1e-3)
    assert state["knee_l_flex"] == pytest.approx(0.0, abs=1e-9)


def test_the_hips_ankle_coupling_runs_the_other_way():
    """Dorsiflexion is pitch minus hip plus knee, so a flexing hip gives back."""
    assert ANKLE_PER["hip"] == pytest.approx(-90.0 / 45.0)
    assert ANKLE_PER["knee"] == pytest.approx(145.0 / 45.0)
    rig = Rig(gain=0.0, hip_gain=-8.0)
    state = {"knee_l_flex": 0.0, "hip_l_flex": 0.0, "ankle_l_flex": 0.0}
    _lock(rig).apply(state)
    assert state["ankle_l_flex"] == pytest.approx(
        state["hip_l_flex"] * ANKLE_PER["hip"], rel=1e-6)


def test_the_knee_is_spent_before_the_hip_when_both_would_do():
    """Both joints reach the foot equally; the cheap one should do the work.

    A gap small enough to close in one step, so the ratio is the costs' and
    not ``MAX_STEP``'s: clipping the knee's share leaves the hip carrying more
    than it was asked to.
    """
    rig = Rig(left_high=1.0, gain=-8.0, hip_gain=-8.0)
    state = {"knee_l_flex": 0.0, "hip_l_flex": 0.0, "ankle_l_flex": 0.0}
    _lock(rig).apply(state)
    assert abs(state["knee_l_flex"]) > abs(state["hip_l_flex"])
    # ...in the ratio their costs set, the knee being the cheaper.
    assert (abs(state["knee_l_flex"]) / abs(state["hip_l_flex"])
            == pytest.approx(JOINT_COST["hip"] / JOINT_COST["knee"], rel=1e-3))
