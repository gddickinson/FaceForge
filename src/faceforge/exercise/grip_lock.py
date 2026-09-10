"""Keep hands that grip a fixed bar from sliding along it.

A pull-up is authored as two arm poses (dead hang, chin over the bar) and the
player interpolates the joint angles between them.  The hands' positions are
whatever those angles produce, so as the elbows flex the grip width changes
(measured: 54.5 units at the hang, 68.7 at the top -- each hand slid 7 units
along the bar).  A real pull-up is a closed chain: the hands stay put and the
shoulder angles follow.

This lock closes the chain one joint at a time.  After the player writes a
frame's joint angles it re-poses the skeleton, measures where each hand's
grip point (:func:`faceforge.body.hand_points.finger_ring_centre`) sits along
the line between the hands, and adjusts that side's shoulder abduction so the
offset returns to the value calibrated on the first frame.  Two Newton steps
with a finite-difference slope are enough per frame; the slope is measured
each time because its sign flips as the arm passes the vertical.

Only abduction is adjusted: it is the angle that moves the hand along the
bar, and leaving flexion and elbow flexion as authored keeps the movement's
character.  The ground lock still sets the body's height and its mean
position under the bar.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from faceforge.body.hand_points import finger_ring_centre
from faceforge.core.state import BodyState

#: Normalised abduction step for the finite-difference slope.
_PROBE_STEP = 0.02
#: Bound on the normalised abduction the lock may write (2.0 = 180 deg; the
#: joint-limit file is applied by the caller afterwards).  The first version
#: bounded it at 1.3, below the 1.83 a dead hang needs, so every hang frame
#: was clipped to 117 deg even with a zero step and the hands swung 54 units
#: along the bar before the next step could pull them back.
_ABDUCT_LIMIT = 2.0
#: Largest normalised step per iteration, so a poor local slope cannot fling
#: the arm across the body.
_MAX_STEP = 0.5
_SIDES = ("R", "L")


class GripWidthLock:
    """Hold each hand's offset along the hand line by adjusting shoulder abduction."""

    def __init__(self, body_animation: Any, scene: Any, pivots: dict,
                 iterations: int = 2) -> None:
        self.body_animation = body_animation
        self.scene = scene
        self.pivots = pivots
        self.iterations = max(1, int(iterations))
        self.targets: dict[str, float] | None = None
        self.axis: np.ndarray | None = None
        self._state = BodyState()

    # -- forward kinematics -------------------------------------------------------

    def _pose(self, state_dict: dict) -> None:
        self._state.set_from_js_dict(state_dict)
        self.body_animation.apply(self._state, 0.0)
        self.scene.update()

    def _offsets(self) -> dict[str, float] | None:
        """Each hand's position along the hand line, measured from the trunk.

        The reference is the midpoint of the two shoulder pivots, not of the
        two hands: offsets from the hand midpoint always sum to zero, which
        would leave one equation for two unknowns.
        """
        rings = {s: finger_ring_centre(self.pivots, s) for s in _SIDES}
        if any(r is None for r in rings.values()):
            return None
        shoulders = [self.pivots.get(f"shoulder_{s}") for s in _SIDES]
        if all(n is not None for n in shoulders):
            mid = 0.5 * sum(np.asarray(n.get_world_position(), dtype=np.float64)
                            for n in shoulders)
        else:
            mid = 0.5 * (rings["R"] + rings["L"])
        if self.axis is None:
            line = rings["R"] - rings["L"]
            n = float(np.linalg.norm(line))
            if n < 1e-6:
                return None
            self.axis = line / n
        return {s: float((rings[s] - mid) @ self.axis) for s in _SIDES}

    # -- api ------------------------------------------------------------------------

    @property
    def calibrated(self) -> bool:
        return self.targets is not None

    def calibrate(self, state_dict: dict) -> bool:
        """Remember each hand's offset along the hand line for this pose."""
        self._pose(state_dict)
        offsets = self._offsets()
        if offsets is None:
            return False
        self.targets = offsets
        return True

    def apply(self, state_dict: dict) -> dict[str, float]:
        """Adjust ``shoulder_{r,l}_abduct`` in ``state_dict`` (in place).

        Returns the residual offset error per side before the last step.
        Each side's offset depends on BOTH abductions (the mid-hand point
        moves with either hand), so the step solves the 2x2 finite-difference
        Jacobian rather than one slope per side.
        """
        if self.targets is None:
            if not self.calibrate(state_dict):
                return {}
        keys = {s: f"shoulder_{s.lower()}_abduct" for s in _SIDES}
        errors: dict[str, float] = {}
        for _ in range(self.iterations):
            self._pose(state_dict)
            offsets = self._offsets()
            if offsets is None:
                return errors
            err = np.array([self.targets[s] - offsets[s] for s in _SIDES])
            errors = {s: float(e) for s, e in zip(_SIDES, err)}
            if float(np.abs(err).max()) < 1e-3:
                break
            jac = np.zeros((2, 2))
            for j, side in enumerate(_SIDES):
                probe = dict(state_dict)
                probe[keys[side]] = float(state_dict.get(keys[side], 0.0)) + _PROBE_STEP
                self._pose(probe)
                probed = self._offsets()
                if probed is None:
                    return errors
                jac[:, j] = [(probed[s] - offsets[s]) / _PROBE_STEP for s in _SIDES]
            if abs(float(np.linalg.det(jac))) < 1e-9:
                break
            delta = np.clip(np.linalg.solve(jac, err), -_MAX_STEP, _MAX_STEP)
            for j, side in enumerate(_SIDES):
                base = float(state_dict.get(keys[side], 0.0))
                state_dict[keys[side]] = float(np.clip(base + delta[j], -_ABDUCT_LIMIT, _ABDUCT_LIMIT))
        return errors
