"""Both feet on the floor, not only the lowest one.

The ground lock translates the whole body so that its lowest support sits on
the floor.  That is one translation for two feet, and it is enough only while
the feet end a pose at the same height.  They do not, because the donor is a
real body and its legs are not identical: measured on the rest skeleton, the
right shank runs [-0.68, 1.35, -46.75] from knee to ankle and the left
[-2.66, 5.77, -46.55] -- about seven degrees apart -- and the femurs differ by
1.8 units in length.

Standing, that costs nothing: both ankles sit within 0.4 units of each other.
Bend the same joints by the same angles, though, and the difference is
amplified by the rotation: at the bottom of a bodyweight squat the two ankles
ended 6.8 units apart, so the ground lock planted the right foot and left the
left one hanging in the air.

This closes the gap one leg at a time, the way
:class:`faceforge.exercise.grip_lock.GripWidthLock` closes a grip: after the
player writes a frame's joint angles, measure each foot, and adjust the higher
leg's knee until it reaches the lower one.  The slope is measured by a probe
step each iteration rather than assumed, because whether extending a knee
raises or lowers the foot depends on where the shank is pointing, and that
changes through the movement.

The knee is what the solve moves, and the ankle follows it.  The sole is flat
when ankle dorsiflexion equals pitch minus hip plus knee, so a knee the solver
has changed leaves the ankle owing exactly that change; without it, bringing
the left foot down in a squat drove its toes 2.2 units through the floor and
levered the heel 17 units into the air, a sole tilted 43 degrees.  Both are
written in degrees through `dof_ranges`, because the knee's range is 145
degrees and the ankle's 45 and the identity is an identity between angles.

The hip is left as the clip authored it, which keeps the movement's shape; the
ground lock still decides the body's height and where it stands.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from faceforge.body.dof_ranges import dof_range
from faceforge.body.ground_contact import height_pivot_names
from faceforge.core.state import BodyState

#: Normalised knee step used to measure the slope.
PROBE_STEP = 0.02
#: Largest normalised knee change per iteration, so a poor local slope cannot
#: snap the leg straight.
MAX_STEP = 0.35
#: Knees flex; they do not hyperextend.  1.0 is the full 145 degrees.
KNEE_RANGE = (0.0, 1.0)
#: Height difference below which the feet count as level, in body units.
TOLERANCE = 0.05

#: Largest gap the lock will try to close, in body units.  It exists to undo
#: the donor's leg asymmetry, which is worth 6.8 units at the bottom of a
#: squat; a gap much larger than that is the pose asking for it -- a foot on a
#: box in a step-up (44 units), the trailing leg of a split squat (65).
#: Without this bound the solve chased those too: on a Romanian deadlift,
#: whose knees are nearly straight and cannot extend further to reach, it
#: drove a knee through 45 degrees and still missed by 19 units.
MAX_GAP = 8.0

#: Fraction of the starting gap the solve must get under for its work to be
#: kept.  Demanding TOLERANCE outright threw away good partial corrections --
#: a squat frame that closed 6.5 units to 0.3 was reverted whole -- while
#: keeping everything left 45 degrees of useless knee on a Romanian deadlift.
ACCEPT_FRACTION = 0.5

SIDES = ("R", "L")

#: Normalised ankle change per unit of normalised knee change that keeps the
#: sole flat: the identity is between degrees, and the two ranges differ.
ANKLE_PER_KNEE = (dof_range("knee_{s}_flex").degrees
                  / dof_range("ankle_{s}_flex").degrees)


class FootLevelLock:
    """Bring the higher foot down to the lower one by bending that knee."""

    def __init__(self, body_animation: Any, scene: Any, pivots: dict,
                 iterations: int = 4) -> None:
        self.body_animation = body_animation
        self.scene = scene
        self.pivots = pivots
        self.iterations = max(1, int(iterations))
        self._state = BodyState()
        #: Each foot's height above the lower one after the last call.
        self.last_error: dict[str, float] = {}

    # -- forward kinematics ---------------------------------------------------

    def _pose(self, state_dict: dict) -> None:
        self._state.set_from_js_dict(state_dict)
        self.body_animation.apply(self._state, 0.0)
        self.scene.update()

    def heights(self) -> dict[str, float] | None:
        """The lowest contact pivot on each side, in world y."""
        out: dict[str, float] = {}
        for side in SIDES:
            ys = [float(np.asarray(node.get_world_position(),
                                   dtype=np.float64)[1])
                  for node in (self.pivots.get(n)
                               for n in height_pivot_names("feet", side))
                  if node is not None]
            if not ys:
                return None
            out[side] = min(ys)
        return out

    # -- api ------------------------------------------------------------------

    def apply(self, state_dict: dict) -> dict[str, float]:
        """Adjust ``knee_{r,l}_flex`` and its ankle in ``state_dict`` (in place).

        Returns each foot's remaining height above the target.  The target is
        the lower foot as the frame arrived, fixed for the whole solve:
        recomputing it each iteration let an overshoot make the other foot the
        higher one, and the two then traded places instead of converging.
        """
        keys = {s: f"knee_{s.lower()}_flex" for s in SIDES}
        ankles = {s: f"ankle_{s.lower()}_flex" for s in SIDES}
        self._pose(state_dict)
        start = self.heights()
        if start is None:
            return {}
        target = min(start.values())
        errors = {s: start[s] - target for s in SIDES}
        if max(errors.values()) > MAX_GAP:
            # The pose wants the feet apart; that is not this lock's business.
            self.last_error = errors
            return errors
        # Everything the solve may write, as the clip authored it.  If it does
        # not get there, the pose goes back: a lock that cannot close the gap
        # must not leave a bent knee behind for nothing.  Measured before this,
        # a Romanian deadlift -- knees near straight, so the foot is out of
        # reach downward -- took 45 degrees of knee and still missed by 19.
        authored = {k: state_dict.get(k) for k in
                    list(keys.values()) + list(ankles.values())}
        for _ in range(self.iterations):
            high = max(SIDES, key=lambda s: errors[s])
            if errors[high] <= TOLERANCE:
                break
            base = float(state_dict.get(keys[high], 0.0))
            ankle_base = float(state_dict.get(ankles[high], 0.0))
            # The probe moves the ankle too, because the step will.  Measuring
            # the slope with the ankle held still answers a different question
            # and the step then overshoots: the left foot went 1.9 units past
            # the right instead of meeting it.
            probe = dict(state_dict)
            probe[keys[high]] = base + PROBE_STEP
            probe[ankles[high]] = ankle_base + PROBE_STEP * ANKLE_PER_KNEE
            self._pose(probe)
            probed = self.heights()
            if probed is None:
                break
            slope = (probed[high] - (errors[high] + target)) / PROBE_STEP
            if abs(slope) < 1e-6:
                break
            step = float(np.clip(-errors[high] / slope, -MAX_STEP, MAX_STEP))
            moved = float(np.clip(base + step, *KNEE_RANGE))
            if abs(moved - base) < 1e-9:
                break
            state_dict[keys[high]] = moved
            # The sole stays flat only if the ankle takes the same angle.
            state_dict[ankles[high]] = ankle_base + (moved - base) * ANKLE_PER_KNEE
            self._pose(state_dict)
            now = self.heights()
            if now is None:
                break
            errors = {s: now[s] - target for s in SIDES}
        start_gap = max(start[s] - target for s in SIDES)
        if max(errors.values()) > max(TOLERANCE, ACCEPT_FRACTION * start_gap):
            for key, value in authored.items():
                if value is None:
                    state_dict.pop(key, None)
                else:
                    state_dict[key] = value
        self.last_error = errors
        return errors
