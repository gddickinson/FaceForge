"""Keep the feet (or hands) on the floor while the joints move.

The rig's root is the pelvis.  Flex the hips and knees and the feet rise off
the floor, because nothing moves the root down; pitch the whole body forward
about its origin near the head and the feet swing metres away.  Real
animation systems solve this with root motion or foot IK.  This module does
the minimal honest version: after the pose is applied and world matrices are
fresh, it measures where the anchor (the lowest foot pivot, or the wrists)
ended up, compares that with where it should be, and translates the wrapper
node by the difference.  One translation, no accumulation, converges in one
frame because the wrapper moves every body point rigidly.

Targets come from the pivots' REST positions -- the sum of local translations
up the pivot chain, which is the body-frame position when every joint is at
identity -- transformed by the orientation's base placement.  Calibration
therefore needs no posed scene.  (``JointPivotSetup.joint_positions`` cannot
be used directly: limb joints are recorded in the body frame but digit
pivots relative to their parent.)
"""

from __future__ import annotations

import numpy as np

from faceforge.body.hand_points import finger_ring_centre

from faceforge.core.math_utils import mat4_compose, vec3

#: Pivot names whose LOWEST world point defines the floor contact, per anchor.
_HEIGHT_PIVOTS = {
    "feet": tuple(f"ankle_{s}" for s in "RL") + tuple(
        f"toe_{s}_{d}_mt" for s in "RL" for d in range(1, 6)),
    "hands": ("wrist_R", "wrist_L"),
}
#: Pivots whose MEAN defines the horizontal position, per anchor.
_PLANE_PIVOTS = {"feet": ("ankle_R", "ankle_L"), "hands": ("wrist_R", "wrist_L")}


def rest_body_position(node, stop_name: str = "bodyRoot") -> np.ndarray:
    """Body-frame rest position of a pivot: local translations summed up to ``stop_name``."""
    total = np.zeros(3)
    while node is not None and node.name != stop_name:
        total += np.asarray(node.position, dtype=np.float64)
        node = node.parent
    return total


class GroundLock:
    """Per-frame re-anchoring of the wrapper so a chosen support stays put."""

    def __init__(self, anchor: str = "feet", lock_horizontal: bool = True,
                 side: str | None = None) -> None:
        if anchor not in ("feet", "hands", "none"):
            raise ValueError(f"anchor must be feet, hands or none, not {anchor!r}")
        if side not in (None, "R", "L"):
            raise ValueError(f"side must be None, 'R' or 'L', not {side!r}")
        self.anchor = anchor
        self.lock_horizontal = lock_horizontal
        #: Restrict the anchor to one limb (the front foot of a lunge, whose
        #: trailing foot is on its toes or a bench and must not be taken for
        #: the floor contact).
        self.side = side
        self._target_y: float | None = None
        self._target_xz: np.ndarray | None = None
        self.last_delta = np.zeros(3)

    def _height_pivots(self) -> tuple[str, ...]:
        names = _HEIGHT_PIVOTS[self.anchor]
        return names if self.side is None else tuple(n for n in names if f"_{self.side}" in n)

    def _plane_pivots(self) -> tuple[str, ...]:
        names = _PLANE_PIVOTS[self.anchor]
        return names if self.side is None else tuple(n for n in names if n.endswith(self.side))

    # -- calibration -------------------------------------------------------------

    @property
    def calibrated(self) -> bool:
        return self.anchor == "none" or self._target_y is not None

    #: Height of the wrist pivot above the floor when the palm is on it.
    HAND_CONTACT_HEIGHT = 3.0

    def calibrate(self, pivots: dict, base_position, base_quaternion,
                  floor_y: float | None = None) -> None:
        """Targets from the pivots' rest positions under the base placement.

        For feet the rest height is the answer (standing places the soles on
        the floor).  Hands rest at the body's sides, so their height target is
        the floor plus the palm's thickness unless ``floor_y`` is None.
        """
        if self.anchor == "none":
            return
        m = mat4_compose(vec3(*base_position), np.asarray(base_quaternion, dtype=np.float64),
                         vec3(1.0, 1.0, 1.0))
        ys = []
        for name in self._height_pivots():
            node = pivots.get(name)
            if node is not None:
                p = rest_body_position(node)
                ys.append(float((m @ np.array([p[0], p[1], p[2], 1.0]))[1]))
        plane = []
        for name in self._plane_pivots():
            node = pivots.get(name)
            if node is not None:
                p = rest_body_position(node)
                w = m @ np.array([p[0], p[1], p[2], 1.0])
                plane.append(np.array([w[0], w[2]]))
        if not ys or not plane:
            raise ValueError(f"no {self.anchor} pivots to calibrate against")
        self._target_y = min(ys)
        if self.anchor == "hands" and floor_y is not None:
            self._target_y = float(floor_y) + self.HAND_CONTACT_HEIGHT
        self._target_xz = np.mean(plane, axis=0)

    def set_target(self, point) -> None:
        """Anchor to an explicit world point (e.g. the hands on a pull-up bar).

        Hands anchored to a point are GRIPPING it, so from here on the hand
        pivots measured are the closed-finger ring centres, not the wrists:
        with the wrists at bar height the fingers waved 13 units above the
        bar in every pull-up render.  Hands on the floor keep the wrists.
        """
        p = np.asarray(point, dtype=np.float64)
        self._target_y = float(p[1])
        self._target_xz = np.array([p[0], p[2]])
        self._grip = self.anchor == "hands"

    # -- per frame -----------------------------------------------------------------

    def measure(self, pivots: dict) -> tuple[float, np.ndarray] | None:
        """Current ``(lowest y, mean xz)`` of the anchor pivots, or None."""
        if self.anchor == "none":
            return None
        ys = []
        for name in self._height_pivots():
            w = self._anchor_point(pivots, name)
            if w is not None:
                ys.append(float(w[1]))
        plane = []
        for name in self._plane_pivots():
            w = self._anchor_point(pivots, name)
            if w is not None:
                plane.append(np.array([float(w[0]), float(w[2])]))
        if not ys or not plane:
            return None
        return min(ys), np.mean(plane, axis=0)

    def _anchor_point(self, pivots: dict, name: str) -> np.ndarray | None:
        """World point measured for anchor pivot ``name``: the grip ring when gripping."""
        if getattr(self, "_grip", False) and name.startswith("wrist_"):
            ring = finger_ring_centre(pivots, name[-1])
            if ring is not None:
                return ring
        node = pivots.get(name)
        if node is None:
            return None
        return np.asarray(node.get_world_position(), dtype=np.float64)

    def update(self, wrapper, pivots: dict, lift: float = 0.0,
               travel: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
        """Translate ``wrapper`` so the anchor returns to its target (+ ``lift``/``travel``).

        Call after the scene's world matrices are up to date; the caller must
        update them again afterwards.  Returns the translation applied.
        """
        delta = np.zeros(3)
        if not self.calibrated or self.anchor == "none":
            self.last_delta = delta
            return delta
        measured = self.measure(pivots)
        if measured is None:
            self.last_delta = delta
            return delta
        y, xz = measured
        delta[1] = (self._target_y + lift) - y
        if self.lock_horizontal and self._target_xz is not None:
            delta[0] = (self._target_xz[0] + travel[0]) - xz[0]
            delta[2] = (self._target_xz[1] + travel[1]) - xz[1]
        if np.any(np.abs(delta) > 1e-6):
            pos = np.asarray(wrapper.position, dtype=np.float64) + delta
            wrapper.set_position(float(pos[0]), float(pos[1]), float(pos[2]))
        self.last_delta = delta
        return delta
