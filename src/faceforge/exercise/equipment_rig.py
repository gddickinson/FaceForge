"""Keep hand-held equipment in the hands, every frame.

The finger pivots' world positions (after the scene update) say where the
closed hands are: the joints of the four fingers form a ring round a held
object, and the object's axis passes through the ring's centre.  A
two-handed item sits between the two grip points with its +X axis along the
line between them; a one-handed item sits at that hand's grip point.  A rig
without finger pivots falls back to the wrist pushed a little way along the
forearm.  Static equipment (a bench, a bar) is simply placed.

Everything here is geometry on numpy vectors, so it is testable with fake
nodes that only expose ``get_world_position``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from faceforge.body.hand_points import finger_ring_centre
from faceforge.core.math_utils import quat_from_axis_angle, quat_multiply, normalize, vec3
from faceforge.core.scene_graph import SceneNode
from faceforge.exercise.model import EquipmentSpec

_X = vec3(1.0, 0.0, 0.0)
_Y = vec3(0.0, 1.0, 0.0)

#: How far past the wrist pivot (along the forearm) the palm centre is.
DEFAULT_GRIP_OFFSET = 7.0


def align_x_to(direction: np.ndarray) -> np.ndarray:
    """A quaternion rotating local +X onto ``direction``."""
    d = np.asarray(direction, dtype=np.float64)
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    d = d / n
    c = float(np.clip(np.dot(_X, d), -1.0, 1.0))
    if c > 1.0 - 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    if c < -1.0 + 1e-9:
        return quat_from_axis_angle(_Y, math.pi)
    axis = np.cross(_X, d)
    return quat_from_axis_angle(axis, math.acos(c))


def _world(node) -> np.ndarray:
    return np.asarray(node.get_world_position(), dtype=np.float64)


@dataclass
class RiggedItem:
    node: SceneNode
    spec: EquipmentSpec
    grip_offset: float = DEFAULT_GRIP_OFFSET
    #: Extra world-space drop for held items (e.g. a kettlebell hangs lower).
    hang: float = 0.0
    #: Revolutions per second about the handle axis (a jump rope), 0 = none.
    spin: float = 0.0


class EquipmentRig:
    """Owns the equipment nodes of a demonstration and places them per frame."""

    def __init__(self) -> None:
        self.items: list[RiggedItem] = []

    def add(self, node: SceneNode, spec: EquipmentSpec, grip_offset: float | None = None,
            hang: float = 0.0, spin: float = 0.0) -> RiggedItem:
        item = RiggedItem(node, spec,
                          DEFAULT_GRIP_OFFSET if grip_offset is None else grip_offset, hang,
                          spin)
        if spec.attach == "static":
            node.set_position(*spec.position)
            rx, ry, rz = (math.radians(a) for a in spec.rotation_deg)
            q = quat_multiply(quat_from_axis_angle(_Y, ry),
                              quat_multiply(quat_from_axis_angle(_X, rx),
                                            quat_from_axis_angle(vec3(0, 0, 1), rz)))
            node.set_quaternion(q)
        self.items.append(item)
        return item

    def clear(self) -> list[SceneNode]:
        nodes = [i.node for i in self.items]
        self.items = []
        return nodes

    @property
    def nodes(self) -> list[SceneNode]:
        return [i.node for i in self.items]

    # -- per frame -------------------------------------------------------------

    @staticmethod
    def palm_point(pivots: dict, side: str, grip_offset: float) -> np.ndarray | None:
        """World position of the palm: the wrist pushed along the forearm."""
        wrist = pivots.get(f"wrist_{side}")
        if wrist is None:
            return None
        w = _world(wrist)
        elbow = pivots.get(f"elbow_{side}")
        if elbow is None or grip_offset == 0.0:
            return w
        forearm = w - _world(elbow)
        n = float(np.linalg.norm(forearm))
        if n < 1e-9:
            return w
        return w + forearm / n * grip_offset

    @classmethod
    def grip_point(cls, pivots: dict, side: str, grip_offset: float) -> np.ndarray | None:
        """World centre of the closed fingers: where a held bar's axis passes.

        See :func:`faceforge.body.hand_points.finger_ring_centre`; without
        finger pivots the ``palm_point`` fallback applies.
        """
        centre = finger_ring_centre(pivots, side)
        if centre is not None:
            return centre
        return cls.palm_point(pivots, side, grip_offset)

    def update(self, pivots: dict, time: float = 0.0) -> None:
        """Place every held item from the current wrist/elbow world positions."""
        for item in self.items:
            attach = item.spec.attach
            if attach == "static":
                continue
            if attach == "hands":
                pr = self.grip_point(pivots, "R", item.grip_offset)
                pl = self.grip_point(pivots, "L", item.grip_offset)
                if pr is None or pl is None:
                    continue
                centre = (pr + pl) / 2.0
                axis = pr - pl
                q = align_x_to(axis) if np.linalg.norm(axis) > 1e-6 else None
            else:
                side = "R" if attach == "hand_r" else "L"
                centre = self.grip_point(pivots, side, item.grip_offset)
                if centre is None:
                    continue
                pr = self.grip_point(pivots, "R", 0.0)
                pl = self.grip_point(pivots, "L", 0.0)
                axis = (pr - pl) if (pr is not None and pl is not None) else _X
                q = align_x_to(axis) if np.linalg.norm(axis) > 1e-6 else None
            if item.hang:
                centre = centre + np.array([0.0, -item.hang, 0.0])
            item.node.set_position(float(centre[0]), float(centre[1]), float(centre[2]))
            if q is not None:
                if item.spin:
                    q = quat_multiply(q, quat_from_axis_angle(_X, 2 * math.pi * item.spin * time))
                item.node.set_quaternion(q)
