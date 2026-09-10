"""The body's joint degrees of freedom: range, sign convention and anatomical name.

One table, consulted by everything that converts between the normalised DOF
values on :class:`~faceforge.core.state.BodyState` and degrees:

* :mod:`faceforge.body.body_animation` scales the value by the range here;
* the exercise catalogue authors poses in degrees and converts them here;
* the motion description turns a DOF change into an anatomical term here.

Body frame (measured, see ``docs/exercise_animation.md``): +Z superior, -Y
anterior, +X right.  A value of ``+1.0`` means the *positive* term below; a
negative value means the *negative* term.  Ranges are the angle in degrees
that a value of 1.0 produces; the joint limits in
``assets/config/body_joint_limits.json`` decide how far past 1.0 a DOF may go
(shoulder flexion reaches 2.0 = 180 degrees for an overhead press).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DofRange:
    """Range and naming for one DOF pattern (``{s}`` = ``r`` or ``l``)."""

    pattern: str
    degrees: float
    positive: str
    negative: str
    joint: str

    def degrees_for(self, value: float) -> float:
        return value * self.degrees

    def value_for(self, degrees: float) -> float:
        return degrees / self.degrees


#: Every pose DOF, in the order the Body tab shows them.
DOF_RANGES: tuple[DofRange, ...] = (
    DofRange("spine_flex", 45.0, "flexion", "extension", "spine"),
    DofRange("spine_lat_bend", 30.0, "right lateral flexion", "left lateral flexion", "spine"),
    DofRange("spine_rotation", 30.0, "rotation", "rotation", "spine"),
    DofRange("shoulder_{s}_flex", 90.0, "flexion", "extension", "shoulder"),
    DofRange("shoulder_{s}_abduct", 90.0, "abduction", "adduction", "shoulder"),
    DofRange("shoulder_{s}_rotate", 70.0, "external rotation", "internal rotation", "shoulder"),
    DofRange("elbow_{s}_flex", 145.0, "flexion", "extension", "elbow"),
    DofRange("forearm_{s}_rotate", 90.0, "supination", "pronation", "forearm"),
    DofRange("wrist_{s}_flex", 70.0, "flexion", "extension", "wrist"),
    DofRange("wrist_{s}_deviate", 30.0, "ulnar deviation", "radial deviation", "wrist"),
    DofRange("hip_{s}_flex", 90.0, "flexion", "extension", "hip"),
    DofRange("hip_{s}_abduct", 45.0, "abduction", "adduction", "hip"),
    DofRange("hip_{s}_rotate", 45.0, "external rotation", "internal rotation", "hip"),
    DofRange("knee_{s}_flex", 145.0, "flexion", "extension", "knee"),
    DofRange("ankle_{s}_flex", 45.0, "dorsiflexion", "plantarflexion", "ankle"),
    DofRange("ankle_{s}_invert", 30.0, "inversion", "eversion", "ankle"),
    DofRange("finger_curl_{s}", 90.0, "finger flexion", "finger extension", "hand"),
    DofRange("finger_spread_{s}", 12.0, "finger abduction", "finger adduction", "hand"),
    DofRange("thumb_op_{s}", 50.0, "thumb opposition", "thumb reposition", "hand"),
    DofRange("toe_curl_{s}", 75.0, "toe flexion", "toe extension", "foot"),
    DofRange("toe_spread_{s}", 8.0, "toe abduction", "toe adduction", "foot"),
)

_BY_PATTERN: dict[str, DofRange] = {r.pattern: r for r in DOF_RANGES}

#: Every concrete BodyState pose field name, expanded for both sides.
POSE_DOF_FIELDS: tuple[str, ...] = tuple(
    name
    for r in DOF_RANGES
    for name in ([r.pattern] if "{s}" not in r.pattern
                 else [r.pattern.replace("{s}", "r"), r.pattern.replace("{s}", "l")])
)


def dof_pattern(field: str) -> str:
    """``"hip_r_flex"`` -> ``"hip_{s}_flex"``; unsided names pass through."""
    if field in _BY_PATTERN:
        return field
    for side in ("_r_", "_l_"):
        if side in field:
            return field.replace(side, "_{s}_", 1)
    if field.endswith(("_r", "_l")):
        return field[:-2] + "_{s}"
    return field


def dof_side(field: str) -> str | None:
    """``"R"``, ``"L"`` or ``None`` for an unsided DOF such as ``spine_flex``."""
    if "_r_" in field or field.endswith("_r"):
        return "R"
    if "_l_" in field or field.endswith("_l"):
        return "L"
    return None


def dof_range(field: str) -> DofRange:
    """The :class:`DofRange` for a concrete field.  Raises ``KeyError`` if unknown."""
    return _BY_PATTERN[dof_pattern(field)]


def is_pose_dof(field: str) -> bool:
    return dof_pattern(field) in _BY_PATTERN


def dof_to_degrees(field: str, value: float) -> float:
    return dof_range(field).degrees_for(value)


def degrees_to_dof(field: str, degrees: float) -> float:
    return dof_range(field).value_for(degrees)


def dof_term(field: str, value: float) -> str:
    """The anatomical term for the *direction* of ``value`` on ``field``."""
    r = dof_range(field)
    return r.positive if value >= 0 else r.negative
