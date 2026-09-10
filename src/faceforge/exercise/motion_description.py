"""Turn a change of pose into the words a physiotherapist would use.

``describe_transition(a, b)`` compares two poses DOF by DOF, converts the
change to degrees with :mod:`faceforge.body.dof_ranges`, and names it:
a hip going from 20 to 110 degrees of flexion is "hip flexion 20 -> 110";
the same hip going back is "hip extension 110 -> 20".  Bilateral movements
that match on both sides are merged into one line ("hips"), so a squat reads
as three lines rather than six.
"""

from __future__ import annotations

from dataclasses import dataclass

from faceforge.body.dof_ranges import dof_range, dof_side, dof_to_degrees, is_pose_dof

_PLURALS = {"hip": "hips", "knee": "knees", "ankle": "ankles", "shoulder": "shoulders",
            "elbow": "elbows", "wrist": "wrists", "forearm": "forearms", "hand": "hands",
            "foot": "feet", "spine": "spine"}


@dataclass(frozen=True)
class JointMotion:
    """One joint moving through one DOF between two poses."""

    joint: str            # "hip", "knee", ...
    side: str | None      # "R", "L" or None
    term: str             # "flexion", "extension", "abduction", ...
    start_deg: float      # signed degrees of the DOF's positive term
    end_deg: float
    field: str            # the BodyState field

    @property
    def delta_deg(self) -> float:
        return self.end_deg - self.start_deg

    def label(self, both_sides: bool = False) -> str:
        joint = _PLURALS.get(self.joint, self.joint) if both_sides else self.joint
        side = "" if (both_sides or self.side is None) else f" ({self.side})"
        return (f"{joint.capitalize()}{side}: {self.term} "
                f"{abs(self.start_deg):.0f}° → {abs(self.end_deg):.0f}°")


def describe_transition(pose_a: dict[str, float], pose_b: dict[str, float],
                        min_delta_deg: float = 4.0) -> list[JointMotion]:
    """Every DOF that changes by at least ``min_delta_deg``, largest first."""
    motions: list[JointMotion] = []
    for field_name in sorted(set(pose_a) | set(pose_b)):
        if not is_pose_dof(field_name):
            continue
        a = dof_to_degrees(field_name, float(pose_a.get(field_name, 0.0)))
        b = dof_to_degrees(field_name, float(pose_b.get(field_name, 0.0)))
        if abs(b - a) < min_delta_deg:
            continue
        r = dof_range(field_name)
        term = r.positive if b > a else r.negative
        motions.append(JointMotion(r.joint, dof_side(field_name), term, a, b, field_name))
    motions.sort(key=lambda m: -abs(m.delta_deg))
    return motions


def summarise_motions(motions: list[JointMotion], max_lines: int = 6) -> list[str]:
    """Merge matching left/right motions and render one line each."""
    lines: list[str] = []
    used: set[int] = set()
    for i, m in enumerate(motions):
        if i in used:
            continue
        partner = None
        if m.side is not None:
            for j, n in enumerate(motions):
                if (j != i and j not in used and n.joint == m.joint and n.term == m.term
                        and n.side not in (None, m.side)
                        and abs(n.start_deg - m.start_deg) < 3 and abs(n.end_deg - m.end_deg) < 3):
                    partner = j
                    break
        if partner is not None:
            used.add(partner)
            lines.append(m.label(both_sides=True))
        else:
            lines.append(m.label())
        used.add(i)
        if len(lines) >= max_lines:
            break
    return lines
