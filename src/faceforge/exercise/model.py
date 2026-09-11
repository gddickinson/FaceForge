"""The data model for an exercise demonstration.

An :class:`ExerciseDefinition` is pure data: it names the equipment, the setup
cues, a sequence of :class:`Phase` objects (each the pose reached at the END
of that phase, how long it takes to get there, and what kind of contraction
it is), the muscles involved by role, and the sources the numbers came from.
Everything that moves the model -- keyframes, activation levels, movement
descriptions -- is derived from this by :mod:`faceforge.exercise.clip_builder`.

Poses are dictionaries of normalised :class:`~faceforge.core.state.BodyState`
DOF values (``{"hip_r_flex": 1.2, ...}``); author them in degrees with
:mod:`faceforge.exercise.pose_library` rather than by hand.  Trunk inclination
is NOT a body DOF: the arms hang off the pelvis root, not the thoracic chain,
so leaning the trunk is done by pitching the whole body (``Phase.pitch``) and
flexing the hips to keep the thighs where they were.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class Role(str, Enum):
    PRIMARY = "primary"          # agonist / prime mover
    SECONDARY = "secondary"      # synergist
    STABILISER = "stabiliser"    # isometric, holds a segment still


class PhaseKind(str, Enum):
    ECCENTRIC = "eccentric"      # the agonists lengthen under load
    CONCENTRIC = "concentric"    # the agonists shorten
    ISOMETRIC = "isometric"      # a hold at constant length
    TRANSITION = "transition"    # unloaded repositioning (e.g. recovery on a rower)


class Category(str, Enum):
    LOWER_BODY = "Lower body"
    UPPER_PUSH = "Upper body: push"
    UPPER_PULL = "Upper body: pull"
    CORE = "Core and trunk"
    CONDITIONING = "Cardio and conditioning"
    ATHLETIC = "Athletic and power"


#: Default peak level for a role when the definition does not give one.
ROLE_PEAK: dict[Role, float] = {Role.PRIMARY: 0.95, Role.SECONDARY: 0.6, Role.STABILISER: 0.3}


@dataclass(frozen=True)
class MuscleUse:
    """One functional muscle group in one role.

    ``peak`` is the activation (0-1, read as a fraction of maximal voluntary
    contraction) at the hardest point of the movement; ``side`` restricts a
    bilateral group to one limb for unilateral exercises.
    """

    group: str
    role: Role
    peak: float | None = None
    side: str | None = None
    note: str = ""

    @property
    def level(self) -> float:
        return ROLE_PEAK[self.role] if self.peak is None else float(self.peak)


@dataclass(frozen=True)
class Phase:
    """The pose reached at the end of this phase, and how to get there."""

    name: str
    kind: PhaseKind
    duration: float                      # seconds to reach ``pose`` from the previous phase
    pose: dict[str, float]               # BodyState DOF -> normalised value
    pitch: float = 0.0                   # trunk pitch forward, degrees (whole-body)
    roll: float = 0.0                    # lateral lean, degrees, +right
    yaw: float = 0.0                     # turn about the vertical, degrees
    lift: float = 0.0                    # extra height off the ground anchor, world units
    travel: tuple[float, float] = (0.0, 0.0)  # anchor floor offset (x, z), world units
    pivot: tuple[float, float, float] | None = None  # body point pitch/roll/yaw turn about
    position: tuple[float, float, float] | None = None  # absolute wrapper position override
    orientation: str | None = None       # override the exercise orientation for this phase
    cues: tuple[str, ...] = ()
    activation: dict[str, float] = field(default_factory=dict)  # group -> level override
    easing: str = "ease_in_out"


@dataclass(frozen=True)
class EquipmentSpec:
    """A piece of equipment and how it follows the body.

    ``attach``: ``"hands"`` (centred between the wrists, e.g. a barbell),
    ``"hand_r"`` / ``"hand_l"`` (one per hand), or ``"static"`` (fixed in the
    room at ``position``).
    """

    kind: str
    attach: str = "hands"
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    params: dict[str, Any] = field(default_factory=dict)
    #: How far below the hands the item's origin sits (world units); ``None``
    #: takes the kind's default (a kettlebell hangs 10 below the grip in a
    #: swing, but sits against the chest in a goblet hold).
    hang: float | None = None


@dataclass(frozen=True)
class ExerciseDefinition:
    """Everything the app needs to demonstrate and explain one exercise."""

    id: str
    name: str
    category: Category
    description: str
    phases: tuple[Phase, ...]
    muscles: tuple[MuscleUse, ...]
    equipment: tuple[EquipmentSpec, ...] = ()
    setup: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    physio_notes: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    orientation: str = "standing"        # standing | supine | prone | seated | hanging | side
    anchor: str = "feet"                 # feet | hands | none  (what the ground lock holds still)
    anchor_side: str | None = None       # "R"/"L": anchor one limb only (front foot of a lunge)
    lock_horizontal: bool = True         # keep the anchor's floor position, not only its height
    base_position: tuple[float, float, float] | None = None
    anchor_point: tuple[float, float, float] | None = None  # world target for the anchor
    default_reps: int = 3
    camera: str = "three_quarter"
    camera_target: tuple[float, float, float] | None = None  # look-at override (world)
    unilateral: bool = False
    tags: tuple[str, ...] = ()

    # -- derived -------------------------------------------------------------

    @property
    def rep_duration(self) -> float:
        return float(sum(p.duration for p in self.phases))

    @property
    def equipment_names(self) -> tuple[str, ...]:
        return tuple(e.kind for e in self.equipment)

    def muscles_by_role(self, role: Role) -> tuple[MuscleUse, ...]:
        return tuple(m for m in self.muscles if m.role is role)

    @property
    def muscle_groups(self) -> tuple[str, ...]:
        seen: list[str] = []
        for m in self.muscles:
            if m.group not in seen:
                seen.append(m.group)
        return tuple(seen)

    # -- serialisation ----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        for ph in d["phases"]:
            ph["kind"] = ph["kind"].value if isinstance(ph["kind"], PhaseKind) else ph["kind"]
        for m in d["muscles"]:
            m["role"] = m["role"].value if isinstance(m["role"], Role) else m["role"]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExerciseDefinition":
        phases = tuple(
            Phase(
                name=p["name"], kind=PhaseKind(p["kind"]), duration=float(p["duration"]),
                pose=dict(p.get("pose", {})), pitch=float(p.get("pitch", 0.0)),
                roll=float(p.get("roll", 0.0)), yaw=float(p.get("yaw", 0.0)),
                lift=float(p.get("lift", 0.0)),
                travel=tuple(p.get("travel", (0.0, 0.0))),
                pivot=tuple(p["pivot"]) if p.get("pivot") is not None else None,
                position=tuple(p["position"]) if p.get("position") is not None else None,
                orientation=p.get("orientation"), cues=tuple(p.get("cues", ())),
                activation=dict(p.get("activation", {})),
                easing=p.get("easing", "ease_in_out"),
            )
            for p in d["phases"]
        )
        muscles = tuple(
            MuscleUse(group=m["group"], role=Role(m["role"]), peak=m.get("peak"),
                      side=m.get("side"), note=m.get("note", ""))
            for m in d["muscles"]
        )
        equipment = tuple(
            EquipmentSpec(kind=e["kind"], attach=e.get("attach", "hands"),
                          position=tuple(e.get("position", (0.0, 0.0, 0.0))),
                          rotation_deg=tuple(e.get("rotation_deg", (0.0, 0.0, 0.0))),
                          params=dict(e.get("params", {})),
                          hang=(None if e.get("hang") is None else float(e["hang"])))
            for e in d.get("equipment", ())
        )
        base = d.get("base_position")
        return cls(
            id=d["id"], name=d["name"], category=Category(d["category"]),
            description=d.get("description", ""), phases=phases, muscles=muscles,
            equipment=equipment, setup=tuple(d.get("setup", ())),
            errors=tuple(d.get("errors", ())), physio_notes=tuple(d.get("physio_notes", ())),
            sources=tuple(d.get("sources", ())), orientation=d.get("orientation", "standing"),
            anchor=d.get("anchor", "feet"), anchor_side=d.get("anchor_side"),
            lock_horizontal=bool(d.get("lock_horizontal", True)),
            base_position=tuple(base) if base is not None else None,
            anchor_point=(tuple(d["anchor_point"]) if d.get("anchor_point") is not None
                          else None),
            default_reps=int(d.get("default_reps", 3)), camera=d.get("camera", "three_quarter"),
            camera_target=(tuple(d["camera_target"]) if d.get("camera_target") is not None
                           else None),
            unilateral=bool(d.get("unilateral", False)), tags=tuple(d.get("tags", ())),
        )


VALID_ORIENTATIONS = ("standing", "supine", "prone", "seated", "hanging", "side")
VALID_ANCHORS = ("feet", "hands", "none")
VALID_ATTACH = ("hands", "hand_r", "hand_l", "static")


def validate_definition(
    defn: ExerciseDefinition,
    known_groups: set[str],
    dof_limits: dict[str, tuple[float, float]] | None = None,
    known_equipment: set[str] | None = None,
) -> list[str]:
    """Return every problem with *defn*; an empty list means it is usable."""
    problems: list[str] = []
    pre = f"{defn.id}: "
    if not defn.id or not defn.name:
        problems.append(pre + "id and name are required")
    if not defn.phases:
        problems.append(pre + "no phases")
    if not defn.muscles:
        problems.append(pre + "no muscles")
    if not defn.muscles_by_role(Role.PRIMARY):
        problems.append(pre + "no primary mover")
    if defn.orientation not in VALID_ORIENTATIONS:
        problems.append(pre + f"orientation {defn.orientation!r} not in {VALID_ORIENTATIONS}")
    for ph in defn.phases:
        if ph.orientation is not None and ph.orientation not in VALID_ORIENTATIONS:
            problems.append(pre + f"phase {ph.name!r}: orientation {ph.orientation!r} unknown")
    if defn.anchor not in VALID_ANCHORS:
        problems.append(pre + f"anchor {defn.anchor!r} not in {VALID_ANCHORS}")
    if not defn.sources:
        problems.append(pre + "no sources")
    for ph in defn.phases:
        if ph.duration <= 0:
            problems.append(pre + f"phase {ph.name!r} has non-positive duration")
        for dof, value in ph.pose.items():
            if dof_limits is not None and dof in dof_limits:
                lo, hi = dof_limits[dof]
                if not (lo - 1e-9 <= value <= hi + 1e-9):
                    problems.append(pre + f"phase {ph.name!r}: {dof}={value:.2f} outside "
                                    f"[{lo}, {hi}]")
            elif dof_limits is not None:
                problems.append(pre + f"phase {ph.name!r}: unknown DOF {dof!r}")
        for key in ph.activation:
            group, _, side = key.partition(":")
            if group not in known_groups or side not in ("", "R", "L"):
                problems.append(pre + f"phase {ph.name!r}: unknown muscle group {key!r}")
    for m in defn.muscles:
        if m.group not in known_groups:
            problems.append(pre + f"unknown muscle group {m.group!r}")
        if m.side not in (None, "R", "L"):
            problems.append(pre + f"muscle side {m.side!r} must be None, 'R' or 'L'")
    for e in defn.equipment:
        if e.attach not in VALID_ATTACH:
            problems.append(pre + f"equipment attach {e.attach!r} not in {VALID_ATTACH}")
        if known_equipment is not None and e.kind not in known_equipment:
            problems.append(pre + f"unknown equipment {e.kind!r}")
    return problems
