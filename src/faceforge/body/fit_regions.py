"""The skeleton as a tree of regions, for fitting it inside a body surface.

The sex morph scales the skeleton *by bone*: every bone takes a factor read
from its name, and the joints move so the articulations stay shut
(:mod:`faceforge.body.skeleton_morph`).  That is the right model for a change
of proportion, because a proportion change is what the factors describe.

Fitting the skeleton into a body-surface mesh is a different problem.  The
surface is a MakeHuman figure and the skeleton a BodyParts3D cadaver: two
different bodies, whose limbs differ in *direction* as well as in length --
measured on the shipped pair, the forearm axes differ by 15 degrees and the
shank axes by 10.  So a region carries a rotation as well as a scale, and the
rotations compose down the chain the way a pose does: turning the humerus at
the shoulder carries the forearm, the hand and every finger with it, and the
forearm's own rotation is then a correction *relative* to that.

::

    pelvis ── lumbar ── thorax ── neck ── head
      │                    ├── girdle_R ── upperarm_R ── forearm_R ── hand_R ── fingers_R
      │                    └── girdle_L ── ...
      ├── thigh_R ── shank_R ── foot_R ── toes_R
      └── thigh_L ── ...

Each region's anchor is the joint it hangs from, and the parent carries it, so
the chain cannot come apart however the parameters are chosen::

    R(r) = R(parent(r)) @ Rlocal(r)        rotations compose, as a pose does
    M(r) = R(r) @ diag(scale(r))
    A'(r) = T(parent(r))(A(r))             the anchor, moved by the parent
    T(r)(x) = A'(r) + M(r) (x - A(r))      everything else, about the anchor

Only two regions carry a translation of their own: the pelvis, because it
hangs from nothing, and the head, because the skull is a group of its own
rather than a bone on a cervical pivot, so moving it opens no joint surface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np
from numpy.typing import NDArray

Vec3 = NDArray[np.float64]

#: Regions allowed to move bodily as well as turn and scale.
OFFSET_REGIONS: tuple[str, ...] = ("pelvis", "head")


@dataclass(frozen=True)
class RegionDef:
    """One region: what it hangs from, and where."""

    name: str
    parent: Optional[str]
    anchor: str


#: The region tree, parents before children so one pass resolves every anchor.
REGIONS: tuple[RegionDef, ...] = (
    RegionDef("pelvis", None, "pelvic_centre"),
    RegionDef("lumbar", "pelvis", "lumbosacral"),
    RegionDef("thorax", "lumbar", "thoracolumbar"),
    RegionDef("neck", "thorax", "cervicothoracic"),
    RegionDef("head", "neck", "craniocervical"),
) + tuple(
    r
    for side in ("R", "L")
    for r in (
        RegionDef(f"girdle_{side}", "thorax", f"sternoclavicular_{side}"),
        RegionDef(f"upperarm_{side}", f"girdle_{side}", f"shoulder_{side}"),
        RegionDef(f"forearm_{side}", f"upperarm_{side}", f"elbow_{side}"),
        RegionDef(f"hand_{side}", f"forearm_{side}", f"wrist_{side}"),
        RegionDef(f"fingers_{side}", f"hand_{side}", f"knuckle_{side}"),
        RegionDef(f"thigh_{side}", "pelvis", f"hip_{side}"),
        RegionDef(f"shank_{side}", f"thigh_{side}", f"knee_{side}"),
        RegionDef(f"foot_{side}", f"shank_{side}", f"ankle_{side}"),
        RegionDef(f"toes_{side}", f"foot_{side}", f"ball_{side}"),
    )
)

REGION_NAMES: tuple[str, ...] = tuple(r.name for r in REGIONS)

#: Which region a node belongs to, by name.  A node that matches nothing
#: inherits its parent's region, so only the heads of subtrees are listed.
#: ``{side}`` is filled from the last capturing group.
_MEMBERSHIP: tuple[tuple[str, str], ...] = (
    (r"^shoulder_([RL])_pivot$", "upperarm_{side}"),
    (r"^elbow_([RL])_pivot$", "forearm_{side}"),
    (r"^wrist_([RL])_pivot$", "hand_{side}"),
    # The knuckle is the break: metacarpals are the hand, phalanges the
    # fingers, which is where the two bodies' hands differ most.
    (r"^finger_([RL])_\d+_mc_pivot$", "hand_{side}"),
    (r"^finger_([RL])_\d+_(?:prox|mid|dist)_pivot$", "fingers_{side}"),
    (r"^hip_([RL])_pivot$", "thigh_{side}"),
    (r"^knee_([RL])_pivot$", "shank_{side}"),
    (r"^ankle_([RL])_pivot$", "foot_{side}"),
    (r"^toe_([RL])_\d+_mt_pivot$", "foot_{side}"),
    (r"^toe_([RL])_\d+_(?:prox|mid|dist)_pivot$", "toes_{side}"),
    (r"^(?:clavicle|scapula)_([RL])_pivot$", "girdle_{side}"),
    (r"^skullGroup$", "head"),
    (r"^vertebraeGroup$", "neck"),
    (r"^(?:thoracic_spine|rib_cage|upper_limb)$", "thorax"),
    (r"^lumbar_spine$", "lumbar"),
)

#: Subtrees the fit never touches.  The body-surface mesh is the target it is
#: fitted *to*, so moving it would be circular; the rest are soft tissue with
#: their own deformers, which rebuild their vertex buffers every frame from
#: rest arrays this would not be writing.
SKIP_SUBTREES: frozenset[str] = frozenset({
    "bodyMeshGroup", "faceGroup", "faceFeatureGroup", "fasciaGroup",
    "brainGroup", "stlMuscleGroup", "exprMuscleGroup", "platysmaGroup",
    "neckMuscleGroup",
})

#: The region a node under ``bodyRoot`` belongs to until something says otherwise.
ROOT_REGION = "pelvis"


def region_of(name: str, parent_region: str) -> str:
    """The region a node belongs to, given its name and its parent's region."""
    for pattern, target in _MEMBERSHIP:
        m = re.fullmatch(pattern, name or "")
        if m is None:
            continue
        side = m.group(1) if m.groups() else ""
        return target.format(side=side)
    return parent_region


def rotation_matrix(degrees: Any) -> NDArray:
    """Rodrigues rotation from a rotation vector given in degrees."""
    v = np.radians(np.asarray(degrees, dtype=np.float64).reshape(3))
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        return np.eye(3)
    k = v / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def anchors(root: Any, joint_positions: dict[str, Any] | None,
            offset_of) -> dict[str, Vec3]:
    """Every region anchor, in body coordinates, from the unfitted skeleton.

    ``offset_of(node)`` gives a node's rest position in body coordinates.  The
    spinal anchors are read from the pivots themselves rather than named,
    because the columns' lengths differ between asset sets.
    """
    jp = {k: np.asarray(v, dtype=np.float64) for k, v in (joint_positions or {}).items()}
    out: dict[str, Vec3] = {}
    for key in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle"):
        for side in ("R", "L"):
            v = jp.get(f"{key}_{side}")
            if v is not None:
                out[f"{key}_{side}"] = v.copy()

    pivots = {n.name: n for n in _walk(root)
              if n.name and "pivot" in n.name.lower()}
    for side in ("R", "L"):
        node = pivots.get(f"clavicle_{side}_pivot")
        if node is not None:
            out[f"sternoclavicular_{side}"] = offset_of(node)
        # The knuckles and the ball of the foot: where the phalanges start.
        for key, prefix in (("knuckle", "finger"), ("ball", "toe")):
            heads = [v for name, v in jp.items()
                     if name.startswith(f"{prefix}_{side}_")
                     and name.endswith("_prox")]
            if heads:
                out[f"{key}_{side}"] = np.mean(heads, axis=0)

    hips = [out[k] for k in ("hip_R", "hip_L") if k in out]
    out["pelvic_centre"] = (np.mean(hips, axis=0) if hips
                            else np.array([0.0, 0.0, -80.0]))

    out["lumbosacral"] = _lowest_of(pivots, "lumbar_spine_pivot_", offset_of,
                                    np.array([0.0, 0.0, -76.0]))
    out["thoracolumbar"] = _lowest_of(pivots, "thoracic_spine_pivot_", offset_of,
                                      np.array([0.0, 0.0, -48.0]))
    cervical = [offset_of(n) for name, n in pivots.items()
                if name.startswith("vertebrae_pivot_")]
    if cervical:
        zs = [float(p[2]) for p in cervical]
        out["cervicothoracic"] = cervical[int(np.argmin(zs))]
        out["craniocervical"] = cervical[int(np.argmax(zs))]
    else:                                    # pragma: no cover - defensive
        out["cervicothoracic"] = np.array([0.0, 0.0, -10.0])
        out["craniocervical"] = np.array([0.0, 0.0, 6.0])
    return out


def _lowest_of(pivots: dict, prefix: str, offset_of, fallback: Vec3) -> Vec3:
    """The lowest pivot whose name starts with ``prefix``: a column's base."""
    points = [offset_of(n) for name, n in pivots.items() if name.startswith(prefix)]
    if not points:
        return fallback
    return points[int(np.argmin([float(p[2]) for p in points]))]


class RegionTransforms:
    """The resolved affine of every region, anchors and rotations carried.

    ``table`` maps a region name to ``{"rotation": (rx, ry, rz) in degrees,
    "scale": (sx, sy, sz), "offset": (x, y, z)}``; every key is optional and
    defaults to no change.  ``amount`` blends the whole fit toward identity, so
    the GUI can show it part-applied and a test can check that zero changes
    nothing -- and it blends the *parameters*, not the matrices, so a
    half-applied rotation is a half rotation rather than a squashed one.
    """

    def __init__(self, table: dict[str, dict], anchor_points: dict[str, Vec3],
                 amount: float = 1.0) -> None:
        self._anchor = anchor_points
        self._amount = float(np.clip(amount, 0.0, 1.0))
        self._mat: dict[str, NDArray] = {}
        self._rot: dict[str, NDArray] = {}
        self._src: dict[str, Vec3] = {}
        self._dst: dict[str, Vec3] = {}
        for rd in REGIONS:
            entry = table.get(rd.name) or {}
            local = rotation_matrix(
                self._amount * np.asarray(
                    entry.get("rotation", (0.0, 0.0, 0.0)), dtype=np.float64))
            parent_rot = (np.eye(3) if rd.parent is None
                          else self._rot[rd.parent])
            rot = parent_rot @ local
            scale = np.asarray(entry.get("scale", (1.0, 1.0, 1.0)),
                               dtype=np.float64)
            scale = 1.0 + self._amount * (scale - 1.0)
            src = anchor_points.get(rd.anchor)
            if src is None:                  # pragma: no cover - defensive
                src = np.zeros(3)
            base = (src if rd.parent is None
                    else self.apply(rd.parent, src[None, :])[0])
            off = np.asarray(entry.get("offset", (0.0, 0.0, 0.0)),
                             dtype=np.float64)
            self._rot[rd.name] = rot
            self._mat[rd.name] = rot @ np.diag(scale)
            self._src[rd.name] = np.asarray(src, dtype=np.float64)
            self._dst[rd.name] = base + self._amount * off

    @property
    def amount(self) -> float:
        return self._amount

    def matrix(self, region: str) -> NDArray:
        return self._mat.get(region, np.eye(3))

    def anchor_pair(self, region: str) -> tuple[Vec3, Vec3]:
        """The region's anchor before and after the fit."""
        zero = np.zeros(3)
        return self._src.get(region, zero), self._dst.get(region, zero)

    def apply(self, region: str, points: NDArray) -> NDArray:
        """Move ``points`` (body coordinates, (N, 3)) by ``region``'s affine."""
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        src, dst = self.anchor_pair(region)
        return dst + (p - src) @ self.matrix(region).T

    def is_identity(self) -> bool:
        """True when applying this would move nothing."""
        for rd in REGIONS:
            if not np.allclose(self._mat[rd.name], np.eye(3), atol=1e-9):
                return False
            if not np.allclose(self._src[rd.name], self._dst[rd.name], atol=1e-9):
                return False
        return True


def blend_tables(male: dict[str, dict], female: dict[str, dict],
                 gender: float) -> dict[str, dict]:
    """One table per sex, lerped: the surface the fit targets is itself lerped."""
    g = float(np.clip(gender, 0.0, 1.0))
    out: dict[str, dict] = {}
    for name in REGION_NAMES:
        a = male.get(name) or {}
        b = female.get(name) or {}
        entry: dict[str, list[float]] = {}
        for key, default in (("rotation", 0.0), ("scale", 1.0), ("offset", 0.0)):
            va = np.asarray(a.get(key, (default,) * 3), dtype=np.float64)
            vb = np.asarray(b.get(key, (default,) * 3), dtype=np.float64)
            entry[key] = (va * (1.0 - g) + vb * g).tolist()
        out[name] = entry
    return out


def _walk(node: Any) -> Iterable[Any]:
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(n.children)
