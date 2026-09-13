"""The skeleton as a tree of regions, for fitting it inside a body surface.

The sex morph scales the skeleton *by bone*: every bone takes a factor read
from its name, and the joints move so the articulations stay shut
(:mod:`faceforge.body.skeleton_morph`).  That is the right model for a change
of proportion, because a proportion change is what the factors describe.

Fitting the skeleton into a body-surface mesh is a different problem.  The
surface is a MakeHuman figure and the skeleton is a BodyParts3D cadaver: two
different bodies, whose limbs differ in *direction* as well as in length --
measured on the shipped pair, the forearm axes differ by 15 degrees and the
shank axes by 10.  A per-axis scale cannot express that, so a region here
carries a full 3x3 matrix.

A region is a segment of the skeleton plus everything hanging off it, and it
is anchored at the joint it hangs from::

    trunk ── neck ── head
      ├── girdle_R ── upperarm_R ── forearm_R ── hand_R
      ├── girdle_L ── ...
      ├── thigh_R  ── shank_R ── foot_R
      └── thigh_L  ── ...

Each region's anchor is carried by its parent, so the chain cannot come
apart however the matrices are chosen:

    A'(r) = T(parent(r))(A(r))          the anchor, moved by the parent
    T(r)(x) = A'(r) + M(r) (x - A(r))   everything else, about the anchor

Only the trunk has nowhere to hang from, so only the trunk carries a
translation of its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np
from numpy.typing import NDArray

Vec3 = NDArray[np.float64]


@dataclass(frozen=True)
class RegionDef:
    """One region: what it hangs from, and where."""

    name: str
    parent: Optional[str]
    anchor: str


#: The region tree, parents before children so one pass resolves every anchor.
REGIONS: tuple[RegionDef, ...] = (
    RegionDef("trunk", None, "pelvis"),
    RegionDef("neck", "trunk", "cervicothoracic"),
    RegionDef("head", "neck", "craniocervical"),
) + tuple(
    r
    for side in ("R", "L")
    for r in (
        RegionDef(f"girdle_{side}", "trunk", f"sternoclavicular_{side}"),
        RegionDef(f"upperarm_{side}", f"girdle_{side}", f"shoulder_{side}"),
        RegionDef(f"forearm_{side}", f"upperarm_{side}", f"elbow_{side}"),
        RegionDef(f"hand_{side}", f"forearm_{side}", f"wrist_{side}"),
        RegionDef(f"thigh_{side}", "trunk", f"hip_{side}"),
        RegionDef(f"shank_{side}", f"thigh_{side}", f"knee_{side}"),
        RegionDef(f"foot_{side}", f"shank_{side}", f"ankle_{side}"),
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
    (r"^hip_([RL])_pivot$", "thigh_{side}"),
    (r"^knee_([RL])_pivot$", "shank_{side}"),
    (r"^ankle_([RL])_pivot$", "foot_{side}"),
    (r"^(?:clavicle|scapula)_([RL])_pivot$", "girdle_{side}"),
    (r"^skullGroup$", "head"),
    (r"^vertebraeGroup$", "neck"),
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


def region_of(name: str, parent_region: str) -> str:
    """The region a node belongs to, given its name and its parent's region."""
    for pattern, target in _MEMBERSHIP:
        m = re.fullmatch(pattern, name or "")
        if m is None:
            continue
        side = m.group(1) if m.groups() else ""
        return target.format(side=side)
    return parent_region


def anchors(root: Any, joint_positions: dict[str, Any] | None,
            offset_of) -> dict[str, Vec3]:
    """Every region anchor, in body coordinates, from the unfitted skeleton.

    ``offset_of(node)`` gives a node's rest position in body coordinates; the
    cervical anchors are read from the cervical pivots themselves rather than
    named, because the column's length differs between asset sets.
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

    hips = [out[k] for k in ("hip_R", "hip_L") if k in out]
    out["pelvis"] = (np.mean(hips, axis=0) if hips
                     else np.array([0.0, 0.0, -80.0]))

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


class RegionTransforms:
    """The resolved affine of every region, anchors already carried.

    ``table`` maps a region name to a 3x3 matrix, and the trunk may also carry
    a 3-vector ``offset``.  ``amount`` blends the whole fit toward identity, so
    the GUI can show it part-applied and a test can check that zero changes
    nothing.
    """

    def __init__(self, table: dict[str, dict], anchor_points: dict[str, Vec3],
                 amount: float = 1.0) -> None:
        self._anchor = anchor_points
        self._amount = float(np.clip(amount, 0.0, 1.0))
        self._mat: dict[str, NDArray] = {}
        self._src: dict[str, Vec3] = {}
        self._dst: dict[str, Vec3] = {}
        eye = np.eye(3)
        for rd in REGIONS:
            entry = table.get(rd.name) or {}
            m = np.asarray(entry.get("matrix", eye), dtype=np.float64).reshape(3, 3)
            m = eye + self._amount * (m - eye)
            src = anchor_points.get(rd.anchor)
            if src is None:                  # pragma: no cover - defensive
                src = np.zeros(3)
            base = (src if rd.parent is None
                    else self.apply(rd.parent, src[None, :])[0])
            # An offset moves the region bodily, on top of where its parent
            # carried it.  The trunk needs one because it hangs from nothing;
            # the head needs one because the skull is not parented to a
            # cervical pivot -- it is a group of its own, and the
            # atlanto-occipital "joint" here is a reference point, not a
            # contact surface that a translation could open.
            off = np.asarray(entry.get("offset", (0.0, 0.0, 0.0)),
                             dtype=np.float64)
            dst = base + self._amount * off
            self._mat[rd.name] = m
            self._src[rd.name] = np.asarray(src, dtype=np.float64)
            self._dst[rd.name] = np.asarray(dst, dtype=np.float64)

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
    eye = np.eye(3)
    out: dict[str, dict] = {}
    for name in REGION_NAMES:
        a = male.get(name) or {}
        b = female.get(name) or {}
        ma = np.asarray(a.get("matrix", eye), dtype=np.float64).reshape(3, 3)
        mb = np.asarray(b.get("matrix", eye), dtype=np.float64).reshape(3, 3)
        oa = np.asarray(a.get("offset", (0.0, 0.0, 0.0)), dtype=np.float64)
        ob = np.asarray(b.get("offset", (0.0, 0.0, 0.0)), dtype=np.float64)
        out[name] = {"matrix": (ma * (1.0 - g) + mb * g).tolist(),
                     "offset": (oa * (1.0 - g) + ob * g).tolist()}
    return out


def _walk(node: Any) -> Iterable[Any]:
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(n.children)
