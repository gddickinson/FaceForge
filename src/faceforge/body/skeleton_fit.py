"""Fit the skeleton inside the body-surface mesh, instead of the other way.

The body-surface mesh is a MakeHuman figure; the skeleton is a BodyParts3D
cadaver.  They are different bodies, and until now the only way they were
reconciled was by deforming the *surface* onto the skeleton -- which every
measurement in ``docs/sex_morph.md`` shows damages the mesh, and which is
switched off (``gender_morph.WARP_SURFACE_TO_SKELETON``).

Left alone, the two disagree badly.  Measured on the shipped pair, 65.7% of
the skeleton's vertices lie outside the body surface, a median of 2.4 units
out and 11.4 at the 95th percentile; the cranium is outside along its whole
length, the scapula and humerus by 12 to 19 units, and every bone of the foot.

This module moves the skeleton instead.  The skeleton is a tree of regions
(:mod:`faceforge.body.fit_regions`), each carrying a 3x3 matrix about the
joint it hangs from, and each region's anchor is carried by its parent -- so
a limb may be shortened, turned and thinned to lie inside its sleeve without
any articulation coming apart.  The matrices are solved offline against the
real surface by ``tools/fit_skeleton_to_skin.py`` and shipped in
``assets/config/skeleton_fit.json``; applying them at runtime is a walk of
the scene graph.

What it does not touch: the body-surface mesh itself (it is the target), and
every soft-tissue group with its own deformer -- the face, the facial, jaw and
neck muscles, the fascia markers and the brain.  Muscles and skin bound to the
skinning follow the joints the way they follow a pose; the caller re-snapshots
them exactly as the sex morph does.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import numpy as np
from numpy.typing import NDArray

from faceforge.body.fit_regions import (
    REGION_NAMES, ROOT_REGION, RegionTransforms, SKIP_SUBTREES, anchors,
    blend_tables, region_of, tree_distance,
)

#: Region name to its column in the per-query weight table.
REGION_INDEX: dict[str, int] = {n: i for i, n in enumerate(REGION_NAMES)}
from faceforge.body.skeleton_field import sampled_warp
from faceforge.core.config_loader import load_config

logger = logging.getLogger(__name__)

Vec3 = NDArray[np.float64]

#: Where the solved matrices live.  A missing or unreadable file leaves the
#: fit as the identity, so the option is always safe to switch on.
CONFIG_NAME = "skeleton_fit.json"

#: Points sampled from each bone as control points for the soft-tissue field.
#: The centroid alone is not enough: a long bone that turns as well as scales
#: displaces its two ends in opposite directions, and a single point in the
#: middle records neither.
SAMPLES_PER_BONE = 120

#: Control points blended per query point, and the distance added to every
#: weight.  Measured on the muscle layers, as the worst muscle's 99th
#: percentile edge stretch and how far the quadriceps stand through the skin
#: (they stood 3.86 out before the fit touched them):
#:
#:     neighbours / smoothing   worst p99   thigh out
#:     64 / 16                    2.148        3.44
#:     32 / 8                     1.944        4.17
#:     16 / 3                     1.963        1.35
#:     24 / 4                     1.641        1.23
#:     32 / 4                     1.611        1.17
FIELD_NEIGHBOURS = 32
FIELD_SMOOTHING = 4.0

#: How many steps along the region tree a control point may be from the
#: region a query point actually sits in.  Space is not the body: the finger
#: bones hang beside the thigh, close enough to take 31% of the weight on the
#: quadriceps and seven steps away in the skeleton.  Carrying that weight, and
#: with it a 92-degree pronation extrapolated 30 units, put the quadriceps 16
#: units through the skin.
REGION_REACH = 2.0

#: Lattice the field is sampled on.  Sampling it is the whole cost of
#: switching the option on; 3 units is fine enough that the lattice is not
#: what limits the answer.
FIELD_LATTICE = 3.0


#: The skin that came with the skeleton.  It was scanned from these bones and
#: already fits them, so the fit never carries it: pulling it onto the surface
#: mesh's proportions is precisely the distortion this feature exists to avoid.
OWN_SKIN_SOURCES = frozenset({"FMA7163"})
OWN_SKIN_NAMES = frozenset({"Skin"})


def is_own_skin(mesh: Any) -> bool:
    """True for the skeleton's own skin layer, matched by source id or name."""
    if mesh is None:
        return False
    source = getattr(mesh, "source_id", "") or ""
    if source in OWN_SKIN_SOURCES:
        return True
    return (getattr(mesh, "name", "") or "") in OWN_SKIN_NAMES


def _is_pivot(node: Any) -> bool:
    return "pivot" in (getattr(node, "name", "") or "").lower()


def _walk(node: Any) -> Iterable[Any]:
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(n.children)


def node_offset(node: Any, stop: str = "bodyRoot") -> Vec3:
    """A node's position in body coordinates, by summing its ancestors'.

    Pivot rotations are identity at rest, so the sum *is* the world position,
    and reading it this way does not depend on the scene having been updated
    or on whatever pose happens to be applied.
    """
    out = np.zeros(3)
    n = node
    while n is not None and getattr(n, "name", "") != stop:
        out = out + np.asarray(n.position, dtype=np.float64)
        n = n.parent
    return out


def load_table() -> dict[str, dict[str, dict]]:
    """The solved fit, as ``{"male": {...}, "female": {...}}``."""
    empty = {name: {} for name in REGION_NAMES}
    try:
        cfg = load_config(CONFIG_NAME)
    except (OSError, ValueError) as exc:
        logger.info("No skeleton fit shipped (%s): the option will do nothing",
                    exc)
        return {"male": dict(empty), "female": dict(empty)}
    out = {}
    for sex in ("male", "female"):
        table = cfg.get(sex) or {}
        out[sex] = {name: (table.get(name) or {}) for name in REGION_NAMES}
    return out


@dataclass
class _Rest:
    """The skeleton as it stood before the fit was applied."""

    pivot_positions: dict[int, Vec3] = field(default_factory=dict)
    mesh_positions: dict[int, NDArray[np.float32]] = field(default_factory=dict)
    joint_positions: dict[str, Vec3] = field(default_factory=dict)


class SkeletonFit:
    """Moves and deforms the skeleton so it sits inside a body-surface mesh.

    ``apply`` always starts from the unfitted skeleton: if a fit is already in
    place it is undone first, so the option can be toggled and the sex slider
    moved in any order without the two compounding.
    """

    def __init__(self, table: dict[str, dict[str, dict]] | None = None) -> None:
        self._table = table if table is not None else load_table()
        self._rest = _Rest()
        self._applied = False
        self._amount = 0.0
        self._control: tuple[NDArray, NDArray] | None = None
        self._transforms: Any = None

    # -- state ---------------------------------------------------------------

    @property
    def applied(self) -> bool:
        """Whether a fit is currently in place on the skeleton."""
        return self._applied

    @property
    def amount(self) -> float:
        return self._amount

    @property
    def available(self) -> bool:
        """False when no solved fit is shipped, so the option would do nothing."""
        for sex in ("male", "female"):
            for entry in self._table.get(sex, {}).values():
                if any(entry.get(key) for key in ("rotation", "scale", "offset")):
                    return True
        return False

    # -- capture and restore -------------------------------------------------

    def _capture(self, root: Any, joint_positions: dict[str, Any] | None) -> None:
        """Snapshot every pivot, and the joint table.

        Bone geometry is snapshotted lazily, as each mesh is written, so that
        ``reset`` restores exactly the meshes the fit touched and nothing
        else: the excluded soft tissue is rebuilt from the morph's own
        captured original and must not be handed a second one here.
        """
        r = _Rest()
        for node in self._fit_nodes(root):
            if _is_pivot(node):
                r.pivot_positions[id(node)] = np.asarray(
                    node.position, dtype=np.float64).copy()
        if joint_positions:
            r.joint_positions = {k: np.asarray(v, dtype=np.float64).copy()
                                 for k, v in joint_positions.items()}
        self._rest = r

    def reset(self, root: Any, joint_positions: dict[str, Any] | None = None) -> None:
        """Put the skeleton back exactly as it was before the fit."""
        if not self._applied:
            return
        r = self._rest
        for node in self._fit_nodes(root):
            rest = r.pivot_positions.get(id(node))
            if rest is not None:
                node.set_position(float(rest[0]), float(rest[1]), float(rest[2]))
            mesh = getattr(node, "mesh", None)
            if mesh is None:
                continue
            pos = r.mesh_positions.get(id(mesh))
            if pos is None:
                continue
            mesh.geometry.positions = pos.copy()
            mesh.rest_positions = pos.copy()
            mesh.needs_update = True
        if joint_positions is not None and r.joint_positions:
            for k, v in r.joint_positions.items():
                if k in joint_positions:
                    joint_positions[k] = v.copy()
        self._applied = False
        self._amount = 0.0
        self._control = None
        self._transforms = None
        logger.info("Skeleton fit removed")

    # -- apply ---------------------------------------------------------------

    def apply(self, root: Any, amount: float = 1.0, gender: float = 0.0,
              joint_positions: dict[str, Any] | None = None,
              exclude: set[int] | None = None) -> dict[str, int]:
        """Fit the skeleton into the body surface at ``gender``.

        ``amount`` blends the whole fit toward the unfitted skeleton, so zero
        is exactly the skeleton the caller passed in.  ``exclude`` is the set
        of ``id(mesh)`` the skinning owns: soft tissue follows the joints, it
        is not transformed here.
        """
        self.reset(root, joint_positions)
        amount = float(np.clip(amount, 0.0, 1.0))
        if amount <= 0.0:
            return {"bones": 0, "pivots": 0}

        self._capture(root, joint_positions)
        table = blend_tables(self._table.get("male", {}),
                             self._table.get("female", {}), gender)
        points = anchors(root, joint_positions, node_offset)
        transforms = RegionTransforms(table, points, amount)
        if transforms.is_identity():
            logger.info("Skeleton fit at gender %.2f is the identity: nothing "
                        "to apply", gender)
            return {"bones": 0, "pivots": 0}

        stats = {"bones": 0, "pivots": 0}
        src: list[Vec3] = []
        dst: list[int] = []
        self._walk_apply(root, transforms, ROOT_REGION, np.zeros(3), np.zeros(3),
                         exclude or set(), stats, src, dst)
        self._applied = True
        self._amount = amount
        self._transforms = transforms
        self._control = ((np.asarray(src), np.asarray(dst, dtype=np.int64))
                         if src else None)
        if joint_positions is not None:
            self._update_joint_positions(root, joint_positions)
        logger.info("Skeleton fitted to the body surface at gender %.2f "
                    "(amount %.2f): %d bones, %d joints moved",
                    gender, amount, stats["bones"], stats["pivots"])
        return stats

    def _walk_apply(self, node: Any, transforms: RegionTransforms,
                    region: str, rest_parent: Vec3, new_parent: Vec3,
                    exclude: set[int], stats: dict[str, int],
                    src: list[Vec3], dst: list[Vec3]) -> None:
        for child in node.children:
            name = getattr(child, "name", "") or ""
            if name in SKIP_SUBTREES:
                continue
            child_region = region_of(name, region)
            rest_local = self._rest.pivot_positions.get(
                id(child), np.asarray(child.position, dtype=np.float64))
            rest_body = rest_parent + rest_local
            if _is_pivot(child):
                new_body = transforms.apply(child_region, rest_body[None, :])[0]
                local = new_body - new_parent
                child.set_position(float(local[0]), float(local[1]), float(local[2]))
                stats["pivots"] += 1
                src.append(rest_body)
                dst.append(REGION_INDEX[child_region])
            else:
                new_body = new_parent + np.asarray(child.position, dtype=np.float64)

            mesh = getattr(child, "mesh", None)
            if mesh is not None and name and id(mesh) not in exclude:
                if self._transform_mesh(mesh, transforms, child_region,
                                        rest_body, new_body, src, dst):
                    stats["bones"] += 1
            self._walk_apply(child, transforms, child_region, rest_body,
                             new_body, exclude, stats, src, dst)

    def _transform_mesh(self, mesh: Any, transforms: RegionTransforms,
                        region: str, rest_body: Vec3, new_body: Vec3,
                        src: list[Vec3], dst: list[int]) -> bool:
        rest = self._rest.mesh_positions.get(id(mesh))
        if rest is None:
            authored = (mesh.rest_positions if mesh.rest_positions is not None
                        else mesh.geometry.positions)
            if authored is None:
                return False
            rest = np.asarray(authored, dtype=np.float32).copy()
            self._rest.mesh_positions[id(mesh)] = rest
        pts = np.asarray(rest, dtype=np.float64).reshape(-1, 3) + rest_body
        moved = transforms.apply(region, pts)
        step = max(1, len(pts) // SAMPLES_PER_BONE)
        sample = pts[::step]
        src.extend(sample)
        dst.extend([REGION_INDEX[region]] * len(sample))
        flat = (moved - new_body).reshape(-1).astype(np.float32)
        mesh.geometry.positions = flat
        mesh.rest_positions = flat.copy()
        mesh.needs_update = True
        return True

    def _fit_nodes(self, root: Any) -> Iterable[Any]:
        """Every node the fit may write to: the skeleton, not the soft tissue."""
        stack = [root]
        while stack:
            n = stack.pop()
            if (getattr(n, "name", "") or "") in SKIP_SUBTREES:
                continue
            yield n
            stack.extend(n.children)

    def _update_joint_positions(self, root: Any,
                                joint_positions: dict[str, Any]) -> None:
        """Refresh ``JointSetup.joint_positions`` so rest-pose consumers agree.

        The ground lock and the equipment rig measure against these; left at
        the unfitted skeleton's values they would put the feet through the
        floor.
        """
        by_name = {n.name: n for n in _walk(root) if _is_pivot(n) and n.name}
        for key in list(joint_positions):
            node = by_name.get(f"{key}_pivot")
            if node is None:
                continue
            joint_positions[key] = node_offset(node)

    # -- the change as a smooth field ----------------------------------------

    def displacement_field(self, sampled: bool = True) -> Optional[Any]:
        """A smooth warp taking the unfitted body's soft tissue to the fitted one.

        Routing a translation of the joints through the articulated skinning
        tears the skin at every chain boundary, so the soft tissue is carried
        by a field instead -- the same reasoning as
        :meth:`SkeletonMorph.displacement_field`.

        It is *not* the same field.  The sex morph uses a thin-plate spline,
        which is an interpolant: outside the hull of its control points it
        extrapolates, and a fit that turns the shoulder girdle by eight
        degrees made it extrapolate hard -- the trapezius and deltoid came
        away from the thorax in wings.

        Nor does it blend the displacements themselves, which was the next
        thing tried.  A displacement blend cannot extrapolate a *scale*: shrink
        a femur by a tenth and the blend carries the bone's own surface
        correctly, but a muscle eight units outside it barely moves, because
        every measured displacement nearby is on the bone.  Rendered, the
        quadriceps ballooned out past the leg -- 15 units through the skin
        where they had been 4 -- and no neighbourhood or smoothing changed it.

        Nor a blend of the region *matrices*, which extrapolates correctly but
        is not closed: a weighted average of two rotation matrices is not a
        rotation, and with a forearm turned 92 degrees the averages collapse.
        The worst muscle's 99th-percentile edge stretch went from 2.21 to 5.37.

        So the regions' transforms are blended in the terms they are made of:
        rotations as quaternions, which stay rotations however they are mixed;
        scales and anchors linearly.  That extrapolates a scale outward from a
        bone, which is what "follow the skeleton" means, and it survives a
        right-angled turn.  It is dual-quaternion skinning with the regions as
        bones and inverse distance for weights.
        """
        if self._control is None or self._transforms is None:
            return None
        pts, regions = self._control
        if len(pts) == 0:
            return None
        warp = _inverse_distance_warp(pts, regions, self._transforms)
        if not sampled:
            return warp

        # Sampling the field on its lattice is the whole cost of switching the
        # option on, and the toggle has to answer inside the 16 ms render
        # timer.  Nothing is sampled until a mesh actually asks to be moved.
        cache: dict[str, Any] = {}

        def lazy(query: NDArray) -> NDArray:
            field = cache.get("field")
            if field is None:
                field = cache["field"] = sampled_warp(
                    warp, pts, spacing=FIELD_LATTICE)
            return field(query)

        return lazy


def _inverse_distance_warp(points: NDArray, regions: NDArray, transforms: Any):
    """Blend where each region *puts* a point, weighted by inverse distance.

    Not a blend of the displacements, which cannot extrapolate a scale: shrink
    a femur by a tenth and a muscle eight units outside it barely moves,
    because every measured displacement nearby is on the bone.  Rendered, the
    quadriceps ballooned out past the leg.  And not a blend of the matrices,
    which is not closed under averaging: with a forearm turned 92 degrees the
    averages collapse.  Each region's affine is applied to the query point and
    the *results* are mixed, every one of which is correct.

    With one guard.  An affine extrapolates, and a region's affine evaluated
    far outside it extrapolates wildly: the finger bones hang beside the
    thigh, and the fingers' affine -- carrying that same 92-degree turn --
    moves a point on the quadriceps by 50 units.  Weighted at 0.31 by nothing
    but proximity, that was the whole of a 16-unit error.  So no region may
    move a point further than it moved its own bones.
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    ids = np.asarray(regions, dtype=np.int64).reshape(-1)
    names = list(REGION_NAMES)
    mats = np.stack([transforms.matrix(n) for n in names])
    src = np.stack([transforms.anchor_pair(n)[0] for n in names])
    dst = np.stack([transforms.anchor_pair(n)[1] for n in names])
    steps = tree_distance()
    reach = np.zeros(len(names))
    for r, name in enumerate(names):
        own = ids == r
        if own.any():
            moved = dst[r] + (pts[own] - src[r]) @ mats[r].T
            reach[r] = float(np.linalg.norm(moved - pts[own], axis=1).max())
    tree = cKDTree(pts)
    k = int(min(FIELD_NEIGHBOURS, len(pts)))

    def warp(query: NDArray) -> NDArray:
        q = np.asarray(query, dtype=np.float64).reshape(-1, 3)
        if len(q) == 0:
            return np.zeros((0, 3))
        d, idx = tree.query(q, k=k)
        if d.ndim == 1:
            d = d[:, None]
            idx = idx[:, None]
        near = ids[idx]                                # (Q, k) region per neighbour
        # Whichever region owns the nearest bone is the part of the body this
        # point belongs to; a neighbour from further than REGION_REACH steps
        # away along the tree is a different part that merely hangs close.
        keep = steps[near[:, 0][:, None], near] <= REGION_REACH
        w = np.where(keep, 1.0 / (d + FIELD_SMOOTHING), 0.0)
        total = w.sum(axis=1, keepdims=True)
        w = np.divide(w, total, out=np.zeros_like(w), where=total > 0)
        share = np.zeros((len(q), len(names)), dtype=np.float64)
        np.add.at(share, (np.repeat(np.arange(len(q)), k), ids[idx].ravel()),
                  w.ravel())
        out = np.zeros_like(q)
        for r in range(len(names)):
            weight = share[:, r]
            live = weight > 1e-9
            if not live.any():
                continue
            delta = (dst[r] + (q[live] - src[r]) @ mats[r].T) - q[live]
            size = np.linalg.norm(delta, axis=1, keepdims=True)
            limit = max(reach[r], 1e-9)
            delta = delta * np.minimum(1.0, limit / np.maximum(size, 1e-9))
            out[live] += weight[live, None] * delta
        return out

    return warp
