"""The head's soft tissue, which owns its own rest pose and so gets left behind.

Everything below the neck is bound to the skinning, so when the skeleton moves
-- a sex morph, or a fit into the body-surface mesh -- the soft tissue is
carried with it by a displacement field and the bindings are re-snapshotted.

The head is not like that.  The neck, jaw and expression muscles, the face and
the face features each keep their own copy of their rest pose and rebuild
their vertex buffers from it every frame, because each has its own deformer:
head-follow, jaw angle, action units.  Nothing was rewriting those copies, so
when the skull moved -- and it now moves a good deal, being reshaped for sex
and seated lower on the neck, and moved again by the skeleton fit -- the
muscles on it stayed exactly where they were.

The brain is the same problem arrived at from the other end.  It hangs off
``brainGroup`` rather than off the skull, deliberately, so that it stays
visible when the skull is hidden; the cost is that nothing in the scene graph
carries it when the skull moves.  Measured with the fit on, the skull came
down from z +0.1..+27.6 to -21.7..+2.5 and the brain stayed at +5.6..+27.1 --
a whole head-height above the body, outside the skin entirely.  It is soft
tissue inside the cranial vault, so it belongs here.

This walks those systems and moves their rest poses by the same field the
skinning gets.  Each system's original is captured the first time and every
later call recomputes from it, so the result never compounds and switching a
fit off puts everything back.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

#: Attribute the captured original is parked on, alongside the live rest.
BASE_ATTR = "_head_tissue_base"


@dataclass
class Owned:
    """One rest-pose array a head system owns, and how to put it back."""

    owner: Any
    field: str
    mesh: Any
    refresh: Optional[Callable[[], None]] = None

    def base(self) -> NDArray:
        """The rest pose as it was before anything moved it."""
        stored = getattr(self.owner, BASE_ATTR, None) or {}
        if self.field not in stored:
            stored = dict(stored)
            stored[self.field] = np.array(getattr(self.owner, self.field),
                                          copy=True)
            setattr(self.owner, BASE_ATTR, stored)
        return stored[self.field]

    def write(self, points: NDArray) -> None:
        flat = np.asarray(points, dtype=np.float64).reshape(-1).astype(
            np.asarray(getattr(self.owner, self.field)).dtype)
        setattr(self.owner, self.field, flat)
        if self.mesh is not None:
            self.mesh.geometry.positions[:len(flat)] = flat
            self.mesh.needs_update = True
        if self.refresh is not None:
            self.refresh()


def brain_meshes(brain_group: Any) -> list[Any]:
    """Every mesh under ``brainGroup``.

    Their vertices are already in body coordinates -- the group sits on
    ``bodyRoot`` with an identity transform -- so the field can be read at
    them directly.
    """
    out: list[Any] = []
    if brain_group is None:
        return out
    stack = [brain_group]
    while stack:
        node = stack.pop()
        stack.extend(getattr(node, "children", ()))
        mesh = getattr(node, "mesh", None)
        if mesh is not None and getattr(mesh, "rest_positions", None) is not None:
            out.append(mesh)
    return out


def collect(pipeline: Any, brain_group: Any = None) -> list[Owned]:
    """Every rest pose in the head that no one else is moving."""
    from faceforge.anatomy import neck_fibre_strain

    out: list[Owned] = []

    neck = getattr(pipeline, "neck_muscles", None)
    for md in getattr(neck, "_muscles", ()) or ():
        # The fibre axis, the centroids and the radial offsets are all
        # measured from the rest pose, so they are re-derived after it moves.
        out.append(Owned(md, "rest_positions", md.mesh,
                         lambda md=md: neck_fibre_strain.init_fiber_geometry(md)))

    for system in ("jaw_muscles", "expression_muscles"):
        holder = getattr(pipeline, system, None)
        for md in getattr(holder, "_muscles", ()) or ():
            out.append(Owned(md, "rest_positions", md.mesh))

    facs = getattr(pipeline, "facs_engine", None)
    if facs is not None and getattr(facs, "_rest", None) is not None:
        out.append(Owned(facs, "_rest", getattr(facs, "_mesh", None)))

    features = getattr(pipeline, "face_features", None)
    for group in ("_features", "_eyeballs"):
        for md in getattr(features, group, ()) or ():
            if getattr(md, "rest_positions", None) is None:
                continue
            # A mesh under a joint is expressed in that joint's frame, so a
            # field evaluated in body coordinates would be read at the wrong
            # place; and it rides with the joint anyway.
            if _under_a_pivot(getattr(md, "node", None)):
                continue
            out.append(Owned(md, "rest_positions", getattr(md, "mesh", None)))

    for mesh in brain_meshes(brain_group):
        out.append(Owned(mesh, "rest_positions", mesh))
    return out


def _under_a_pivot(node: Any) -> bool:
    while node is not None:
        if "pivot" in (getattr(node, "name", "") or "").lower():
            return True
        node = getattr(node, "parent", None)
    return False


def owned_meshes(pipeline: Any, brain_group: Any = None) -> set[int]:
    """``id(mesh)`` of everything in the head that owns its own rest pose.

    The skeleton morph is handed this as an exclusion.  Without it a muscle
    whose name reads like a bone is scaled as one -- "Zygomatic Maj." matches
    the zygomatic bone's pattern -- and then fights the head's own deformer
    for the same vertex buffer.
    """
    return {id(r.mesh) for r in collect(pipeline, brain_group)
            if r.mesh is not None}


def rebase(pipeline: Any, warp: Optional[Callable[[NDArray], NDArray]],
           exclude: set[int] | None = None, brain_group: Any = None) -> int:
    """Move every head rest pose by ``warp``.  ``None`` puts them all back."""
    skip = exclude or set()
    records = collect(pipeline, brain_group)
    moved = 0
    for record in records:
        if record.mesh is not None and id(record.mesh) in skip:
            continue
        base = record.base()
        pts = base.reshape(-1, 3).astype(np.float64)
        record.write(pts if warp is None else pts + warp(pts))
        moved += 1

    # The platysma measures its own rest from the expression muscles', so it
    # has to be told: it spans the jaw to the chest and is the one head muscle
    # a body change pulls on from both ends.
    platysma = getattr(pipeline, "platysma", None)
    muscles = getattr(getattr(pipeline, "expression_muscles", None), "_muscles", None)
    if platysma is not None and muscles and hasattr(platysma, "register"):
        try:
            platysma.register(muscles)
        except Exception as exc:                     # noqa: BLE001 - logged
            logger.warning("Platysma not re-registered after a head move: %s", exc)

    if moved:
        logger.info("Head soft tissue moved with the skull: %d rest poses",
                    moved)
    return moved
