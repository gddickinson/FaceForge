"""The part of cranial sex the merged skull mesh cannot express on its own.

The skull arrives as one mesh, so the sex morph can only give it one factor,
and the published dimensions disagree about which factor that should be: head
breadth is 14.5/15.2 = 0.954 female to male, while bizygomatic breadth -- the
width across the cheekbones -- is 12.7/13.7 = 0.927.  The vault and the face
do not shrink together.  Scaling the whole skull at 0.954 leaves the face
about three per cent too wide; scaling it at 0.927 makes the braincase far too
small, and cranial capacity is the better-measured of the two.

So the mesh takes the vault's factor from ``gender_dimorphism.json`` and this
adds the difference back over the lower skull, graded by height so there is no
seam.  The grade runs from the brow down, which also narrows the mastoid
process -- larger in males, and one of the features a forensic anthropologist
sexes a skull by -- for the same reason and at the same time.

It is applied about the midline, where x = 0, so it is exact: no anchor has to
be chosen and nothing can drift off centre.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

#: Meshes this applies to: the merged skull, and nothing else.
SKULL_MESHES: frozenset[str] = frozenset({"cranium"})

#: Extra lateral narrowing of the lower skull at gender 1, on top of the
#: whole-skull factor: 0.927 / 0.954 = 0.972.
FACE_NARROWING = 0.972

#: Where the grade runs, as a fraction of the skull's height measured down
#: from the crown.  0.35 is about the brow; by 0.85, near the alveolar margin,
#: it is fully applied.
GRADE_FROM = 0.35
GRADE_TO = 0.85


def facial_grade(points: NDArray, z_top: float, z_bottom: float) -> NDArray:
    """Smoothstep from 0 at the crown to 1 over the face, per vertex."""
    height = max(z_top - z_bottom, 1e-6)
    depth = (z_top - np.asarray(points, dtype=np.float64)[:, 2]) / height
    t = np.clip((depth - GRADE_FROM) / max(GRADE_TO - GRADE_FROM, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def apply(root: Any, gender: float, exclude: set[int] | None = None) -> int:
    """Narrow the lower skull further than the vault.  Returns meshes changed.

    Call after the skeleton has been scaled; it reads the mesh as it stands and
    writes it back, so it composes with whatever scaled it.
    """
    g = float(max(0.0, min(1.0, gender)))
    skip = exclude or set()
    changed = 0
    stack = [root]
    while stack:
        node = stack.pop()
        stack.extend(node.children)
        mesh = getattr(node, "mesh", None)
        if mesh is None or (node.name or "") not in SKULL_MESHES:
            continue
        if id(mesh) in skip:
            continue
        source = (mesh.rest_positions if mesh.rest_positions is not None
                  else mesh.geometry.positions)
        if source is None:
            continue
        pts = np.asarray(source, dtype=np.float64).reshape(-1, 3)
        if not len(pts):
            continue
        if g > 0.0:
            t = facial_grade(pts, float(pts[:, 2].max()), float(pts[:, 2].min()))
            factor = 1.0 + t * g * (FACE_NARROWING - 1.0)
            pts = pts.copy()
            pts[:, 0] *= factor
        flat = pts.reshape(-1).astype(np.float32)
        mesh.geometry.positions = flat
        mesh.rest_positions = flat.copy()
        mesh.needs_update = True
        changed += 1
    if changed and g > 0.0:
        logger.info("Skull face narrowed to %.3f of the vault at gender %.2f",
                    1.0 + g * (FACE_NARROWING - 1.0), g)
    return changed
