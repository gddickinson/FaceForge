"""Triangles that are not skin, found by moving the body rather than looking at it.

The BP3D skin arrives with the arm welded to the chest and the hand welded to
the hip: 1,443 lateral-chest vertices touch arm skin directly, at a shortest
surface path of 0.11 units against a median of 36.74.  Telling those welds
from the genuine armpit rim by geometry alone has been tried and does not
work -- it needs a dihedral test, and the mesh's winding is too inconsistent
to support one.

Moving the body settles it.  Real skin stretches a little when the arm turns;
a weld spanning the gap between two limbs stretches without limit, and draws
as a web.  Measured with the skeleton fit on, which pronates the forearm 92
degrees, 554 edges of 2,379,747 pass three times their rest length.  Those
triangles are dropped from what is drawn; the vertices stay, so nothing that
indexes them is disturbed, and the full index buffer is kept so the mesh comes
back whole when the body does.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

#: Stretch past which a triangle is a weld rather than skin.  At 3.0 only 703
#: of 1,586,498 triangles went and the webs were still there, because a web is
#: a fan and most of its triangles are nearer 2 than 3.  Real skin over a
#: moving joint reaches about 1.6 at the 99th percentile, so 2.0 leaves it
#: alone.
WEB_STRETCH = 2.0

#: Where the whole index buffer is kept while part of it is hidden.
FULL_ATTR = "_weld_full_indices"


def cull(mesh: Any, rest: NDArray, limit: float = WEB_STRETCH) -> int:
    """Drop the triangles that have stretched past ``limit``.  Returns how many.

    ``rest`` is the mesh's geometry before it moved.  Passing the current
    positions back restores the whole mesh.
    """
    geometry = getattr(mesh, "geometry", None)
    if geometry is None or geometry.indices is None:
        return 0
    full = getattr(mesh, FULL_ATTR, None)
    if full is None:
        full = np.asarray(geometry.indices).copy()
        setattr(mesh, FULL_ATTR, full)

    tris = full.reshape(-1, 3)
    before = np.asarray(rest, dtype=np.float64).reshape(-1, 3)
    now = np.asarray(mesh.rest_positions if mesh.rest_positions is not None
                     else geometry.positions, dtype=np.float64).reshape(-1, 3)
    if len(before) != len(now):
        return 0

    keep = np.ones(len(tris), dtype=bool)
    for a, b in ((0, 1), (1, 2), (2, 0)):
        was = np.linalg.norm(before[tris[:, a]] - before[tris[:, b]], axis=1)
        is_ = np.linalg.norm(now[tris[:, a]] - now[tris[:, b]], axis=1)
        keep &= is_ <= limit * np.maximum(was, 1e-6)

    dropped = int((~keep).sum())
    geometry.indices = (full if not dropped
                        else tris[keep].reshape(-1).astype(full.dtype))
    mesh.needs_update = True
    if dropped:
        logger.info("%s: %d of %d triangles are welds, not skin; not drawn",
                    getattr(mesh, "name", "?"), dropped, len(tris))
    return dropped


def restore(mesh: Any) -> bool:
    """Put the whole mesh back."""
    full = getattr(mesh, FULL_ATTR, None)
    geometry = getattr(mesh, "geometry", None)
    if full is None or geometry is None:
        return False
    geometry.indices = full
    mesh.needs_update = True
    return True
