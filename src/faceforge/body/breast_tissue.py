"""Mammary tissue: the lens between the skin and the chest wall.

The BodyParts3D set is a male cadaver and has no breast in it, and the first
attempt here built one out of the difference between the two body surfaces --
the shape a female chest has and a male one does not.  That is the wrong
object twice over.  The MakeHuman female base is modest: measured, the chest
front gains one unit of projection over the male, so the tissue came out a
sliver.  And anatomically the inner boundary of the breast is not the male
skin, it is the pectoral fascia, several units deeper.

So it is built as what it is: a lens over ribs two to six, from the skin down
to the chest wall, thickest at the nipple and tapering to nothing at its base.
Men have mammary tissue too -- a thin disc behind the areola -- so the depth
runs from a little to a lot with the slider rather than from nothing.

The surface normals of this mesh point *inward*; that is a property of the
asset and it inverted the first attempt's whole selection, so the outward
direction is taken as the negated normal and checked against the anatomy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

#: Where to look for the nipple: the chest, off the midline, and above the
#: costal margin.
CHEST_Z = (-68.0, -42.0)
NIPPLE_X = (4.0, 22.0)

#: Radius of the tissue's base, in body units.  A breast base is 10 to 12 cm
#: across and the model is 200 units to 176 cm, so 9 units is about 8 cm.
BASE_RADIUS = 9.0

#: Depth at the nipple, male and female.  About 0.7 cm and 3.5 cm: men have a
#: disc of tissue behind the areola, women a cone.
DEPTH = (0.8, 4.0)

#: Smallest patch worth keeping, in vertices: enough to drop stray specks
#: without dropping either breast.
MIN_PATCH = 24


@dataclass
class BreastTissue:
    """A lens under the skin of the chest, deepest at the nipple."""

    #: Vertex indices into the source surface, in patch order.
    source: NDArray
    #: Triangles of the shell, indexing 2 * len(source) vertices: the outer
    #: face first, then the inner.
    faces: NDArray
    #: The skin at those vertices at each sex, the inward direction, and how
    #: much of the full depth each vertex carries.
    skin_male: NDArray
    skin_female: NDArray
    inward: NDArray
    profile: NDArray

    @property
    def vertex_count(self) -> int:
        return 2 * len(self.source)

    def positions(self, gender: float) -> NDArray:
        """The lens at a given sex: the skin outside, the chest wall inside."""
        g = float(max(0.0, min(1.0, gender)))
        skin = self.skin_male * (1.0 - g) + self.skin_female * g
        depth = DEPTH[0] + g * (DEPTH[1] - DEPTH[0])
        inner = skin + self.inward * (depth * self.profile)[:, None]
        return np.vstack([skin, inner])

    def signed_volume(self, gender: float) -> float:
        """Positive when the triangles wind outward, by the divergence theorem."""
        p = self.positions(gender)
        a, b, c = p[self.faces[:, 0]], p[self.faces[:, 1]], p[self.faces[:, 2]]
        return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)

    def volume(self, gender: float) -> float:
        """Enclosed volume, in body units cubed."""
        return abs(self.signed_volume(gender))


def _boundary_edges(faces: NDArray) -> NDArray:
    """Edges of a patch that belong to only one of its triangles."""
    pairs = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    keyed = np.sort(pairs, axis=1)
    uniq, inverse, counts = np.unique(keyed, axis=0, return_inverse=True,
                                      return_counts=True)
    once = counts[inverse] == 1
    return pairs[once]


def _components(selected: NDArray, faces: NDArray, n: int) -> NDArray:
    """Drop specks: keep only patches of at least ``MIN_PATCH`` vertices."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    keep = selected[edges[:, 0]] & selected[edges[:, 1]]
    edges = edges[keep]
    if not len(edges):
        return selected
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                       shape=(n, n))
    count, label = connected_components(graph, directed=False)
    sizes = np.bincount(label[selected], minlength=count)
    big = np.isin(label, np.flatnonzero(sizes >= MIN_PATCH))
    return selected & big


def _nipples(points: NDArray, outward: NDArray) -> list[NDArray]:
    """The most forward point of each side of the chest."""
    found = []
    for sign in (1.0, -1.0):
        band = ((points[:, 2] > CHEST_Z[0]) & (points[:, 2] < CHEST_Z[1])
                & (points[:, 0] * sign > NIPPLE_X[0])
                & (points[:, 0] * sign < NIPPLE_X[1])
                & (outward[:, 1] < -0.3))
        if band.sum() < 8:
            continue
        idx = np.flatnonzero(band)
        found.append(points[idx[np.argmin(points[idx, 1])]])
    return found


def build(male: NDArray, female: NDArray, faces: NDArray,
          normals: NDArray) -> Optional[BreastTissue]:
    """The lens between the skin and the chest wall, or None if there is none."""
    m = np.asarray(male, dtype=np.float64).reshape(-1, 3)
    f = np.asarray(female, dtype=np.float64).reshape(-1, 3)
    tris = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    n = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    # This mesh winds inward, so its vertex normals point into the body.
    outward = -n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)

    nipples = _nipples(f, outward)
    if len(nipples) < 2:
        logger.info("No chest found to build breast tissue on")
        return None

    radius = np.min([np.linalg.norm(f - p, axis=1) for p in nipples], axis=0)
    selected = radius < BASE_RADIUS
    if selected.sum() < 2 * MIN_PATCH:
        return None
    selected = _components(selected, tris, len(m))

    patch = tris[selected[tris].all(axis=1)]
    if not len(patch):
        return None
    source = np.unique(patch)
    remap = np.full(len(m), -1, dtype=np.int64)
    remap[source] = np.arange(len(source))
    local = remap[patch]
    count = len(source)

    # A lens: full depth under the nipple, nothing at the base.
    t = np.clip(radius[source] / BASE_RADIUS, 0.0, 1.0)
    profile = 1.0 - t * t

    outer = local
    inner = local[:, ::-1] + count          # reversed: it faces the other way
    band = []
    for a, b in _boundary_edges(local):
        # A closed surface is consistently wound when each directed edge is
        # seen once, so a wall meeting the outer boundary edge a -> b must
        # itself run b -> a; the other way round leaves the band inside out.
        band.append((b, a, a + count))
        band.append((b, a + count, b + count))
    shell = (np.vstack([outer, inner, np.asarray(band, dtype=np.int64)])
             if band else np.vstack([outer, inner]))

    tissue = BreastTissue(source=source, faces=shell,
                          skin_male=m[source], skin_female=f[source],
                          inward=outward[source] * -1.0, profile=profile)
    # The source mesh winds inward, so the patch inherits that and the shell
    # comes out inside-out.  A negative enclosed volume is the test for it.
    if tissue.signed_volume(1.0) < 0.0:
        tissue.faces = shell[:, ::-1].copy()
    logger.info("Breast tissue built: %d vertices, %d triangles, volume "
                "%.0f male and %.0f female", tissue.vertex_count, len(shell),
                tissue.volume(0.0), tissue.volume(1.0))
    return tissue
