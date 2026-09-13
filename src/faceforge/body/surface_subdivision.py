"""Loop subdivision for the body-surface mesh.

The male and female surfaces are MakeHuman base meshes: 10,582 vertices and
21,160 triangles for a whole body, with a median edge of 1.13 units but a 95th
percentile of 3.75 and a worst of 7.72.  That is the ceiling on two things.
The figure is visibly faceted where it is drawn at partial opacity, and the
female-minus-male soft-tissue field (:mod:`faceforge.body.skin_morph`) is
measured on those same vertices, so it is only as detailed as they are.

Loop subdivision is the right refinement here rather than a midpoint split: a
midpoint split quadruples the triangle count and changes nothing, because the
surface it describes is the same faceted one.  Loop converges to a smooth
limit surface, so the silhouette improves as well as the density.

The mesh is closed and genus 0 -- V - E + F = 10582 - 31740 + 21160 = 2 -- so
there are no boundary rules to get wrong.  Both sexes share a topology and are
subdivided by the same operator, so they keep it: the morph between them is
still a vertex-for-vertex lerp.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def _edges(faces: NDArray) -> tuple[NDArray, NDArray]:
    """Unique undirected edges, and each face's three edge indices."""
    pairs = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    keyed = np.sort(pairs, axis=1)
    uniq, inverse = np.unique(keyed, axis=0, return_inverse=True)
    return uniq, inverse.reshape(3, len(faces)).T


def _beta(valence: NDArray) -> NDArray:
    """Loop's weight for an original vertex's neighbours, Warren's variant."""
    n = np.maximum(valence, 3).astype(np.float64)
    inner = 0.375 + 0.25 * np.cos(2.0 * np.pi / n)
    return (0.625 - inner * inner) / n


def subdivide(positions: NDArray, faces: NDArray
              ) -> tuple[NDArray, NDArray]:
    """One Loop step.  Returns ``(positions, faces)`` for the refined mesh."""
    from scipy.sparse import coo_matrix

    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    v = len(pos)
    edges, face_edges = _edges(f)
    e = len(edges)

    # Odd vertices: 3/8 of the edge's own ends, 1/8 of the two opposite
    # corners.  Each interior edge is opposite exactly two corners; summing
    # over faces finds them without building an adjacency structure.
    odd = 0.375 * (pos[edges[:, 0]] + pos[edges[:, 1]])
    opposite = np.zeros((e, 3))
    for corner in range(3):
        # face_edges[:, k] is the edge (k, k+1); the corner opposite it is k+2.
        np.add.at(opposite, face_edges[:, corner], pos[f[:, (corner + 2) % 3]])
    odd = odd + 0.125 * opposite

    # Even vertices: pulled toward the mean of their neighbours.
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    adjacency = coo_matrix((np.ones(len(rows)), (rows, cols)),
                           shape=(v, v)).tocsr()
    valence = np.asarray(adjacency.sum(axis=1)).ravel()
    neighbour_sum = adjacency @ pos
    b = _beta(valence)[:, None]
    even = (1.0 - valence[:, None] * b) * pos + b * neighbour_sum

    new_pos = np.vstack([even, odd])
    m = v + face_edges                                   # odd vertex per edge
    a, bb, c = f[:, 0], f[:, 1], f[:, 2]
    m0, m1, m2 = m[:, 0], m[:, 1], m[:, 2]               # (a,b) (b,c) (c,a)
    new_faces = np.vstack([
        np.stack([a, m0, m2], axis=1),
        np.stack([m0, bb, m1], axis=1),
        np.stack([m2, m1, c], axis=1),
        np.stack([m0, m1, m2], axis=1),
    ])
    return new_pos, new_faces


def subdivide_pair(male: NDArray, female: NDArray, faces: NDArray,
                   levels: int = 1) -> tuple[NDArray, NDArray, NDArray]:
    """Subdivide both surfaces with one operator, so they keep one topology."""
    m = np.asarray(male, dtype=np.float64).reshape(-1, 3)
    fm = np.asarray(female, dtype=np.float64).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    for _ in range(max(0, int(levels))):
        m, refined = subdivide(m, f)
        fm, _ = subdivide(fm, f)
        f = refined
    return m, fm, f
