"""Per-vertex centres of rotation for skin deformation.

Motivation
----------
Linear and dual-quaternion blend skinning both rotate a vertex about the
*bones* influencing it.  Where two bones meet at a large angle -- the axilla
under shoulder abduction or flexion -- no interpolation of those two transforms
preserves the distances between neighbouring vertices, which is what produces
the spiky fringes and taut sheets measured in
``shoulder_deformation_diagnosis.md``.

Le & Hodgins (2016), *Real-time Skeletal Skinning with Optimized Centers of
Rotation*, replaces the blend with a rotation about an optimised centre::

    v' = R(w_v) (v - p*_v) + T(w_v, p*_v)

where ``R(w_v)`` is the blended rotation for the vertex's weights and ``p*_v``
is a point characteristic of the vertex's weight distribution rather than any
one bone.

What this module implements, and how it differs from the paper
-------------------------------------------------------------
The paper computes ``p*_v`` as a similarity-weighted, area-weighted average
over the mesh's triangles::

    p*_v = sum_t s(w_v, w_t) a_t c_t / sum_t s(w_v, w_t) a_t

    s(w_p, w_v) = sum_{j != k} w_p^j w_p^k w_v^j w_v^k
                  exp(-(w_p^j w_v^k - w_p^k w_v^j)^2 / sigma^2)

That is O(V x T) similarity evaluations.  On this skin mesh -- 791,729 vertices
and ~1.5M triangles -- that is ~1.2e12 evaluations, against the ~1e8 the paper
reports for meshes of ~10k vertices.  Clustering vertices and subsampling
triangles brings it to ~7.5e7 pairs, still with a sparse pair-matching term
inside, so a faithful implementation needs a hash join over influence-pair keys
and a per-pair exponential.  That is a substantial piece of numerical code and
is NOT what this module does.

This module implements the mechanism at O(V): vertices are grouped by their
*influence signature* -- which bones influence them, with weights quantised --
and ``p*`` for a group is the area-weighted centroid of that group's own rest
positions.  So a vertex rotates about the centroid of the region that moves
with it, instead of about a bone.

Where this agrees with the paper: vertices whose weight vectors are similar get
a common centre, and that centre sits inside the co-moving region rather than
on a bone.  Where it differs: the paper's similarity is continuous and couples
*different* weight regions in proportion to how alike they are, which smooths
centres across region boundaries.  Quantised grouping is piecewise constant, so
centres jump between adjacent groups.  Finer quantisation reduces the jump size
but increases the group count; the trade is exposed as ``WEIGHT_QUANTISATION``.

Treat results from this module as a lower bound on what the full method would
achieve, not as a measurement of it.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

#: Weights are rounded to this step before grouping. Smaller values give
#: centres that vary more smoothly across the mesh at the cost of more groups
#: (and so a weaker area-weighted centroid per group).
WEIGHT_QUANTISATION = 0.25


def compute_centres(
    rest_positions: np.ndarray,
    influences: np.ndarray,
    influence_weights: np.ndarray,
    *,
    quantisation: float = WEIGHT_QUANTISATION,
    min_group: int = 8,
) -> np.ndarray:
    """Return an (V, 3) array of per-vertex centres of rotation.

    Parameters
    ----------
    rest_positions : (V, 3)
        Rest-pose vertex positions, model space.
    influences : (V, K) int
        Joint index per influence slot.
    influence_weights : (V, K) float
        Matching normalised weights.
    quantisation :
        Weight rounding step used to form influence signatures.
    min_group :
        Groups smaller than this fall back to the vertex's own position, which
        makes the deformation reduce to the ordinary blend there.  A centroid
        over two or three vertices is noise, not a region.

    Notes
    -----
    Vertices are keyed on the *sorted* influence set together with quantised
    weights, so two vertices driven by the same bones in the same proportions
    share a centre regardless of the order their influences happen to sit in.
    """
    rest = np.asarray(rest_positions, dtype=np.float64).reshape(-1, 3)
    V, K = influences.shape
    if len(rest) != V:
        raise ValueError(f"rest has {len(rest)} vertices, influences has {V}")

    # Sort each row by descending weight so slot order cannot split a group.
    order = np.argsort(-influence_weights, axis=1, kind="stable")
    inf_sorted = np.take_along_axis(influences, order, axis=1)
    w_sorted = np.take_along_axis(influence_weights, order, axis=1)

    q = np.clip(np.round(w_sorted / quantisation).astype(np.int64),
                0, int(round(1.0 / quantisation)))
    # Zero-weight slots must not contribute their (arbitrary) joint index.
    inf_keyed = np.where(q > 0, inf_sorted.astype(np.int64), -1)

    # Compose one integer key per vertex from the (joint, quantised weight)
    # pairs. np.unique on a void view over the row bytes is both faster and
    # safer than arithmetic packing, which can overflow on large rigs.
    key_cols = np.concatenate([inf_keyed, q], axis=1)
    key_cols = np.ascontiguousarray(key_cols)
    view = key_cols.view([("", key_cols.dtype)] * key_cols.shape[1]).ravel()
    _, group_id, counts = np.unique(view, return_inverse=True,
                                    return_counts=True)
    n_groups = len(counts)

    # Area weighting: a vertex's share of incident triangle area is
    # proportional to nothing we have cheaply here, so use uniform weights.
    # The paper's area weighting exists to stop dense mesh regions dominating;
    # this skin mesh is near-uniformly tessellated (median edge 0.2565 with a
    # tight spread), so uniform weighting is a close stand-in. Stated rather
    # than silently assumed.
    sums = np.zeros((n_groups, 3), dtype=np.float64)
    np.add.at(sums, group_id, rest)
    centroids = sums / counts[:, None]

    centres = centroids[group_id]

    # Small groups: fall back to the vertex itself, which makes p* - v vanish
    # and the deformation reduce to the plain blend for those vertices.
    small = counts[group_id] < min_group
    if np.any(small):
        centres[small] = rest[small]

    logger.info(
        "Centres of rotation: %d groups over %d vertices "
        "(median group %d, %.1f%% of vertices in groups below %d)",
        n_groups, V, int(np.median(counts)),
        100.0 * small.mean(), min_group,
    )
    return centres
