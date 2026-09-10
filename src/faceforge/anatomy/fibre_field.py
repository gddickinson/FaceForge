"""Harmonic fibre interpolation for a muscle with two authored footprints.

Why this exists
---------------
A muscle whose belly lies on no bone -- latissimus dorsi, the pectorals,
teres major, the deltoid -- was skinned by blending the RIGID images of its
two attachment joints.  A belly vertex 40 units below the shoulder has a
humerus image that swings through a 40-unit arc when the arm elevates, and a
half-weight blend of that arc is exactly what the user saw: the lats bowing
away from the trunk as a loop at the dead hang of a pull-up, the pectorals
peaking above the clavicle.  Every later pass (pinning, stretch clamp, hull
bound) then fought the arc and tore the ends instead.

What a muscle does is simpler.  Each fibre runs from origin to insertion and
straightens between wherever those two ends are NOW.  That is harmonic
interpolation of the attachment displacements across the mesh: the footprints
move rigidly with their bones (a Dirichlet boundary) and the belly solves
Laplace's equation between them, so it translates and stretches along its own
length and never bows.

Because each footprint moves rigidly, ``T(s) = R s + t``, the boundary data is
an affine function of rest position and its harmonic extension is linear in
the six numbers that change per frame.  Eight scalar harmonic solves at bind
time -- the extensions of {1, x, y, z} restricted to each footprint -- reduce
every frame to two small matrix products::

    h(p) = sum over footprints f of  (R_f - I) m_f(p) + t_f c_f(p)

where ``c_f`` is the harmonic weight of footprint f at p (1 on f, 0 on the
other footprint) and ``m_f`` the harmonic extension of f's rest coordinates.
On a footprint vertex ``c_f = 1`` and ``m_f = s``, so ``h = T(s) - s``: the
ends land on their bones by construction.  If neither joint moved, both deltas
are the identity and ``h = 0``: the containment invariant holds by
construction too.

Mesh components that touch no footprint are left to the skinning solver; a
harmonic solve on them would be singular and their motion is not determined
by these two attachments.
"""

from __future__ import annotations

import hashlib
import logging
import os
import zipfile
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import factorized

logger = logging.getLogger(__name__)

_CACHE_VERSION = 1

#: Minimum geodesic separation between the two footprints, as a fraction of
#: the muscle's geodesic length.  See :func:`trim_footprints`.
GAP_FRACTION = 0.25
#: The share of each footprint nearest the OTHER footprint that is dropped,
#: and the share that is always kept.  Proximity-seeded sets are far larger
#: than real attachments (a deltoid "origin" of 35% of its vertices, because
#: the deep surface touches the scapular neck; a teres major "origin" of
#: 57%): the far end of each set is the attachment, the near end is contact.
#: Measured on the field alone at the pull-up dead hang, dropping 30% took
#: latissimus dorsi p99 4.96x -> 3.64x and teres minor 5.96x -> 5.34x; 50%
#: and 70% cut further but left specks that tore (deltoid acromial max 17x
#: -> 102x).  An ABSOLUTE gap alone emptied the insertion of teres major and
#: the abdominal pectoralis, whose seeds overlap, so each set always keeps
#: its farthest ``KEEP_MIN`` share.
DROP_QUANTILE = 0.3
KEEP_MIN = 0.3


def trim_footprints(g_origin: np.ndarray, g_insertion: np.ndarray,
                    origin_idx: np.ndarray, insertion_idx: np.ndarray,
                    fraction: float = GAP_FRACTION,
                    drop: float = DROP_QUANTILE,
                    keep_min: float = KEEP_MIN) -> tuple[np.ndarray, np.ndarray]:
    """Keep the far end of each footprint; drop its members near the other.

    Footprints seeded by bone proximity touch or overlap wherever a muscle
    wraps its own joint: the deltoid's deep surface lies on the humeral head
    right beside its acromial origin, teres major's two ends meet in the
    axilla.  A harmonic field between ADJACENT boundary vertices that move
    with bones 110 degrees apart has no interior to interpolate across and
    tears along the seam (measured at the pull-up dead hang: teres major
    stretch p99 10.9x, max 55x).

    Per set, a member survives when its geodesic distance to the other set
    is at least ``max(min(fraction * L, q[1 - keep_min]), q[drop])`` where
    ``L`` is the farthest any vertex lies from either set and ``q`` the
    quantiles of the set's own distances: an absolute gap of a quarter of
    the muscle, but never less than the set's farthest ``keep_min`` share,
    and never more than its nearest ``drop`` share removed.  Vertices in
    both sets belong to neither.  Members the other set cannot reach
    (separate mesh components) are kept; there is nothing to separate them
    from.

    ``g_origin`` / ``g_insertion`` are geodesic distances of every vertex to
    the sets (``scipy.sparse.csgraph.dijkstra`` with ``min_only=True``).
    """
    o = np.unique(np.asarray(origin_idx, dtype=np.int64))
    i = np.unique(np.asarray(insertion_idx, dtype=np.int64))
    shared = np.intersect1d(o, i)
    if len(shared):
        o, i = np.setdiff1d(o, shared), np.setdiff1d(i, shared)
    finite = np.isfinite(g_origin) & np.isfinite(g_insertion)
    if not finite.any() or not len(o) or not len(i):
        return o, i
    length = max(float(g_origin[finite].max()), float(g_insertion[finite].max()))

    def survivors(members: np.ndarray, to_other: np.ndarray) -> np.ndarray:
        d = to_other[members]
        fin = np.isfinite(d)
        if not fin.any():
            return members
        floor = min(fraction * length, float(np.quantile(d[fin], 1.0 - keep_min)))
        if drop > 0.0:
            floor = max(floor, float(np.quantile(d[fin], drop)))
        return members[~fin | (d >= floor)]

    return survivors(o, g_insertion), survivors(i, g_origin)


def _prune_specks(adj: csr_matrix, idx: np.ndarray, min_neighbours: int = 1) -> np.ndarray:
    """Drop set members with no neighbour in the set (fewer than ``min_neighbours``).

    A lone Dirichlet vertex is a point load on the Laplacian: the field
    around it is a spike, and its edges tear (subscapularis max 9x -> 93x
    from one such speck after trimming).
    """
    if not len(idx):
        return idx
    member = np.zeros(adj.shape[0], dtype=bool)
    member[idx] = True
    inside = np.asarray(adj[idx] @ member.astype(np.float64)).ravel()
    return idx[inside >= min_neighbours]


@dataclass
class FibreField:
    """Bind-time harmonic data for one muscle; see the module docstring."""

    #: Vertex indices the field drives: both footprints plus every interior
    #: vertex in a mesh component that touches a footprint.
    solved: np.ndarray
    #: (S, 2) harmonic weights of the origin and insertion footprints.
    c: np.ndarray
    #: (S, 2, 3) harmonic extension of each footprint's rest coordinates.
    m: np.ndarray
    #: (S, 3) rest positions of the solved vertices.
    rest: np.ndarray

    def displacement(self, delta_origin: np.ndarray,
                     delta_insertion: np.ndarray) -> np.ndarray:
        """(S, 3) displacement from rest for the two joints' 4x4 deltas."""
        h = np.zeros((len(self.solved), 3), dtype=np.float64)
        for k, delta in enumerate((delta_origin, delta_insertion)):
            d = np.asarray(delta, dtype=np.float64)
            a = d[:3, :3] - np.eye(3)
            h += self.m[:, k, :] @ a.T + np.outer(self.c[:, k], d[:3, 3])
        return h

    def apply(self, positions: np.ndarray, delta_origin: np.ndarray,
              delta_insertion: np.ndarray) -> int:
        """Overwrite the solved vertices of ``positions`` ((V, 3), in place)."""
        h = self.displacement(delta_origin, delta_insertion)
        positions[self.solved] = (self.rest + h).astype(positions.dtype)
        return len(self.solved)


def _uniform_adjacency(n: int, edges: np.ndarray) -> csr_matrix:
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    e = e[(e[:, 0] < n) & (e[:, 1] < n) & (e[:, 0] != e[:, 1])]
    rows = np.concatenate([e[:, 0], e[:, 1]])
    cols = np.concatenate([e[:, 1], e[:, 0]])
    adj = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    adj.data[:] = 1.0          # duplicate edges must not weight the Laplacian
    return adj


def build_fibre_field(rest: np.ndarray, edges: np.ndarray,
                      origin_idx: np.ndarray,
                      insertion_idx: np.ndarray) -> FibreField | None:
    """Solve the eight harmonic functions for a mesh with two footprints.

    Returns None when the footprints are empty, overlap completely, or reach
    no interior vertex -- there is then nothing to interpolate.
    """
    rest = np.asarray(rest, dtype=np.float64).reshape(-1, 3)
    n = len(rest)
    o = np.unique(np.asarray(origin_idx, dtype=np.int64))
    i = np.unique(np.asarray(insertion_idx, dtype=np.int64))
    o = o[(o >= 0) & (o < n)]
    i = i[(i >= 0) & (i < n)]
    # A vertex authored into both sets belongs to neither boundary condition.
    shared = np.intersect1d(o, i)
    if len(shared):
        o = np.setdiff1d(o, shared)
        i = np.setdiff1d(i, shared)
    if not len(o) or not len(i):
        return None

    adj = _uniform_adjacency(n, edges)
    o = _prune_specks(adj, o)
    i = _prune_specks(adj, i)
    if not len(o) or not len(i):
        return None
    boundary = np.zeros(n, dtype=bool)
    boundary[o] = True
    boundary[i] = True
    ncomp, labels = connected_components(adj, directed=False)
    touched = np.zeros(ncomp, dtype=bool)
    touched[labels[boundary]] = True
    keep = touched[labels]
    interior = keep & ~boundary
    b_idx = np.where(boundary)[0]
    i_idx = np.where(interior)[0]

    # Boundary data: [c_o, m_o(3), c_i, m_i(3)].
    values = np.zeros((n, 8), dtype=np.float64)
    values[o, 0] = 1.0
    values[o, 1:4] = rest[o]
    values[i, 4] = 1.0
    values[i, 5:8] = rest[i]

    if len(i_idx):
        lap = (diags(np.asarray(adj.sum(axis=1)).ravel()) - adj).tocsr()
        l_ii = lap[i_idx][:, i_idx].tocsc()
        l_ib = lap[i_idx][:, b_idx]
        rhs = -(l_ib @ values[b_idx])
        try:
            solve = factorized(l_ii)
        except (RuntimeError, ValueError) as exc:   # singular / out of memory
            logger.warning("Fibre field: factorisation failed (%s)", exc)
            return None
        for k in range(8):
            values[i_idx, k] = solve(np.ascontiguousarray(rhs[:, k]))

    solved = np.where(keep)[0]
    v = values[solved]
    c = np.clip(v[:, [0, 4]], 0.0, 1.0)
    m = np.stack([v[:, 1:4], v[:, 5:8]], axis=1)
    return FibreField(solved=solved, c=c, m=m, rest=rest[solved].copy())


# ── Disk cache: the eight solves cost seconds per muscle at every load ───────

def _key(rest: np.ndarray, edges: np.ndarray, o: np.ndarray, i: np.ndarray) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(f"fibre v{_CACHE_VERSION}".encode())
    for arr in (rest, edges, o, i):
        a = np.ascontiguousarray(np.asarray(arr))
        h.update(f"|{a.shape}{a.dtype.str}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def cached_fibre_field(rest: np.ndarray, edges: np.ndarray,
                       origin_idx: np.ndarray,
                       insertion_idx: np.ndarray) -> FibreField | None:
    """``build_fibre_field`` through the skinning disk cache when enabled."""
    from faceforge.body import skinning_cache

    if not skinning_cache.enabled():
        return build_fibre_field(rest, edges, origin_idx, insertion_idx)
    key = _key(rest, edges, origin_idx, insertion_idx)
    path = skinning_cache.cache_dir() / f"fibre.{key}.npz"
    try:
        with np.load(path) as z:
            return FibreField(solved=z["solved"], c=z["c"], m=z["m"], rest=z["rest"])
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        pass
    field = build_fibre_field(rest, edges, origin_idx, insertion_idx)
    if field is None:
        return None
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(tmp, solved=field.solved, c=field.c, m=field.m, rest=field.rest)
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
    return field
