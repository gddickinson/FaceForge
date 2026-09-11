"""Carry the sex difference in soft tissue onto the model's own skin.

The skeleton morph changes the bones and the skinning carries the skin with
them, but bone is only half of what differs between the sexes on the surface.
The other half is where the soft tissue sits: breast, gluteal and thigh fat,
and a waist that is narrower relative to the hip.  None of that follows from a
scaled skeleton, and without it the morph reads as a change of size rather
than a change of sex.

The difference is not invented here.  The application already loads a matched
male/female pair of body-surface meshes, and once both are warped onto the
skeleton the vector between them is a measured sex difference with a shared
topology.  Most of that vector is the stature difference, which the skeleton
already produces; applying it again would shrink the body twice.  So two
things are removed from it:

* the **uniform size change** -- the female mesh is rescaled to the male's
  stature before the difference is taken, leaving shape only;
* the **tangential component** -- only the part of the residual that runs
  radially out from the body's own vertical axis is kept.  Radial displacement
  is exactly "more or less tissue here", which is what a fat distribution is,
  and it cannot change a limb's length or a body's height.

Measured on the shipped meshes, the residual that survives is 0.8 units over
the head (there is essentially no soft-tissue sex difference there, which is
the right answer and a useful check) against 4.2 at the chest, 5.2 at the
waist and 4.5 over the hip and buttock.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

import numpy as np
from numpy.typing import NDArray

from faceforge.body.edge_relaxation import enforce_edge_range

logger = logging.getLogger(__name__)

#: Neighbours used to interpolate the field onto a target mesh.
FIELD_K = 24

#: Width of the interpolation kernel, in body units.  The source mesh's own
#: vertex spacing is about 1, so a kernel several times that averages over a
#: patch rather than latching onto whichever single source vertex happens to be
#: nearest -- which is what made the field jump.
FIELD_SIGMA = 3.0

#: The field is trusted in full within ``FIELD_TRUST_NEAR`` of the source
#: surface and fades to nothing by ``FIELD_TRUST_FAR``.  The two surfaces are
#: different bodies: the model's skin is a median 2.8 units from the surface
#: the field was measured on, 9.3 at the 95th percentile and 16.4 at worst.
#: A hard cut-off there is what produced the visible shards -- 14606 vertices
#: fell outside it and received nothing while their neighbours received the
#: full field, so the mesh tore along the boundary.
FIELD_TRUST_NEAR = 6.0
FIELD_TRUST_FAR = 14.0

#: Laplacian passes over the *target* mesh's own edges.  The source surface and
#: the target surface are different bodies, so near a boundary -- the armpit
#: above all -- neighbouring target vertices can draw on source vertices from
#: different limbs and receive very different vectors.  Measured before
#: smoothing: the field jumped by up to 8.0 units across a skin edge whose
#: median length is 0.26, which tore 15372 edges to under half their length.
FIELD_SMOOTH_ITER = 6
FIELD_SMOOTH_STRENGTH = 0.5

#: Laplacian passes over the *source* mesh before the field is interpolated.
#: This is where the heavy smoothing belongs: the source has 10.5 k vertices
#: about 1 unit apart, so 60 passes reach several units, while the same reach
#: on the 790 k-vertex skin (0.26-unit edges) would need hundreds.  A fat
#: distribution is smooth by nature, so nothing real is lost.
SOURCE_SMOOTH_ITER = 60

#: The field is finally held inside an edge-length band on the target mesh, so
#: that however the two surfaces disagree the skin cannot be torn.  Smoothing
#: alone got the worst edges from 0.006 to 0.019 of their rest length; the
#: constraint is what makes "no tearing" a property rather than a hope.
FIELD_MAX_STRETCH = 0.25
FIELD_MAX_COMPRESSION = 0.25
FIELD_CONSTRAINT_SWEEPS = 20


def radial_component(points: NDArray, delta: NDArray,
                     axis_xy: tuple[float, float] | None = None) -> NDArray:
    """Keep only the part of ``delta`` pointing away from the body's long axis.

    The body is star-shaped about a vertical axis, so the outward direction at
    a vertex is well defined without needing a surface normal -- which matters
    here because the triangle winding of both surface meshes is inconsistent
    and their normals cannot be trusted.
    """
    pts = np.asarray(points, dtype=np.float64)
    d = np.asarray(delta, dtype=np.float64)
    if axis_xy is None:
        axis_xy = (float(np.median(pts[:, 0])), float(np.median(pts[:, 1])))
    out = pts[:, :2] - np.asarray(axis_xy, dtype=np.float64)
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    unit = np.divide(out, np.maximum(norm, 1e-9))
    radial = np.einsum("ij,ij->i", d[:, :2], unit)
    result = np.zeros_like(d)
    result[:, :2] = unit * radial[:, None]
    return result


class SkinShapeMorph:
    """A measured soft-tissue sex field, interpolated onto any mesh.

    Parameters
    ----------
    source_points, source_delta :
        The male surface and the shape-only displacement to the female one,
        both ``(N, 3)`` in body coordinates.
    """

    def __init__(self, source_points: NDArray, source_delta: NDArray) -> None:
        self._pts = np.asarray(source_points, dtype=np.float64)
        self._delta = np.asarray(source_delta, dtype=np.float64)
        self._tree = None
        self._rest: dict[int, NDArray[np.float32]] = {}
        self._field: dict[int, NDArray[np.float64]] = {}

    @classmethod
    def from_pair(cls, male: NDArray, female: NDArray,
                  indices: Optional[NDArray] = None) -> "SkinShapeMorph":
        """Build the field from a matched male/female surface pair.

        ``indices`` are the pair's shared triangles; given them, the field is
        smoothed on the source before it is ever interpolated.
        """
        m = np.asarray(male, dtype=np.float64)
        f = np.asarray(female, dtype=np.float64)
        hm = float(m[:, 2].max() - m[:, 2].min())
        hf = float(f[:, 2].max() - f[:, 2].min())
        scaled = f.copy()
        if hf > 1e-6:
            scaled[:, 2] -= f[:, 2].min()
            scaled *= hm / hf
            scaled[:, 2] += m[:, 2].min()
        for ax in (0, 1):
            scaled[:, ax] -= float(np.median(scaled[:, ax]) - np.median(m[:, ax]))
        delta = radial_component(m, scaled - m)
        delta = cls.smooth_on_mesh(delta, indices, len(m),
                                   iterations=SOURCE_SMOOTH_ITER)
        return cls(m, delta)

    # -- field ---------------------------------------------------------------

    def delta_for(self, points: NDArray) -> NDArray:
        """The field sampled at arbitrary points, by inverse-distance weighting."""
        try:
            from scipy.spatial import cKDTree
        except ImportError:                                   # pragma: no cover
            return np.zeros((len(points), 3))
        if self._tree is None:
            self._tree = cKDTree(self._pts)
        k = min(FIELD_K, len(self._pts))
        dist, idx = self._tree.query(np.asarray(points, dtype=np.float64), k=k)
        if dist.ndim == 1:
            dist = dist[:, None]
            idx = idx[:, None]
        w = np.exp(-(dist / FIELD_SIGMA) ** 2)
        total = w.sum(axis=1, keepdims=True)
        w = np.divide(w, np.maximum(total, 1e-12))
        out = np.einsum("qk,qkj->qj", w, self._delta[idx])
        # Confidence from the nearest source vertex, faded rather than cut.
        nearest = dist.min(axis=1)
        span = max(FIELD_TRUST_FAR - FIELD_TRUST_NEAR, 1e-9)
        trust = np.clip((FIELD_TRUST_FAR - nearest) / span, 0.0, 1.0)
        # Smoothstep, so the weight's own derivative is continuous too.
        trust = trust * trust * (3.0 - 2.0 * trust)
        return out * trust[:, None]

    @staticmethod
    def smooth_on_mesh(field: NDArray, indices: Optional[NDArray], n_vertices: int,
                       iterations: int = FIELD_SMOOTH_ITER,
                       strength: float = FIELD_SMOOTH_STRENGTH) -> NDArray:
        """Laplacian-smooth a displacement field over a mesh's own edges."""
        if indices is None or iterations <= 0:
            return field
        tris = np.asarray(indices).reshape(-1, 3)
        e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
        src = np.concatenate([e[:, 0], e[:, 1]])
        dst = np.concatenate([e[:, 1], e[:, 0]])
        counts = np.bincount(src, minlength=n_vertices).astype(np.float64)
        has = counts > 0
        inv = np.zeros(n_vertices, dtype=np.float64)
        inv[has] = 1.0 / counts[has]
        out = np.array(field, dtype=np.float64, copy=True)
        for _ in range(iterations):
            avg = np.empty_like(out)
            for c in range(3):
                avg[:, c] = np.bincount(src, weights=out[dst, c], minlength=n_vertices) * inv
            out[has] = (1.0 - strength) * out[has] + strength * avg[has]
        return out

    @staticmethod
    def constrain(field: NDArray, rest: NDArray, indices: Optional[NDArray]) -> NDArray:
        """Hold the displaced mesh inside an edge-length band around its rest pose."""
        if indices is None:
            return field
        tris = np.asarray(indices).reshape(-1, 3)
        edges = np.unique(np.sort(np.concatenate(
            [tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1), axis=0)
        rest_len = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
        keep = rest_len > 1e-6
        edges, rest_len = edges[keep], rest_len[keep]
        pos = rest + field
        report = enforce_edge_range(
            pos, edges, rest_len,
            max_stretch=FIELD_MAX_STRETCH, max_compression=FIELD_MAX_COMPRESSION,
            iterations=FIELD_CONSTRAINT_SWEEPS)
        logger.info("Skin field constrained: %s in %d sweeps",
                    "converged" if report["converged"] else "residual remains",
                    report["iterations_run"])
        return pos - rest

    # -- application ---------------------------------------------------------

    def apply(self, bindings: Iterable[Any], gender: float,
              name_filter: Optional[str] = "skin") -> int:
        """Displace the rest pose of every matching mesh by ``gender`` x the field."""
        n = 0
        for binding in bindings:
            mesh = getattr(binding, "mesh", None)
            if mesh is None or getattr(binding, "is_muscle", False):
                continue
            if name_filter and name_filter not in (mesh.name or "").lower():
                continue
            if self._apply_one(binding, mesh, gender):
                n += 1
        return n

    def _apply_one(self, binding: Any, mesh: Any, gender: float) -> bool:
        if mesh.rest_positions is None:
            return False
        key = id(mesh)
        rest = self._rest.get(key)
        if rest is None:
            rest = self._rest[key] = np.asarray(mesh.rest_positions, dtype=np.float32).copy()
        field = self._field.get(key)
        if field is None:
            pts = rest.reshape(-1, 3).astype(np.float64)
            field = self.delta_for(pts)
            field = self.smooth_on_mesh(field, mesh.geometry.indices, len(pts))
            field = self.constrain(field, pts, mesh.geometry.indices)
            self._field[key] = field
            logger.info("Skin sex field on %s: median %.2f, max %.2f over %d vertices",
                        mesh.name, float(np.median(np.linalg.norm(field, axis=1))),
                        float(np.linalg.norm(field, axis=1).max()), len(field))
        g = float(max(0.0, min(1.0, gender)))
        out = (rest.reshape(-1, 3).astype(np.float64) + field * g)
        mesh.rest_positions = out.reshape(-1).astype(np.float32)
        binding._rest_f64 = None
        binding._pos_h = None
        mesh.needs_update = True
        return True
