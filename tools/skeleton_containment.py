"""How far the skeleton sticks out of the body-surface mesh.

One number decides whether a fit helped: the signed distance from a bone
vertex to the surface, positive when the vertex is outside it.  The sign is
not taken from the mesh's winding -- the MakeHuman surface winds inward, and
assuming otherwise silently inverts every measurement -- but calibrated once
against a point known to be inside the body.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

from faceforge.body.fit_regions import ROOT_REGION, SKIP_SUBTREES, region_of

#: Vertices sampled per bone mesh.  The skeleton has ~200 of them, so this is
#: a cloud of order 10^5 -- enough that a percentile is stable, small enough
#: that a solve can project it thousands of times.
PER_BONE = 400


def bone_points(root: Any, per_bone: int = PER_BONE, seed: int = 0,
                exclude: set[int] | None = None
                ) -> tuple[NDArray, NDArray, NDArray]:
    """Sampled skeleton vertices in body coordinates.

    Returns ``(points, region, name)``: the cloud, the fit region each point
    belongs to, and the node it came from.
    """
    rng = np.random.default_rng(seed)
    skip = exclude or set()
    pts: list[NDArray] = []
    regions: list[str] = []
    names: list[str] = []
    stack: list[tuple[Any, str, NDArray]] = [(root, ROOT_REGION, np.zeros(3))]
    while stack:
        node, region, offset = stack.pop()
        for child in node.children:
            name = getattr(child, "name", "") or ""
            if name in SKIP_SUBTREES:
                continue
            child_region = region_of(name, region)
            child_offset = offset + np.asarray(child.position, dtype=np.float64)
            mesh = getattr(child, "mesh", None)
            if mesh is not None and name and id(mesh) not in skip:
                geo = mesh.geometry
                p = np.asarray(geo.positions, dtype=np.float64).reshape(-1, 3)
                p = p[:geo.vertex_count]
                if len(p):
                    if len(p) > per_bone:
                        p = p[rng.choice(len(p), per_bone, replace=False)]
                    pts.append(p + child_offset)
                    regions += [child_region] * len(p)
                    names += [name] * len(p)
            stack.append((child, child_region, child_offset))
    if not pts:
        return np.zeros((0, 3)), np.zeros(0, dtype=object), np.zeros(0, dtype=object)
    return np.vstack(pts), np.array(regions), np.array(names)


class SurfaceDepth:
    """Signed distance to a closed surface, positive outside."""

    def __init__(self, positions: NDArray, triangles: NDArray,
                 inside_probe: NDArray | None = None) -> None:
        self.pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        self.tris = np.asarray(triangles).reshape(-1, 3)
        probe = (np.asarray(inside_probe, dtype=np.float64).reshape(1, 3)
                 if inside_probe is not None
                 else self.pos.mean(axis=0).reshape(1, 3))
        self._sign = 1.0
        # Calibrate: the probe is inside, so its depth must come out negative.
        self._sign = -1.0 if float(self(probe)[0]) > 0 else 1.0

    def __call__(self, points: NDArray) -> NDArray:
        from faceforge.body.surface_projection import closest_points_on_surface

        q = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(q) == 0:
            return np.zeros(0)
        cp, sq, ti = closest_points_on_surface(q, self.pos, self.tris, k=16)
        a = self.pos[self.tris[ti, 0]]
        b = self.pos[self.tris[ti, 1]]
        c = self.pos[self.tris[ti, 2]]
        n = np.cross(b - a, c - a)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        side = np.sign(np.einsum("ij,ij->i", q - cp, n))
        return self._sign * np.sqrt(sq) * side


def summarise(depth: NDArray) -> dict[str, float]:
    """The four numbers a fit is judged on."""
    if len(depth) == 0:
        return {"outside_pct": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "outside_pct": float(100.0 * (depth > 0).mean()),
        "median": float(np.median(depth)),
        "p95": float(np.percentile(depth, 95)),
        "max": float(depth.max()),
    }


def report(tag: str, depth: NDArray, groups: NDArray | None = None,
           top: int = 0) -> dict[str, float]:
    """Print the summary, optionally worst groups first.  Returns the summary."""
    s = summarise(depth)
    print(f"[{tag}] n={len(depth)}  outside {s['outside_pct']:.1f}%  "
          f"median {s['median']:+.2f}  p95 {s['p95']:.2f}  max {s['max']:.2f}")
    if groups is not None and top:
        rows = []
        for g in sorted(set(groups.tolist())):
            d = depth[groups == g]
            rows.append((float(np.median(d)), g, summarise(d), len(d)))
        for _, g, st, n in sorted(rows, reverse=True)[:top]:
            print(f"    {g:<14}{n:>7}  outside {st['outside_pct']:5.1f}%  "
                  f"median {st['median']:+6.2f}  p95 {st['p95']:6.2f}  "
                  f"max {st['max']:6.2f}")
    return s


def surface_of(gender_morph: Any) -> tuple[NDArray, NDArray]:
    """The body-surface mesh's positions and triangles, as it is now."""
    mesh = gender_morph.body_mesh
    geo = mesh.geometry
    pos = np.asarray(geo.positions, dtype=np.float64).reshape(-1, 3)[:geo.vertex_count]
    tris = np.asarray(geo.indices).reshape(-1, 3)
    return pos, tris


def iter_regions(order_first: Iterable[str] = ()) -> list[str]:
    """Region names, parents before children."""
    from faceforge.body.fit_regions import REGIONS

    names = [r.name for r in REGIONS]
    return [n for n in order_first if n in names] + [
        n for n in names if n not in set(order_first)]
