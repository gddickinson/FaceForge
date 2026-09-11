"""Turn a skeleton change into a smooth spatial warp for the soft tissue.

A change of proportion is not a pose.  Routed through the articulated skinning
it tears: the joints *translate* rather than rotate, two skin vertices either
side of a chain boundary follow different joints, and the cross-chain
divergence clamp -- which exists so that a moving arm does not drag the trunk
-- stops them being blended back together.  Measured on the model's skin at
gender 1, that put 27747 edges beyond twice their rest length and 6666 below
half, and the body came out visibly shredded at the waist, the hip and the
shoulders.

Interpolating the joint displacements over space instead gives a field that
varies on the scale of a limb rather than of a mesh edge, so it cannot tear;
and between two joints it reproduces exactly the linear stretch of the segment
between them, which is what a proportion change is.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np
from numpy.typing import NDArray

#: Ridge added to the spline system.  The control points include joints that
#: very nearly coincide (a wrist and the metacarpals hanging off it), which
#: makes the interpolation matrix ill-conditioned; a small ridge trades exact
#: interpolation at those points for a stable solve.
FIELD_SMOOTHING = 1e-3

#: Spacing of the lattice the warp is sampled on, in body units.  The field
#: varies on the scale of a limb, so sampling it every few units and
#: interpolating costs a fraction of evaluating the spline per vertex and
#: differs from it by far less than the mesh's own edge length.
LATTICE_SPACING = 2.0


def _walk(node: Any) -> Iterable[Any]:
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(n.children)


def rest_offset(node: Any, pivot_rest: dict[int, NDArray]) -> NDArray:
    """A node's captured (pre-morph) position in body coordinates."""
    out = np.zeros(3)
    n = node
    while n is not None and getattr(n, "name", "") != "bodyRoot":
        rest = pivot_rest.get(id(n))
        out = out + (rest if rest is not None
                     else np.asarray(n.position, dtype=np.float64))
        n = n.parent
    return out


def control_points(root: Any, pivot_rest: dict[int, NDArray],
                   bone_rest: dict[int, NDArray],
                   has_scale: Callable[[str], Any],
                   node_offset: Callable[[Any], NDArray],
                   is_pivot: Callable[[Any], bool]) -> tuple[NDArray, NDArray]:
    """Where the skeleton was and where it went, as ``(positions, displacements)``.

    Every joint contributes one control point, and so does every bone that
    hangs off a group rather than a joint -- the pelvis, the sternum, the
    cranium -- so no region of the body is left without one.
    """
    pos: list[NDArray] = []
    disp: list[NDArray] = []
    for node in _walk(root):
        if is_pivot(node):
            rest = pivot_rest.get(id(node))
            if rest is None:
                continue
            before = rest_offset(node.parent, pivot_rest) + rest
            after = node_offset(node)
            pos.append(before)
            disp.append(after - before)
            continue
        mesh = getattr(node, "mesh", None)
        if mesh is None or node.parent is None or is_pivot(node.parent):
            continue
        rest = bone_rest.get(id(mesh))
        if rest is None or has_scale(node.name or "") is None:
            continue
        before = (np.asarray(rest, dtype=np.float64).reshape(-1, 3).mean(axis=0)
                  + rest_offset(node, pivot_rest))
        g = mesh.geometry
        after = (np.asarray(g.positions, dtype=np.float64).reshape(-1, 3).mean(axis=0)
                 + node_offset(node))
        pos.append(before)
        disp.append(after - before)
    if not pos:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.asarray(pos, dtype=np.float64), np.asarray(disp, dtype=np.float64)


def displacement_warp(points: NDArray, displacements: NDArray,
                      smoothing: float | None = None) -> Callable[[NDArray], NDArray]:
    """A thin-plate spline through the joint displacements.

    A spline rather than a weighted average, because the field has to
    *reproduce* the skeleton change rather than blur it: a Gaussian average
    over control points 10 units apart returned barely half of a control
    point's own displacement at the control point itself, so the body would
    have moved about half as far as its bones.

    The thin-plate spline is the natural choice for the job.  It interpolates
    the data exactly, it is the smoothest interpolant in the bending-energy
    sense, and -- because it carries an explicit affine term -- it reproduces
    an affine change exactly: scale a skeleton uniformly and the soft tissue
    scales uniformly with it, which is the case that must be exactly right.
    """
    pts = np.asarray(points, dtype=np.float64)
    disp = np.asarray(displacements, dtype=np.float64)
    n = len(pts)
    if n == 0:
        return lambda q: np.zeros((len(np.asarray(q, dtype=np.float64).reshape(-1, 3)), 3))
    if n < 4:
        mean = disp.mean(axis=0)
        return lambda q: np.tile(mean, (len(np.asarray(q, dtype=np.float64).reshape(-1, 3)), 1))

    lam = FIELD_SMOOTHING if smoothing is None else float(smoothing)
    # phi(r) = r is the biharmonic kernel in three dimensions.
    K = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    K[np.diag_indices(n)] = 0.0
    K += lam * np.eye(n)
    P = np.concatenate([np.ones((n, 1)), pts], axis=1)          # (n, 4)
    A = np.zeros((n + 4, n + 4), dtype=np.float64)
    A[:n, :n] = K
    A[:n, n:] = P
    A[n:, :n] = P.T
    rhs = np.zeros((n + 4, 3), dtype=np.float64)
    rhs[:n] = disp
    try:
        sol = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:                              # pragma: no cover
        sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
    coeff, affine = sol[:n], sol[n:]

    def warp(query: NDArray) -> NDArray:
        q = np.asarray(query, dtype=np.float64).reshape(-1, 3)
        out = np.empty_like(q)
        # Chunked so a dense mesh never allocates an (M, n) matrix at once.
        step = max(1, int(4_000_000 // n))
        for lo in range(0, len(q), step):
            block = q[lo:lo + step]
            r = np.linalg.norm(block[:, None, :] - pts[None, :, :], axis=2)
            out[lo:lo + step] = (r @ coeff
                                 + affine[0]
                                 + block @ affine[1:])
        return out

    return warp


def sampled_warp(warp: Callable[[NDArray], NDArray], points: NDArray,
                 spacing: float | None = None,
                 margin: float = 12.0) -> Callable[[NDArray], NDArray]:
    """Sample a smooth warp on a lattice and interpolate it trilinearly.

    Evaluating the spline itself is O(vertices x control points): 6.2 million
    soft-tissue vertices against 171 joints took 25 seconds.  The field is
    smooth by construction, so sampling it on a lattice a few units across and
    interpolating is the same answer for a fraction of the work.  Points
    outside the lattice fall back to the exact warp, so nothing is clamped.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    step = float(LATTICE_SPACING if spacing is None else spacing)
    lo = pts.min(axis=0) - margin
    hi = pts.max(axis=0) + margin
    counts = np.maximum(np.ceil((hi - lo) / step).astype(int) + 1, 2)
    axes = [lo[a] + np.arange(counts[a]) * step for a in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    values = warp(grid).reshape(counts[0], counts[1], counts[2], 3)

    def interp(query: NDArray) -> NDArray:
        q = np.asarray(query, dtype=np.float64).reshape(-1, 3)
        f = (q - lo) / step
        i0 = np.floor(f).astype(np.int64)
        inside = np.all((i0 >= 0) & (i0 < np.asarray(counts) - 1), axis=1)
        out = np.empty_like(q)
        if not inside.all():
            outside = ~inside
            out[outside] = warp(q[outside])
        if not inside.any():
            return out
        idx = i0[inside]
        t = (f[inside] - idx)
        acc = np.zeros((idx.shape[0], 3), dtype=np.float64)
        for dx in (0, 1):
            wx = t[:, 0] if dx else 1.0 - t[:, 0]
            for dy in (0, 1):
                wy = t[:, 1] if dy else 1.0 - t[:, 1]
                for dz in (0, 1):
                    wz = t[:, 2] if dz else 1.0 - t[:, 2]
                    acc += (wx * wy * wz)[:, None] * values[
                        idx[:, 0] + dx, idx[:, 1] + dy, idx[:, 2] + dz]
        out[inside] = acc
        return out

    return interp
