"""Vectorised per-vertex operations shared by the skinning passes.

``transform_points`` / ``rotate_vectors`` are the gathered einsum the
skinning has always used, kept in one place so every pass computes a
joint's image of a vertex the same way.  Measured alternatives, 1 million
vertices, float64 (the numbers that decided this):

    joints used   gathered einsum   per-joint masked matmuls   one wide matmul + pick
        1              37 ms                12 ms                     33 ms
        2              35 ms                49 ms                     39 ms
        4              34 ms                56 ms                     38 ms
       10              36 ms                99 ms                     67 ms

The (V, 4, 4) gather looks wasteful (128 bytes per vertex) but numpy streams
it at memory bandwidth; grouping vertices by joint costs a mask pass and a
scattered write per joint and loses from two joints up.

``accumulate_rows`` replaces ``np.add.at`` (an unbuffered loop) with one
``np.bincount`` per column: the same sums, an order of magnitude faster
(0.92 s -> 0.31 s per frame for the face-normal accumulation of the muscles
that had a positional pass).
"""

from __future__ import annotations

import numpy as np


def transform_points(delta_stack: np.ndarray, joint_idx: np.ndarray,
                     points_h: np.ndarray) -> np.ndarray:
    """``(delta[joint_idx[v]] @ points_h[v])[:3]`` for every vertex; fresh ``(V, 3)`` float64."""
    if len(points_h) == 0:
        return np.empty((0, 3), dtype=np.float64)
    return np.einsum('vij,vj->vi', delta_stack[joint_idx], points_h)[:, :3]


def rotate_vectors(delta_stack: np.ndarray, joint_idx: np.ndarray,
                   vectors: np.ndarray) -> np.ndarray:
    """``delta[joint_idx[v]][:3, :3] @ vectors[v]`` for every vertex; fresh ``(V, 3)``."""
    if len(vectors) == 0:
        return np.empty((0, 3), dtype=np.float64)
    return np.einsum('vij,vj->vi', delta_stack[joint_idx, :3, :3], vectors)


def used_joints(joint_idx: np.ndarray, secondary_idx: np.ndarray | None,
                n_joints: int) -> np.ndarray:
    """Sorted joint indices that drive at least one vertex (bincount, O(V), no sort)."""
    counts = np.bincount(np.asarray(joint_idx).ravel(), minlength=n_joints)
    if secondary_idx is not None:
        counts = counts + np.bincount(np.asarray(secondary_idx).ravel(), minlength=n_joints)
    return np.flatnonzero(counts[:n_joints])


def accumulate_rows(index: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    """Sum the rows of ``values`` into ``n`` bins by ``index`` (``np.add.at`` semantics)."""
    index = np.asarray(index).ravel()
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        return np.bincount(index, weights=values, minlength=n)[:n]
    out = np.empty((n, values.shape[1]), dtype=np.float64)
    for c in range(values.shape[1]):
        out[:, c] = np.bincount(index, weights=values[:, c], minlength=n)[:n]
    return out
