"""The shared per-vertex skinning operations give the numbers the passes always computed."""

from __future__ import annotations

import numpy as np

from faceforge.body.skinning_ops import (
    accumulate_rows, rotate_vectors, transform_points, used_joints,
)
from faceforge.core.math_utils import mat4_compose, quat_from_axis_angle, vec3


def _deltas(n_joints: int, rng) -> np.ndarray:
    out = []
    for _ in range(n_joints):
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        q = quat_from_axis_angle(vec3(*axis), rng.uniform(-2, 2))
        out.append(mat4_compose(rng.normal(size=3) * 10, q, np.ones(3)))
    return np.stack(out)


def test_transform_and_rotate_are_the_per_vertex_joint_products():
    rng = np.random.default_rng(3)
    D = _deltas(7, rng)
    V = 2000
    ji = rng.choice([0, 2, 5, 6], size=V)
    pts = rng.normal(size=(V, 3)) * 20
    pos_h = np.concatenate([pts, np.ones((V, 1))], axis=1)
    got = transform_points(D, ji, pos_h)
    want = np.stack([(D[j] @ p)[:3] for j, p in zip(ji, pos_h)])
    np.testing.assert_allclose(got, want, atol=1e-9)
    assert got.dtype == np.float64 and got.flags.writeable
    vec = rng.normal(size=(V, 3))
    want_n = np.stack([D[j][:3, :3] @ v for j, v in zip(ji, vec)])
    np.testing.assert_allclose(rotate_vectors(D, ji, vec), want_n, atol=1e-9)
    assert transform_points(D, ji[:0], pos_h[:0]).shape == (0, 3)
    assert rotate_vectors(D, ji[:0], vec[:0]).shape == (0, 3)


def test_used_joints_is_the_sorted_union_of_primary_and_secondary():
    ji = np.array([5, 5, 2, 9, 2])
    si = np.array([5, 7, 2, 9, 0])
    np.testing.assert_array_equal(used_joints(ji, si, 12), [0, 2, 5, 7, 9])
    np.testing.assert_array_equal(used_joints(ji, None, 12), [2, 5, 9])


def test_accumulate_rows_matches_add_at():
    rng = np.random.default_rng(1)
    n = 300
    tri = rng.integers(0, n, size=(1200, 3))
    fn = rng.normal(size=(1200, 3))
    want = np.zeros((n, 3))
    for k in range(3):
        np.add.at(want, tri[:, k], fn)
    got = accumulate_rows(tri.ravel(order="F"), np.tile(fn, (3, 1)), n)
    np.testing.assert_allclose(got, want, atol=1e-12)
    got1 = accumulate_rows(np.array([0, 0, 2]), np.array([1.0, 2.0, 5.0]), 5)
    np.testing.assert_allclose(got1, [3.0, 0.0, 5.0, 0.0, 0.0])
