"""Tests for math_utils module."""

import numpy as np
import pytest

from faceforge.core.math_utils import (
    vec3, vec4, mat4_identity, mat4_translation, mat4_scale,
    mat4_rotation_x, mat4_rotation_y, mat4_rotation_z,
    mat4_from_quaternion, mat4_compose, mat4_perspective, mat4_look_at,
    mat4_inverse, mat3_normal,
    quat_identity, quat_from_euler, quat_from_axis_angle,
    quat_multiply, quat_conjugate, quat_normalize, quat_slerp,
    quat_rotate_vec3,
    normalize, lerp, lerp_vec3, clamp, deg_to_rad, rad_to_deg,
    transform_point, transform_direction,
)


def test_vec3():
    v = vec3(1, 2, 3)
    assert v.shape == (3,)
    np.testing.assert_array_equal(v, [1, 2, 3])


def test_mat4_identity():
    m = mat4_identity()
    np.testing.assert_array_equal(m, np.eye(4))


def test_mat4_translation():
    m = mat4_translation(1, 2, 3)
    p = transform_point(m, vec3(0, 0, 0))
    np.testing.assert_array_almost_equal(p, [1, 2, 3])


def test_mat4_scale():
    m = mat4_scale(2, 3, 4)
    p = transform_point(m, vec3(1, 1, 1))
    np.testing.assert_array_almost_equal(p, [2, 3, 4])


def test_mat4_rotation_x():
    m = mat4_rotation_x(np.pi / 2)
    p = transform_point(m, vec3(0, 1, 0))
    np.testing.assert_array_almost_equal(p, [0, 0, 1], decimal=10)


def test_mat4_rotation_y():
    m = mat4_rotation_y(np.pi / 2)
    p = transform_point(m, vec3(0, 0, 1))
    np.testing.assert_array_almost_equal(p, [1, 0, 0], decimal=10)


def test_mat4_rotation_z():
    m = mat4_rotation_z(np.pi / 2)
    p = transform_point(m, vec3(1, 0, 0))
    np.testing.assert_array_almost_equal(p, [0, 1, 0], decimal=10)


def test_quat_identity():
    q = quat_identity()
    np.testing.assert_array_equal(q, [0, 0, 0, 1])


def test_quat_from_axis_angle():
    q = quat_from_axis_angle(vec3(0, 1, 0), np.pi / 2)
    v = quat_rotate_vec3(q, vec3(0, 0, 1))
    np.testing.assert_array_almost_equal(v, [1, 0, 0], decimal=10)


def test_quat_multiply_identity():
    q = quat_from_axis_angle(vec3(1, 0, 0), 0.5)
    result = quat_multiply(q, quat_identity())
    np.testing.assert_array_almost_equal(result, q)


def test_quat_slerp_endpoints():
    a = quat_identity()
    b = quat_from_axis_angle(vec3(0, 1, 0), np.pi / 2)
    # t=0 should return a
    np.testing.assert_array_almost_equal(quat_slerp(a, b, 0.0), a)
    # t=1 should return b
    np.testing.assert_array_almost_equal(quat_slerp(a, b, 1.0), b)


def test_quat_slerp_midpoint():
    a = quat_identity()
    b = quat_from_axis_angle(vec3(0, 1, 0), np.pi)
    mid = quat_slerp(a, b, 0.5)
    # Midpoint should be 90 degrees
    v = quat_rotate_vec3(mid, vec3(0, 0, 1))
    np.testing.assert_array_almost_equal(v, [1, 0, 0], decimal=5)


def test_mat4_from_quaternion_roundtrip():
    q = quat_from_euler(0.3, 0.5, 0.7, "XYZ")
    m = mat4_from_quaternion(q)
    # Rotating a vector should give same result
    v = vec3(1, 2, 3)
    r1 = quat_rotate_vec3(q, v)
    r2 = transform_point(m, v)
    np.testing.assert_array_almost_equal(r1, r2, decimal=10)


def test_mat4_compose():
    pos = vec3(1, 2, 3)
    q = quat_identity()
    scale = vec3(2, 2, 2)
    m = mat4_compose(pos, q, scale)
    p = transform_point(m, vec3(1, 0, 0))
    np.testing.assert_array_almost_equal(p, [3, 2, 3])


def test_mat4_inverse():
    m = mat4_translation(5, 10, 15)
    mi = mat4_inverse(m)
    result = m @ mi
    np.testing.assert_array_almost_equal(result, np.eye(4), decimal=10)


def test_normalize():
    v = normalize(vec3(3, 0, 0))
    np.testing.assert_array_almost_equal(v, [1, 0, 0])


def test_normalize_zero():
    v = normalize(vec3(0, 0, 0))
    np.testing.assert_array_equal(v, [0, 0, 0])


def test_lerp():
    assert lerp(0, 10, 0.5) == 5
    assert lerp(0, 10, 0.0) == 0
    assert lerp(0, 10, 1.0) == 10


def test_clamp():
    assert clamp(5, 0, 10) == 5
    assert clamp(-1, 0, 10) == 0
    assert clamp(15, 0, 10) == 10


def test_deg_rad_roundtrip():
    assert abs(rad_to_deg(deg_to_rad(45.0)) - 45.0) < 1e-10


def test_mat4_look_at():
    eye = vec3(0, 0, 5)
    target = vec3(0, 0, 0)
    up = vec3(0, 1, 0)
    m = mat4_look_at(eye, target, up)
    # Eye should transform to origin in view space
    p = transform_point(m, eye)
    np.testing.assert_array_almost_equal(p, [0, 0, 0], decimal=10)


def test_mat4_perspective():
    m = mat4_perspective(deg_to_rad(60), 1.0, 0.1, 100.0)
    # Should be a valid projection matrix (non-zero diagonal)
    assert m[0, 0] != 0
    assert m[1, 1] != 0
    assert m[3, 2] == -1.0
