"""NumPy-backed math utilities: Vec3, Quaternion, Mat4 operations.

Provides lightweight wrappers and utility functions for 3D math.
Vectors are plain numpy arrays; quaternions are [x, y, z, w] arrays.
Matrices are 4x4 numpy arrays in column-major order (OpenGL convention).
"""

import numpy as np
from numpy.typing import NDArray

# Type aliases
Vec3 = NDArray[np.float64]
Vec4 = NDArray[np.float64]
Mat3 = NDArray[np.float64]
Mat4 = NDArray[np.float64]
Quat = NDArray[np.float64]  # [x, y, z, w]


def vec3(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Vec3:
    return np.array([x, y, z], dtype=np.float64)


def vec4(x: float = 0.0, y: float = 0.0, z: float = 0.0, w: float = 1.0) -> Vec4:
    return np.array([x, y, z, w], dtype=np.float64)


def mat4_identity() -> Mat4:
    return np.eye(4, dtype=np.float64)


def mat3_identity() -> Mat3:
    return np.eye(3, dtype=np.float64)


def mat4_translation(x: float, y: float, z: float) -> Mat4:
    m = np.eye(4, dtype=np.float64)
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m


def mat4_scale(sx: float, sy: float, sz: float) -> Mat4:
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = sx
    m[1, 1] = sy
    m[2, 2] = sz
    return m


def mat4_rotation_x(angle_rad: float) -> Mat4:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    m = np.eye(4, dtype=np.float64)
    m[1, 1] = c
    m[1, 2] = -s
    m[2, 1] = s
    m[2, 2] = c
    return m


def mat4_rotation_y(angle_rad: float) -> Mat4:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = c
    m[0, 2] = s
    m[2, 0] = -s
    m[2, 2] = c
    return m


def mat4_rotation_z(angle_rad: float) -> Mat4:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = c
    m[0, 1] = -s
    m[1, 0] = s
    m[1, 1] = c
    return m


def mat4_from_quaternion(q: Quat) -> Mat4:
    """Convert quaternion [x,y,z,w] to 4x4 rotation matrix."""
    x, y, z, w = q
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = 1 - 2 * (y * y + z * z)
    m[0, 1] = 2 * (x * y - z * w)
    m[0, 2] = 2 * (x * z + y * w)
    m[1, 0] = 2 * (x * y + z * w)
    m[1, 1] = 1 - 2 * (x * x + z * z)
    m[1, 2] = 2 * (y * z - x * w)
    m[2, 0] = 2 * (x * z - y * w)
    m[2, 1] = 2 * (y * z + x * w)
    m[2, 2] = 1 - 2 * (x * x + y * y)
    return m


def mat4_compose(position: Vec3, quaternion: Quat, scale: Vec3) -> Mat4:
    """Compose TRS matrix from position, quaternion rotation, and scale."""
    m = mat4_from_quaternion(quaternion)
    m[0, :3] *= scale[0]
    m[1, :3] *= scale[1]
    m[2, :3] *= scale[2]
    m[0, 3] = position[0]
    m[1, 3] = position[1]
    m[2, 3] = position[2]
    return m


def mat4_perspective(fov_rad: float, aspect: float, near: float, far: float) -> Mat4:
    """Create perspective projection matrix."""
    f = 1.0 / np.tan(fov_rad / 2.0)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def mat4_look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4:
    """Create view matrix (camera look-at)."""
    f = normalize(target - eye)
    s = normalize(np.cross(f, up))
    # Degenerate case: forward is parallel to up (e.g. looking straight down).
    # Fall back to an alternative up vector.
    if np.linalg.norm(s) < 1e-6:
        alt_up = np.array([0.0, 0.0, -1.0]) if abs(f[1]) > 0.9 else np.array([0.0, 1.0, 0.0])
        s = normalize(np.cross(f, alt_up))
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float64)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def mat4_inverse(m: Mat4) -> Mat4:
    return np.linalg.inv(m)


def mat3_normal(m: Mat4) -> Mat3:
    """Extract normal matrix (inverse transpose of upper-left 3x3)."""
    return np.linalg.inv(m[:3, :3]).T


# Quaternion operations

def quat_identity() -> Quat:
    return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def quat_from_euler(x: float, y: float, z: float, order: str = "XYZ") -> Quat:
    """Create quaternion from Euler angles (radians)."""
    cx, sx = np.cos(x / 2), np.sin(x / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    cz, sz = np.cos(z / 2), np.sin(z / 2)

    if order == "XYZ":
        return np.array([
            sx * cy * cz + cx * sy * sz,
            cx * sy * cz - sx * cy * sz,
            cx * cy * sz + sx * sy * cz,
            cx * cy * cz - sx * sy * sz,
        ], dtype=np.float64)
    elif order == "YXZ":
        return np.array([
            sx * cy * cz + cx * sy * sz,
            cx * sy * cz - sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
            cx * cy * cz + sx * sy * sz,
        ], dtype=np.float64)
    elif order == "ZYX":
        return np.array([
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
            cx * cy * cz + sx * sy * sz,
        ], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported Euler order: {order}")


def quat_from_axis_angle(axis: Vec3, angle: float) -> Quat:
    """Create quaternion from axis-angle."""
    half = angle / 2
    s = np.sin(half)
    a = normalize(axis)
    return np.array([a[0] * s, a[1] * s, a[2] * s, np.cos(half)], dtype=np.float64)


def quat_multiply(a: Quat, b: Quat) -> Quat:
    """Multiply two quaternions (a * b)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dtype=np.float64)


def quat_conjugate(q: Quat) -> Quat:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def quat_normalize(q: Quat) -> Quat:
    n = np.linalg.norm(q)
    if n < 1e-10:
        return quat_identity()
    return q / n


def quat_slerp(a: Quat, b: Quat, t: float) -> Quat:
    """Spherical linear interpolation between two quaternions."""
    dot = np.dot(a, b)
    if dot < 0:
        b = -b
        dot = -dot
    if dot > 0.9995:
        result = a + t * (b - a)
        return quat_normalize(result)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    if sin_theta < 1e-10:
        return a.copy()
    wa = np.sin((1 - t) * theta) / sin_theta
    wb = np.sin(t * theta) / sin_theta
    return quat_normalize(wa * a + wb * b)


def quat_rotate_vec3(q: Quat, v: Vec3) -> Vec3:
    """Rotate a vector by a quaternion."""
    qv = q[:3]
    w = q[3]
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


# Vector operations

def normalize(v: Vec3) -> Vec3:
    n = np.linalg.norm(v)
    if n < 1e-10:
        return np.zeros_like(v)
    return v / n


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_vec3(a: Vec3, b: Vec3, t: float) -> Vec3:
    return a + (b - a) * t


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def deg_to_rad(degrees: float) -> float:
    return degrees * np.pi / 180.0


def rad_to_deg(radians: float) -> float:
    return radians * 180.0 / np.pi


def transform_point(m: Mat4, p: Vec3) -> Vec3:
    """Transform a point by a 4x4 matrix."""
    v = np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
    r = m @ v
    return r[:3]


def transform_direction(m: Mat4, d: Vec3) -> Vec3:
    """Transform a direction by a 4x4 matrix (ignores translation)."""
    return (m[:3, :3] @ d)


def world_to_screen(
    world_pos: Vec3,
    view_proj: Mat4,
    viewport_w: int,
    viewport_h: int,
) -> tuple[float, float, bool]:
    """Project a 3D world point to 2D screen coordinates.

    Returns (screen_x, screen_y, in_front) where in_front is True if the
    point is in front of the camera (clip w > 0).
    """
    v = np.array([world_pos[0], world_pos[1], world_pos[2], 1.0], dtype=np.float64)
    clip = view_proj @ v
    if abs(clip[3]) < 1e-10:
        return 0.0, 0.0, False
    ndc_x = clip[0] / clip[3]
    ndc_y = clip[1] / clip[3]
    screen_x = (ndc_x * 0.5 + 0.5) * viewport_w
    screen_y = (1.0 - (ndc_y * 0.5 + 0.5)) * viewport_h
    return screen_x, screen_y, clip[3] > 0


# ── Batch (vectorized) quaternion operations ──────────────────────────

def batch_mat3_to_quat(R: NDArray) -> NDArray:
    """Convert (N, 3, 3) rotation matrices to (N, 4) quaternions [x, y, z, w].

    Uses Shepperd's method with masked branching for numerical stability.
    """
    N = len(R)
    q = np.zeros((N, 4), dtype=np.float64)
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    # Case 1: trace > 0
    m1 = trace > 0
    if m1.any():
        s = 0.5 / np.sqrt(trace[m1] + 1.0)
        q[m1, 3] = 0.25 / s
        q[m1, 0] = (R[m1, 2, 1] - R[m1, 1, 2]) * s
        q[m1, 1] = (R[m1, 0, 2] - R[m1, 2, 0]) * s
        q[m1, 2] = (R[m1, 1, 0] - R[m1, 0, 1]) * s

    # Case 2: R[0,0] is largest diagonal
    m2 = ~m1 & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if m2.any():
        s = 2.0 * np.sqrt(1.0 + R[m2, 0, 0] - R[m2, 1, 1] - R[m2, 2, 2])
        q[m2, 3] = (R[m2, 2, 1] - R[m2, 1, 2]) / s
        q[m2, 0] = 0.25 * s
        q[m2, 1] = (R[m2, 0, 1] + R[m2, 1, 0]) / s
        q[m2, 2] = (R[m2, 0, 2] + R[m2, 2, 0]) / s

    # Case 3: R[1,1] is largest diagonal
    m3 = ~m1 & ~m2 & (R[:, 1, 1] > R[:, 2, 2])
    if m3.any():
        s = 2.0 * np.sqrt(1.0 + R[m3, 1, 1] - R[m3, 0, 0] - R[m3, 2, 2])
        q[m3, 3] = (R[m3, 0, 2] - R[m3, 2, 0]) / s
        q[m3, 0] = (R[m3, 0, 1] + R[m3, 1, 0]) / s
        q[m3, 1] = 0.25 * s
        q[m3, 2] = (R[m3, 1, 2] + R[m3, 2, 1]) / s

    # Case 4: R[2,2] is largest diagonal
    m4 = ~m1 & ~m2 & ~m3
    if m4.any():
        s = 2.0 * np.sqrt(1.0 + R[m4, 2, 2] - R[m4, 0, 0] - R[m4, 1, 1])
        q[m4, 3] = (R[m4, 1, 0] - R[m4, 0, 1]) / s
        q[m4, 0] = (R[m4, 0, 2] + R[m4, 2, 0]) / s
        q[m4, 1] = (R[m4, 1, 2] + R[m4, 2, 1]) / s
        q[m4, 2] = 0.25 * s

    return q


def batch_quat_multiply(a: NDArray, b: NDArray) -> NDArray:
    """Multiply (N, 4) quaternions [x, y, z, w]: result = a * b."""
    ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.column_stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def batch_quat_rotate(q: NDArray, v: NDArray) -> NDArray:
    """Rotate (N, 3) vectors by (N, 4) quaternions [x, y, z, w]."""
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    # t = 2 * cross(q.xyz, v)
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    # result = v + qw * t + cross(q.xyz, t)
    return np.column_stack([
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    ])


def batch_mat4_to_dual_quat(M: NDArray) -> NDArray:
    """Convert (N, 4, 4) transform matrices to (N, 8) dual quaternions.

    Returns array where [:, 0:4] is the real part (rotation quaternion
    [x, y, z, w]) and [:, 4:8] is the dual part encoding translation.

    The dual quaternion representation interpolates rotations properly
    (via the real quaternion) avoiding the volume collapse artifacts of
    linear position blending.
    """
    R = M[:, :3, :3]  # (N, 3, 3)
    t = M[:, :3, 3]   # (N, 3)

    q_r = batch_mat3_to_quat(R)  # (N, 4) [x, y, z, w]

    # Dual part: q_d = 0.5 * pure_quat(t) * q_r
    # where pure_quat(t) = [tx, ty, tz, 0]
    N = len(M)
    t_quat = np.zeros((N, 4), dtype=np.float64)
    t_quat[:, :3] = t

    q_d = 0.5 * batch_quat_multiply(t_quat, q_r)

    return np.concatenate([q_r, q_d], axis=1)  # (N, 8)
