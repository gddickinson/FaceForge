"""Procedural mesh builders for scene environment objects.

All functions return :class:`BufferGeometry` with positions + normals,
compatible with the existing GL rendering pipeline.
"""

import math

import numpy as np

from faceforge.core.mesh import BufferGeometry


def make_box(width: float, height: float, depth: float) -> BufferGeometry:
    """Create a box with unique normals per face (24 verts, 12 tris).

    Centered at origin. Dimensions along X, Y, Z respectively.
    """
    hw, hh, hd = width / 2, height / 2, depth / 2

    # 6 faces, 4 verts each = 24 verts
    # Order: +X, -X, +Y, -Y, +Z, -Z
    positions = []
    normals = []
    indices = []

    faces = [
        # (normal, 4 corners)
        ([1, 0, 0],  [(hw, -hh, -hd), (hw, hh, -hd), (hw, hh, hd), (hw, -hh, hd)]),
        ([-1, 0, 0], [(-hw, -hh, hd), (-hw, hh, hd), (-hw, hh, -hd), (-hw, -hh, -hd)]),
        ([0, 1, 0],  [(-hw, hh, -hd), (-hw, hh, hd), (hw, hh, hd), (hw, hh, -hd)]),
        ([0, -1, 0], [(-hw, -hh, hd), (-hw, -hh, -hd), (hw, -hh, -hd), (hw, -hh, hd)]),
        ([0, 0, 1],  [(-hw, -hh, hd), (hw, -hh, hd), (hw, hh, hd), (-hw, hh, hd)]),
        ([0, 0, -1], [(hw, -hh, -hd), (-hw, -hh, -hd), (-hw, hh, -hd), (hw, hh, -hd)]),
    ]

    for normal, corners in faces:
        base = len(positions)
        for c in corners:
            positions.append(c)
            normals.append(normal)
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])

    pos = np.array(positions, dtype=np.float32).ravel()
    nrm = np.array(normals, dtype=np.float32).ravel()
    idx = np.array(indices, dtype=np.uint32)

    return BufferGeometry(positions=pos, normals=nrm, indices=idx)


def make_plane(
    width: float, depth: float,
    segments_w: int = 1, segments_d: int = 1,
) -> BufferGeometry:
    """Create a subdivided plane in the XZ plane, normal pointing +Y.

    Centered at origin with Y=0.
    """
    verts = []
    norms = []
    idxs = []

    for iz in range(segments_d + 1):
        for ix in range(segments_w + 1):
            x = (ix / segments_w - 0.5) * width
            z = (iz / segments_d - 0.5) * depth
            verts.append((x, 0.0, z))
            norms.append((0.0, 1.0, 0.0))

    cols = segments_w + 1
    for iz in range(segments_d):
        for ix in range(segments_w):
            a = iz * cols + ix
            b = a + 1
            c = a + cols
            d = c + 1
            idxs.extend([a, c, b, b, c, d])

    pos = np.array(verts, dtype=np.float32).ravel()
    nrm = np.array(norms, dtype=np.float32).ravel()
    idx = np.array(idxs, dtype=np.uint32)

    return BufferGeometry(positions=pos, normals=nrm, indices=idx)


def make_cylinder(
    radius: float, height: float, segments: int = 16,
) -> BufferGeometry:
    """Create a cylinder along the Y axis, centered at origin.

    Includes top and bottom caps.
    """
    positions = []
    normals = []
    indices = []
    half_h = height / 2

    # --- Side ---
    side_base = 0
    for i in range(segments + 1):
        theta = (i / segments) * 2 * math.pi
        nx = math.cos(theta)
        nz = math.sin(theta)
        x = radius * nx
        z = radius * nz
        # Bottom vertex
        positions.append((x, -half_h, z))
        normals.append((nx, 0.0, nz))
        # Top vertex
        positions.append((x, half_h, z))
        normals.append((nx, 0.0, nz))

    for i in range(segments):
        b = side_base + i * 2
        indices.extend([b, b + 2, b + 1, b + 1, b + 2, b + 3])

    # --- Top cap ---
    top_center = len(positions) // 3
    positions.append((0, half_h, 0))
    normals.append((0, 1, 0))
    for i in range(segments):
        theta = (i / segments) * 2 * math.pi
        positions.append((radius * math.cos(theta), half_h, radius * math.sin(theta)))
        normals.append((0, 1, 0))
    for i in range(segments):
        n = top_center + 1 + i
        nn = top_center + 1 + (i + 1) % segments
        indices.extend([top_center, n, nn])

    # --- Bottom cap ---
    bot_center = len(positions) // 3
    positions.append((0, -half_h, 0))
    normals.append((0, -1, 0))
    for i in range(segments):
        theta = (i / segments) * 2 * math.pi
        positions.append((radius * math.cos(theta), -half_h, radius * math.sin(theta)))
        normals.append((0, -1, 0))
    for i in range(segments):
        n = bot_center + 1 + i
        nn = bot_center + 1 + (i + 1) % segments
        indices.extend([bot_center, nn, n])  # reversed winding

    pos = np.array(positions, dtype=np.float32).ravel()
    nrm = np.array(normals, dtype=np.float32).ravel()
    idx = np.array(indices, dtype=np.uint32)

    return BufferGeometry(positions=pos, normals=nrm, indices=idx)


def make_disc(radius: float, segments: int = 16) -> BufferGeometry:
    """Create a flat disc in the XZ plane at Y=0, normal pointing -Y.

    Used for lamp shade bottom.
    """
    positions = [(0.0, 0.0, 0.0)]
    normals_list = [(0.0, -1.0, 0.0)]
    indices = []

    for i in range(segments):
        theta = (i / segments) * 2 * math.pi
        positions.append((radius * math.cos(theta), 0.0, radius * math.sin(theta)))
        normals_list.append((0.0, -1.0, 0.0))

    for i in range(segments):
        n = 1 + i
        nn = 1 + (i + 1) % segments
        indices.extend([0, nn, n])  # CW from below

    pos = np.array(positions, dtype=np.float32).ravel()
    nrm = np.array(normals_list, dtype=np.float32).ravel()
    idx = np.array(indices, dtype=np.uint32)

    return BufferGeometry(positions=pos, normals=nrm, indices=idx)


def make_sphere(radius: float, segments: int = 16, rings: int = 12) -> BufferGeometry:
    """Create a UV sphere centred at the origin."""
    positions = []
    normals = []
    indices = []
    for r in range(rings + 1):
        phi = math.pi * r / rings           # 0 at +Y pole, pi at -Y pole
        y = math.cos(phi)
        ring_r = math.sin(phi)
        for s_ in range(segments + 1):
            theta = 2 * math.pi * s_ / segments
            nx, nz = ring_r * math.cos(theta), ring_r * math.sin(theta)
            positions.append((radius * nx, radius * y, radius * nz))
            normals.append((nx, y, nz))
    cols = segments + 1
    for r in range(rings):
        for s_ in range(segments):
            a = r * cols + s_
            b = a + cols
            indices.extend([a, b, a + 1, a + 1, b, b + 1])
    pos = np.array(positions, dtype=np.float32).ravel()
    nrm = np.array(normals, dtype=np.float32).ravel()
    idx = np.array(indices, dtype=np.uint32)
    return BufferGeometry(positions=pos, normals=nrm, indices=idx)


def make_torus(radius: float, tube: float, segments: int = 24, sides: int = 8) -> BufferGeometry:
    """Create a torus in the XZ plane (axis +Y), centred at the origin."""
    positions = []
    normals = []
    indices = []
    for i in range(segments + 1):
        u = 2 * math.pi * i / segments
        cu, su = math.cos(u), math.sin(u)
        for j in range(sides + 1):
            v = 2 * math.pi * j / sides
            cv, sv = math.cos(v), math.sin(v)
            positions.append(((radius + tube * cv) * cu, tube * sv, (radius + tube * cv) * su))
            normals.append((cv * cu, sv, cv * su))
    cols = sides + 1
    for i in range(segments):
        for j in range(sides):
            a = i * cols + j
            b = a + cols
            indices.extend([a, a + 1, b, b, a + 1, b + 1])
    pos = np.array(positions, dtype=np.float32).ravel()
    nrm = np.array(normals, dtype=np.float32).ravel()
    idx = np.array(indices, dtype=np.uint32)
    return BufferGeometry(positions=pos, normals=nrm, indices=idx)
