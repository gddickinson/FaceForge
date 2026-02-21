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
