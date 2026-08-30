"""Frozen pre-optimization STL parser, kept as an equivalence oracle.

These are byte-for-byte the implementations that shipped before
``faceforge.loaders.stl_parser`` was vectorised (per-triangle
``struct.unpack_from`` loop; per-vertex dict weld).  They are slow on
purpose: their only job is to give the regression tests something
independent to compare the fast path against, so the vectorised kernels
cannot silently drift.

Do not "optimise" this module.
"""

from __future__ import annotations

import struct

import numpy as np

from faceforge.core.mesh import BufferGeometry


def ref_parse_binary_stl(data: bytes) -> BufferGeometry:
    """Original per-triangle struct.unpack_from parser."""
    if len(data) < 84:
        raise ValueError("Invalid STL: too short")

    tri_count = struct.unpack_from("<I", data, 80)[0]
    expected_size = 84 + tri_count * 50
    if len(data) < expected_size:
        raise ValueError(f"Invalid STL: expected {expected_size} bytes, got {len(data)}")

    vert_count = tri_count * 3
    positions = np.empty(vert_count * 3, dtype=np.float32)
    normals = np.empty(vert_count * 3, dtype=np.float32)

    offset = 84
    for i in range(tri_count):
        nx, ny, nz = struct.unpack_from("<3f", data, offset)
        offset += 12
        for j in range(3):
            vi = (i * 3 + j) * 3
            x, y, z = struct.unpack_from("<3f", data, offset)
            positions[vi] = x
            positions[vi + 1] = y
            positions[vi + 2] = z
            normals[vi] = nx
            normals[vi + 1] = ny
            normals[vi + 2] = nz
            offset += 12
        offset += 2  # attribute byte count

    return BufferGeometry(
        positions=positions,
        normals=normals,
        vertex_count=vert_count,
    )


def ref_build_indexed_geometry(
    geom: BufferGeometry, tolerance: float = 1e-5
) -> BufferGeometry:
    """Original dict-based duplicate-vertex weld."""
    pos = geom.positions.reshape(-1, 3)
    nrm = geom.normals.reshape(-1, 3)
    vert_count = len(pos)

    scale = 1.0 / tolerance
    quantized = (pos * scale).astype(np.int64)

    vertex_map: dict[tuple, int] = {}
    unique_positions = []
    unique_normals = []
    index_remap = np.empty(vert_count, dtype=np.uint32)

    for i in range(vert_count):
        key = (quantized[i, 0], quantized[i, 1], quantized[i, 2])
        if key in vertex_map:
            idx = vertex_map[key]
            unique_normals[idx] += nrm[i]
            index_remap[i] = idx
        else:
            idx = len(unique_positions)
            vertex_map[key] = idx
            unique_positions.append(pos[i].copy())
            unique_normals.append(nrm[i].copy())
            index_remap[i] = idx

    out_pos = np.array(unique_positions, dtype=np.float32)
    out_nrm = np.array(unique_normals, dtype=np.float32)
    lengths = np.linalg.norm(out_nrm, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-10)
    out_nrm /= lengths

    return BufferGeometry(
        positions=out_pos.ravel(),
        normals=out_nrm.ravel(),
        indices=index_remap,
        vertex_count=len(unique_positions),
    )
