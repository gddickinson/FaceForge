"""Binary STL parser with indexed geometry support."""

import struct
from pathlib import Path

import numpy as np

from faceforge.core.mesh import BufferGeometry


def parse_binary_stl(data: bytes) -> BufferGeometry:
    """Parse binary STL data into a BufferGeometry with per-triangle vertices.

    Binary STL format:
    - 80 bytes header
    - 4 bytes uint32 triangle count
    - Per triangle (50 bytes):
      - 12 bytes normal (3x float32)
      - 36 bytes vertices (3x 3x float32)
      - 2 bytes attribute byte count
    """
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
        # Normal (same for all 3 vertices of this triangle)
        nx, ny, nz = struct.unpack_from("<3f", data, offset)
        offset += 12

        # 3 vertices
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


def build_indexed_geometry(geom: BufferGeometry, tolerance: float = 1e-5) -> BufferGeometry:
    """Convert non-indexed geometry to indexed by merging duplicate vertices.

    Uses a spatial hash to find and merge vertices within tolerance.
    This reduces ~957k triangle vertices to ~160k unique vertices.
    """
    pos = geom.positions.reshape(-1, 3)
    nrm = geom.normals.reshape(-1, 3)
    vert_count = len(pos)

    # Quantize positions for hashing
    scale = 1.0 / tolerance
    quantized = (pos * scale).astype(np.int64)

    # Hash map: quantized position tuple → unique index
    vertex_map: dict[tuple, int] = {}
    unique_positions = []
    unique_normals = []
    index_remap = np.empty(vert_count, dtype=np.uint32)

    for i in range(vert_count):
        key = (quantized[i, 0], quantized[i, 1], quantized[i, 2])
        if key in vertex_map:
            idx = vertex_map[key]
            # Accumulate normals for averaging
            unique_normals[idx] += nrm[i]
            index_remap[i] = idx
        else:
            idx = len(unique_positions)
            vertex_map[key] = idx
            unique_positions.append(pos[i].copy())
            unique_normals.append(nrm[i].copy())
            index_remap[i] = idx

    # Normalize accumulated normals
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


def load_stl_file(path: Path, indexed: bool = True) -> BufferGeometry:
    """Load an STL file and optionally build indexed geometry."""
    with open(path, "rb") as f:
        data = f.read()

    geom = parse_binary_stl(data)
    if indexed:
        geom = build_indexed_geometry(geom)
    return geom
