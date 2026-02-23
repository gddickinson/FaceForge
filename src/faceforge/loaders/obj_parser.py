"""Wavefront OBJ parser → BufferGeometry."""

import numpy as np
from numpy.typing import NDArray

from faceforge.core.mesh import BufferGeometry


def parse_obj(text: str) -> BufferGeometry:
    """Parse a Wavefront OBJ string into indexed BufferGeometry.

    Supports ``v``, ``vn``, and ``f`` lines.  Quads are triangulated
    into two triangles each.  If vertex normals are absent, flat face
    normals are computed and averaged to per-vertex normals.

    Parameters
    ----------
    text : str
        The OBJ file contents.

    Returns
    -------
    BufferGeometry
        Indexed geometry with positions, normals, and triangle indices.
    """
    positions: list[list[float]] = []
    normals: list[list[float]] = []
    face_verts: list[list[int]] = []   # vertex indices per face
    face_norms: list[list[int]] = []   # normal indices per face (may be empty)

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        key = parts[0]

        if key == "v" and len(parts) >= 4:
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif key == "vn" and len(parts) >= 4:
            normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif key == "f":
            vi: list[int] = []
            ni: list[int] = []
            for token in parts[1:]:
                indices = token.split("/")
                vi.append(int(indices[0]) - 1)  # OBJ is 1-based
                if len(indices) >= 3 and indices[2]:
                    ni.append(int(indices[2]) - 1)
            face_verts.append(vi)
            if ni:
                face_norms.append(ni)

    pos_arr = np.array(positions, dtype=np.float32).reshape(-1)  # Flat (V*3,)
    vertex_count = len(positions)

    # Triangulate faces (fan triangulation for quads/ngons)
    tri_indices: list[int] = []
    tri_norm_indices: list[int] = []
    has_normals = len(face_norms) == len(face_verts) and len(normals) > 0

    for fi, fv in enumerate(face_verts):
        fn = face_norms[fi] if has_normals else []
        for k in range(1, len(fv) - 1):
            tri_indices.extend([fv[0], fv[k], fv[k + 1]])
            if has_normals:
                tri_norm_indices.extend([fn[0], fn[k], fn[k + 1]])

    idx_arr = np.array(tri_indices, dtype=np.uint32)

    # Build normals array
    if has_normals and len(normals) == vertex_count:
        # Per-vertex normals directly from vn lines (common case)
        norm_arr = np.array(normals, dtype=np.float32).reshape(-1)
    else:
        # Compute flat face normals, then average to per-vertex
        norm_arr = _compute_vertex_normals(
            pos_arr.reshape(-1, 3), idx_arr,
        ).reshape(-1)

    return BufferGeometry(
        positions=pos_arr,
        normals=norm_arr,
        indices=idx_arr,
        vertex_count=vertex_count,
    )


def _compute_vertex_normals(
    positions: NDArray[np.float32],
    indices: NDArray[np.uint32],
) -> NDArray[np.float32]:
    """Compute smooth per-vertex normals from triangle indices."""
    normals = np.zeros_like(positions)
    n_tris = len(indices) // 3

    for t in range(n_tris):
        i0, i1, i2 = indices[t * 3], indices[t * 3 + 1], indices[t * 3 + 2]
        v0, v1, v2 = positions[i0], positions[i1], positions[i2]
        edge1 = v1 - v0
        edge2 = v2 - v0
        fn = np.cross(edge1, edge2)
        normals[i0] += fn
        normals[i1] += fn
        normals[i2] += fn

    # Normalize
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-8)
    normals /= lengths
    return normals


def load_obj_file(path) -> BufferGeometry:
    """Load an OBJ file from disk.

    Parameters
    ----------
    path : str or Path
        Path to the ``.obj`` file.

    Returns
    -------
    BufferGeometry
    """
    with open(path, "r") as f:
        text = f.read()
    return parse_obj(text)
