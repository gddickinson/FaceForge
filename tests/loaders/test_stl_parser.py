"""Tests for STL parser."""

import struct
import numpy as np
import pytest

from faceforge.core.mesh import BufferGeometry
from faceforge.loaders.stl_parser import parse_binary_stl, build_indexed_geometry


def _make_stl_data(triangles: list[tuple]) -> bytes:
    """Create minimal binary STL data from triangle definitions.

    Each triangle is ((nx,ny,nz), (v0x,v0y,v0z), (v1x,v1y,v1z), (v2x,v2y,v2z)).
    """
    header = b"\x00" * 80
    tri_count = len(triangles)
    data = header + struct.pack("<I", tri_count)
    for normal, v0, v1, v2 in triangles:
        data += struct.pack("<3f", *normal)
        data += struct.pack("<3f", *v0)
        data += struct.pack("<3f", *v1)
        data += struct.pack("<3f", *v2)
        data += struct.pack("<H", 0)  # attribute byte count
    return data


def test_parse_single_triangle():
    stl = _make_stl_data([
        ((0, 0, 1), (0, 0, 0), (1, 0, 0), (0, 1, 0)),
    ])
    geom = parse_binary_stl(stl)
    assert geom.vertex_count == 3
    assert len(geom.positions) == 9
    assert len(geom.normals) == 9
    np.testing.assert_array_almost_equal(geom.positions[:3], [0, 0, 0])
    np.testing.assert_array_almost_equal(geom.normals[:3], [0, 0, 1])


def test_parse_two_triangles():
    stl = _make_stl_data([
        ((0, 0, 1), (0, 0, 0), (1, 0, 0), (0, 1, 0)),
        ((0, 0, 1), (1, 0, 0), (1, 1, 0), (0, 1, 0)),
    ])
    geom = parse_binary_stl(stl)
    assert geom.vertex_count == 6
    assert geom.triangle_count == 2


def test_parse_invalid_stl():
    with pytest.raises(ValueError):
        parse_binary_stl(b"too short")


def test_build_indexed_geometry():
    """Two triangles sharing an edge should merge shared vertices."""
    stl = _make_stl_data([
        ((0, 0, 1), (0, 0, 0), (1, 0, 0), (0, 1, 0)),
        ((0, 0, 1), (1, 0, 0), (1, 1, 0), (0, 1, 0)),
    ])
    geom = parse_binary_stl(stl)
    indexed = build_indexed_geometry(geom)

    # Should merge shared vertices: 6 original → 4 unique
    assert indexed.vertex_count == 4
    assert indexed.has_indices
    assert len(indexed.indices) == 6  # Still 6 index entries for 2 triangles


def test_indexed_preserves_shape():
    """Indexed geometry should represent the same mesh."""
    stl = _make_stl_data([
        ((0, 0, 1), (0, 0, 0), (1, 0, 0), (0, 1, 0)),
    ])
    geom = parse_binary_stl(stl)
    indexed = build_indexed_geometry(geom)

    # With a single triangle, all 3 vertices are unique
    assert indexed.vertex_count == 3
    assert indexed.triangle_count == 1
