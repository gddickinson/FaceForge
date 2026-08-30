"""AssetManager / load_stl_batch cache-sharing tests.

``AssetManager.get_stl`` and ``load_stl_batch`` are two independent load
paths over the same STL files, and the batch path cannot reuse the
manager's in-memory geometry because it applies the BP3D coordinate
transform in place.  They instead share the on-disk welded-geometry cache,
so a mesh welded by one path is never welded again by the other.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from faceforge.loaders import stl_parser as sp
from faceforge.loaders.asset_manager import AssetManager
from faceforge.loaders.stl_batch_loader import CoordinateTransform, load_stl_batch


def _cube_stl(offset: float = 0.0) -> bytes:
    """A small closed mesh with shared corners, so welding does real work."""
    c = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float64) + offset
    faces = [
        (0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
        (0, 4, 5), (0, 5, 1), (2, 6, 7), (2, 7, 3),
        (1, 5, 6), (1, 6, 2), (0, 3, 7), (0, 7, 4),
    ]
    data = b"\x00" * 80 + struct.pack("<I", len(faces))
    for f in faces:
        v = c[list(f)]
        n = np.cross(v[1] - v[0], v[2] - v[0])
        n = n / max(np.linalg.norm(n), 1e-9)
        data += struct.pack("<3f", *n)
        for k in range(3):
            data += struct.pack("<3f", *v[k])
        data += struct.pack("<H", 0)
    return data


@pytest.fixture
def stl_dir(tmp_path):
    d = tmp_path / "stl"
    d.mkdir()
    (d / "FMA00001.stl").write_bytes(_cube_stl())
    return d


def test_get_stl_uses_the_mesh_cache(stl_dir, tmp_path):
    cache = tmp_path / "cache"
    am = AssetManager(stl_dir=stl_dir, mesh_cache_dir=cache)
    first = am.get_stl("FMA00001")
    assert list(cache.glob("*.npz")), "get_stl did not populate the mesh cache"

    # A fresh manager (empty in-memory cache) must hit the disk cache and
    # return identical geometry.
    am2 = AssetManager(stl_dir=stl_dir, mesh_cache_dir=cache)
    second = am2.get_stl("FMA00001")
    assert second.vertex_count == first.vertex_count
    assert np.array_equal(second.positions, first.positions)
    assert np.array_equal(second.indices, first.indices)


def test_batch_and_get_stl_share_the_disk_cache(stl_dir, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setenv("FACEFORGE_MESH_CACHE_DIR", str(cache))

    # Batch path welds first and writes the cache entry.
    defs = [{"name": "cube", "stl": "FMA00001", "color": 0xCCCCCC}]
    load_stl_batch(defs, stl_dir=stl_dir, transform=CoordinateTransform())
    entries = list(cache.glob("*.npz"))
    assert len(entries) == 1, "load_stl_batch did not use the mesh cache"

    # get_stl on the same file must consume that entry rather than re-weld.
    calls: list[int] = []
    real_build = sp.build_indexed_geometry

    def counting_build(geom, tolerance=1e-5):
        calls.append(1)
        return real_build(geom, tolerance)

    monkeypatch.setattr(sp, "build_indexed_geometry", counting_build)
    am = AssetManager(stl_dir=stl_dir)
    geom = am.get_stl("FMA00001")
    assert calls == [], "get_stl re-welded a mesh already in the cache"
    assert geom.vertex_count == 8  # cube corners after welding


def test_batch_transform_does_not_corrupt_get_stl_geometry(stl_dir, tmp_path):
    """The batch loader transforms coordinates in place; the manager's cached
    geometry must not be the same object, or it would be double-transformed."""
    cache = tmp_path / "cache"
    am = AssetManager(stl_dir=stl_dir, mesh_cache_dir=cache)
    before = am.get_stl("FMA00001").positions.copy()

    defs = [{"name": "cube", "stl": "FMA00001", "color": 0xCCCCCC}]
    result = load_stl_batch(defs, stl_dir=stl_dir, transform=CoordinateTransform())
    assert result.meshes, "batch load produced no meshes"

    after = am.get_stl("FMA00001").positions
    assert np.array_equal(before, after)
    # And the batch mesh really was transformed (so the check above is not
    # passing trivially).
    assert not np.array_equal(
        result.meshes[0].geometry.positions[:9], before[:9]
    )
