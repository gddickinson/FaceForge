"""Equivalence and cache-correctness tests for the vectorised STL parser.

``parse_binary_stl`` and ``build_indexed_geometry`` were rewritten from
Python loops to numpy kernels.  These tests pin the rewrite against the
frozen pre-optimization implementations in ``_reference_stl`` so the
kernels cannot silently drift, and cover the welded-geometry disk cache
(hit, staleness, atomic write, cold == warm).
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np
import pytest

from faceforge.constants import STL_DIR
from faceforge.core.mesh import BufferGeometry
from faceforge.loaders import stl_parser as sp
from faceforge.loaders.stl_parser import (
    build_indexed_geometry,
    load_stl_file,
    parse_binary_stl,
)

from tests.loaders._reference_stl import (
    ref_build_indexed_geometry,
    ref_parse_binary_stl,
)


# ── helpers ─────────────────────────────────────────────────────────

def _stl_bytes(triangles) -> bytes:
    data = b"\x00" * 80 + struct.pack("<I", len(triangles))
    for normal, v0, v1, v2 in triangles:
        data += struct.pack("<3f", *normal)
        data += struct.pack("<3f", *v0)
        data += struct.pack("<3f", *v1)
        data += struct.pack("<3f", *v2)
        data += struct.pack("<H", 0)
    return data


def _random_soup(n_tri: int, seed: int, welds: bool = True) -> bytes:
    """A triangle soup with deliberately repeated vertices to exercise welding."""
    rng = np.random.default_rng(seed)
    # Coordinates snapped to a coarse lattice so distinct triangles share
    # vertices after quantization.
    lattice = np.round(rng.normal(0, 4, (n_tri * 3 if not welds else 40, 3)), 1)
    tris = []
    for i in range(n_tri):
        idx = rng.integers(0, len(lattice), 3)
        v = lattice[idx]
        nrm = np.cross(v[1] - v[0], v[2] - v[0])
        ln = np.linalg.norm(nrm)
        nrm = nrm / ln if ln > 1e-9 else np.array([0.0, 0.0, 1.0])
        tris.append((tuple(nrm), tuple(v[0]), tuple(v[1]), tuple(v[2])))
    return _stl_bytes(tris)


def _deref(geom: BufferGeometry, attr: str) -> np.ndarray:
    """Attribute values in triangle-corner order — invariant to vertex ordering."""
    arr = getattr(geom, attr).reshape(-1, 3)
    return arr[geom.indices].astype(np.float64)


def _real_sample(n: int = 6) -> list[Path]:
    """A size-stratified sample of real BodyParts3D meshes, if present."""
    if not STL_DIR.exists():
        return []
    files = sorted(STL_DIR.glob("*.stl"), key=lambda p: p.stat().st_size)
    if len(files) < n:
        return files
    idx = np.linspace(0, len(files) - 1, n).round().astype(int)
    return [files[i] for i in idx]


# ── parse equivalence ───────────────────────────────────────────────

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_parse_matches_reference_bit_exactly(seed):
    data = _random_soup(400, seed)
    ref = ref_parse_binary_stl(data)
    got = parse_binary_stl(data)
    assert got.vertex_count == ref.vertex_count
    # A frombuffer reinterpretation of the same little-endian float32s must
    # be bit-identical, not merely close.
    assert np.array_equal(got.positions, ref.positions)
    assert np.array_equal(got.normals, ref.normals)


def test_parse_zero_triangles():
    geom = parse_binary_stl(_stl_bytes([]))
    assert geom.vertex_count == 0
    assert len(geom.positions) == 0


def test_parse_ignores_trailing_bytes():
    data = _stl_bytes([((0, 0, 1), (0, 0, 0), (1, 0, 0), (0, 1, 0))])
    assert np.array_equal(
        parse_binary_stl(data).positions,
        parse_binary_stl(data + b"junkjunk").positions,
    )


def test_parsed_positions_are_writable():
    """load_stl_batch transforms coordinates in place, so the buffer must
    not stay attached to the read-only input bytes."""
    geom = parse_binary_stl(_stl_bytes([((0, 0, 1), (0, 0, 0), (1, 0, 0), (0, 1, 0))]))
    geom.positions[0] = 5.0  # must not raise
    assert geom.positions[0] == 5.0


# ── weld equivalence ────────────────────────────────────────────────

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_weld_matches_reference(seed):
    data = _random_soup(400, seed)
    ref = ref_build_indexed_geometry(ref_parse_binary_stl(data))
    got = build_indexed_geometry(parse_binary_stl(data))

    # Same number of welded vertices, and identical geometry once
    # dereferenced through the index buffer (vertex *order* differs: the
    # fast path emits sorted-unique rather than first-appearance order).
    assert got.vertex_count == ref.vertex_count
    assert len(got.indices) == len(ref.indices)
    assert np.array_equal(_deref(got, "positions"), _deref(ref, "positions"))
    # Normals are an accumulate-then-normalize sum, so they agree to
    # float32 summation-order rounding rather than bit-exactly.
    assert np.abs(_deref(got, "normals") - _deref(ref, "normals")).max() < 1e-5


def test_weld_empty_geometry():
    got = build_indexed_geometry(
        BufferGeometry(positions=np.empty(0, np.float32),
                       normals=np.empty(0, np.float32), vertex_count=0)
    )
    assert got.vertex_count == 0
    assert got.indices is not None and len(got.indices) == 0


def test_weld_normals_are_unit_length():
    got = build_indexed_geometry(parse_binary_stl(_random_soup(300, 7)))
    lengths = np.linalg.norm(got.normals.reshape(-1, 3), axis=1)
    assert np.abs(lengths - 1.0).max() < 1e-5


# `slow`: this parametrization reads real BP3D meshes, and the size-stratified
# sample deliberately includes the largest one in the dataset -- 7.78 s of the
# module's measured 8.31 s is the single FMA7163 case.  The synthetic
# equivalence and cache tests above cover the same code paths in milliseconds,
# so the fast tier keeps its coverage of the parser without the dataset.
@pytest.mark.slow
@pytest.mark.skipif(not _real_sample(), reason="BodyParts3D STL assets not present")
@pytest.mark.parametrize("path", _real_sample(), ids=lambda p: p.stem)
def test_real_mesh_equivalence(path):
    """Size-stratified real BP3D meshes: positions bit-identical, unique
    vertex counts identical, normals within float32 rounding."""
    data = path.read_bytes()
    ref_raw = ref_parse_binary_stl(data)
    got_raw = parse_binary_stl(data)
    assert np.array_equal(got_raw.positions, ref_raw.positions)
    assert np.array_equal(got_raw.normals, ref_raw.normals)

    ref = ref_build_indexed_geometry(ref_raw)
    got = build_indexed_geometry(got_raw)
    assert got.vertex_count == ref.vertex_count
    assert np.array_equal(_deref(got, "positions"), _deref(ref, "positions"))
    assert np.abs(_deref(got, "normals") - _deref(ref, "normals")).max() < 1e-5


# ── mesh cache ──────────────────────────────────────────────────────

@pytest.fixture
def stl_on_disk(tmp_path):
    p = tmp_path / "mesh.stl"
    p.write_bytes(_random_soup(500, 11))
    return p


def test_cache_roundtrip_is_identical(stl_on_disk, tmp_path):
    cache = tmp_path / "cache"
    cold = load_stl_file(stl_on_disk, cache_dir=cache)   # miss: parses + writes
    assert len(list(cache.iterdir())) == 1
    warm = load_stl_file(stl_on_disk, cache_dir=cache)   # hit
    uncached = load_stl_file(stl_on_disk, use_cache=False)

    for a, b in ((cold, warm), (cold, uncached)):
        assert a.vertex_count == b.vertex_count
        assert np.array_equal(a.positions, b.positions)
        assert np.array_equal(a.normals, b.normals)
        assert np.array_equal(a.indices, b.indices)


def test_cache_invalidated_by_mtime(stl_on_disk, tmp_path):
    cache = tmp_path / "cache"
    first = load_stl_file(stl_on_disk, cache_dir=cache)
    stl_on_disk.write_bytes(_random_soup(120, 12))
    os.utime(stl_on_disk, (1_000_000, 1_000_000))  # older than the cache entry
    second = load_stl_file(stl_on_disk, cache_dir=cache)
    # Staleness is keyed on (mtime_ns, size), not on "cache newer than
    # source", so an *older* replacement still invalidates.
    assert len(second.indices) != len(first.indices)
    assert np.array_equal(
        second.positions, load_stl_file(stl_on_disk, use_cache=False).positions
    )


def test_partial_cache_entry_is_ignored(stl_on_disk, tmp_path):
    cache = tmp_path / "cache"
    load_stl_file(stl_on_disk, cache_dir=cache)
    entry = next(cache.iterdir())
    truncated = entry.read_bytes()[: len(entry.read_bytes()) // 2]
    entry.write_bytes(truncated)
    got = load_stl_file(stl_on_disk, cache_dir=cache)
    assert np.array_equal(
        got.positions, load_stl_file(stl_on_disk, use_cache=False).positions
    )


def test_cache_write_is_atomic(stl_on_disk, tmp_path, monkeypatch):
    """A process killed mid-write must leave no readable partial entry."""
    cache = tmp_path / "cache"

    real_savez = np.savez
    def die_after_write(file, **kw):
        real_savez(file, **kw)
        raise KeyboardInterrupt("killed after temp write, before replace")

    monkeypatch.setattr(np, "savez", die_after_write)
    with pytest.raises(KeyboardInterrupt):
        load_stl_file(stl_on_disk, cache_dir=cache)
    monkeypatch.undo()

    # Only a .tmp.npz may exist; the real entry name must be absent.
    final = [f for f in cache.iterdir() if not f.name.endswith(".tmp.npz")]
    assert final == []
    got = load_stl_file(stl_on_disk, cache_dir=cache)
    assert np.array_equal(
        got.positions, load_stl_file(stl_on_disk, use_cache=False).positions
    )


def test_cache_dir_is_configurable(stl_on_disk, tmp_path, monkeypatch):
    explicit = tmp_path / "explicit"
    load_stl_file(stl_on_disk, cache_dir=explicit)
    assert list(explicit.glob("*.npz"))

    via_env = tmp_path / "from_env"
    monkeypatch.setenv("FACEFORGE_MESH_CACHE_DIR", str(via_env))
    assert sp.mesh_cache_dir() == via_env
    load_stl_file(stl_on_disk)
    assert list(via_env.glob("*.npz"))

    via_setter = tmp_path / "from_setter"
    sp.set_mesh_cache_dir(via_setter)
    try:
        assert sp.mesh_cache_dir() == via_setter
        load_stl_file(stl_on_disk)
        assert list(via_setter.glob("*.npz"))
    finally:
        sp.set_mesh_cache_dir(None)


def test_cache_can_be_disabled_by_env(stl_on_disk, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setenv("FACEFORGE_MESH_CACHE_OFF", "1")
    load_stl_file(stl_on_disk, cache_dir=cache)
    assert not cache.exists()


def test_unindexed_load_is_not_cached(stl_on_disk, tmp_path):
    cache = tmp_path / "cache"
    geom = load_stl_file(stl_on_disk, indexed=False, cache_dir=cache)
    assert not geom.has_indices
    assert not cache.exists()
