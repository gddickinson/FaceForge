"""The binding-solve disk cache must be exact, keyed correctly, and optional.

``register_skin_mesh`` on the full-body skin mesh (791,729 vertices, 148
joints) costs ~34 s of deterministic CPU on the main thread, so the solve is
memoised on disk.  A cache that returned *nearly* the right weights would be
worse than no cache, so these tests assert bitwise equality rather than
closeness, and check that every input the solve reads is part of the key.

The real skin mesh is far too large for a unit test, so ``MIN_VERTS`` is
lowered and a synthetic grid mesh is bound instead: the cache code path is
identical, only the size differs.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceforge.body import skinning_cache
from faceforge.body.soft_tissue import SkinJoint, SoftTissueSkinning
from faceforge.core.mesh import BufferGeometry, MeshInstance
from faceforge.core.math_utils import mat4_identity
from faceforge.core.scene_graph import SceneNode


def _grid_mesh(nx: int = 40, ny: int = 40) -> MeshInstance:
    """A triangulated plane in the XZ plane, so edges and topology are real."""
    xs = np.linspace(-20.0, 20.0, nx)
    zs = np.linspace(-40.0, 40.0, ny)
    gx, gz = np.meshgrid(xs, zs, indexing="ij")
    positions = np.stack(
        [gx.ravel(), np.zeros(gx.size), gz.ravel()], axis=1,
    ).astype(np.float32)

    tris = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a = i * ny + j
            tris.append((a, a + 1, a + ny))
            tris.append((a + 1, a + ny + 1, a + ny))
    indices = np.array(tris, dtype=np.uint32).ravel()

    geom = BufferGeometry(
        positions=positions.ravel(),
        normals=np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), nx * ny),
        indices=indices,
        vertex_count=nx * ny,
    )
    mesh = MeshInstance(name="grid", geometry=geom)
    mesh.rest_positions = positions.ravel().copy()
    return mesh


def _skinning(chain_count: int = 2) -> SoftTissueSkinning:
    """Two vertical chains of three joints each, straddling the midline."""
    obj = SoftTissueSkinning()
    for chain in range(chain_count):
        x = -10.0 + 20.0 * chain
        for k in range(3):
            z = -30.0 + 30.0 * k
            rest = mat4_identity()
            rest[0, 3] = x
            rest[2, 3] = z
            joint = SkinJoint(
                name=f"c{chain}j{k}",
                node=SceneNode(name=f"c{chain}j{k}"),
                rest_world=rest,
                segment_start=np.array([x, 0.0, z]),
                segment_end=np.array([x, 0.0, z + 15.0]),
                chain_id=chain,
            )
            obj.joints.append(joint)
    obj.chain_count = chain_count
    return obj


@pytest.fixture
def small_cache(tmp_path, monkeypatch):
    """Point the cache at tmp_path and make the grid mesh cache-eligible."""
    monkeypatch.setattr(skinning_cache, "MIN_VERTS", 100)
    monkeypatch.delenv(skinning_cache._ENV_CACHE_OFF, raising=False)
    skinning_cache.set_cache_dir(tmp_path)
    yield tmp_path
    skinning_cache.set_cache_dir(None)


def _solve(mesh):
    return _skinning()._solve_skin_binding(
        mesh, is_muscle=False, allowed_chains={0, 1},
        spatial_limit=25.0, chain_z_margin=15.0,
    )


def test_cache_round_trip_is_bitwise_identical(small_cache):
    mesh = _grid_mesh()

    skinning_cache.set_cache_dir(None)          # solve with no cache at all
    import os
    os.environ[skinning_cache._ENV_CACHE_OFF] = "1"
    try:
        fresh = _solve(mesh)
    finally:
        del os.environ[skinning_cache._ENV_CACHE_OFF]
    skinning_cache.set_cache_dir(small_cache)

    assert fresh is not None
    stored = _solve(mesh)                       # miss: solves and writes
    hit = _solve(mesh)                          # hit: reads
    assert len(list(small_cache.glob("binding.*.npz"))) == 1

    names = ("joint_indices", "secondary_indices", "weights", "edges",
             "influences", "influence_weights")
    assert len(fresh) == len(names), (
        f"the solve returns {len(fresh)} arrays but this test names "
        f"{len(names)}; a new one was added without being covered here"
    )
    for name, a, b, c in zip(names, fresh, stored, hit, strict=True):
        assert np.array_equal(a, b), f"{name}: uncached vs cache-miss differ"
        assert np.array_equal(a, c), f"{name}: uncached vs cache-hit differ"
        assert a.dtype == c.dtype, f"{name}: dtype changed through the cache"

    # The multi-influence arrays are the reason CACHE_VERSION went to 2: a v1
    # entry carries none, and serving one would silently deform with two
    # influences while the solver believes it produced four.
    inf, infw = fresh[4], fresh[5]
    assert inf is not None and infw is not None, \
        "skin solve produced no multi-influence arrays"
    assert inf.shape == infw.shape, f"{inf.shape} vs {infw.shape}"
    assert inf.shape[1] == _skinning().SKIN_INFLUENCES
    np.testing.assert_allclose(
        infw.sum(axis=1), 1.0, atol=1e-6,
        err_msg="influence weights must form a partition of unity per vertex",
    )
    assert (infw >= 0.0).all(), "negative influence weight"


def test_cache_is_skipped_below_min_verts(small_cache, monkeypatch):
    """Small meshes must not pay the hashing cost."""
    monkeypatch.setattr(skinning_cache, "MIN_VERTS", 10_000_000)
    assert _solve(_grid_mesh()) is not None
    assert list(small_cache.glob("binding.*.npz")) == []


def test_cache_can_be_disabled_by_env(small_cache, monkeypatch):
    monkeypatch.setenv(skinning_cache._ENV_CACHE_OFF, "1")
    assert not skinning_cache.enabled()
    assert _solve(_grid_mesh()) is not None
    assert list(small_cache.glob("binding.*.npz")) == []


def test_key_changes_when_geometry_changes(small_cache):
    mesh = _grid_mesh()
    _solve(mesh)
    moved = _grid_mesh()
    moved.rest_positions = moved.rest_positions + np.float32(3.0)
    _solve(moved)
    assert len(list(small_cache.glob("binding.*.npz"))) == 2, (
        "moving the mesh must not reuse the previous binding"
    )


def test_key_changes_when_a_tunable_changes(small_cache):
    """A tunable that alters the solve must invalidate the entry."""
    mesh = _grid_mesh()
    obj = _skinning()
    obj._solve_skin_binding(
        mesh, allowed_chains={0, 1}, spatial_limit=25.0, chain_z_margin=15.0)
    assert len(list(small_cache.glob("binding.*.npz"))) == 1

    other = _skinning()
    other.CROSS_CHAIN_RADIUS = 5.0
    other._solve_skin_binding(
        mesh, allowed_chains={0, 1}, spatial_limit=25.0, chain_z_margin=15.0)
    assert len(list(small_cache.glob("binding.*.npz"))) == 2


def test_corrupt_entry_falls_back_to_solving(small_cache):
    mesh = _grid_mesh()
    fresh = _solve(mesh)
    entry = next(iter(small_cache.glob("binding.*.npz")))
    entry.write_bytes(b"not an npz file")
    again = _solve(mesh)
    assert again is not None
    assert np.array_equal(fresh[0], again[0])


def test_unreadable_cache_dir_is_not_fatal(tmp_path, monkeypatch):
    """A read-only cache location degrades to solving, it does not raise."""
    monkeypatch.setattr(skinning_cache, "MIN_VERTS", 100)
    blocker = tmp_path / "blocked"
    blocker.write_text("I am a file, not a directory")
    skinning_cache.set_cache_dir(blocker / "sub")
    try:
        assert _solve(_grid_mesh()) is not None
    finally:
        skinning_cache.set_cache_dir(None)
