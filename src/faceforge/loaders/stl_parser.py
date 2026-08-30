"""Binary STL parser with indexed geometry support.

Both hot paths here are vectorised: parsing is a single ``np.frombuffer``
over a 50-byte structured dtype, and the duplicate-vertex weld is a
lexsort + bincount instead of a per-vertex dict loop.  ``load_stl_file``
additionally memoises the welded result in an ``.npz`` cache keyed on the
source file's mtime and size, so repeat runs skip both stages.
"""

from __future__ import annotations

import os
import struct
import zipfile
from pathlib import Path

import numpy as np

from faceforge.core.mesh import BufferGeometry

# 50-byte binary-STL triangle record: 3 float32 normal, 3x3 float32 verts,
# uint16 attribute count.  itemsize is exactly 50 with no padding.
_TRI_DTYPE = np.dtype([("normal", "<3f4"), ("verts", "<3,3f4"), ("attr", "<u2")])
assert _TRI_DTYPE.itemsize == 50, "structured dtype must match the 50-byte record"


# ── welded-geometry disk cache ──────────────────────────────────────
#
# Welding a full BodyParts3D mesh set costs seconds of pure CPU that is
# identical on every run, so the indexed result is memoised on disk.  The
# cache lives outside the repo by default (so it neither pollutes the
# working tree nor needs a .gitignore entry) and is keyed on the source
# path plus its (mtime_ns, size) so an edited or replaced STL invalidates
# its entry in either time direction.

CACHE_VERSION = 1
_ENV_CACHE_DIR = "FACEFORGE_MESH_CACHE_DIR"
_ENV_CACHE_OFF = "FACEFORGE_MESH_CACHE_OFF"

_cache_dir_override: Path | None = None


def default_mesh_cache_dir() -> Path:
    """Platform cache location used when no directory is configured."""
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "faceforge" / "meshes"


def set_mesh_cache_dir(path: Path | str | None) -> None:
    """Override the welded-geometry cache directory (None restores default)."""
    global _cache_dir_override
    _cache_dir_override = Path(path) if path is not None else None


def mesh_cache_dir() -> Path:
    """Cache directory currently in effect (override > env var > default)."""
    if _cache_dir_override is not None:
        return _cache_dir_override
    env = os.environ.get(_ENV_CACHE_DIR)
    if env:
        return Path(env)
    return default_mesh_cache_dir()


def mesh_cache_enabled() -> bool:
    """False when FACEFORGE_MESH_CACHE_OFF is set to a truthy value."""
    return os.environ.get(_ENV_CACHE_OFF, "").strip().lower() not in (
        "1", "true", "yes", "on",
    )


def _cache_path(cache_dir: Path, path: Path) -> Path:
    """Cache filename for a source STL: readable stem plus a path digest.

    The digest keeps two same-named STLs in different directories apart.
    """
    import hashlib

    digest = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:12]
    return cache_dir / f"{path.stem}.{digest}.v{CACHE_VERSION}.npz"


def _cache_load(cf: Path, st: os.stat_result) -> BufferGeometry | None:
    """Read a cache entry, or None if absent/stale/unreadable."""
    try:
        with np.load(cf) as z:
            key = z["key"]
            if int(key[0]) != st.st_mtime_ns or int(key[1]) != st.st_size:
                return None
            return BufferGeometry(
                positions=z["p"],
                normals=z["n"],
                indices=z["i"],
                vertex_count=int(key[2]),
            )
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        # Missing, truncated or written by an incompatible version.
        return None


def _cache_store(cf: Path, st: os.stat_result, geom: BufferGeometry) -> None:
    """Write a cache entry atomically; silently give up if the FS says no."""
    # np.savez appends '.npz' unless the name already ends in it, so the
    # temp name must itself end in .npz for the rename to land correctly.
    # The pid keeps concurrent writers from sharing a temp file.
    tmp = cf.with_name(f"{cf.stem}.{os.getpid()}.tmp.npz")
    try:
        cf.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            tmp,
            p=geom.positions,
            n=geom.normals,
            i=geom.indices,
            key=np.array(
                [st.st_mtime_ns, st.st_size, geom.vertex_count], dtype=np.int64
            ),
        )
        # os.replace is atomic within a filesystem, so a process killed
        # mid-write leaves the temp file behind, never a partial entry.
        os.replace(tmp, cf)
    except OSError:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


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

    # One frombuffer instead of a Python loop over triangles: the record
    # layout is fixed, so the whole file is a single strided read.
    rec = np.frombuffer(data, dtype=_TRI_DTYPE, count=tri_count, offset=84)

    # .copy() detaches from the read-only input buffer so the downstream
    # in-place coordinate transform can write to it.
    positions = rec["verts"].reshape(-1, 3).ravel().copy()
    # One normal per triangle, repeated for its 3 vertices.
    normals = np.repeat(rec["normal"], 3, axis=0).ravel()
    vert_count = tri_count * 3

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
    pos = geom.positions.reshape(-1, 3)[: geom.vertex_count]
    nrm = geom.normals.reshape(-1, 3)[: geom.vertex_count]
    n = len(pos)
    if n == 0:
        return BufferGeometry(
            positions=np.empty(0, np.float32),
            normals=np.empty(0, np.float32),
            indices=np.empty(0, np.uint32),
            vertex_count=0,
        )

    # Quantize positions for hashing (same tolerance/semantics as before)
    scale = 1.0 / tolerance
    q = (pos * scale).astype(np.int64)
    # int32 keys halve sort bandwidth; fall back to int64 on overflow.
    if np.abs(q).max() < 2**31 - 1:
        q = q.astype(np.int32)

    # Group identical quantized triples with three integer sorts.  The
    # unique-vertex *ordering* is sorted rather than first-appearance,
    # which is invisible to anything reading positions via the indices.
    order = np.lexsort((q[:, 2], q[:, 1], q[:, 0]))
    sq = q[order]
    is_new = np.empty(n, dtype=bool)
    is_new[0] = True
    np.any(sq[1:] != sq[:-1], axis=1, out=is_new[1:])
    group = np.cumsum(is_new) - 1
    index_remap = np.empty(n, dtype=np.int64)
    index_remap[order] = group
    first_idx = order[is_new]
    n_unique = len(first_idx)

    out_pos = pos[first_idx].astype(np.float32, copy=True)

    # Accumulate then normalize shared normals, as before.  bincount per
    # component is measurably faster than np.add.at, which runs an
    # unbuffered (non-vectorised) ufunc loop.
    out_nrm = np.empty((n_unique, 3), dtype=np.float32)
    for c in range(3):
        out_nrm[:, c] = np.bincount(
            index_remap, weights=nrm[:, c], minlength=n_unique
        )
    lengths = np.maximum(np.linalg.norm(out_nrm, axis=1, keepdims=True), 1e-10)
    out_nrm /= lengths

    return BufferGeometry(
        positions=out_pos.ravel(),
        normals=out_nrm.ravel(),
        indices=index_remap.astype(np.uint32),
        vertex_count=n_unique,
    )


def load_stl_file(
    path: Path,
    indexed: bool = True,
    *,
    use_cache: bool = True,
    cache_dir: Path | str | None = None,
) -> BufferGeometry:
    """Load an STL file and optionally build indexed geometry.

    Args:
        path: Source ``.stl`` file.
        indexed: Weld duplicate vertices into indexed geometry.
        use_cache: Read/write the welded-geometry disk cache.  Only applies
            when ``indexed`` is True — the non-indexed path is a single
            ``frombuffer`` and has nothing worth caching.
        cache_dir: Cache directory override; defaults to
            :func:`mesh_cache_dir`.

    A cold (cache-miss) load produces geometry identical to a run with the
    cache disabled; the cache only stores what the weld already computed.
    """
    path = Path(path)
    caching = indexed and use_cache and mesh_cache_enabled()

    cf = st = None
    if caching:
        try:
            st = path.stat()
            cf = _cache_path(Path(cache_dir) if cache_dir else mesh_cache_dir(), path)
        except OSError:
            caching = False
    if caching:
        hit = _cache_load(cf, st)
        if hit is not None:
            return hit

    with open(path, "rb") as f:
        data = f.read()

    geom = parse_binary_stl(data)
    if indexed:
        geom = build_indexed_geometry(geom)
        if caching:
            _cache_store(cf, st, geom)
    return geom
