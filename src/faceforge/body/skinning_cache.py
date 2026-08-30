"""Disk cache for the soft-tissue binding solve.

Why this exists
---------------
``SoftTissueSkinning.register_skin_mesh`` assigns every vertex of a mesh to
a bone segment.  For the small muscle/organ meshes that is a few
milliseconds, but the full-body skin mesh is 791,729 vertices against 148
joints, and the solve materialises several ``(V, S, 3)`` float64 arrays plus
a Dijkstra pass over ~2.4M mesh edges.  Measured on the reference machine it
costs ~34 s of pure CPU and blocks the main thread, which owns the 16 ms
render timer — a ~2,000 frame freeze the first time the user ticks "Skin".

The solve is deterministic: identical mesh geometry, identical joint rest
configuration and identical tunables always produce identical
``joint_indices`` / ``secondary_indices`` / ``weights``.  So the result is
memoised on disk, exactly as ``faceforge.loaders.stl_parser`` memoises
welded geometry, and every run after the first is a file read.

Design notes
------------
* Keyed on a blake2b digest of *everything the solve reads*: the rest
  positions, the index buffer, the per-joint rest translations / chain IDs /
  bone segments, the call parameters, and every public scalar attribute of
  the skinning object.  Hashing the tunables automatically (rather than from
  an enumerated list) means a future tunable cannot silently serve a stale
  binding.
* Only applied above ``MIN_VERTS``.  Below that the solve is cheaper than
  hashing its inputs, so the hundreds of muscle registrations pay nothing.
* A miss, a corrupt file or a read-only cache directory is never fatal — the
  caller falls back to solving.
"""

from __future__ import annotations

import hashlib
import os
import zipfile
from pathlib import Path

import numpy as np

# Bump when the meaning of a stored array changes.
CACHE_VERSION = 1

# Vertex count above which memoising beats solving.  The full-body skin mesh
# is ~792k vertices; muscle meshes are typically 1k-30k.
MIN_VERTS = 100_000

_ENV_CACHE_DIR = "FACEFORGE_SKIN_CACHE_DIR"
_ENV_CACHE_OFF = "FACEFORGE_SKIN_CACHE_OFF"

_cache_dir_override: Path | None = None


def default_cache_dir() -> Path:
    """Platform cache location used when no directory is configured."""
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "faceforge" / "skinning"


def set_cache_dir(path: Path | str | None) -> None:
    """Override the binding cache directory (None restores the default)."""
    global _cache_dir_override
    _cache_dir_override = Path(path) if path is not None else None


def cache_dir() -> Path:
    """Directory currently in effect (override > env var > default)."""
    if _cache_dir_override is not None:
        return _cache_dir_override
    env = os.environ.get(_ENV_CACHE_DIR)
    if env:
        return Path(env)
    return default_cache_dir()


def enabled() -> bool:
    """False when FACEFORGE_SKIN_CACHE_OFF is set to a truthy value."""
    return os.environ.get(_ENV_CACHE_OFF, "").strip().lower() not in (
        "1", "true", "yes", "on",
    )


def scalar_tunables(obj: object) -> list[tuple[str, str]]:
    """Every public int/float/bool/str attribute of *obj*, sorted by name.

    Used so the cache key covers all solve tunables without an enumerated
    list that could fall behind the code.
    """
    out: list[tuple[str, str]] = []
    for name in sorted(dir(obj)):
        if name.startswith("_"):
            continue
        # A property whose getter raises would otherwise abort key building.
        try:
            value = getattr(obj, name)
        except AttributeError:      # pragma: no cover - defensive
            continue
        if isinstance(value, (int, float, bool, str)):
            out.append((name, repr(value)))
    return out


def binding_key(
    *,
    positions: np.ndarray,
    indices: np.ndarray | None,
    joint_rest: np.ndarray,
    joint_chains: np.ndarray,
    seg_starts: np.ndarray,
    seg_ends: np.ndarray,
    seg_indices: np.ndarray,
    seg_chains: np.ndarray,
    params: tuple,
    tunables: list[tuple[str, str]],
) -> str:
    """Digest of every input the binding solve reads."""
    h = hashlib.blake2b(digest_size=20)
    h.update(f"v{CACHE_VERSION}\n".encode())
    for arr in (positions, joint_rest, joint_chains,
                seg_starts, seg_ends, seg_indices, seg_chains):
        a = np.ascontiguousarray(arr)
        h.update(f"|{a.shape}{a.dtype.str}".encode())
        h.update(a.tobytes())
    if indices is None:
        h.update(b"|noidx")
    else:
        a = np.ascontiguousarray(indices)
        h.update(f"|{a.shape}{a.dtype.str}".encode())
        h.update(a.tobytes())
    h.update(repr(params).encode())
    h.update(repr(tunables).encode())
    return h.hexdigest()


def _path(key: str) -> Path:
    return cache_dir() / f"binding.{key}.v{CACHE_VERSION}.npz"


def load(key: str) -> dict[str, np.ndarray] | None:
    """Read a cached solve, or None if absent/corrupt/unreadable."""
    try:
        with np.load(_path(key)) as z:
            joint_indices = z["ji"]
            secondary_indices = z["si"]
            weights = z["w"]
            edges = z["e"]
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return None
    if not (len(joint_indices) == len(secondary_indices) == len(weights)):
        return None                      # truncated / mismatched entry
    return {
        "joint_indices": joint_indices,
        "secondary_indices": secondary_indices,
        "weights": weights,
        # A zero-row edge array means "the solve had no precomputed edges".
        "edges": edges if edges.size else None,
    }


def store(
    key: str,
    *,
    joint_indices: np.ndarray,
    secondary_indices: np.ndarray,
    weights: np.ndarray,
    edges: np.ndarray | None,
) -> bool:
    """Write a cache entry atomically.  Returns False if the FS says no."""
    target = _path(key)
    # np.savez appends '.npz' unless the name already ends in it, so the temp
    # name must itself end in .npz for the rename to land correctly.  The pid
    # keeps concurrent writers from sharing a temp file.
    tmp = target.with_name(f"{target.stem}.{os.getpid()}.tmp.npz")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            tmp,
            ji=joint_indices,
            si=secondary_indices,
            w=weights,
            e=(np.empty((0, 2), dtype=np.int32) if edges is None
               else np.ascontiguousarray(edges)),
        )
        # os.replace is atomic within a filesystem, so a process killed
        # mid-write leaves a temp file behind, never a partial entry.
        os.replace(tmp, target)
        return True
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:                  # pragma: no cover - defensive
            pass
        return False
