"""Fingerprinting the anatomy config set.

A SceneState is only a reproducibility claim if the inputs it does *not*
contain are pinned.  The configs under ``assets/config`` are exactly those
inputs: they decide which STL file is loaded under which display name, its
default colour, opacity and shininess, which structures belong to which layer
toggle, joint limits, gender-dimorphism bone scales, and -- through
``fma_labels.json`` -- what each structure is called in the FMA.  Change any of
them and the same state file renders a different picture.

So the fingerprint is recorded on capture and checked on load.  A mismatch is a
loud warning, never an error: re-rendering a stored state against updated
configs is a legitimate thing to do, and it is the *silent* version of it that
destroys the value of the file.

What is hashed
--------------
Every ``*.json`` under ``assets/config`` (recursively), by content, keyed by
POSIX-style relative path so the digest is identical on macOS and Linux and
independent of where the repo is checked out.  Nothing else: STL geometry is
identified per structure by ``source_id`` in the provenance block, and hashing
934 meshes on every save would cost seconds for no added protection.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from faceforge.core.scene_state.model import ConfigFingerprint, SceneStateError

logger = logging.getLogger(__name__)

CONFIG_GLOB = "*.json"
FMA_LABELS_NAME = "fma_labels.json"

#: (root, stat-signature) -> fingerprint.  Recomputing the digest costs a read
#: of ~500 KB across ~50 files; a capture-heavy loop (the golden-image harness
#: captures 16 states in a row) would otherwise pay it 16 times.  The key
#: includes every file's size and mtime_ns, so an edit invalidates the entry --
#: a plain path key would hand back a stale digest and defeat the check.
_CACHE: dict[tuple[str, tuple], ConfigFingerprint] = {}


def default_config_root() -> Path:
    from faceforge.constants import CONFIG_DIR

    return Path(CONFIG_DIR)


def _config_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob(CONFIG_GLOB) if p.is_file())


def _stat_signature(root: Path, files: list[Path]) -> tuple:
    return tuple(
        (p.relative_to(root).as_posix(), st.st_size, st.st_mtime_ns)
        for p, st in ((p, p.stat()) for p in files)
    )


def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint_configs(root: Path | str | None = None, *, use_cache: bool = True) -> ConfigFingerprint:
    """Hash the anatomy config set rooted at *root* (default ``assets/config``).

    The combined digest is over ``"<relpath>\\n<file-sha256>\\n"`` for each file
    in sorted relative-path order.  Including the path means a rename is a
    change (it is: configs are looked up by name), and the trailing newlines
    make the concatenation unambiguous, so no pair of distinct file sets can
    collide by string-boundary accident.
    """
    root = Path(root) if root is not None else default_config_root()
    if not root.is_dir():
        raise SceneStateError(
            f"config root {root} is not a directory; cannot fingerprint the config set"
        )
    files = _config_files(root)
    sig = _stat_signature(root, files)
    key = (str(root.resolve()), sig)
    if use_cache and key in _CACHE:
        return _CACHE[key]

    combined = hashlib.sha256()
    fma_digest = ""
    for path in files:
        rel = path.relative_to(root).as_posix()
        d = _file_digest(path)
        combined.update(rel.encode("utf-8"))
        combined.update(b"\n")
        combined.update(d.encode("ascii"))
        combined.update(b"\n")
        if rel == FMA_LABELS_NAME:
            fma_digest = d

    fp = ConfigFingerprint(
        algorithm="sha256",
        digest=combined.hexdigest(),
        file_count=len(files),
        # Recorded as the directory name only.  An absolute path would make the
        # digest block differ between two checkouts of the same commit and turn
        # every state file into a machine-specific artefact.
        root=root.name,
        fma_labels_digest=fma_digest,
    )
    if use_cache:
        _CACHE[key] = fp
    return fp


def clear_cache() -> None:
    """Drop the fingerprint cache.  Used by tests that rewrite config files."""
    _CACHE.clear()


def describe_mismatch(stored: ConfigFingerprint, current: ConfigFingerprint) -> str | None:
    """Human-readable description of how two fingerprints differ, or None.

    Kept separate from the warning site so the same text can be reused by a CLI
    or a report generator without triggering a warning.
    """
    if stored == current:
        return None
    if not stored.digest:
        return (
            "this state file carries no config fingerprint, so there is no way to "
            "tell whether the configs it was captured against match the ones "
            "loaded now"
        )
    parts = []
    if stored.algorithm != current.algorithm:
        parts.append(f"algorithm {stored.algorithm!r} -> {current.algorithm!r}")
    if stored.file_count != current.file_count:
        parts.append(f"file count {stored.file_count} -> {current.file_count}")
    if stored.root != current.root:
        parts.append(f"config root {stored.root!r} -> {current.root!r}")
    if stored.digest != current.digest:
        parts.append(f"config digest {stored.digest[:12]}… -> {current.digest[:12]}…")
    if stored.fma_labels_digest != current.fma_labels_digest:
        parts.append(
            f"{FMA_LABELS_NAME} digest {stored.fma_labels_digest[:12] or '(none)'}… -> "
            f"{current.fma_labels_digest[:12] or '(none)'}…"
        )
    return "; ".join(parts) if parts else "fingerprints differ in an unclassified field"
