"""Reading and writing SceneState files.

File format
-----------
A single UTF-8 JSON object, keys sorted, two-space indent, one trailing
newline.  Sorted keys and stable sequence order are not cosmetic: a state file
is meant to live in git next to the paper whose figure it reproduces, and a
file whose key order depends on dict iteration is not diffable.

Top-level keys::

    schema_version   int    the format version of this file
    metadata         object non-comparable: when and by what it was written
    config           object anatomy-config fingerprint (see confighash)
    camera           object position / target / up / fov / near / far / aspect
    viewport         object width / height
    lighting         object ambient, directional, optional point light
    render           object global render mode, clear colour, clip plane,
                            scene transform
    structures       array  one entry per mesh, sorted by scene-graph path:
                            visibility, material override, render-mode
                            override, FMA provenance
    face             object every scalar field of FaceState (all 12 AUs, gaze,
                            head rotation, expression name, auto toggles)
    body             object every scalar field of BodyState (39 pose DOFs,
                            physiology settings and flags, gender)
    target_au        object interpolation targets for the AUs
    target_head      object interpolation targets for head rotation
    target_body      object interpolation targets for the body
    morph            object gender actually applied to the morph system,
                            faceGroup alignment
    assets           object tier, skull mode, STL dir, layer visibility

Float exactness
---------------
``json.dumps`` writes a Python float with ``float.__repr__``, which is by
definition the shortest decimal string that reads back as the identical
double; ``json.loads`` parses it with ``strtod``, which is correctly rounded.
So a finite float survives an arbitrary number of save/load cycles exactly --
including ``0.1``, ``1e-8``, ``-0.0`` and the 17-significant-digit values that
come out of a camera orbit.  There is no ``round()``, no ``%.6f`` and no
``str()`` anywhere in this package.  Non-finite floats are rejected at
construction time (see :func:`model.as_float`) rather than written as the
non-standard ``NaN``/``Infinity`` tokens that ``json`` would otherwise emit.

Version policy
--------------
``SCHEMA_VERSION`` is the version this build writes.  On load:

* **equal** -- loaded directly.
* **newer than this build** -- refused with :class:`SceneStateVersionError`.
  A newer file may contain fields whose absence changes the render, and a
  partial load would produce a picture that is not the one the file describes
  while reporting success.  Guessing is the one thing a reproducibility format
  must never do.
* **older** -- migrated forward through the chain in :data:`MIGRATIONS`, one
  version at a time, and then loaded.  If any step in the chain is missing the
  load is refused, naming the version that cannot be upgraded.  As of
  ``SCHEMA_VERSION = 1`` there are no released older versions, so
  :data:`MIGRATIONS` is empty and any ``schema_version < 1`` is refused as
  unsupported; :func:`register_migration` is the hook a future version-2
  format uses (and is what the test suite exercises to prove the chain runs).
* **missing, or not an int** -- refused as malformed.
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Mapping

from faceforge.core.scene_state import confighash
from faceforge.core.scene_state.model import (
    AssetState,
    CameraState,
    ConfigFingerprint,
    ConfigFingerprintMismatch,
    LightingState,
    MorphState,
    RenderState,
    SceneState,
    SceneStateFormatError,
    SceneStateVersionError,
    StructureState,
    ViewportState,
    as_int,
    require_keys,
    validate_dataclass_dict,
)

logger = logging.getLogger(__name__)

#: The format version this build writes.  Bump when a change would make an
#: older reader mis-render a newer file, and register a migration at the same
#: time.
SCHEMA_VERSION = 1

GENERATOR = "faceforge.core.scene_state"

#: from_version -> callable(payload_dict) -> payload_dict at from_version + 1.
#: Migrations receive and return the raw JSON payload, before model validation,
#: so a migration can add, rename or drop keys freely.
MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {}

_TOP_KEYS = (
    "schema_version", "metadata", "config", "camera", "viewport", "lighting",
    "render", "structures", "face", "body", "target_au", "target_head",
    "target_body", "morph", "assets",
)


def register_migration(
    from_version: int, fn: Callable[[dict[str, Any]], dict[str, Any]]
) -> None:
    """Register an upgrade from *from_version* to ``from_version + 1``."""
    if from_version in MIGRATIONS:
        raise ValueError(f"a migration from version {from_version} is already registered")
    MIGRATIONS[from_version] = fn


def new_metadata(**extra: Any) -> dict[str, Any]:
    """Build a fresh non-comparable metadata block.

    Called by ``capture``, never by ``save``: regenerating the timestamp on
    every write would mean save -> load -> save is not byte-identical, and the
    exact-round-trip guarantee is the whole point.
    """
    md: dict[str, Any] = {
        "generator": GENERATOR,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "comparable": False,
    }
    md.update(extra)
    return md


# ---------------------------------------------------------------------------
# dict <-> SceneState
# ---------------------------------------------------------------------------


def to_dict(state: SceneState) -> dict[str, Any]:
    """Full JSON-ready payload, including ``schema_version`` and metadata."""
    from faceforge.core.state import BodyState, FaceState, TargetAU, TargetHead

    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": dict(state.metadata),
        "config": state.config.to_dict(),
        "camera": state.camera.to_dict(),
        "viewport": state.viewport.to_dict(),
        "lighting": state.lighting.to_dict(),
        "render": state.render.to_dict(),
        "structures": [s.to_dict() for s in state.structures],
        "face": validate_dataclass_dict(FaceState, state.face, what="face"),
        "body": validate_dataclass_dict(BodyState, state.body, what="body"),
        "target_au": validate_dataclass_dict(TargetAU, state.target_au, what="target_au"),
        "target_head": validate_dataclass_dict(
            TargetHead, state.target_head, what="target_head"
        ),
        "target_body": validate_dataclass_dict(
            BodyState, state.target_body, what="target_body"
        ),
        "morph": state.morph.to_dict(),
        "assets": state.assets.to_dict(),
    }


def from_dict(payload: Mapping[str, Any]) -> SceneState:
    """Build a :class:`SceneState` from an already-version-checked payload."""
    from faceforge.core.state import BodyState, FaceState, TargetAU, TargetHead

    require_keys(payload, _TOP_KEYS, what="state")
    md = payload["metadata"]
    if not isinstance(md, Mapping):
        raise SceneStateFormatError(
            f"state.metadata: expected an object, got {type(md).__name__}"
        )
    structures = payload["structures"]
    if not isinstance(structures, list):
        raise SceneStateFormatError(
            f"state.structures: expected an array, got {type(structures).__name__}"
        )
    return SceneState(
        camera=CameraState.from_dict(payload["camera"]),
        viewport=ViewportState.from_dict(payload["viewport"]),
        lighting=LightingState.from_dict(payload["lighting"]),
        render=RenderState.from_dict(payload["render"]),
        structures=tuple(
            StructureState.from_dict(s, what=f"structures[{i}]")
            for i, s in enumerate(structures)
        ),
        face=validate_dataclass_dict(FaceState, payload["face"], what="face"),
        body=validate_dataclass_dict(BodyState, payload["body"], what="body"),
        target_au=validate_dataclass_dict(TargetAU, payload["target_au"], what="target_au"),
        target_head=validate_dataclass_dict(
            TargetHead, payload["target_head"], what="target_head"
        ),
        target_body=validate_dataclass_dict(
            BodyState, payload["target_body"], what="target_body"
        ),
        morph=MorphState.from_dict(payload["morph"]),
        assets=AssetState.from_dict(payload["assets"]),
        config=ConfigFingerprint.from_dict(payload["config"]),
        metadata=dict(md),
    )


# ---------------------------------------------------------------------------
# Canonical form: the comparable payload
# ---------------------------------------------------------------------------


def canonical_payload(state: SceneState) -> dict[str, Any]:
    """:func:`to_dict` minus the non-comparable ``metadata`` block."""
    d = to_dict(state)
    d.pop("metadata")
    return d


def _dump_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"


def canonical_json(state: SceneState) -> str:
    """The comparable payload as canonical JSON text.

    Two states describing the same render produce identical text here even if
    they were written on different days, which is what makes this the right
    thing to hash or to assert equal.
    """
    return _dump_json(canonical_payload(state))


def payload_digest(state: SceneState) -> str:
    """sha256 of :func:`canonical_json` -- a one-line identity for a render."""
    import hashlib

    return hashlib.sha256(canonical_json(state).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Text and file I/O
# ---------------------------------------------------------------------------


def dumps(state: SceneState) -> str:
    """Serialise *state* to canonical JSON text, metadata included."""
    return _dump_json(to_dict(state))


def loads(
    text: str,
    *,
    check_config: bool = True,
    config_root: Path | str | None = None,
    source: str = "<string>",
) -> SceneState:
    """Parse a state file, enforcing the version policy and config check."""
    try:
        payload = json.loads(text)
    except ValueError as exc:
        raise SceneStateFormatError(f"{source}: not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SceneStateFormatError(
            f"{source}: expected a JSON object at the top level, got "
            f"{type(payload).__name__}"
        )
    payload = _migrate(payload, source=source)
    state = from_dict(payload)
    if check_config:
        verify_config(state, config_root=config_root, source=source)
    return state


def _migrate(payload: dict[str, Any], *, source: str) -> dict[str, Any]:
    """Apply the version policy documented in this module's docstring."""
    if "schema_version" not in payload:
        raise SceneStateFormatError(
            f"{source}: no schema_version.  Every SceneState file must declare its "
            "format version; a file without one cannot be safely interpreted."
        )
    version = payload["schema_version"]
    try:
        version = as_int(version, what=f"{source}: schema_version")
    except SceneStateFormatError as exc:
        raise SceneStateFormatError(str(exc)) from exc

    if version == SCHEMA_VERSION:
        return payload
    if version > SCHEMA_VERSION:
        raise SceneStateVersionError(
            f"{source}: schema_version {version} is newer than this build "
            f"supports (max {SCHEMA_VERSION}).  Refusing to load: a newer file may "
            "contain fields that change the render, and loading it partially would "
            "silently produce a different picture from the one it describes.  "
            "Update FaceForge, or re-capture the state with this build."
        )

    current = version
    while current < SCHEMA_VERSION:
        fn = MIGRATIONS.get(current)
        if fn is None:
            raise SceneStateVersionError(
                f"{source}: schema_version {version} cannot be upgraded -- no "
                f"migration is registered from version {current} to {current + 1} "
                f"(this build writes version {SCHEMA_VERSION}).  Refusing to load "
                "rather than guess at the missing fields."
            )
        payload = fn(payload)
        if not isinstance(payload, dict):
            raise SceneStateFormatError(
                f"{source}: migration {current} -> {current + 1} returned "
                f"{type(payload).__name__}, not a payload dict"
            )
        current += 1
        payload["schema_version"] = current
        logger.info("%s: migrated state schema %d -> %d", source, current - 1, current)
    return payload


def verify_config(
    state: SceneState,
    *,
    config_root: Path | str | None = None,
    source: str = "<string>",
) -> str | None:
    """Compare the state's config fingerprint against the configs on disk.

    Returns ``None`` when they match, otherwise the mismatch description --
    after emitting it both as a :class:`ConfigFingerprintMismatch` warning and
    at ``logging.WARNING``.  Two channels on purpose: the warning is what a
    test or a ``-W error`` run can trap, and the log line is what a user
    running the app actually sees.
    """
    current = confighash.fingerprint_configs(config_root)
    detail = confighash.describe_mismatch(state.config, current)
    if detail is None:
        return None
    message = (
        f"{source}: anatomy configs have changed since this state was captured "
        f"({detail}).  The render this state file describes was produced against "
        "different configs, so re-rendering it now may not reproduce that figure. "
        "Re-capture the state, or check out the configs it was captured against."
    )
    logger.warning("%s", message)
    warnings.warn(message, ConfigFingerprintMismatch, stacklevel=3)
    return detail


def save(state: SceneState, path: Path | str) -> Path:
    """Write *state* to *path*.  Returns the path written.

    Written to a sibling ``.tmp`` file and moved into place, so an interrupted
    save cannot leave a truncated state file that later loads as a subtly
    different scene.
    """
    path = Path(path)
    text = dumps(state)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


def load(
    path: Path | str,
    *,
    check_config: bool = True,
    config_root: Path | str | None = None,
) -> SceneState:
    """Read a state file from *path*."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SceneStateFormatError(f"no such state file: {path}") from exc
    return loads(
        text, check_config=check_config, config_root=config_root, source=str(path)
    )
