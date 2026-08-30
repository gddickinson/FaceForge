"""Versioned, hash-stamped serialisation of everything a render depends on.

The problem this solves: until now a FaceForge render could not be reproduced
from a saved file.  A figure in a paper could not be regenerated -- not even by
its author, who would have to remember a camera position, a render mode, 39
pose DOFs, 12 Action Unit values, a lighting setup and which structures were
loaded and visible.  A ``SceneState`` is that record, in one diffable JSON file
that can sit in git next to the manuscript.

Quick start
-----------
Save the current scene::

    from faceforge.core import scene_state

    state = scene_state.capture_scene_state(
        scene=scene, camera=gl_widget.camera, lights=gl_widget.lights,
        renderer=gl_widget.renderer, state=state_manager,
    )
    scene_state.save(state, "figure_3a.state.json")

Reproduce it::

    state = scene_state.load("figure_3a.state.json")   # warns on config drift
    report = scene_state.apply_scene_state(
        state, scene=scene, camera=gl_widget.camera, lights=gl_widget.lights,
        renderer=gl_widget.renderer, state_manager=state_manager,
    )
    assert report.ok, report.summary()

Guarantees
----------
* **Exact round trip.**  ``save -> load -> save`` is byte-identical, and the
  reloaded state compares equal to the original.  Floats are written with
  ``float.__repr__`` and read with ``strtod``, so every finite double survives
  unchanged -- there is no rounding anywhere in this package.
* **Versioned.**  A newer ``schema_version`` is refused with a clear message
  rather than partially loaded; an older one is migrated or refused.  See
  :mod:`faceforge.core.scene_state.codec` for the full policy.
* **Config-stamped.**  The anatomy config set and the FMA crosswalk are hashed
  into the file.  Loading against changed configs warns loudly (it does not
  fail: re-rendering against updated configs is legitimate, doing it unknowingly
  is not).
* **Citable.**  Every structure records its BodyParts3D ``source_id``, its FMA
  ``ontology_id`` and the FMA ``preferred_label``, so the contents of a figure
  can be cited in standard terminology rather than app display names.
* **Deterministic.**  Sorted keys, structures sorted by scene-graph path, and
  the only timestamp in the file lives in a ``metadata`` block that is excluded
  from equality and from :func:`canonical_json`.
"""

from faceforge.core.scene_state.binding import (
    ApplyReport,
    apply_animation,
    apply_camera,
    apply_lighting,
    apply_render,
    apply_scene_state,
    capture_camera,
    capture_lighting,
    capture_render,
    capture_scene_state,
    mesh_paths,
    scene_paths,
    structures_missing_provenance,
)
from faceforge.core.scene_state.codec import (
    MIGRATIONS,
    SCHEMA_VERSION,
    canonical_json,
    canonical_payload,
    dumps,
    from_dict,
    load,
    loads,
    new_metadata,
    payload_digest,
    register_migration,
    save,
    to_dict,
    verify_config,
)
from faceforge.core.scene_state.confighash import (
    describe_mismatch,
    fingerprint_configs,
)
from faceforge.core.scene_state.model import (
    AssetState,
    CameraState,
    ClipPlaneState,
    ConfigFingerprint,
    ConfigFingerprintMismatch,
    LightingState,
    MaterialState,
    MorphState,
    PointLightState,
    ProvenanceState,
    RenderState,
    SceneState,
    SceneStateError,
    SceneStateFormatError,
    SceneStateVersionError,
    StructureState,
    ViewportState,
    render_mode_names,
)

__all__ = [
    "SCHEMA_VERSION",
    "MIGRATIONS",
    # model
    "SceneState",
    "CameraState",
    "ViewportState",
    "LightingState",
    "PointLightState",
    "RenderState",
    "ClipPlaneState",
    "StructureState",
    "MaterialState",
    "ProvenanceState",
    "MorphState",
    "AssetState",
    "ConfigFingerprint",
    "SceneStateError",
    "SceneStateFormatError",
    "SceneStateVersionError",
    "ConfigFingerprintMismatch",
    "render_mode_names",
    # codec
    "dumps",
    "loads",
    "save",
    "load",
    "to_dict",
    "from_dict",
    "canonical_json",
    "canonical_payload",
    "payload_digest",
    "new_metadata",
    "register_migration",
    "verify_config",
    # config fingerprint
    "fingerprint_configs",
    "describe_mismatch",
    # binding
    "capture_scene_state",
    "apply_scene_state",
    "ApplyReport",
    "capture_camera",
    "capture_lighting",
    "capture_render",
    "apply_camera",
    "apply_lighting",
    "apply_render",
    "apply_animation",
    "scene_paths",
    "mesh_paths",
    "structures_missing_provenance",
]
