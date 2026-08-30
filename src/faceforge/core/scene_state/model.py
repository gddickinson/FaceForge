"""The SceneState data model: everything needed to reproduce a render.

Pure data.  This module imports nothing from ``faceforge.rendering`` and
touches no GL, so a state file can be read, diffed, validated and written by a
script, a test or a CI job that has no graphics context at all.  The adapter
that reads these values off live ``Camera`` / ``LightSetup`` / ``Scene`` /
``GLRenderer`` / ``StateManager`` objects and writes them back lives in
:mod:`faceforge.core.scene_state.binding`.

Design rules that the rest of the package depends on
----------------------------------------------------
* **Every leaf dataclass here is frozen.**  A state is a record of one render,
  not a mutable scratchpad; freezing means a captured state cannot drift
  between capture and save, and it gives structural equality for free.
* **No numpy in the model.**  Vectors are plain ``tuple[float, ...]``.  numpy
  arrays are converted at the boundary (:func:`vec_tuple`).  A model holding
  ``np.ndarray`` would make ``==`` return an array rather than a bool, which
  would silently turn every equality assertion in the test suite into a
  truth-value error or, worse, a pass.
* **Floats are validated, not rounded.**  :func:`as_float` rejects NaN and
  infinity (a non-finite camera position is a bug, not a state worth
  recording) and otherwise passes the value through untouched.  Rounding
  happens nowhere in this package: ``json`` writes ``float.__repr__``, which
  is the shortest string that round-trips to the identical double.
* **Ordering is canonical.**  Sequences that have no meaningful order
  (structures, layer names) are sorted at construction, so two captures of
  the same scene cannot differ by dict iteration order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Any, Iterable, Mapping, Sequence

# ---------------------------------------------------------------------------
# Errors and warnings.  Defined here (the dependency-free module) so that both
# the codec and the binding layer can raise them without a circular import.
# ---------------------------------------------------------------------------


class SceneStateError(Exception):
    """Base class for every SceneState failure."""


class SceneStateFormatError(SceneStateError):
    """A state file, or a value being put into one, is malformed."""


class SceneStateVersionError(SceneStateError):
    """A state file's ``schema_version`` cannot be handled by this build."""


class ConfigFingerprintMismatch(UserWarning):
    """The anatomy configs differ from those the state was captured against.

    A warning, never an error: re-rendering an old state against updated
    configs is a legitimate thing to want to do.  But it must be impossible to
    do *by accident*, because the render it produces is not the render the
    state describes.
    """


# ---------------------------------------------------------------------------
# Scalar / vector coercion
# ---------------------------------------------------------------------------


def as_float(value: Any, *, what: str = "value") -> float:
    """Return *value* as a finite Python float, or raise.

    Accepts anything ``float()`` accepts, including ``np.float32``.  Note that
    ``float(np.float32(0.1))`` is ``0.10000000149011612`` -- the exact double
    the float32 holds.  That is deliberate: the value is preserved bit-for-bit
    rather than being prettified back to ``0.1``, which would be a different
    number from the one that was rendered.
    """
    if isinstance(value, bool):
        raise SceneStateFormatError(f"{what}: expected a float, got bool {value!r}")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise SceneStateFormatError(
            f"{what}: expected a float, got {type(value).__name__} {value!r}"
        ) from exc
    if not math.isfinite(out):
        raise SceneStateFormatError(
            f"{what}: {out!r} is not finite.  A state file must describe a render "
            "that can actually be produced; NaN and infinity cannot be."
        )
    return out


def as_bool(value: Any, *, what: str = "value") -> bool:
    if not isinstance(value, bool):
        raise SceneStateFormatError(
            f"{what}: expected a bool, got {type(value).__name__} {value!r}"
        )
    return value


def as_int(value: Any, *, what: str = "value") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SceneStateFormatError(
            f"{what}: expected an int, got {type(value).__name__} {value!r}"
        )
    return value


def as_str(value: Any, *, what: str = "value") -> str:
    if not isinstance(value, str):
        raise SceneStateFormatError(
            f"{what}: expected a str, got {type(value).__name__} {value!r}"
        )
    return value


def vec_tuple(value: Any, length: int, *, what: str = "vector") -> tuple[float, ...]:
    """Convert a sequence (list, tuple, ``np.ndarray``) to a float tuple."""
    if isinstance(value, (str, bytes)):
        raise SceneStateFormatError(f"{what}: expected a numeric sequence, got {value!r}")
    try:
        items = list(value)
    except TypeError as exc:
        raise SceneStateFormatError(
            f"{what}: expected a sequence of {length} numbers, got "
            f"{type(value).__name__} {value!r}"
        ) from exc
    if len(items) != length:
        raise SceneStateFormatError(
            f"{what}: expected {length} components, got {len(items)}"
        )
    return tuple(as_float(v, what=f"{what}[{i}]") for i, v in enumerate(items))


def mat4_tuple(value: Any, *, what: str = "matrix") -> tuple[tuple[float, ...], ...]:
    """Convert a 4x4 matrix (nested sequence or ``np.ndarray``) to nested tuples."""
    try:
        rows = [list(r) for r in value]
    except TypeError as exc:
        raise SceneStateFormatError(f"{what}: expected a 4x4 matrix, got {value!r}") from exc
    if len(rows) != 4:
        raise SceneStateFormatError(f"{what}: expected 4 rows, got {len(rows)}")
    return tuple(vec_tuple(r, 4, what=f"{what}[{i}]") for i, r in enumerate(rows))


# ---------------------------------------------------------------------------
# Typed dumping of the app's own animation-state dataclasses
# ---------------------------------------------------------------------------

#: Coercion per declared field kind.
_KIND_COERCE = {"bool": as_bool, "int": as_int, "float": as_float, "str": as_str}


def field_kinds(cls: type) -> dict[str, str]:
    """Map field name -> {"bool","int","float","str"} from the dataclass defaults.

    Keyed off the *default value*, not the annotation.  Annotations may be
    strings (under ``from __future__ import annotations``) or typing
    constructs; the defaults on ``FaceState`` / ``BodyState`` are concrete
    literals and are therefore an unambiguous, import-order-independent source
    of truth.

    Why this matters for exactness: a UI that assigns ``state.body.knee_r_flex
    = 1`` puts an *int* on a float field.  Without a declared kind the value
    would serialise as ``1``, and a state that had been through a save/load
    cycle would then serialise as ``1.0`` -- not byte-identical.  Knowing the
    field is a float makes both passes write ``1.0``.

    Fields whose default is not a scalar (``ConstraintState.attachments``, the
    private ``_JS_KEY_MAP``) are omitted: those are solver scratch, not scene
    description, and putting mutable per-frame junk into a file meant to be
    diffed in git would defeat the point.
    """
    kinds: dict[str, str] = {}
    for f in fields(cls):
        if f.name.startswith("_"):
            continue
        d = f.default
        if isinstance(d, bool):
            kinds[f.name] = "bool"
        elif isinstance(d, int):
            kinds[f.name] = "int"
        elif isinstance(d, float):
            kinds[f.name] = "float"
        elif isinstance(d, str):
            kinds[f.name] = "str"
    return kinds


def dump_dataclass(obj: Any) -> dict[str, Any]:
    """Serialise a flat state dataclass (FaceState, BodyState, ...) to a dict."""
    kinds = field_kinds(type(obj))
    out: dict[str, Any] = {}
    for name, kind in kinds.items():
        value = getattr(obj, name)
        where = f"{type(obj).__name__}.{name}"
        if kind == "float":
            out[name] = as_float(value, what=where)
        elif kind == "bool":
            # bool is the only kind that tolerates a widened runtime value:
            # `auto_blink = 1` is what a JSON preset or a Qt checkbox produces.
            out[name] = bool(value)
        elif kind == "int":
            out[name] = as_int(value, what=where)
        else:
            out[name] = as_str(value, what=where)
    return out


def validate_dataclass_dict(
    cls: type, data: Any, *, what: str
) -> dict[str, Any]:
    """Validate a dict against *cls*'s field kinds.  Unknown keys are an error.

    Strict on purpose.  A silently-dropped key is exactly how a state file
    starts rendering differently from the render it claims to describe -- the
    failure mode this whole format exists to prevent.
    """
    if not isinstance(data, Mapping):
        raise SceneStateFormatError(f"{what}: expected an object, got {type(data).__name__}")
    kinds = field_kinds(cls)
    unknown = sorted(set(data) - set(kinds))
    if unknown:
        raise SceneStateFormatError(
            f"{what}: unknown field(s) {unknown} for {cls.__name__}.  "
            f"Known fields: {sorted(kinds)}"
        )
    missing = sorted(set(kinds) - set(data))
    if missing:
        raise SceneStateFormatError(f"{what}: missing field(s) {missing} for {cls.__name__}")
    return {
        name: _KIND_COERCE[kind](data[name], what=f"{what}.{name}")
        for name, kind in kinds.items()
    }


def apply_dataclass_dict(obj: Any, data: Mapping[str, Any]) -> None:
    """Write a validated dict back onto a live state dataclass instance."""
    for name in field_kinds(type(obj)):
        if name in data:
            setattr(obj, name, data[name])


# ---------------------------------------------------------------------------
# Camera / viewport
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraState:
    """Everything ``Camera`` needs to reproduce its view and projection.

    ``aspect`` is stored even though it is derivable from the viewport, because
    ``Camera.set_aspect`` is driven by widget resize events and a state may
    legitimately have been captured with an aspect that does not match the
    viewport it is replayed into.  Recording both makes that visible instead of
    silently reframing the shot.
    """

    position: tuple[float, ...] = (0.0, 0.0, 0.0)
    target: tuple[float, ...] = (0.0, 0.0, 0.0)
    up: tuple[float, ...] = (0.0, 0.0, 1.0)
    fov_deg: float = 50.0
    near: float = 0.1
    far: float = 1000.0
    aspect: float = 1.0

    _KEYS = ("position", "target", "up", "fov_deg", "near", "far", "aspect")

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": list(self.position),
            "target": list(self.target),
            "up": list(self.up),
            "fov_deg": self.fov_deg,
            "near": self.near,
            "far": self.far,
            "aspect": self.aspect,
        }

    @classmethod
    def from_dict(cls, d: Any, *, what: str = "camera") -> CameraState:
        require_keys(d, cls._KEYS, what=what)
        return cls(
            position=vec_tuple(d["position"], 3, what=f"{what}.position"),
            target=vec_tuple(d["target"], 3, what=f"{what}.target"),
            up=vec_tuple(d["up"], 3, what=f"{what}.up"),
            fov_deg=as_float(d["fov_deg"], what=f"{what}.fov_deg"),
            near=as_float(d["near"], what=f"{what}.near"),
            far=as_float(d["far"], what=f"{what}.far"),
            aspect=as_float(d["aspect"], what=f"{what}.aspect"),
        )


@dataclass(frozen=True)
class ViewportState:
    width: int = 1
    height: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {"width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, d: Any, *, what: str = "viewport") -> ViewportState:
        require_keys(d, ("width", "height"), what=what)
        w = as_int(d["width"], what=f"{what}.width")
        h = as_int(d["height"], what=f"{what}.height")
        if w < 1 or h < 1:
            raise SceneStateFormatError(f"{what}: {w}x{h} is not a renderable size")
        return cls(width=w, height=h)


# ---------------------------------------------------------------------------
# Lighting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PointLightState:
    position: tuple[float, ...] = (0.0, 0.0, 0.0)
    color: tuple[float, ...] = (1.0, 0.95, 0.85)
    intensity: float = 1.5
    range: float = 400.0
    enabled: bool = False

    _KEYS = ("position", "color", "intensity", "range", "enabled")

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": list(self.position),
            "color": list(self.color),
            "intensity": self.intensity,
            "range": self.range,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, d: Any, *, what: str = "point_light") -> PointLightState:
        require_keys(d, cls._KEYS, what=what)
        return cls(
            position=vec_tuple(d["position"], 3, what=f"{what}.position"),
            color=vec_tuple(d["color"], 3, what=f"{what}.color"),
            intensity=as_float(d["intensity"], what=f"{what}.intensity"),
            range=as_float(d["range"], what=f"{what}.range"),
            enabled=as_bool(d["enabled"], what=f"{what}.enabled"),
        )


@dataclass(frozen=True)
class LightingState:
    ambient_color: tuple[float, ...] = (0.4, 0.4, 0.45)
    light_dir: tuple[float, ...] = (0.0, 0.0, 1.0)
    light_color: tuple[float, ...] = (0.8, 0.8, 0.75)
    point_light: PointLightState | None = None

    _KEYS = ("ambient_color", "light_dir", "light_color", "point_light")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ambient_color": list(self.ambient_color),
            "light_dir": list(self.light_dir),
            "light_color": list(self.light_color),
            "point_light": None if self.point_light is None else self.point_light.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Any, *, what: str = "lighting") -> LightingState:
        require_keys(d, cls._KEYS, what=what)
        pl = d["point_light"]
        return cls(
            ambient_color=vec_tuple(d["ambient_color"], 3, what=f"{what}.ambient_color"),
            light_dir=vec_tuple(d["light_dir"], 3, what=f"{what}.light_dir"),
            light_color=vec_tuple(d["light_color"], 3, what=f"{what}.light_color"),
            point_light=None if pl is None
            else PointLightState.from_dict(pl, what=f"{what}.point_light"),
        )


# ---------------------------------------------------------------------------
# Global render state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClipPlaneState:
    enabled: bool = False
    normal: tuple[float, ...] = (1.0, 0.0, 0.0)
    offset: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "normal": list(self.normal), "offset": self.offset}

    @classmethod
    def from_dict(cls, d: Any, *, what: str = "clip_plane") -> ClipPlaneState:
        require_keys(d, ("enabled", "normal", "offset"), what=what)
        return cls(
            enabled=as_bool(d["enabled"], what=f"{what}.enabled"),
            normal=vec_tuple(d["normal"], 3, what=f"{what}.normal"),
            offset=as_float(d["offset"], what=f"{what}.offset"),
        )


@dataclass(frozen=True)
class RenderState:
    """Renderer-global settings: mode, background, cutaway, scene transform.

    ``global_mode`` is the render mode that the majority of structures use.
    Per-structure deviations are recorded as overrides on
    :class:`StructureState`, so the common case (one mode for the whole scene,
    which is what the Display tab produces) costs one line in the file rather
    than one line per structure -- while a hand-mixed scene still round-trips
    exactly.
    """

    global_mode: str = "SOLID"
    clear_color: tuple[float, ...] = (0.12, 0.12, 0.15, 1.0)
    clip_plane: ClipPlaneState = field(default_factory=ClipPlaneState)
    scene_transform: tuple[tuple[float, ...], ...] | None = None

    _KEYS = ("global_mode", "clear_color", "clip_plane", "scene_transform")

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_mode": self.global_mode,
            "clear_color": list(self.clear_color),
            "clip_plane": self.clip_plane.to_dict(),
            "scene_transform": None if self.scene_transform is None
            else [list(r) for r in self.scene_transform],
        }

    @classmethod
    def from_dict(cls, d: Any, *, what: str = "render") -> RenderState:
        require_keys(d, cls._KEYS, what=what)
        st = d["scene_transform"]
        return cls(
            global_mode=validate_render_mode(d["global_mode"], what=f"{what}.global_mode"),
            clear_color=vec_tuple(d["clear_color"], 4, what=f"{what}.clear_color"),
            clip_plane=ClipPlaneState.from_dict(d["clip_plane"], what=f"{what}.clip_plane"),
            scene_transform=None if st is None
            else mat4_tuple(st, what=f"{what}.scene_transform"),
        )


def render_mode_names() -> tuple[str, ...]:
    """The valid ``RenderMode`` names, read from the enum at call time.

    Imported lazily so this module has no import-time dependency beyond the
    standard library; that is what lets a state file be validated by a tool
    that never touches the rest of the app.
    """
    from faceforge.core.material import RenderMode

    return tuple(m.name for m in RenderMode)


def validate_render_mode(value: Any, *, what: str = "render_mode") -> str:
    name = as_str(value, what=what)
    valid = render_mode_names()
    if name not in valid:
        raise SceneStateFormatError(
            f"{what}: {name!r} is not a RenderMode.  Valid modes: {list(valid)}"
        )
    return name


# ---------------------------------------------------------------------------
# Per-structure state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaterialState:
    """The subset of :class:`faceforge.core.material.Material` that is scene state.

    ``render_mode`` is excluded here -- it lives on :class:`StructureState` so
    that the global-mode-plus-overrides compression can see it.
    ``Material.visible`` is also excluded: ``MeshInstance.visible`` is the flag
    the renderer actually consults, and recording both would make it ambiguous
    which one a loader should write.
    """

    color: tuple[float, ...] = (0.8, 0.8, 0.8)
    opacity: float = 1.0
    shininess: float = 30.0
    emissive: tuple[float, ...] = (0.0, 0.0, 0.0)
    double_sided: bool = False
    transparent: bool = False
    depth_write: bool = True
    wireframe_color: tuple[float, ...] | None = None
    vertex_colors_active: bool = False

    _KEYS = (
        "color", "opacity", "shininess", "emissive", "double_sided",
        "transparent", "depth_write", "wireframe_color", "vertex_colors_active",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "color": list(self.color),
            "opacity": self.opacity,
            "shininess": self.shininess,
            "emissive": list(self.emissive),
            "double_sided": self.double_sided,
            "transparent": self.transparent,
            "depth_write": self.depth_write,
            "wireframe_color": None if self.wireframe_color is None
            else list(self.wireframe_color),
            "vertex_colors_active": self.vertex_colors_active,
        }

    @classmethod
    def from_dict(cls, d: Any, *, what: str = "material") -> MaterialState:
        require_keys(d, cls._KEYS, what=what)
        wc = d["wireframe_color"]
        return cls(
            color=vec_tuple(d["color"], 3, what=f"{what}.color"),
            opacity=as_float(d["opacity"], what=f"{what}.opacity"),
            shininess=as_float(d["shininess"], what=f"{what}.shininess"),
            emissive=vec_tuple(d["emissive"], 3, what=f"{what}.emissive"),
            double_sided=as_bool(d["double_sided"], what=f"{what}.double_sided"),
            transparent=as_bool(d["transparent"], what=f"{what}.transparent"),
            depth_write=as_bool(d["depth_write"], what=f"{what}.depth_write"),
            wireframe_color=None if wc is None
            else vec_tuple(wc, 3, what=f"{what}.wireframe_color"),
            vertex_colors_active=as_bool(
                d["vertex_colors_active"], what=f"{what}.vertex_colors_active"
            ),
        )


@dataclass(frozen=True)
class ProvenanceState:
    """Anatomical identity of a rendered structure, straight off MeshInstance.

    ``source_id`` is the BodyParts3D mesh id ("FMA52748"); ``ontology_id`` is
    the canonical term ("FMA:52748"); ``preferred_label`` is the FMA preferred
    term, which differs from the display name for the great majority of
    configured structures.  All three are empty for procedural geometry (the
    scan plane, environment meshes, generated eyes), which has no anatomical
    referent and must not be given a fake one.
    """

    source_id: str = ""
    ontology_id: str = ""
    preferred_label: str = ""

    _KEYS = ("source_id", "ontology_id", "preferred_label")

    @property
    def is_anatomical(self) -> bool:
        return bool(self.source_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "ontology_id": self.ontology_id,
            "preferred_label": self.preferred_label,
        }

    @classmethod
    def from_dict(cls, d: Any, *, what: str = "provenance") -> ProvenanceState:
        require_keys(d, cls._KEYS, what=what)
        return cls(
            source_id=as_str(d["source_id"], what=f"{what}.source_id"),
            ontology_id=as_str(d["ontology_id"], what=f"{what}.ontology_id"),
            preferred_label=as_str(d["preferred_label"], what=f"{what}.preferred_label"),
        )


@dataclass(frozen=True)
class StructureState:
    """One mesh in the scene, addressed by its scene-graph path.

    ``path`` is the identity used to match a state back onto a scene.  Mesh
    *names* are not unique across the loaded set, and ``id()`` is not stable
    across processes, so the path -- root-to-node names, with an index suffix
    on same-named siblings -- is the only stable addressable key the scene
    graph offers.

    ``visible`` is ``MeshInstance.visible``; ``node_visible`` is the owning
    ``SceneNode.visible``.  Both are recorded because the app drives them from
    different places (layer toggles set the node, per-structure toggles set the
    mesh) and a render depends on both.

    ``render_mode`` of ``None`` means "inherit ``RenderState.global_mode``".
    """

    path: str
    name: str = ""
    visible: bool = True
    node_visible: bool = True
    render_mode: str | None = None
    material: MaterialState = field(default_factory=MaterialState)
    provenance: ProvenanceState = field(default_factory=ProvenanceState)

    _KEYS = ("path", "name", "visible", "node_visible", "material", "provenance")

    def effective_render_mode(self, global_mode: str) -> str:
        return global_mode if self.render_mode is None else self.render_mode

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "path": self.path,
            "name": self.name,
            "visible": self.visible,
            "node_visible": self.node_visible,
            "material": self.material.to_dict(),
            "provenance": self.provenance.to_dict(),
        }
        # Omitted entirely when inherited: an explicit null would read as "no
        # mode" rather than "the global one", and the point of the override
        # form is that the file stays short and legible.
        if self.render_mode is not None:
            out["render_mode"] = self.render_mode
        return out

    @classmethod
    def from_dict(cls, d: Any, *, what: str = "structure") -> StructureState:
        require_keys(d, cls._KEYS, what=what, optional=("render_mode",))
        rm = d.get("render_mode")
        return cls(
            path=as_str(d["path"], what=f"{what}.path"),
            name=as_str(d["name"], what=f"{what}.name"),
            visible=as_bool(d["visible"], what=f"{what}.visible"),
            node_visible=as_bool(d["node_visible"], what=f"{what}.node_visible"),
            render_mode=None if rm is None
            else validate_render_mode(rm, what=f"{what}.render_mode"),
            material=MaterialState.from_dict(d["material"], what=f"{what}.material"),
            provenance=ProvenanceState.from_dict(d["provenance"], what=f"{what}.provenance"),
        )


# ---------------------------------------------------------------------------
# Assets, morphs, config fingerprint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssetState:
    """Which structures were loaded, and under what load configuration.

    ``tier`` is one of ``faceforge.constants.TIER_*``.  It is optional because
    the loading pipeline does not currently expose a single "current tier"
    value; when a caller knows it, recording it turns "some structures are
    missing" into "this state was captured at a higher tier than the scene it
    is being applied to", which is diagnosable rather than mysterious.
    """

    tier: int | None = None
    skull_mode: str | None = None
    stl_dir: str | None = None
    structure_count: int = 0
    layer_visibility: tuple[tuple[str, bool], ...] = ()

    _KEYS = ("tier", "skull_mode", "stl_dir", "structure_count", "layer_visibility")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "skull_mode": self.skull_mode,
            "stl_dir": self.stl_dir,
            "structure_count": self.structure_count,
            # A mapping in the file (readable, greppable); a sorted tuple of
            # pairs in the model (hashable, order-canonical).
            "layer_visibility": dict(self.layer_visibility),
        }

    @classmethod
    def from_dict(cls, d: Any, *, what: str = "assets") -> AssetState:
        require_keys(d, cls._KEYS, what=what)
        lv = d["layer_visibility"]
        if not isinstance(lv, Mapping):
            raise SceneStateFormatError(
                f"{what}.layer_visibility: expected an object, got {type(lv).__name__}"
            )
        return cls(
            tier=None if d["tier"] is None else as_int(d["tier"], what=f"{what}.tier"),
            skull_mode=None if d["skull_mode"] is None
            else as_str(d["skull_mode"], what=f"{what}.skull_mode"),
            stl_dir=None if d["stl_dir"] is None
            else as_str(d["stl_dir"], what=f"{what}.stl_dir"),
            structure_count=as_int(d["structure_count"], what=f"{what}.structure_count"),
            layer_visibility=tuple(
                (as_str(k, what=f"{what}.layer_visibility key"),
                 as_bool(v, what=f"{what}.layer_visibility[{k}]"))
                for k, v in sorted(lv.items())
            ),
        )


@dataclass(frozen=True)
class MorphState:
    """Morph parameters that are not part of BodyState.

    ``BodyState.gender`` is dumped in the ``body`` block; what is recorded here
    is ``gender_applied`` -- the value actually pushed into
    ``GenderMorphSystem``.  The two differ in normal use: ``app.py`` updates
    ``body.gender`` on every slider tick but only re-morphs the meshes on
    release, so a state captured mid-drag would otherwise claim a body shape
    that was never rendered.

    ``alignment`` is the faceGroup alignment (scale, offsets, rot_x) held by
    the Align tab.  ``None`` when the caller did not supply it.
    """

    gender_applied: float | None = None
    alignment: tuple[tuple[str, float], ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "gender_applied": self.gender_applied,
            "alignment": None if self.alignment is None else dict(self.alignment),
        }

    @classmethod
    def from_dict(cls, d: Any, *, what: str = "morph") -> MorphState:
        require_keys(d, ("gender_applied", "alignment"), what=what)
        al = d["alignment"]
        if al is not None and not isinstance(al, Mapping):
            raise SceneStateFormatError(
                f"{what}.alignment: expected an object or null, got {type(al).__name__}"
            )
        return cls(
            gender_applied=None if d["gender_applied"] is None
            else as_float(d["gender_applied"], what=f"{what}.gender_applied"),
            alignment=None if al is None else tuple(
                (as_str(k, what=f"{what}.alignment key"),
                 as_float(v, what=f"{what}.alignment[{k}]"))
                for k, v in sorted(al.items())
            ),
        )


@dataclass(frozen=True)
class ConfigFingerprint:
    """A digest of the anatomy config set the state was captured against.

    Why this is not optional: the configs decide which STL is loaded under
    which name, with what colour and opacity, and ``fma_labels.json`` decides
    what a structure *is*.  A state file that renders differently against a
    different config set, without saying so, is worse than no state file -- it
    is a reproducibility claim that is false.  ``fma_labels_digest`` is broken
    out separately because the crosswalk changes for provenance reasons that
    need not change a render.
    """

    algorithm: str = "sha256"
    digest: str = ""
    file_count: int = 0
    root: str = ""
    fma_labels_digest: str = ""

    _KEYS = ("algorithm", "digest", "file_count", "root", "fma_labels_digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "digest": self.digest,
            "file_count": self.file_count,
            "root": self.root,
            "fma_labels_digest": self.fma_labels_digest,
        }

    @classmethod
    def from_dict(cls, d: Any, *, what: str = "config") -> ConfigFingerprint:
        require_keys(d, cls._KEYS, what=what)
        return cls(
            algorithm=as_str(d["algorithm"], what=f"{what}.algorithm"),
            digest=as_str(d["digest"], what=f"{what}.digest"),
            file_count=as_int(d["file_count"], what=f"{what}.file_count"),
            root=as_str(d["root"], what=f"{what}.root"),
            fma_labels_digest=as_str(
                d["fma_labels_digest"], what=f"{what}.fma_labels_digest"
            ),
        )


# ---------------------------------------------------------------------------
# The top-level state
# ---------------------------------------------------------------------------


@dataclass
class SceneState:
    """A complete, reproducible description of one render.

    Equality compares everything *except* :attr:`metadata`.  Metadata holds the
    generation timestamp and tool version -- facts about the file, not about the
    render -- and two states that describe the same render must compare equal
    regardless of when they were written.  :attr:`metadata` is carried verbatim
    through a load so that save -> load -> save is byte-identical; it is never
    regenerated on save.
    """

    camera: CameraState = field(default_factory=CameraState)
    viewport: ViewportState = field(default_factory=ViewportState)
    lighting: LightingState = field(default_factory=LightingState)
    render: RenderState = field(default_factory=RenderState)
    structures: tuple[StructureState, ...] = ()
    face: Mapping[str, Any] = field(default_factory=dict)
    body: Mapping[str, Any] = field(default_factory=dict)
    target_au: Mapping[str, Any] = field(default_factory=dict)
    target_head: Mapping[str, Any] = field(default_factory=dict)
    target_body: Mapping[str, Any] = field(default_factory=dict)
    morph: MorphState = field(default_factory=MorphState)
    assets: AssetState = field(default_factory=AssetState)
    config: ConfigFingerprint = field(default_factory=ConfigFingerprint)
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        # Canonical structure order, enforced here rather than trusted from the
        # caller: a capture that walked the scene graph in a different order
        # must still produce an identical file.
        self.structures = tuple(sorted(self.structures, key=lambda s: s.path))
        paths = [s.path for s in self.structures]
        if len(set(paths)) != len(paths):
            seen: set[str] = set()
            dupes = sorted({p for p in paths if p in seen or seen.add(p)})
            raise SceneStateFormatError(
                f"structures: duplicate path(s) {dupes}.  Paths address structures "
                "on apply, so they must be unique."
            )

    # -- convenience views ------------------------------------------------

    @property
    def visible_structures(self) -> tuple[StructureState, ...]:
        """Structures the renderer would actually draw: mesh *and* node visible."""
        return tuple(s for s in self.structures if s.visible and s.node_visible)

    def structure(self, path: str) -> StructureState | None:
        for s in self.structures:
            if s.path == path:
                return s
        return None

    def structure_modes(self) -> dict[str, str]:
        """path -> the render mode that will actually be used."""
        g = self.render.global_mode
        return {s.path: s.effective_render_mode(g) for s in self.structures}


def require_keys(
    d: Any, keys: Iterable[str], *, what: str, optional: Sequence[str] = ()
) -> None:
    """Assert *d* is a mapping with exactly *keys* (plus any of *optional*)."""
    if not isinstance(d, Mapping):
        raise SceneStateFormatError(f"{what}: expected an object, got {type(d).__name__}")
    keys = tuple(keys)
    missing = sorted(k for k in keys if k not in d)
    if missing:
        raise SceneStateFormatError(f"{what}: missing key(s) {missing}")
    extra = sorted(set(d) - set(keys) - set(optional))
    if extra:
        raise SceneStateFormatError(
            f"{what}: unexpected key(s) {extra}.  Expected {sorted(keys)}"
            + (f" (optional: {sorted(optional)})" if optional else "")
        )
