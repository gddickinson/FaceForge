"""Capturing a SceneState from live objects, and applying one back onto them.

This is the only module in the package that knows about ``Camera``,
``LightSetup``, ``GLRenderer``, ``Scene`` and ``StateManager``.  Keeping it
separate from the model means a state file can be validated, diffed and
version-checked with none of those imported -- and it means the round-trip
tests exercise the format itself rather than the app's object graph.

Structure identity
------------------
Structures are addressed by **scene-graph path**, not by name or index.  Mesh
names are not unique across the loaded set (muscle configs reuse names freely),
and index order changes whenever a layer is loaded on demand, so neither is a
stable key.  A path is the ``/``-joined chain of node names from the root, with
``[n]`` appended when a node has same-named siblings; ``%`` and ``/`` inside a
name are percent-escaped so a path parses unambiguously.  Two structural
guarantees follow: the same scene always yields the same paths, and a state
captured before a layer was loaded applies cleanly to a scene that has it
(the extra structures are reported, not silently mutated).
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from faceforge.core.scene_state import codec
from faceforge.core.scene_state.confighash import fingerprint_configs
from faceforge.core.scene_state.model import (
    AssetState,
    CameraState,
    LightingState,
    MaterialState,
    MorphState,
    PointLightState,
    ProvenanceState,
    RenderState,
    ClipPlaneState,
    SceneState,
    SceneStateError,
    StructureState,
    ViewportState,
    apply_dataclass_dict,
    as_float,
    dump_dataclass,
    mat4_tuple,
    vec_tuple,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scene-graph paths
# ---------------------------------------------------------------------------


def _escape(label: str) -> str:
    return label.replace("%", "%25").replace("/", "%2F")


def scene_paths(root: Any) -> list[tuple[str, Any]]:
    """``[(path, node), ...]`` for every node under *root*, root itself excluded.

    Depth-first in child order, which is the order the renderer traverses in,
    so a path list read top to bottom matches the draw order the scene graph
    presents.  Callers that need a canonical order sort by path.
    """
    out: list[tuple[str, Any]] = []

    def walk(node: Any, prefix: str) -> None:
        counts: Counter[str] = Counter()
        totals = Counter(_escape(c.name or "") for c in node.children)
        for child in node.children:
            label = _escape(child.name or "")
            counts[label] += 1
            # An index suffix only where it is needed: adding "[0]" to every
            # unique name would make every path in the file noisier and would
            # break the moment a second same-named node appeared.
            suffix = f"[{counts[label] - 1}]" if totals[label] > 1 else ""
            path = f"{prefix}/{label}{suffix}"
            out.append((path, child))
            walk(child, path)

    walk(root, "")
    return out


def mesh_paths(root: Any) -> list[tuple[str, Any]]:
    """``[(path, node), ...]`` for nodes that carry a mesh."""
    return [(p, n) for p, n in scene_paths(root) if getattr(n, "mesh", None) is not None]


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def _material_state(material: Any) -> MaterialState:
    wc = getattr(material, "wireframe_color", None)
    return MaterialState(
        color=vec_tuple(material.color, 3, what="material.color"),
        opacity=as_float(material.opacity, what="material.opacity"),
        shininess=as_float(material.shininess, what="material.shininess"),
        emissive=vec_tuple(material.emissive, 3, what="material.emissive"),
        double_sided=bool(material.double_sided),
        transparent=bool(material.transparent),
        depth_write=bool(material.depth_write),
        wireframe_color=None if wc is None
        else vec_tuple(wc, 3, what="material.wireframe_color"),
        vertex_colors_active=bool(material.vertex_colors_active),
    )


def _provenance_state(mesh: Any) -> ProvenanceState:
    return ProvenanceState(
        source_id=str(getattr(mesh, "source_id", "") or ""),
        ontology_id=str(getattr(mesh, "ontology_id", "") or ""),
        preferred_label=str(getattr(mesh, "preferred_label", "") or ""),
    )


def _majority_mode(mode_names: Sequence[str]) -> str:
    """The most common render mode, ties broken by ``RenderMode`` declaration order.

    A deterministic compression, not a guess about intent: whichever mode wins,
    every structure that uses a different one gets an explicit override, so the
    round-trip is exact either way.  The tie-break matters because ``Counter``
    ordering is insertion-dependent and insertion order here follows the scene
    graph, which changes as layers load.
    """
    from faceforge.core.material import RenderMode

    if not mode_names:
        return RenderState().global_mode
    order = {m.name: i for i, m in enumerate(RenderMode)}
    counts = Counter(mode_names)
    return min(counts, key=lambda name: (-counts[name], order.get(name, len(order))))


def capture_camera(camera: Any) -> CameraState:
    return CameraState(
        position=vec_tuple(camera.position, 3, what="camera.position"),
        target=vec_tuple(camera.target, 3, what="camera.target"),
        up=vec_tuple(camera.up, 3, what="camera.up"),
        fov_deg=as_float(camera.fov, what="camera.fov"),
        near=as_float(camera.near, what="camera.near"),
        far=as_float(camera.far, what="camera.far"),
        aspect=as_float(camera.aspect, what="camera.aspect"),
    )


def capture_lighting(lights: Any) -> LightingState:
    pl = getattr(lights, "point_light", None)
    return LightingState(
        ambient_color=vec_tuple(lights.ambient_color, 3, what="lights.ambient_color"),
        light_dir=vec_tuple(lights.light_dir, 3, what="lights.light_dir"),
        light_color=vec_tuple(lights.light_color, 3, what="lights.light_color"),
        point_light=None if pl is None else PointLightState(
            position=vec_tuple(pl.position, 3, what="point_light.position"),
            color=vec_tuple(pl.color, 3, what="point_light.color"),
            intensity=as_float(pl.intensity, what="point_light.intensity"),
            range=as_float(pl.range, what="point_light.range"),
            enabled=bool(pl.enabled),
        ),
    )


def capture_render(renderer: Any, *, global_mode: str) -> RenderState:
    if renderer is None:
        return RenderState(global_mode=global_mode)
    st = getattr(renderer, "scene_transform", None)
    return RenderState(
        global_mode=global_mode,
        clear_color=vec_tuple(renderer.CLEAR_COLOR, 4, what="renderer.CLEAR_COLOR"),
        clip_plane=ClipPlaneState(
            enabled=bool(renderer.clip_plane_enabled),
            normal=vec_tuple(renderer.clip_plane[:3], 3, what="renderer.clip_plane"),
            offset=as_float(renderer.clip_plane[3], what="renderer.clip_plane.offset"),
        ),
        scene_transform=None if st is None
        else mat4_tuple(st, what="renderer.scene_transform"),
    )


def capture_scene_state(
    *,
    scene: Any,
    camera: Any,
    lights: Any,
    renderer: Any = None,
    state: Any = None,
    viewport: tuple[int, int] | None = None,
    visibility: Any = None,
    tier: int | None = None,
    skull_mode: str | None = None,
    stl_dir: Path | str | None = None,
    gender_applied: float | None = None,
    alignment: dict[str, float] | None = None,
    global_render_mode: str | None = None,
    config_root: Path | str | None = None,
    metadata: dict[str, Any] | None = None,
) -> SceneState:
    """Read a complete :class:`SceneState` off live objects.

    Only *scene*, *camera* and *lights* are required.  Everything optional is
    recorded as ``null`` when absent rather than being invented -- a state file
    that says "no renderer was captured" is honest; one that says the clip
    plane was off when nobody looked is not.

    Parameters
    ----------
    state
        A :class:`faceforge.core.state.StateManager`.  Without it, the face,
        body and target blocks are captured from freshly-constructed defaults,
        which is what a scene assembled by a script (rather than by the app)
        actually has.
    global_render_mode
        Force the global mode instead of deriving the majority.  Pass this when
        the caller knows the app's last-set global mode.
    """
    from faceforge.core.state import BodyState, FaceState, StateManager, TargetAU, TargetHead

    entries = mesh_paths(scene)
    modes = [n.mesh.material.render_mode.name for _, n in entries]
    global_mode = (
        _majority_mode(modes) if global_render_mode is None else global_render_mode
    )

    structures = tuple(
        StructureState(
            path=path,
            name=str(node.mesh.name or ""),
            visible=bool(node.mesh.visible),
            node_visible=bool(node.visible),
            render_mode=(
                None if node.mesh.material.render_mode.name == global_mode
                else node.mesh.material.render_mode.name
            ),
            material=_material_state(node.mesh.material),
            provenance=_provenance_state(node.mesh),
        )
        for path, node in entries
    )

    if viewport is None:
        # GLRenderer keeps the viewport it was last resized to in _width /
        # _height.  Read rather than required from the caller because the
        # renderer is the component that actually knows, and a state whose
        # aspect disagrees with its viewport is a reproducibility bug.
        w = int(getattr(renderer, "_width", 1) or 1)
        h = int(getattr(renderer, "_height", 1) or 1)
        viewport = (w, h)

    sm = state if state is not None else None
    face_src = sm.face if sm is not None else FaceState()
    body_src = sm.body if sm is not None else BodyState()
    tau_src = sm.target_au if sm is not None else TargetAU()
    thead_src = sm.target_head if sm is not None else TargetHead()
    tbody_src = sm.target_body if sm is not None else BodyState()
    if sm is not None and not isinstance(sm, StateManager):
        logger.debug("capture: state is %s, not StateManager", type(sm).__name__)

    layer_vis: tuple[tuple[str, bool], ...] = ()
    if visibility is not None:
        layer_vis = tuple(
            (name, bool(visibility.is_visible(name)))
            for name in sorted(visibility.get_toggle_names())
        )

    return SceneState(
        camera=capture_camera(camera),
        viewport=ViewportState(width=viewport[0], height=viewport[1]),
        lighting=capture_lighting(lights),
        render=capture_render(renderer, global_mode=global_mode),
        structures=structures,
        face=dump_dataclass(face_src),
        body=dump_dataclass(body_src),
        target_au=dump_dataclass(tau_src),
        target_head=dump_dataclass(thead_src),
        target_body=dump_dataclass(tbody_src),
        morph=MorphState(
            gender_applied=None if gender_applied is None
            else as_float(gender_applied, what="morph.gender_applied"),
            alignment=None if alignment is None else tuple(
                (str(k), as_float(v, what=f"morph.alignment[{k}]"))
                for k, v in sorted(alignment.items())
            ),
        ),
        assets=AssetState(
            tier=tier,
            skull_mode=skull_mode,
            stl_dir=None if stl_dir is None else Path(stl_dir).name,
            structure_count=len(structures),
            layer_visibility=layer_vis,
        ),
        config=fingerprint_configs(config_root),
        metadata=codec.new_metadata(**(metadata or {})),
    )


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


@dataclass
class ApplyReport:
    """What applying a state actually did.  Never raises on a partial match.

    A state captured at tier 3 applied to a tier-1 scene is a real workflow (a
    reviewer opens a figure's state file without loading every muscle), so
    missing structures are reported, not fatal.  What would be fatal is
    reporting success: ``ok`` is False whenever anything was missing or extra,
    so a caller that wants an exact replay can assert on it.
    """

    applied: int = 0
    missing_paths: tuple[str, ...] = ()
    extra_paths: tuple[str, ...] = ()
    camera_applied: bool = False
    lighting_applied: bool = False
    render_applied: bool = False
    animation_applied: bool = False
    notes: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.missing_paths and not self.extra_paths

    def summary(self) -> str:
        bits = [f"{self.applied} structures applied"]
        if self.missing_paths:
            bits.append(
                f"{len(self.missing_paths)} in the state file absent from the scene "
                f"(e.g. {list(self.missing_paths[:3])})"
            )
        if self.extra_paths:
            bits.append(
                f"{len(self.extra_paths)} in the scene absent from the state file "
                f"(e.g. {list(self.extra_paths[:3])})"
            )
        for part in ("camera", "lighting", "render", "animation"):
            if not getattr(self, f"{part}_applied"):
                bits.append(f"{part} not applied (no target supplied)")
        return "; ".join(bits + list(self.notes))


def apply_camera(state: SceneState, camera: Any) -> None:
    from faceforge.core.math_utils import vec3

    c = state.camera
    camera.look_at(vec3(*c.position), vec3(*c.target), vec3(*c.up))
    camera.fov = c.fov_deg
    camera.near = c.near
    camera.far = c.far
    camera.aspect = c.aspect
    # Camera caches its projection behind a private dirty flag and exposes no
    # public marker for it (``set_aspect`` recomputes aspect from a pixel size,
    # which would discard the captured value).  Setting the flag is the only
    # way to make a restored fov/near/far/aspect take effect; a public
    # ``Camera.mark_projection_dirty()`` would be the tidier fix, in a file
    # this track does not own.
    camera._proj_dirty = True
    camera.mark_view_dirty()


def apply_lighting(state: SceneState, lights: Any) -> None:
    from faceforge.core.math_utils import vec3
    from faceforge.rendering.lights import PointLight

    lg = state.lighting
    lights.ambient_color = vec3(*lg.ambient_color)
    # Stored as captured, NOT re-normalised.  LightSetup normalises once at
    # construction; re-normalising on load would change the value that was
    # actually rendered with if anything had written a non-unit direction.
    lights.light_dir = vec3(*lg.light_dir)
    lights.light_color = vec3(*lg.light_color)
    if lg.point_light is None:
        lights.point_light = None
        return
    p = lg.point_light
    lights.point_light = PointLight(
        position=vec3(*p.position),
        color=tuple(p.color),
        intensity=p.intensity,
        range=p.range,
        enabled=p.enabled,
    )


def apply_render(state: SceneState, renderer: Any) -> None:
    import numpy as np

    r = state.render
    # Instance attribute deliberately shadows the GLRenderer.CLEAR_COLOR class
    # attribute -- the same thing app.py's mode-driven background switch does.
    renderer.CLEAR_COLOR = tuple(r.clear_color)
    renderer._bg_color_dirty = True
    if r.clip_plane.enabled:
        renderer.set_clip_plane(tuple(r.clip_plane.normal), r.clip_plane.offset)
    else:
        renderer.clear_clip_plane()
    renderer.scene_transform = (
        None if r.scene_transform is None
        else np.array([list(row) for row in r.scene_transform], dtype=np.float64)
    )


def apply_animation(state: SceneState, state_manager: Any) -> None:
    apply_dataclass_dict(state_manager.face, state.face)
    apply_dataclass_dict(state_manager.body, state.body)
    apply_dataclass_dict(state_manager.target_au, state.target_au)
    apply_dataclass_dict(state_manager.target_head, state.target_head)
    apply_dataclass_dict(state_manager.target_body, state.target_body)


def apply_scene_state(
    state: SceneState,
    *,
    scene: Any = None,
    camera: Any = None,
    lights: Any = None,
    renderer: Any = None,
    state_manager: Any = None,
    visibility: Any = None,
    strict: bool = False,
) -> ApplyReport:
    """Write *state* back onto whichever live objects are supplied.

    With ``strict=True``, a structure set that does not match the state exactly
    raises :class:`SceneStateError` instead of being reported.  Use it in a
    reproducibility check (the render must be *the* render); leave it off in
    interactive use.
    """
    from faceforge.core.material import RenderMode

    report = ApplyReport()

    if camera is not None:
        apply_camera(state, camera)
        report.camera_applied = True
    if lights is not None:
        apply_lighting(state, lights)
        report.lighting_applied = True
    if renderer is not None:
        apply_render(state, renderer)
        report.render_applied = True
    if state_manager is not None:
        apply_animation(state, state_manager)
        report.animation_applied = True

    if scene is not None:
        by_path = dict(mesh_paths(scene))
        global_mode = state.render.global_mode
        applied = 0
        missing = []
        for s in state.structures:
            node = by_path.get(s.path)
            if node is None:
                missing.append(s.path)
                continue
            mesh = node.mesh
            mat = mesh.material
            mat.render_mode = RenderMode[s.effective_render_mode(global_mode)]
            m = s.material
            mat.color = tuple(m.color)
            mat.opacity = m.opacity
            mat.shininess = m.shininess
            mat.emissive = tuple(m.emissive)
            mat.double_sided = m.double_sided
            mat.transparent = m.transparent
            mat.depth_write = m.depth_write
            mat.wireframe_color = (
                None if m.wireframe_color is None else tuple(m.wireframe_color)
            )
            mat.vertex_colors_active = m.vertex_colors_active
            mesh.visible = s.visible
            node.visible = s.node_visible
            applied += 1
        report.applied = applied
        report.missing_paths = tuple(missing)
        report.extra_paths = tuple(
            sorted(set(by_path) - {s.path for s in state.structures})
        )

    if visibility is not None and state.assets.layer_visibility:
        known = set(visibility.get_toggle_names())
        unknown = []
        for name, vis in state.assets.layer_visibility:
            if name in known:
                visibility.set_visible(name, vis)
            else:
                unknown.append(name)
        if unknown:
            report.notes = report.notes + (
                f"{len(unknown)} layer toggle(s) in the state file are not "
                f"registered in this scene: {sorted(unknown)[:5]}",
            )

    if not report.ok:
        message = (
            "applying this state did not reproduce its structure set: "
            + report.summary()
        )
        if strict:
            raise SceneStateError(message)
        logger.warning("%s", message)
    return report


def structures_missing_provenance(
    state: SceneState, *, visible_only: bool = True
) -> tuple[StructureState, ...]:
    """Structures that carry a ``source_id`` but no ``ontology_id``.

    A non-empty result means the state is not fully citable: the structure came
    from a BodyParts3D mesh, so an FMA term exists for it, but the crosswalk
    did not resolve it.  Procedural geometry (no ``source_id``) is excluded --
    it has no anatomical referent to cite.
    """
    pool: Iterable[StructureState] = (
        state.visible_structures if visible_only else state.structures
    )
    return tuple(s for s in pool if s.provenance.source_id and not s.provenance.ontology_id)
