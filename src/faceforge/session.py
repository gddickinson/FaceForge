"""A headless, scriptable FaceForge render session.

Why this module exists
----------------------
Everything FaceForge can draw was reachable only through the GUI.  There was no
way to say "render figure 3a" from a script, which means there was no way to
regenerate a figure in a paper -- not even by its author.
:mod:`faceforge.core.scene_state` supplied one half of the answer (a diffable
file that records *what* to render); this module supplies the other half (a
process that renders it with no window, no Qt and no display).

    from faceforge.session import Session

    with Session.create(width=512, height=512) as s:
        s.load_state_scene(state)          # geometry, from the state's provenance
        s.apply_state("figure_3a.state.json")
        s.save_png("figure_3a.png")

What this module is *not*
-------------------------
It is not a renderer.  Every pixel comes from :class:`faceforge.rendering.renderer.GLRenderer`,
the same class the GUI uses, and the OpenGL context comes from
``tools/glcontext.py``'s ``acquire_offscreen_gl()``.  A second renderer or a
second context-acquisition path would be a second thing to keep correct, and
the pixel-identity claims below only mean anything because there is exactly
one of each.

One session per process, enforced
---------------------------------
OpenGL object names -- vertex array objects above all -- are scoped to a
context, not to the process.  A ``GLRenderer`` that has uploaded a mesh holds a
VAO name that is meaningful only while the context it was created in is
current, and ``MeshInstance.gl_handle`` caches that upload on the *mesh*, which
outlives any renderer.  Two live sessions in one process therefore have two
ways to corrupt each other: a second context acquisition invalidates the first
session's VAO names, and a shared ``MeshInstance`` hands the second renderer a
handle into the first one's dead buffers.  This is not hypothetical -- it is
the failure that broke an earlier benchmark harness in this repo.

So: **a process may have at most one live Session.**
:meth:`Session.create` raises :class:`SessionInUseError` if one already exists,
naming it.  :meth:`Session.close` releases every GL object the session owns --
including clearing ``gl_handle`` on every mesh it drew, via the renderer's own
``remove_mesh`` -- after which a new session can be created and is free to
re-use the very same ``MeshInstance`` objects.  Sessions are sequential, not
concurrent, and that is a documented guarantee rather than an accident.

The context itself is deliberately *not* torn down.  ``tools/glcontext.py``
holds the CGL pixel format and context for the process lifetime on purpose (a
garbage-collected ``CGLContextObj`` takes the rasteriser down mid-render), and
its ``prefer="auto"`` path reuses an already-current context.  A second session
therefore renders in the same context as the first, with its own framebuffer,
its own renderer and freshly uploaded meshes.

Never a blank image
-------------------
``tools/capture_gui_screenshots.py`` destroyed 11 tracked README images by
writing blank frames after its context failed, while still exiting 0.  The
discipline here is the same as ``tools/capture_golden.py``'s:

* no context obtainable -> :class:`NoGLContextError`, listing every attempt;
* a frame that is a single uniform colour -> :class:`BlankFrameError`, and
  nothing is written;
* :meth:`Session.render` records :attr:`Session.last_content_fraction` so a
  caller can gate on "did anything actually draw".

Import cost
-----------
Importing this module pulls in numpy and the standard library, nothing else.
Qt, PyOpenGL and the FaceForge rendering stack are imported inside the
functions that need them, so ``import faceforge.session`` works on a machine
with no GL, no display and no PySide6 -- which is what lets the scan and
scene-building helpers below run in that environment.
"""

from __future__ import annotations

import logging
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "Session",
    "SessionError",
    "SessionInUseError",
    "NoGLContextError",
    "BlankFrameError",
    "AnatomyError",
    "SessionInfo",
    "AnatomyReport",
    "TIER_LOADS",
    "LAYER_LOADERS",
    "SCAN_ORIENTATIONS",
    "SCAN_REDUCTIONS",
    "scan_modes",
    "build_scene",
    "build_scene_from_state",
    "load_structures",
    "load_tier_scene",
    "add_layers",
    "make_asset_manager",
    "TierScene",
    "repo_tool_module",
    "gl_available",
    "png_bytes",
    "plane_frame",
    "scan_scene",
    "scan_to_rgb",
    "content_fraction",
    "write_png",
]


# ---------------------------------------------------------------------------
# Errors.  Every one of these means "nothing was written"; none of them can be
# reached by a session that has already produced an image.
# ---------------------------------------------------------------------------


class SessionError(RuntimeError):
    """Base class for every session failure."""


class SessionInUseError(SessionError):
    """Another Session is live in this process.  See the module docstring."""


class NoGLContextError(SessionError):
    """No OpenGL context could be obtained, so nothing can be rendered."""


class BlankFrameError(SessionError):
    """A render produced a uniform frame -- i.e. produced nothing."""


class AnatomyError(SessionError):
    """Requested anatomy could not be loaded."""


# ---------------------------------------------------------------------------
# PNG writing, from the standard library only.
#
# Pillow is not a declared dependency of this project (only tools/ uses it), and
# a library module that writes a figure must not fail on a machine that lacks
# an undeclared dependency.  zlib is enough: PNG is a length-prefixed chunk
# format and the encoder below emits the canonical minimum -- IHDR, one IDAT,
# IEND -- with filter type 0 on every scanline.  Byte-for-byte deterministic
# for a given array and zlib build, which is what makes "the same command twice
# produces identical files" a checkable claim.
# ---------------------------------------------------------------------------

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def png_bytes(image: np.ndarray) -> bytes:
    """Encode an ``(H, W, 3)`` or ``(H, W, 4)`` uint8 array as PNG bytes."""
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise ValueError(f"png_bytes expects uint8, got {arr.dtype}")
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"png_bytes expects (H, W, 3|4), got shape {arr.shape}")
    height, width, channels = arr.shape
    if height < 1 or width < 1:
        raise ValueError(f"png_bytes expects a non-empty image, got {arr.shape}")
    colour_type = 2 if channels == 3 else 6

    rows = bytearray()
    contiguous = np.ascontiguousarray(arr)
    for y in range(height):
        rows.append(0)                       # filter type 0: None
        rows += contiguous[y].tobytes()

    ihdr = struct.pack(">IIBBBBB", width, height, 8, colour_type, 0, 0, 0)
    return (
        _PNG_SIGNATURE
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(bytes(rows), 9))
        + _png_chunk(b"IEND", b"")
    )


def write_png(path: Path | str, image: np.ndarray) -> Path:
    """Write *image* to *path* as PNG.  Returns the path written.

    Written to a sibling ``.tmp`` and moved into place, so an interrupted write
    cannot leave a truncated PNG that a later comparison reads as a changed
    figure.
    """
    path = Path(path)
    data = png_bytes(image)
    if path.parent != Path(""):
        path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return path


# ---------------------------------------------------------------------------
# Frame content.  Same rule and same tolerance as
# tools.capture_golden.frame_content_fraction; tests/session assert the two
# agree numerically rather than trusting the comment.
# ---------------------------------------------------------------------------

#: A frame must differ from the clear colour on at least this fraction of its
#: pixels before :meth:`Session.render` stops warning about it.
MIN_CONTENT_FRACTION = 0.001

#: Per-channel tolerance, in 8-bit levels, below which a difference from the
#: clear colour is treated as rasteriser dither rather than geometry.
CONTENT_TOLERANCE = 6


def content_fraction(image: np.ndarray, clear_rgb8: Sequence[int]) -> float:
    """Fraction of pixels whose RGB differs perceptibly from the clear colour."""
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"expected HxWx3+ image, got shape {arr.shape}")
    bg = np.asarray(clear_rgb8, dtype=np.int16)[:3]
    delta = np.abs(arr[:, :, :3].astype(np.int16) - bg).max(axis=2)
    return float((delta > CONTENT_TOLERANCE).mean())


def _is_uniform(image: np.ndarray) -> bool:
    flat = np.asarray(image).reshape(-1, image.shape[-1])
    return len(np.unique(flat, axis=0)) <= 1


# ---------------------------------------------------------------------------
# tools/glcontext.py resolution.
#
# ``tools`` is a repo-root package, not part of the installed wheel
# ([tool.setuptools.packages.find] where = ["src"]), so ``import
# tools.glcontext`` succeeds in a source checkout and fails in an install.
# Rather than duplicate the context acquisition -- the one thing this repo has
# exactly one correct implementation of -- the module is located next to the
# package and loaded by path.  A source checkout is the normal deployment
# anyway: ``assets/stl`` is a committed symlink to a dataset outside the repo.
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    """The repo root, assuming the installed layout ``<root>/src/faceforge``."""
    return Path(__file__).resolve().parents[2]


def repo_tool_module(name: str):
    """Import ``tools.<name>``, falling back to loading it by path.

    Raises :class:`SessionError` naming both attempts if it cannot be found.
    """
    import importlib
    import importlib.util
    import sys

    try:
        return importlib.import_module(f"tools.{name}")
    except ImportError:
        pass

    alias = f"faceforge._tool_{name}"
    cached = sys.modules.get(alias)
    if cached is not None:
        return cached

    path = _repo_root() / "tools" / f"{name}.py"
    if not path.is_file():
        raise SessionError(
            f"cannot locate tools/{name}.py.\n"
            f"  tried: import tools.{name}  -> ImportError\n"
            f"  tried: {path}  -> not a file\n"
            "  tools/ is a repo-root package and is not installed with the wheel.\n"
            "  Run from a source checkout, or put the repo root on PYTHONPATH."
        )
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise SessionError(f"cannot load a module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def glcontext_module():
    """Return ``tools.glcontext``, importing it by path if it is not on sys.path."""
    try:
        return repo_tool_module("glcontext")
    except SessionError as exc:
        raise NoGLContextError(
            f"{exc}\nWithout it no OpenGL context can be acquired, so nothing "
            "can be rendered."
        ) from exc


def gl_available(prefer: str = "auto") -> bool:
    """Non-raising probe, for ``pytest.mark.skipif`` and CLI diagnostics.

    Deliberately not used on the render path: a render must fail loudly rather
    than skip, or it silently produces nothing while exiting 0.
    """
    try:
        return bool(glcontext_module().gl_context_available(prefer))
    except Exception:                            # noqa: BLE001 - probe only
        return False


# ---------------------------------------------------------------------------
# Scene construction.  No GL: these run in a process that has never had a
# context, which is what lets `faceforge-cli scan` and `export` work headlessly.
# ---------------------------------------------------------------------------

#: ``tier`` -> a short description of what loading that tier actually calls.
#: Tiers 0-2 map onto real entry points that exist today
#: (:func:`faceforge.anatomy.skull.build_skull` and
#: :class:`faceforge.coordination.loading_pipeline.LoadingPipeline`).  Tiers 3-5
#: are *on-demand layers* in ``app.py``, not phases of the load sequence, so
#: they are reached through ``layers=`` rather than ``tier=``; see
#: :data:`LAYER_LOADERS`.
TIER_LOADS: dict[int, str] = {
    0: "build_skull() only",
    1: "LoadingPipeline.load_head() -- skull, face, jaw/expression/neck muscles, vertebrae",
    2: "load_head() + LoadingPipeline.load_body_skeleton()",
}

#: layer name -> (AssetManager method, positional args).  Every entry is a
#: method that exists on :class:`faceforge.loaders.asset_manager.AssetManager`
#: today; nothing here is invented.
LAYER_LOADERS: dict[str, tuple[str, tuple[Any, ...]]] = {
    "back_muscles": ("load_body_muscles", ("back_muscles.json",)),
    "shoulder_muscles": ("load_body_muscles", ("shoulder_muscles.json",)),
    "arm_muscles": ("load_body_muscles", ("arm_muscles.json",)),
    "torso_muscles": ("load_body_muscles", ("torso_muscles.json",)),
    "hip_muscles": ("load_body_muscles", ("hip_muscles.json",)),
    "leg_muscles": ("load_body_muscles", ("leg_muscles.json",)),
    "hand_muscles": ("load_hand_muscles", ()),
    "foot_muscles": ("load_foot_muscles", ()),
    "organs": ("load_organs", ()),
    "vasculature": ("load_vasculature", ()),
    "brain": ("load_brain", ()),
    "skull_bones": ("load_skull_bones", ()),
    "teeth": ("load_teeth", ()),
    "ligaments": ("load_ligaments", ()),
    "oral": ("load_oral", ()),
    "pelvic_floor": ("load_pelvic_floor", ()),
    "intestinal": ("load_intestinal", ()),
    "cardiac_additional": ("load_cardiac_additional", ()),
    "cns_additional": ("load_cns_additional", ()),
    "skin": ("load_skin", ()),
}


@dataclass
class AnatomyReport:
    """What a load actually put in the scene.  Never silently partial."""

    structures: int = 0
    tier: int | None = None
    layers: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.structures > 0 and not self.failed

    def summary(self) -> str:
        bits = [f"{self.structures} structure(s)"]
        if self.tier is not None:
            bits.append(f"tier {self.tier}")
        if self.layers:
            bits.append(f"layers {list(self.layers)}")
        if self.failed:
            bits.append(f"{len(self.failed)} FAILED to load: {list(self.failed)[:5]}")
        return "; ".join(bits + list(self.notes))


def _unescape(label: str) -> str:
    """Inverse of ``scene_state.binding._escape``.

    ``%2F`` first, then ``%25``: the other order would turn an escaped literal
    ``%`` back into ``%`` before the slash escape was read, and a structure
    genuinely named ``a/b`` would come back as ``a%2Fb``.
    """
    return label.replace("%2F", "/").replace("%25", "%")


def _split_path(path: str) -> list[tuple[str, int]]:
    """``"/skullGroup/Mandible[1]"`` -> ``[("skullGroup", 0), ("Mandible", 1)]``.

    Splitting on ``/`` is safe because ``_escape`` has already replaced every
    literal slash in a node name with ``%2F``.
    """
    out: list[tuple[str, int]] = []
    for raw in path.split("/"):
        if not raw:
            continue
        index = 0
        name = raw
        if name.endswith("]") and "[" in name:
            head, _, tail = name.rpartition("[")
            digits = tail[:-1]
            if digits.isdigit():
                name, index = head, int(digits)
        out.append((_unescape(name), index))
    return out


def _stl_dir(stl_dir: Path | str | None) -> Path:
    if stl_dir is not None:
        return Path(stl_dir)
    from faceforge.constants import STL_DIR

    return Path(STL_DIR)


def _load_geometry(source_id: str, stl_dir: Path, transform: Any | None):
    from faceforge.loaders.stl_parser import load_stl_file

    path = stl_dir / f"{source_id}.stl"
    geom = load_stl_file(path)
    if transform is not None:
        # The same two calls load_stl_batch makes, in the same order.  Applying
        # the transform here rather than reimplementing it keeps one definition
        # of BP3D -> skull coordinates in the tree.
        transform.transform_positions_in_place(geom.positions, geom.vertex_count)
        transform.transform_normals_in_place(geom.normals, geom.vertex_count)
    return geom


def load_structures(
    source_ids: Sequence[str],
    *,
    scene: Any = None,
    group_name: str = "sessionGroup",
    stl_dir: Path | str | None = None,
    transform: Any | None = None,
    color: tuple[float, float, float] = (0.82, 0.76, 0.68),
    opacity: float = 1.0,
) -> tuple[Any, AnatomyReport]:
    """Load an explicit, ordered set of BodyParts3D meshes into a scene.

    The set is explicit and never globbed: a glob would reorder or resize the
    scene whenever the asset directory changed, and every stored reference
    image would silently become incomparable.

    Returns ``(scene, report)``.  Missing files are collected into
    ``report.failed`` and the report is not ``ok`` -- a subset of the requested
    set is not the requested set.
    """
    from faceforge.core.material import Material
    from faceforge.core.mesh import MeshInstance
    from faceforge.core.scene_graph import Scene, SceneNode
    from faceforge.loaders.stl_batch_loader import load_fma_labels

    if not source_ids:
        raise AnatomyError("load_structures called with an empty structure list")

    directory = _stl_dir(stl_dir)
    scene = Scene() if scene is None else scene
    group = SceneNode(name=group_name)
    scene.add(group)

    fma = load_fma_labels()
    loaded: list[str] = []
    failed: list[str] = []
    for source_id in source_ids:
        try:
            geom = _load_geometry(source_id, directory, transform)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("structure %s did not load: %s", source_id, exc)
            failed.append(source_id)
            continue
        entry = fma.get(source_id, {})
        label = entry.get("preferred_label") or source_id
        mesh = MeshInstance(
            name=label,
            geometry=geom,
            material=Material(color=color, opacity=opacity, transparent=opacity < 1.0),
            source_id=source_id,
            ontology_id=(f"FMA:{entry['fma_id']}" if entry.get("fma_id") else ""),
            preferred_label=entry.get("preferred_label", ""),
        )
        node = SceneNode(name=source_id)
        node.mesh = mesh
        group.add(node)
        loaded.append(source_id)

    report = AnatomyReport(
        structures=len(loaded),
        source_ids=tuple(loaded),
        failed=tuple(failed),
        notes=(f"explicit structure set, group {group_name!r}",),
    )
    if failed:
        raise AnatomyError(
            f"{len(failed)} of {len(source_ids)} requested structures are missing "
            f"from {directory}: {failed[:8]}.  A subset of the requested set is "
            "not the requested set; refusing to render it silently."
        )
    return scene, report


def build_scene_from_state(
    state: Any,
    *,
    stl_dir: Path | str | None = None,
    transform: Any | None = None,
    visible_only: bool = False,
) -> tuple[Any, AnatomyReport]:
    """Rebuild the geometry a :class:`SceneState` describes, from its provenance.

    Every structure in a state file records the BodyParts3D ``source_id`` it was
    loaded from and the scene-graph path it sat at, which is enough to rebuild
    the scene the state addresses -- so ``render --state FILE`` needs no other
    argument, and a committed state file is a self-contained figure.

    Two honest limitations, both worth knowing before trusting a reproduction:

    * A state file does **not** record per-node transforms, so this rebuilds an
      identity-posed scene.  States captured from a posed GUI scene record the
      pose in their ``face`` / ``body`` blocks, which are replayed by the
      animation systems, not by geometry loading; use ``tier=`` loading plus
      the app's own pipeline for those.
    * A state file does not record whether the BP3D -> skull coordinate
      transform was applied to the geometry.  ``transform=None`` (the default)
      reproduces raw STL coordinates, which is what a script-built scene has;
      pass ``CoordinateTransform.from_config()`` for a scene built by the app's
      batch loader.
    """
    from faceforge.core.material import Material, RenderMode
    from faceforge.core.mesh import MeshInstance
    from faceforge.core.scene_graph import Scene, SceneNode

    directory = _stl_dir(stl_dir)
    structures = state.visible_structures if visible_only else state.structures
    if not structures:
        raise AnatomyError(
            "this state file records no structures, so there is nothing to build"
        )

    unprovenanced = [s.path for s in structures if not s.provenance.source_id]
    if unprovenanced:
        raise AnatomyError(
            f"{len(unprovenanced)} of {len(structures)} structures in this state "
            f"carry no source_id, so their geometry cannot be located: "
            f"{unprovenanced[:5]}.  Rebuild the scene with --tier instead, or "
            "re-capture the state from a scene whose meshes carry provenance."
        )

    scene = Scene()
    nodes: dict[tuple[tuple[str, int], ...], Any] = {}
    global_mode = state.render.global_mode
    loaded: list[str] = []
    failed: list[str] = []

    for structure in structures:
        segments = _split_path(structure.path)
        if not segments:
            failed.append(structure.path)
            continue
        parent = scene
        key: tuple[tuple[str, int], ...] = ()
        for segment in segments[:-1]:
            key = key + (segment,)
            node = nodes.get(key)
            if node is None:
                node = SceneNode(name=segment[0])
                parent.add(node)
                nodes[key] = node
            parent = node

        source_id = structure.provenance.source_id
        try:
            geom = _load_geometry(source_id, directory, transform)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("structure %s (%s) did not load: %s",
                           structure.path, source_id, exc)
            failed.append(f"{structure.path} ({source_id})")
            continue

        material_state = structure.material
        material = Material(
            color=tuple(material_state.color),
            opacity=material_state.opacity,
            shininess=material_state.shininess,
            emissive=tuple(material_state.emissive),
            render_mode=RenderMode[structure.effective_render_mode(global_mode)],
            double_sided=material_state.double_sided,
            transparent=material_state.transparent,
            depth_write=material_state.depth_write,
            wireframe_color=(None if material_state.wireframe_color is None
                             else tuple(material_state.wireframe_color)),
            vertex_colors_active=material_state.vertex_colors_active,
        )
        mesh = MeshInstance(
            name=structure.name,
            geometry=geom,
            material=material,
            source_id=source_id,
            ontology_id=structure.provenance.ontology_id,
            preferred_label=structure.provenance.preferred_label,
        )
        mesh.visible = structure.visible
        leaf = SceneNode(name=segments[-1][0])
        leaf.mesh = mesh
        leaf.visible = structure.node_visible
        parent.add(leaf)
        nodes[key + (segments[-1],)] = leaf
        loaded.append(source_id)

    if failed:
        raise AnatomyError(
            f"{len(failed)} of {len(structures)} structures in this state could not "
            f"be rebuilt: {failed[:8]}.  Rendering the remainder would produce a "
            "different figure from the one the state describes."
        )

    report = AnatomyReport(
        structures=len(loaded),
        source_ids=tuple(loaded),
        notes=(
            "rebuilt from state provenance"
            + ("" if transform is None else " with the BP3D coordinate transform"),
        ),
    )
    return scene, report


def make_asset_manager(stl_dir: Path | str | None = None) -> Any:
    """A fresh :class:`AssetManager` with its coordinate transform loaded.

    Never share one between sessions: ``AssetManager`` caches ``MeshInstance``
    objects (``load_skull``), and a MeshInstance shared between two sessions
    carries a ``gl_handle`` from whichever renderer uploaded it first.
    """
    from faceforge.loaders.asset_manager import AssetManager

    assets = AssetManager(stl_dir=None if stl_dir is None else Path(stl_dir))
    assets.init_transform()
    return assets


@dataclass
class TierScene:
    """A scene built by the app's own loading pipeline."""

    scene: Any
    named_nodes: dict
    visibility: Any
    assets: Any
    report: AnatomyReport


def load_tier_scene(
    tier: int,
    *,
    skull_mode: str = "original",
    stl_dir: Path | str | None = None,
    assets: Any = None,
) -> TierScene:
    """Build a tier-0/1/2 scene with the app's own pipeline.  No GL required.

    This runs :class:`faceforge.coordination.scene_builder.SceneBuilder` and
    :class:`faceforge.coordination.loading_pipeline.LoadingPipeline` -- the same
    objects ``app.py`` uses -- so the scene-graph shape, the BP3D coordinate
    transform and the per-structure materials are the app's, not this module's.
    Being GL-free matters: ``faceforge-cli scan`` and ``export`` need a loaded
    scene and no framebuffer.
    """
    from faceforge.core.scene_state import mesh_paths

    if tier not in TIER_LOADS:
        loadable = "; ".join(f"{k} = {v}" for k, v in sorted(TIER_LOADS.items()))
        raise AnatomyError(
            f"tier {tier} is not loadable headlessly.  Loadable tiers: "
            f"{loadable}.  Tiers 3-5 of faceforge.constants are on-demand "
            "layers, not load phases -- use layers=, one of "
            f"{sorted(LAYER_LOADERS)}."
        )

    from faceforge.coordination.scene_builder import SceneBuilder
    from faceforge.coordination.visibility import VisibilityManager
    from faceforge.core.events import EventBus

    assets = make_asset_manager(stl_dir) if assets is None else assets
    visibility = VisibilityManager()
    builder = SceneBuilder(assets, visibility)
    scene, named_nodes = builder.build()

    notes: list[str] = [TIER_LOADS[tier]]
    if tier == 0:
        from faceforge.anatomy.skull import build_skull

        group, _meshes, _pivot = build_skull(assets, mode=skull_mode)
        target = named_nodes.get("skullGroup")
        if target is None:
            raise AnatomyError("SceneBuilder produced no skullGroup to fill")
        for child in list(group.children):
            target.add(child)
    else:
        from faceforge.coordination.loading_pipeline import LoadingPipeline

        pipeline = LoadingPipeline(assets, EventBus(), named_nodes)
        pipeline.load_head(skull_mode)
        if tier >= 2:
            pipeline.load_body_skeleton()
        if pipeline.report.degraded:
            notes.append(pipeline.report.summary())

    scene.update()
    structures = len(mesh_paths(scene))
    if structures == 0:
        raise AnatomyError(
            f"tier {tier} loaded no structures at all.  Check the asset set with "
            "`faceforge-cli verify-assets`."
        )
    return TierScene(
        scene=scene,
        named_nodes=named_nodes,
        visibility=visibility,
        assets=assets,
        report=AnatomyReport(structures=structures, tier=tier, notes=tuple(notes)),
    )


def add_layers(
    scene: Any,
    layers: Sequence[str],
    *,
    assets: Any = None,
    stl_dir: Path | str | None = None,
) -> tuple[Any, list[str]]:
    """Add on-demand layers to *scene*.  Returns ``(assets, failed_names)``.

    Every layer name maps to an :class:`AssetManager` method that exists today
    (:data:`LAYER_LOADERS`); nothing here invents a loader.
    """
    unknown = [name for name in layers if name not in LAYER_LOADERS]
    if unknown:
        raise AnatomyError(f"unknown layer(s) {unknown}; known: {sorted(LAYER_LOADERS)}")
    assets = make_asset_manager(stl_dir) if assets is None else assets
    failed: list[str] = []
    for name in layers:
        method_name, args = LAYER_LOADERS[name]
        result = getattr(assets, method_name)(*args)
        scene.add(result.group)
        failed.extend(f"{name}:{f}" for f in result.failed)
    scene.update()
    return assets, failed


def build_scene(
    *,
    state: Any = None,
    tier: int | None = None,
    structures: Sequence[str] | None = None,
    layers: Sequence[str] = (),
    skull_mode: str = "original",
    stl_dir: Path | str | None = None,
    transform: Any | None = None,
    visible_only: bool = False,
) -> tuple[Any, AnatomyReport]:
    """Build a scene from whichever source was requested.  No GL required.

    Exactly one of *state*, *tier* or *structures* chooses the geometry source;
    *layers* is added on top of any of them.  This is the GL-free half of
    :meth:`Session.load_anatomy`, shared with the ``scan`` and ``export``
    commands, which need a scene and no framebuffer.
    """
    chosen = [k for k, v in (("state", state), ("tier", tier),
                             ("structures", structures)) if v is not None]
    if len(chosen) != 1:
        raise AnatomyError(
            "build_scene needs exactly one of state=, tier= or structures=; "
            f"got {chosen or 'none'}"
        )

    assets = None
    if state is not None:
        scene, report = build_scene_from_state(
            state, stl_dir=stl_dir, transform=transform, visible_only=visible_only,
        )
    elif structures is not None:
        scene, report = load_structures(
            structures, stl_dir=stl_dir, transform=transform,
        )
    else:
        built = load_tier_scene(tier, skull_mode=skull_mode, stl_dir=stl_dir)  # type: ignore[arg-type]
        scene, report, assets = built.scene, built.report, built.assets

    if layers:
        from faceforge.core.scene_state import mesh_paths

        _assets, failed = add_layers(scene, layers, assets=assets, stl_dir=stl_dir)
        report = AnatomyReport(
            structures=len(mesh_paths(scene)),
            tier=report.tier,
            layers=tuple(layers),
            source_ids=report.source_ids,
            failed=tuple(list(report.failed) + failed),
            notes=report.notes,
        )
    return scene, report


# ---------------------------------------------------------------------------
# The virtual scanner, headless.  ScannerEngine is pure numpy -- no GL, no Qt --
# so a scan needs no context at all.
# ---------------------------------------------------------------------------

#: Orientation -> (normal, right, up), copied from
#: ``faceforge.scanner.scanner_window.ScannerWindow._compute_plane``.  That
#: method is on a QWidget, so it cannot be called without Qt; duplicating the
#: three frames is the smaller evil, and ``tests/session`` locks the values.
#: Extracting ``_compute_plane`` into ``scanner/scan_plane.py`` would remove the
#: duplication -- see the track report.
SCAN_ORIENTATIONS: dict[str, tuple[tuple[float, float, float], ...]] = {
    "axial": ((0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
    "coronal": ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
    "sagittal": ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
}

#: The reductions ``ScannerEngine.scan`` actually branches on.
SCAN_REDUCTIONS: tuple[str, ...] = ("mean", "max", "min", "sum")


def scan_modes() -> tuple[str, ...]:
    """The scan modes, read from :mod:`faceforge.scanner.tissue_map` at call time."""
    from faceforge.scanner.tissue_map import MODES

    return tuple(MODES)


def plane_frame(orientation: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(normal, right, up)`` for a named scan orientation."""
    try:
        normal, right, up = SCAN_ORIENTATIONS[orientation]
    except KeyError:
        raise ValueError(
            f"unknown orientation {orientation!r}; known: {sorted(SCAN_ORIENTATIONS)}"
        ) from None
    return (
        np.asarray(normal, dtype=np.float64),
        np.asarray(right, dtype=np.float64),
        np.asarray(up, dtype=np.float64),
    )


def scan_scene(
    scene: Any,
    *,
    origin: Sequence[float],
    orientation: str = "axial",
    width: float = 400.0,
    height: float = 400.0,
    depth: float = 10.0,
    resolution: int = 128,
    mode: str = "ct",
    reduction: str = "mean",
    progress_callback: Any = None,
) -> np.ndarray:
    """Run the virtual scanner over *scene*.  No GL context required.

    Returns what :meth:`faceforge.scanner.engine.ScannerEngine.scan` returns:
    ``(res, res)`` float32 intensities for ct/mri/xray, ``(res, res, 3)``
    float32 RGB for ``anatomical``.
    """
    from faceforge.scanner.engine import ScannerEngine
    from faceforge.scanner.tissue_map import TissueMapper

    valid_modes = scan_modes()
    if mode not in valid_modes:
        raise ValueError(f"unknown scan mode {mode!r}; known: {list(valid_modes)}")
    if reduction not in SCAN_REDUCTIONS:
        raise ValueError(
            f"unknown reduction {reduction!r}; known: {list(SCAN_REDUCTIONS)}"
        )
    if resolution < 8:
        raise ValueError(f"resolution {resolution} is too small to be an image")

    normal, right, up = plane_frame(orientation)
    scene.update()
    meshes = scene.collect_meshes()
    if not meshes:
        raise AnatomyError(
            "the scene has no visible meshes, so a scan would be uniformly empty"
        )

    engine = ScannerEngine(TissueMapper())
    engine.cache_meshes(meshes)
    return engine.scan(
        origin=np.asarray(origin, dtype=np.float64),
        normal=normal,
        right=right,
        up=up,
        width=float(width),
        height=float(height),
        depth=float(depth),
        resolution=int(resolution),
        mode=mode,
        reduction=reduction,
        progress_callback=progress_callback,
    )


def scan_to_rgb(image: np.ndarray, mode: str) -> np.ndarray:
    """Colour-map a scan result to ``(H, W, 3)`` uint8.

    The scalar colour maps live in :mod:`faceforge.scanner.tissue_map`; this is
    their vectorised equivalent, and ``tests/session`` asserts the two agree on
    every 8-bit input rather than trusting that they do.  The GUI's own
    vectorised copy (``scanner_window._apply_colormap_vec``) cannot be reused
    here because importing it pulls in Qt.
    """
    arr = np.asarray(image)
    if arr.ndim == 3:
        return np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    value = np.clip(arr.astype(np.float64), 0.0, 1.0)
    if mode in ("ct", "xray", "anatomical"):
        grey = (value * 255).astype(np.uint8)
        return np.stack([grey, grey, grey], axis=-1)
    if mode in ("mri_t1", "mri_t2"):
        red = np.clip(value * 255 + value * (1 - value) * 20, 0, 255).astype(np.uint8)
        green = (value * 255).astype(np.uint8)
        blue = (value * 245).astype(np.uint8)
        return np.stack([red, green, blue], axis=-1)
    raise ValueError(f"unknown scan mode {mode!r}; known: {list(scan_modes())}")


# ---------------------------------------------------------------------------
# The GL session
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionInfo:
    """What a session is rendering with.  Goes straight into a manifest."""

    gl_kind: str
    gl_version: str
    gl_renderer: str
    glsl_version: str
    is_software: bool
    width: int
    height: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "gl_kind": self.gl_kind,
            "gl_version": self.gl_version,
            "gl_renderer": self.gl_renderer,
            "glsl_version": self.glsl_version,
            "is_software": self.is_software,
            "viewport": {"width": self.width, "height": self.height},
        }

    def banner(self) -> str:
        speed = "  [CPU rasteriser: correctness only, NOT a benchmark]" if self.is_software else ""
        return (
            f"GL context: {self.gl_kind}\n"
            f"  GL_RENDERER  {self.gl_renderer}{speed}\n"
            f"  GL_VERSION   {self.gl_version}\n"
            f"  viewport     {self.width}x{self.height}"
        )


class _Framebuffer:
    """A colour+depth FBO the session owns and can free.

    The attachment format -- ``GL_RGBA8`` texture plus ``GL_DEPTH_COMPONENT24``
    renderbuffer -- is the same as ``tools/capture_golden.py``'s ``_make_fbo``,
    which is what makes a session render comparable to a golden capture
    pixel-for-pixel.  ``tests/session`` renders the same scene both ways and
    asserts zero differing pixels, so a divergence in either file fails loudly
    rather than quietly changing every stored reference image.
    """

    def __init__(self, width: int, height: int) -> None:
        from OpenGL.GL import (
            GL_COLOR_ATTACHMENT0, GL_DEPTH_ATTACHMENT, GL_DEPTH_COMPONENT24,
            GL_FRAMEBUFFER, GL_FRAMEBUFFER_COMPLETE, GL_RENDERBUFFER, GL_RGBA,
            GL_RGBA8, GL_TEXTURE_2D, GL_UNSIGNED_BYTE,
            glBindFramebuffer, glBindRenderbuffer, glBindTexture,
            glCheckFramebufferStatus, glFramebufferRenderbuffer,
            glFramebufferTexture2D, glGenFramebuffers, glGenRenderbuffers,
            glGenTextures, glRenderbufferStorage, glTexImage2D,
        )

        self.width = int(width)
        self.height = int(height)
        self.fbo = int(glGenFramebuffers(1))
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        self.tex = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.width, self.height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, None)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, self.tex, 0)
        self.rbo = int(glGenRenderbuffers(1))
        glBindRenderbuffer(GL_RENDERBUFFER, self.rbo)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24,
                              self.width, self.height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                  GL_RENDERBUFFER, self.rbo)
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            self.destroy()
            raise SessionError(
                f"framebuffer incomplete: status=0x{int(status):04X} at "
                f"{self.width}x{self.height}.  Nothing was rendered."
            )

    def bind(self) -> None:
        from OpenGL.GL import GL_FRAMEBUFFER, glBindFramebuffer

        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)

    def read(self) -> np.ndarray:
        """glReadPixels into ``(H, W, 4)`` uint8, flipped to top-down order."""
        from OpenGL.GL import (
            GL_RGBA, GL_UNSIGNED_BYTE, glFinish, glReadPixels,
        )

        glFinish()
        raw = glReadPixels(0, 0, self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE)
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 4)
        # GL origin is bottom-left; an image file's is top-left.
        return np.flipud(arr).copy()

    def destroy(self) -> None:
        from OpenGL.GL import (
            glDeleteFramebuffers, glDeleteRenderbuffers, glDeleteTextures,
        )

        if getattr(self, "rbo", 0):
            glDeleteRenderbuffers(1, [self.rbo])
            self.rbo = 0
        if getattr(self, "tex", 0):
            glDeleteTextures([self.tex])
            self.tex = 0
        if getattr(self, "fbo", 0):
            glDeleteFramebuffers(1, [self.fbo])
            self.fbo = 0


#: Public name for the framebuffer type above.  Exposed so that the GUI's
#: still exporter (:mod:`faceforge.export.video_export`) renders through the
#: *same* colour/depth attachment format as this session and as
#: ``tools/capture_golden.py``, instead of growing a third definition of it
#: that could drift.  There is one FBO layout in this tree, and this is it.
Framebuffer = _Framebuffer

#: The one live session, or None.  Module-level rather than a class attribute so
#: that `del session` cannot leave the guard set while the GL objects are gone.
_ACTIVE: Session | None = None


class Session:
    """A headless render session: one GL context, one renderer, one scene.

    Construct with :meth:`create`; the initialiser is not part of the API
    because a Session is only meaningful with GL resources attached, and there
    is no valid half-built state.
    """

    def __init__(self, *, _private: bool = False) -> None:
        if not _private:
            raise TypeError("use Session.create(), not Session()")
        self._gl_info: Any = None
        self._fb: _Framebuffer | None = None
        self._renderer: Any = None
        self._scene: Any = None
        self._camera: Any = None
        self._lights: Any = None
        self._assets: Any = None
        self._visibility: Any = None
        self._closed = False
        self._frames = 0
        self.last_content_fraction: float | None = None

    # -- lifecycle --------------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        width: int = 512,
        height: int = 512,
        prefer: str = "auto",
    ) -> Session:
        """Acquire GL, build renderer / scene / camera / lights, return a Session.

        Raises :class:`SessionInUseError` if a session is already live in this
        process (see the module docstring), and :class:`NoGLContextError` if no
        context can be obtained.  It never returns a Session that cannot render.
        """
        global _ACTIVE

        if _ACTIVE is not None and not _ACTIVE.closed:
            raise SessionInUseError(
                "a Session is already live in this process "
                f"({_ACTIVE!r}).  OpenGL object names -- VAOs above all -- are "
                "scoped to a context, so two live sessions can hand each other "
                "stale handles.  Close the first session (or use it as a context "
                "manager) before creating another."
            )
        if width < 1 or height < 1:
            raise ValueError(f"{width}x{height} is not a renderable size")

        glcontext = glcontext_module()
        try:
            gl_info = glcontext.acquire_offscreen_gl(prefer)
        except glcontext.GLContextError as exc:
            raise NoGLContextError(str(exc)) from exc

        session = cls(_private=True)
        session._gl_info = gl_info
        _ACTIVE = session
        try:
            from faceforge.core.scene_graph import Scene
            from faceforge.rendering.camera import Camera
            from faceforge.rendering.lights import LightSetup
            from faceforge.rendering.renderer import GLRenderer

            session._fb = _Framebuffer(width, height)
            renderer = GLRenderer()
            renderer.init_gl()
            renderer.resize(width, height)
            session._renderer = renderer
            session._scene = Scene()
            session._camera = Camera()
            session._camera.set_aspect(width, height)
            session._lights = LightSetup()
        except Exception:
            # A half-built session must not hold the process guard, or the next
            # create() reports "already live" for something that never lived.
            session._teardown()
            _ACTIVE = None
            raise
        logger.info("session created: %dx%d, %s", width, height, gl_info.kind)
        return session

    @classmethod
    def active(cls) -> Session | None:
        """The live session in this process, if any."""
        return None if _ACTIVE is None or _ACTIVE.closed else _ACTIVE

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        if self._closed:
            return "<Session closed>"
        w, h = self.size
        return f"<Session {w}x{h} {getattr(self._gl_info, 'kind', '?')} frames={self._frames}>"

    def close(self) -> None:
        """Release every GL object this session owns.  Idempotent.

        The CGL context is intentionally left current: ``tools/glcontext.py``
        holds it for the process lifetime by design, and the next session
        reuses it.  What is released is everything whose name is only valid
        while *this* renderer exists -- shaders, buffers, VAOs, the FBO -- plus
        the ``gl_handle`` cached on every mesh this session drew, so those
        ``MeshInstance`` objects can be handed to a later session safely.
        """
        global _ACTIVE

        if self._closed:
            return
        self._teardown()
        self._closed = True
        if _ACTIVE is self:
            _ACTIVE = None
        logger.info("session closed after %d frame(s)", self._frames)

    def _teardown(self) -> None:
        renderer, scene = self._renderer, self._scene
        if renderer is not None and scene is not None:
            try:
                for mesh in scene.subtree_meshes():
                    renderer.remove_mesh(mesh)
            except Exception as exc:                 # noqa: BLE001 - teardown
                logger.warning("could not release mesh GL handles: %s", exc)
        if renderer is not None:
            try:
                renderer.destroy()
            except Exception as exc:                 # noqa: BLE001 - teardown
                logger.warning("renderer teardown failed: %s", exc)
        if self._fb is not None:
            try:
                self._fb.destroy()
            except Exception as exc:                 # noqa: BLE001 - teardown
                logger.warning("framebuffer teardown failed: %s", exc)
        self._fb = None
        self._renderer = None
        self._scene = None

    def _require_open(self) -> None:
        if self._closed:
            raise SessionError(
                "this Session is closed; its GL objects are gone.  Create a new "
                "one with Session.create()."
            )

    # -- accessors --------------------------------------------------------

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def scene(self) -> Any:
        self._require_open()
        return self._scene

    @property
    def camera(self) -> Any:
        self._require_open()
        return self._camera

    @property
    def lights(self) -> Any:
        self._require_open()
        return self._lights

    @property
    def renderer(self) -> Any:
        self._require_open()
        return self._renderer

    @property
    def gl(self) -> Any:
        """The ``GLContextInfo`` returned by ``acquire_offscreen_gl``."""
        return self._gl_info

    @property
    def size(self) -> tuple[int, int]:
        self._require_open()
        assert self._fb is not None
        return self._fb.width, self._fb.height

    @property
    def frames_rendered(self) -> int:
        return self._frames

    def info(self) -> SessionInfo:
        self._require_open()
        w, h = self.size
        g = self._gl_info
        return SessionInfo(
            gl_kind=g.kind,
            gl_version=g.gl_version,
            gl_renderer=g.gl_renderer,
            glsl_version=g.glsl_version,
            is_software=bool(g.is_software),
            width=w,
            height=h,
        )

    @property
    def clear_rgb8(self) -> tuple[int, int, int]:
        """The renderer's clear colour as 8-bit RGB, read from the renderer."""
        self._require_open()
        colour = self._renderer.CLEAR_COLOR
        return tuple(int(round(c * 255)) for c in colour[:3])   # type: ignore[return-value]

    # -- scene assembly ---------------------------------------------------

    def load_anatomy(
        self,
        *,
        tier: int | None = None,
        structures: Sequence[str] | None = None,
        layers: Sequence[str] | None = None,
        skull_mode: str = "original",
        stl_dir: Path | str | None = None,
        transform: Any | None = None,
    ) -> AnatomyReport:
        """Load anatomy into this session's scene.

        Exactly one of *tier* or *structures* selects how the scene is built:

        ``structures``
            An explicit, ordered list of BodyParts3D ids (``["FMA52734", ...]``)
            loaded straight from ``assets/stl``.  Deterministic and fast; this
            is what a golden capture or a script-built figure wants.
        ``tier``
            0, 1 or 2 -- see :data:`TIER_LOADS`.  Runs the app's own
            :class:`~faceforge.coordination.loading_pipeline.LoadingPipeline`,
            so the scene-graph shape, the coordinate transform and the
            per-structure materials are the app's, not this module's.  Replaces
            the session scene with the pipeline's.

        ``layers`` adds any of :data:`LAYER_LOADERS` on top (the on-demand
        layers that ``app.py`` loads when a toggle is switched on).  Tiers 3-5
        of ``faceforge.constants`` are those layers rather than load phases, so
        they are requested by name here rather than as a tier.
        """
        self._require_open()
        if tier is not None and structures is not None:
            raise AnatomyError("pass tier= or structures=, not both")
        if tier is None and structures is None and not layers:
            raise AnatomyError("load_anatomy needs tier=, structures= or layers=")

        report = AnatomyReport()
        if structures is not None:
            _, report = load_structures(
                structures, scene=self._scene, stl_dir=stl_dir, transform=transform,
            )
        elif tier is not None:
            report = self._load_tier(tier, skull_mode=skull_mode, stl_dir=stl_dir)

        if layers:
            report = self._load_layers(layers, report, stl_dir=stl_dir)

        self._scene.update()
        logger.info("anatomy loaded: %s", report.summary())
        return report

    def _load_tier(self, tier: int, *, skull_mode: str,
                   stl_dir: Path | str | None) -> AnatomyReport:
        built = load_tier_scene(
            tier, skull_mode=skull_mode, stl_dir=stl_dir, assets=self._assets,
        )
        # Replacing the scene is the honest move: the pipeline builds its own
        # group tree and writes into it, so grafting it under a scene this
        # session already owned would produce paths that no captured state
        # matches.
        self._swap_scene(built.scene)
        self._visibility = built.visibility
        self._assets = built.assets
        return built.report

    def _load_layers(self, layers: Sequence[str], report: AnatomyReport,
                     *, stl_dir: Path | str | None) -> AnatomyReport:
        from faceforge.core.scene_state import mesh_paths

        self._assets, failed = add_layers(
            self._scene, layers, assets=self._assets, stl_dir=stl_dir,
        )
        return AnatomyReport(
            structures=len(mesh_paths(self._scene)),
            tier=report.tier,
            layers=tuple(layers),
            source_ids=report.source_ids,
            failed=tuple(list(report.failed) + failed),
            notes=report.notes,
        )

    def _swap_scene(self, scene: Any) -> None:
        """Point the session at a different scene, releasing the old one's GL."""
        old = self._scene
        if old is not None and self._renderer is not None and old is not scene:
            for mesh in old.subtree_meshes():
                self._renderer.remove_mesh(mesh)
        self._scene = scene

    def adopt_scene(self, scene: Any, *, visibility: Any = None) -> None:
        """Render *scene* from now on, releasing the previous scene's GL objects.

        The public form of what :meth:`load_anatomy` and
        :meth:`load_state_scene` do internally, for callers that built a scene
        with :func:`build_scene` (which needs no GL) and now want to render it.
        """
        self._require_open()
        self._swap_scene(scene)
        if visibility is not None:
            self._visibility = visibility
        self._scene.update()

    def load_state_scene(
        self,
        state: Any,
        *,
        stl_dir: Path | str | None = None,
        transform: Any | None = None,
        visible_only: bool = False,
    ) -> AnatomyReport:
        """Rebuild the geometry a SceneState describes and adopt it as the scene.

        See :func:`build_scene_from_state` for what a state file does and does
        not record.
        """
        self._require_open()
        scene, report = build_scene_from_state(
            state, stl_dir=stl_dir, transform=transform, visible_only=visible_only,
        )
        self._swap_scene(scene)
        self._scene.update()
        return report

    # -- state ------------------------------------------------------------

    def apply_state(
        self,
        state_or_path: Any,
        *,
        strict: bool = False,
        check_config: bool = True,
        resize_to_state: bool = True,
    ) -> Any:
        """Apply a :class:`SceneState` (or a path to one) to this session.

        Returns the :class:`~faceforge.core.scene_state.ApplyReport`.  With
        ``strict=True`` a structure set that does not match the state exactly
        raises instead of being reported -- which is what a reproducibility
        check wants.

        ``resize_to_state`` resizes the framebuffer to the state's viewport
        *before* the camera is applied, so the restored aspect ratio is the
        captured one and the framing is exact.  Pass ``False`` to render the
        same shot at a different size.
        """
        self._require_open()
        from faceforge.core.scene_state import apply_scene_state, codec

        state = state_or_path
        if isinstance(state_or_path, (str, Path)):
            state = codec.load(state_or_path, check_config=check_config)

        if resize_to_state:
            self.resize(state.viewport.width, state.viewport.height, update_aspect=False)

        report = apply_scene_state(
            state,
            scene=self._scene,
            camera=self._camera,
            lights=self._lights,
            renderer=self._renderer,
            visibility=self._visibility,
            strict=strict,
        )
        self._scene.update()
        return report

    def capture_state(self, **kwargs: Any) -> Any:
        """Capture a :class:`SceneState` from this session.

        Keyword arguments are passed through to
        :func:`faceforge.core.scene_state.capture_scene_state`; ``scene``,
        ``camera``, ``lights``, ``renderer`` and ``viewport`` are supplied from
        the session and must not be overridden.
        """
        self._require_open()
        from faceforge.core.scene_state import capture_scene_state

        clash = {"scene", "camera", "lights", "renderer", "viewport"} & set(kwargs)
        if clash:
            raise TypeError(
                f"capture_state supplies {sorted(clash)} from the session; "
                "passing them again would record something other than what was "
                "rendered"
            )
        kwargs.setdefault("visibility", self._visibility)
        return capture_scene_state(
            scene=self._scene,
            camera=self._camera,
            lights=self._lights,
            renderer=self._renderer,
            viewport=self.size,
            **kwargs,
        )

    def save_state(self, path: Path | str, **kwargs: Any) -> Path:
        """Capture and write a state file.  Returns the path written."""
        from faceforge.core.scene_state import codec

        return codec.save(self.capture_state(**kwargs), path)

    # -- rendering --------------------------------------------------------

    def resize(self, width: int, height: int, *, update_aspect: bool = True) -> None:
        """Resize the framebuffer, recreating it.  A no-op if unchanged.

        ``update_aspect=True`` also reframes the camera for the new pixel
        aspect, which is what a caller asking for a different output size
        wants.  :meth:`apply_state` passes ``False``, because the state records
        the aspect it was rendered with.
        """
        self._require_open()
        assert self._fb is not None
        width, height = int(width), int(height)
        if width < 1 or height < 1:
            raise ValueError(f"{width}x{height} is not a renderable size")
        if (width, height) != (self._fb.width, self._fb.height):
            self._fb.destroy()
            self._fb = _Framebuffer(width, height)
            self._renderer.resize(width, height)
            # A framebuffer change invalidates the renderer's mirror of GL
            # state; the renderer exposes exactly this for the purpose.
            self._renderer.invalidate_state_cache()
        if update_aspect:
            self._camera.set_aspect(width, height)

    def render(
        self,
        width: int | None = None,
        height: int | None = None,
        *,
        allow_blank: bool = False,
    ) -> np.ndarray:
        """Render one frame and return it as an ``(H, W, 4)`` uint8 array.

        Raises :class:`BlankFrameError` if the frame is a single uniform colour
        -- the render produced nothing, and returning it would let a caller
        write a blank figure and exit 0.  ``allow_blank=True`` is for tests
        that deliberately render an empty scene.
        """
        self._require_open()
        assert self._fb is not None
        if width is not None or height is not None:
            current_w, current_h = self.size
            self.resize(width or current_w, height or current_h)

        self._fb.bind()
        self._renderer.render(self._scene, self._camera, self._lights)
        image = self._fb.read()
        self._frames += 1

        if _is_uniform(image) and not allow_blank:
            raise BlankFrameError(
                f"the frame is a single uniform colour {image.reshape(-1, 4)[0].tolist()} "
                f"at {self._fb.width}x{self._fb.height}: the render produced "
                "nothing.  Refusing to return it as an image.  Check that the "
                "scene has visible meshes and that the camera is outside them."
            )
        self.last_content_fraction = content_fraction(image, self.clear_rgb8)
        if self.last_content_fraction < MIN_CONTENT_FRACTION:
            logger.warning(
                "only %.4f%% of pixels differ from the clear colour (floor %.3f%%): "
                "this frame is very nearly blank",
                self.last_content_fraction * 100.0, MIN_CONTENT_FRACTION * 100.0,
            )
        return image

    def save_png(
        self,
        path: Path | str,
        width: int | None = None,
        height: int | None = None,
        *,
        image: np.ndarray | None = None,
        allow_blank: bool = False,
    ) -> Path:
        """Render (or write the supplied frame) to a PNG.  Returns the path."""
        frame = self.render(width, height, allow_blank=allow_blank) if image is None else image
        return write_png(path, frame)

    def export_still(
        self,
        path: Path | str,
        width: int | None = None,
        height: int | None = None,
        *,
        allow_blank: bool = False,
    ) -> Any:
        """Write a publication still rendered *at* ``width x height``.

        The difference from :meth:`save_png` is not the pixels -- both go
        through this session's FBO -- but the contract.  This checks the
        requested size against ``GL_MAX_TEXTURE_SIZE``,
        ``GL_MAX_RENDERBUFFER_SIZE`` and ``GL_MAX_VIEWPORT_DIMS`` before
        allocating anything, restores the previous size if the render fails, and
        returns a :class:`faceforge.export.still.StillResult` recording the
        driver limits alongside the file, so a caller can tell a true 4K render
        from a clamped one.  See :mod:`faceforge.export.still`.
        """
        from faceforge.export.still import export_still

        return export_still(self, path, width, height, allow_blank=allow_blank)

    def gl_size_limits(self) -> Any:
        """The maximum still size this GL context can render.

        Returns a :class:`faceforge.export.still.GLSizeLimits`.
        """
        self._require_open()
        from faceforge.export.still import query_size_limits

        return query_size_limits()
