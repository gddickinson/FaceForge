"""Render a fixed, reproducible scene to PNG through a real framebuffer object.

This is the capture half of golden-image validation.  It renders the same
meshes, from the same camera, under the same light, at the same size, in every
one of the 16 render modes, and writes one PNG per mode plus a manifest
describing exactly what was rendered.  ``tools/compare_golden.py`` then diffs
two such directories.

Determinism
-----------
Nothing here is discovered at runtime:

* The mesh list is the explicit, ordered ``FIXED_MESHES`` below.  It is never
  globbed -- a glob would reorder or resize the scene whenever the asset
  directory changed, and every stored reference image would silently become
  incomparable.
* The camera is a named preset.  Presets are defined relative to the loaded
  set's own centroid and bounding radius, so a preset frames the subject
  identically for a given ``--meshes`` count, and the resulting absolute
  eye/target/up go into the manifest as plain numbers.
* Lighting is ``LightSetup()`` at its defaults with the point light off.
* No random numbers are used anywhere.  There is consequently no seed to set;
  if that ever changes, seed it here and record the seed in the manifest.

Blank-image safety
------------------
``tools/capture_gui_screenshots.py`` destroyed 11 tracked README images by
writing blank frames after its GL context failed, while still exiting 0.  This
script cannot do that:

1. It refuses to start without a real GL context (:mod:`tools.glcontext`
   raises rather than returning a sentinel).
2. It refuses to write into a non-empty directory unless ``--force``.
3. Every rendered frame is checked for content -- a frame that is uniformly
   one colour, or that is indistinguishable from the clear colour, is a
   failure, not an output.
4. PNGs are written to a ``.partial`` staging directory and only moved into
   place once *every* mode has passed.  A failed run leaves no manifest, and
   ``compare_golden.py`` refuses to compare a directory without one.

Usage
-----
    python -m tools.capture_golden --out captures/ref
    python -m tools.capture_golden --out captures/cur --modes SOLID,XRAY --size 256x256
    python -m tools.capture_golden --selftest      # no GL needed

The software rasteriser reached by :mod:`tools.glcontext` is a correctness
tool, not a benchmark.  Frame times printed by this script through it are two
orders of magnitude off hardware and must never be quoted as performance.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger("capture_golden")

REPO_ROOT = Path(__file__).resolve().parent.parent
STL_SUBDIR = Path("assets/stl")

# ----------------------------------------------------------------------------
# The fixed scene: the cranium plus mandible, ordered.  --meshes N takes the
# first N entries, so N is meaningful and monotonic rather than arbitrary.
# Triangle counts are the STL header counts measured on 2026-08-29 and are
# re-verified at capture time; a mismatch means the assets changed under us and
# the capture is reported as non-comparable rather than quietly diffed.
# ----------------------------------------------------------------------------
FIXED_MESHES: tuple[tuple[str, str], ...] = (
    ("FMA52734", "Frontal bone"),
    ("FMA52788", "Right parietal bone"),
    ("FMA52789", "Left parietal bone"),
    ("FMA52735", "Occipital bone"),
    ("FMA52738", "Right temporal bone"),
    ("FMA52739", "Left temporal bone"),
    ("FMA52736", "Sphenoid bone"),
    ("FMA52740", "Ethmoid"),
    ("FMA53649", "Right maxilla"),
    ("FMA53650", "Left maxilla"),
    ("FMA52748", "Mandible"),
    ("FMA52892", "Right zygomatic bone"),
    ("FMA52893", "Left zygomatic bone"),
    ("FMA53647", "Right nasal bone"),
    ("FMA53648", "Left nasal bone"),
    ("FMA9710", "Vomer"),
)

MAX_MESHES = len(FIXED_MESHES)

# All 16 RenderMode members, in enum declaration order.  Spelled out rather
# than read from the enum so that adding a mode to the enum does not silently
# change what a "full" capture means; the selftest asserts the two agree.
ALL_MODES: tuple[str, ...] = (
    "SOLID", "WIREFRAME", "XRAY", "POINTS", "OPAQUE",
    "ILLUSTRATION", "SEPIA", "COLOR_ATLAS", "PEN_INK", "MEDICAL",
    "HOLOGRAM", "CARTOON", "PORCELAIN", "BLUEPRINT", "THERMAL", "ETHEREAL",
)


@dataclass(frozen=True)
class CameraPreset:
    """A camera placement, expressed relative to the subject's own bounds.

    ``direction`` is the unit offset from subject centroid to eye, in the
    app's Z-up clinical frame (+X left-to-right, +Y posterior, +Z superior).
    ``distance`` multiplies the subject's bounding radius.
    """

    direction: tuple[float, float, float]
    up: tuple[float, float, float]
    distance: float


CAMERA_PRESETS: dict[str, CameraPreset] = {
    # Facing the front of the skull.  -Y is anterior in this frame.
    "anterior": CameraPreset((0.0, -1.0, 0.0), (0.0, 0.0, 1.0), 2.8),
    "left_lateral": CameraPreset((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), 2.8),
    "right_lateral": CameraPreset((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), 2.8),
    # Looking straight down; up must not be parallel to the view direction.
    "superior": CameraPreset((0.0, 0.0, 1.0), (0.0, -1.0, 0.0), 2.8),
    # Three-quarter view: the one that shows the most shading variation, so
    # it is the most sensitive to a lighting or normal regression.
    "oblique": CameraPreset((-0.62, -0.68, 0.39), (0.0, 0.0, 1.0), 2.9),
}

DEFAULT_CAMERA = "oblique"
DEFAULT_SIZE = (512, 512)
MIN_SIZE, MAX_SIZE = 64, 4096

# Renderer clear colour as 8-bit RGB.  GLRenderer.CLEAR_COLOR is
# (0.12, 0.12, 0.15, 1.0) -> (31, 31, 38).  Read from the class at runtime so
# the two cannot drift; this literal is only the selftest's expectation.
EXPECTED_CLEAR_RGB8 = (31, 31, 38)

# A frame must differ from the clear colour on at least this fraction of
# pixels to count as having rendered something.  The measured content fraction
# for the full 16-mesh skull at the oblique preset is 8-30% depending on mode;
# HOLOGRAM is the sparsest.  0.1% is far below any real mode and far above the
# zero a blank frame produces.
MIN_CONTENT_FRACTION = 0.001


class CaptureError(RuntimeError):
    """A capture could not be completed.  Nothing valid was written."""


# ----------------------------------------------------------------------------
# Pure helpers -- no GL, no filesystem.  These are what --selftest exercises.
# ----------------------------------------------------------------------------

def parse_size(text: str) -> tuple[int, int]:
    """Parse ``WxH`` and clamp-check it.  Raises ValueError on anything else."""
    if not isinstance(text, str):
        raise ValueError(f"size must be a string like 512x512, got {type(text).__name__}")
    parts = text.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"size must look like WxH, got {text!r}")
    try:
        w, h = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"size must be two integers, got {text!r}") from exc
    for label, v in (("width", w), ("height", h)):
        if not MIN_SIZE <= v <= MAX_SIZE:
            raise ValueError(f"{label} {v} outside [{MIN_SIZE}, {MAX_SIZE}]")
    return w, h


def parse_modes(text: str | None) -> list[str]:
    """Parse a comma-separated mode list, preserving ALL_MODES order.

    Order is normalised deliberately: two captures that asked for the same set
    of modes in a different order must produce identical manifests, or
    compare_golden would refuse to diff them.
    """
    if text is None or text.strip().lower() in ("", "all"):
        return list(ALL_MODES)
    wanted = [p.strip().upper() for p in text.split(",") if p.strip()]
    if not wanted:
        raise ValueError("empty mode list")
    unknown = [m for m in wanted if m not in ALL_MODES]
    if unknown:
        raise ValueError(f"unknown render mode(s): {unknown}; known: {list(ALL_MODES)}")
    seen = set(wanted)
    return [m for m in ALL_MODES if m in seen]


def resolve_mesh_count(n: int | None) -> int:
    if n is None:
        return MAX_MESHES
    if not isinstance(n, int) or isinstance(n, bool):
        raise ValueError(f"meshes must be an int, got {type(n).__name__}")
    if not 1 <= n <= MAX_MESHES:
        raise ValueError(f"meshes {n} outside [1, {MAX_MESHES}]")
    return n


def resolve_camera(name: str) -> CameraPreset:
    if name not in CAMERA_PRESETS:
        raise ValueError(f"unknown camera preset {name!r}; known: {sorted(CAMERA_PRESETS)}")
    return CAMERA_PRESETS[name]


def stl_dir() -> Path:
    return REPO_ROOT / STL_SUBDIR


def mesh_paths(count: int) -> list[tuple[str, str, Path]]:
    """(fma_id, label, path) for the first *count* fixed meshes.

    Raises CaptureError listing every missing file rather than skipping them:
    a capture of a subset of the fixed scene is not the fixed scene.
    """
    out, missing = [], []
    for fma_id, label in FIXED_MESHES[:count]:
        p = stl_dir() / f"{fma_id}.stl"
        if not p.is_file():
            missing.append(str(p))
        out.append((fma_id, label, p))
    if missing:
        raise CaptureError(
            f"{len(missing)} of the {count} fixed meshes are missing:\n  "
            + "\n  ".join(missing)
            + "\nThe fixed scene cannot be rendered.  Check the assets/stl symlink."
        )
    return out


def stl_triangle_count(path: Path) -> int:
    """Triangle count from the binary STL header (no parse, no allocation)."""
    with open(path, "rb") as f:
        f.seek(80)
        return int.from_bytes(f.read(4), "little")


def git_commit() -> str:
    """Short commit hash, or a marker.  Never raises."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=10, check=False,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
        return "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def git_dirty() -> bool | None:
    """True if the tree has uncommitted changes.  None if unknown.

    Recorded in the manifest because 'same commit' is not 'same code' when the
    tree is dirty -- and a golden-image loop exists precisely to be run on a
    dirty tree, so this must be visible rather than assumed.
    """
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=15, check=False,
        )
        if r.returncode != 0:
            return None
        return bool(r.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None


def camera_placement(
    centroid: np.ndarray, radius: float, preset: CameraPreset
) -> tuple[list[float], list[float], list[float]]:
    """Absolute (eye, target, up) for a preset against a subject's bounds."""
    d = np.asarray(preset.direction, dtype=np.float64)
    norm = float(np.linalg.norm(d))
    if norm == 0.0:
        raise ValueError("camera preset direction must be non-zero")
    eye = centroid + (d / norm) * radius * preset.distance
    return (
        [round(float(v), 6) for v in eye],
        [round(float(v), 6) for v in centroid],
        [float(v) for v in preset.up],
    )


def frame_content_fraction(img: np.ndarray, clear_rgb8: tuple[int, int, int]) -> float:
    """Fraction of pixels whose RGB differs perceptibly from the clear colour.

    Tolerance 6/255 absorbs the dither and rounding a rasteriser applies to a
    flat background without hiding real geometry, whose darkest shaded pixels
    in these modes sit far further from the background than that.
    """
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError(f"expected HxWx3+ image, got shape {img.shape}")
    bg = np.asarray(clear_rgb8, dtype=np.int16)
    delta = np.abs(img[:, :, :3].astype(np.int16) - bg).max(axis=2)
    return float((delta > 6).mean())


def validate_frame(mode: str, img: np.ndarray, clear_rgb8: tuple[int, int, int]) -> float:
    """Raise CaptureError if *img* is blank or degenerate.  Returns content fraction."""
    flat = img.reshape(-1, img.shape[2])
    if len(np.unique(flat, axis=0)) <= 1:
        raise CaptureError(
            f"mode {mode}: frame is a single uniform colour "
            f"{flat[0].tolist()} -- the render produced nothing.  Refusing to write it."
        )
    frac = frame_content_fraction(img, clear_rgb8)
    if frac < MIN_CONTENT_FRACTION:
        raise CaptureError(
            f"mode {mode}: only {frac * 100:.4f}% of pixels differ from the clear "
            f"colour (floor {MIN_CONTENT_FRACTION * 100:.3f}%) -- effectively a blank "
            "frame.  Refusing to write it."
        )
    return frac


def prepare_out_dir(out: Path, force: bool) -> Path:
    """Create *out*, refusing to clobber existing content unless *force*.

    Returns the staging directory.  Nothing is moved into *out* until the
    caller calls :func:`commit_staging`.
    """
    out = Path(out)
    if out.exists():
        if not out.is_dir():
            raise CaptureError(f"--out {out} exists and is not a directory")
        existing = [p.name for p in out.iterdir() if p.name != ".partial"]
        if existing and not force:
            raise CaptureError(
                f"--out {out} is not empty ({len(existing)} entries, e.g. "
                f"{sorted(existing)[:4]}).\n"
                "Refusing to overwrite: tools/capture_gui_screenshots.py destroyed 11 "
                "tracked README images this way.  Pass --force if you are certain, or "
                "capture into a fresh directory."
            )
    staging = out / ".partial"
    if staging.exists():
        # Leftovers from an earlier failed run.  Safe to clear: this directory
        # is created by this script and holds nothing else.
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    return staging


def commit_staging(staging: Path, out: Path, names: list[str]) -> list[Path]:
    """Move validated files from staging into *out*.  Called only on success."""
    moved = []
    for name in names:
        src = staging / name
        if not src.is_file():
            raise CaptureError(f"staged file missing at commit time: {src}")
        dst = out / name
        os.replace(src, dst)
        moved.append(dst)
    shutil.rmtree(staging, ignore_errors=True)
    return moved


def build_manifest(
    *,
    modes: list[str],
    meshes: list[tuple[str, str, Path]],
    tri_counts: list[int],
    size: tuple[int, int],
    camera_name: str,
    eye: list[float],
    target: list[float],
    up: list[float],
    fov: float,
    near: float,
    far: float,
    clear_rgb8: tuple[int, int, int],
    gl_info: dict,
    content_fractions: dict[str, float],
    frame_ms: dict[str, float],
    label: str,
) -> dict:
    """The manifest is the contract compare_golden checks before diffing."""
    return {
        "schema_version": 1,
        "tool": "tools/capture_golden.py",
        "label": label,
        "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "gl": gl_info,
        "viewport": {"width": size[0], "height": size[1]},
        "clear_color_rgb8": list(clear_rgb8),
        "camera": {
            "preset": camera_name,
            "eye": eye,
            "target": target,
            "up": up,
            "fov_deg": fov,
            "near": near,
            "far": far,
        },
        "lighting": {
            "setup": "LightSetup() defaults",
            "point_light_enabled": False,
        },
        "determinism": {
            "mesh_list": "explicit FIXED_MESHES, never globbed",
            "rng_used": False,
            "seed": None,
        },
        "modes": modes,
        "meshes": [
            {"fma_id": fid, "label": lab, "file": p.name, "triangles": t}
            for (fid, lab, p), t in zip(meshes, tri_counts, strict=True)
        ],
        "total_triangles": int(sum(tri_counts)),
        "content_fraction": {k: round(v, 6) for k, v in content_fractions.items()},
        "frame_ms": {k: round(v, 3) for k, v in frame_ms.items()},
        "files": {m: f"{m}.png" for m in modes},
    }


def manifest_comparability_key(manifest: dict) -> dict:
    """The subset of a manifest that two captures must share to be diffable.

    Deliberately excludes ``git_commit``: comparing across commits is the
    entire point of a regression check.  Includes the GL renderer string,
    because the pixel-level noise floor between two different rasterisers is
    unmeasured and a cross-driver diff would report change everywhere.
    """
    return {
        "viewport": manifest.get("viewport"),
        "camera": {
            k: v for k, v in (manifest.get("camera") or {}).items()
            if k in ("preset", "eye", "target", "up", "fov_deg", "near", "far")
        },
        "modes": manifest.get("modes"),
        "meshes": [
            {"fma_id": m.get("fma_id"), "file": m.get("file"), "triangles": m.get("triangles")}
            for m in (manifest.get("meshes") or [])
        ],
        "clear_color_rgb8": manifest.get("clear_color_rgb8"),
        "gl_renderer": (manifest.get("gl") or {}).get("gl_renderer"),
    }


# ----------------------------------------------------------------------------
# The GL path.  Everything above is exercised by --selftest; this is not.
# ----------------------------------------------------------------------------

def _make_fbo(width: int, height: int):
    """Create and bind a colour+depth FBO.  Raises if incomplete."""
    from OpenGL.GL import (
        GL_COLOR_ATTACHMENT0, GL_DEPTH_ATTACHMENT, GL_DEPTH_COMPONENT24,
        GL_FRAMEBUFFER, GL_FRAMEBUFFER_COMPLETE, GL_RENDERBUFFER, GL_RGBA,
        GL_RGBA8, GL_TEXTURE_2D, GL_UNSIGNED_BYTE,
        glBindFramebuffer, glBindRenderbuffer, glBindTexture,
        glCheckFramebufferStatus, glFramebufferRenderbuffer,
        glFramebufferTexture2D, glGenFramebuffers, glGenRenderbuffers,
        glGenTextures, glRenderbufferStorage, glTexImage2D,
    )

    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
    rbo = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, rbo)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo)
    status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
    if status != GL_FRAMEBUFFER_COMPLETE:
        raise CaptureError(
            f"framebuffer incomplete: status=0x{int(status):04X} at {width}x{height}. "
            "Nothing was rendered; refusing to continue."
        )
    return fbo, tex, rbo


def _read_frame(width: int, height: int) -> np.ndarray:
    """glReadPixels into an HxWx4 uint8 array, flipped to top-down image order."""
    from OpenGL.GL import GL_RGBA, GL_UNSIGNED_BYTE, glFinish, glReadPixels

    glFinish()
    raw = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)
    # GL origin is bottom-left; PNG is top-left.
    return np.flipud(arr).copy()


def capture(
    out: Path,
    *,
    modes: list[str] | None = None,
    mesh_count: int | None = None,
    size: tuple[int, int] = DEFAULT_SIZE,
    camera_name: str = DEFAULT_CAMERA,
    label: str = "capture",
    force: bool = False,
    prefer: str = "auto",
) -> dict:
    """Render the fixed scene in every requested mode.  Returns the manifest.

    Raises :class:`CaptureError` (or :class:`tools.glcontext.GLContextError`)
    rather than writing anything partial or blank.
    """
    from PIL import Image

    from tools.glcontext import acquire_offscreen_gl

    modes = list(ALL_MODES) if modes is None else parse_modes(",".join(modes))
    count = resolve_mesh_count(mesh_count)
    preset = resolve_camera(camera_name)
    width, height = size
    if not MIN_SIZE <= width <= MAX_SIZE or not MIN_SIZE <= height <= MAX_SIZE:
        raise ValueError(f"size {width}x{height} outside [{MIN_SIZE}, {MAX_SIZE}]")

    entries = mesh_paths(count)
    out = Path(out)
    staging = prepare_out_dir(out, force)

    gl_info = acquire_offscreen_gl(prefer)
    logger.info("%s", gl_info.banner())
    if gl_info.is_software:
        logger.warning(
            "software rasteriser: frame times below are CPU numbers and are NOT "
            "renderer performance"
        )

    # Imported after the context exists: these modules touch GL at call time,
    # and importing them first only makes a context failure harder to read.
    from faceforge.core.material import Material, RenderMode
    from faceforge.core.math_utils import vec3
    from faceforge.core.mesh import MeshInstance
    from faceforge.core.scene_graph import Scene, SceneNode
    from faceforge.loaders.stl_parser import load_stl_file
    from faceforge.rendering.camera import Camera
    from faceforge.rendering.lights import LightSetup
    from faceforge.rendering.renderer import GLRenderer

    clear_rgb8 = tuple(int(round(c * 255)) for c in GLRenderer.CLEAR_COLOR[:3])

    _make_fbo(width, height)

    renderer = GLRenderer()
    renderer.init_gl()
    renderer.resize(width, height)

    scene = Scene()
    tri_counts: list[int] = []
    mesh_instances: list[MeshInstance] = []
    for fma_id, lab, path in entries:
        geom = load_stl_file(path)
        header_tris = stl_triangle_count(path)
        if geom.triangle_count != header_tris:
            raise CaptureError(
                f"{path.name}: STL header declares {header_tris} triangles but the "
                f"parsed geometry has {geom.triangle_count}.  The assets changed; "
                "this capture would not be comparable to a stored reference."
            )
        tri_counts.append(header_tris)
        mi = MeshInstance(
            name=lab, geometry=geom,
            material=Material(color=(0.82, 0.76, 0.68), opacity=1.0),
            source_id=fma_id,
        )
        node = SceneNode(name=fma_id)
        node.mesh = mi
        scene.add(node)
        mesh_instances.append(mi)

    all_pos = np.concatenate([m.positions.reshape(-1, 3) for m in mesh_instances])
    centroid = all_pos.mean(axis=0).astype(np.float64)
    radius = float(np.linalg.norm(all_pos - centroid, axis=1).max())
    eye, target, up = camera_placement(centroid, radius, preset)

    camera = Camera()
    camera.set_aspect(width, height)
    camera.look_at(vec3(*eye), vec3(*target), vec3(*up))
    lights = LightSetup()

    logger.info(
        "scene: %d meshes, %d triangles, camera=%s, %dx%d, %d modes",
        len(entries), sum(tri_counts), camera_name, width, height, len(modes),
    )

    content: dict[str, float] = {}
    frame_ms: dict[str, float] = {}
    written: list[str] = []

    try:
        for mode_name in modes:
            mode = RenderMode[mode_name]
            for mi in mesh_instances:
                mi.material.render_mode = mode
            t0 = time.perf_counter()
            renderer.render(scene, camera, lights)
            img = _read_frame(width, height)
            frame_ms[mode_name] = (time.perf_counter() - t0) * 1000.0
            content[mode_name] = validate_frame(mode_name, img, clear_rgb8)
            fname = f"{mode_name}.png"
            Image.fromarray(img, mode="RGBA").save(staging / fname, format="PNG", optimize=False)
            written.append(fname)
            logger.info(
                "  %-13s content=%5.2f%%  %7.1f ms  -> %s",
                mode_name, content[mode_name] * 100.0, frame_ms[mode_name], fname,
            )
    except Exception:
        logger.error(
            "capture failed after %d/%d modes; leaving %s in place and writing no "
            "manifest, so the directory cannot be mistaken for a valid capture",
            len(written), len(modes), staging,
        )
        raise

    manifest = build_manifest(
        modes=modes, meshes=entries, tri_counts=tri_counts, size=(width, height),
        camera_name=camera_name, eye=eye, target=target, up=up,
        fov=float(camera.fov), near=float(camera.near), far=float(camera.far),
        clear_rgb8=clear_rgb8, gl_info=gl_info.as_manifest_dict(),
        content_fractions=content, frame_ms=frame_ms, label=label,
    )
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    commit_staging(staging, out, [*written, "manifest.json"])
    logger.info("capture complete: %d PNGs + manifest.json in %s", len(written), out)
    return manifest


# ----------------------------------------------------------------------------
# Selftest: everything except the GL calls, so it runs where GL is absent.
# ----------------------------------------------------------------------------

def selftest() -> int:
    """Verify every non-GL invariant.  Returns a process exit code."""
    failures: list[str] = []

    def check(name: str, fn) -> None:
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as exc:
            failures.append(f"{name}: {exc}")
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001 - selftest reports, never crashes
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")

    print("capture_golden --selftest (no GL context required)")

    def modes_match_enum() -> None:
        from faceforge.core.material import RenderMode
        enum_names = [m.name for m in RenderMode]
        assert list(ALL_MODES) == enum_names, (
            f"ALL_MODES drifted from RenderMode.\n  here: {list(ALL_MODES)}\n  enum: {enum_names}"
        )
        assert len(ALL_MODES) == 16, f"expected 16 modes, have {len(ALL_MODES)}"
    check("ALL_MODES matches RenderMode enum exactly (16 modes)", modes_match_enum)

    def clear_colour_matches() -> None:
        from faceforge.rendering.renderer import GLRenderer
        got = tuple(int(round(c * 255)) for c in GLRenderer.CLEAR_COLOR[:3])
        assert got == EXPECTED_CLEAR_RGB8, f"clear colour is {got}, expected {EXPECTED_CLEAR_RGB8}"
    check("clear colour matches GLRenderer.CLEAR_COLOR", clear_colour_matches)

    def mesh_list_sane() -> None:
        ids = [m[0] for m in FIXED_MESHES]
        assert len(ids) == len(set(ids)), "FIXED_MESHES contains duplicates"
        assert len(FIXED_MESHES) == 16, f"expected 16 fixed meshes, have {len(FIXED_MESHES)}"
    check("FIXED_MESHES is 16 unique ids", mesh_list_sane)

    def mesh_files_exist() -> None:
        entries = mesh_paths(MAX_MESHES)
        tris = [stl_triangle_count(p) for _, _, p in entries]
        assert all(t > 0 for t in tris), f"some meshes report 0 triangles: {tris}"
        print(f"        {len(entries)} meshes, {sum(tris)} triangles total")
    check("all 16 fixed STL files exist and have triangles", mesh_files_exist)

    def size_parsing() -> None:
        assert parse_size("512x512") == (512, 512)
        assert parse_size("1920X1080") == (1920, 1080)
        for bad in ("0x0", "63x64", "5000x100", "abc", "512", "512x", "-1x-1", "512x512x512"):
            try:
                parse_size(bad)
            except ValueError:
                continue
            raise AssertionError(f"parse_size accepted {bad!r}")
    check("parse_size accepts valid and rejects invalid sizes", size_parsing)

    def mode_parsing() -> None:
        assert parse_modes(None) == list(ALL_MODES)
        assert parse_modes("all") == list(ALL_MODES)
        assert parse_modes("XRAY,SOLID") == ["SOLID", "XRAY"], "order must normalise"
        assert parse_modes("solid") == ["SOLID"], "must be case-insensitive"
        for bad in ("NOPE", "SOLID,NOPE", ","):
            try:
                parse_modes(bad)
            except ValueError:
                continue
            raise AssertionError(f"parse_modes accepted {bad!r}")
    check("parse_modes normalises order and rejects unknown modes", mode_parsing)

    def mesh_count_clamping() -> None:
        assert resolve_mesh_count(None) == MAX_MESHES
        assert resolve_mesh_count(1) == 1
        for bad in (0, -3, MAX_MESHES + 1, 1.5, "8", True):
            try:
                resolve_mesh_count(bad)
            except ValueError:
                continue
            raise AssertionError(f"resolve_mesh_count accepted {bad!r}")
    check("resolve_mesh_count rejects out-of-range and wrong types", mesh_count_clamping)

    def camera_presets_valid() -> None:
        assert DEFAULT_CAMERA in CAMERA_PRESETS
        for name, p in CAMERA_PRESETS.items():
            d = np.asarray(p.direction, float)
            u = np.asarray(p.up, float)
            assert np.linalg.norm(d) > 0, f"{name}: zero direction"
            assert np.linalg.norm(u) > 0, f"{name}: zero up"
            cross = np.linalg.norm(np.cross(d / np.linalg.norm(d), u / np.linalg.norm(u)))
            assert cross > 1e-3, f"{name}: up is parallel to view direction (degenerate)"
            assert p.distance > 1.0, f"{name}: distance {p.distance} puts eye inside the subject"
        try:
            resolve_camera("nope")
        except ValueError:
            pass
        else:
            raise AssertionError("resolve_camera accepted an unknown preset")
    check("camera presets are non-degenerate and outside the subject", camera_presets_valid)

    def placement_is_deterministic() -> None:
        c = np.array([1.0, 2.0, 3.0])
        a = camera_placement(c, 10.0, CAMERA_PRESETS["oblique"])
        b = camera_placement(c, 10.0, CAMERA_PRESETS["oblique"])
        assert a == b, "camera_placement is not deterministic"
        eye = np.asarray(a[0])
        assert abs(np.linalg.norm(eye - c) - 29.0) < 1e-6, (
            f"oblique eye distance {np.linalg.norm(eye - c)} != radius*2.9"
        )
    check("camera_placement is deterministic and honours the distance factor", placement_is_deterministic)

    def blank_detection() -> None:
        clear = EXPECTED_CLEAR_RGB8
        uniform = np.zeros((8, 8, 4), np.uint8)
        uniform[:, :, :3] = clear
        uniform[:, :, 3] = 255
        try:
            validate_frame("SOLID", uniform, clear)
        except CaptureError:
            pass
        else:
            raise AssertionError("a uniform clear-colour frame was accepted")
        black = np.zeros((8, 8, 4), np.uint8)
        black[:, :, 3] = 255
        try:
            validate_frame("SOLID", black, clear)
        except CaptureError:
            pass
        else:
            raise AssertionError("a uniform black frame was accepted")
        # One lit pixel in 64 is 1.56%, above the 0.1% floor.
        one = uniform.copy()
        one[4, 4, :3] = (200, 180, 160)
        frac = validate_frame("SOLID", one, clear)
        assert abs(frac - 1.0 / 64.0) < 1e-9, f"content fraction {frac} != 1/64"
        # Sub-floor content must still be rejected.
        sparse = np.zeros((200, 200, 4), np.uint8)
        sparse[:, :, :3] = clear
        sparse[:, :, 3] = 255
        sparse[0, 0, :3] = (255, 255, 255)   # 1/40000 = 0.0025%, below floor
        try:
            validate_frame("SOLID", sparse, clear)
        except CaptureError:
            pass
        else:
            raise AssertionError("a frame with sub-floor content was accepted")
    check("validate_frame rejects uniform and sub-floor frames", blank_detection)

    def content_tolerance() -> None:
        clear = EXPECTED_CLEAR_RGB8
        img = np.zeros((10, 10, 4), np.uint8)
        img[:, :, :3] = clear
        img[:, :, 3] = 255
        img[0, 0, :3] = tuple(c + 5 for c in clear)   # within tolerance
        assert frame_content_fraction(img, clear) == 0.0, "5/255 dither counted as content"
        img[0, 1, :3] = tuple(c + 40 for c in clear)  # real content
        assert frame_content_fraction(img, clear) == 0.01
    check("frame_content_fraction ignores dither but not geometry", content_tolerance)

    def out_dir_guard() -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "cap"
            st = prepare_out_dir(d, force=False)
            assert st.name == ".partial" and st.is_dir()
            (st / "x.png").write_bytes(b"stub")
            # Committing moves staged files and removes staging.
            commit_staging(st, d, ["x.png"])
            assert (d / "x.png").is_file() and not st.exists()
            # A now non-empty directory must be refused without --force.
            try:
                prepare_out_dir(d, force=False)
            except CaptureError:
                pass
            else:
                raise AssertionError("prepare_out_dir clobbered a non-empty directory")
            st2 = prepare_out_dir(d, force=True)
            assert st2.is_dir(), "--force did not produce a staging directory"
            assert (d / "x.png").is_file(), "--force destroyed existing files before rendering"
    check("prepare_out_dir refuses to clobber; --force stages without destroying", out_dir_guard)

    def commit_is_atomic_in_intent() -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "cap"
            st = prepare_out_dir(d, force=False)
            (st / "a.png").write_bytes(b"a")
            try:
                commit_staging(st, d, ["a.png", "missing.png"])
            except CaptureError:
                pass
            else:
                raise AssertionError("commit_staging accepted a missing staged file")
    check("commit_staging fails loudly on a missing staged file", commit_is_atomic_in_intent)

    def manifest_shape() -> None:
        entries = mesh_paths(3)
        tris = [stl_triangle_count(p) for _, _, p in entries]
        m = build_manifest(
            modes=["SOLID", "XRAY"], meshes=entries, tri_counts=tris, size=(256, 128),
            camera_name="anterior", eye=[1.0, 2.0, 3.0], target=[0.0, 0.0, 0.0],
            up=[0.0, 0.0, 1.0], fov=45.0, near=1.0, far=5000.0,
            clear_rgb8=EXPECTED_CLEAR_RGB8,
            gl_info={"gl_renderer": "test", "kind": "none"},
            content_fractions={"SOLID": 0.2, "XRAY": 0.1},
            frame_ms={"SOLID": 1.0, "XRAY": 2.0}, label="selftest",
        )
        for key in ("schema_version", "git_commit", "gl", "viewport", "camera",
                    "modes", "meshes", "total_triangles", "files", "determinism"):
            assert key in m, f"manifest missing {key}"
        assert m["total_triangles"] == sum(tris)
        assert m["viewport"] == {"width": 256, "height": 128}
        assert m["files"] == {"SOLID": "SOLID.png", "XRAY": "XRAY.png"}
        assert m["determinism"]["rng_used"] is False
        json.dumps(m)  # must be serialisable
        k = manifest_comparability_key(m)
        assert "git_commit" not in k, "comparability key must not include the commit"
        assert k["gl_renderer"] == "test"
    check("build_manifest is complete and JSON-serialisable", manifest_shape)

    def git_metadata() -> None:
        c = git_commit()
        assert isinstance(c, str) and c, "git_commit returned nothing"
        d = git_dirty()
        assert d is None or isinstance(d, bool)
        print(f"        commit={c} dirty={d}")
    check("git metadata collection never raises", git_metadata)

    print()
    if failures:
        print(f"SELFTEST FAILED: {len(failures)} problem(s)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("SELFTEST PASSED — all non-GL invariants hold.")
    print("NOT covered by selftest: FBO creation, shader compilation, draw calls,")
    print("glReadPixels.  Those need a GL context; run without --selftest to exercise them.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="capture_golden",
        description="Render the fixed FaceForge scene to PNG via an FBO, plus a manifest.",
    )
    p.add_argument("--out", type=Path, help="output directory (must be empty unless --force)")
    p.add_argument("--modes", default=None,
                   help=f"comma-separated subset of {','.join(ALL_MODES)} (default: all)")
    p.add_argument("--meshes", type=int, default=None,
                   help=f"how many of the {MAX_MESHES} fixed meshes to load (default: all)")
    p.add_argument("--size", default=f"{DEFAULT_SIZE[0]}x{DEFAULT_SIZE[1]}", help="WxH")
    p.add_argument("--camera", default=DEFAULT_CAMERA, choices=sorted(CAMERA_PRESETS))
    p.add_argument("--label", default="capture", help="label recorded in the manifest")
    p.add_argument("--force", action="store_true", help="allow writing into a non-empty --out")
    p.add_argument("--prefer", default="auto", choices=("auto", "hardware", "software"),
                   help="GL context preference; 'software' is reproducible across machines")
    p.add_argument("--selftest", action="store_true",
                   help="verify all non-GL invariants and exit (no GL context needed)")
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s", stream=sys.stdout,
    )

    if args.selftest:
        return selftest()

    if args.out is None:
        p.error("--out is required (or pass --selftest)")

    try:
        size = parse_size(args.size)
        modes = parse_modes(args.modes)
        capture(
            args.out, modes=modes, mesh_count=args.meshes, size=size,
            camera_name=args.camera, label=args.label, force=args.force, prefer=args.prefer,
        )
    except Exception as exc:  # noqa: BLE001 - top-level CLI reporting
        logger.error("CAPTURE FAILED: %s: %s", type(exc).__name__, exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
