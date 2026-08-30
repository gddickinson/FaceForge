#!/usr/bin/env python3
"""Real-GPU frame-time benchmark for the FaceForge render pipeline.

Creates a genuine OpenGL 3.3 core context through PySide6, loads a configurable
number of real BodyParts3D STL meshes into a FaceForge ``Scene``, and reports
true frame times per render mode and per mesh count.  Use it to validate (or
refute) the Python-side predictions produced headlessly by the glrec harness,
which cannot see driver or GPU cost at all.

Requires a machine with a window server.  On macOS that means running it from a
normal desktop session -- not over plain ssh, and not with
``QT_QPA_PLATFORM=offscreen`` (Qt's offscreen and minimal plugins refuse to
create a GL context; the cocoa platform hangs when no window server is
reachable rather than reporting an error, so a hang here means the session
cannot see a display).

Examples
--------
    # default sweep: 50/200/500/900 meshes, SOLID mode
    python bench_render_real.py

    # every render mode at 500 meshes, 300 frames each, 1920x1200
    python bench_render_real.py --meshes 500 --modes all --frames 300 \
        --size 1920x1200 --csv frames.csv

    # A/B the prototype optimisations against the shipped renderer
    python bench_render_real.py --meshes 500 --compare-fast

    # verify data loading and scene assembly without touching the GPU
    python bench_render_real.py --selftest

Reported per row
----------------
median / p95 / min frame time in ms and the implied fps.  The median is the
number to quote; p95 exposes hitching.  ``glFinish()`` is called before each
timestamp so the GPU is drained and the measurement is a true frame time rather
than a command-submission time.
"""
from __future__ import annotations

import argparse
import csv
import random
import statistics
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------
# Repo discovery: work whether or not faceforge is installed on the path
# --------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent


def _add_repo_to_path(repo: Path | None) -> Path:
    candidates = []
    if repo is not None:
        candidates.append(Path(repo).expanduser().resolve())
    candidates += [_HERE, _HERE.parent, Path.cwd(),
                   Path("/Users/george/Documents/GitHub/FaceForge")]
    for c in candidates:
        for root in (c, *c.parents):
            if (root / "src" / "faceforge" / "__init__.py").is_file():
                sys.path.insert(0, str(root / "src"))
                return root
    raise SystemExit(
        "Could not locate the FaceForge repo. Pass --repo /path/to/FaceForge.")


# --------------------------------------------------------------------------
def parse_size(text: str) -> tuple[int, int]:
    w, _, h = text.lower().partition("x")
    return int(w), int(h)


def build_scene(stl_dir: Path, n_meshes: int, seed: int = 0, verbose: bool = True):
    """Load *n_meshes* real STL files and hang them off a scene graph.

    The layout mirrors FaceForge's own: a flat-ish tree of anatomy groups, each
    holding ~25 structure nodes, every structure its own MeshInstance with its
    own colour -- which is what makes the renderer's per-mesh uniform traffic
    unavoidable in the unoptimised path.
    """
    from faceforge.core.material import Material, RenderMode
    from faceforge.core.mesh import MeshInstance
    from faceforge.core.scene_graph import Scene, SceneNode
    from faceforge.loaders.stl_parser import load_stl_file

    files = sorted(stl_dir.glob("*.stl"))
    if not files:
        raise SystemExit(f"No .stl files under {stl_dir}")
    rng = random.Random(seed)
    if n_meshes <= len(files):
        chosen = files[:n_meshes]
    else:                       # reuse geometry when asked for more than exist
        chosen = files + [rng.choice(files) for _ in range(n_meshes - len(files))]

    scene = Scene()
    body = SceneNode("bodyRoot")
    scene.add(body)
    meshes = []
    group = None
    tris = 0
    t0 = time.perf_counter()
    for i, path in enumerate(chosen):
        if i % 25 == 0:
            group = SceneNode(f"anatomyGroup_{i // 25}")
            body.add(group)
        geom = load_stl_file(path, indexed=True)
        tris += geom.triangle_count
        # Mirror what stl_batch_loader.py does for a config entry with no
        # explicit opacity: opacity 0.7, transparent True.  That is the state
        # 941 of the 950 configured structures are actually in.
        mat = Material(color=(rng.uniform(0.35, 0.95), rng.uniform(0.3, 0.8),
                              rng.uniform(0.3, 0.8)),
                       opacity=0.7, transparent=True,
                       render_mode=RenderMode.SOLID)
        m = MeshInstance(name=f"{path.stem}_{i}", geometry=geom, material=mat)
        node = SceneNode(f"node_{path.stem}_{i}")
        node.mesh = m
        group.add(node)
        meshes.append(m)
        if verbose and (i + 1) % 100 == 0:
            print(f"    loaded {i + 1}/{len(chosen)} meshes", flush=True)
    if verbose:
        print(f"    {len(meshes)} meshes, {tris:,} triangles, "
              f"{time.perf_counter() - t0:.1f} s to load", flush=True)
    return scene, meshes, tris


def frame_scene(scene, camera) -> None:
    """Point the camera at the scene's bounding box so nothing is culled away."""
    import numpy as np
    pts = []
    for mesh, world in scene.collect_meshes():
        p = np.asarray(mesh.geometry.positions, dtype=np.float64).reshape(-1, 3)
        lo, hi = p.min(axis=0), p.max(axis=0)
        pts.append(world[:3, :3] @ lo + world[:3, 3])
        pts.append(world[:3, :3] @ hi + world[:3, 3])
    pts = np.array(pts)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center = (lo + hi) * 0.5
    radius = float(np.linalg.norm(hi - center)) or 1.0
    dist = radius / np.tan(np.radians(camera.fov * 0.5)) * 1.15
    camera.far = max(camera.far, dist + radius * 3)
    camera.set_target(*center)
    camera.set_position(center[0] + dist * 0.6, center[1] - dist * 0.7,
                        center[2] + dist * 0.35)


# --------------------------------------------------------------------------
class BenchWidget:
    """Owns the GL context and runs the timing loop inside paintGL."""

    def __init__(self, size, frames, warmup, orbit, use_fast, fast_opts):
        from PySide6.QtOpenGLWidgets import QOpenGLWidget
        from faceforge.rendering.gl_widget import create_gl_format

        self.size = size
        self.frames = frames
        self.warmup = warmup
        self.orbit = orbit
        self.use_fast = use_fast
        self.fast_opts = fast_opts

        outer = self

        class _W(QOpenGLWidget):
            def __init__(self):
                super().__init__()
                self.setFormat(create_gl_format())
                self.resize(*outer.size)
                self.ready = False
                self.times: list[float] = []
                self.gl_version = ""
                self.renderer_name = ""
                self.scene = None
                self.camera = None
                self.lights = None
                self.renderer = None
                self.done = False

            def initializeGL(self):
                from OpenGL.GL import (GL_RENDERER, GL_VERSION, glGetString)
                self.gl_version = glGetString(GL_VERSION).decode()
                self.renderer_name = glGetString(GL_RENDERER).decode()
                self.renderer.init_gl()
                self.ready = True

            def resizeGL(self, w, h):
                dpr = self.devicePixelRatio()
                self.camera.set_aspect(w, h)
                self.renderer.resize(int(w * dpr), int(h * dpr))

            def paintGL(self):
                from OpenGL.GL import glFinish
                if self.done:
                    return
                n = outer.frames + outer.warmup
                for i in range(n):
                    if outer.orbit:
                        outer.spin(self.camera, i)
                    t0 = time.perf_counter()
                    self.renderer.render(self.scene, self.camera, self.lights)
                    glFinish()
                    dt = (time.perf_counter() - t0) * 1e3
                    if i >= outer.warmup:
                        self.times.append(dt)
                self.done = True

        self.widget = _W()

    @staticmethod
    def spin(camera, i):
        """Rotate the camera a little each frame: a static camera lets a
        dirty-flag scene graph and any driver-side caching look better than
        they are in interactive use."""
        import numpy as np
        d = camera.position - camera.target
        r = np.hypot(d[0], d[1])
        a = np.arctan2(d[1], d[0]) + 0.01
        camera.set_position(camera.target[0] + r * np.cos(a),
                            camera.target[1] + r * np.sin(a),
                            camera.position[2])


class PrototypeUnavailable(RuntimeError):
    """--compare-fast was requested but renderer_fast.py is not installed."""


def make_renderer(use_fast: bool, fast_opts: dict):
    if not use_fast:
        from faceforge.rendering.renderer import GLRenderer
        return GLRenderer(), "shipped"
    sys.path.insert(0, str(_HERE))
    try:
        from renderer_fast import FastRenderer      # prototype, same folder
    except ModuleNotFoundError as exc:
        raise PrototypeUnavailable(
            "--compare-fast needs renderer_fast.py next to this script "
            f"(looked in {_HERE}).\n"
            "That module is the standalone optimisation prototype and is not "
            "part of the repo.\n"
            "Once the optimisations are merged into "
            "faceforge/rendering/renderer.py you no longer need it: just run "
            "this script twice, before and after, and compare the CSVs."
        ) from exc
    return FastRenderer(**fast_opts), "prototype"


def run_case(app, stl_dir, n_meshes, mode_name, size, frames, warmup, orbit,
             use_fast, fast_opts, scene_cache, force_opaque=False):
    from faceforge.core.material import RenderMode
    from faceforge.rendering.camera import Camera
    from faceforge.rendering.lights import LightSetup

    key = (n_meshes, use_fast)
    if key not in scene_cache:
        print(f"  building scene: {n_meshes} meshes "
              f"({'prototype' if use_fast else 'shipped'} graph)", flush=True)
        scene_cache[key] = build_scene(stl_dir, n_meshes)
    scene, meshes, tris = scene_cache[key]

    mode = RenderMode[mode_name]
    for m in meshes:
        m.material.render_mode = mode
        if force_opaque:
            m.material.transparent = False
            m.material.opacity = 1.0

    # Each case gets a fresh QOpenGLWidget, and Qt on macOS falls back to an
    # UNSHARED context ("Could not create NSOpenGLContext with shared
    # context").  VAO names are per-context by specification and are never
    # shared even when context sharing succeeds, so any handle cached on a
    # mesh by a previous case is invalid here and glBindVertexArray would
    # raise GL_INVALID_OPERATION (1282).
    #
    # Drop the stale handles so the new context re-uploads.  Deliberately do
    # NOT call GLMesh.destroy(): the owning context is already gone, so the
    # delete calls would be issued against the wrong context.  The old buffers
    # die with their context.
    for m in meshes:
        m.gl_handle = None

    camera = Camera()
    camera.set_aspect(*size)
    frame_scene(scene, camera)
    renderer, tag = make_renderer(use_fast, fast_opts)

    bw = BenchWidget(size, frames, warmup, orbit, use_fast, fast_opts)
    w = bw.widget
    w.scene, w.camera, w.lights, w.renderer = scene, camera, LightSetup(), renderer
    w.show()
    # pump the event loop until paintGL has finished the timing loop
    deadline = time.time() + 600
    while not w.done and time.time() < deadline:
        app.processEvents()
        if w.ready and not w.done:
            w.update()
        time.sleep(0.001)
    if use_fast and getattr(renderer, "opt", {}).get("batch_static"):
        pass    # batches are built lazily on the prototype's first frame
    t = w.times
    w.hide()
    if not t:
        raise SystemExit("paintGL never ran -- no GL context? See the docstring.")
    row = dict(renderer=tag, opaque_forced=force_opaque,
               meshes=n_meshes, triangles=tris, mode=mode_name,
               width=size[0], height=size[1], frames=len(t),
               median_ms=round(statistics.median(t), 3),
               p95_ms=round(sorted(t)[int(len(t) * 0.95) - 1], 3),
               min_ms=round(min(t), 3), max_ms=round(max(t), 3),
               fps_median=round(1000 / statistics.median(t), 1),
               gl_version=w.gl_version, gpu=w.renderer_name)
    return row


FAST_FULL = dict(no_diag=True, frame_uniforms=True, state_cache=True,
                 sort_state=True, rigid_normal=True, buf_reuse=True,
                 frustum_cull=True, trim_binds=True)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=None,
                    help="path to the FaceForge checkout (auto-detected otherwise)")
    ap.add_argument("--stl-dir", default=None,
                    help="directory of .stl files (default: <repo>/assets/stl)")
    ap.add_argument("--meshes", default="50,200,500,900",
                    help="comma-separated mesh counts (default 50,200,500,900)")
    ap.add_argument("--modes", default="SOLID",
                    help="comma-separated RenderMode names, or 'all'")
    ap.add_argument("--frames", type=int, default=200, help="timed frames per case")
    ap.add_argument("--warmup", type=int, default=30, help="untimed warm-up frames")
    ap.add_argument("--size", default="1600x1000", help="viewport WxH")
    ap.add_argument("--no-orbit", action="store_true",
                    help="hold the camera still (flatters transform caching)")
    ap.add_argument("--compare-fast", action="store_true",
                    help="also run the renderer_fast.py prototype for each case")
    ap.add_argument("--fast-only", action="store_true")
    ap.add_argument("--csv", default=None, help="write all rows to this CSV")
    ap.add_argument("--force-opaque", action="store_true",
                    help="set transparent=False, opacity=1.0 on every material. "
                         "stl_batch_loader.py:133 defaults opacity to 0.7 and "
                         "transparent to True, and 941 of the 950 configured "
                         "structures do not override it -- so the whole anatomy "
                         "normally renders alpha-blended with depth writes off. "
                         "This flag measures what making it opaque is worth on "
                         "real hardware, which the headless harness cannot see.")
    ap.add_argument("--selftest", action="store_true",
                    help="load data and build a scene, but never touch the GPU")
    args = ap.parse_args(argv)

    repo = _add_repo_to_path(args.repo)
    stl_dir = Path(args.stl_dir).expanduser() if args.stl_dir else repo / "assets" / "stl"
    if not stl_dir.is_dir():
        raise SystemExit(f"STL directory not found: {stl_dir}")

    from faceforge.core.material import RenderMode
    counts = [int(x) for x in args.meshes.split(",") if x.strip()]
    modes = ([m.name for m in RenderMode] if args.modes.strip().lower() == "all"
             else [m.strip().upper() for m in args.modes.split(",") if m.strip()])
    for m in modes:
        if m not in RenderMode.__members__:
            raise SystemExit(f"Unknown render mode {m!r}. "
                             f"Choose from: {', '.join(RenderMode.__members__)}")
    size = parse_size(args.size)

    print(f"repo:     {repo}")
    print(f"stl dir:  {stl_dir}  ({len(list(stl_dir.glob('*.stl')))} files)")
    print(f"counts:   {counts}")
    print(f"modes:    {', '.join(modes)}")
    print(f"viewport: {size[0]}x{size[1]}   frames: {args.frames} "
          f"(+{args.warmup} warm-up)   camera: "
          f"{'static' if args.no_orbit else 'orbiting'}")
    print()

    if args.selftest:
        scene, meshes, tris = build_scene(stl_dir, min(counts[0], 25))
        from faceforge.rendering.camera import Camera
        cam = Camera()
        cam.set_aspect(*size)
        frame_scene(scene, cam)
        print(f"SELFTEST OK: {len(meshes)} meshes, {tris:,} triangles, "
              f"{len(scene.collect_meshes())} collected")
        print(f"  camera at {cam.position.round(1)} looking at "
              f"{cam.target.round(1)}, far={cam.far:.0f}")
        print("  GPU was not touched. Run without --selftest on a machine with "
              "a display for real frame times.")
        return 0

    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv[:1])

    variants = []
    if not args.fast_only:
        variants.append((False, {}))
    if args.compare_fast or args.fast_only:
        # Fail before the first (slow) scene build rather than midway through.
        try:
            make_renderer(True, FAST_FULL)
        except PrototypeUnavailable as exc:
            if args.fast_only:
                print(f"\nERROR: {exc}", file=sys.stderr)
                return 2
            print(f"\nWARNING: {exc}\n"
                  "Continuing with the shipped renderer only.\n", file=sys.stderr)
        else:
            variants.append((True, FAST_FULL))

    rows = []
    cache: dict = {}
    for n in counts:
        for mode in modes:
            for use_fast, opts in variants:
                row = run_case(app, stl_dir, n, mode, size, args.frames,
                               args.warmup, not args.no_orbit, use_fast, opts,
                               cache, args.force_opaque)
                rows.append(row)
                print(f"  {row['renderer']:<9} {n:>4} meshes  {mode:<12} "
                      f"median {row['median_ms']:>7.2f} ms  "
                      f"p95 {row['p95_ms']:>7.2f} ms  "
                      f"{row['fps_median']:>6.1f} fps", flush=True)

    print()
    if rows:
        print(f"GL: {rows[0]['gl_version']}")
        print(f"GPU: {rows[0]['gpu']}")
    print()
    print(f"{'renderer':<10}{'meshes':>8}{'mode':<14}{'median ms':>11}"
          f"{'p95 ms':>9}{'fps':>8}")
    for r in rows:
        print(f"{r['renderer']:<10}{r['meshes']:>8}{r['mode']:<14}"
              f"{r['median_ms']:>11.2f}{r['p95_ms']:>9.2f}{r['fps_median']:>8.1f}")

    if args.compare_fast:
        print()
        print("shipped vs prototype (median frame time):")
        for n in counts:
            for mode in modes:
                a = next((r for r in rows if r["meshes"] == n and r["mode"] == mode
                          and r["renderer"] == "shipped"), None)
                b = next((r for r in rows if r["meshes"] == n and r["mode"] == mode
                          and r["renderer"] == "prototype"), None)
                if a and b:
                    print(f"  {n:>4} meshes {mode:<12} "
                          f"{a['median_ms']:>7.2f} -> {b['median_ms']:>7.2f} ms  "
                          f"({a['median_ms'] / b['median_ms']:.2f}x)")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
