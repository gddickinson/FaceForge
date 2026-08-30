"""One-command GPU frame-time benchmark -- run this on your own display.

    python tests/rendering/bench_gpu.py

Everything else in this directory measures the *Python* half of a frame through
the GL recorder, because the analysis sandbox has no graphics subsystem at all
(CGLChoosePixelFormat returns kCGLBadConnection, CGMainDisplayID() == 0,
MTLCreateSystemDefaultDevice() returns NULL).  That means the fragment-side
half of the rendering work -- which is what the opacity default and the
clip-plane change are aimed at -- could not be measured where these fixes were
written.  This script is how you measure it.

It opens a real window, builds a synthetic scene with a realistic triangle
budget, and reports median frame time for each configuration:

    opaque        depth writes on, early-Z active         <- the fixed default
    blended       glDepthMask(False), early-Z defeated    <- the old default
    clip on       cutaway enabled (gl_ClipDistance path)
    clip off      cutaway disabled

The opaque/blended pair is the measurement that matters: with 941 of ~950
structures previously defaulting to opacity 0.7, every frame ran with depth
writes off across the whole anatomy.

Options:
    --meshes N      structures in the scene            (default 500)
    --tris N        triangles per structure            (default 4000)
    --frames N      timed frames per configuration     (default 240)
    --mode NAME     render mode, e.g. SOLID, XRAY      (default SOLID)
    --size WxH      window size                        (default 1600x1000)
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np  # noqa: E402


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--meshes", type=int, default=500)
    p.add_argument("--tris", type=int, default=4000)
    p.add_argument("--frames", type=int, default=240)
    p.add_argument("--mode", default="SOLID")
    p.add_argument("--size", default="1600x1000")
    return p.parse_args(argv[1:])


def _build_scene(n_meshes: int, n_tris: int, mode, *, opacity: float,
                 transparent: bool):
    from faceforge.core.material import Material
    from faceforge.core.mesh import BufferGeometry, MeshInstance
    from faceforge.core.scene_graph import Scene, SceneNode

    rng = np.random.default_rng(0)
    scene = Scene()
    # A shell of outward-facing triangles per structure, so the depth complexity
    # is realistic: structures overlap in depth and occlude one another, which
    # is precisely the situation early-Z exists for.
    for i in range(n_meshes):
        n_verts = n_tris * 3
        centre = rng.normal(0.0, 55.0, 3)
        pos = (centre + rng.normal(0.0, 18.0, size=(n_verts, 3))).astype(np.float32)
        nrm = pos - centre.astype(np.float32)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-6)
        mat = Material(color=(0.55 + 0.3 * rng.random(), 0.45, 0.42),
                       opacity=opacity, transparent=transparent,
                       render_mode=mode, shininess=15.0)
        mesh = MeshInstance(
            name=f"structure_{i}",
            geometry=BufferGeometry(positions=pos.ravel(),
                                    normals=nrm.astype(np.float32).ravel(),
                                    vertex_count=n_verts),
            material=mat)
        node = SceneNode(name=f"node_{i}")
        node.mesh = mesh
        scene.add(node)
    scene.update()
    return scene


def main(argv: list[str]) -> int:
    args = _parse(argv)
    w, h = (int(x) for x in args.size.lower().split("x"))

    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtGui import QSurfaceFormat
        from PySide6.QtWidgets import QApplication
        from PySide6.QtOpenGLWidgets import QOpenGLWidget
    except ImportError as exc:            # pragma: no cover
        print(f"PySide6 is required: {exc}")
        return 2

    from OpenGL.GL import GL_RENDERER, GL_VENDOR, GL_VERSION, glGetString

    from faceforge.core.material import RenderMode
    from faceforge.rendering.camera import Camera
    from faceforge.rendering.lights import LightSetup
    from faceforge.rendering.renderer import GLRenderer

    mode = RenderMode[args.mode.upper()]

    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    fmt.setSwapInterval(0)                # vsync off, or every result is 60 fps
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv[:1])

    configs = [
        ("opaque,   clip off", dict(opacity=1.0, transparent=False), False),
        ("blended,  clip off", dict(opacity=0.7, transparent=True), False),
        ("opaque,   clip on ", dict(opacity=1.0, transparent=False), True),
        ("blended,  clip on ", dict(opacity=0.7, transparent=True), True),
    ]
    results: list[tuple[str, float, float]] = []

    class Bench(QOpenGLWidget):
        def __init__(self) -> None:
            super().__init__()
            self.renderer = GLRenderer()
            self.camera = Camera()
            self.lights = LightSetup()
            self.resize(w, h)
            self._cfg = -1
            self._scene = None
            self._times: list[float] = []
            self._warmup = 0
            self._t0 = 0.0
            self._info_printed = False

        def initializeGL(self) -> None:
            self.renderer.init_gl()
            self.renderer.resize(w, h)
            self._next_config()

        def _next_config(self) -> None:
            self._cfg += 1
            if self._cfg >= len(configs):
                self._report()
                app.quit()
                return
            label, matkw, clip = configs[self._cfg]
            self._scene = _build_scene(args.meshes, args.tris, mode, **matkw)
            if clip:
                self.renderer.set_clip_plane((1.0, 0.0, 0.0), 0.0)
            else:
                self.renderer.clear_clip_plane()
            self._times = []
            self._warmup = 30
            print(f"  running {label} ...", flush=True)

        def paintGL(self) -> None:
            if not self._info_printed:
                self._info_printed = True
                def s(x):
                    v = glGetString(x)
                    return v.decode() if isinstance(v, bytes) else str(v)
                print(f"GL_VENDOR   {s(GL_VENDOR)}")
                print(f"GL_RENDERER {s(GL_RENDERER)}")
                print(f"GL_VERSION  {s(GL_VERSION)}")
                tris = args.meshes * args.tris
                print(f"\n{args.meshes} structures x {args.tris} tris = "
                      f"{tris / 1e6:.1f}M triangles, {w}x{h}, mode={mode.name}, "
                      f"{args.frames} timed frames per configuration\n")

            t = time.perf_counter()
            self.renderer.render(self._scene, self.camera, self.lights)
            # glFinish would be needed for a true GPU-side time; without it this
            # measures submit-to-submit interval, which with vsync off and a
            # fully GPU-bound scene converges to the frame time.
            from OpenGL.GL import glFinish
            glFinish()
            dt = (time.perf_counter() - t) * 1000.0

            if self._warmup > 0:
                self._warmup -= 1
            else:
                self._times.append(dt)
                if len(self._times) >= args.frames:
                    label = configs[self._cfg][0]
                    med = statistics.median(self._times)
                    p95 = sorted(self._times)[int(0.95 * len(self._times)) - 1]
                    results.append((label, med, p95))
                    self._next_config()
            self.update()

        def _report(self) -> None:
            print(f"\n{'configuration':<22}{'median ms':>10}{'p95 ms':>9}"
                  f"{'fps':>8}")
            for label, med, p95 in results:
                print(f"{label:<22}{med:>10.2f}{p95:>9.2f}{1000.0 / med:>8.1f}")
            if len(results) >= 2:
                gain = results[1][1] / results[0][1]
                print(f"\nopaque vs blended (clip off): {gain:.2f}x "
                      "-- this is the early-Z effect the opacity default "
                      "controls")
            if len(results) >= 3:
                print(f"clip on vs off (opaque):     "
                      f"{results[2][1] / results[0][1]:.2f}x")

    widget = Bench()
    widget.show()
    QTimer.singleShot(0, widget.update)
    app.exec()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
