"""Headless GL-call / Python-time benchmark for the renderer hot path.

Not a test (no ``test_`` prefix, so pytest ignores it).  Run it directly to
measure the two quantities that decide whether the app hits 60 fps on a real
display: the number of GL entry points a frame issues, and the Python time
spent issuing them.

    python tests/rendering/bench_glrec.py            # 500 and 900 meshes
    python tests/rendering/bench_glrec.py 50 200 500 900

Because ``tests.support.glrec`` stands in for ``OpenGL.GL``, the numbers are
exact call counts, not estimates.  The wall time is Python-side marshalling
only -- there is no driver and no GPU behind it -- so treat ms/frame as the
*CPU* half of the frame budget.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np  # noqa: E402

from tests.support import glrec  # noqa: E402

REC = glrec.install()

from faceforge.core.material import Material, RenderMode  # noqa: E402
from faceforge.core.mesh import BufferGeometry, MeshInstance  # noqa: E402
from faceforge.core.scene_graph import Scene, SceneNode  # noqa: E402
from faceforge.rendering.camera import Camera  # noqa: E402
from faceforge.rendering.lights import LightSetup  # noqa: E402
from faceforge.rendering.renderer import GLRenderer  # noqa: E402

TRIS = 200


def _geometry(rng: np.random.Generator) -> BufferGeometry:
    n = TRIS * 3
    pos = rng.normal(0.0, 40.0, size=n * 3).astype(np.float32)
    nrm = np.tile([0.0, 0.0, 1.0], n).astype(np.float32)
    return BufferGeometry(positions=pos, normals=nrm, vertex_count=n)


def build_scene(n_meshes: int, mode: RenderMode = RenderMode.SOLID,
                *, opacity: float = 1.0, transparent: bool = False):
    rng = np.random.default_rng(0)
    scene = Scene()
    for i in range(n_meshes):
        mat = Material(render_mode=mode, opacity=opacity, transparent=transparent)
        mesh = MeshInstance(name=f"m{i}", geometry=_geometry(rng), material=mat)
        node = SceneNode(name=f"n{i}")
        node.mesh = mesh
        node.set_position(*(rng.normal(0.0, 60.0, 3).tolist()))
        scene.add(node)
    scene.update()
    return scene


def measure(n_meshes: int, mode: RenderMode = RenderMode.SOLID,
            frames: int = 60, **kw) -> dict:
    scene = build_scene(n_meshes, mode, **kw)
    cam, lights = Camera(), LightSetup()
    r = GLRenderer()
    r.init_gl()
    r.resize(1600, 1000)
    r.render(scene, cam, lights)          # warm-up: uploads VAOs/VBOs
    r.render(scene, cam, lights)

    REC.reset()
    r.render(scene, cam, lights)
    calls = REC.total
    groups = REC.group()

    t0 = time.perf_counter()
    for _ in range(frames):
        r.render(scene, cam, lights)
    ms = (time.perf_counter() - t0) * 1000.0 / frames

    r.destroy()
    return {
        "meshes": n_meshes, "calls": calls, "per_mesh": calls / n_meshes,
        "ms": ms, "fps": 1000.0 / ms if ms else float("inf"), "groups": groups,
    }


def main(argv: list[str]) -> None:
    sizes = [int(a) for a in argv[1:]] or [500, 900]
    print(f"{'meshes':>7} {'GL/frame':>9} {'GL/mesh':>8} {'ms/frame':>9} "
          f"{'cpu-fps':>8}   uniform/draw/bind/state")
    for n in sizes:
        m = measure(n)
        g = m["groups"]
        print(f"{m['meshes']:>7} {m['calls']:>9} {m['per_mesh']:>8.2f} "
              f"{m['ms']:>9.2f} {m['fps']:>8.1f}   "
              f"{g['uniform']}/{g['draw']}/{g['bind']}/{g['state']}")

    print("\nper render mode, 300 meshes:")
    print(f"{'mode':<14} {'GL/mesh':>8} {'ms/frame':>9}")
    for mode in RenderMode:
        m = measure(300, mode, frames=30)
        print(f"{mode.name:<14} {m['per_mesh']:>8.2f} {m['ms']:>9.2f}")


if __name__ == "__main__":
    main(sys.argv)
