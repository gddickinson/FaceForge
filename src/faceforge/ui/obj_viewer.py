"""A standalone window that displays one OBJ file.

Its own process, its own GL context, nothing of the application in it: the
point of "save the pose and look at it" is to see what a *file* contains, and
a viewer sharing the app's scene graph would show the app's idea of the pose
rather than the exported one.  Run it directly --

    python -m faceforge.ui.obj_viewer results/exercise_obj/back_squat.obj

-- or let the exercise tab's "Save OBJ and view" button launch it.

Orbit with the left mouse button, pan with the middle, zoom with the wheel;
the camera is framed on the file's own bounds, so a barbell and a fingertip
both arrive in shot.

The file is turned Y-up to Z-up on the way in.  An exercise is exported from
the gym scene, whose world is Y-up (the scene wrapper stands the body on its
feet), while `OrbitControls` measures its polar angle from +Z -- the body
frame's own convention.  Loading a Y-up file into Z-up controls leaves the
model lying on its side and orbiting about its length.
"""

from __future__ import annotations

import logging
import math
import re
import sys
from pathlib import Path

import numpy as np

from faceforge.core.material import Material
from faceforge.core.mesh import BufferGeometry, MeshInstance
from faceforge.core.scene_graph import Scene, SceneNode
from faceforge.loaders.obj_parser import parse_obj
from faceforge.rendering.gl_widget import GLViewport

logger = logging.getLogger(__name__)

#: The figure is grey; the point of the viewer is shape, not colour.
MESH_COLOUR = 0xC8C8C8
#: How much of the frame the model fills.
FRAME_MARGIN = 1.25
#: Y-up (the gym scene) to Z-up (the body frame, and what OrbitControls uses).
_Y_UP_TO_Z_UP = np.array([[1.0, 0.0, 0.0],
                          [0.0, 0.0, -1.0],
                          [0.0, 1.0, 0.0]], dtype=np.float64)


#: Above this many bytes the pure-Python parser is not worth waiting for.
#: Measured on a full exercise export (556 MB, 3.46 M vertices):
#: `loaders.obj_parser.parse_obj` extrapolates to ~105 s, the vectorised
#: reader below does it in 8.
FAST_READ_BYTES = 8_000_000


def read_obj_fast(path: Path) -> BufferGeometry:
    """A vectorised reader for the big files this viewer is given.

    `loaders.obj_parser.parse_obj` walks the file a line at a time, which is
    right for the app's own assets and hopeless for a whole posed scene: a
    back squat with the skeleton on exports 3.46 M vertices and 6.93 M
    triangles.  This does the same job with three regex passes and numpy.
    Anything it does not recognise falls back to the real parser.
    """
    raw = path.read_bytes()
    # Each pass is freed before the next starts: the three lists of matches
    # together are several times the file, and the file can be half a gigabyte.
    v = re.findall(rb"(?m)^v (.*)$", raw)
    if not v:
        raise ValueError("no vertices")
    positions = np.array(b" ".join(v).split(), dtype=np.float32)
    del v
    f = re.findall(rb"(?m)^f (.*)$", raw)
    if not f:
        raise ValueError("no faces")
    if positions.size % 3:
        raise ValueError("a vertex line does not have three coordinates")
    positions = positions.reshape(-1, 3)

    face_text = b" ".join(f)
    del f
    sample = face_text.split(b" ", 1)[0]
    if b"//" in sample:                       # a//a  -- what this app writes
        idx = np.array(face_text.replace(b"//", b" ").split(), dtype=np.int64)[::2]
    elif b"/" in sample:                      # a/b or a/b/c
        stride = sample.count(b"/") + 1
        idx = np.array(face_text.replace(b"/", b" ").split(), dtype=np.int64)[::stride]
    else:                                     # bare indices
        idx = np.array(face_text.split(), dtype=np.int64)
    if idx.size % 3:
        raise ValueError("only triangles are supported by the fast reader")
    indices = (idx - 1).astype(np.uint32)     # OBJ is 1-based
    del face_text, idx

    vn = re.findall(rb"(?m)^vn (.*)$", raw)
    del raw
    if vn:
        normals = np.array(b" ".join(vn).split(), dtype=np.float32).reshape(-1, 3)
        del vn
    else:
        normals = np.zeros_like(positions)
    if len(normals) != len(positions):        # per-face normals: let the real
        raise ValueError("normal count does not match vertex count")
    return BufferGeometry(positions=positions.ravel(), normals=normals.ravel(),
                          indices=indices)


def load_geometry(path: Path) -> BufferGeometry:
    """The fast reader for a big file, the real parser for anything else."""
    if path.stat().st_size >= FAST_READ_BYTES:
        try:
            return read_obj_fast(path)
        except Exception as exc:
            logger.warning("fast OBJ read failed (%s); falling back to the parser", exc)
    return parse_obj(path.read_text())


def _turn_z_up(geometry) -> None:
    """Rotate positions and normals from the scene's Y-up into Z-up, in place."""
    for attr in ("positions", "normals"):
        data = getattr(geometry, attr, None)
        if data is None:
            continue
        v = np.asarray(data, dtype=np.float32).reshape(-1, 3)
        setattr(geometry, attr, (v @ _Y_UP_TO_Z_UP.T).astype(np.float32).ravel())


def scene_from_obj(path: Path) -> tuple[Scene, np.ndarray, np.ndarray]:
    """Load *path* into a one-node scene, and return it with its bounds."""
    geometry = load_geometry(path)
    if getattr(geometry, "vertex_count", 0) == 0:
        raise ValueError(f"{path.name} has no geometry in it")
    _turn_z_up(geometry)
    material = Material.from_hex(MESH_COLOUR)
    node = SceneNode(path.stem)
    node.mesh = MeshInstance(name=path.stem, geometry=geometry, material=material)
    scene = Scene()
    scene.add(node)
    scene.update()
    v = np.asarray(geometry.positions, dtype=np.float64).reshape(-1, 3)
    return scene, v.min(axis=0), v.max(axis=0)


def frame_camera(viewport: GLViewport, lo: np.ndarray, hi: np.ndarray) -> None:
    """Point the camera at the model and back off far enough to see all of it."""
    centre = (lo + hi) / 2.0
    radius = max(float(np.linalg.norm(hi - lo)) / 2.0, 1.0)
    fov = math.radians(viewport.camera.fov)
    distance = radius * FRAME_MARGIN / max(math.tan(fov / 2.0), 1e-3)
    viewport.camera.set_target(*centre)
    # Three-quarter, and above the middle: straight down an axis hides the
    # limb that matters on half the exercises.
    eye = centre + np.array([distance * 0.62, distance * 0.62, distance * 0.35])
    viewport.camera.set_position(*(float(c) for c in eye))
    viewport.camera.far = max(viewport.camera.far, distance * 6.0)
    # OrbitControls derives its spherical coordinates in its constructor, so a
    # camera moved afterwards leaves it with a stale radius and the first drag
    # snaps the model away.  Rebuild it on the camera we have just placed.
    from faceforge.rendering.orbit_controls import OrbitControls

    viewport.orbit_controls = OrbitControls(viewport.camera)
    viewport.orbit_controls.max_radius = max(viewport.orbit_controls.max_radius,
                                             distance * 4.0)


def build_window(path: Path):
    """The viewer window for *path*, ready to show."""
    from PySide6.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget

    scene, lo, hi = scene_from_obj(path)
    viewport = GLViewport()
    viewport.scene = scene
    frame_camera(viewport, lo, hi)

    window = QMainWindow()
    window.setWindowTitle(f"FaceForge — {path.name}")
    central = QWidget()
    layout = QVBoxLayout(central)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    layout.addWidget(viewport, 1)
    size = hi - lo
    caption = QLabel(
        f"{path}    {scene.collect_meshes()[0][0].geometry.vertex_count:,} vertices    "
        f"bounds {size[0]:.0f} x {size[1]:.0f} x {size[2]:.0f} units")
    caption.setStyleSheet("padding: 4px 8px; color: #b8bcc4; background: #1b1e24;")
    layout.addWidget(caption)
    window.setCentralWidget(central)
    window.resize(1000, 800)
    return window


def launch(path: Path):
    """Open this viewer on *path* in a separate process, and return it.

    A separate process, not another window: the viewer builds its own GL
    context and its own scene, and a crash or a slow load in it must not take
    the application down with it.  ``PYTHONPATH`` carries the running
    interpreter's own import path so this works from a source checkout as well
    as an installed package.
    """
    import os
    import subprocess

    env = dict(os.environ)
    roots = [p for p in sys.path if p and Path(p).is_dir()]
    env["PYTHONPATH"] = os.pathsep.join(roots + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    return subprocess.Popen([sys.executable, "-m", "faceforge.ui.obj_viewer", str(path)],
                            env=env, start_new_session=True)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__.strip().splitlines()[0])
        print("usage: python -m faceforge.ui.obj_viewer <file.obj>")
        return 2
    path = Path(argv[0]).expanduser()
    if not path.is_file():
        print(f"no such file: {path}")
        return 2

    from PySide6.QtGui import QSurfaceFormat
    from PySide6.QtWidgets import QApplication

    from faceforge.rendering.gl_widget import create_gl_format

    # Before QApplication, and it has to be: Qt reads the default format when
    # it creates the first OpenGL surface, and one set afterwards is ignored on
    # macOS -- silently giving a legacy context.  (This is not what made the
    # viewer open blank; that was the GL error Qt leaves queued behind it, and
    # the fix for it is `gl_widget.drain_gl_errors`.  The context here was
    # measured as 4.1 core either way.  Setting the format is still right.)
    if QApplication.instance() is None:
        QSurfaceFormat.setDefaultFormat(create_gl_format())
    app = QApplication.instance() or QApplication([])
    try:
        window = build_window(path)
    except Exception as exc:                       # a bad file is not a crash
        logger.error("cannot display %s: %s", path, exc)
        print(f"cannot display {path}: {exc}")
        return 1
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
