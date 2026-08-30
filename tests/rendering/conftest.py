"""Fixtures that let the real renderer run without a GL context.

There is no window server in CI (and Qt's ``offscreen``/``minimal`` platform
plugins refuse ``createPlatformOpenGLContext``), so the renderer could not be
tested at all.  ``tests.support.glrec`` installs a counting stand-in for
``OpenGL.GL`` into :data:`sys.modules`; these fixtures swap it in, re-import
``faceforge.rendering`` against it, and put the real modules back afterwards.

Because ``faceforge.rendering.*`` binds GL entry points at import time
(``from OpenGL.GL import glDrawArrays, ...``), the rendering package must be
evicted from ``sys.modules`` and re-imported *inside* the fixture -- otherwise
the already-bound real functions would be used.
"""

import sys

import numpy as np
import pytest

from tests.support import glrec


def _evict(prefixes: tuple[str, ...]) -> dict[str, object]:
    """Remove and return every loaded module whose name starts with *prefixes*."""
    saved = {n: m for n, m in sys.modules.items() if n.startswith(prefixes)}
    for name in saved:
        del sys.modules[name]
    return saved


@pytest.fixture
def gl_env():
    """Yield ``(recorder, faceforge.rendering module namespace)``.

    The recorder counts every GL entry point invoked while the fixture is
    active.  Real ``OpenGL`` and ``faceforge.rendering`` modules are restored
    on teardown so other test modules are unaffected.
    """
    saved_ff = _evict(("faceforge.rendering",))
    saved_gl = _evict(("OpenGL",))

    rec = glrec.install()
    try:
        import importlib

        mods = {
            name: importlib.import_module(f"faceforge.rendering.{name}")
            for name in ("renderer", "gl_mesh", "shader_program", "camera", "lights")
        }
        yield rec, mods
    finally:
        glrec.uninstall()
        _evict(("faceforge.rendering", "OpenGL"))
        sys.modules.update(saved_gl)
        sys.modules.update(saved_ff)


@pytest.fixture
def make_mesh():
    """Return a factory building a single-triangle :class:`MeshInstance`."""
    from faceforge.core.material import Material
    from faceforge.core.mesh import BufferGeometry, MeshInstance

    def _factory(name: str = "m", *, opacity: float = 1.0, transparent: bool = False):
        geom = BufferGeometry(
            positions=np.array(
                [0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32),
            normals=np.array(
                [0, 0, 1, 0, 0, 1, 0, 0, 1], dtype=np.float32),
            vertex_count=3,
        )
        mat = Material()
        mat.opacity = opacity
        mat.transparent = transparent
        return MeshInstance(name=name, geometry=geom, material=mat)

    return _factory


@pytest.fixture
def scene_with(make_mesh):
    """Return a factory building a :class:`Scene` holding *n* meshes."""
    from faceforge.core.scene_graph import Scene, SceneNode

    def _factory(n: int):
        scene = Scene()
        meshes = []
        for i in range(n):
            node = SceneNode(name=f"node{i}")
            node.mesh = make_mesh(f"mesh{i}")
            meshes.append(node.mesh)
            scene.add(node)
        scene.update()
        return scene, meshes

    return _factory
