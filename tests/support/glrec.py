"""Headless OpenGL recorder for FaceForge.

Installs a fake ``OpenGL.GL`` module into ``sys.modules`` so that the real
renderer can be driven without a GL context or a window server.  Every GL
entry point becomes a counting stub, which lets us measure the *exact* number
of driver calls a frame costs -- the quantity that dominates PyOpenGL
render loops -- plus the Python-side marshalling time.

Fidelity notes
--------------
* ``glGetUniformLocation`` returns -1 for uniforms that do not appear in the
  program's shader sources, exactly like a real driver.  Sources are tracked
  through ``glShaderSource``/``glAttachShader``, so per-mode uniform-upload
  counts match reality instead of being inflated.
* ``glGen*``/``glCreate*`` hand out unique non-zero integer names.
* Compile/link status queries report success.

Usage
-----
    import glrec
    rec = glrec.install()          # BEFORE importing faceforge.rendering
    ...
    rec.reset()
    renderer.render(scene, camera, lights)
    print(rec.total, rec.counts.most_common(10))
"""

from __future__ import annotations

import collections
import itertools
import re
import sys
import types

__all__ = ["GLRecorder", "install", "uninstall"]


class GLRecorder:
    """Counts GL entry-point invocations and (optionally) records arguments."""

    def __init__(self) -> None:
        self.counts: collections.Counter = collections.Counter()
        self.calls: list[tuple] = []
        self.record_args: bool = False
        # --- emulated driver state ---
        self._names = itertools.count(1)
        self._const_ids: dict[str, int] = {}
        self._const_counter = itertools.count(0x1000)
        self._shader_src: dict[int, str] = {}     # shader handle -> source
        self._program_src: dict[int, str] = {}    # program handle -> concat source
        self._uniform_loc: dict[tuple[int, str], int] = {}
        self._uniform_counter = itertools.count(0)

    # ------------------------------------------------------------------
    # Measurement API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear counters (keeps emulated driver state such as programs)."""
        self.counts.clear()
        self.calls.clear()

    @property
    def total(self) -> int:
        """Total number of GL calls recorded since the last reset."""
        return sum(self.counts.values())

    def summary(self, top: int = 15) -> str:
        lines = [f"total GL calls: {self.total}"]
        for name, n in self.counts.most_common(top):
            lines.append(f"  {n:>8d}  {name}")
        return "\n".join(lines)

    def group(self) -> dict[str, int]:
        """Aggregate call counts into coarse categories."""
        buckets = {
            "uniform": 0, "draw": 0, "bind": 0, "state": 0,
            "buffer": 0, "query": 0, "other": 0,
        }
        for name, n in self.counts.items():
            if name.startswith("glUniform"):
                buckets["uniform"] += n
            elif name.startswith("glDraw"):
                buckets["draw"] += n
            elif name.startswith("glBind") or name == "glUseProgram":
                buckets["bind"] += n
            elif name.startswith("glBuffer"):
                buckets["buffer"] += n
            elif name.startswith(("glGet", "glIs")):
                buckets["query"] += n
            elif name.startswith((
                "glEnable", "glDisable", "glDepth", "glCull", "glPolygon",
                "glBlend", "glClear", "glViewport", "glPointSize",
                "glLineWidth", "glColorMask",
            )):
                buckets["state"] += n
            else:
                buckets["other"] += n
        return buckets

    # ------------------------------------------------------------------
    # Fake-driver internals
    # ------------------------------------------------------------------

    def _constant(self, name: str) -> int:
        cid = self._const_ids.get(name)
        if cid is None:
            cid = next(self._const_counter)
            self._const_ids[name] = cid
        return cid

    def _dispatch(self, name: str, args: tuple):
        self.counts[name] += 1
        if self.record_args:
            self.calls.append((name, args))

        # --- object creation ---------------------------------------------
        if name in ("glGenBuffers", "glGenVertexArrays", "glGenTextures",
                    "glGenFramebuffers", "glGenRenderbuffers",
                    "glGenQueries"):
            n = args[0] if args else 1
            if n == 1:
                return next(self._names)
            return [next(self._names) for _ in range(n)]

        if name == "glCreateShader":
            h = next(self._names)
            self._shader_src[h] = ""
            return h

        if name == "glCreateProgram":
            h = next(self._names)
            self._program_src[h] = ""
            return h

        # --- shader source tracking (for realistic uniform locations) ----
        if name == "glShaderSource":
            handle, src = args[0], args[1]
            if isinstance(src, (list, tuple)):
                src = "\n".join(str(s) for s in src)
            self._shader_src[handle] = str(src)
            return None

        if name == "glAttachShader":
            prog, shader = args[0], args[1]
            self._program_src[prog] = (
                self._program_src.get(prog, "") + "\n"
                + self._shader_src.get(shader, "")
            )
            return None

        # --- status queries ----------------------------------------------
        if name in ("glGetShaderiv", "glGetProgramiv"):
            return 1
        if name in ("glGetShaderInfoLog", "glGetProgramInfoLog"):
            return b""
        if name == "glGetString":
            return b"OpenGL 3.3 (faceforge GLRecorder)"
        if name == "glGetError":
            return 0
        if name == "glGetIntegerv":
            return 0

        # --- uniform / attribute locations -------------------------------
        if name == "glGetUniformLocation":
            prog, uname = args[0], args[1]
            if isinstance(uname, bytes):
                uname = uname.decode()
            key = (prog, uname)
            loc = self._uniform_loc.get(key)
            if loc is None:
                src = self._program_src.get(prog, "")
                # Match a real driver: absent (or optimised-out) uniforms -> -1
                declared = re.search(
                    r"\buniform\b[^;]*\b" + re.escape(uname) + r"\b", src
                )
                used = src.count(uname) > 1 if declared else False
                loc = next(self._uniform_counter) if (declared and used) else -1
                self._uniform_loc[key] = loc
            return loc

        if name == "glGetAttribLocation":
            return 0

        return None

    # ------------------------------------------------------------------

    def make_module(self, modname: str) -> types.ModuleType:
        """Build a fake GL module whose attributes are constants or stubs."""
        mod = types.ModuleType(modname)
        rec = self

        def _getattr(attr: str):
            if attr.startswith("__"):
                raise AttributeError(attr)
            if attr.startswith("GL_") or attr.isupper():
                return rec._constant(attr)

            def _stub(*args, **kwargs):
                return rec._dispatch(attr, args)

            _stub.__name__ = attr
            setattr(mod, attr, _stub)   # cache so repeat lookups are cheap
            return _stub

        mod.__getattr__ = _getattr
        return mod


_INSTALLED: dict[str, object] = {}


def install(record_args: bool = False) -> GLRecorder:
    """Replace the ``OpenGL`` package in ``sys.modules`` with a recorder.

    Must be called *before* any module that does ``from OpenGL.GL import ...``.
    Returns the :class:`GLRecorder` collecting the calls.
    """
    rec = GLRecorder()
    rec.record_args = record_args

    root = types.ModuleType("OpenGL")
    root.ERROR_CHECKING = False
    root.ARRAY_SIZE_CHECKING = False
    root.STORE_POINTERS = False
    root.__path__ = []          # mark as a package
    root.__version__ = "recorder"

    gl = rec.make_module("OpenGL.GL")
    root.GL = gl

    submods = {"OpenGL": root, "OpenGL.GL": gl}
    for extra in ("OpenGL.GLU", "OpenGL.GLUT", "OpenGL.arrays",
                  "OpenGL.GL.shaders", "OpenGL.error"):
        m = rec.make_module(extra)
        submods[extra] = m

    for k, v in submods.items():
        _INSTALLED[k] = sys.modules.get(k)
        sys.modules[k] = v

    return rec


def uninstall() -> None:
    """Restore whatever was in ``sys.modules`` before :func:`install`."""
    for k, old in _INSTALLED.items():
        if old is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = old
    _INSTALLED.clear()
