"""Acquire a headless OpenGL context on macOS, or fail loudly.

Why this module exists
---------------------
Every previous attempt to render FaceForge without a window failed, and the
2026-08 audit recorded the failures as a hard environment constraint:

    Qt offscreen/minimal plugins   refuse createPlatformOpenGLContext
    QT_QPA_PLATFORM=cocoa          hangs indefinitely (a probe burned 600 s)
    glfw window creation           hangs for the same reason
    CGLChoosePixelFormat           err=10017 kCGLBadConnection
    CGMainDisplayID()              returns 0, zero active displays
    MTLCreateSystemDefaultDevice() returns NULL
    Mesa / llvmpipe / osmesa       no macOS build exists

All of that is still true, and it is still true that the sandbox is denied a
CoreGraphics window-server connection.  What was *not* tried is asking CGL for
a specific renderer instead of letting it choose.  The default
``CGLChoosePixelFormat`` search consults the window server to enumerate
displays and their attached GPUs, which is the step that returns
kCGLBadConnection.  Naming Apple's software rasteriser explicitly --

    kCGLPFARendererID (70) = kCGLRendererGenericFloatID (0x00020400)

-- skips display enumeration entirely, because a software renderer is not
attached to a display.  Measured in the sandbox on 2026-08-29:

    kCGLPFARendererID=GenericFloat, GL3_Core profile
      -> GL_VERSION  = 4.1 APPLE-23.1.1
         GL_RENDERER = Apple Software Renderer
         GL_MAX_CLIP_DISTANCES = 8
         FBO status  = GL_FRAMEBUFFER_COMPLETE
         glReadPixels returns correct pixels

That is a real GLSL compiler, a real rasteriser and a real framebuffer, which
is everything golden-image capture needs.  It is CPU-only and roughly two
orders of magnitude slower than the M1 Max, so it is a validation tool, not a
benchmark: never quote a frame time measured through it as renderer
performance.

Acquisition order
-----------------
1. A context that is already current (the caller is inside a Qt/GLFW GUI).
2. Hardware-accelerated CGL -- works in a logged-in GUI session, where it
   selects the Metal-backed driver.
3. Software CGL -- the sandbox path above.

The chosen path is reported in :class:`GLContextInfo.kind` and lands in the
capture manifest, so a comparison can refuse to diff a software capture
against a hardware one.

There is no fourth path and no silent success: :func:`acquire_offscreen_gl`
raises :class:`GLContextError` listing every attempt and its error code.  That
is deliberate.  ``tools/capture_gui_screenshots.py`` destroyed 11 tracked
README images by writing blank frames after its context failed while still
exiting 0; a loud exception is the fix.
"""

from __future__ import annotations

import ctypes
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_OPENGL_FRAMEWORK = "/System/Library/Frameworks/OpenGL.framework/OpenGL"

# CGL pixel-format attributes (CGLTypes.h).  Named here so the attribute lists
# below read as intent rather than as integers.
kCGLPFAAllRenderers = 1
kCGLPFADoubleBuffer = 5
kCGLPFAColorSize = 8
kCGLPFADepthSize = 12
kCGLPFARendererID = 70
kCGLPFAAccelerated = 73
kCGLPFAAllowOfflineRenderers = 96
kCGLPFAOpenGLProfile = 99

kCGLOGLPVersion_Legacy = 0x1000
kCGLOGLPVersion_GL3_Core = 0x3200
kCGLOGLPVersion_GL4_Core = 0x4100

# Apple's software rasteriser.  Not attached to a display, hence reachable
# without a window-server connection.
kCGLRendererGenericFloatID = 0x00020400

GL_VERSION = 0x1F02
GL_RENDERER = 0x1F01
GL_VENDOR = 0x1F00
GL_SHADING_LANGUAGE_VERSION = 0x8B8C


class GLContextError(RuntimeError):
    """No usable OpenGL context could be created.

    Raised instead of returning a sentinel so a caller cannot mistake failure
    for success and go on to write blank images.
    """


@dataclass
class GLContextInfo:
    """What was acquired, for the manifest and for the log banner."""

    kind: str                     # "existing" | "cgl-hardware" | "cgl-software"
    gl_version: str
    gl_renderer: str
    gl_vendor: str
    glsl_version: str
    max_clip_distances: int
    attempts: list[str] = field(default_factory=list)

    @property
    def is_software(self) -> bool:
        return "software" in self.gl_renderer.lower() or self.kind == "cgl-software"

    def banner(self) -> str:
        speed = "  [CPU rasteriser: correctness only, NOT a benchmark]" if self.is_software else ""
        return (
            f"GL context: {self.kind}\n"
            f"  GL_VERSION   {self.gl_version}\n"
            f"  GL_RENDERER  {self.gl_renderer}{speed}\n"
            f"  GL_VENDOR    {self.gl_vendor}\n"
            f"  GLSL         {self.glsl_version}\n"
            f"  MAX_CLIP_DISTANCES {self.max_clip_distances}"
        )

    def as_manifest_dict(self) -> dict:
        return {
            "kind": self.kind,
            "gl_version": self.gl_version,
            "gl_renderer": self.gl_renderer,
            "gl_vendor": self.gl_vendor,
            "glsl_version": self.glsl_version,
            "max_clip_distances": self.max_clip_distances,
            "is_software": self.is_software,
        }


# Keeping the CGL pixel format and context alive for the process lifetime.
# ctypes does not retain them, and a garbage-collected CGLContextObj takes the
# rasteriser down with it mid-render.
_held: list = []


def _cgl() -> ctypes.CDLL:
    try:
        return ctypes.CDLL(_OPENGL_FRAMEWORK)
    except OSError as exc:  # pragma: no cover - macOS always has this
        raise GLContextError(f"cannot load {_OPENGL_FRAMEWORK}: {exc}") from exc


def _gl_string(cgl: ctypes.CDLL, enum: int) -> str:
    cgl.glGetString.restype = ctypes.c_char_p
    raw = cgl.glGetString(ctypes.c_uint(enum))
    return raw.decode("utf-8", "replace") if raw else "<null>"


def _try_cgl(attrs: list[int], label: str, attempts: list[str]) -> bool:
    """Try one CGL attribute list.  True if a context is now current."""
    cgl = _cgl()
    arr = (ctypes.c_int * (len(attrs) + 1))(*(list(attrs) + [0]))
    pf = ctypes.c_void_p()
    npix = ctypes.c_int()
    err = cgl.CGLChoosePixelFormat(arr, ctypes.byref(pf), ctypes.byref(npix))
    if err != 0 or not pf.value:
        attempts.append(f"{label}: CGLChoosePixelFormat err={err} (npix={npix.value})")
        return False
    ctx = ctypes.c_void_p()
    err = cgl.CGLCreateContext(pf, None, ctypes.byref(ctx))
    if err != 0 or not ctx.value:
        attempts.append(f"{label}: CGLCreateContext err={err}")
        return False
    err = cgl.CGLSetCurrentContext(ctx)
    if err != 0:
        attempts.append(f"{label}: CGLSetCurrentContext err={err}")
        return False
    _held.append((pf, ctx))
    attempts.append(f"{label}: OK")
    return True


def _context_is_current() -> bool:
    """True if some other layer (Qt, GLFW) already made a context current."""
    cgl = _cgl()
    cgl.CGLGetCurrentContext.restype = ctypes.c_void_p
    try:
        return bool(cgl.CGLGetCurrentContext())
    except Exception:  # noqa: BLE001 - probing, absence is a valid answer
        return False


def _describe(kind: str, attempts: list[str]) -> GLContextInfo:
    cgl = _cgl()
    max_clip = ctypes.c_int(0)
    try:
        # GL_MAX_CLIP_DISTANCES = 0x0D32
        cgl.glGetIntegerv(ctypes.c_uint(0x0D32), ctypes.byref(max_clip))
    except Exception:  # noqa: BLE001 - informational only
        pass
    return GLContextInfo(
        kind=kind,
        gl_version=_gl_string(cgl, GL_VERSION),
        gl_renderer=_gl_string(cgl, GL_RENDERER),
        gl_vendor=_gl_string(cgl, GL_VENDOR),
        glsl_version=_gl_string(cgl, GL_SHADING_LANGUAGE_VERSION),
        max_clip_distances=int(max_clip.value),
        attempts=attempts,
    )


def acquire_offscreen_gl(prefer: str = "auto") -> GLContextInfo:
    """Make an OpenGL context current, or raise :class:`GLContextError`.

    Args:
        prefer: ``"auto"`` tries existing -> hardware -> software.
            ``"hardware"`` refuses to fall back to the software rasteriser,
            which is what a performance measurement wants.  ``"software"``
            forces the CPU path, so a capture can be made reproducible across
            machines with different GPUs.

    Never returns without a current context.  Callers must not guard this with
    a bare ``except`` and continue: the entire point is that a failed context
    stops the run before any image is written.
    """
    if prefer not in ("auto", "hardware", "software"):
        raise ValueError(f"prefer must be auto|hardware|software, got {prefer!r}")

    attempts: list[str] = []

    if prefer == "auto" and _context_is_current():
        attempts.append("existing current context: OK")
        return _describe("existing", attempts)

    profile = [kCGLPFAOpenGLProfile, kCGLOGLPVersion_GL3_Core]
    buffers = [kCGLPFAColorSize, 24, kCGLPFADepthSize, 24]

    if prefer in ("auto", "hardware"):
        hw = [kCGLPFAAccelerated, *profile, *buffers]
        if _try_cgl(hw, "cgl-hardware(accelerated,GL3_Core)", attempts):
            return _describe("cgl-hardware", attempts)
        hw_off = [kCGLPFAAccelerated, kCGLPFAAllowOfflineRenderers, *profile, *buffers]
        if _try_cgl(hw_off, "cgl-hardware(accelerated,offline,GL3_Core)", attempts):
            return _describe("cgl-hardware", attempts)

    if prefer in ("auto", "software"):
        sw = [kCGLPFARendererID, kCGLRendererGenericFloatID, *profile, *buffers]
        if _try_cgl(sw, "cgl-software(GenericFloat,GL3_Core)", attempts):
            return _describe("cgl-software", attempts)
        sw_off = [
            kCGLPFARendererID, kCGLRendererGenericFloatID,
            kCGLPFAAllowOfflineRenderers, *profile, *buffers,
        ]
        if _try_cgl(sw_off, "cgl-software(GenericFloat,offline,GL3_Core)", attempts):
            return _describe("cgl-software", attempts)

    detail = "\n".join(f"    {a}" for a in attempts)
    raise GLContextError(
        f"no OpenGL context could be created (prefer={prefer}).\n"
        f"  attempts:\n{detail}\n"
        "  err=10017 is kCGLBadConnection: no window-server connection.\n"
        "  If every CGL attempt failed, this process cannot render.  Run the\n"
        "  capture from a logged-in GUI session, or via tools/render_agent.py."
    )


def gl_context_available(prefer: str = "auto") -> bool:
    """Non-raising probe, for ``pytest.mark.skipif``.

    Deliberately not used by the capture path: a capture must crash rather
    than skip, or it silently produces nothing while exiting 0.
    """
    if os.environ.get("FACEFORGE_NO_GL"):
        return False
    try:
        acquire_offscreen_gl(prefer)
    except GLContextError:
        return False
    except Exception:  # noqa: BLE001 - probe must not propagate anything
        return False
    return True


if __name__ == "__main__":  # pragma: no cover - manual diagnostic
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        info = acquire_offscreen_gl("auto")
    except GLContextError as exc:
        print(f"FAILED\n{exc}")
        raise SystemExit(1) from exc
    print(info.banner())
    for a in info.attempts:
        print(f"  attempt: {a}")
