"""True-resolution offscreen stills, and the evidence that they are one.

The problem
-----------
:meth:`faceforge.export.video_export.VideoExporter.export_screenshot` produces a
"high resolution" still by calling ``QOpenGLWidget.grabFramebuffer()`` and then
``QImage.scaled(w, h, SmoothTransformation)``.  That is interpolation, not
resolution.  A 700x700 viewport scaled to 4000x4000 is a 4000x4000 file holding
700x700 of information: the geometry was rasterised once, at the small size, and
every triangle edge, specular highlight and thin vessel is frozen at the coarse
sampling before the resize ever runs.  Nothing above the small image's Nyquist
frequency can appear, because the renderer was never asked for it.

The fix
-------
Render *at* the requested size through a framebuffer object.  The scene is
rasterised at 2048x2048, so a 2048x2048 file holds 2048x2048 of information:
edges are sampled where they actually are, and features narrower than a
viewport pixel exist.  :class:`faceforge.session.Session` already owns exactly
the right FBO (colour texture + depth renderbuffer, the same attachment format
as ``tools/capture_golden.py``); this module drives it, bounds-checks the
request against the driver's real limits first, and restores the session's
previous size if anything fails.

Proving it
----------
"Bigger file" is not "higher resolution", so :func:`resolution_evidence`
measures the difference instead of asserting it.  Given the same scene rendered
small and large, it bicubic-upscales the small one to the large one's size and
compares:

* **Spectral band energy above the small render's Nyquist.**  This is the
  decisive number.  An upscale cannot put energy above the source's Nyquist
  frequency -- there is none to put there; bicubic contributes only a little
  ringing.  A true render at 4x fills that band with real detail.
* **Pixel disagreement** between the true render and the upscale.
* **Gradient energy**, a coarser proxy for edge sharpness.
* **A downsample check**: area-averaging the large render back down should
  approximately reproduce the small one.  Without it, a large "difference"
  could just mean the two renders show different things.

:mod:`tests.export.test_still` runs this at 512 vs 2048 and asserts the
direction and magnitude of each number.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: Smallest still worth writing.  Below this the FBO works but the result is
#: not an image of anything.
MIN_STILL_SIZE = 8


class StillExportError(RuntimeError):
    """A still that cannot be rendered at the requested size."""


class StillSizeError(StillExportError, ValueError):
    """The requested size exceeds what the GL implementation can allocate.

    A subclass of :class:`ValueError` so that callers which already treat bad
    arguments as usage errors -- the CLI does -- classify it correctly without
    a second except clause.
    """


# ---------------------------------------------------------------------------
# Driver limits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GLSizeLimits:
    """What the current GL implementation will actually allocate.

    Three separate limits bind an FBO render, and the smallest wins:

    ``GL_MAX_TEXTURE_SIZE``
        the colour attachment is a 2D texture.
    ``GL_MAX_RENDERBUFFER_SIZE``
        the depth attachment is a renderbuffer.
    ``GL_MAX_VIEWPORT_DIMS``
        ``glViewport`` silently clamps beyond this, which is the dangerous one:
        the FBO allocates, the render succeeds, and the image is a correct
        render of a *smaller* viewport letterboxed into a larger buffer.  That
        is precisely the "silently truncated image" failure this module exists
        to prevent, so it is checked up front rather than discovered later.
    """

    max_texture_size: int
    max_renderbuffer_size: int
    max_viewport_width: int
    max_viewport_height: int

    @property
    def max_width(self) -> int:
        return min(self.max_texture_size, self.max_renderbuffer_size,
                   self.max_viewport_width)

    @property
    def max_height(self) -> int:
        return min(self.max_texture_size, self.max_renderbuffer_size,
                   self.max_viewport_height)

    @property
    def max_square(self) -> int:
        return min(self.max_width, self.max_height)

    def as_dict(self) -> dict[str, Any]:
        return {
            "GL_MAX_TEXTURE_SIZE": self.max_texture_size,
            "GL_MAX_RENDERBUFFER_SIZE": self.max_renderbuffer_size,
            "GL_MAX_VIEWPORT_DIMS": [self.max_viewport_width,
                                     self.max_viewport_height],
            "max_still": [self.max_width, self.max_height],
        }

    def check(self, width: int, height: int) -> None:
        """Raise :class:`StillSizeError` if ``width x height`` cannot be drawn.

        Named in the message: which limit was hit, what it is, and what the
        largest workable size would be.  A caller that gets this exception can
        act on it without querying GL itself.
        """
        width, height = int(width), int(height)
        if width < MIN_STILL_SIZE or height < MIN_STILL_SIZE:
            raise StillSizeError(
                f"{width}x{height} is below the {MIN_STILL_SIZE}px floor for a "
                "still; nothing useful can be rendered at that size"
            )
        failures = []
        if max(width, height) > self.max_texture_size:
            failures.append(
                f"GL_MAX_TEXTURE_SIZE={self.max_texture_size} (the colour "
                "attachment is a 2D texture)")
        if max(width, height) > self.max_renderbuffer_size:
            failures.append(
                f"GL_MAX_RENDERBUFFER_SIZE={self.max_renderbuffer_size} (the "
                "depth attachment is a renderbuffer)")
        if width > self.max_viewport_width or height > self.max_viewport_height:
            failures.append(
                f"GL_MAX_VIEWPORT_DIMS=[{self.max_viewport_width}, "
                f"{self.max_viewport_height}] -- glViewport would clamp and "
                "the image would be a smaller render inside a larger buffer")
        if failures:
            raise StillSizeError(
                f"cannot render a {width}x{height} still on this GL "
                f"implementation: exceeds " + "; ".join(failures) +
                f".  The largest still this context can render is "
                f"{self.max_width}x{self.max_height}."
            )


def query_size_limits() -> GLSizeLimits:
    """Read the limits from the *current* GL context.

    Requires a current context -- call it from inside a live
    :class:`faceforge.session.Session`, not before creating one.
    """
    from OpenGL.GL import (
        GL_MAX_RENDERBUFFER_SIZE, GL_MAX_TEXTURE_SIZE, GL_MAX_VIEWPORT_DIMS,
        glGetIntegerv,
    )

    max_tex = int(glGetIntegerv(GL_MAX_TEXTURE_SIZE))
    max_rb = int(glGetIntegerv(GL_MAX_RENDERBUFFER_SIZE))
    dims = np.asarray(glGetIntegerv(GL_MAX_VIEWPORT_DIMS)).ravel()
    if dims.size < 2:                                    # pragma: no cover
        raise StillExportError(
            "GL_MAX_VIEWPORT_DIMS returned fewer than two values; this GL "
            "context is not reporting its limits and a still cannot be "
            "size-checked against it"
        )
    return GLSizeLimits(
        max_texture_size=max_tex,
        max_renderbuffer_size=max_rb,
        max_viewport_width=int(dims[0]),
        max_viewport_height=int(dims[1]),
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StillResult:
    """One written still, plus what it was rendered through."""

    path: Path
    width: int
    height: int
    bytes_written: int
    limits: GLSizeLimits
    content_fraction: float | None
    upscaled_from: tuple[int, int] | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def megapixels(self) -> float:
        return self.width * self.height / 1e6

    def as_dict(self) -> dict[str, Any]:
        return {
            "out": str(self.path),
            "size": [self.width, self.height],
            "megapixels": round(self.megapixels, 3),
            "bytes": self.bytes_written,
            "rendered_at_full_resolution": self.upscaled_from is None,
            "content_fraction": self.content_fraction,
            "gl_limits": self.limits.as_dict(),
            "notes": list(self.notes),
        }


def render_still(
    session: Any,
    width: int,
    height: int,
    *,
    allow_blank: bool = False,
    limits: GLSizeLimits | None = None,
) -> np.ndarray:
    """Render one frame at exactly ``width x height`` through the session's FBO.

    The size is checked against the driver's limits *before* anything is
    allocated, so an over-large request raises :class:`StillSizeError` with the
    offending limit named instead of producing a clamped or truncated image.

    The session's previous size is restored if the render fails, because a
    session left holding a destroyed framebuffer would fail every later render
    with an unrelated error.
    """
    width, height = int(width), int(height)
    limits = limits or query_size_limits()
    limits.check(width, height)

    previous = session.size
    try:
        image = session.render(width, height, allow_blank=allow_blank)
    except Exception:
        if session.size != previous and not session.closed:
            try:
                session.resize(*previous)
            except Exception as exc:                      # noqa: BLE001
                logger.warning("could not restore the session to %dx%d: %s",
                               previous[0], previous[1], exc)
        raise

    got = (image.shape[1], image.shape[0])
    if got != (width, height):
        raise StillExportError(
            f"asked for a {width}x{height} still and got {got[0]}x{got[1]}.  "
            "Refusing to write it: an image whose size does not match the "
            "request is either truncated or clamped, and neither is a "
            "publication still."
        )
    return image


def frame_scene(
    session: Any,
    *,
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    up: tuple[float, float, float] = (0.0, 1.0, 0.0),
    distance: float = 2.6,
) -> dict[str, Any]:
    """Point the session's camera at its scene, from *direction*.

    Needed because a still rendered without a SceneState has no camera to
    inherit, and the default camera sits at the origin looking down -Z --
    which, for BodyParts3D geometry centred 1.5 m up the +Z axis, produces a
    frame that is almost entirely background.  A near-blank publication still
    that exits 0 is exactly the outcome this module exists to prevent, so the
    scene is framed rather than left to chance.

    The placement rule is the same one ``tools/capture_golden.py`` uses -- eye
    at ``centroid + direction * radius * distance``, target at the centroid --
    so a framed still and a golden capture of the same scene look at it the
    same way.  Returns the measured bounds for the manifest.
    """
    from faceforge.core.math_utils import vec3

    scene = session.scene
    scene.update()
    pairs = scene.collect_meshes()
    if not pairs:
        raise StillExportError(
            "the scene has no visible meshes, so there is nothing to frame"
        )

    points = []
    for mesh, world in pairs:
        positions = np.asarray(mesh.geometry.positions,
                               dtype=np.float64).reshape(-1, 3)
        if not len(positions):
            continue
        world = np.asarray(world, dtype=np.float64)
        points.append((world[:3, :3] @ positions.T).T + world[:3, 3])
    if not points:
        raise StillExportError("every visible mesh in the scene is empty")

    all_points = np.concatenate(points)
    lo, hi = all_points.min(axis=0), all_points.max(axis=0)
    centroid = (lo + hi) / 2.0
    radius = float(np.linalg.norm(all_points - centroid, axis=1).max())
    if radius <= 0.0:                                    # pragma: no cover
        radius = 1.0

    axis = np.asarray(direction, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    eye = centroid + axis * radius * float(distance)
    session.camera.look_at(vec3(*eye), vec3(*centroid), vec3(*up))
    return {
        "centroid": [float(v) for v in centroid],
        "radius": radius,
        "bounds_min": [float(v) for v in lo],
        "bounds_max": [float(v) for v in hi],
        "eye": [float(v) for v in eye],
    }


def export_still(
    session: Any,
    path: Path | str,
    width: int | None = None,
    height: int | None = None,
    *,
    allow_blank: bool = False,
) -> StillResult:
    """Render at ``width x height`` and write it as a PNG.  Returns a receipt.

    ``width``/``height`` default to the session's current size.  Unlike a
    window grab there is no coupling to any on-screen widget: the requested
    size *is* the rasterisation size.
    """
    from faceforge.session import write_png

    path = Path(path)
    current_w, current_h = session.size
    width = current_w if width is None else int(width)
    height = current_h if height is None else int(height)

    limits = query_size_limits()
    image = render_still(session, width, height,
                         allow_blank=allow_blank, limits=limits)
    write_png(path, image)

    return StillResult(
        path=path,
        width=width,
        height=height,
        bytes_written=path.stat().st_size,
        limits=limits,
        content_fraction=getattr(session, "last_content_fraction", None),
        upscaled_from=None,
        notes=(
            f"rasterised at {width}x{height} through an offscreen framebuffer; "
            "no window grab and no resampling step is involved.",
        ),
    )


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def luminance(image: np.ndarray) -> np.ndarray:
    """Rec. 601 luma of an ``(H, W, 3|4)`` uint8 image, as float64 in 0..1."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr.astype(np.float64) / 255.0
    rgb = arr[..., :3].astype(np.float64) / 255.0
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def bicubic_upscale(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Bicubic-resample *image* to ``width x height`` -- the thing being beaten.

    Pillow's ``BICUBIC`` is the same class of filter as Qt's
    ``SmoothTransformation``: a fixed reconstruction kernel over the small
    render's samples.  It is the strongest fair stand-in for the old
    grab-then-scale path, and it is what the true render is measured against.
    """
    from PIL import Image

    arr = np.asarray(image)
    mode = "RGBA" if arr.ndim == 3 and arr.shape[2] == 4 else "RGB"
    src = Image.fromarray(arr[..., :4] if mode == "RGBA" else arr[..., :3], mode)
    return np.asarray(src.resize((int(width), int(height)), Image.BICUBIC))


def _hann2d(shape: tuple[int, int]) -> np.ndarray:
    """Separable Hann window.

    Without it, the hard image border leaks broadband energy into every
    frequency bin and both images look full of high-frequency detail -- which
    would flatter the upscale and make the comparison meaningless.
    """
    h, w = shape
    return np.outer(np.hanning(h), np.hanning(w))


def band_energy_above(gray: np.ndarray, cutoff_cycles_per_px: float) -> float:
    """Fraction of spectral power above *cutoff*, DC excluded.

    ``cutoff_cycles_per_px`` is in cycles per pixel of *this* image's grid, so
    the Nyquist limit of a source image ``factor`` times smaller sits at
    ``0.5 / factor``.
    """
    win = _hann2d(gray.shape)
    spectrum = np.fft.fft2((gray - gray.mean()) * win)
    power = np.abs(spectrum) ** 2
    fy = np.fft.fftfreq(gray.shape[0])[:, None]
    fx = np.fft.fftfreq(gray.shape[1])[None, :]
    radius = np.sqrt(fy ** 2 + fx ** 2)
    total = float(power.sum())
    if total <= 0.0:
        return 0.0
    return float(power[radius > cutoff_cycles_per_px].sum() / total)


def gradient_energy(gray: np.ndarray) -> float:
    """Mean squared finite-difference gradient magnitude."""
    gy, gx = np.gradient(gray)
    return float(np.mean(gx ** 2 + gy ** 2))


def area_downsample(image: np.ndarray, factor: int) -> np.ndarray:
    """Box-average *image* down by an integer *factor*, per channel."""
    arr = np.asarray(image)
    h, w = arr.shape[:2]
    if h % factor or w % factor:
        raise ValueError(
            f"{h}x{w} is not divisible by {factor}; an area downsample would "
            "have to crop, and a cropped comparison is not a comparison"
        )
    if arr.ndim == 2:
        return arr.reshape(h // factor, factor, w // factor, factor).mean((1, 3))
    c = arr.shape[2]
    return arr.astype(np.float64).reshape(
        h // factor, factor, w // factor, factor, c).mean((1, 3))


def resolution_evidence(small: np.ndarray, large: np.ndarray) -> dict[str, Any]:
    """Measure whether *large* holds detail *small* cannot supply.

    Both images must be renders of the same scene at the same aspect ratio,
    ``large`` the same integer multiple of ``small`` on both axes.  A single
    factor is required (rather than one per axis) because the Nyquist cutoff
    the comparison turns on is a single number.  Returns measured numbers only
    -- no verdict; the thresholds live in the test that calls this.
    """
    small = np.asarray(small)
    large = np.asarray(large)
    factors = tuple(large.shape[a] / small.shape[a] for a in (0, 1))
    factor = int(round(factors[0]))
    if factor < 2 or any(abs(f - factor) > 1e-9 for f in factors):
        raise ValueError(
            f"large/small = {factors} (rows, cols) is not one integer factor "
            "of at least 2 on both axes"
        )

    upscaled = bicubic_upscale(small, large.shape[1], large.shape[0])
    true_gray = luminance(large)
    up_gray = luminance(upscaled)

    # The band the small render physically cannot describe.
    cutoff = 0.5 / factor
    true_band = band_energy_above(true_gray, cutoff)
    up_band = band_energy_above(up_gray, cutoff)

    diff = np.abs(true_gray - up_gray)
    down = area_downsample(large, factor)
    small_f = np.asarray(small).astype(np.float64)
    if down.ndim == 3 and small_f.ndim == 3:
        down = down[..., :small_f.shape[2]]
    downsample_mad = float(np.mean(np.abs(down - small_f)))

    return {
        "small_size": [int(small.shape[1]), int(small.shape[0])],
        "large_size": [int(large.shape[1]), int(large.shape[0])],
        "factor": factor,
        "nyquist_cutoff_cycles_per_px": cutoff,
        "band_energy_above_small_nyquist": {
            "true_render": true_band,
            "bicubic_upscale": up_band,
            "ratio": (true_band / up_band) if up_band > 0 else float("inf"),
        },
        "pixel_disagreement_vs_upscale": {
            "mean_abs_luma": float(diff.mean()),
            "rms_luma": float(np.sqrt((diff ** 2).mean())),
            "max_abs_luma": float(diff.max()),
            "fraction_of_pixels_differing_over_1_255": float(
                (diff > 1.0 / 255.0).mean()),
        },
        "gradient_energy": {
            "true_render": gradient_energy(true_gray),
            "bicubic_upscale": gradient_energy(up_gray),
        },
        "downsample_check": {
            "mean_abs_rgba_0_255": downsample_mad,
            "note": "area-averaging the large render by the factor should "
                    "approximately reproduce the small render; a small value "
                    "means the two renders show the same scene, so the "
                    "high-band difference is detail rather than a different "
                    "picture.  Valid only for modes whose primitives scale "
                    "with the image: WIREFRAME and POINTS set line and point "
                    "width in PIXELS, so a 1 px line covers four times less "
                    "of the subject at 4x and downsampling legitimately "
                    "disagrees (measured 12.5/255 on a wireframe skull "
                    "against 1.0/255 on solid geometry).  Read it as a "
                    "same-scene guard for solid modes only.",
        },
    }
