"""Is the "high resolution" still actually high resolution?

The old path -- ``QOpenGLWidget.grabFramebuffer()`` then ``QImage.scaled(...,
SmoothTransformation)`` -- produces a large *file* holding a small image's worth
of information.  These tests establish that the FBO path does not, by measuring
the difference rather than asserting it.

The decisive test is
:func:`test_the_large_render_holds_detail_an_upscale_cannot_supply`.  It renders
the same scene at 512 and at 2048, bicubic-upscales the 512 to 2048 (the fair
stand-in for the old path) and compares the spectral energy above the 512
render's Nyquist frequency.  An upscale cannot put information there; a render
at four times the sampling rate can.  The asserted thresholds are deliberately
well inside the measured margin, so the test fails on a regression to
interpolation rather than on rasteriser noise.

Marked ``slow``: needs a GL context, and a 2048x2048 software render is a few
seconds.  Every timing in this module is a CPU rasteriser timing and is
therefore *not* a renderer benchmark -- nothing here reports one.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.slow

from faceforge.export import still as st                   # noqa: E402
from tests.export.conftest import box_mesh                 # noqa: E402

#: The reference and the still.  A factor of 4 makes the Nyquist argument easy
#: to state: nothing above 0.125 cycles/px of the 2048 grid can come from a 512
#: render.
SMALL, LARGE = 512, 2048


@pytest.fixture(scope="module")
def detail_scene_factory():
    """A grid of small boxes: lots of genuine high-frequency structure.

    A single large box would make the resolution argument almost entirely about
    its silhouette.  A 10x10 grid of 7 mm boxes on a 12 mm pitch puts thin gaps
    and many edges across the frame, which is what a real anatomical scene has
    and what an upscale demonstrably cannot reconstruct.
    """
    def build():
        from faceforge.core.scene_graph import Scene, SceneNode

        scene = Scene()
        for ix in range(10):
            for iy in range(10):
                mesh = box_mesh(
                    f"Mandible_{ix}_{iy}", source_id="FMA52748",
                    ontology_id="FMA:52748", preferred_label="Mandible",
                    centre=((ix - 5) * 12.0, (iy - 5) * 12.0, 0.0),
                    size=(7.0, 7.0, 7.0),
                )
                node = SceneNode(name=f"b{ix}_{iy}")
                node.mesh = mesh
                scene.add(node)
        scene.update()
        return scene
    return build


@pytest.fixture
def gl_session(detail_scene_factory):
    """One live session with the detail scene framed, closed on teardown."""
    from faceforge import session as fs
    from faceforge.core.math_utils import vec3

    try:
        session = fs.Session.create(width=256, height=256, prefer="auto")
    except fs.SessionError as exc:
        pytest.skip(f"no usable GL context: {exc}")
    try:
        session.adopt_scene(detail_scene_factory())
        session.camera.set_aspect(1, 1)
        session.camera.look_at(vec3(0.0, 0.0, 168.0), vec3(0.0, 0.0, 0.0),
                               vec3(0.0, 1.0, 0.0))
        yield session
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Driver limits, reported rather than assumed
# ---------------------------------------------------------------------------


def test_the_gl_size_limits_are_readable_and_consistent(gl_session):
    limits = gl_session.gl_size_limits()
    assert limits.max_texture_size >= 1024
    assert limits.max_renderbuffer_size >= 1024
    assert limits.max_viewport_width >= 1024
    assert limits.max_square == min(
        limits.max_texture_size, limits.max_renderbuffer_size,
        limits.max_viewport_width, limits.max_viewport_height,
    )
    # All three go in the manifest; a caller must be able to see which binds.
    assert set(limits.as_dict()) == {
        "GL_MAX_TEXTURE_SIZE", "GL_MAX_RENDERBUFFER_SIZE",
        "GL_MAX_VIEWPORT_DIMS", "max_still",
    }


def test_a_size_past_the_limit_fails_and_names_the_limit(gl_session):
    """It must fail clearly, never silently produce a truncated image."""
    limits = gl_session.gl_size_limits()
    over = limits.max_square + 1
    with pytest.raises(st.StillSizeError) as excinfo:
        st.render_still(gl_session, over, over)

    message = str(excinfo.value)
    assert f"{over}x{over}" in message
    assert "GL_MAX_TEXTURE_SIZE" in message
    assert str(limits.max_square) in message, (
        "the message must say what size would work"
    )


def test_the_session_still_works_after_an_oversize_request(gl_session):
    """A refused still must not leave the session holding a dead framebuffer."""
    limits = gl_session.gl_size_limits()
    before = gl_session.size
    with pytest.raises(st.StillSizeError):
        st.render_still(gl_session, limits.max_square + 1, limits.max_square + 1)
    assert gl_session.size == before

    image = st.render_still(gl_session, 128, 128)
    assert image.shape == (128, 128, 4)


def test_a_size_below_the_floor_is_refused(gl_session):
    with pytest.raises(st.StillSizeError, match="below the .* floor"):
        st.render_still(gl_session, 4, 4)


# ---------------------------------------------------------------------------
# The still is rasterised at the requested size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [(256, 256), (640, 480), (1024, 512)])
def test_the_image_comes_back_at_exactly_the_requested_size(gl_session, size):
    width, height = size
    image = st.render_still(gl_session, width, height)
    assert image.shape == (height, width, 4)


def test_export_still_writes_a_png_of_the_requested_size(gl_session, tmp_path):
    out = tmp_path / "figure.png"
    result = gl_session.export_still(out, 1024, 768)

    assert result.width == 1024 and result.height == 768
    assert result.upscaled_from is None
    assert result.bytes_written == out.stat().st_size > 0
    assert result.limits.max_square >= 1024

    from PIL import Image

    with Image.open(out) as img:
        assert img.size == (1024, 768)


def _content_box(image, clear_rgb8):
    """Bounding box of non-clear pixels: ``(row0, row1, col0, col1)``."""
    clear = np.asarray(clear_rgb8, dtype=np.int16)
    mask = np.abs(image[..., :3].astype(np.int16) - clear).max(axis=2) > 2
    assert mask.any(), "the render is entirely clear colour"
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])


def test_the_gl_viewport_covers_the_whole_framebuffer(gl_session):
    """The direct check: ask GL what viewport it just rasterised into.

    ``glViewport`` silently clamps a request past ``GL_MAX_VIEWPORT_DIMS``, and
    a clamped viewport renders the scene into a sub-rectangle of a larger
    buffer -- a correct picture of the wrong size, letterboxed, with no error
    anywhere.  Reading ``GL_VIEWPORT`` back after the render settles it without
    inference.
    """
    from OpenGL.GL import GL_VIEWPORT, glGetIntegerv

    for width, height in [(1024, 384), (384, 1024), (1024, 1024), (640, 480)]:
        st.render_still(gl_session, width, height)
        viewport = [int(v) for v in np.asarray(glGetIntegerv(GL_VIEWPORT)).ravel()]
        assert viewport == [0, 0, width, height], (
            f"asked for {width}x{height} but GL rasterised into {viewport}"
        )


def test_content_is_centred_and_scales_with_the_output_height(gl_session):
    """Letterboxing, ruled out geometrically as well as by ``GL_VIEWPORT``.

    The scene is a square grid of boxes drawn by a perspective camera with a
    fixed *vertical* field of view and an aspect that follows the output size
    (``mat4_perspective(fov, width/height, ...)``).  Two consequences follow,
    and together they exclude a clamped sub-rectangle of any kind:

    * the content's vertical extent is a **constant fraction of the output
      height** at every size and aspect -- a clamp to a fixed sub-rectangle
      would instead pin the extent to a constant number of *pixels*, so the
      fraction would fall as the request grew;
    * the content is **centred** -- a clamped viewport is anchored at an
      origin, so its content cannot be.

    Measured on this scene: 0.75 of the output height at 1024x384, 512x192,
    384x1024 and 1024x1024 alike.  Horizontal extent is *not* checked the same
    way, and deliberately so: with a vertical FOV, a frame narrower than the
    subject crops it at the sides, which is what 384x1024 does (content reaches
    within 2 % of both edges).  That crop is itself evidence the viewport spans
    the full width rather than a sub-rectangle, so the horizontal assertion is
    centring only.
    """
    clear = gl_session.clear_rgb8
    height_fractions = {}
    for width, height in [(1024, 384), (512, 192), (384, 1024), (1024, 1024)]:
        image = st.render_still(gl_session, width, height)
        row0, row1, col0, col1 = _content_box(image, clear)

        col_centre = ((col0 + col1) / 2) / width
        row_centre = ((row0 + row1) / 2) / height
        assert col_centre == pytest.approx(0.5, abs=0.05), (
            f"{width}x{height}: content is centred at column fraction "
            f"{col_centre:.3f}, not near 0.5 -- consistent with a viewport "
            "anchored at an origin rather than covering the buffer"
        )
        assert row_centre == pytest.approx(0.5, abs=0.06), (
            f"{width}x{height}: content row centre {row_centre:.3f}"
        )
        height_fractions[(width, height)] = (row1 - row0 + 1) / height

    spread = max(height_fractions.values()) - min(height_fractions.values())
    assert spread < 0.02, (
        f"content height as a fraction of the output height varies across "
        f"sizes: {height_fractions} -- the render is not scaling with the "
        "requested size"
    )

    # And the tall frame crops the subject at both side edges, which a
    # sub-rectangle viewport could not do.
    tall = st.render_still(gl_session, 384, 1024)
    _r0, _r1, col0, col1 = _content_box(tall, clear)
    assert col0 < 0.03 * 384 and col1 > 0.97 * 384, (col0, col1)


# ---------------------------------------------------------------------------
# The claim: this is resolution, not interpolation
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def evidence(detail_scene_factory):
    """Render the same scene at 512 and 2048 once, and measure.

    Module-scoped: the 2048 render is the expensive part and four tests read
    the same numbers from it.
    """
    from faceforge import session as fs
    from faceforge.core.math_utils import vec3

    try:
        session = fs.Session.create(width=256, height=256, prefer="auto")
    except fs.SessionError as exc:
        pytest.skip(f"no usable GL context: {exc}")
    try:
        session.adopt_scene(detail_scene_factory())
        session.camera.set_aspect(1, 1)
        session.camera.look_at(vec3(0.0, 0.0, 168.0), vec3(0.0, 0.0, 0.0),
                               vec3(0.0, 1.0, 0.0))
        limits = session.gl_size_limits()
        if limits.max_square < LARGE:
            pytest.skip(f"this GL context caps stills at {limits.max_square}px")
        small = st.render_still(session, SMALL, SMALL)
        large = st.render_still(session, LARGE, LARGE)
    finally:
        session.close()
    return {"small": small, "large": large,
            "metrics": st.resolution_evidence(small, large)}


def test_the_two_renders_show_the_same_scene(evidence):
    """Guards the comparison: a large 'difference' must not be a different shot."""
    check = evidence["metrics"]["downsample_check"]
    assert check["mean_abs_rgba_0_255"] < 5.0, (
        f"area-downsampling the {LARGE}px render disagrees with the {SMALL}px "
        f"render by {check['mean_abs_rgba_0_255']:.2f}/255 on average, so the "
        "two are not the same view and the detail comparison is meaningless"
    )


def test_the_large_render_holds_detail_an_upscale_cannot_supply(evidence):
    """The load-bearing test of this whole step.

    Above the 512 render's Nyquist frequency there is nothing for a bicubic
    upscale to reconstruct.  The true 2048 render has real content there.
    Measured on this scene: 2.16 % of spectral energy for the true render
    against 0.25 % for the upscale, a factor of 8.5.  The thresholds below sit
    well inside that so the test reports a regression, not noise.
    """
    band = evidence["metrics"]["band_energy_above_small_nyquist"]
    assert band["true_render"] > 4 * band["bicubic_upscale"], (
        f"the {LARGE}px render holds only {band['ratio']:.2f}x the "
        f"above-Nyquist energy of an upscaled {SMALL}px render; at 1x it would "
        "be interpolation wearing a bigger filename"
    )
    assert band["true_render"] > 0.005
    assert band["bicubic_upscale"] < 0.01, (
        "the upscale should have almost no above-Nyquist energy; more than a "
        "little means the measurement is picking up an artefact"
    )


def test_the_true_render_is_sharper_by_gradient_energy_too(evidence):
    """A second, independent statistic, in case the spectral one is subtle."""
    grad = evidence["metrics"]["gradient_energy"]
    assert grad["true_render"] > 1.3 * grad["bicubic_upscale"]


def test_the_two_images_actually_disagree_pixel_by_pixel(evidence):
    disagreement = evidence["metrics"]["pixel_disagreement_vs_upscale"]
    assert disagreement["fraction_of_pixels_differing_over_1_255"] > 0.02
    assert disagreement["max_abs_luma"] > 0.1


def test_resolution_evidence_refuses_a_non_integer_factor(evidence):
    small = evidence["small"]
    with pytest.raises(ValueError, match="integer factor"):
        st.resolution_evidence(small, small[:600, :600])


def test_a_bicubic_upscale_is_what_it_claims_to_be(evidence):
    """Sanity-check the comparison baseline itself."""
    up = st.bicubic_upscale(evidence["small"], LARGE, LARGE)
    assert up.shape[:2] == (LARGE, LARGE)
    # Upscaling then area-downsampling by the same factor is near-lossless,
    # which is the property that makes it a fair stand-in for the old path.
    back = st.area_downsample(up, LARGE // SMALL)
    assert np.abs(back - evidence["small"].astype(np.float64)).mean() < 6.0
