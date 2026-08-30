"""The GUI's still export, exercised without Qt.

``VideoExporter.export_screenshot`` used to grab the widget's framebuffer and
scale it up.  It now renders at the requested size through an offscreen FBO in
the widget's own GL context.  That change is only worth anything if it is
tested, and testing it through Qt is not possible in this environment
(``QT_QPA_PLATFORM=cocoa`` hangs; ``offscreen`` gives no usable GL context).

So the test drives the exporter through the small protocol it actually needs --
``scene``, ``renderer``, ``camera``, ``lights``, ``width()``, ``height()``,
``makeCurrent()`` -- backed by a real headless GL context and the real
``GLRenderer`` from a :class:`~faceforge.session.Session`.  Everything under
test is the production code path; only the Qt widget is stood in for.  What is
therefore *not* covered here is Qt's own context handling
(``makeCurrent``/``doneCurrent`` on a live ``QOpenGLWidget``), and this module
does not claim it is.

Marked ``slow``: needs a GL context.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.slow

from faceforge.export import still as st                   # noqa: E402
from faceforge.export.video_export import VideoExporter    # noqa: E402
from tests.export.conftest import box_mesh                 # noqa: E402


class StubGLWidget:
    """The subset of ``GLViewport`` that the still path touches.

    ``makeCurrent``/``doneCurrent`` are no-ops because the context this
    session's renderer lives in is already current -- which is exactly what a
    real ``QOpenGLWidget.makeCurrent()`` would arrange.
    """

    def __init__(self, session, size=(320, 240)):
        self.scene = session.scene
        self.renderer = session.renderer
        self.camera = session.camera
        self.lights = session.lights
        self._size = size
        self.make_current_calls = 0
        self.done_current_calls = 0
        self.grab_calls = 0

    def width(self):
        return self._size[0]

    def height(self):
        return self._size[1]

    def makeCurrent(self):                               # noqa: N802 - Qt name
        self.make_current_calls += 1

    def doneCurrent(self):                               # noqa: N802 - Qt name
        self.done_current_calls += 1

    def grabFramebuffer(self):                           # noqa: N802 - Qt name
        """Records that the upscaling fallback was reached."""
        self.grab_calls += 1
        raise AssertionError(
            "the FBO path should not have fallen back to a window grab"
        )


@pytest.fixture
def stub_widget():
    from faceforge import session as fs
    from faceforge.core.math_utils import vec3
    from faceforge.core.scene_graph import Scene, SceneNode

    try:
        session = fs.Session.create(width=320, height=240, prefer="auto")
    except fs.SessionError as exc:
        pytest.skip(f"no usable GL context: {exc}")
    try:
        scene = Scene()
        for ix in range(6):
            for iy in range(6):
                mesh = box_mesh(f"Mandible_{ix}_{iy}", source_id="FMA52748",
                                centre=((ix - 3) * 12.0, (iy - 3) * 12.0, 0.0),
                                size=(7.0, 7.0, 7.0))
                node = SceneNode(name=f"b{ix}_{iy}")
                node.mesh = mesh
                scene.add(node)
        scene.update()
        session.adopt_scene(scene)
        session.camera.look_at(vec3(0.0, 0.0, 120.0), vec3(0.0, 0.0, 0.0),
                               vec3(0.0, 1.0, 0.0))
        yield StubGLWidget(session)
    finally:
        session.close()


def test_export_screenshot_renders_at_the_requested_size(stub_widget, tmp_path):
    """A 1024px still from a 320px widget must be a 1024px *render*."""
    exporter = VideoExporter(stub_widget)
    out = tmp_path / "still.png"
    assert exporter.export_screenshot(str(out), width=1024, height=1024) is True
    assert exporter.last_screenshot_method == "offscreen-fbo"
    assert stub_widget.grab_calls == 0

    from PIL import Image

    with Image.open(out) as img:
        assert img.size == (1024, 1024)


def test_the_gui_still_beats_an_upscale_of_the_widget_sized_grab(
    stub_widget, tmp_path,
):
    """The audit finding, measured on the GUI path itself.

    The widget is 320x240.  A 960x720 still through the FBO is compared with a
    bicubic upscale of a 320x240 render of the same view -- which is what the
    old code produced.  The comparison is on the square centre crop so the
    factor is an exact integer, which
    :func:`faceforge.export.still.resolution_evidence` requires.
    """
    exporter = VideoExporter(stub_widget)
    big = tmp_path / "big.png"
    exporter.export_still(str(big), 960, 960)
    small = tmp_path / "small.png"
    exporter.export_still(str(small), 320, 320)

    from PIL import Image

    with Image.open(big) as img:
        large_arr = np.asarray(img.convert("RGBA"))
    with Image.open(small) as img:
        small_arr = np.asarray(img.convert("RGBA"))

    metrics = st.resolution_evidence(small_arr, large_arr)
    band = metrics["band_energy_above_small_nyquist"]
    assert band["true_render"] > 3 * band["bicubic_upscale"], band
    assert metrics["downsample_check"]["mean_abs_rgba_0_255"] < 6.0


def test_the_live_viewport_is_restored_afterwards(stub_widget, tmp_path):
    """An export must not leave the on-screen view resized or distorted."""
    before_aspect = stub_widget.camera.aspect
    exporter = VideoExporter(stub_widget)
    exporter.export_still(str(tmp_path / "s.png"), 800, 600)

    assert stub_widget.camera.aspect == pytest.approx(320 / 240)
    assert stub_widget.camera.aspect == pytest.approx(before_aspect)
    assert stub_widget.make_current_calls == 1
    assert stub_widget.done_current_calls == 1


def test_an_oversize_still_is_refused_and_the_viewport_survives(
    stub_widget, tmp_path,
):
    limits = st.query_size_limits()
    over = limits.max_square + 1
    exporter = VideoExporter(stub_widget)

    with pytest.raises(st.StillSizeError):
        exporter.export_still(str(tmp_path / "huge.png"), over, over)
    assert not (tmp_path / "huge.png").exists()
    assert stub_widget.camera.aspect == pytest.approx(320 / 240)

    # A normal export still works after the refusal.
    assert exporter.export_screenshot(str(tmp_path / "ok.png"),
                                      width=512, height=512) is True
    assert exporter.last_screenshot_method == "offscreen-fbo"


def test_upscaling_can_be_refused_outright(stub_widget, tmp_path):
    """A figure pipeline should be able to say "never interpolate"."""
    stub_widget.scene = None            # forces the FBO path to give up
    exporter = VideoExporter(stub_widget)
    assert exporter.export_screenshot(
        str(tmp_path / "no.png"), width=512, height=512,
        allow_upscale_fallback=False,
    ) is False
    assert not (tmp_path / "no.png").exists()
    assert stub_widget.grab_calls == 0
