"""PySide6 QOpenGLWidget subclass bridging Qt and OpenGL rendering."""

import logging
import traceback
from typing import Optional

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent, QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from faceforge.core.scene_graph import Scene
from faceforge.rendering.camera import Camera
from faceforge.rendering.lights import LightSetup
from faceforge.rendering.orbit_controls import OrbitControls
from faceforge.rendering.renderer import GLRenderer

logger = logging.getLogger(__name__)


def create_gl_format() -> QSurfaceFormat:
    """Create an OpenGL 3.3 core-profile surface format with multisampling."""
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setSamples(4)
    fmt.setDepthBufferSize(24)
    fmt.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
    return fmt


class GLViewport(QOpenGLWidget):
    """OpenGL viewport widget that renders a Scene using the GLRenderer.

    Set the ``scene`` attribute to a :class:`Scene` instance before the first
    paint.  The widget drives rendering at ~60 fps via a QTimer.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFormat(create_gl_format())

        # Rendering components (created lazily in initializeGL)
        self.renderer: GLRenderer = GLRenderer()
        self.camera: Camera = Camera()
        self.orbit_controls: OrbitControls = OrbitControls(self.camera)
        self.lights: LightSetup = LightSetup()

        # Scene -- set externally before rendering starts
        self.scene: Optional[Scene] = None

        # Refresh timer (~60 fps)
        self._timer = QTimer(self)
        self._timer.setInterval(16)  # ~60 fps
        self._timer.timeout.connect(self._on_timer)

        # Accept focus for keyboard events
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Track mouse for hover and eye tracking
        self.setMouseTracking(True)

        # Optional callback for mouse move (eye tracking)
        self.mouse_move_callback = None

    # ------------------------------------------------------------------
    # QOpenGLWidget overrides
    # ------------------------------------------------------------------

    def initializeGL(self) -> None:
        """Called once when the GL context is ready."""
        try:
            logger.info("GLViewport: initialising OpenGL.")
            self.renderer.init_gl()
            self._timer.start()
        except Exception:
            logger.error("initializeGL failed:\n%s", traceback.format_exc())

    def resizeGL(self, w: int, h: int) -> None:
        """Called on every resize.

        Qt passes logical dimensions; we scale by devicePixelRatio for
        the actual framebuffer size (needed on Retina/HiDPI displays).
        """
        dpr = self.devicePixelRatio()
        pw = int(w * dpr)
        ph = int(h * dpr)
        self.camera.set_aspect(w, h)
        self.renderer.resize(pw, ph)

    def paintGL(self) -> None:
        """Called each frame to render the scene."""
        try:
            if self.scene is not None:
                self.orbit_controls.update()
                self.renderer.render(self.scene, self.camera, self.lights)
        except Exception:
            logger.error("paintGL failed:\n%s", traceback.format_exc())

    # ------------------------------------------------------------------
    # Mouse events -> orbit controls
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        button = self._qt_button_to_int(event.button())
        if button is not None:
            pos = event.position()
            self.orbit_controls.on_mouse_press(pos.x(), pos.y(), button)
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        self.orbit_controls.on_mouse_move(pos.x(), pos.y())
        # Feed mouse position to eye tracking
        if self.mouse_move_callback is not None:
            self.mouse_move_callback(pos.x(), pos.y())
        self.update()  # request repaint
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self.orbit_controls.on_mouse_release()
        event.accept()

    def wheelEvent(self, event: QWheelEvent) -> None:
        # angleDelta().y() is typically +/-120 per notch
        delta = event.angleDelta().y() / 120.0
        self.orbit_controls.on_scroll(delta)
        self.update()
        event.accept()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Explicitly release GL resources. Call before the widget is destroyed."""
        self._timer.stop()
        self.makeCurrent()
        self.renderer.destroy()
        self.doneCurrent()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_timer(self) -> None:
        """Timer callback to drive continuous rendering."""
        self.update()

    @staticmethod
    def _qt_button_to_int(qt_button) -> int | None:
        """Map Qt mouse button to orbit-control button constant."""
        if qt_button == Qt.MouseButton.LeftButton:
            return OrbitControls.BUTTON_LEFT
        if qt_button == Qt.MouseButton.MiddleButton:
            return OrbitControls.BUTTON_MIDDLE
        if qt_button == Qt.MouseButton.RightButton:
            return OrbitControls.BUTTON_RIGHT
        return None
