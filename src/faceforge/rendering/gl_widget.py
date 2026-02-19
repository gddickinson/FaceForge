"""PySide6 QOpenGLWidget subclass bridging Qt and OpenGL rendering."""

import logging
import traceback
from typing import Optional

import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent, QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from faceforge.core.scene_graph import Scene
from faceforge.rendering.camera import Camera
from faceforge.rendering.lights import LightSetup
from faceforge.rendering.orbit_controls import OrbitControls
from faceforge.rendering.renderer import GLRenderer
from faceforge.rendering.selection_tool import SelectionTool

logger = logging.getLogger(__name__)

# Pre-allocated identity matrices (module-level, avoid per-frame allocation)
_IDENTITY4 = np.eye(4, dtype=np.float64)
_IDENTITY3 = np.eye(3, dtype=np.float64)


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

        # Selection tool (set externally when skinning is ready)
        self.selection_tool: Optional[SelectionTool] = None

        # Cached GL resources for selection overlay (avoid per-frame churn)
        self._sel_vao: int = 0
        self._sel_vbo: int = 0
        self._sel_vbo_capacity: int = 0  # bytes
        self._lasso_vao: int = 0
        self._lasso_vbo: int = 0
        self._lasso_vbo_capacity: int = 0

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

                # Render selected vertex highlights via GL_POINTS
                self._draw_selection_points()
                # Render lasso / selection border overlay via GL_LINES
                self._draw_lasso_overlay()
        except Exception:
            logger.error("paintGL failed:\n%s", traceback.format_exc())

    # ------------------------------------------------------------------
    # Selection overlay rendering (cached GL resources)
    # ------------------------------------------------------------------

    def _ensure_overlay_vao(self, attr: str, cap_attr: str, data: np.ndarray) -> tuple:
        """Ensure a cached VAO/VBO exists and has enough capacity for *data*.

        Returns (vao, vbo, vertex_count).
        """
        from OpenGL.GL import (
            GL_ARRAY_BUFFER, GL_STREAM_DRAW, GL_FLOAT, GL_FALSE,
            glGenVertexArrays, glGenBuffers, glBindVertexArray,
            glBindBuffer, glBufferData, glBufferSubData,
            glVertexAttribPointer, glEnableVertexAttribArray,
        )

        vao = getattr(self, f"_{attr}_vao")
        vbo = getattr(self, f"_{attr}_vbo")
        capacity = getattr(self, f"_{attr}_vbo_capacity")
        nbytes = data.nbytes

        if vao == 0:
            # First-time creation
            vao = glGenVertexArrays(1)
            vbo = glGenBuffers(1)
            setattr(self, f"_{attr}_vao", vao)
            setattr(self, f"_{attr}_vbo", vbo)
            capacity = 0

        if nbytes > capacity:
            # (Re)allocate buffer — grow with headroom to avoid frequent reallocs
            new_cap = max(nbytes, capacity * 2, 4096)
            glBindVertexArray(vao)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, new_cap, None, GL_STREAM_DRAW)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
            glEnableVertexAttribArray(0)
            glBindVertexArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            setattr(self, f"_{attr}_vbo_capacity", new_cap)

        # Stream data into existing buffer
        from OpenGL.GL import glBindBuffer as _bb, glBufferSubData as _bsd
        _bb(GL_ARRAY_BUFFER, vbo)
        _bsd(GL_ARRAY_BUFFER, 0, nbytes, data)
        _bb(GL_ARRAY_BUFFER, 0)

        return vao, vbo

    def _draw_selection_points(self) -> None:
        """Render selected vertices as bright yellow GL_POINTS."""
        if self.selection_tool is None or self.selection_tool.selection.total_count == 0:
            return

        positions = self.selection_tool.get_selected_positions()
        if positions is None or len(positions) == 0:
            return

        from OpenGL.GL import (
            GL_POINTS, GL_DEPTH_TEST,
            glPointSize, glDrawArrays, glEnable, glDisable,
            glBindVertexArray,
        )
        from faceforge.core.material import RenderMode
        from faceforge.core.math_utils import mat3_normal

        shader = self.renderer.get_shader(RenderMode.SOLID)
        shader.use()

        view = self.camera.get_view_matrix()
        proj = self.camera.get_projection_matrix()

        shader.set_uniform_mat4("uModelView", view)
        shader.set_uniform_mat4("uProjection", proj)
        try:
            normal_mat = mat3_normal(view)
        except np.linalg.LinAlgError:
            normal_mat = _IDENTITY3
        shader.set_uniform_mat3("uNormalMatrix", normal_mat)
        shader.set_uniform_vec3("uColor", (1.0, 1.0, 0.2))
        shader.set_uniform_float("uOpacity", 1.0)
        shader.set_uniform_float("uShininess", 1.0)
        shader.set_uniform_int("uUseVertexColor", 0)

        data = np.ascontiguousarray(positions, dtype=np.float32)
        vao, _ = self._ensure_overlay_vao("sel", "sel", data)

        glPointSize(6.0)
        glDisable(GL_DEPTH_TEST)
        glBindVertexArray(vao)
        glDrawArrays(GL_POINTS, 0, len(positions))
        glBindVertexArray(0)
        glEnable(GL_DEPTH_TEST)

    def _draw_lasso_overlay(self) -> None:
        """Render lasso polygon border via GL_LINE_STRIP with cached VBO."""
        if self.selection_tool is None or not self.selection_tool.active:
            return

        pts = self.selection_tool.lasso_points
        if not self.selection_tool._drag_started or len(pts) < 2:
            return

        from OpenGL.GL import (
            GL_LINE_STRIP, GL_DEPTH_TEST, GL_BLEND,
            GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
            glDrawArrays, glEnable, glDisable, glLineWidth, glBlendFunc,
            glBindVertexArray,
        )
        from faceforge.core.material import RenderMode

        shader = self.renderer.get_shader(RenderMode.SOLID)
        shader.use()

        # Orthographic projection: pixel coords → NDC
        w = float(self.width())
        h = float(self.height())
        ortho = np.array([
            [2.0 / w,  0.0,      0.0, -1.0],
            [0.0,     -2.0 / h,  0.0,  1.0],
            [0.0,      0.0,     -1.0,  0.0],
            [0.0,      0.0,      0.0,  1.0],
        ], dtype=np.float64)

        shader.set_uniform_mat4("uModelView", _IDENTITY4)
        shader.set_uniform_mat4("uProjection", ortho)
        shader.set_uniform_mat3("uNormalMatrix", _IDENTITY3)
        shader.set_uniform_float("uOpacity", 0.9)
        shader.set_uniform_float("uShininess", 1.0)
        shader.set_uniform_int("uUseVertexColor", 0)

        # Build vertex array — close polygon if 3+ points
        n = len(pts) + (1 if len(pts) >= 3 else 0)
        verts = np.empty((n, 3), dtype=np.float32)
        for i, p in enumerate(pts):
            verts[i, 0] = p[0]
            verts[i, 1] = p[1]
            verts[i, 2] = 0.0
        if len(pts) >= 3:
            verts[-1] = verts[0]

        vao, _ = self._ensure_overlay_vao("lasso", "lasso", verts)

        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Pass 1: bright yellow border
        shader.set_uniform_vec3("uColor", (1.0, 1.0, 0.0))
        glLineWidth(2.0)
        glBindVertexArray(vao)
        glDrawArrays(GL_LINE_STRIP, 0, n)
        glBindVertexArray(0)

        # Pass 2: wider glow
        shader.set_uniform_vec3("uColor", (1.0, 0.9, 0.0))
        shader.set_uniform_float("uOpacity", 0.4)
        glLineWidth(4.0)
        glBindVertexArray(vao)
        glDrawArrays(GL_LINE_STRIP, 0, n)
        glBindVertexArray(0)

        glLineWidth(1.0)
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)

    # ------------------------------------------------------------------
    # Mouse events -> orbit controls
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        button = self._qt_button_to_int(event.button())
        pos = event.position()

        # Selection tool intercepts left button when active
        if (self.selection_tool is not None
                and self.selection_tool.active
                and button == OrbitControls.BUTTON_LEFT):
            mods = event.modifiers()
            if self.selection_tool.on_mouse_press(pos.x(), pos.y(), 1, mods):
                event.accept()
                return

        if button is not None:
            self.orbit_controls.on_mouse_press(pos.x(), pos.y(), button)
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()

        # Selection tool lasso drag
        if (self.selection_tool is not None
                and self.selection_tool.active
                and self.selection_tool.on_mouse_move(pos.x(), pos.y())):
            self.update()
            event.accept()
            return

        self.orbit_controls.on_mouse_move(pos.x(), pos.y())
        # Feed mouse position to eye tracking
        if self.mouse_move_callback is not None:
            self.mouse_move_callback(pos.x(), pos.y())
        self.update()  # request repaint
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        pos = event.position()

        # Selection tool finalize
        if (self.selection_tool is not None
                and self.selection_tool.active
                and self.selection_tool._dragging):
            vp = self.camera.get_view_projection()
            mods = event.modifiers()
            self.selection_tool.on_mouse_release(
                pos.x(), pos.y(), vp,
                self.width(), self.height(), mods,
            )
            self.update()
            event.accept()
            return

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
        # Clean up cached overlay buffers
        from OpenGL.GL import glDeleteBuffers, glDeleteVertexArrays
        for attr in ("sel", "lasso"):
            vao = getattr(self, f"_{attr}_vao", 0)
            vbo = getattr(self, f"_{attr}_vbo", 0)
            if vbo:
                glDeleteBuffers(1, [vbo])
            if vao:
                glDeleteVertexArrays(1, [vao])
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
