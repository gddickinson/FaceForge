"""Timeline canvas widget for keyframe visualisation and interaction.

Custom QPainter-based widget that draws keyframe diamonds on a time axis,
supports mouse interaction for selecting, moving, and zooming.
"""

from PySide6.QtWidgets import QWidget, QMenu
from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont,
    QMouseEvent, QWheelEvent, QPaintEvent,
)


# Colour constants
_BG_COLOR = QColor(30, 32, 38)
_GRID_COLOR = QColor(50, 52, 60)
_TRACK_BG = QColor(38, 40, 48)
_KEYFRAME_COLOR = QColor(220, 180, 60)
_KEYFRAME_SELECTED = QColor(100, 200, 255)
_CURSOR_COLOR = QColor(255, 80, 80, 180)
_TEXT_COLOR = QColor(180, 180, 180)


class TimelineCanvas(QWidget):
    """Custom widget for keyframe timeline visualisation.

    Signals
    -------
    keyframe_moved(int, float)
        Emitted when a keyframe is dragged to a new time.
    keyframe_selected(int)
        Emitted when a keyframe is clicked.
    """

    keyframe_moved = Signal(int, float)
    keyframe_selected = Signal(int)

    def __init__(self, clip=None, parent=None):
        super().__init__(parent)
        self._clip = clip
        self._duration = 20.0
        self._cursor_time = 0.0

        # View state
        self._time_offset = 0.0    # left edge time
        self._pixels_per_sec = 40.0  # zoom level
        self._track_height = 24
        self._header_height = 30

        # Interaction state
        self._selected_kf = -1
        self._dragging_kf = -1
        self._drag_start_x = 0.0
        self._drag_original_time = 0.0
        self._panning = False
        self._pan_start = QPointF()

        self.setMinimumHeight(200)
        self.setMinimumWidth(400)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    @property
    def cursor_time(self) -> float:
        return self._cursor_time

    @property
    def selected_keyframe_index(self) -> int:
        return self._selected_kf

    def set_clip(self, clip) -> None:
        """Set the animation clip to display."""
        self._clip = clip
        if clip and clip.duration > 0:
            self._duration = clip.duration
        self._selected_kf = -1
        self.update()

    def set_cursor_time(self, t: float) -> None:
        """Set the playback cursor position."""
        self._cursor_time = t
        self.update()

    def set_duration(self, duration: float) -> None:
        self._duration = max(1.0, duration)
        self.update()

    # ── Coordinate conversion ──

    def _time_to_x(self, t: float) -> float:
        return (t - self._time_offset) * self._pixels_per_sec

    def _x_to_time(self, x: float) -> float:
        return x / self._pixels_per_sec + self._time_offset

    # ── Paint ──

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # Background
        painter.fillRect(0, 0, w, h, _BG_COLOR)

        # Time axis header
        painter.setPen(QPen(_TEXT_COLOR))
        painter.setFont(QFont("Courier New", 8))

        # Grid lines
        grid_interval = self._get_grid_interval()
        t = (int(self._time_offset / grid_interval)) * grid_interval
        while t <= self._time_offset + w / self._pixels_per_sec:
            x = self._time_to_x(t)
            if 0 <= x <= w:
                painter.setPen(QPen(_GRID_COLOR))
                painter.drawLine(int(x), self._header_height, int(x), h)
                painter.setPen(QPen(_TEXT_COLOR))
                label = f"{t:.1f}s" if t < 60 else f"{int(t//60)}:{t%60:04.1f}"
                painter.drawText(int(x) + 2, self._header_height - 4, label)
            t += grid_interval

        # Track backgrounds
        if self._clip:
            n_kf = len(self._clip.keyframes)
            y = self._header_height
            painter.fillRect(0, y, w, self._track_height, _TRACK_BG)

            # Draw connections between keyframes
            if n_kf > 1:
                painter.setPen(QPen(QColor(100, 100, 120), 1))
                for i in range(n_kf - 1):
                    x1 = self._time_to_x(self._clip.keyframes[i].time)
                    x2 = self._time_to_x(self._clip.keyframes[i + 1].time)
                    cy = y + self._track_height // 2
                    painter.drawLine(int(x1), cy, int(x2), cy)

            # Draw keyframe diamonds
            for i, kf in enumerate(self._clip.keyframes):
                x = self._time_to_x(kf.time)
                cy = y + self._track_height // 2
                size = 6

                if i == self._selected_kf:
                    color = _KEYFRAME_SELECTED
                else:
                    color = _KEYFRAME_COLOR

                painter.setBrush(QBrush(color))
                painter.setPen(QPen(color.darker(130), 1))

                # Diamond shape
                points = [
                    QPointF(x, cy - size),
                    QPointF(x + size, cy),
                    QPointF(x, cy + size),
                    QPointF(x - size, cy),
                ]
                painter.drawPolygon(points)

        # Cursor line
        cursor_x = self._time_to_x(self._cursor_time)
        if 0 <= cursor_x <= w:
            painter.setPen(QPen(_CURSOR_COLOR, 2))
            painter.drawLine(int(cursor_x), 0, int(cursor_x), h)

        painter.end()

    def _get_grid_interval(self) -> float:
        """Choose grid interval based on zoom level."""
        pixels = self._pixels_per_sec
        if pixels > 100:
            return 0.5
        elif pixels > 40:
            return 1.0
        elif pixels > 15:
            return 2.0
        elif pixels > 5:
            return 5.0
        else:
            return 10.0

    # ── Mouse events ──

    def mousePressEvent(self, event: QMouseEvent) -> None:
        pos = event.position()

        if event.button() == Qt.MouseButton.LeftButton:
            # Check if clicking on a keyframe
            kf_idx = self._hit_test_keyframe(pos.x(), pos.y())
            if kf_idx >= 0:
                self._selected_kf = kf_idx
                self._dragging_kf = kf_idx
                self._drag_start_x = pos.x()
                self._drag_original_time = self._clip.keyframes[kf_idx].time
                self.keyframe_selected.emit(kf_idx)
            else:
                # Click on empty area: move cursor
                self._cursor_time = max(0.0, self._x_to_time(pos.x()))
                self._selected_kf = -1
            self.update()

        elif event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = pos

        elif event.button() == Qt.MouseButton.RightButton:
            kf_idx = self._hit_test_keyframe(pos.x(), pos.y())
            if kf_idx >= 0:
                self._show_context_menu(event.globalPosition().toPoint(), kf_idx)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()

        if self._dragging_kf >= 0:
            # Drag keyframe horizontally
            dx = pos.x() - self._drag_start_x
            new_time = self._drag_original_time + dx / self._pixels_per_sec
            new_time = max(0.0, new_time)
            self.keyframe_moved.emit(self._dragging_kf, new_time)
            self.update()

        elif self._panning:
            dx = pos.x() - self._pan_start.x()
            self._time_offset -= dx / self._pixels_per_sec
            self._time_offset = max(0.0, self._time_offset)
            self._pan_start = pos
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._dragging_kf = -1
        self._panning = False

    def wheelEvent(self, event: QWheelEvent) -> None:
        # Zoom with scroll wheel
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 1.0 / 1.15

        # Zoom toward mouse position
        mouse_time = self._x_to_time(event.position().x())
        self._pixels_per_sec = max(2.0, min(500.0,
                                             self._pixels_per_sec * factor))
        # Adjust offset to keep mouse position stable
        new_mouse_x = (mouse_time - self._time_offset) * self._pixels_per_sec
        self._time_offset = mouse_time - event.position().x() / self._pixels_per_sec
        self._time_offset = max(0.0, self._time_offset)
        self.update()

    def _hit_test_keyframe(self, x: float, y: float) -> int:
        """Test if a point is near a keyframe diamond. Returns index or -1."""
        if self._clip is None:
            return -1

        kf_y = self._header_height + self._track_height // 2
        hit_radius = 8

        for i, kf in enumerate(self._clip.keyframes):
            kf_x = self._time_to_x(kf.time)
            if abs(x - kf_x) < hit_radius and abs(y - kf_y) < hit_radius:
                return i
        return -1

    def _show_context_menu(self, pos, kf_idx: int) -> None:
        """Show right-click context menu for a keyframe."""
        menu = QMenu(self)

        delete_action = menu.addAction("Delete Keyframe")
        delete_action.triggered.connect(
            lambda: self._delete_keyframe(kf_idx))

        easing_menu = menu.addMenu("Easing")
        for easing in ("linear", "ease_in", "ease_out", "ease_in_out"):
            action = easing_menu.addAction(easing)
            action.triggered.connect(
                lambda checked, e=easing: self._set_easing(kf_idx, e))

        menu.exec(pos)

    def _delete_keyframe(self, idx: int) -> None:
        if self._clip and 0 <= idx < len(self._clip.keyframes):
            self._clip.keyframes.pop(idx)
            self._selected_kf = -1
            self.update()

    def _set_easing(self, idx: int, easing: str) -> None:
        if self._clip and 0 <= idx < len(self._clip.keyframes):
            self._clip.keyframes[idx].easing = easing
