"""QPainter 2D overlay for structure labels projected from 3D."""

from typing import Optional

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPainter, QColor, QFont, QPen, QFontMetrics

from faceforge.core.math_utils import Vec3, Mat4, world_to_screen


class LabelOverlay(QWidget):
    """Transparent overlay that draws 2D text labels at projected 3D positions.

    Parented to the GL viewport widget, same pattern as LoadingOverlay.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._enabled: bool = False
        self._labels: list[tuple[str, Vec3]] = []  # (name, world_pos)
        self._view_proj: Optional[Mat4] = None

        self._font = QFont("monospace", 9)
        self._text_color = QColor(230, 230, 230)
        self._outline_color = QColor(10, 10, 10)

        self.hide()

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        self.setVisible(enabled)

    def set_labels(self, labels: list[tuple[str, Vec3]]) -> None:
        self._labels = labels

    def set_view_proj(self, vp: Mat4) -> None:
        self._view_proj = vp

    def paintEvent(self, event) -> None:
        if not self._enabled or self._view_proj is None or not self._labels:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(self._font)
        fm = QFontMetrics(self._font)

        w = self.width()
        h = self.height()

        outline_pen = QPen(self._outline_color)
        outline_pen.setWidth(2)
        text_pen = QPen(self._text_color)

        for name, world_pos in self._labels:
            sx, sy, in_front = world_to_screen(world_pos, self._view_proj, w, h)
            if not in_front:
                continue
            if sx < -50 or sx > w + 50 or sy < -20 or sy > h + 20:
                continue

            ix, iy = int(sx), int(sy)

            # Draw text with dark outline for readability
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    painter.setPen(outline_pen)
                    painter.drawText(ix + dx, iy + dy, name)

            painter.setPen(text_pen)
            painter.drawText(ix, iy, name)

        painter.end()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.parent():
            self.setGeometry(self.parent().rect())
