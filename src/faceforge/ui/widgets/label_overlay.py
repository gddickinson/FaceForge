"""QPainter 2D overlay for structure labels projected from 3D.

Supports two modes:
- Simple mode: text at projected 3D positions (original behavior).
- Illustration mode: Grey's Anatomy-style labels with leader lines arranged
  in left/right columns, used by illustration presets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import (
    QPainter, QColor, QFont, QPen, QFontMetrics, QBrush, QPainterPath,
)

from faceforge.core.math_utils import Vec3, Mat4, world_to_screen


@dataclass
class LabelDef:
    """Definition of a single illustration label."""
    mesh_name: str          # mesh.name to look up (e.g., "Temporalis L")
    display_text: str       # label text shown to user
    side: str = "auto"      # "left", "right", or "auto" (use mesh X coord)


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

        # Simple-mode fonts/colors
        self._font = QFont("monospace", 9)
        self._text_color = QColor(230, 230, 230)
        self._outline_color = QColor(10, 10, 10)

        # Illustration mode state
        self._illustration_mode: bool = False
        self._illustration_labels: list[LabelDef] = []
        self._illustration_positions: dict[str, Vec3] = {}  # mesh_name → world pos

        # Illustration fonts/colors (Grey's Anatomy aesthetic)
        self._illust_font = QFont("Georgia", 10)
        self._illust_font.setItalic(True)
        self._illust_text_color = QColor(200, 200, 200)       # light on dark bg
        self._illust_line_color = QColor(160, 160, 160, 180)   # subtle leader lines
        self._illust_dot_color = QColor(180, 60, 60)           # red anchor dots

        self.hide()

    # ── Public API ──

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        self.setVisible(enabled)

    def set_labels(self, labels: list[tuple[str, Vec3]]) -> None:
        self._labels = labels

    def set_view_proj(self, vp: Mat4) -> None:
        self._view_proj = vp

    def set_illustration_labels(
        self,
        labels: list[LabelDef],
        world_positions: dict[str, Vec3],
    ) -> None:
        """Enter illustration mode with the given labels and 3D positions."""
        self._illustration_mode = True
        self._illustration_labels = labels
        self._illustration_positions = world_positions

    def clear_illustration_labels(self) -> None:
        """Return to simple label mode."""
        self._illustration_mode = False
        self._illustration_labels = []
        self._illustration_positions = {}

    # ── Paint ──

    def paintEvent(self, event) -> None:
        if not self._enabled or self._view_proj is None:
            return

        if self._illustration_mode and self._illustration_labels:
            self._paint_illustration_labels()
        elif self._labels:
            self._paint_simple_labels()

    def _paint_simple_labels(self) -> None:
        """Original simple label rendering."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(self._font)

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

    def _paint_illustration_labels(self) -> None:
        """Grey's Anatomy-style label rendering with leader lines."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        if w < 100 or h < 100:
            painter.end()
            return

        # Margins for label columns
        left_margin = 40
        right_margin = 40
        top_margin = int(h * 0.10)
        bottom_margin = int(h * 0.10)
        usable_height = h - top_margin - bottom_margin

        # Project all labels to screen space
        left_items: list[tuple[LabelDef, float, float]] = []   # (label, sx, sy)
        right_items: list[tuple[LabelDef, float, float]] = []

        for label_def in self._illustration_labels:
            world_pos = self._illustration_positions.get(label_def.mesh_name)
            if world_pos is None:
                continue

            sx, sy, in_front = world_to_screen(world_pos, self._view_proj, w, h)
            if not in_front:
                continue

            # Determine side
            if label_def.side == "left":
                left_items.append((label_def, sx, sy))
            elif label_def.side == "right":
                right_items.append((label_def, sx, sy))
            else:
                # Auto: use world X coordinate (positive X = right side of body)
                if world_pos[0] >= 0:
                    right_items.append((label_def, sx, sy))
                else:
                    left_items.append((label_def, sx, sy))

        # Sort each column by projected Y (top to bottom)
        left_items.sort(key=lambda t: t[2])
        right_items.sort(key=lambda t: t[2])

        # Distribute labels evenly in each column
        fm = QFontMetrics(self._illust_font)
        min_spacing = max(20, fm.height() + 4)

        def distribute_y(items, count):
            """Return evenly spaced Y positions within usable area."""
            if count <= 0:
                return []
            if count == 1:
                return [top_margin + usable_height // 2]
            spacing = max(min_spacing, usable_height / (count - 1))
            # Center the block if it doesn't fill the space
            total = spacing * (count - 1)
            start = top_margin + max(0, (usable_height - total)) // 2
            return [int(start + i * spacing) for i in range(count)]

        left_ys = distribute_y(left_items, len(left_items))
        right_ys = distribute_y(right_items, len(right_items))

        # Set up pens and fonts
        painter.setFont(self._illust_font)

        line_pen = QPen(self._illust_line_color)
        line_pen.setWidthF(1.0)

        text_pen = QPen(self._illust_text_color)

        dot_brush = QBrush(self._illust_dot_color)
        dot_pen = QPen(self._illust_dot_color)
        dot_radius = 3.0

        # Draw left column
        for i, (label_def, anchor_sx, anchor_sy) in enumerate(left_items):
            label_y = left_ys[i]
            text_x = left_margin
            text_width = fm.horizontalAdvance(label_def.display_text)

            # Leader line: from text end → midpoint → anchor
            line_start_x = text_x + text_width + 6
            line_start_y = label_y
            mid_x = left_margin + int((anchor_sx - left_margin) * 0.5)
            mid_y = label_y

            painter.setPen(line_pen)
            painter.drawLine(
                QPointF(line_start_x, line_start_y),
                QPointF(mid_x, mid_y),
            )
            painter.drawLine(
                QPointF(mid_x, mid_y),
                QPointF(anchor_sx, anchor_sy),
            )

            # Anchor dot
            painter.setPen(dot_pen)
            painter.setBrush(dot_brush)
            painter.drawEllipse(
                QPointF(anchor_sx, anchor_sy), dot_radius, dot_radius,
            )
            painter.setBrush(Qt.BrushStyle.NoBrush)

            # Label text
            painter.setPen(text_pen)
            painter.drawText(text_x, label_y + fm.ascent() // 2, label_def.display_text)

        # Draw right column
        for i, (label_def, anchor_sx, anchor_sy) in enumerate(right_items):
            label_y = right_ys[i]
            text_width = fm.horizontalAdvance(label_def.display_text)
            text_x = w - right_margin - text_width

            # Leader line: from text start → midpoint → anchor
            line_start_x = text_x - 6
            line_start_y = label_y
            mid_x = anchor_sx + int((line_start_x - anchor_sx) * 0.5)
            mid_y = label_y

            painter.setPen(line_pen)
            painter.drawLine(
                QPointF(line_start_x, line_start_y),
                QPointF(mid_x, mid_y),
            )
            painter.drawLine(
                QPointF(mid_x, mid_y),
                QPointF(anchor_sx, anchor_sy),
            )

            # Anchor dot
            painter.setPen(dot_pen)
            painter.setBrush(dot_brush)
            painter.drawEllipse(
                QPointF(anchor_sx, anchor_sy), dot_radius, dot_radius,
            )
            painter.setBrush(Qt.BrushStyle.NoBrush)

            # Label text
            painter.setPen(text_pen)
            painter.drawText(text_x, label_y + fm.ascent() // 2, label_def.display_text)

        painter.end()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.parent():
            self.setGeometry(self.parent().rect())
