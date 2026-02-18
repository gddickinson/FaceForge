"""Color picker button + QColorDialog."""

from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton, QColorDialog
from PySide6.QtGui import QColor
from PySide6.QtCore import Signal


class ColorPicker(QWidget):
    """A row with a label and a small coloured button that opens a colour dialog."""

    color_changed = Signal(QColor)

    def __init__(
        self,
        label: str,
        initial_color: QColor,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._color = QColor(initial_color)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        layout.setSpacing(6)

        self._label = QLabel(label)
        self._label.setObjectName("sliderLabel")
        self._label.setFixedWidth(80)
        layout.addWidget(self._label)

        self._button = QPushButton()
        self._button.setObjectName("colorButton")
        self._button.setFixedSize(28, 22)
        self._button.clicked.connect(self._pick_color)
        layout.addWidget(self._button)

        layout.addStretch()

        self._apply_swatch()

    # ── Public API ──

    @property
    def color(self) -> QColor:
        return QColor(self._color)

    def set_color(self, color: QColor) -> None:
        """Set the colour programmatically."""
        self._color = QColor(color)
        self._apply_swatch()

    # ── Internal ──

    def _apply_swatch(self) -> None:
        self._button.setStyleSheet(
            f"background-color: {self._color.name()}; border: 1px solid #252830;"
        )

    def _pick_color(self) -> None:
        color = QColorDialog.getColor(self._color, self, "Choose Color")
        if color.isValid():
            self._color = color
            self._apply_swatch()
            self.color_changed.emit(QColor(self._color))
