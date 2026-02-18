"""Grid of expression preset buttons."""

from PySide6.QtWidgets import QWidget, QGridLayout, QPushButton, QSizePolicy
from PySide6.QtCore import Signal


class ExpressionGrid(QWidget):
    """A grid of buttons for selecting expression presets.

    Arranges buttons in a 2-column grid. The currently active expression
    is visually highlighted.
    """

    expression_selected = Signal(str)

    _COLUMNS = 2

    def __init__(
        self,
        expressions: list[str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._buttons: dict[str, QPushButton] = {}
        self._active: str | None = None

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(4)

        for i, name in enumerate(expressions):
            btn = QPushButton(name.capitalize())
            btn.setObjectName("expressionButton")
            btn.setCheckable(True)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.clicked.connect(lambda checked, n=name: self._on_clicked(n))
            row = i // self._COLUMNS
            col = i % self._COLUMNS
            layout.addWidget(btn, row, col)
            self._buttons[name] = btn

    # ── Public API ──

    def set_active(self, name: str) -> None:
        """Highlight *name* as the active expression, un-highlighting the rest."""
        self._active = name
        for btn_name, btn in self._buttons.items():
            btn.setChecked(btn_name == name)

    # ── Internal ──

    def _on_clicked(self, name: str) -> None:
        self.set_active(name)
        self.expression_selected.emit(name)
