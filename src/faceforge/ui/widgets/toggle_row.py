"""Label + QCheckBox toggle widget."""

from PySide6.QtWidgets import QWidget, QHBoxLayout, QCheckBox
from PySide6.QtCore import Signal


class ToggleRow(QWidget):
    """A row containing a labelled checkbox toggle."""

    toggled = Signal(bool)

    def __init__(
        self,
        label: str,
        default: bool = True,
        indent: int = 0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(indent, 1, 0, 1)
        layout.setSpacing(6)

        self._checkbox = QCheckBox(label)
        self._checkbox.setChecked(default)
        self._checkbox.toggled.connect(self._on_toggled)
        layout.addWidget(self._checkbox)

    # ── Public API ──

    def set_checked(self, checked: bool) -> None:
        """Set the toggle state programmatically."""
        self._checkbox.setChecked(checked)

    @property
    def is_checked(self) -> bool:
        """Current toggle state."""
        return self._checkbox.isChecked()

    # ── Internal ──

    def _on_toggled(self, checked: bool) -> None:
        self.toggled.emit(checked)
