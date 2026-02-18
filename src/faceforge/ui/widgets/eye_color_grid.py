"""Grid of eye color preset buttons with color swatches."""

from PySide6.QtWidgets import QWidget, QGridLayout, QPushButton, QSizePolicy
from PySide6.QtCore import Signal

from faceforge.core.config_loader import load_config


def _load_eye_colors() -> dict[str, dict]:
    """Load eye color presets from config."""
    try:
        return load_config("eye_colors.json")
    except Exception:
        return {"brown": {"mid": [0.42, 0.26, 0.13]}}


class EyeColorGrid(QWidget):
    """A grid of buttons for selecting eye color presets.

    Each button shows the color name with a tinted background.
    Emits *color_selected(name, r, g, b)* when clicked.
    """

    color_selected = Signal(str, float, float, float)

    _COLUMNS = 3

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._buttons: dict[str, QPushButton] = {}
        self._active: str | None = None
        self._colors = _load_eye_colors()

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(4)

        for i, (name, data) in enumerate(self._colors.items()):
            r, g, b = data.get("mid", [0.5, 0.5, 0.5])
            btn = QPushButton(name.capitalize())
            btn.setCheckable(True)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            # Color swatch via stylesheet
            ri, gi, bi = int(r * 255), int(g * 255), int(b * 255)
            # Use lighter text for dark backgrounds
            luma = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "#fff" if luma < 0.4 else "#000"
            btn.setStyleSheet(
                f"QPushButton {{ background-color: rgb({ri},{gi},{bi}); "
                f"color: {text_color}; border: 1px solid #555; border-radius: 3px; "
                f"padding: 4px; font-weight: bold; }}"
                f"QPushButton:checked {{ border: 2px solid #fff; }}"
            )
            btn.clicked.connect(
                lambda checked, n=name, cr=r, cg=g, cb=b: self._on_clicked(n, cr, cg, cb)
            )
            row = i // self._COLUMNS
            col = i % self._COLUMNS
            layout.addWidget(btn, row, col)
            self._buttons[name] = btn

    def set_active(self, name: str) -> None:
        """Highlight *name* as the active color."""
        self._active = name
        for btn_name, btn in self._buttons.items():
            btn.setChecked(btn_name == name)

    def _on_clicked(self, name: str, r: float, g: float, b: float) -> None:
        self.set_active(name)
        self.color_selected.emit(name, r, g, b)
