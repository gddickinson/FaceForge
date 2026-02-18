"""Alignment tab: face-to-skull alignment sliders."""

from PySide6.QtWidgets import QScrollArea, QWidget, QVBoxLayout, QPushButton
from PySide6.QtCore import Qt

from faceforge.core.events import EventBus, EventType
from faceforge.core.state import StateManager
from faceforge.constants import (
    DEFAULT_FACE_SCALE,
    DEFAULT_FACE_OFFSET_X,
    DEFAULT_FACE_OFFSET_Y,
    DEFAULT_FACE_OFFSET_Z,
    DEFAULT_FACE_ROT_X_DEG,
)
from faceforge.ui.widgets.section_label import SectionLabel
from faceforge.ui.widgets.slider_row import SliderRow


# Default alignment values
_DEFAULTS = {
    "scale": DEFAULT_FACE_SCALE,
    "offset_x": DEFAULT_FACE_OFFSET_X,
    "offset_y": DEFAULT_FACE_OFFSET_Y,
    "offset_z": DEFAULT_FACE_OFFSET_Z,
    "rot_x": DEFAULT_FACE_ROT_X_DEG,
    "rot_y": 0.0,
    "rot_z": 0.0,
}


class AlignTab(QScrollArea):
    """Tab with sliders for adjusting face-to-skull alignment.

    Publishes ``ALIGNMENT_CHANGED`` events on every slider change.
    """

    def __init__(
        self,
        event_bus: EventBus,
        state: StateManager,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._bus = event_bus
        self._state = state

        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QScrollArea.Shape.NoFrame)

        container = QWidget()
        self._layout = QVBoxLayout(container)
        self._layout.setContentsMargins(8, 4, 8, 8)
        self._layout.setSpacing(2)
        self.setWidget(container)

        self._sliders: dict[str, SliderRow] = {}

        # ── Scale & Position ──
        self._layout.addWidget(SectionLabel("Scale & Position"))

        self._add_slider("scale", "Scale", 0.5, 2.0, _DEFAULTS["scale"], decimals=3)
        self._add_slider("offset_x", "Offset X", -20.0, 20.0, _DEFAULTS["offset_x"])
        self._add_slider("offset_y", "Offset Y", -30.0, 10.0, _DEFAULTS["offset_y"])
        self._add_slider("offset_z", "Offset Z", -10.0, 30.0, _DEFAULTS["offset_z"])

        # ── Rotation ──
        self._layout.addWidget(SectionLabel("Rotation"))

        self._add_slider("rot_x", "Rot X", 0.0, 180.0, _DEFAULTS["rot_x"], decimals=1)
        self._add_slider("rot_y", "Rot Y", -90.0, 90.0, _DEFAULTS["rot_y"], decimals=1)
        self._add_slider("rot_z", "Rot Z", -90.0, 90.0, _DEFAULTS["rot_z"], decimals=1)

        # ── Reset ──
        self._layout.addSpacing(12)
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.setObjectName("resetButton")
        reset_btn.clicked.connect(self._on_reset)
        self._layout.addWidget(reset_btn)

        self._layout.addStretch()

    # ── Helpers ──

    def _add_slider(
        self,
        field: str,
        label: str,
        min_val: float,
        max_val: float,
        default: float,
        decimals: int = 2,
    ) -> None:
        slider = SliderRow(label, min_val=min_val, max_val=max_val, default=default, decimals=decimals)
        slider.value_changed.connect(lambda v, f=field: self._on_changed(f, v))
        self._sliders[field] = slider
        self._layout.addWidget(slider)

    # ── Slots ──

    def _on_changed(self, field: str, value: float) -> None:
        self._bus.publish(EventType.ALIGNMENT_CHANGED, field=field, value=value)

    def _on_reset(self) -> None:
        for field, default in _DEFAULTS.items():
            self._sliders[field].set_value(default)
            self._bus.publish(EventType.ALIGNMENT_CHANGED, field=field, value=default)

    # ── External Updates ──

    def get_alignment(self) -> dict[str, float]:
        """Return current alignment values as a dict."""
        return {field: slider.value for field, slider in self._sliders.items()}
