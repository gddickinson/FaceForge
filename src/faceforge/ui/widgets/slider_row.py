"""Label + QSlider + value display widget."""

from PySide6.QtWidgets import QWidget, QHBoxLayout, QSlider, QLabel, QSizePolicy
from PySide6.QtCore import Qt, Signal


class SliderRow(QWidget):
    """Horizontal row: fixed-width label, slider, and numeric value display.

    Maps an integer slider range (0-1000) to a floating-point value
    between *min_val* and *max_val*.
    """

    value_changed = Signal(float)

    _SLIDER_STEPS = 1000

    def __init__(
        self,
        label: str,
        min_val: float = -1.0,
        max_val: float = 1.0,
        default: float = 0.0,
        decimals: int = 2,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._min_val = min_val
        self._max_val = max_val
        self._decimals = decimals
        self._emitting = True

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        layout.setSpacing(6)

        # Label
        self._label = QLabel(label)
        self._label.setObjectName("sliderLabel")
        self._label.setFixedWidth(80)
        self._label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self._label)

        # Slider
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, self._SLIDER_STEPS)
        self._slider.setSingleStep(1)
        self._slider.setPageStep(50)
        self._slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self._slider)

        # Value display
        self._value_label = QLabel()
        self._value_label.setObjectName("valueLabel")
        self._value_label.setFixedWidth(40)
        self._value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self._value_label)

        # Set initial value
        self.set_value(default)

    # ── Properties ──

    @property
    def value(self) -> float:
        """Current float value."""
        return self._int_to_float(self._slider.value())

    @property
    def slider(self) -> QSlider:
        """The underlying QSlider (for connecting to signals like sliderReleased)."""
        return self._slider

    # ── Public API ──

    def set_value(self, val: float) -> None:
        """Set the slider value programmatically *without* emitting value_changed."""
        self._emitting = False
        clamped = max(self._min_val, min(self._max_val, val))
        self._slider.setValue(self._float_to_int(clamped))
        self._update_display(clamped)
        self._emitting = True

    def set_range(self, min_val: float, max_val: float) -> None:
        """Update the float range."""
        self._min_val = min_val
        self._max_val = max_val
        self.set_value(self.value)

    # ── Internal ──

    def _float_to_int(self, val: float) -> int:
        frac = (val - self._min_val) / (self._max_val - self._min_val)
        return round(frac * self._SLIDER_STEPS)

    def _int_to_float(self, ival: int) -> float:
        frac = ival / self._SLIDER_STEPS
        return self._min_val + frac * (self._max_val - self._min_val)

    def _update_display(self, val: float) -> None:
        self._value_label.setText(f"{val:.{self._decimals}f}")

    def _on_slider_changed(self, ival: int) -> None:
        val = self._int_to_float(ival)
        self._update_display(val)
        if self._emitting:
            self.value_changed.emit(val)
