"""Animation transport controls: play/pause, stop, speed, timeline slider."""

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QComboBox, QSlider, QLabel,
    QSizePolicy,
)
from PySide6.QtCore import Qt, Signal

from faceforge.core.events import EventBus, EventType


class TransportControls(QWidget):
    """Compact horizontal transport bar for animation playback.

    Emits animation events via EventBus:
      ANIM_PLAY, ANIM_PAUSE, ANIM_STOP, ANIM_SEEK, ANIM_SPEED
    """

    def __init__(self, event_bus: EventBus, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._bus = event_bus
        self._is_playing = False
        self._seeking = False  # True while user is dragging the slider
        self._duration = 0.0

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Play/Pause button
        self._play_btn = QPushButton("\u25B6")  # ▶
        self._play_btn.setObjectName("transportButton")
        self._play_btn.setFixedWidth(32)
        self._play_btn.setToolTip("Play / Pause")
        self._play_btn.clicked.connect(self._on_play_pause)
        layout.addWidget(self._play_btn)

        # Stop button
        self._stop_btn = QPushButton("\u23F9")  # ⏹
        self._stop_btn.setObjectName("transportButton")
        self._stop_btn.setFixedWidth(32)
        self._stop_btn.setToolTip("Stop (reset to start)")
        self._stop_btn.clicked.connect(self._on_stop)
        layout.addWidget(self._stop_btn)

        # Speed combo
        self._speed_combo = QComboBox()
        self._speed_combo.addItems(["0.25x", "0.5x", "1x", "2x"])
        self._speed_combo.setCurrentText("1x")
        self._speed_combo.setFixedWidth(60)
        self._speed_combo.currentTextChanged.connect(self._on_speed_changed)
        layout.addWidget(self._speed_combo)

        # Timeline slider
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 1000)
        self._slider.setValue(0)
        self._slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._slider.sliderPressed.connect(self._on_slider_pressed)
        self._slider.sliderReleased.connect(self._on_slider_released)
        self._slider.valueChanged.connect(self._on_slider_value_changed)
        layout.addWidget(self._slider)

        # Time label
        self._time_label = QLabel("0:00 / 0:00")
        self._time_label.setFixedWidth(80)
        self._time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._time_label)

        self.setFixedHeight(40)

    # ── Public API ────────────────────────────────────────────────

    def set_progress(self, progress: float, current_time: float = 0.0,
                     duration: float = 0.0) -> None:
        """Update slider and time label without re-emitting seek.

        Called each frame by the simulation loop.
        """
        self._duration = duration
        if not self._seeking:
            self._slider.blockSignals(True)
            self._slider.setValue(int(progress * 1000))
            self._slider.blockSignals(False)
        self._time_label.setText(
            f"{_format_time(current_time)} / {_format_time(duration)}"
        )

    def set_playing(self, playing: bool) -> None:
        """Update play/pause button state."""
        self._is_playing = playing
        self._play_btn.setText("\u23F8" if playing else "\u25B6")  # ⏸ or ▶

    def set_duration(self, duration: float) -> None:
        """Set the clip duration for time display."""
        self._duration = duration

    # ── Slots ─────────────────────────────────────────────────────

    def _on_play_pause(self) -> None:
        if self._is_playing:
            self._bus.publish(EventType.ANIM_PAUSE)
            self.set_playing(False)
        else:
            self._bus.publish(EventType.ANIM_PLAY)
            self.set_playing(True)

    def _on_stop(self) -> None:
        self._bus.publish(EventType.ANIM_STOP)
        self.set_playing(False)

    def _on_speed_changed(self, text: str) -> None:
        try:
            speed = float(text.replace("x", ""))
        except ValueError:
            speed = 1.0
        self._bus.publish(EventType.ANIM_SPEED, speed=speed)

    def _on_slider_pressed(self) -> None:
        self._seeking = True

    def _on_slider_released(self) -> None:
        self._seeking = False
        pos = self._slider.value() / 1000.0
        self._bus.publish(EventType.ANIM_SEEK, position=pos)

    def _on_slider_value_changed(self, value: int) -> None:
        if self._seeking:
            pos = value / 1000.0
            self._bus.publish(EventType.ANIM_SEEK, position=pos)


def _format_time(seconds: float) -> str:
    """Format seconds as M:SS."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"
