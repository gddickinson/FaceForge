"""Animate tab: expressions grid, AU sliders, eye controls, head rotation, auto-animation toggles, speech."""

from PySide6.QtWidgets import (
    QScrollArea, QWidget, QVBoxLayout, QHBoxLayout, QSizePolicy,
    QLineEdit, QPushButton,
)
from PySide6.QtCore import Qt

from faceforge.core.events import EventBus, EventType
from faceforge.core.state import StateManager, AU_IDS
from faceforge.core.config_loader import load_config
from faceforge.ui.widgets.section_label import SectionLabel
from faceforge.ui.widgets.slider_row import SliderRow
from faceforge.ui.widgets.toggle_row import ToggleRow
from faceforge.ui.widgets.expression_grid import ExpressionGrid
from faceforge.ui.widgets.eye_color_grid import EyeColorGrid


# AU display names keyed by ID
_AU_NAMES: dict[str, str] = {}


def _load_au_names() -> dict[str, str]:
    """Load AU id-to-name mapping from config."""
    global _AU_NAMES
    if not _AU_NAMES:
        try:
            defs = load_config("au_definitions.json")
            _AU_NAMES = {entry["id"]: entry["name"] for entry in defs}
        except Exception:
            _AU_NAMES = {au: au for au in AU_IDS}
    return _AU_NAMES


class AnimateTab(QScrollArea):
    """Tab containing expression presets, AU sliders, eye/head controls, and
    auto-animation toggles.

    All slider changes are published through the *event_bus*.
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

        # Scroll area setup
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QScrollArea.Shape.NoFrame)

        container = QWidget()
        self._layout = QVBoxLayout(container)
        self._layout.setContentsMargins(8, 4, 8, 8)
        self._layout.setSpacing(2)
        self.setWidget(container)

        # Load configs
        au_names = _load_au_names()
        self._expressions = self._load_expressions()

        # ── 1. Expressions ──
        self._layout.addWidget(SectionLabel("Expressions"))
        self._expr_grid = ExpressionGrid(list(self._expressions.keys()))
        self._expr_grid.expression_selected.connect(self._on_expression_selected)
        self._expr_grid.set_active("neutral")
        self._layout.addWidget(self._expr_grid)

        # ── 2. Action Units ──
        self._layout.addWidget(SectionLabel("Action Units"))
        self._au_sliders: dict[str, SliderRow] = {}
        for au_id in AU_IDS:
            name = au_names.get(au_id, au_id)
            label = f"{au_id} {name}"
            slider = SliderRow(label, min_val=0.0, max_val=1.0, default=0.0, decimals=2)
            slider.value_changed.connect(lambda val, aid=au_id: self._on_au_changed(aid, val))
            self._au_sliders[au_id] = slider
            self._layout.addWidget(slider)

        # ── 3. Eyes ──
        self._layout.addWidget(SectionLabel("Eyes"))
        self._eye_x = SliderRow("Look X", min_val=-1.0, max_val=1.0, default=0.0)
        self._eye_x.value_changed.connect(lambda v: self._on_eye_changed("eye_look_x", v))
        self._layout.addWidget(self._eye_x)

        self._eye_y = SliderRow("Look Y", min_val=-1.0, max_val=1.0, default=0.0)
        self._eye_y.value_changed.connect(lambda v: self._on_eye_changed("eye_look_y", v))
        self._layout.addWidget(self._eye_y)

        self._pupil = SliderRow("Pupil", min_val=0.0, max_val=1.0, default=0.5)
        self._pupil.value_changed.connect(lambda v: self._on_eye_changed("pupil_dilation", v))
        self._layout.addWidget(self._pupil)

        # Eye color presets
        self._eye_color_grid = EyeColorGrid()
        self._eye_color_grid.color_selected.connect(self._on_eye_color_selected)
        self._eye_color_grid.set_active("brown")
        self._layout.addWidget(self._eye_color_grid)

        # ── 4. Head ──
        self._layout.addWidget(SectionLabel("Head"))
        self._head_yaw = SliderRow("Yaw", min_val=-1.0, max_val=1.0, default=0.0)
        self._head_yaw.value_changed.connect(lambda v: self._on_head_changed("head_yaw", v))
        self._layout.addWidget(self._head_yaw)

        self._head_pitch = SliderRow("Pitch", min_val=-1.0, max_val=1.0, default=0.0)
        self._head_pitch.value_changed.connect(lambda v: self._on_head_changed("head_pitch", v))
        self._layout.addWidget(self._head_pitch)

        self._head_roll = SliderRow("Roll", min_val=-1.0, max_val=1.0, default=0.0)
        self._head_roll.value_changed.connect(lambda v: self._on_head_changed("head_roll", v))
        self._layout.addWidget(self._head_roll)

        # ── 5. Ear Wiggle ──
        self._layout.addWidget(SectionLabel("Ear Wiggle"))
        self._ear_wiggle = SliderRow("Wiggle", min_val=0.0, max_val=1.0, default=0.0)
        self._ear_wiggle.value_changed.connect(self._on_ear_wiggle_changed)
        self._layout.addWidget(self._ear_wiggle)

        # ── 6. Auto Animation ──
        self._layout.addWidget(SectionLabel("Auto Animation"))

        self._auto_blink = ToggleRow("Auto Blink", default=False)
        self._auto_blink.toggled.connect(
            lambda v: self._bus.publish(EventType.AUTO_BLINK_TOGGLED, enabled=v)
        )
        self._layout.addWidget(self._auto_blink)

        self._auto_breathing = ToggleRow("Auto Breathing", default=False)
        self._auto_breathing.toggled.connect(
            lambda v: self._bus.publish(EventType.AUTO_BREATHING_TOGGLED, enabled=v)
        )
        self._layout.addWidget(self._auto_breathing)

        self._eye_tracking = ToggleRow("Eye Tracking", default=False)
        self._eye_tracking.toggled.connect(
            lambda v: self._bus.publish(EventType.EYE_TRACKING_TOGGLED, enabled=v)
        )
        self._layout.addWidget(self._eye_tracking)

        self._micro_expr = ToggleRow("Micro Expressions", default=False)
        self._micro_expr.toggled.connect(
            lambda v: self._bus.publish(EventType.MICRO_EXPRESSIONS_TOGGLED, enabled=v)
        )
        self._layout.addWidget(self._micro_expr)

        # ── 7. Speech ──
        self._layout.addWidget(SectionLabel("Speech"))
        speech_row = QHBoxLayout()
        speech_row.setSpacing(4)
        self._speech_input = QLineEdit()
        self._speech_input.setPlaceholderText("Type text to speak...")
        self._speech_input.setStyleSheet(
            "QLineEdit { background: rgba(40, 42, 50, 0.9); color: #ccc; "
            "border: 1px solid #555; border-radius: 4px; padding: 4px 8px; }"
        )
        speech_row.addWidget(self._speech_input, stretch=1)
        self._speak_btn = QPushButton("Speak")
        self._speak_btn.setObjectName("speakButton")
        self._speak_btn.clicked.connect(self._on_speak_clicked)
        speech_row.addWidget(self._speak_btn)
        speech_widget = QWidget()
        speech_widget.setLayout(speech_row)
        self._layout.addWidget(speech_widget)

        self._speech_speed = SliderRow("Speed", min_val=0.5, max_val=2.0, default=1.0)
        self._layout.addWidget(self._speech_speed)

        # Push everything up
        self._layout.addStretch()

    # ── Config Loading ──

    @staticmethod
    def _load_expressions() -> dict[str, dict]:
        """Load expression presets from config."""
        try:
            return load_config("expressions.json")
        except Exception:
            return {"neutral": {au: 0.0 for au in AU_IDS}}

    # ── Slots ──

    def _on_expression_selected(self, name: str) -> None:
        preset = self._expressions.get(name, {})
        # Extract AU values (exclude head rotation keys)
        au_vals = {k: v for k, v in preset.items() if k.startswith("AU")}
        # Update slider UI without re-emitting
        for au_id, slider in self._au_sliders.items():
            slider.set_value(au_vals.get(au_id, 0.0))
        # Handle head rotation in preset (e.g. "thinking" has headYaw/headPitch)
        head_yaw = preset.get("headYaw", 0.0)
        head_pitch = preset.get("headPitch", 0.0)
        head_roll = preset.get("headRoll", 0.0)
        self._head_yaw.set_value(head_yaw)
        self._head_pitch.set_value(head_pitch)
        self._head_roll.set_value(head_roll)
        # Publish
        self._bus.publish(EventType.EXPRESSION_SET, name=name, values=au_vals)
        if head_yaw or head_pitch or head_roll:
            self._bus.publish(
                EventType.HEAD_ROTATION_CHANGED,
                head_yaw=head_yaw,
                head_pitch=head_pitch,
                head_roll=head_roll,
            )

    def _on_au_changed(self, au_id: str, value: float) -> None:
        self._bus.publish(EventType.AU_CHANGED, au_id=au_id, value=value)

    def _on_eye_changed(self, field: str, value: float) -> None:
        self._bus.publish(EventType.AU_CHANGED, au_id=field, value=value)

    def _on_head_changed(self, field: str, value: float) -> None:
        kwargs = {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0}
        kwargs[field] = value
        # Read current values from other sliders
        kwargs["head_yaw"] = self._head_yaw.value
        kwargs["head_pitch"] = self._head_pitch.value
        kwargs["head_roll"] = self._head_roll.value
        self._bus.publish(EventType.HEAD_ROTATION_CHANGED, **kwargs)

    def _on_ear_wiggle_changed(self, value: float) -> None:
        self._bus.publish(EventType.AU_CHANGED, au_id="ear_wiggle", value=value)

    def _on_speak_clicked(self) -> None:
        text = self._speech_input.text().strip()
        if text:
            speed = self._speech_speed.value
            self._bus.publish(EventType.SPEECH_PLAY, text=text, speed=speed)

    def _on_eye_color_selected(self, name: str, r: float, g: float, b: float) -> None:
        self._bus.publish(EventType.EYE_COLOR_SET, name=name, color=(r, g, b))

    # ── External Updates ──

    def sync_from_state(self) -> None:
        """Update all widgets to reflect current state (e.g. after undo)."""
        face = self._state.face
        for au_id, slider in self._au_sliders.items():
            slider.set_value(face.get_au(au_id))
        self._eye_x.set_value(face.eye_look_x)
        self._eye_y.set_value(face.eye_look_y)
        self._pupil.set_value(face.pupil_dilation)
        self._head_yaw.set_value(face.head_yaw)
        self._head_pitch.set_value(face.head_pitch)
        self._head_roll.set_value(face.head_roll)
        self._ear_wiggle.set_value(face.ear_wiggle)
        self._expr_grid.set_active(face.current_expression)
        self._eye_color_grid.set_active(face.eye_color)
        self._auto_blink.set_checked(face.auto_blink)
        self._auto_breathing.set_checked(face.auto_breathing)
        self._eye_tracking.set_checked(face.eye_tracking)
        self._micro_expr.set_checked(face.micro_expressions)
