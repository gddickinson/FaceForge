"""Display tab: render mode, camera presets, colors."""

from PySide6.QtWidgets import (
    QScrollArea, QWidget, QVBoxLayout, QGridLayout, QHBoxLayout,
    QPushButton, QSizePolicy, QComboBox, QLabel,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

from faceforge.core.events import EventBus, EventType
from faceforge.core.state import StateManager
from faceforge.core.material import RenderMode
from faceforge.ui.widgets.section_label import SectionLabel
from faceforge.ui.widgets.color_picker import ColorPicker
from faceforge.ui.widgets.toggle_row import ToggleRow
from faceforge.ui.widgets.transport_controls import TransportControls


# Render modes with display labels
_RENDER_MODES = [
    (RenderMode.WIREFRAME, "Wireframe"),
    (RenderMode.SOLID, "Solid"),
    (RenderMode.XRAY, "X-Ray"),
    (RenderMode.POINTS, "Points"),
    (RenderMode.OPAQUE, "Opaque"),
]

# Camera preset definitions: (id, label)
_BODY_CAMERA_PRESETS = [
    ("body_front", "Front"),
    ("body_left", "Left"),
    ("body_right", "Right"),
    ("body_top", "Top"),
    ("body_back", "Back"),
    ("body_three_quarter", "3/4 View"),
]

_HEAD_CAMERA_PRESETS = [
    ("head_front", "Front"),
    ("head_left", "Left"),
    ("head_right", "Right"),
    ("head_top", "Top"),
    ("head_back", "Back"),
    ("head_three_quarter", "3/4 View"),
]


class DisplayTab(QScrollArea):
    """Tab for render mode selection, camera presets, and colour pickers.

    Publishes ``RENDER_MODE_CHANGED``, ``CAMERA_PRESET``, and
    ``COLOR_CHANGED`` events.
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

        # ── 0. Overlay ──
        self._layout.addWidget(SectionLabel("Overlay"))
        self._labels_toggle = ToggleRow("Show Labels", default=False)
        self._labels_toggle.toggled.connect(self._on_labels_toggled)
        self._layout.addWidget(self._labels_toggle)

        # ── 1. Render Mode ──
        self._layout.addWidget(SectionLabel("Render Mode"))
        self._mode_buttons: dict[RenderMode, QPushButton] = {}
        mode_grid = QGridLayout()
        mode_grid.setSpacing(4)
        for i, (mode, label) in enumerate(_RENDER_MODES):
            btn = QPushButton(label)
            btn.setObjectName("renderModeButton")
            btn.setCheckable(True)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.clicked.connect(lambda checked, m=mode: self._on_render_mode(m))
            mode_grid.addWidget(btn, i // 2, i % 2)
            self._mode_buttons[mode] = btn
        # Default to wireframe
        self._mode_buttons[RenderMode.WIREFRAME].setChecked(True)
        mode_widget = QWidget()
        mode_widget.setLayout(mode_grid)
        self._layout.addWidget(mode_widget)

        # ── 2. Camera Presets ──
        self._camera_buttons: dict[str, QPushButton] = {}

        # Body views
        self._layout.addWidget(SectionLabel("Body View"))
        body_cam_grid = QGridLayout()
        body_cam_grid.setSpacing(4)
        for i, (preset_id, label) in enumerate(_BODY_CAMERA_PRESETS):
            btn = QPushButton(label)
            btn.setObjectName("cameraPresetButton")
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.clicked.connect(lambda checked, pid=preset_id: self._on_camera_preset(pid))
            body_cam_grid.addWidget(btn, i // 3, i % 3)
            self._camera_buttons[preset_id] = btn
        body_cam_widget = QWidget()
        body_cam_widget.setLayout(body_cam_grid)
        self._layout.addWidget(body_cam_widget)

        # Head views
        self._layout.addWidget(SectionLabel("Head View"))
        head_cam_grid = QGridLayout()
        head_cam_grid.setSpacing(4)
        for i, (preset_id, label) in enumerate(_HEAD_CAMERA_PRESETS):
            btn = QPushButton(label)
            btn.setObjectName("cameraPresetButton")
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.clicked.connect(lambda checked, pid=preset_id: self._on_camera_preset(pid))
            head_cam_grid.addWidget(btn, i // 3, i % 3)
            self._camera_buttons[preset_id] = btn
        head_cam_widget = QWidget()
        head_cam_widget.setLayout(head_cam_grid)
        self._layout.addWidget(head_cam_widget)

        # ── 3. Colors ──
        self._layout.addWidget(SectionLabel("Colors"))

        self._skull_color = ColorPicker("Skull", QColor(0xD4, 0xA5, 0x74))
        self._skull_color.color_changed.connect(
            lambda c: self._on_color_changed("skull", c)
        )
        self._layout.addWidget(self._skull_color)

        self._face_color = ColorPicker("Face", QColor(0xE0, 0xB8, 0x98))
        self._face_color.color_changed.connect(
            lambda c: self._on_color_changed("face", c)
        )
        self._layout.addWidget(self._face_color)

        self._bg_color = ColorPicker("Background", QColor(0x0A, 0x0B, 0x0E))
        self._bg_color.color_changed.connect(
            lambda c: self._on_color_changed("background", c)
        )
        self._layout.addWidget(self._bg_color)

        # ── 4. Scene View ──
        self._layout.addWidget(SectionLabel("Scene View"))

        self._scene_toggle = QPushButton("Scene View: OFF")
        self._scene_toggle.setObjectName("sceneToggleButton")
        self._scene_toggle.setCheckable(True)
        self._scene_toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._scene_toggle.clicked.connect(self._on_scene_toggled)
        self._layout.addWidget(self._scene_toggle)

        self._scene_camera_combo = QComboBox()
        self._scene_camera_combo.addItems([
            "overhead", "side", "head_end", "foot_end", "corner",
        ])
        self._scene_camera_combo.setEnabled(False)
        self._scene_camera_combo.currentTextChanged.connect(self._on_scene_camera_changed)
        self._layout.addWidget(self._scene_camera_combo)

        # ── 4b. Wrapper Transform Debug ──
        self._layout.addWidget(SectionLabel("Wrapper Nudge"))

        self._nudge_buttons: list[QPushButton] = []
        nudge_step = 10.0
        rot_step = 15.0  # degrees

        # Position: +/- X, Y, Z
        for axis_label, axis_key in [("X", "px"), ("Y", "py"), ("Z", "pz")]:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"Pos {axis_label}:"))
            minus = QPushButton(f"-{int(nudge_step)}")
            minus.setFixedWidth(50)
            minus.clicked.connect(lambda _, a=axis_key: self._on_nudge(a, -nudge_step))
            row.addWidget(minus)
            plus = QPushButton(f"+{int(nudge_step)}")
            plus.setFixedWidth(50)
            plus.clicked.connect(lambda _, a=axis_key: self._on_nudge(a, nudge_step))
            row.addWidget(plus)
            self._nudge_buttons.extend([minus, plus])
            wrapper = QWidget()
            wrapper.setLayout(row)
            self._layout.addWidget(wrapper)

        # Rotation: +/- around X, Y, Z
        for axis_label, axis_key in [("Rot X", "rx"), ("Rot Y", "ry"), ("Rot Z", "rz")]:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{axis_label}:"))
            minus = QPushButton(f"-{int(rot_step)}°")
            minus.setFixedWidth(50)
            minus.clicked.connect(lambda _, a=axis_key: self._on_nudge(a, -rot_step))
            row.addWidget(minus)
            plus = QPushButton(f"+{int(rot_step)}°")
            plus.setFixedWidth(50)
            plus.clicked.connect(lambda _, a=axis_key: self._on_nudge(a, rot_step))
            row.addWidget(plus)
            self._nudge_buttons.extend([minus, plus])
            wrapper = QWidget()
            wrapper.setLayout(row)
            self._layout.addWidget(wrapper)

        # Reset button
        reset_btn = QPushButton("Reset to Supine Default")
        reset_btn.clicked.connect(lambda: self._on_nudge("reset", 0))
        self._nudge_buttons.append(reset_btn)
        self._layout.addWidget(reset_btn)

        # Enable/disable with scene mode
        for btn in self._nudge_buttons:
            btn.setEnabled(False)

        # ── 5. Animation ──
        self._layout.addWidget(SectionLabel("Animation"))

        self._anim_clip_combo = QComboBox()
        self._anim_clip_combo.setEnabled(False)
        self._anim_clip_combo.currentTextChanged.connect(self._on_anim_clip_selected)
        self._layout.addWidget(self._anim_clip_combo)

        self._transport = TransportControls(event_bus)
        self._transport.setEnabled(False)
        self._layout.addWidget(self._transport)

        self._layout.addStretch()

    # ── Slots ──

    def _on_labels_toggled(self, enabled: bool) -> None:
        self._bus.publish(EventType.LABELS_TOGGLED, enabled=enabled)

    def _on_render_mode(self, mode: RenderMode) -> None:
        for m, btn in self._mode_buttons.items():
            btn.setChecked(m == mode)
        self._bus.publish(EventType.RENDER_MODE_CHANGED, mode=mode)

    def _on_camera_preset(self, preset_id: str) -> None:
        self._bus.publish(EventType.CAMERA_PRESET, preset=preset_id)

    def _on_color_changed(self, target: str, color: QColor) -> None:
        rgb = (color.redF(), color.greenF(), color.blueF())
        self._bus.publish(EventType.COLOR_CHANGED, target=target, color=rgb)

    def _on_nudge(self, axis: str, delta: float) -> None:
        self._bus.publish(EventType.SCENE_WRAPPER_NUDGE, axis=axis, delta=delta)

    def _on_scene_toggled(self, checked: bool) -> None:
        self._scene_toggle.setText(f"Scene View: {'ON' if checked else 'OFF'}")
        self._scene_camera_combo.setEnabled(checked)
        self._anim_clip_combo.setEnabled(checked)
        self._transport.setEnabled(checked)
        for btn in self._nudge_buttons:
            btn.setEnabled(checked)
        if not checked:
            # Stop animation when leaving scene mode
            self._bus.publish(EventType.ANIM_STOP)
            self._transport.set_playing(False)
        self._bus.publish(EventType.SCENE_MODE_TOGGLED, enabled=checked)

    def _on_scene_camera_changed(self, preset: str) -> None:
        if self._scene_toggle.isChecked():
            self._bus.publish(EventType.SCENE_CAMERA_CHANGED, preset=preset)

    def _on_anim_clip_selected(self, clip_name: str) -> None:
        if clip_name:
            self._bus.publish(EventType.ANIM_CLIP_SELECTED, clip_name=clip_name)

    # ── Public API ──

    def set_render_mode(self, mode: RenderMode) -> None:
        """Highlight the specified render mode button."""
        for m, btn in self._mode_buttons.items():
            btn.setChecked(m == mode)

    def set_animation_clips(self, clip_names: list[str]) -> None:
        """Populate the clip selection combo with available clips."""
        self._anim_clip_combo.blockSignals(True)
        self._anim_clip_combo.clear()
        self._anim_clip_combo.addItems(clip_names)
        self._anim_clip_combo.blockSignals(False)

    def update_animation_progress(self, progress: float, current_time: float,
                                  duration: float) -> None:
        """Update the transport controls with current playback state."""
        self._transport.set_progress(progress, current_time, duration)

    @property
    def transport(self) -> TransportControls:
        """Access the transport controls widget."""
        return self._transport
