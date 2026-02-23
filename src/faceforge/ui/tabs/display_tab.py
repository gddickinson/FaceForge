"""Display tab: render mode, camera presets, colors, labels, clip plane."""

from PySide6.QtWidgets import (
    QScrollArea, QWidget, QVBoxLayout, QGridLayout, QHBoxLayout,
    QPushButton, QSizePolicy, QComboBox, QLabel, QCheckBox, QSlider,
    QSpinBox,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor

from faceforge.core.events import EventBus, EventType
from faceforge.core.state import StateManager
from faceforge.core.material import RenderMode
from faceforge.ui.widgets.section_label import SectionLabel
from faceforge.ui.widgets.color_picker import ColorPicker
from faceforge.ui.widgets.toggle_row import ToggleRow
from faceforge.ui.widgets.transport_controls import TransportControls


# ── Render mode groups ───────────────────────────────────────────────────

_CLINICAL_MODES = [
    (RenderMode.WIREFRAME, "Wireframe"),
    (RenderMode.SOLID, "Solid"),
    (RenderMode.XRAY, "X-Ray"),
    (RenderMode.POINTS, "Points"),
    (RenderMode.OPAQUE, "Opaque"),
]

_ILLUSTRATION_MODES = [
    (RenderMode.ILLUSTRATION, "B&W Textbook"),
    (RenderMode.SEPIA, "Sepia Vintage"),
    (RenderMode.COLOR_ATLAS, "Colour Atlas"),
    (RenderMode.PEN_INK, "Pen & Ink"),
    (RenderMode.MEDICAL, "Medical Atlas"),
]

_CREATIVE_MODES = [
    (RenderMode.HOLOGRAM, "Hologram"),
    (RenderMode.CARTOON, "Cartoon"),
    (RenderMode.PORCELAIN, "Porcelain"),
    (RenderMode.BLUEPRINT, "Blueprint"),
    (RenderMode.THERMAL, "Thermal"),
    (RenderMode.ETHEREAL, "Ethereal"),
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

# Background colour presets: (name, hex)
_BG_PRESETS = [
    ("Dark",      "#0A0B0E"),
    ("Charcoal",  "#1E1E24"),
    ("Slate",     "#2C3040"),
    ("Midnight",  "#0D1B2A"),
    ("Paper",     "#F5F0E8"),
    ("Cream",     "#FFFDD0"),
    ("White",     "#FFFFFF"),
    ("Black",     "#000000"),
    ("Sage",      "#2A3A2A"),
    ("Navy",      "#101828"),
]

# Label font families
_FONT_FAMILIES = [
    "Georgia", "Times New Roman", "Garamond",
    "Arial", "Helvetica", "Verdana",
    "Courier New", "monospace",
]

# Leader line styles
_LINE_STYLES = [
    ("Solid", Qt.PenStyle.SolidLine),
    ("Dashed", Qt.PenStyle.DashLine),
    ("Dotted", Qt.PenStyle.DotLine),
    ("Dash-Dot", Qt.PenStyle.DashDotLine),
]


class DisplayTab(QScrollArea):
    """Tab for render mode selection, camera presets, colours, labels,
    clip plane, and scene controls.

    Publishes ``RENDER_MODE_CHANGED``, ``CAMERA_PRESET``,
    ``COLOR_CHANGED``, ``LABELS_TOGGLED``, ``CLIP_PLANE_CHANGED`` events.
    """

    # Signal emitted when label style settings change (listened to by app.py)
    label_style_changed = Signal()

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

        # ── 0b. Label Style ──
        self._layout.addWidget(SectionLabel("Label Style"))

        # Font family
        font_row = QHBoxLayout()
        font_row.addWidget(QLabel("Font:"))
        self._label_font_combo = QComboBox()
        self._label_font_combo.addItems(_FONT_FAMILIES)
        self._label_font_combo.setCurrentText("Georgia")
        self._label_font_combo.currentTextChanged.connect(lambda _: self.label_style_changed.emit())
        font_row.addWidget(self._label_font_combo)
        font_row_w = QWidget()
        font_row_w.setLayout(font_row)
        self._layout.addWidget(font_row_w)

        # Font size
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Size:"))
        self._label_size_spin = QSpinBox()
        self._label_size_spin.setRange(6, 28)
        self._label_size_spin.setValue(10)
        self._label_size_spin.valueChanged.connect(lambda _: self.label_style_changed.emit())
        size_row.addWidget(self._label_size_spin)

        self._label_italic = QCheckBox("Italic")
        self._label_italic.setChecked(True)
        self._label_italic.toggled.connect(lambda _: self.label_style_changed.emit())
        size_row.addWidget(self._label_italic)

        self._label_bold = QCheckBox("Bold")
        self._label_bold.toggled.connect(lambda _: self.label_style_changed.emit())
        size_row.addWidget(self._label_bold)
        size_row_w = QWidget()
        size_row_w.setLayout(size_row)
        self._layout.addWidget(size_row_w)

        # Line width + dot size
        line_row = QHBoxLayout()
        line_row.addWidget(QLabel("Line:"))
        self._label_line_width = QSlider(Qt.Orientation.Horizontal)
        self._label_line_width.setRange(1, 5)
        self._label_line_width.setValue(1)
        self._label_line_width.setFixedWidth(60)
        self._label_line_width.valueChanged.connect(lambda _: self.label_style_changed.emit())
        line_row.addWidget(self._label_line_width)
        self._label_line_width_lbl = QLabel("1")
        self._label_line_width_lbl.setFixedWidth(12)
        self._label_line_width.valueChanged.connect(
            lambda v: self._label_line_width_lbl.setText(str(v)))
        line_row.addWidget(self._label_line_width_lbl)

        line_row.addWidget(QLabel("Dot:"))
        self._label_dot_size = QSlider(Qt.Orientation.Horizontal)
        self._label_dot_size.setRange(1, 8)
        self._label_dot_size.setValue(3)
        self._label_dot_size.setFixedWidth(60)
        self._label_dot_size.valueChanged.connect(lambda _: self.label_style_changed.emit())
        line_row.addWidget(self._label_dot_size)
        self._label_dot_size_lbl = QLabel("3")
        self._label_dot_size_lbl.setFixedWidth(12)
        self._label_dot_size.valueChanged.connect(
            lambda v: self._label_dot_size_lbl.setText(str(v)))
        line_row.addWidget(self._label_dot_size_lbl)
        line_row_w = QWidget()
        line_row_w.setLayout(line_row)
        self._layout.addWidget(line_row_w)

        # Line style
        style_row = QHBoxLayout()
        style_row.addWidget(QLabel("Style:"))
        self._label_line_style = QComboBox()
        self._label_line_style.addItems([name for name, _ in _LINE_STYLES])
        self._label_line_style.currentIndexChanged.connect(lambda _: self.label_style_changed.emit())
        style_row.addWidget(self._label_line_style)
        style_row_w = QWidget()
        style_row_w.setLayout(style_row)
        self._layout.addWidget(style_row_w)

        # Text / line / dot colour pickers
        self._label_text_color = ColorPicker("Text", QColor(200, 200, 200))
        self._label_text_color.color_changed.connect(lambda _: self.label_style_changed.emit())
        self._layout.addWidget(self._label_text_color)

        self._label_line_color = ColorPicker("Line", QColor(160, 160, 160))
        self._label_line_color.color_changed.connect(lambda _: self.label_style_changed.emit())
        self._layout.addWidget(self._label_line_color)

        self._label_dot_color = ColorPicker("Dot", QColor(180, 60, 60))
        self._label_dot_color.color_changed.connect(lambda _: self.label_style_changed.emit())
        self._layout.addWidget(self._label_dot_color)

        # ── 1. Render Mode: Clinical ──
        self._layout.addWidget(SectionLabel("Clinical Render"))
        self._mode_buttons: dict[RenderMode, QPushButton] = {}
        self._build_mode_grid(_CLINICAL_MODES, cols=3)

        # ── 1b. Render Mode: Illustration ──
        self._layout.addWidget(SectionLabel("Illustration Render"))
        self._build_mode_grid(_ILLUSTRATION_MODES, cols=3)

        # ── 1c. Render Mode: Creative ──
        self._layout.addWidget(SectionLabel("Creative Render"))
        self._build_mode_grid(_CREATIVE_MODES, cols=3)

        # Default to wireframe
        self._mode_buttons[RenderMode.WIREFRAME].setChecked(True)

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

        # Background colour presets
        bg_preset_grid = QGridLayout()
        bg_preset_grid.setSpacing(3)
        for i, (name, hex_str) in enumerate(_BG_PRESETS):
            btn = QPushButton()
            btn.setFixedSize(24, 20)
            btn.setToolTip(name)
            btn.setStyleSheet(
                f"background-color: {hex_str}; border: 1px solid #404040; border-radius: 3px;"
            )
            btn.clicked.connect(lambda _, h=hex_str, n=name: self._on_bg_preset(h))
            bg_preset_grid.addWidget(btn, i // 5, i % 5)
        bg_preset_widget = QWidget()
        bg_preset_widget.setLayout(bg_preset_grid)
        self._layout.addWidget(bg_preset_widget)

        # ── 4. Clip Plane ──
        self._layout.addWidget(SectionLabel("Clip Plane"))

        clip_row = QHBoxLayout()
        self._clip_enable = QCheckBox("Enable")
        self._clip_enable.toggled.connect(self._on_clip_changed)
        clip_row.addWidget(self._clip_enable)

        self._clip_axis = QComboBox()
        self._clip_axis.addItems(["Sagittal (X)", "Coronal (Y)", "Axial (Z)"])
        self._clip_axis.currentIndexChanged.connect(lambda _: self._on_clip_changed())
        clip_row.addWidget(self._clip_axis)

        self._clip_flip = QCheckBox("Flip")
        self._clip_flip.toggled.connect(self._on_clip_changed)
        clip_row.addWidget(self._clip_flip)

        clip_row_widget = QWidget()
        clip_row_widget.setLayout(clip_row)
        self._layout.addWidget(clip_row_widget)

        offset_row = QHBoxLayout()
        offset_row.addWidget(QLabel("Offset:"))
        self._clip_offset = QSlider(Qt.Orientation.Horizontal)
        self._clip_offset.setRange(-200, 200)
        self._clip_offset.setValue(0)
        self._clip_offset.valueChanged.connect(lambda _: self._on_clip_changed())
        offset_row.addWidget(self._clip_offset)
        self._clip_offset_label = QLabel("0")
        self._clip_offset_label.setFixedWidth(30)
        offset_row.addWidget(self._clip_offset_label)
        offset_row_widget = QWidget()
        offset_row_widget.setLayout(offset_row)
        self._layout.addWidget(offset_row_widget)

        # ── 5. Scene View ──
        self._layout.addWidget(SectionLabel("Scene View"))

        # Scene type selector
        scene_type_row = QHBoxLayout()
        scene_type_row.addWidget(QLabel("Scene:"))
        self._scene_type_combo = QComboBox()
        self._scene_type_combo.addItems(["Examination Room", "Dance Studio"])
        self._scene_type_combo.setEnabled(True)
        scene_type_row.addWidget(self._scene_type_combo)
        scene_type_widget = QWidget()
        scene_type_widget.setLayout(scene_type_row)
        self._layout.addWidget(scene_type_widget)

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

        # ── 5b. Wrapper Transform Debug ──
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
            minus = QPushButton(f"-{int(rot_step)}\u00B0")
            minus.setFixedWidth(50)
            minus.clicked.connect(lambda _, a=axis_key: self._on_nudge(a, -rot_step))
            row.addWidget(minus)
            plus = QPushButton(f"+{int(rot_step)}\u00B0")
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

        # ── 6. Tools ──
        self._layout.addWidget(SectionLabel("Tools"))

        tools_grid = QGridLayout()
        tools_grid.setSpacing(4)

        self._export_btn = QPushButton("Export Video")
        self._export_btn.setObjectName("toolButton")
        self._export_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        tools_grid.addWidget(self._export_btn, 0, 0)

        self._quiz_btn = QPushButton("Anatomy Quiz")
        self._quiz_btn.setObjectName("toolButton")
        self._quiz_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        tools_grid.addWidget(self._quiz_btn, 0, 1)

        self._timeline_btn = QPushButton("Edit Timeline")
        self._timeline_btn.setObjectName("toolButton")
        self._timeline_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        tools_grid.addWidget(self._timeline_btn, 1, 0)

        self._compare_btn = QPushButton("Compare")
        self._compare_btn.setObjectName("toolButton")
        self._compare_btn.setCheckable(True)
        self._compare_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        tools_grid.addWidget(self._compare_btn, 1, 1)

        tools_widget = QWidget()
        tools_widget.setLayout(tools_grid)
        self._layout.addWidget(tools_widget)

        # ── 7. Animation ──
        self._layout.addWidget(SectionLabel("Animation"))

        self._anim_clip_combo = QComboBox()
        self._anim_clip_combo.setEnabled(False)
        self._anim_clip_combo.currentTextChanged.connect(self._on_anim_clip_selected)
        self._layout.addWidget(self._anim_clip_combo)

        self._transport = TransportControls(event_bus)
        self._transport.setEnabled(False)
        self._layout.addWidget(self._transport)

        self._layout.addStretch()

    # ── Helpers ──

    def _build_mode_grid(self, modes: list, cols: int = 3) -> None:
        """Build a grid of render mode buttons and add to layout."""
        grid = QGridLayout()
        grid.setSpacing(4)
        for i, (mode, label) in enumerate(modes):
            btn = QPushButton(label)
            btn.setObjectName("renderModeButton")
            btn.setCheckable(True)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.clicked.connect(lambda checked, m=mode: self._on_render_mode(m))
            grid.addWidget(btn, i // cols, i % cols)
            self._mode_buttons[mode] = btn
        widget = QWidget()
        widget.setLayout(grid)
        self._layout.addWidget(widget)

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

    def _on_bg_preset(self, hex_str: str) -> None:
        color = QColor(hex_str)
        self._bg_color.set_color(color)
        self._on_color_changed("background", color)

    def _on_clip_changed(self, _=None) -> None:
        enabled = self._clip_enable.isChecked()
        axis_map = {0: "x", 1: "y", 2: "z"}
        axis = axis_map.get(self._clip_axis.currentIndex(), "x")
        offset = self._clip_offset.value()
        flip = self._clip_flip.isChecked()
        self._clip_offset_label.setText(str(offset))
        self._bus.publish(
            EventType.CLIP_PLANE_CHANGED,
            enabled=enabled, axis=axis, offset=float(offset), flip=flip,
        )

    def set_clip_plane(self, enabled: bool, axis: str = "x",
                       offset: float = 0.0, flip: bool = False) -> None:
        """Programmatically set clip plane state (used by illustration presets)."""
        self._clip_enable.blockSignals(True)
        self._clip_axis.blockSignals(True)
        self._clip_offset.blockSignals(True)
        self._clip_flip.blockSignals(True)

        self._clip_enable.setChecked(enabled)
        axis_idx = {"x": 0, "y": 1, "z": 2}.get(axis, 0)
        self._clip_axis.setCurrentIndex(axis_idx)
        self._clip_offset.setValue(int(offset))
        self._clip_flip.setChecked(flip)
        self._clip_offset_label.setText(str(int(offset)))

        self._clip_enable.blockSignals(False)
        self._clip_axis.blockSignals(False)
        self._clip_offset.blockSignals(False)
        self._clip_flip.blockSignals(False)

        # Fire the event
        self._bus.publish(
            EventType.CLIP_PLANE_CHANGED,
            enabled=enabled, axis=axis, offset=float(offset), flip=flip,
        )

    def _on_nudge(self, axis: str, delta: float) -> None:
        self._bus.publish(EventType.SCENE_WRAPPER_NUDGE, axis=axis, delta=delta)

    def _on_scene_toggled(self, checked: bool) -> None:
        self._scene_toggle.setText(f"Scene View: {'ON' if checked else 'OFF'}")
        self._scene_camera_combo.setEnabled(checked)
        self._anim_clip_combo.setEnabled(checked)
        self._transport.setEnabled(checked)
        self._scene_type_combo.setEnabled(not checked)  # lock while active
        for btn in self._nudge_buttons:
            btn.setEnabled(checked)
        if not checked:
            # Stop animation when leaving scene mode
            self._bus.publish(EventType.ANIM_STOP)
            self._transport.set_playing(False)
        # Determine scene type from combo
        scene_type = "dance_studio" if self._scene_type_combo.currentIndex() == 1 else "examination"
        self._bus.publish(EventType.SCENE_MODE_TOGGLED, enabled=checked, scene_type=scene_type)

        # Update camera presets for scene type
        if checked:
            self._update_camera_presets(scene_type)

    def _update_camera_presets(self, scene_type: str) -> None:
        """Update camera combo items for the given scene type."""
        self._scene_camera_combo.blockSignals(True)
        self._scene_camera_combo.clear()
        if scene_type == "dance_studio":
            self._scene_camera_combo.addItems([
                "front", "front_wide", "side_left", "side_right",
                "overhead", "corner", "low_front",
            ])
        else:
            self._scene_camera_combo.addItems([
                "overhead", "side", "head_end", "foot_end", "corner",
            ])
        self._scene_camera_combo.blockSignals(False)

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

    # ── Label style getters ──

    def get_label_style(self) -> dict:
        """Return current label style settings as a dict."""
        style_idx = self._label_line_style.currentIndex()
        _, qt_style = _LINE_STYLES[style_idx] if style_idx < len(_LINE_STYLES) else _LINE_STYLES[0]
        return {
            "font_family": self._label_font_combo.currentText(),
            "font_size": self._label_size_spin.value(),
            "italic": self._label_italic.isChecked(),
            "bold": self._label_bold.isChecked(),
            "line_width": self._label_line_width.value(),
            "line_style": qt_style,
            "dot_size": self._label_dot_size.value(),
            "text_color": self._label_text_color.color,
            "line_color": self._label_line_color.color,
            "dot_color": self._label_dot_color.color,
        }
