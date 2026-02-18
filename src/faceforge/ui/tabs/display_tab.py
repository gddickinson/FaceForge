"""Display tab: render mode, camera presets, colors."""

from PySide6.QtWidgets import (
    QScrollArea, QWidget, QVBoxLayout, QGridLayout, QPushButton, QSizePolicy,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

from faceforge.core.events import EventBus, EventType
from faceforge.core.state import StateManager
from faceforge.core.material import RenderMode
from faceforge.ui.widgets.section_label import SectionLabel
from faceforge.ui.widgets.color_picker import ColorPicker
from faceforge.ui.widgets.toggle_row import ToggleRow


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

    # ── Public API ──

    def set_render_mode(self, mode: RenderMode) -> None:
        """Highlight the specified render mode button."""
        for m, btn in self._mode_buttons.items():
            btn.setChecked(m == mode)
