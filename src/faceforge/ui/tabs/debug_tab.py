"""Debug tab: debug visualization toggles, stats."""

from PySide6.QtWidgets import QScrollArea, QWidget, QVBoxLayout, QLabel, QPushButton, QTextEdit
from PySide6.QtCore import Qt, Signal

from faceforge.core.events import EventBus, EventType
from faceforge.core.state import StateManager
from faceforge.ui.widgets.section_label import SectionLabel
from faceforge.ui.widgets.toggle_row import ToggleRow


class DebugTab(QScrollArea):
    """Tab with debug overlay toggles and live rendering statistics.

    Toggles publish ``LAYER_TOGGLED`` events with ``debug_`` prefixed layer IDs.
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

        # ── Debug Overlays ──
        self._layout.addWidget(SectionLabel("Debug Overlays"))

        self._toggles: dict[str, ToggleRow] = {}

        debug_layers = [
            ("debug_bone_markers", "Bone Markers", False),
            ("debug_attachment_points", "Attachment Points", False),
            ("debug_tension_heatmap", "Tension Heatmap", False),
            ("debug_body_follow", "Body Follow Vectors", False),
        ]

        for layer_id, label, default in debug_layers:
            toggle = ToggleRow(label, default=default)
            toggle.toggled.connect(
                lambda checked, lid=layer_id: self._on_debug_toggled(lid, checked)
            )
            self._toggles[layer_id] = toggle
            self._layout.addWidget(toggle)

        # ── Stats ──
        self._layout.addWidget(SectionLabel("Statistics"))

        self._vertex_label = QLabel("Vertices: 0")
        self._vertex_label.setObjectName("statsLabel")
        self._layout.addWidget(self._vertex_label)

        self._face_label = QLabel("Faces: 0")
        self._face_label.setObjectName("statsLabel")
        self._layout.addWidget(self._face_label)

        self._fps_label = QLabel("FPS: 0")
        self._fps_label.setObjectName("statsLabel")
        self._layout.addWidget(self._fps_label)

        self._draw_calls_label = QLabel("Draw calls: 0")
        self._draw_calls_label.setObjectName("statsLabel")
        self._layout.addWidget(self._draw_calls_label)

        # ── Skinning Diagnostics ──
        self._layout.addWidget(SectionLabel("Skinning Diagnostics"))

        self._diag_btn = QPushButton("Run Diagnostic")
        self._diag_btn.clicked.connect(self._on_run_diagnostic)
        self._layout.addWidget(self._diag_btn)

        self._diag_output = QTextEdit()
        self._diag_output.setReadOnly(True)
        self._diag_output.setMaximumHeight(200)
        self._diag_output.setObjectName("diagOutput")
        self._diag_output.setPlaceholderText("Click 'Run Diagnostic' to analyze skinning bindings...")
        self._layout.addWidget(self._diag_output)

        self._layout.addStretch()

        # Callback set by app.py to run diagnostics
        self._diag_callback = None

    # ── Slots ──

    def _on_debug_toggled(self, layer_id: str, visible: bool) -> None:
        self._bus.publish(EventType.LAYER_TOGGLED, layer=layer_id, visible=visible)

    def _on_run_diagnostic(self) -> None:
        if self._diag_callback is not None:
            result = self._diag_callback()
            self._diag_output.setPlainText(result)
        else:
            self._diag_output.setPlainText("Diagnostic not available (soft tissue not loaded)")

    def set_diagnostic_callback(self, callback) -> None:
        """Set the callback that returns a diagnostic report string."""
        self._diag_callback = callback

    # ── Public API ──

    def update_stats(
        self,
        vertices: int = 0,
        faces: int = 0,
        fps: float = 0.0,
        draw_calls: int = 0,
    ) -> None:
        """Update the statistics display."""
        self._vertex_label.setText(f"Vertices: {vertices:,}")
        self._face_label.setText(f"Faces: {faces:,}")
        self._fps_label.setText(f"FPS: {fps:.1f}")
        self._draw_calls_label.setText(f"Draw calls: {draw_calls}")
