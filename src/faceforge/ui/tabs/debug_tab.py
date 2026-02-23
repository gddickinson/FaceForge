"""Debug tab: debug visualization toggles, stats, vertex selection, overrides."""

from PySide6.QtWidgets import (
    QScrollArea, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTextEdit, QComboBox,
)
from PySide6.QtCore import Qt, Signal

from faceforge.core.events import EventBus, EventType
from faceforge.core.state import StateManager
from faceforge.ui.widgets.section_label import SectionLabel
from faceforge.ui.widgets.toggle_row import ToggleRow


class DebugTab(QScrollArea):
    """Tab with debug overlay toggles, live rendering statistics,
    stretch/chain visualization, vertex selection, and override management.

    Toggles publish ``LAYER_TOGGLED`` events with ``debug_`` prefixed layer IDs.
    """

    # Signals for the new features
    stretch_viz_toggled = Signal(bool)
    chain_viz_toggled = Signal(bool)
    region_viz_toggled = Signal(bool)
    selection_mode_toggled = Signal(bool)
    reassign_clicked = Signal(str)  # target chain name
    region_reassign_clicked = Signal(str)  # target region name
    clear_selection_clicked = Signal()
    undo_clicked = Signal()
    save_overrides_clicked = Signal()
    load_overrides_clicked = Signal()

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

        # ── Visualization ──
        self._layout.addWidget(SectionLabel("Visualization"))

        self._stretch_toggle = ToggleRow("Stretch Heatmap", default=False)
        self._stretch_toggle.toggled.connect(self._on_stretch_toggled)
        self._layout.addWidget(self._stretch_toggle)

        self._chain_toggle = ToggleRow("Chain Assignment", default=False)
        self._chain_toggle.toggled.connect(self._on_chain_toggled)
        self._layout.addWidget(self._chain_toggle)

        self._region_toggle = ToggleRow("Region Labels", default=False)
        self._region_toggle.toggled.connect(self._on_region_toggled)
        self._layout.addWidget(self._region_toggle)

        # ── Vertex Selection ──
        self._layout.addWidget(SectionLabel("Vertex Selection"))

        self._select_toggle = ToggleRow("Selection Mode", default=False)
        self._select_toggle.toggled.connect(self._on_selection_toggled)
        self._layout.addWidget(self._select_toggle)

        self._selected_label = QLabel("Selected: 0 vertices")
        self._selected_label.setObjectName("statsLabel")
        self._layout.addWidget(self._selected_label)

        # Chain selector + reassign button
        chain_row = QWidget()
        chain_layout = QHBoxLayout(chain_row)
        chain_layout.setContentsMargins(0, 0, 0, 0)
        chain_layout.setSpacing(4)

        self._chain_combo = QComboBox()
        self._chain_combo.setMinimumWidth(100)
        chain_layout.addWidget(self._chain_combo, 1)

        self._reassign_btn = QPushButton("Reassign")
        self._reassign_btn.clicked.connect(self._on_reassign)
        self._reassign_btn.setEnabled(False)
        chain_layout.addWidget(self._reassign_btn)
        self._layout.addWidget(chain_row)

        # Region selector + set region button
        region_row = QWidget()
        region_layout = QHBoxLayout(region_row)
        region_layout.setContentsMargins(0, 0, 0, 0)
        region_layout.setSpacing(4)

        self._region_combo = QComboBox()
        self._region_combo.setMinimumWidth(100)
        region_layout.addWidget(self._region_combo, 1)

        self._region_reassign_btn = QPushButton("Set Region")
        self._region_reassign_btn.clicked.connect(self._on_region_reassign)
        self._region_reassign_btn.setEnabled(False)
        region_layout.addWidget(self._region_reassign_btn)
        self._layout.addWidget(region_row)

        # Clear and Undo buttons
        btn_row = QWidget()
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(4)

        self._clear_btn = QPushButton("Clear Selection")
        self._clear_btn.clicked.connect(self._on_clear_selection)
        btn_layout.addWidget(self._clear_btn)

        self._undo_btn = QPushButton("Undo")
        self._undo_btn.clicked.connect(self._on_undo)
        self._undo_btn.setEnabled(False)
        btn_layout.addWidget(self._undo_btn)
        self._layout.addWidget(btn_row)

        # ── Overrides ──
        self._layout.addWidget(SectionLabel("Overrides"))

        override_row = QWidget()
        override_layout = QHBoxLayout(override_row)
        override_layout.setContentsMargins(0, 0, 0, 0)
        override_layout.setSpacing(4)

        self._save_btn = QPushButton("Save Overrides")
        self._save_btn.clicked.connect(self._on_save_overrides)
        override_layout.addWidget(self._save_btn)

        self._load_btn = QPushButton("Load Overrides")
        self._load_btn.clicked.connect(self._on_load_overrides)
        override_layout.addWidget(self._load_btn)
        self._layout.addWidget(override_row)

        self._override_label = QLabel("0 overrides active")
        self._override_label.setObjectName("statsLabel")
        self._layout.addWidget(self._override_label)

        self._region_override_label = QLabel("0 region overrides")
        self._region_override_label.setObjectName("statsLabel")
        self._layout.addWidget(self._region_override_label)

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

    def _on_stretch_toggled(self, checked: bool) -> None:
        # Mutually exclusive with chain/region viz
        print(f"[DEBUG] _on_stretch_toggled({checked})")
        if checked:
            if self._chain_toggle.is_checked:
                self._chain_toggle.set_checked(False)
            if self._region_toggle.is_checked:
                self._region_toggle.set_checked(False)
        print(f"[DEBUG] emitting stretch_viz_toggled({checked})")
        self.stretch_viz_toggled.emit(checked)

    def _on_chain_toggled(self, checked: bool) -> None:
        # Mutually exclusive with stretch/region viz
        print(f"[DEBUG] _on_chain_toggled({checked})")
        if checked:
            if self._stretch_toggle.is_checked:
                self._stretch_toggle.set_checked(False)
            if self._region_toggle.is_checked:
                self._region_toggle.set_checked(False)
        print(f"[DEBUG] emitting chain_viz_toggled({checked})")
        self.chain_viz_toggled.emit(checked)

    def _on_region_toggled(self, checked: bool) -> None:
        # Mutually exclusive with stretch/chain viz
        print(f"[DEBUG] _on_region_toggled({checked})")
        if checked:
            if self._stretch_toggle.is_checked:
                self._stretch_toggle.set_checked(False)
            if self._chain_toggle.is_checked:
                self._chain_toggle.set_checked(False)
        print(f"[DEBUG] emitting region_viz_toggled({checked})")
        self.region_viz_toggled.emit(checked)

    def _on_selection_toggled(self, checked: bool) -> None:
        self.selection_mode_toggled.emit(checked)

    def _on_reassign(self) -> None:
        chain_name = self._chain_combo.currentText()
        if chain_name:
            self.reassign_clicked.emit(chain_name)

    def _on_region_reassign(self) -> None:
        region_name = self._region_combo.currentText()
        if region_name:
            self.region_reassign_clicked.emit(region_name)

    def _on_clear_selection(self) -> None:
        self.clear_selection_clicked.emit()

    def _on_undo(self) -> None:
        self.undo_clicked.emit()

    def _on_save_overrides(self) -> None:
        self.save_overrides_clicked.emit()

    def _on_load_overrides(self) -> None:
        self.load_overrides_clicked.emit()

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

    def set_chain_names(self, names: list[str]) -> None:
        """Populate the chain selector combo box."""
        self._chain_combo.clear()
        self._chain_combo.addItems(names)

    def set_region_names(self, names: list[str]) -> None:
        """Populate the region selector combo box."""
        self._region_combo.clear()
        self._region_combo.addItems(names)

    def update_selection_count(self, count: int) -> None:
        """Update the selected vertex count label."""
        self._selected_label.setText(f"Selected: {count} vertices")
        self._reassign_btn.setEnabled(count > 0)
        self._region_reassign_btn.setEnabled(count > 0)

    def set_undo_enabled(self, enabled: bool) -> None:
        """Enable/disable the undo button."""
        self._undo_btn.setEnabled(enabled)

    def set_override_count(self, count: int) -> None:
        """Update the override count label."""
        self._override_label.setText(f"{count} overrides active")

    def update_region_override_count(self, count: int) -> None:
        """Update the region override count label."""
        self._region_override_label.setText(f"{count} region overrides")

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
