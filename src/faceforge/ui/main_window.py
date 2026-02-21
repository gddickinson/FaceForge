"""Main window: assembles all UI components around GL viewport."""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QStatusBar, QLabel, QSizePolicy, QFileDialog, QMenuBar,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QAction

from faceforge.core.events import EventBus, EventType
from faceforge.core.state import StateManager
from faceforge.rendering.gl_widget import GLViewport
from faceforge.ui.control_panel import ControlPanel
from faceforge.ui.info_panel import InfoPanel
from faceforge.ui.style import DARK_THEME
from faceforge.ui.widgets.loading_overlay import LoadingOverlay


class MainWindow(QMainWindow):
    """Main application window.

    Layout: [InfoPanel | GLViewport | ControlPanel]
    with status bar at bottom showing vertex/face/FPS counts.
    """

    def __init__(
        self,
        event_bus: EventBus,
        state: StateManager,
        gl_widget: GLViewport,
        parent=None,
    ):
        super().__init__(parent)
        self.event_bus = event_bus
        self.state = state
        self.gl_widget = gl_widget

        self.setWindowTitle("FaceForge — Full-Body Anatomical Simulation")
        self.resize(1400, 900)
        self.setStyleSheet(DARK_THEME)

        # Central layout
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left info panel
        self.info_panel = InfoPanel(state)
        self.info_panel.setFixedWidth(220)
        main_layout.addWidget(self.info_panel)

        # GL viewport (stretches to fill)
        gl_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        main_layout.addWidget(gl_widget)

        # Right control panel
        self.control_panel = ControlPanel(event_bus, state)
        main_layout.addWidget(self.control_panel)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        mono = QFont("monospace", 9)

        self.vert_label = QLabel("Verts: 0")
        self.vert_label.setFont(mono)
        self.face_label = QLabel("Faces: 0")
        self.face_label.setFont(mono)
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setFont(mono)

        self.status_bar.addPermanentWidget(self.vert_label)
        self.status_bar.addPermanentWidget(self.face_label)
        self.status_bar.addPermanentWidget(self.fps_label)

        # Menu bar
        self._build_menu_bar()

        # Loading overlay
        self.loading_overlay = LoadingOverlay(gl_widget)
        self.loading_overlay.hide()

        # Subscribe to loading events
        event_bus.subscribe(EventType.LOADING_PHASE, self._on_loading_phase)
        event_bus.subscribe(EventType.LOADING_PROGRESS, self._on_loading_progress)
        event_bus.subscribe(EventType.LOADING_COMPLETE, self._on_loading_complete)

        # Status update timer (2Hz)
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._update_status)
        self._status_timer.start(500)

        # FPS tracking
        self._frame_count_at_last_update = 0

    def _build_menu_bar(self) -> None:
        """Create the application menu bar."""
        menu_bar = self.menuBar()

        # ── File menu ──
        file_menu = menu_bar.addMenu("&File")

        export_glb_action = QAction("Export GLB (Blender)...", self)
        export_glb_action.setShortcut("Ctrl+E")
        export_glb_action.triggered.connect(self._export_glb)
        file_menu.addAction(export_glb_action)

    def _export_glb(self) -> None:
        """Export visible meshes to a GLB file."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export GLB",
            "faceforge_export.glb",
            "GLB Files (*.glb);;All Files (*)",
        )
        if not path:
            return

        from faceforge.export.glb_exporter import export_glb

        scene = self.gl_widget.scene
        if scene is None:
            self.status_bar.showMessage("No scene to export", 3000)
            return

        try:
            count = export_glb(scene, path)
            self.status_bar.showMessage(
                f"Exported {count} meshes to {path}", 5000,
            )
        except Exception as e:
            self.status_bar.showMessage(f"Export failed: {e}", 5000)

    def _on_loading_phase(self, phase: str = "", **kw):
        self.loading_overlay.show_loading(phase, 0.0)

    def _on_loading_progress(self, progress: float = 0.0, **kw):
        self.loading_overlay.show_loading("", progress)

    def _on_loading_complete(self, **kw):
        self.loading_overlay.hide()

    def _update_status(self):
        """Update status bar with scene stats."""
        scene = self.gl_widget.scene
        if scene is None:
            return

        # Count meshes
        total_verts = 0
        total_faces = 0
        meshes = scene.collect_meshes()
        for mesh, _ in meshes:
            total_verts += mesh.geometry.vertex_count
            total_faces += mesh.geometry.triangle_count

        self.vert_label.setText(f"Verts: {total_verts:,}")
        self.face_label.setText(f"Faces: {total_faces:,}")

        # FPS from frame counter
        frames = self.state.frame_count - self._frame_count_at_last_update
        fps = frames * 2  # Timer fires at 2Hz
        self._frame_count_at_last_update = self.state.frame_count
        self.fps_label.setText(f"FPS: {fps}")
        self.state.fps = fps

        # Update info panel
        self.info_panel.update_display()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Keep loading overlay sized to GL viewport
        if hasattr(self, 'loading_overlay'):
            self.loading_overlay.setGeometry(self.gl_widget.geometry())
