"""The auxiliary windows: scanner, export, quiz, timeline, comparison.

Each window is built on first request and then kept, so reopening one returns
to the state the user left it in and does not repay construction cost.  The
dialog classes are imported inside the open methods rather than at module
level: several pull in heavy submodules, and a session that never opens the
scanner should not pay to import it.

The scanner is the exception to "keep it": its window is rebuilt if it has been
closed, because closing it hides the 3D scan-plane visualisation and a rebuild
is what re-establishes the plane.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Buttons on the display tab, and the method each opens.
TOOL_BUTTONS = {
    "_export_btn": "open_export",
    "_quiz_btn": "open_quiz",
    "_timeline_btn": "open_timeline",
}

#: Opacity a mesh not being quizzed is dimmed to.
QUIZ_DIMMED_OPACITY = 0.2


class ToolsController:
    """Owns the auxiliary windows and the buttons that open them."""

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        self.scanner_window: Any = None
        self.export_dialog: Any = None
        self.quiz_dialog: Any = None
        self.timeline_editor: Any = None
        self.comparison_dialog: Any = None

    def connect(self) -> None:
        """Wire the window's tool buttons and the scanner menu action."""
        self.ctx.window.scanner_requested.connect(self.open_scanner)

    def connect_display_buttons(self) -> None:
        """Connect the display tab's tool buttons, where they exist.

        Guarded with ``hasattr`` because the display tab is built from a
        declarative spec: a button that has not been added yet is a missing
        feature, not a wiring error.
        """
        display_tab = self.ctx.control_panel.display_tab
        for attr, method in TOOL_BUTTONS.items():
            if hasattr(display_tab, attr):
                getattr(display_tab, attr).clicked.connect(getattr(self, method))
        if hasattr(display_tab, "_compare_btn"):
            display_tab._compare_btn.toggled.connect(self.on_compare_toggled)

    # -- Virtual scanner ---------------------------------------------------

    def open_scanner(self) -> None:
        """Open the virtual scanner, or raise it if already open."""
        from faceforge.scanner.scanner_window import ScannerWindow

        if self.scanner_window is not None and self.scanner_window.isVisible():
            self.scanner_window.raise_()
            self.scanner_window.activateWindow()
            return

        window = ScannerWindow(self.ctx.window, self.ctx.scanner_engine)
        self.scanner_window = window

        window.scan_requested.connect(self._on_scan)
        window.plane_changed.connect(self.update_scan_plane)
        window.closed.connect(
            lambda: self.ctx.scan_plane_viz.set_visible(False))
        window.show()

        # Show the plane immediately, with the window's initial parameters.
        self.ctx.scan_plane_viz.set_visible(True)
        self.update_scan_plane(window.plane_params)

    def _on_scan(self) -> None:
        """Run a scan of the current scene.

        The 3D plane is updated first so the user sees where the slice is being
        taken from while the scan runs; the scan itself is synchronous and
        pumps events from inside the scanner window.
        """
        self.update_scan_plane(self.scanner_window.plane_params)
        self.scanner_window.run_scan(self.ctx.scene.collect_meshes())

    def update_scan_plane(self, params: dict) -> None:
        self.ctx.scan_plane_viz.update(
            origin=params["origin"], normal=params["normal"],
            right=params["right"], up=params["up"],
            width=params["width"], height=params["height"],
        )

    # -- Export ------------------------------------------------------------

    def open_export(self) -> None:
        from faceforge.ui.export_dialog import ExportDialog

        if self.export_dialog is None:
            self.export_dialog = ExportDialog(self.ctx.window)
            self.export_dialog.export_requested.connect(self.run_export)
        self.export_dialog.show()

    def run_export(self, config: dict) -> None:
        """Dispatch one export request to the video exporter.

        Width and height default to the live viewport size, so an export with
        no size chosen matches what the user is looking at.
        """
        gl = self.ctx.gl_widget
        mode = config.get("mode", "turntable")
        output = config.get("output_path", "export.mp4")
        fps = config.get("fps", 30)
        width = config.get("width", gl.width())
        height = config.get("height", gl.height())

        if mode == "screenshot":
            self.ctx.video_exporter.export_screenshot(
                output, width=width, height=height)
        elif mode == "turntable":
            self.ctx.video_exporter.export_turntable(
                output, duration=config.get("duration", 10.0), fps=fps,
                width=width, height=height)
        elif mode == "animation":
            self.ctx.video_exporter.export_animation(
                self.ctx.anim_player, self.ctx.simulation, output,
                fps=fps, width=width, height=height)

    # -- Quiz --------------------------------------------------------------

    def open_quiz(self) -> None:
        from faceforge.ui.quiz_dialog import QuizDialog

        if self.quiz_dialog is None:
            self.quiz_dialog = QuizDialog(self.ctx.quiz_engine, self.ctx.window)
            self.quiz_dialog.highlight_requested.connect(self.highlight_mesh)
            self.quiz_dialog.clear_highlight.connect(self.clear_highlight)
            self.quiz_dialog.quiz_click_mode.connect(self.set_quiz_click_mode)
        self.quiz_dialog.show()

    def highlight_mesh(self, mesh_name: str) -> None:
        for mesh, _ in self.ctx.scene.collect_meshes():
            mesh.material.opacity = (
                1.0 if mesh.name == mesh_name else QUIZ_DIMMED_OPACITY)

    def clear_highlight(self) -> None:
        for mesh, _ in self.ctx.scene.collect_meshes():
            mesh.material.opacity = 1.0

    def set_quiz_click_mode(self, enabled: bool) -> None:
        """Route viewport clicks to the quiz dialog while identify mode is on."""
        gl = self.ctx.gl_widget
        gl.quiz_click_mode = enabled
        gl.quiz_click_callback = self._on_mesh_clicked if enabled else None

    def _on_mesh_clicked(self, name: str) -> None:
        if self.quiz_dialog is not None:
            self.quiz_dialog.on_mesh_clicked(name)

    # -- Timeline editor ---------------------------------------------------

    def open_timeline(self) -> None:
        from faceforge.ui.timeline_editor import TimelineEditor

        if self.timeline_editor is None:
            self.timeline_editor = TimelineEditor(self.ctx.window)
            self.timeline_editor.set_animation_player(self.ctx.anim_player)
            self.timeline_editor.set_state_refs(self.ctx.state, self.ctx.camera)
        self.timeline_editor.show()

    # -- Comparison view ---------------------------------------------------

    def on_compare_toggled(self, checked: bool) -> None:
        """Enter or leave side-by-side comparison rendering."""
        gl = self.ctx.gl_widget
        if not checked:
            gl.comparison_mode = False
            if self.comparison_dialog is not None:
                self.comparison_dialog.hide()
            return

        from faceforge.ui.comparison_dialog import ComparisonDialog

        if self.comparison_dialog is None:
            self.comparison_dialog = ComparisonDialog(self.ctx.window)
            self.comparison_dialog.config_changed.connect(
                self.set_comparison_configs)
        gl.comparison_mode = True
        self.comparison_dialog.show()

    def set_comparison_configs(self, left_cfg: Any, right_cfg: Any) -> None:
        self.ctx.gl_widget.comparison_left_config = left_cfg
        self.ctx.gl_widget.comparison_right_config = right_cfg
