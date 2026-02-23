"""Export dialog for video/GIF/screenshot settings."""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QLabel, QFileDialog, QProgressBar, QGroupBox,
)
from PySide6.QtCore import Qt, Signal


class ExportDialog(QDialog):
    """Modal dialog for configuring video/GIF/screenshot export.

    Emits ``export_requested`` with the chosen settings.
    """

    export_requested = Signal(dict)

    def __init__(self, parent=None, ffmpeg_available: bool = False):
        super().__init__(parent)
        self.setWindowTitle("Export Video / Screenshot")
        self.setMinimumWidth(350)
        self._ffmpeg_available = ffmpeg_available
        self._output_path = ""

        layout = QVBoxLayout(self)

        # Export mode
        mode_group = QGroupBox("Export Mode")
        mode_layout = QFormLayout(mode_group)

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Turntable", "Animation", "Screenshot"])
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addRow("Mode:", self._mode_combo)

        # Format
        self._format_combo = QComboBox()
        formats = []
        if ffmpeg_available:
            formats.append("MP4 (.mp4)")
        formats.extend(["GIF (.gif)", "PNG Sequence"])
        self._format_combo.addItems(formats)
        mode_layout.addRow("Format:", self._format_combo)

        layout.addWidget(mode_group)

        # Settings
        settings_group = QGroupBox("Settings")
        settings_layout = QFormLayout(settings_group)

        self._fps_spin = QSpinBox()
        self._fps_spin.setRange(1, 60)
        self._fps_spin.setValue(30)
        settings_layout.addRow("FPS:", self._fps_spin)

        self._duration_spin = QDoubleSpinBox()
        self._duration_spin.setRange(1.0, 120.0)
        self._duration_spin.setValue(10.0)
        self._duration_spin.setSuffix(" s")
        settings_layout.addRow("Duration:", self._duration_spin)

        self._width_spin = QSpinBox()
        self._width_spin.setRange(0, 3840)
        self._width_spin.setValue(0)
        self._width_spin.setSpecialValueText("Auto")
        settings_layout.addRow("Width:", self._width_spin)

        self._height_spin = QSpinBox()
        self._height_spin.setRange(0, 2160)
        self._height_spin.setValue(0)
        self._height_spin.setSpecialValueText("Auto")
        settings_layout.addRow("Height:", self._height_spin)

        layout.addWidget(settings_group)

        # Output path
        path_layout = QHBoxLayout()
        self._path_label = QLabel("No file selected")
        self._path_label.setStyleSheet("color: #888;")
        path_layout.addWidget(self._path_label)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._on_browse)
        path_layout.addWidget(browse_btn)
        layout.addLayout(path_layout)

        # Progress
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        # Buttons
        btn_layout = QHBoxLayout()
        self._export_btn = QPushButton("Export")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._on_export)
        btn_layout.addWidget(self._export_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        self._on_mode_changed(0)

    def _on_mode_changed(self, index: int) -> None:
        is_screenshot = index == 2
        self._fps_spin.setEnabled(not is_screenshot)
        self._duration_spin.setEnabled(index == 0)  # Turntable only

    def _on_browse(self) -> None:
        mode_idx = self._mode_combo.currentIndex()
        fmt_text = self._format_combo.currentText()

        if mode_idx == 2:  # Screenshot
            filters = "Images (*.png *.jpg *.bmp)"
        elif "MP4" in fmt_text:
            filters = "Video (*.mp4)"
        elif "GIF" in fmt_text:
            filters = "GIF (*.gif)"
        else:
            filters = "All (*)"

        path, _ = QFileDialog.getSaveFileName(self, "Export To", "", filters)
        if path:
            self._output_path = path
            self._path_label.setText(path)
            self._path_label.setStyleSheet("color: #ccc;")
            self._export_btn.setEnabled(True)

    def _on_export(self) -> None:
        settings = {
            "mode": self._mode_combo.currentText().lower(),
            "format": self._format_combo.currentText(),
            "fps": self._fps_spin.value(),
            "duration": self._duration_spin.value(),
            "width": self._width_spin.value(),
            "height": self._height_spin.value(),
            "output_path": self._output_path,
        }
        self._progress.setVisible(True)
        self.export_requested.emit(settings)

    def set_progress(self, value: float) -> None:
        """Update export progress (0.0 to 1.0)."""
        self._progress.setValue(int(value * 100))
        if value >= 1.0:
            self._progress.setVisible(False)
            self.accept()

    def get_settings(self) -> dict:
        """Return current export settings."""
        return {
            "mode": self._mode_combo.currentText().lower(),
            "format": self._format_combo.currentText(),
            "fps": self._fps_spin.value(),
            "duration": self._duration_spin.value(),
            "width": self._width_spin.value(),
            "height": self._height_spin.value(),
            "output_path": self._output_path,
        }
