"""Scanner window: QDialog popup with cross-section image display and controls.

Provides controls for scan mode, position, orientation, dimensions,
reduction, and resolution. Displays the resulting image with
mouse-position coordinate readout.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QPushButton, QSizePolicy, QScrollArea, QFrame,
    QApplication,
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QImage, QPixmap

from faceforge.scanner.engine import ScannerEngine
from faceforge.scanner.tissue_map import TissueMapper
from faceforge.ui.widgets.slider_row import SliderRow
from faceforge.ui.widgets.section_label import SectionLabel


class ScannerWindow(QDialog):
    """Popup window for the virtual scanner."""

    scan_requested = Signal()
    plane_changed = Signal(dict)   # emitted on any param change (for 3D plane viz)
    closed = Signal()

    def __init__(self, parent: QWidget, engine: ScannerEngine) -> None:
        super().__init__(parent)
        self.engine = engine
        self._auto_update = False
        self._scanning = False

        self.setWindowTitle("Virtual Scanner")
        self.resize(900, 600)
        self.setMinimumSize(600, 400)

        # State (must be set before widgets, since eventFilter can fire during setup)
        self._last_image: NDArray | None = None
        self._last_image_size: tuple[int, int] = (0, 0)

        # Main layout: image (left) | controls (right)
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # ── Image area (left, expanding) ──
        image_container = QVBoxLayout()
        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._image_label.setMinimumSize(256, 256)
        self._image_label.setStyleSheet("QLabel { background-color: #000; border: 1px solid #333; }")
        self._image_label.setMouseTracking(True)
        self._image_label.installEventFilter(self)
        image_container.addWidget(self._image_label)

        # Status line under image
        self._status_label = QLabel("Click 'Scan' to generate an image")
        self._status_label.setStyleSheet("QLabel { color: #8b8e99; font-size: 11px; }")
        image_container.addWidget(self._status_label)

        main_layout.addLayout(image_container, stretch=1)

        # ── Controls panel (right, fixed width) ──
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFixedWidth(220)
        controls_scroll.setFrameShape(QFrame.Shape.NoFrame)

        controls = QWidget()
        ctrl_layout = QVBoxLayout(controls)
        ctrl_layout.setContentsMargins(4, 4, 4, 4)
        ctrl_layout.setSpacing(4)

        # -- Scan Mode --
        ctrl_layout.addWidget(SectionLabel("Scan Mode"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["CT", "MRI T1", "MRI T2", "X-ray", "Anatomical"])
        self._mode_combo.currentIndexChanged.connect(self._on_param_changed)
        ctrl_layout.addWidget(self._mode_combo)

        # -- Orientation --
        ctrl_layout.addWidget(SectionLabel("Orientation"))
        self._orient_combo = QComboBox()
        self._orient_combo.addItems(["Axial", "Coronal", "Sagittal"])
        self._orient_combo.currentIndexChanged.connect(self._on_param_changed)
        ctrl_layout.addWidget(self._orient_combo)

        # -- Position --
        ctrl_layout.addWidget(SectionLabel("Position"))
        self._pos_x = SliderRow("X", min_val=-100, max_val=100, default=0, decimals=0)
        self._pos_y = SliderRow("Y", min_val=-100, max_val=100, default=0, decimals=0)
        self._pos_z = SliderRow("Z", min_val=-200, max_val=20, default=-80, decimals=0)
        for s in (self._pos_x, self._pos_y, self._pos_z):
            s.value_changed.connect(lambda _: self._on_param_changed())
            ctrl_layout.addWidget(s)

        # -- Dimensions --
        ctrl_layout.addWidget(SectionLabel("Dimensions"))
        self._width_slider = SliderRow("Width", min_val=10, max_val=200, default=80, decimals=0)
        self._height_slider = SliderRow("Height", min_val=10, max_val=200, default=80, decimals=0)
        self._depth_slider = SliderRow("Depth", min_val=1, max_val=50, default=5, decimals=0)
        for s in (self._width_slider, self._height_slider, self._depth_slider):
            s.value_changed.connect(lambda _: self._on_param_changed())
            ctrl_layout.addWidget(s)

        # -- Processing --
        ctrl_layout.addWidget(SectionLabel("Reduction"))
        self._reduction_combo = QComboBox()
        self._reduction_combo.addItems(["Mean", "Max", "Min", "Sum (X-ray)"])
        self._reduction_combo.currentIndexChanged.connect(self._on_param_changed)
        ctrl_layout.addWidget(self._reduction_combo)

        # -- Resolution --
        ctrl_layout.addWidget(SectionLabel("Resolution"))
        self._res_combo = QComboBox()
        self._res_combo.addItems(["64", "128", "256", "512"])
        self._res_combo.setCurrentIndex(1)  # default 128
        self._res_combo.currentIndexChanged.connect(self._on_param_changed)
        ctrl_layout.addWidget(self._res_combo)

        # -- Buttons --
        ctrl_layout.addSpacing(8)
        self._scan_btn = QPushButton("Scan")
        self._scan_btn.setStyleSheet(
            "QPushButton { background: #4fd1c5; color: #0a0b0e; font-weight: bold; "
            "padding: 6px; border-radius: 4px; }"
            "QPushButton:hover { background: #38b2ac; }"
        )
        self._scan_btn.clicked.connect(self._do_scan)
        ctrl_layout.addWidget(self._scan_btn)

        self._auto_btn = QPushButton("Auto-update: OFF")
        self._auto_btn.setCheckable(True)
        self._auto_btn.clicked.connect(self._toggle_auto)
        ctrl_layout.addWidget(self._auto_btn)

        ctrl_layout.addStretch()
        controls_scroll.setWidget(controls)
        main_layout.addWidget(controls_scroll)

    # ── Public API ──

    @property
    def scan_params(self) -> dict:
        """Current scanner parameters as a dict."""
        mode_map = {0: "ct", 1: "mri_t1", 2: "mri_t2", 3: "xray", 4: "anatomical"}
        orient_map = {0: "axial", 1: "coronal", 2: "sagittal"}
        reduction_map = {0: "mean", 1: "max", 2: "min", 3: "sum"}

        mode = mode_map.get(self._mode_combo.currentIndex(), "ct")
        orientation = orient_map.get(self._orient_combo.currentIndex(), "axial")
        reduction = reduction_map.get(self._reduction_combo.currentIndex(), "mean")
        resolution = int(self._res_combo.currentText())

        origin, normal, right, up = self._compute_plane(orientation)

        return {
            "origin": origin,
            "normal": normal,
            "right": right,
            "up": up,
            "width": self._width_slider.value,
            "height": self._height_slider.value,
            "depth": self._depth_slider.value,
            "resolution": resolution,
            "mode": mode,
            "reduction": reduction,
        }

    @property
    def plane_params(self) -> dict:
        """Lightweight plane-only params for 3D visualization."""
        orient_map = {0: "axial", 1: "coronal", 2: "sagittal"}
        orientation = orient_map.get(self._orient_combo.currentIndex(), "axial")
        origin, normal, right, up = self._compute_plane(orientation)
        return {
            "origin": origin,
            "normal": normal,
            "right": right,
            "up": up,
            "width": self._width_slider.value,
            "height": self._height_slider.value,
        }

    def run_scan(self, meshes: list) -> None:
        """Cache meshes and run the scan, keeping the UI alive via processEvents.

        Called directly from app.py — runs synchronously but pumps the event
        loop after each mesh so the window stays responsive.
        """
        if self._scanning:
            return
        self._scanning = True
        self._scan_btn.setEnabled(False)
        self._scan_btn.setText("Scanning...")
        self._status_label.setText("Caching meshes...")
        QApplication.processEvents()

        params = self.scan_params

        # Cache meshes
        self.engine.cache_meshes(meshes)
        n_cached = len(self.engine._cache)
        self._status_label.setText(f"Scanning {n_cached} meshes...")
        QApplication.processEvents()

        # Progress callback pumps event loop
        def _progress(frac: float) -> None:
            pct = int(frac * 100)
            self._status_label.setText(f"Scanning... {pct}%")
            QApplication.processEvents()

        try:
            image = self.engine.scan(
                origin=params["origin"],
                normal=params["normal"],
                right=params["right"],
                up=params["up"],
                width=params["width"],
                height=params["height"],
                depth=params["depth"],
                resolution=params["resolution"],
                mode=params["mode"],
                reduction=params["reduction"],
                progress_callback=_progress,
            )
            self.display_image(image)
        except Exception as exc:
            self._status_label.setText(f"Scan failed: {exc}")
            import traceback
            traceback.print_exc()

        self._scanning = False
        self._scan_btn.setEnabled(True)
        self._scan_btn.setText("Scan")

    def display_image(self, image: NDArray) -> None:
        """Display a scan result image.

        Parameters
        ----------
        image : (H, W) float32 grayscale or (H, W, 3) float32 RGB
        """
        self._last_image = image

        mode_idx = self._mode_combo.currentIndex()
        mode_map = {0: "ct", 1: "mri_t1", 2: "mri_t2", 3: "xray", 4: "anatomical"}
        mode = mode_map.get(mode_idx, "ct")

        if image.ndim == 3:
            h, w, _ = image.shape
            rgb = np.clip(image * 255, 0, 255).astype(np.uint8)
        else:
            h, w = image.shape
            rgb = _apply_colormap_vec(image, mode)

        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = rgb
        rgba[:, :, 3] = 255
        qimg = QImage(rgba.data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()

        self._last_image_size = (w, h)
        pixmap = QPixmap.fromImage(qimg)

        label_size = self._image_label.size()
        scaled = pixmap.scaled(
            label_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._image_label.setPixmap(scaled)
        self._status_label.setText(f"Scan complete: {w}x{h}")

    # ── Events ──

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)

    def eventFilter(self, obj, event):
        """Show mouse coordinates in status line when hovering over image."""
        if obj is self._image_label and self._last_image is not None:
            from PySide6.QtCore import QEvent
            if event.type() == QEvent.Type.MouseMove:
                pos = event.position() if hasattr(event, 'position') else event.pos()
                self._update_mouse_info(pos)
        return super().eventFilter(obj, event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._last_image is not None:
            self.display_image(self._last_image)

    # ── Internal ──

    def _compute_plane(self, orientation: str):
        """Compute origin, normal, right, up vectors for the given orientation."""
        origin = np.array([
            self._pos_x.value,
            self._pos_y.value,
            self._pos_z.value,
        ], dtype=np.float64)

        if orientation == "axial":
            normal = np.array([0.0, 0.0, -1.0])
            right = np.array([1.0, 0.0, 0.0])
            up = np.array([0.0, -1.0, 0.0])
        elif orientation == "coronal":
            normal = np.array([0.0, -1.0, 0.0])
            right = np.array([1.0, 0.0, 0.0])
            up = np.array([0.0, 0.0, -1.0])
        else:  # sagittal
            normal = np.array([1.0, 0.0, 0.0])
            right = np.array([0.0, -1.0, 0.0])
            up = np.array([0.0, 0.0, -1.0])

        return origin, normal, right, up

    def _do_scan(self) -> None:
        """Trigger a scan — emits scan_requested for app.py to call run_scan()."""
        if self._scanning:
            return
        self.scan_requested.emit()

    def _toggle_auto(self, checked: bool) -> None:
        self._auto_update = checked
        self._auto_btn.setText(f"Auto-update: {'ON' if checked else 'OFF'}")
        if checked:
            self._do_scan()

    def _on_param_changed(self, _=None) -> None:
        self.plane_changed.emit(self.plane_params)
        if self._auto_update and not self._scanning:
            self._do_scan()

    def _update_mouse_info(self, pos) -> None:
        """Update status with scan coordinates under mouse."""
        if self._last_image is None:
            return

        pixmap = self._image_label.pixmap()
        if pixmap is None:
            return

        label_w = self._image_label.width()
        label_h = self._image_label.height()
        pm_w = pixmap.width()
        pm_h = pixmap.height()

        x_offset = (label_w - pm_w) / 2
        y_offset = (label_h - pm_h) / 2

        ix = pos.x() - x_offset
        iy = pos.y() - y_offset

        if ix < 0 or iy < 0 or ix >= pm_w or iy >= pm_h:
            return

        img_w, img_h = self._last_image_size
        img_x = int(ix / pm_w * img_w)
        img_y = int(iy / pm_h * img_h)

        if img_x < 0 or img_y < 0 or img_x >= img_w or img_y >= img_h:
            return

        if self._last_image.ndim == 3:
            val = self._last_image[img_y, img_x]
            self._status_label.setText(
                f"Pixel ({img_x}, {img_y}) RGB: ({val[0]:.2f}, {val[1]:.2f}, {val[2]:.2f})"
            )
        else:
            val = self._last_image[img_y, img_x]
            self._status_label.setText(f"Pixel ({img_x}, {img_y}) Value: {val:.3f}")


# ── Vectorized colormap helpers ──────────────────────────────────────

def _apply_colormap_vec(image: NDArray, mode: str) -> NDArray:
    """Apply colormap to a (H, W) float32 image, returning (H, W, 3) uint8 RGB."""
    v = np.clip(image, 0.0, 1.0)

    if mode in ("ct", "xray"):
        g = (v * 255).astype(np.uint8)
        return np.stack([g, g, g], axis=-1)

    if mode in ("mri_t1", "mri_t2"):
        r = np.clip(v * 255 + v * (1 - v) * 20, 0, 255).astype(np.uint8)
        g = (v * 255).astype(np.uint8)
        b = (v * 245).astype(np.uint8)
        return np.stack([r, g, b], axis=-1)

    g = (v * 255).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)
