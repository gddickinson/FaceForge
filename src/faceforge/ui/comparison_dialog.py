"""Comparison mode dialog for side-by-side anatomy views.

Configures left/right viewport settings for split-screen comparison.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QPushButton, QComboBox, QLabel, QGroupBox, QCheckBox,
)
from PySide6.QtCore import Signal


# Layer presets for quick comparison setups
_LAYER_PRESETS = {
    "Skeleton": [
        "skull", "thoracic", "lumbar", "ribs", "pelvis",
        "upper_limb_skel", "lower_limb_skel", "hands_skel", "feet_skel",
        "vertebrae",
    ],
    "Muscular": [
        "back_muscles", "shoulder_muscles", "arm_muscles",
        "torso_muscles", "hip_muscles", "leg_muscles",
        "jaw_muscles", "expression_muscles", "neck_muscles",
    ],
    "Organs": ["organs"],
    "Vasculature": ["vasculature"],
    "Brain": ["brain"],
    "Skin": ["skin"],
    "All Anatomy": [
        "skull", "thoracic", "lumbar", "ribs", "pelvis",
        "upper_limb_skel", "lower_limb_skel",
        "back_muscles", "shoulder_muscles", "arm_muscles",
        "torso_muscles", "hip_muscles", "leg_muscles",
        "organs", "vasculature",
    ],
}

# Render mode options
_RENDER_MODES = [
    "Current", "Wireframe", "Solid", "X-Ray", "Points", "Opaque",
    "Hologram", "Blueprint", "Thermal",
]


class ComparisonDialog(QDialog):
    """Dialog for configuring side-by-side comparison views.

    Signals
    -------
    config_changed(dict, dict)
        Emitted with (left_config, right_config) dicts.
    """

    config_changed = Signal(dict, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Comparative Anatomy View")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        # Mode selector
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Compare:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems([
            "Layer Comparison",
            "Render Mode Comparison",
            "Before/After Pathology",
        ])
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self._mode_combo)
        layout.addLayout(mode_layout)

        # Left config
        left_group = QGroupBox("Left View (A)")
        left_form = QFormLayout(left_group)

        self._left_layers = QComboBox()
        self._left_layers.addItems(list(_LAYER_PRESETS.keys()))
        self._left_layers.setCurrentText("Skeleton")
        left_form.addRow("Layers:", self._left_layers)

        self._left_render = QComboBox()
        self._left_render.addItems(_RENDER_MODES)
        left_form.addRow("Render:", self._left_render)

        layout.addWidget(left_group)

        # Right config
        right_group = QGroupBox("Right View (B)")
        right_form = QFormLayout(right_group)

        self._right_layers = QComboBox()
        self._right_layers.addItems(list(_LAYER_PRESETS.keys()))
        self._right_layers.setCurrentText("Muscular")
        right_form.addRow("Layers:", self._right_layers)

        self._right_render = QComboBox()
        self._right_render.addItems(_RENDER_MODES)
        right_form.addRow("Render:", self._right_render)

        layout.addWidget(right_group)

        # Sync camera
        self._sync_camera = QCheckBox("Synchronize Camera")
        self._sync_camera.setChecked(True)
        layout.addWidget(self._sync_camera)

        # Apply button
        btn_layout = QHBoxLayout()
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._apply)
        btn_layout.addWidget(apply_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)

        layout.addLayout(btn_layout)

    def _on_mode_changed(self, index: int) -> None:
        # Enable/disable controls based on mode
        is_render = index == 1
        self._left_layers.setEnabled(not is_render)
        self._right_layers.setEnabled(not is_render)
        self._left_render.setEnabled(True)
        self._right_render.setEnabled(True)

    def _apply(self) -> None:
        left_config = {
            "layers": _LAYER_PRESETS.get(self._left_layers.currentText(), []),
            "render_mode": self._left_render.currentText().lower(),
            "sync_camera": self._sync_camera.isChecked(),
        }
        right_config = {
            "layers": _LAYER_PRESETS.get(self._right_layers.currentText(), []),
            "render_mode": self._right_render.currentText().lower(),
            "sync_camera": self._sync_camera.isChecked(),
        }
        self.config_changed.emit(left_config, right_config)

    def get_configs(self) -> tuple[dict, dict]:
        """Return current left and right configurations."""
        left = {
            "layers": _LAYER_PRESETS.get(self._left_layers.currentText(), []),
            "render_mode": self._left_render.currentText().lower(),
        }
        right = {
            "layers": _LAYER_PRESETS.get(self._right_layers.currentText(), []),
            "render_mode": self._right_render.currentText().lower(),
        }
        return left, right
