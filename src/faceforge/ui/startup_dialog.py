"""Startup dialog for choosing an initial layer configuration preset."""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGridLayout, QWidget, QSizePolicy, QTabWidget, QScrollArea,
)
from PySide6.QtCore import Qt

from faceforge.ui.style import DARK_THEME


# ── Preset definitions ──────────────────────────────────────────────────
# Each preset maps to a dict of layer_id → True/False.
# Layers not listed inherit their default from the layers tab.
# "on_demand" entries trigger on-demand loading (organs, vasculature, muscles, etc.)

# All skeleton layers
_SKELETON = {
    "skull": True, "vertebrae": True, "teeth": True,
    "thoracic": True, "lumbar": True, "ribs": True, "pelvis": True,
    "upper_limb_skel": True, "lower_limb_skel": True,
    "hands_skel": True, "feet_skel": True,
}

# All head soft tissue
_HEAD_SOFT = {
    "face": True, "jaw_muscles": True, "expression_muscles": True,
    "neck_muscles": True, "eyes": True, "ears": True,
    "nose_cart": True, "eyebrows": True, "throat": True,
}

# All body muscles
_BODY_MUSCLES = {
    "back_muscles": True, "shoulder_muscles": True, "arm_muscles": True,
    "torso_muscles": True, "hip_muscles": True, "leg_muscles": True,
}

# All off (used as a base to override from)
_ALL_OFF = {
    "skull": False, "face": False, "jaw_muscles": False,
    "expression_muscles": False, "neck_muscles": False,
    "vertebrae": False, "eyes": False, "ears": False,
    "nose_cart": False, "eyebrows": False, "throat": False,
    "teeth": False,
    "thoracic": False, "lumbar": False, "ribs": False, "pelvis": False,
    "upper_limb_skel": False, "lower_limb_skel": False,
    "hands_skel": False, "feet_skel": False,
    "skin": False,
    "back_muscles": False, "shoulder_muscles": False, "arm_muscles": False,
    "torso_muscles": False, "hip_muscles": False, "leg_muscles": False,
    "organs": False, "vasculature": False, "brain": False,
}

PRESETS: dict[str, dict] = {}


def _preset(name, description, icon, layers, camera="body_front"):
    PRESETS[name] = {"description": description, "icon": icon, "layers": layers, "camera": camera}


_preset(
    "Skeleton",
    "Bones, vertebrae, and teeth only",
    "\u2620",  # skull
    {**_ALL_OFF, **_SKELETON},
)

_preset(
    "Full Anatomy",
    "All layers including skin",
    "\u2605",  # star
    {**_SKELETON, **_HEAD_SOFT,
     "skin": True,
     **_BODY_MUSCLES,
     "organs": True, "vasculature": True, "brain": True},
)

_preset(
    "Full Anatomy (No Skin)",
    "Everything visible except body skin",
    "\u2606",  # empty star
    {**_SKELETON, **_HEAD_SOFT,
     "skin": False,
     **_BODY_MUSCLES,
     "organs": True, "vasculature": True, "brain": True},
)

_preset(
    "Cardiovascular",
    "Skeleton, heart, and blood vessels",
    "\u2665",  # heart
    {**_ALL_OFF, **_SKELETON,
     "organs": True, "vasculature": True},
)

_preset(
    "Respiratory",
    "Skeleton with lungs and airways",
    "\u2601",  # cloud
    {**_ALL_OFF, **_SKELETON,
     "organs": True, "throat": True},
)

_preset(
    "Digestive",
    "Skeleton with digestive organs",
    "\u2600",  # sun
    {**_ALL_OFF, **_SKELETON,
     "organs": True},
)

_preset(
    "Muscular",
    "Skeleton with all muscle groups",
    "\u270A",  # fist
    {**_ALL_OFF, **_SKELETON,
     **_BODY_MUSCLES,
     "jaw_muscles": True, "expression_muscles": True, "neck_muscles": True},
)

_preset(
    "Nervous System",
    "Skeleton, brain, and spinal structures",
    "\u26A1",  # lightning
    {**_ALL_OFF, **_SKELETON,
     "brain": True, "eyes": True},
)

_preset(
    "Head Detail",
    "All head structures, no body soft tissue",
    "\u263A",  # smiley
    {**_ALL_OFF, **_SKELETON, **_HEAD_SOFT},
    camera="head_front",
)

_preset(
    "Head Only",
    "Head and neck only, no body skeleton",
    "\U0001F9D1",  # person silhouette
    {**_ALL_OFF,
     "skull": True, "vertebrae": True, "teeth": True,
     **_HEAD_SOFT},
    camera="head_front",
)

_preset(
    "Nervous System Only",
    "Brain and eyes, no skeleton or muscles",
    "\U0001F9E0",  # brain
    {**_ALL_OFF, "brain": True, "eyes": True},
    camera="body_front",
)

_preset(
    "Default",
    "Standard startup: skeleton and teeth",
    "\u25CB",  # circle
    {},  # empty = use layers tab defaults
)


# ── Dialog ──────────────────────────────────────────────────────────────

class StartupDialog(QDialog):
    """Modal dialog shown at startup for choosing a layer preset or illustration.

    Returns the chosen preset name via ``selected_preset`` after ``exec()``.
    If the user closes the dialog, ``selected_preset`` is ``"Default"``.
    For illustration presets, ``selected_illustration`` is set instead.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("FaceForge — Choose Configuration")
        self.setStyleSheet(DARK_THEME)
        self.setMinimumSize(520, 560)
        self.setModal(True)
        self.selected_preset: str = "Default"
        self.selected_illustration: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        # Title
        title = QLabel("Choose a Configuration")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: 600; color: #4fd1c5;")
        layout.addWidget(title)

        subtitle = QLabel("Select a layer preset or anatomical illustration")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("font-size: 11px; color: #8b8e99; margin-bottom: 8px;")
        layout.addWidget(subtitle)

        # Tab widget
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabBar::tab {
                padding: 6px 18px;
                font-size: 12px;
            }
        """)

        # ── Tab 1: Configurations ──
        config_scroll = QScrollArea()
        config_scroll.setWidgetResizable(True)
        config_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        config_container = QWidget()
        config_grid = QGridLayout(config_container)
        config_grid.setSpacing(8)

        preset_names = list(PRESETS.keys())
        cols = 2
        for i, name in enumerate(preset_names):
            info = PRESETS[name]
            btn = self._make_preset_button(name, info)
            btn.clicked.connect(lambda _, n=name: self._on_preset_clicked(n))
            config_grid.addWidget(btn, i // cols, i % cols)

        config_scroll.setWidget(config_container)
        tabs.addTab(config_scroll, "Configurations")

        # ── Tab 2: Illustrations ──
        from faceforge.ui.illustration_presets import ILLUSTRATION_PRESETS
        illust_scroll = QScrollArea()
        illust_scroll.setWidgetResizable(True)
        illust_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        illust_container = QWidget()
        illust_grid = QGridLayout(illust_container)
        illust_grid.setSpacing(8)

        illust_names = list(ILLUSTRATION_PRESETS.keys())
        for i, name in enumerate(illust_names):
            p = ILLUSTRATION_PRESETS[name]
            info = {"icon": p.icon, "description": p.description}
            btn = self._make_preset_button(name, info, obj_name="illustButton")
            btn.clicked.connect(lambda _, n=name: self._on_illustration_clicked(n))
            illust_grid.addWidget(btn, i // cols, i % cols)

        illust_scroll.setWidget(illust_container)
        tabs.addTab(illust_scroll, "Illustrations")

        layout.addWidget(tabs)

    def _make_preset_button(self, name: str, info: dict,
                            obj_name: str = "presetButton") -> QPushButton:
        icon = info["icon"]
        desc = info["description"]
        btn = QPushButton(f" {icon}  {name}\n      {desc}")
        btn.setObjectName(obj_name)
        btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        btn.setMinimumHeight(52)
        btn.setStyleSheet(f"""
            QPushButton#{obj_name} {{
                text-align: left;
                padding: 8px 12px;
                font-size: 12px;
                line-height: 1.4;
                background-color: #12141a;
                border: 1px solid #252830;
                border-radius: 6px;
            }}
            QPushButton#{obj_name}:hover {{
                background-color: #1a1d26;
                border-color: #4fd1c5;
            }}
            QPushButton#{obj_name}:pressed {{
                background-color: #4fd1c5;
                color: #0a0b0e;
            }}
        """)
        return btn

    def _on_preset_clicked(self, name: str) -> None:
        self.selected_preset = name
        self.accept()

    def _on_illustration_clicked(self, name: str) -> None:
        self.selected_illustration = name
        self.accept()


def apply_preset(preset_name: str, layers_tab, event_bus, gl_widget=None) -> None:
    """Apply a layer preset by toggling layers in the UI.

    Parameters
    ----------
    preset_name : str
        Key into ``PRESETS``.
    layers_tab : LayersTab
        The layers tab instance for programmatic toggle updates.
    event_bus : EventBus
        For publishing LAYER_TOGGLED events (on-demand layers).
    gl_widget : GLViewport, optional
        If provided, set camera to the preset's camera view.
    """
    from faceforge.core.events import EventType

    preset = PRESETS.get(preset_name)
    if preset is None:
        return

    # Apply camera positioning
    if gl_widget is not None and "camera" in preset:
        event_bus.publish(EventType.CAMERA_PRESET, preset=preset["camera"])

    if not preset["layers"]:
        return  # "Default" or unknown — use tab defaults

    layers = preset["layers"]

    # On-demand layers use CollapsibleSection widgets (not simple ToggleRow).
    # They need an explicit LAYER_TOGGLED event to trigger loading in app.py.
    _COLLAPSIBLE_LAYERS = {
        "back_muscles", "shoulder_muscles", "arm_muscles",
        "torso_muscles", "hip_muscles", "leg_muscles",
        "organs", "vasculature", "brain",
    }

    for layer_id, visible in layers.items():
        if layer_id in _COLLAPSIBLE_LAYERS:
            # Collapsible section: update UI and fire event for on-demand loading
            section = layers_tab._sections.get(layer_id)
            if section is not None and visible:
                section.set_master_checked(True)
                event_bus.publish(EventType.LAYER_TOGGLED, layer=layer_id, visible=True)
            elif section is not None and not visible:
                section.set_master_checked(False)
        else:
            # Simple toggle layers (including on-demand ones like "skin")
            # set_layer_visible fires the toggled signal → LAYER_TOGGLED event
            layers_tab.set_layer_visible(layer_id, visible)
