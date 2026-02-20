"""Layers tab: visibility toggles for all anatomical groups."""

from PySide6.QtWidgets import QScrollArea, QWidget, QVBoxLayout, QHBoxLayout, QRadioButton, QButtonGroup
from PySide6.QtCore import Qt

from faceforge.core.events import EventBus, EventType
from faceforge.core.state import StateManager
from faceforge.ui.widgets.section_label import SectionLabel
from faceforge.ui.widgets.toggle_row import ToggleRow
from faceforge.ui.widgets.collapsible_section import CollapsibleSection


# Layer definitions: (toggle_id, display_label, default_visible)
_HEAD_LAYERS = [
    ("skull", "Skull", True),
    ("face", "Face Skin", False),
    ("jaw_muscles", "Jaw Muscles", False),
    ("expression_muscles", "Expression Muscles", False),
    ("neck_muscles", "Neck Muscles", False),
    ("vertebrae", "Vertebrae", True),
    ("eyes", "Eyes", False),
    ("ears", "Ears", False),
    ("nose_cart", "Nasal Cartilages", False),
    ("eyebrows", "Eyebrows", False),
    ("throat", "Throat", False),
    ("teeth", "Teeth", True),
]

_BODY_SKELETON_LAYERS = [
    ("thoracic", "Thoracic Spine", True),
    ("lumbar", "Lumbar Spine", True),
    ("ribs", "Rib Cage", True),
    ("pelvis", "Pelvis", True),
    ("upper_limb_skel", "Upper Limb", True),
    ("lower_limb_skel", "Lower Limb", True),
    ("hands_skel", "Hands", True),
    ("feet_skel", "Feet", True),
]

_MUSCLE_GROUPS = [
    ("back_muscles", "Back Muscles"),
    ("shoulder_muscles", "Shoulder Muscles"),
    ("arm_muscles", "Arm Muscles"),
    ("torso_muscles", "Torso Muscles"),
    ("hip_muscles", "Hip Muscles"),
    ("leg_muscles", "Leg Muscles"),
]

_COLLAPSIBLE_GROUPS = [
    ("vasculature", "Vasculature"),
    ("brain", "Brain"),
]

_ADDITIONAL_ANATOMY_GROUPS = [
    ("hand_muscles", "Hand Muscles"),
    ("foot_muscles", "Foot Muscles"),
    ("pelvic_floor", "Pelvic Floor"),
    ("ligaments", "Ligaments & Tendons"),
    ("oral", "Oral Structures"),
    ("cardiac_additional", "Cardiac Detail"),
    ("intestinal", "Intestinal Detail"),
    ("cns_additional", "CNS Additional"),
]


class LayersTab(QScrollArea):
    """Tab with visibility toggles for every anatomical layer group.

    Toggle changes publish ``LAYER_TOGGLED`` events via the event bus.
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

        self._toggles: dict[str, ToggleRow] = {}
        self._sections: dict[str, CollapsibleSection] = {}

        # ── Head ──
        self._layout.addWidget(SectionLabel("Head"))

        # Skull mode radio buttons
        skull_mode_row = QWidget()
        skull_mode_layout = QHBoxLayout(skull_mode_row)
        skull_mode_layout.setContentsMargins(8, 2, 0, 2)
        skull_mode_layout.setSpacing(12)
        self._skull_mode_group = QButtonGroup(self)
        self._radio_original = QRadioButton("Original Skull")
        self._radio_bp3d = QRadioButton("BP3D Skull")
        self._radio_original.setChecked(True)
        self._skull_mode_group.addButton(self._radio_original, 0)
        self._skull_mode_group.addButton(self._radio_bp3d, 1)
        skull_mode_layout.addWidget(self._radio_original)
        skull_mode_layout.addWidget(self._radio_bp3d)
        skull_mode_layout.addStretch()
        self._layout.addWidget(skull_mode_row)
        self._skull_mode_group.idToggled.connect(self._on_skull_mode_toggled)

        self._add_layer_group(_HEAD_LAYERS)

        # ── Body Skeleton ──
        self._layout.addWidget(SectionLabel("Body Skeleton"))
        self._add_layer_group(_BODY_SKELETON_LAYERS)

        # ── Body Surface ──
        self._layout.addWidget(SectionLabel("Body Surface"))
        self._add_layer_group([("skin", "Skin", False)])

        # ── Body Soft Tissue ──
        self._layout.addWidget(SectionLabel("Body Soft Tissue"))

        # Muscle groups as collapsible sections
        for group_id, title in _MUSCLE_GROUPS:
            section = CollapsibleSection(title)
            section.master_toggled.connect(
                lambda checked, gid=group_id: self._on_layer_toggled(gid, checked)
            )
            section.child_toggled.connect(
                lambda tid, checked: self._on_layer_toggled(tid, checked)
            )
            self._sections[group_id] = section
            self._layout.addWidget(section)

        # Organs as collapsible section
        organs_section = CollapsibleSection("Organs")
        organs_section.master_toggled.connect(
            lambda checked: self._on_layer_toggled("organs", checked)
        )
        organs_section.child_toggled.connect(
            lambda tid, checked: self._on_layer_toggled(tid, checked)
        )
        self._sections["organs"] = organs_section
        self._layout.addWidget(organs_section)

        # Vasculature and brain as collapsible sections
        for group_id, title in _COLLAPSIBLE_GROUPS:
            section = CollapsibleSection(title)
            section.master_toggled.connect(
                lambda checked, gid=group_id: self._on_layer_toggled(gid, checked)
            )
            section.child_toggled.connect(
                lambda tid, checked: self._on_layer_toggled(tid, checked)
            )
            self._sections[group_id] = section
            self._layout.addWidget(section)

        # ── Additional Anatomy ──
        self._layout.addWidget(SectionLabel("Additional Anatomy"))
        for group_id, title in _ADDITIONAL_ANATOMY_GROUPS:
            section = CollapsibleSection(title)
            section.master_toggled.connect(
                lambda checked, gid=group_id: self._on_layer_toggled(gid, checked)
            )
            section.child_toggled.connect(
                lambda tid, checked: self._on_layer_toggled(tid, checked)
            )
            self._sections[group_id] = section
            self._layout.addWidget(section)

        # ── Debug / Constraints ──
        self._layout.addWidget(SectionLabel("Debug / Constraints"))
        self._add_layer_group([
            ("fascia", "Fascia Targets", False),
        ])

        # ── Integument & Misc ──
        self._layout.addWidget(SectionLabel("Integument & Misc"))
        self._add_layer_group([
            ("head_hair", "Head Hair", False),
            ("pubic_hair", "Pubic Hair", False),
            ("epicranial_aponeurosis", "Epicranial Aponeurosis", False),
            ("spinal_central_canal", "Spinal Central Canal", False),
        ])

        self._layout.addStretch()

    # ── Helpers ──

    def _add_layer_group(self, layers: list[tuple[str, str, bool]]) -> None:
        for toggle_id, label, default in layers:
            row = ToggleRow(label, default=default)
            row.toggled.connect(
                lambda checked, tid=toggle_id: self._on_layer_toggled(tid, checked)
            )
            self._toggles[toggle_id] = row
            self._layout.addWidget(row)

    # ── Slots ──

    def _on_layer_toggled(self, layer_id: str, visible: bool) -> None:
        self._bus.publish(EventType.LAYER_TOGGLED, layer=layer_id, visible=visible)

    def _on_skull_mode_toggled(self, button_id: int, checked: bool) -> None:
        if not checked:
            return
        mode = "bp3d" if button_id == 1 else "original"
        self._bus.publish(EventType.SKULL_MODE_CHANGED, mode=mode)

    # ── Public API ──

    def set_layer_visible(self, layer_id: str, visible: bool) -> None:
        """Programmatically set a layer toggle."""
        toggle = self._toggles.get(layer_id)
        if toggle is not None:
            toggle.set_checked(visible)

    def on_structures_loaded(self, group_id: str = "", items: list = None, **kw) -> None:
        """Populate a collapsible section after its structures load.

        Parameters
        ----------
        group_id : str
            The section key (e.g. ``"organs"``, ``"back_muscles"``).
        items : list[dict]
            Each dict has ``toggle_id``, ``name``, and optionally ``category``.
        """
        if items is None:
            return
        section = self._sections.get(group_id)
        if section is None:
            return

        # Group items by category if present (try "category" then "type")
        categories: dict[str, list[dict]] = {}
        uncategorized: list[dict] = []
        for item in items:
            cat = item.get("category") or item.get("type")
            if cat:
                categories.setdefault(cat, []).append(item)
            else:
                uncategorized.append(item)

        # Suppress child_toggled signals during bulk population
        section._updating_master = True

        # Add categorized items with headers
        for cat_name in sorted(categories):
            section.add_category_header(cat_name)
            for item in categories[cat_name]:
                section.add_item(item["toggle_id"], item["name"], default=True)

        # Add uncategorized items
        for item in uncategorized:
            section.add_item(item["toggle_id"], item["name"], default=True)

        section._updating_master = False
        section._update_master_state()
