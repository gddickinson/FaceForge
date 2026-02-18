"""Body tab: pose presets, spine/joint sliders, breathing."""

from PySide6.QtWidgets import (
    QScrollArea, QWidget, QVBoxLayout, QGridLayout, QPushButton, QSizePolicy,
)
from PySide6.QtCore import Qt

from faceforge.core.events import EventBus, EventType
from faceforge.core.state import StateManager
from faceforge.core.config_loader import load_config
from faceforge.ui.widgets.section_label import SectionLabel
from faceforge.ui.widgets.slider_row import SliderRow
from faceforge.ui.widgets.toggle_row import ToggleRow


class BodyTab(QScrollArea):
    """Tab containing body pose presets, joint sliders, and breathing controls.

    Joint slider changes publish ``BODY_STATE_CHANGED`` events.
    Pose preset buttons publish ``BODY_POSE_SET`` events.
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

        self._poses = self._load_poses()
        self._sliders: dict[str, SliderRow] = {}

        # ── 1. Body Poses ──
        self._layout.addWidget(SectionLabel("Body Poses"))
        self._pose_buttons: dict[str, QPushButton] = {}
        pose_grid = QGridLayout()
        pose_grid.setSpacing(4)
        for i, name in enumerate(self._poses.keys()):
            btn = QPushButton(name.capitalize())
            btn.setObjectName("poseButton")
            btn.setCheckable(True)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.clicked.connect(lambda checked, n=name: self._on_pose_selected(n))
            pose_grid.addWidget(btn, i // 2, i % 2)
            self._pose_buttons[name] = btn
        pose_widget = QWidget()
        pose_widget.setLayout(pose_grid)
        self._layout.addWidget(pose_widget)

        # ── 2. Spine ──
        self._layout.addWidget(SectionLabel("Spine"))
        self._add_slider("spine_flex", "Flex", -1.0, 1.0)
        self._add_slider("spine_lat_bend", "Lat Bend", -1.0, 1.0)
        self._add_slider("spine_rotation", "Rotation", -1.0, 1.0)

        # ── 3. Shoulders ──
        self._layout.addWidget(SectionLabel("Shoulders"))
        self._add_slider("shoulder_r_abduct", "R Abduct", -1.0, 1.0)
        self._add_slider("shoulder_r_flex", "R Flex", -1.0, 1.0)
        self._add_slider("shoulder_r_rotate", "R Rotate", -1.0, 1.0)
        self._add_slider("shoulder_l_abduct", "L Abduct", -1.0, 1.0)
        self._add_slider("shoulder_l_flex", "L Flex", -1.0, 1.0)
        self._add_slider("shoulder_l_rotate", "L Rotate", -1.0, 1.0)

        # ── 4. Elbows ──
        self._layout.addWidget(SectionLabel("Elbows"))
        self._add_slider("elbow_r_flex", "R Flex", -1.0, 1.0)
        self._add_slider("elbow_l_flex", "L Flex", -1.0, 1.0)

        # ── 5. Hips ──
        self._layout.addWidget(SectionLabel("Hips"))
        self._add_slider("hip_r_flex", "R Flex", -1.0, 1.0)
        self._add_slider("hip_r_abduct", "R Abduct", -1.0, 1.0)
        self._add_slider("hip_r_rotate", "R Rotate", -1.0, 1.0)
        self._add_slider("hip_l_flex", "L Flex", -1.0, 1.0)
        self._add_slider("hip_l_abduct", "L Abduct", -1.0, 1.0)
        self._add_slider("hip_l_rotate", "L Rotate", -1.0, 1.0)

        # ── 6. Knees ──
        self._layout.addWidget(SectionLabel("Knees"))
        self._add_slider("knee_r_flex", "R Flex", -1.0, 1.0)
        self._add_slider("knee_l_flex", "L Flex", -1.0, 1.0)

        # ── 7. Ankles ──
        self._layout.addWidget(SectionLabel("Ankles"))
        self._add_slider("ankle_r_flex", "R Flex", -1.0, 1.0)
        self._add_slider("ankle_r_invert", "R Invert", -1.0, 1.0)
        self._add_slider("ankle_l_flex", "L Flex", -1.0, 1.0)
        self._add_slider("ankle_l_invert", "L Invert", -1.0, 1.0)

        # ── 8. Wrists ──
        self._layout.addWidget(SectionLabel("Wrists"))
        self._add_slider("wrist_r_flex", "R Flex", -1.0, 1.0)
        self._add_slider("wrist_r_deviate", "R Deviate", -1.0, 1.0)
        self._add_slider("forearm_r_rotate", "R Forearm Rot", -1.0, 1.0)
        self._add_slider("wrist_l_flex", "L Flex", -1.0, 1.0)
        self._add_slider("wrist_l_deviate", "L Deviate", -1.0, 1.0)
        self._add_slider("forearm_l_rotate", "L Forearm Rot", -1.0, 1.0)

        # ── 9. Hands ──
        self._layout.addWidget(SectionLabel("Hands"))
        self._add_slider("finger_curl_r", "R Curl", -1.0, 1.0)
        self._add_slider("finger_spread_r", "R Spread", -1.0, 1.0)
        self._add_slider("thumb_op_r", "R Thumb Op", 0.0, 1.0)
        self._add_slider("finger_curl_l", "L Curl", -1.0, 1.0)
        self._add_slider("finger_spread_l", "L Spread", -1.0, 1.0)
        self._add_slider("thumb_op_l", "L Thumb Op", 0.0, 1.0)

        # ── 10. Feet ──
        self._layout.addWidget(SectionLabel("Feet"))
        self._add_slider("toe_curl_r", "R Curl", -1.0, 1.0)
        self._add_slider("toe_spread_r", "R Spread", -1.0, 1.0)
        self._add_slider("toe_curl_l", "L Curl", -1.0, 1.0)
        self._add_slider("toe_spread_l", "L Spread", -1.0, 1.0)

        # ── 11. Breathing ──
        self._layout.addWidget(SectionLabel("Breathing"))
        self._add_slider("breath_depth", "Depth", 0.0, 1.0, default=0.3)
        self._add_slider("breath_rate", "Rate", 0.0, 1.0, default=0.25)

        self._auto_breath = ToggleRow("Auto Breathing", default=True)
        self._auto_breath.toggled.connect(self._on_auto_breath_toggled)
        self._layout.addWidget(self._auto_breath)

        self._layout.addStretch()

    # ── Helpers ──

    def _add_slider(
        self,
        field: str,
        label: str,
        min_val: float = -1.0,
        max_val: float = 1.0,
        default: float = 0.0,
    ) -> SliderRow:
        slider = SliderRow(label, min_val=min_val, max_val=max_val, default=default)
        slider.value_changed.connect(lambda v, f=field: self._on_joint_changed(f, v))
        self._sliders[field] = slider
        self._layout.addWidget(slider)
        return slider

    @staticmethod
    def _load_poses() -> dict[str, dict]:
        try:
            return load_config("body_poses.json")
        except Exception:
            return {"anatomical": {}}

    # ── Slots ──

    def _on_pose_selected(self, name: str) -> None:
        # Highlight button
        for btn_name, btn in self._pose_buttons.items():
            btn.setChecked(btn_name == name)
        # Read pose values and update sliders
        pose = self._poses.get(name, {})
        # The pose dict uses JS camelCase keys; map to Python snake_case
        js_to_py = self._state.body._JS_KEY_MAP
        for js_key, value in pose.items():
            py_key = js_to_py.get(js_key, js_key)
            if py_key in self._sliders:
                self._sliders[py_key].set_value(value)
        # Zero out sliders not in the pose
        all_pose_py_keys = {js_to_py.get(k, k) for k in pose.keys()}
        for field, slider in self._sliders.items():
            if field not in all_pose_py_keys and field not in ("breath_depth", "breath_rate"):
                slider.set_value(0.0)
        self._bus.publish(EventType.BODY_POSE_SET, name=name, values=pose)

    def _on_joint_changed(self, field: str, value: float) -> None:
        self._bus.publish(EventType.BODY_STATE_CHANGED, field=field, value=value)

    def _on_auto_breath_toggled(self, enabled: bool) -> None:
        self._bus.publish(EventType.BODY_STATE_CHANGED, field="auto_breath_body", value=enabled)

    # ── External Updates ──

    def sync_from_state(self) -> None:
        """Sync all sliders from the body state."""
        body = self._state.body
        for field, slider in self._sliders.items():
            val = getattr(body, field, 0.0)
            slider.set_value(val)
        self._auto_breath.set_checked(body.auto_breath_body)
