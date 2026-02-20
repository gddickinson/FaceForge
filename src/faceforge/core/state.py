"""Application state management for face and body parameters."""

from dataclasses import dataclass, field, fields
from typing import Any


# All 12 Action Units
AU_IDS = ["AU1", "AU2", "AU4", "AU5", "AU6", "AU9", "AU12", "AU15", "AU20", "AU22", "AU25", "AU26"]


@dataclass
class FaceState:
    """Current face animation state."""
    # Action Units (0-1)
    AU1: float = 0.0   # Inner Brow Raise
    AU2: float = 0.0   # Outer Brow Raise
    AU4: float = 0.0   # Brow Lower
    AU5: float = 0.0   # Upper Lid Raise
    AU6: float = 0.0   # Cheek Raise
    AU9: float = 0.0   # Nose Wrinkle
    AU12: float = 0.0  # Lip Corner Pull
    AU15: float = 0.0  # Lip Corner Drop
    AU20: float = 0.0  # Lip Stretch
    AU22: float = 0.0  # Lip Funneler
    AU25: float = 0.0  # Lips Part
    AU26: float = 0.0  # Jaw Drop

    # Eyes
    eye_look_x: float = 0.0
    eye_look_y: float = 0.0
    blink_amount: float = 0.0

    # Head rotation (-1 to 1)
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    head_roll: float = 0.0

    # Ear wiggle
    ear_wiggle: float = 0.0

    # Pupil dilation (0=constricted, 1=dilated)
    pupil_dilation: float = 0.5

    # Eye color preset name
    eye_color: str = "brown"

    # Auto-animation toggles (all off by default for performance)
    auto_breathing: bool = False
    auto_blink: bool = False
    eye_tracking: bool = False
    micro_expressions: bool = False

    # Internal timers (not user-controlled)
    current_expression: str = "neutral"
    blink_timer: float = 0.0
    next_blink_time: float = 3.0
    breath_phase: float = 0.0
    micro_timer: float = 0.0

    def get_au(self, au_id: str) -> float:
        return getattr(self, au_id, 0.0)

    def set_au(self, au_id: str, value: float) -> None:
        setattr(self, au_id, max(0.0, min(1.0, value)))

    def get_au_dict(self) -> dict[str, float]:
        return {au: self.get_au(au) for au in AU_IDS}

    def set_aus_from_dict(self, aus: dict[str, float]) -> None:
        for au_id, value in aus.items():
            if au_id in AU_IDS:
                self.set_au(au_id, value)


@dataclass
class TargetAU:
    """Target AU values for smooth interpolation."""
    AU1: float = 0.0
    AU2: float = 0.0
    AU4: float = 0.0
    AU5: float = 0.0
    AU6: float = 0.0
    AU9: float = 0.0
    AU12: float = 0.0
    AU15: float = 0.0
    AU20: float = 0.0
    AU22: float = 0.0
    AU25: float = 0.0
    AU26: float = 0.0

    def get(self, au_id: str) -> float:
        return getattr(self, au_id, 0.0)

    def set(self, au_id: str, value: float) -> None:
        setattr(self, au_id, value)

    def set_from_dict(self, aus: dict[str, float]) -> None:
        for au_id, value in aus.items():
            if hasattr(self, au_id):
                setattr(self, au_id, value)

    def to_dict(self) -> dict[str, float]:
        return {au: getattr(self, au) for au in AU_IDS}


@dataclass
class TargetHead:
    """Target head rotation for smooth interpolation."""
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    head_roll: float = 0.0


@dataclass
class BodyState:
    """Current body animation state."""
    # Spine (-1 to 1)
    spine_flex: float = 0.0
    spine_lat_bend: float = 0.0
    spine_rotation: float = 0.0

    # Shoulders (3DOF each, -1 to 1)
    shoulder_r_abduct: float = 0.0
    shoulder_r_flex: float = 0.0
    shoulder_r_rotate: float = 0.0
    shoulder_l_abduct: float = 0.0
    shoulder_l_flex: float = 0.0
    shoulder_l_rotate: float = 0.0

    # Elbows (1DOF)
    elbow_r_flex: float = 0.0
    elbow_l_flex: float = 0.0

    # Forearms (1DOF pronation/supination)
    forearm_r_rotate: float = 0.0
    forearm_l_rotate: float = 0.0

    # Wrists (2DOF)
    wrist_r_flex: float = 0.0
    wrist_r_deviate: float = 0.0
    wrist_l_flex: float = 0.0
    wrist_l_deviate: float = 0.0

    # Hips (3DOF each)
    hip_r_flex: float = 0.0
    hip_r_abduct: float = 0.0
    hip_r_rotate: float = 0.0
    hip_l_flex: float = 0.0
    hip_l_abduct: float = 0.0
    hip_l_rotate: float = 0.0

    # Knees (1DOF)
    knee_r_flex: float = 0.0
    knee_l_flex: float = 0.0

    # Ankles (2DOF)
    ankle_r_flex: float = 0.0
    ankle_r_invert: float = 0.0
    ankle_l_flex: float = 0.0
    ankle_l_invert: float = 0.0

    # Breathing
    breath_depth: float = 0.3
    breath_rate: float = 0.25
    breath_phase_body: float = 0.0
    auto_breath_body: bool = False

    # Hands
    finger_curl_r: float = 0.0
    finger_spread_r: float = 0.0
    thumb_op_r: float = 0.0
    finger_curl_l: float = 0.0
    finger_spread_l: float = 0.0
    thumb_op_l: float = 0.0

    # Feet / toes
    toe_curl_r: float = 0.0
    toe_spread_r: float = 0.0
    toe_curl_l: float = 0.0
    toe_spread_l: float = 0.0

    # Mapping from JS camelCase to Python snake_case
    _JS_KEY_MAP: dict[str, str] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._JS_KEY_MAP = {
            "spineFlex": "spine_flex", "spineLatBend": "spine_lat_bend",
            "spineRotation": "spine_rotation",
            "shoulderRAbduct": "shoulder_r_abduct", "shoulderRFlex": "shoulder_r_flex",
            "shoulderRRotate": "shoulder_r_rotate",
            "shoulderLAbduct": "shoulder_l_abduct", "shoulderLFlex": "shoulder_l_flex",
            "shoulderLRotate": "shoulder_l_rotate",
            "elbowRFlex": "elbow_r_flex", "elbowLFlex": "elbow_l_flex",
            "forearmRRotate": "forearm_r_rotate", "forearmLRotate": "forearm_l_rotate",
            "wristRFlex": "wrist_r_flex", "wristRDeviate": "wrist_r_deviate",
            "wristLFlex": "wrist_l_flex", "wristLDeviate": "wrist_l_deviate",
            "hipRFlex": "hip_r_flex", "hipRAbduct": "hip_r_abduct",
            "hipRRotate": "hip_r_rotate",
            "hipLFlex": "hip_l_flex", "hipLAbduct": "hip_l_abduct",
            "hipLRotate": "hip_l_rotate",
            "kneeRFlex": "knee_r_flex", "kneeLFlex": "knee_l_flex",
            "ankleRFlex": "ankle_r_flex", "ankleRInvert": "ankle_r_invert",
            "ankleLFlex": "ankle_l_flex", "ankleLInvert": "ankle_l_invert",
            "breathDepth": "breath_depth", "breathRate": "breath_rate",
            "breathPhaseBody": "breath_phase_body", "autoBreathBody": "auto_breath_body",
            "fingerCurlR": "finger_curl_r", "fingerSpreadR": "finger_spread_r",
            "thumbOpR": "thumb_op_r",
            "fingerCurlL": "finger_curl_l", "fingerSpreadL": "finger_spread_l",
            "thumbOpL": "thumb_op_l",
            "toeCurlR": "toe_curl_r", "toeSpreadR": "toe_spread_r",
            "toeCurlL": "toe_curl_l", "toeSpreadL": "toe_spread_l",
        }

    def set_from_js_dict(self, d: dict[str, float]) -> None:
        """Set values from a dict using JS camelCase keys."""
        for js_key, value in d.items():
            py_key = self._JS_KEY_MAP.get(js_key, js_key)
            if hasattr(self, py_key):
                setattr(self, py_key, value)

    def to_dict(self) -> dict[str, float]:
        """Return all numeric fields as a dict."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name != "_JS_KEY_MAP" and isinstance(getattr(self, f.name), (int, float))
        }


@dataclass
class ConstraintState:
    """Neck constraint solver state."""
    attachments: list = field(default_factory=list)
    tensions: list = field(default_factory=list)
    total_excess: float = 0.0
    # Smoothed total_excess to prevent frame-to-frame oscillation / jitter.
    # Head rotation soft-clamp reads this instead of raw total_excess.
    smoothed_total_excess: float = 0.0
    spine_compensation_yaw: float = 0.0
    spine_compensation_pitch: float = 0.0
    spine_compensation_roll: float = 0.0
    spine_compensation_magnitude: float = 0.0


class StateManager:
    """Central state container."""

    def __init__(self):
        self.face = FaceState()
        self.target_au = TargetAU()
        self.target_head = TargetHead()
        self.target_ear_wiggle: float = 0.0
        self.body = BodyState()
        self.target_body = BodyState()
        self.constraints = ConstraintState()

        # Render modes
        self.render_modes: dict[str, str] = {"skull": "wireframe", "face": "wireframe"}

        # Frame stats
        self.frame_count: int = 0
        self.fps: float = 0.0
