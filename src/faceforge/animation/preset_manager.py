"""Expression and body pose preset management."""

from faceforge.core.state import (
    FaceState, BodyState, TargetAU, TargetHead, StateManager, AU_IDS,
)
from faceforge.core.config_loader import load_config


class PresetManager:
    """Manages expression and body pose presets loaded from JSON config."""

    def __init__(self):
        self.expressions: dict[str, dict] = {}
        self.body_poses: dict[str, dict] = {}

    def load(self) -> None:
        """Load presets from config files."""
        self.expressions = load_config("expressions.json")
        self.body_poses = load_config("body_poses.json")

    def get_expression_names(self) -> list[str]:
        return list(self.expressions.keys())

    def get_body_pose_names(self) -> list[str]:
        return list(self.body_poses.keys())

    def set_expression(self, name: str, state: StateManager) -> None:
        """Apply an expression preset to target AU values."""
        preset = self.expressions.get(name)
        if preset is None:
            return

        # Reset all AU targets to 0
        for au_id in AU_IDS:
            state.target_au.set(au_id, 0.0)

        # Apply preset values
        for key, value in preset.items():
            if key in AU_IDS:
                state.target_au.set(key, value)
            elif key == "headYaw":
                state.target_head.head_yaw = value
            elif key == "headPitch":
                state.target_head.head_pitch = value
            elif key == "headRoll":
                state.target_head.head_roll = value

        state.face.current_expression = name

    def set_body_pose(self, name: str, state: StateManager) -> None:
        """Apply a body pose preset to target body values."""
        preset = self.body_poses.get(name)
        if preset is None:
            return

        state.target_body.set_from_js_dict(preset)
