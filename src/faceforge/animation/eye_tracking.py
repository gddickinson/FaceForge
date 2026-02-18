"""Mouse-follow eye positioning."""

from faceforge.core.state import FaceState


class EyeTracking:
    """Updates eye look direction based on mouse/cursor position.

    Maps normalized screen coordinates (-1..1) to eyeLookX/Y targets.
    """

    SENSITIVITY = 0.8
    SMOOTHING = 6.0

    def __init__(self):
        self._target_x = 0.0
        self._target_y = 0.0

    def set_cursor_position(self, norm_x: float, norm_y: float) -> None:
        """Set normalized cursor position (-1..1 from viewport center)."""
        self._target_x = max(-1.0, min(1.0, norm_x * self.SENSITIVITY))
        self._target_y = max(-1.0, min(1.0, norm_y * self.SENSITIVITY))

    def update(self, face: FaceState, dt: float) -> None:
        if not face.eye_tracking:
            return

        t = min(1.0, self.SMOOTHING * dt)
        face.eye_look_x += (self._target_x - face.eye_look_x) * t
        face.eye_look_y += (self._target_y - face.eye_look_y) * t
