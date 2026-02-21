"""Mouse-driven orbit, pan, and zoom controls using spherical coordinates."""

import math

import numpy as np

from faceforge.core.math_utils import Vec3, normalize, vec3, clamp
from faceforge.rendering.camera import Camera


class OrbitControls:
    """Orbits the camera around a target point.

    Uses spherical coordinates (theta, phi, radius) with optional damping
    and auto-rotation.

    Parameters
    ----------
    camera : Camera
        The camera whose position will be updated.
    """

    # Mouse button constants
    BUTTON_LEFT = 1
    BUTTON_MIDDLE = 2
    BUTTON_RIGHT = 3

    def __init__(self, camera: Camera) -> None:
        self.camera = camera
        self.target: Vec3 = camera.target.copy()

        # Spherical coordinates relative to target
        self._theta: float = 0.0   # horizontal angle (radians)
        self._phi: float = math.pi / 2  # vertical angle (radians)
        self._radius: float = 0.0

        # Limits
        self.min_radius: float = 1.0
        self.max_radius: float = 500.0
        self.min_phi: float = 0.05  # avoid gimbal lock at poles
        self.max_phi: float = math.pi - 0.05

        # Sensitivity
        self.rotate_speed: float = 0.005
        self.pan_speed: float = 0.05
        self.zoom_speed: float = 0.1

        # Damping (0 = no damping, 1 = full freeze)
        self.damping: float = 0.1
        self._damping_theta: float = 0.0
        self._damping_phi: float = 0.0

        # Auto-rotation (degrees per second, 0 = disabled)
        self.auto_rotate: float = 0.0

        # Interaction state
        self._active_button: int | None = None
        self._last_x: float = 0.0
        self._last_y: float = 0.0

        # Initialise spherical coords from current camera position
        self._sync_from_camera()

    # ------------------------------------------------------------------
    # Mouse event handlers
    # ------------------------------------------------------------------

    def on_mouse_press(self, x: float, y: float, button: int) -> None:
        """Begin an orbit (left), pan (right), or zoom (middle) drag."""
        self._active_button = button
        self._last_x = x
        self._last_y = y

    def on_mouse_move(self, x: float, y: float) -> None:
        """Process a mouse-move during an active drag."""
        if self._active_button is None:
            return

        dx = x - self._last_x
        dy = y - self._last_y
        self._last_x = x
        self._last_y = y

        if self._active_button == self.BUTTON_LEFT:
            self._orbit(dx, dy)
        elif self._active_button == self.BUTTON_RIGHT:
            self._pan(dx, dy)
        elif self._active_button == self.BUTTON_MIDDLE:
            self._zoom_drag(dy)

    def on_mouse_release(self) -> None:
        """End the current drag."""
        self._active_button = None

    def on_scroll(self, delta: float) -> None:
        """Zoom in/out via scroll wheel. Positive *delta* zooms in."""
        factor = 1.0 - delta * self.zoom_speed
        self._radius = clamp(self._radius * factor, self.min_radius, self.max_radius)
        self._apply_to_camera()

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(self, dt: float = 1.0 / 60.0) -> None:
        """Apply damping and auto-rotation. Call once per frame."""
        if self.auto_rotate != 0.0:
            self._theta += math.radians(self.auto_rotate) * dt
            self._apply_to_camera()

        # Damping (decay towards zero angular velocity)
        if self.damping > 0:
            self._damping_theta *= (1.0 - self.damping)
            self._damping_phi *= (1.0 - self.damping)
            if abs(self._damping_theta) > 1e-6 or abs(self._damping_phi) > 1e-6:
                self._theta += self._damping_theta
                self._phi = clamp(
                    self._phi + self._damping_phi, self.min_phi, self.max_phi,
                )
                self._apply_to_camera()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset to initial camera position."""
        from faceforge.constants import DEFAULT_CAMERA_POS, DEFAULT_CAMERA_TARGET

        self.camera.position = vec3(*DEFAULT_CAMERA_POS)
        self.camera.target = vec3(*DEFAULT_CAMERA_TARGET)
        self.target = self.camera.target.copy()
        self._sync_from_camera()

    def reset_from_camera(self) -> None:
        """Re-sync orbit state from current camera position/target."""
        self.target = self.camera.target.copy()
        self._damping_theta = 0.0
        self._damping_phi = 0.0
        self._sync_from_camera()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _orbit(self, dx: float, dy: float) -> None:
        d_theta = -dx * self.rotate_speed
        d_phi = -dy * self.rotate_speed

        self._theta += d_theta
        self._phi = clamp(self._phi + d_phi, self.min_phi, self.max_phi)

        self._damping_theta = d_theta
        self._damping_phi = d_phi

        self._apply_to_camera()

    def _pan(self, dx: float, dy: float) -> None:
        """Pan the camera (and target) perpendicular to the view direction."""
        offset = self.camera.position - self.target
        dist = float(np.linalg.norm(offset))

        # Build local right/up vectors from the view matrix
        view = self.camera.get_view_matrix()
        right = vec3(view[0, 0], view[0, 1], view[0, 2])
        up = vec3(view[1, 0], view[1, 1], view[1, 2])

        pan = (-dx * right + dy * up) * self.pan_speed * dist * 0.002
        self.target = self.target + pan
        self._apply_to_camera()

    def _zoom_drag(self, dy: float) -> None:
        factor = 1.0 + dy * 0.005
        self._radius = clamp(self._radius * factor, self.min_radius, self.max_radius)
        self._apply_to_camera()

    def _sync_from_camera(self) -> None:
        """Derive spherical coords from the current camera position/target."""
        offset = self.camera.position - self.target
        self._radius = float(np.linalg.norm(offset))
        if self._radius < 1e-6:
            self._radius = 1.0
            return

        n = offset / self._radius
        self._phi = math.acos(clamp(float(n[1]), -1.0, 1.0))
        self._theta = math.atan2(float(n[0]), float(n[2]))

    def _apply_to_camera(self) -> None:
        """Write spherical coords back to the camera position."""
        sin_phi = math.sin(self._phi)
        x = self._radius * sin_phi * math.sin(self._theta)
        y = self._radius * math.cos(self._phi)
        z = self._radius * sin_phi * math.cos(self._theta)

        self.camera.position = self.target + vec3(x, y, z)
        self.camera.target = self.target.copy()
        self.camera.mark_view_dirty()
