"""Perspective camera with view and projection matrices."""

import numpy as np

from faceforge.core.math_utils import (
    Mat4,
    Vec3,
    deg_to_rad,
    mat4_identity,
    mat4_look_at,
    mat4_perspective,
    vec3,
)
from faceforge.constants import DEFAULT_CAMERA_POS, DEFAULT_CAMERA_TARGET


class Camera:
    """A perspective camera that produces view and projection matrices.

    Parameters
    ----------
    fov : float
        Vertical field-of-view in degrees.
    near : float
        Near clipping plane distance.
    far : float
        Far clipping plane distance.
    """

    def __init__(
        self,
        fov: float = 50.0,
        near: float = 0.1,
        far: float = 1000.0,
    ) -> None:
        self.fov = fov
        self.near = near
        self.far = far
        self.aspect: float = 1.0

        self.position: Vec3 = vec3(*DEFAULT_CAMERA_POS)
        self.target: Vec3 = vec3(*DEFAULT_CAMERA_TARGET)
        self.up: Vec3 = vec3(0.0, 1.0, 0.0)

        # Cached matrices (recomputed on demand)
        self._view_dirty: bool = True
        self._proj_dirty: bool = True
        self._view: Mat4 = mat4_identity()
        self._proj: Mat4 = mat4_identity()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_aspect(self, width: int, height: int) -> None:
        """Update the aspect ratio from viewport dimensions."""
        if height > 0:
            self.aspect = width / height
            self._proj_dirty = True

    def get_view_matrix(self) -> Mat4:
        """Return the current view (camera) matrix."""
        if self._view_dirty:
            self._view = mat4_look_at(self.position, self.target, self.up)
            self._view_dirty = False
        return self._view

    def get_projection_matrix(self) -> Mat4:
        """Return the current perspective projection matrix."""
        if self._proj_dirty:
            fov_rad = deg_to_rad(self.fov)
            self._proj = mat4_perspective(fov_rad, self.aspect, self.near, self.far)
            self._proj_dirty = False
        return self._proj

    def get_view_projection(self) -> Mat4:
        """Return ``projection @ view``."""
        return self.get_projection_matrix() @ self.get_view_matrix()

    # ------------------------------------------------------------------
    # Convenience mutators (mark view dirty)
    # ------------------------------------------------------------------

    def look_at(self, eye: Vec3, target: Vec3, up: Vec3 | None = None) -> None:
        self.position = eye.copy()
        self.target = target.copy()
        if up is not None:
            self.up = up.copy()
        self._view_dirty = True

    def set_position(self, x: float, y: float, z: float) -> None:
        self.position = vec3(x, y, z)
        self._view_dirty = True

    def set_target(self, x: float, y: float, z: float) -> None:
        self.target = vec3(x, y, z)
        self._view_dirty = True

    def mark_view_dirty(self) -> None:
        """Call after externally modifying ``position`` or ``target``."""
        self._view_dirty = True
