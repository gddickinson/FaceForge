"""Manages scene mode state and body repositioning.

Uses a **wrapper node** approach: a ``scene_wrapper`` SceneNode is inserted
between the scene root and ``bodyRoot``.  The supine rotation + table
positioning is applied to ``scene_wrapper``, keeping ``bodyRoot`` at identity.

The soft tissue skinning system receives a reference to ``scene_wrapper``
and cancels its transform from joint delta matrices, avoiding double-rotation.

Activating scene mode:
  - Creates ``scene_wrapper`` and reparents ``bodyRoot`` under it
  - Applies supine quaternion + table position to ``scene_wrapper``
  - Parents the environment root into the scene
  - Switches camera to side preset
  - Enables point light

Deactivating restores everything to clinical defaults.
"""

import logging
import math

import numpy as np

from faceforge.core.math_utils import (
    Mat4,
    mat4_compose,
    quat_from_axis_angle,
    quat_multiply,
    quat_identity,
    vec3,
)
from faceforge.core.scene_graph import Scene, SceneNode
from faceforge.rendering.camera import Camera
from faceforge.rendering.lights import LightSetup, PointLight
from faceforge.scene.scene_environment import SceneEnvironment, TABLE_HEIGHT

logger = logging.getLogger(__name__)

# Body standing height: feet near Y=-170, head near Y=0, facing +Z.
# Table: X-axis = length (200), Z-axis = width (80).
# Body should lie along X with head toward +X.
#
# Combined supine quaternion (empirically determined):
#   -90°X (standing→lying) then +90°Y (head→+X) then +90°Z (face-up)
#   Result: (-0.7071, 0, 0.7071, 0)
_Q_SUPINE = quat_multiply(
    quat_from_axis_angle(vec3(0, 0, 1), math.pi / 2),
    quat_multiply(
        quat_from_axis_angle(vec3(0, 1, 0), math.pi / 2),
        quat_from_axis_angle(vec3(1, 0, 0), -math.pi / 2),
    ),
)

# Upright but facing +X (for standing after climbing off table)
_Q_UPRIGHT_X = quat_from_axis_angle(vec3(0, 1, 0), math.pi / 2)

# Identity quaternion (facing +Z, normal clinical view)
_Q_IDENTITY = quat_identity()

# Body rests on table surface — empirically Y≈105 places body on tabletop
# (TABLE_HEIGHT=90, body back-surface is ~15 units from origin after rotation)
_BODY_TABLE_Y = TABLE_HEIGHT + 15

# After supine rotation, body extends from X=0 (head) to X=170 (feet).
# To center on the table (X=0), offset X by half the body length.
_BODY_CENTER_X = -85.0

# Scene camera presets: (position, target)
# Body center is at (_BODY_CENTER_X, _BODY_TABLE_Y, 0) after supine rotation.
_BODY_TARGET = (_BODY_CENTER_X, _BODY_TABLE_Y, 0)
_SCENE_CAMERA_PRESETS: dict[str, tuple[tuple, tuple]] = {
    "overhead":  ((_BODY_CENTER_X, 220, 5),               _BODY_TARGET),
    "side":      ((_BODY_CENTER_X, 140, 180),             _BODY_TARGET),
    "head_end":  ((_BODY_CENTER_X + 120, 140, 0),         _BODY_TARGET),
    "foot_end":  ((_BODY_CENTER_X - 120, 140, 0),         _BODY_TARGET),
    "corner":    ((_BODY_CENTER_X + 80, 160, 120),        _BODY_TARGET),
}


class SceneModeController:
    """Orchestrates activation / deactivation of the scene environment."""

    def __init__(self) -> None:
        self._active: bool = False
        self._environment: SceneEnvironment = SceneEnvironment()
        self._env_built: bool = False

        # Wrapper node that sits between scene and bodyRoot.
        # Carries the supine rotation + table position.
        self._wrapper: SceneNode = SceneNode("scene_wrapper")

        # References saved on activation
        self._scene: Scene | None = None
        self._body_root: SceneNode | None = None

        # Saved state for deactivation restore
        self._saved_cam_pos: np.ndarray | None = None
        self._saved_cam_target: np.ndarray | None = None

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def wrapper_node(self) -> SceneNode:
        """The wrapper node (for soft tissue wrapper cancellation)."""
        return self._wrapper

    @property
    def scene_transform(self) -> Mat4 | None:
        """Legacy: returns None. Renderer-level transform is no longer used."""
        return None

    # ------------------------------------------------------------------
    # Transform API (for animation player)
    # ------------------------------------------------------------------

    def set_wrapper_transform(
        self,
        pos: tuple[float, float, float] | None,
        quat: tuple[float, float, float, float] | None,
    ) -> None:
        """Update the wrapper node transform (called by animation player)."""
        if pos is not None:
            self._wrapper.set_position(*pos)
        if quat is not None:
            self._wrapper.set_quaternion(np.array(quat, dtype=np.float64))

    # ------------------------------------------------------------------
    # Activation / Deactivation
    # ------------------------------------------------------------------

    def activate(
        self,
        body_root: SceneNode,
        scene: Scene,
        camera: Camera,
        light_setup: LightSetup,
    ) -> None:
        """Enter scene mode: reparent body under wrapper, build room, enable light."""
        if self._active:
            return

        self._scene = scene
        self._body_root = body_root

        # Build environment lazily (once)
        if not self._env_built:
            self._environment.build()
            self._env_built = True

        env_root = self._environment.root

        # Parent environment into scene
        scene.add(env_root)

        # Reparent bodyRoot under the wrapper node, then add wrapper to scene.
        # SceneNode.add() handles removing bodyRoot from its current parent.
        self._wrapper.set_position(*(_BODY_CENTER_X, _BODY_TABLE_Y, 0))
        self._wrapper.set_quaternion(_Q_SUPINE.copy())
        self._wrapper.add(body_root)
        scene.add(self._wrapper)

        logger.info(
            "Wrapper transform: pos=(%s, %s, %s), quat=%s",
            _BODY_CENTER_X, _BODY_TABLE_Y, 0,
            _Q_SUPINE.round(3),
        )

        # Save camera state
        self._saved_cam_pos = camera.position.copy()
        self._saved_cam_target = camera.target.copy()

        # Set camera to side view (overhead is degenerate with Y-up)
        self.set_camera_preset(camera, "side")

        # Enable point light
        light_pos = self._environment.get_light_position()
        if light_setup.point_light is None:
            light_setup.point_light = PointLight(
                position=light_pos,
                color=(1.0, 0.95, 0.85),
                intensity=1.5,
                range=400.0,
                enabled=True,
            )
        else:
            light_setup.point_light.position = light_pos
            light_setup.point_light.enabled = True

        self._active = True
        logger.info("Scene mode activated (wrapper node approach).")

    def deactivate(
        self,
        body_root: SceneNode,
        scene: Scene,
        camera: Camera,
        light_setup: LightSetup,
    ) -> None:
        """Exit scene mode: reparent body back, remove room, clear transform."""
        if not self._active:
            return

        # Reparent bodyRoot back to scene directly (removes from wrapper)
        scene.add(body_root)

        # Remove wrapper and environment from scene
        scene.remove(self._wrapper)
        env_root = self._environment.root
        if env_root is not None:
            scene.remove(env_root)

        # Reset wrapper to identity for clean state
        self._wrapper.set_position(0, 0, 0)
        self._wrapper.set_quaternion(_Q_IDENTITY.copy())

        # Restore camera
        if self._saved_cam_pos is not None:
            camera.set_position(*self._saved_cam_pos)
        if self._saved_cam_target is not None:
            camera.set_target(*self._saved_cam_target)

        # Disable point light
        if light_setup.point_light is not None:
            light_setup.point_light.enabled = False

        self._active = False
        self._scene = None
        self._body_root = None
        logger.info("Scene mode deactivated.")

    # ------------------------------------------------------------------
    # Camera presets
    # ------------------------------------------------------------------

    def set_camera_preset(self, camera: Camera, preset_name: str) -> None:
        """Apply one of the scene camera presets."""
        data = _SCENE_CAMERA_PRESETS.get(preset_name)
        if data is None:
            logger.warning("Unknown scene camera preset: %s", preset_name)
            return
        pos, target = data
        camera.set_position(*pos)
        camera.set_target(*target)

    @staticmethod
    def get_preset_names() -> list[str]:
        """Return the list of available scene camera preset names."""
        return list(_SCENE_CAMERA_PRESETS.keys())

    def set_render_mode(self, mode) -> None:
        """Propagate render mode to environment meshes."""
        self._environment.set_render_mode(mode)
