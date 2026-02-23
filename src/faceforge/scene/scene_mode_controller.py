"""Manages scene mode state and body repositioning.

Uses a **wrapper node** approach: a ``scene_wrapper`` SceneNode is inserted
between the scene root and ``bodyRoot``.  The supine rotation + table
positioning is applied to ``scene_wrapper``, keeping ``bodyRoot`` at identity.

The soft tissue skinning system receives a reference to ``scene_wrapper``
and cancels its transform from joint delta matrices, avoiding double-rotation.

Supports two scene types:
  - **examination** (default): body supine on table, side camera
  - **dance_studio**: body standing upright on floor, front camera

Activating scene mode:
  - Creates ``scene_wrapper`` and reparents ``bodyRoot`` under it
  - Applies appropriate quaternion + position to ``scene_wrapper``
  - Parents the environment root into the scene
  - Switches camera to default preset
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

# Standing upright, facing +Z (toward camera) — Rx(-90°)
_Q_STANDING_Z = quat_from_axis_angle(vec3(1, 0, 0), -math.pi / 2)

# Identity quaternion (facing +Z, normal clinical view)
_Q_IDENTITY = quat_identity()

# Body rests on table surface — empirically Y≈105 places body on tabletop
# (TABLE_HEIGHT=90, body back-surface is ~15 units from origin after rotation)
_BODY_TABLE_Y = TABLE_HEIGHT + 15

# After supine rotation, body extends from X=0 (head) to X=170 (feet).
# To center on the table (X=0), offset X by half the body length.
_BODY_CENTER_X = -85.0

# Dance studio: body stands at Y=203 (feet on floor).
# Body Z-axis: head≈0, feet≈-200.  Rx(-90°) maps body Z → world Y.
# Wrapper Y = 200 + ~3 units clearance ≈ 203.
_STAND_Y = 203.0

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

# Dance studio camera presets: body center at roughly (0, 100, 0)
_DANCE_TARGET = (0, 100, 0)
_DANCE_CAMERA_PRESETS: dict[str, tuple[tuple, tuple]] = {
    "front":       ((0, 120, 350),         _DANCE_TARGET),
    "front_wide":  ((0, 140, 450),         _DANCE_TARGET),
    "side_left":   ((-300, 120, 0),        _DANCE_TARGET),
    "side_right":  ((300, 120, 0),         _DANCE_TARGET),
    "overhead":    ((0, 380, 10),          _DANCE_TARGET),
    "corner":      ((220, 150, 250),       _DANCE_TARGET),
    "low_front":   ((0, 30, 320),          _DANCE_TARGET),
}


class SceneModeController:
    """Orchestrates activation / deactivation of the scene environment."""

    def __init__(self) -> None:
        self._active: bool = False
        self._scene_type: str = "examination"
        self._environments: dict[str, SceneEnvironment] = {}
        self._active_environment: SceneEnvironment | None = None

        # Wrapper node that sits between scene and bodyRoot.
        # Carries the supine rotation + table position.
        self._wrapper: SceneNode = SceneNode("scene_wrapper")

        # References saved on activation
        self._scene: Scene | None = None
        self._body_root: SceneNode | None = None

        # Saved state for deactivation restore
        self._saved_cam_pos: np.ndarray | None = None
        self._saved_cam_target: np.ndarray | None = None
        self._saved_cam_up: np.ndarray | None = None

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def scene_type(self) -> str:
        return self._scene_type

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

    def _get_environment(self, scene_type: str) -> SceneEnvironment:
        """Get or create the environment for the given scene type."""
        if scene_type not in self._environments:
            env = SceneEnvironment(scene_type=scene_type)
            env.build()
            self._environments[scene_type] = env
        return self._environments[scene_type]

    def activate(
        self,
        body_root: SceneNode,
        scene: Scene,
        camera: Camera,
        light_setup: LightSetup,
        scene_type: str = "examination",
    ) -> None:
        """Enter scene mode: reparent body under wrapper, build room, enable light."""
        if self._active:
            return

        self._scene = scene
        self._body_root = body_root
        self._scene_type = scene_type

        # Build/retrieve environment
        env = self._get_environment(scene_type)
        self._active_environment = env

        # Parent environment into scene
        scene.add(env.root)

        # Reparent bodyRoot under the wrapper node, then add wrapper to scene.
        if scene_type == "dance_studio":
            # Standing upright facing +Z, feet on floor
            self._wrapper.set_position(0, _STAND_Y, 0)
            self._wrapper.set_quaternion(_Q_STANDING_Z.copy())
            logger.info(
                "Dance studio wrapper: pos=(0, %s, 0), quat=%s",
                _STAND_Y, _Q_STANDING_Z.round(3),
            )
        else:
            # Supine on table
            self._wrapper.set_position(*(_BODY_CENTER_X, _BODY_TABLE_Y, 0))
            self._wrapper.set_quaternion(_Q_SUPINE.copy())
            logger.info(
                "Wrapper transform: pos=(%s, %s, %s), quat=%s",
                _BODY_CENTER_X, _BODY_TABLE_Y, 0,
                _Q_SUPINE.round(3),
            )

        self._wrapper.add(body_root)
        scene.add(self._wrapper)

        # Save camera state
        self._saved_cam_pos = camera.position.copy()
        self._saved_cam_target = camera.target.copy()
        self._saved_cam_up = camera.up.copy()

        # Scene mode uses Y-up (room has floor at Y=0, ceiling at Y=height)
        camera.up = vec3(0.0, 1.0, 0.0)
        camera._view_dirty = True

        # Set camera to default preset
        if scene_type == "dance_studio":
            self.set_camera_preset(camera, "front")
        else:
            self.set_camera_preset(camera, "side")

        # Enable point light
        light_pos = env.get_light_position()
        if light_setup.point_light is None:
            light_setup.point_light = PointLight(
                position=light_pos,
                color=(1.0, 0.95, 0.85),
                intensity=2.0 if scene_type == "dance_studio" else 1.5,
                range=500.0 if scene_type == "dance_studio" else 400.0,
                enabled=True,
            )
        else:
            light_setup.point_light.position = light_pos
            light_setup.point_light.intensity = 2.0 if scene_type == "dance_studio" else 1.5
            light_setup.point_light.range = 500.0 if scene_type == "dance_studio" else 400.0
            light_setup.point_light.enabled = True

        self._active = True
        logger.info("Scene mode activated: %s", scene_type)

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
        if self._active_environment is not None and self._active_environment.root is not None:
            scene.remove(self._active_environment.root)
        self._active_environment = None

        # Reset wrapper to identity for clean state
        self._wrapper.set_position(0, 0, 0)
        self._wrapper.set_quaternion(_Q_IDENTITY.copy())

        # Restore camera
        if self._saved_cam_up is not None:
            camera.up = self._saved_cam_up.copy()
            camera._view_dirty = True
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
        presets = self._get_camera_presets()
        data = presets.get(preset_name)
        if data is None:
            logger.warning("Unknown scene camera preset: %s", preset_name)
            return
        pos, target = data
        camera.set_position(*pos)
        camera.set_target(*target)

    def get_preset_names(self) -> list[str]:
        """Return the list of available scene camera preset names."""
        return list(self._get_camera_presets().keys())

    def _get_camera_presets(self) -> dict[str, tuple[tuple, tuple]]:
        """Return the camera presets for the current scene type."""
        if self._scene_type == "dance_studio":
            return _DANCE_CAMERA_PRESETS
        return _SCENE_CAMERA_PRESETS

    def set_render_mode(self, mode) -> None:
        """Propagate render mode to environment meshes."""
        if self._active_environment is not None:
            self._active_environment.set_render_mode(mode)
