"""Scene camera presets follow a look-at override while a demonstration moves the body."""

from __future__ import annotations

import numpy as np

from faceforge.rendering.camera import Camera
from faceforge.scene.scene_mode_controller import SceneModeController


def _gym_controller() -> SceneModeController:
    ctrl = SceneModeController()
    ctrl._scene_type = "gym"          # the presets are chosen by scene type
    return ctrl


def test_presets_keep_their_offset_from_the_overridden_target():
    ctrl = _gym_controller()
    cam = Camera()
    ctrl.set_camera_preset(cam, "front")
    standing_pos = np.array(cam.position, dtype=float)
    standing_target = np.array(cam.target, dtype=float)

    ctrl.set_camera_target_override((0.0, 190.0, 0.0))     # a hanging body
    assert ctrl.camera_target_override == (0.0, 190.0, 0.0)
    ctrl.set_camera_preset(cam, "front")
    np.testing.assert_allclose(cam.target, [0.0, 190.0, 0.0])
    np.testing.assert_allclose(np.array(cam.position) - np.array(cam.target),
                               standing_pos - standing_target)

    # An explicit target still wins, and clearing the override restores the preset.
    ctrl.set_camera_preset(cam, "front", target=(5.0, 50.0, 0.0))
    np.testing.assert_allclose(cam.target, [5.0, 50.0, 0.0])
    ctrl.set_camera_target_override(None)
    ctrl.set_camera_preset(cam, "front")
    np.testing.assert_allclose(cam.target, standing_target)
    np.testing.assert_allclose(cam.position, standing_pos)
