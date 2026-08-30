"""Stubs for exercising controllers without Qt, GL or anatomy assets.

What is real and what is stubbed, and why
-----------------------------------------
The ``EventBus``, ``StateManager``, ``RenderMode`` and ``AppContext`` used in
these tests are the **real** classes.  They are pure Python with no Qt or GL
dependency, so stubbing them would mean asserting against a copy of the thing
under test -- the tests would still pass if ``StateManager`` renamed a field.

Stubbed here instead: anything that reads a file, uploads to the GPU or needs a
window.  These stubs record what was asked of them, which is what the
assertions are about -- a controller's job is to make the right calls on its
collaborators, in the right order.
"""

from __future__ import annotations

import numpy as np

from faceforge.core.material import RenderMode

#: World matrix handed out by :class:`FakeScene` -- identity, so a transformed
#: point equals the untransformed one and assertions stay readable.
IDENTITY = np.eye(4, dtype=np.float64)


class FakeMaterial:
    def __init__(self, mode=RenderMode.WIREFRAME):
        self.render_mode = mode
        self.color = (0.8, 0.8, 0.8)
        self.opacity = 1.0
        self.vertex_colors_active = False


class FakeGeometry:
    def __init__(self, centre=(0.0, 0.0, 0.0)):
        self._centre = centre
        self.vertex_count = 3
        self.has_indices = True

    def get_bounding_center(self):
        return self._centre


class FakeMesh:
    def __init__(self, name="mesh", mode=RenderMode.WIREFRAME, centre=(0, 0, 0)):
        self.name = name
        self.material = FakeMaterial(mode)
        self.geometry = FakeGeometry(centre)
        self.rest_pose_stored = 0

    def store_rest_pose(self):
        self.rest_pose_stored += 1


class FakeNode:
    """A scene node with just enough graph behaviour for the controllers."""

    def __init__(self, name="node", mesh=None):
        self.name = name
        self.mesh = mesh
        self.children: list[FakeNode] = []
        self.visible = True
        self.forced_world_updates = 0

    def add(self, child):
        self.children.append(child)

    def remove(self, child):
        self.children.remove(child)

    def find(self, name):
        if self.name == name:
            return self
        for child in self.children:
            hit = child.find(name)
            if hit is not None:
                return hit
        return None

    def update_world_matrix(self, force=False):
        self.forced_world_updates += 1


class FakeScene:
    """Collects meshes from a flat list, in registration order."""

    def __init__(self, meshes=()):
        self.meshes = list(meshes)
        self.updates = 0

    def collect_meshes(self):
        return [(m, IDENTITY) for m in self.meshes]

    def update(self):
        self.updates += 1


class FakeVisibility:
    def __init__(self):
        self.registered: list[tuple[str, object]] = []
        self.set_calls: list[tuple[str, bool]] = []

    def register(self, toggle_id, node):
        self.registered.append((toggle_id, node))

    def set_visible(self, toggle_id, visible):
        self.set_calls.append((toggle_id, visible))


class FakeRenderer:
    CLEAR_COLOR = (0.05, 0.05, 0.07, 1.0)

    def __init__(self):
        self.CLEAR_COLOR = FakeRenderer.CLEAR_COLOR
        self._bg_color_dirty = False
        self.clip = "unset"

    def set_clip_plane(self, normal, offset):
        self.clip = (normal, offset)

    def clear_clip_plane(self):
        self.clip = None


class FakeCamera:
    def __init__(self):
        self.position = (0.0, 0.0, 0.0)
        self.target = (0.0, 0.0, 0.0)

    def set_position(self, x, y, z):
        self.position = (x, y, z)

    def set_target(self, x, y, z):
        self.target = (x, y, z)

    def get_view_projection(self):
        return "view_proj"


class FakeOrbit:
    def __init__(self):
        self.resets = 0

    def reset_from_camera(self):
        self.resets += 1


class FakeGLWidget:
    def __init__(self, scene=None):
        self.camera = FakeCamera()
        self.renderer = FakeRenderer()
        self.orbit_controls = FakeOrbit()
        self.scene = scene
        self.lights = object()
        self.mouse_move_callback = None
        self.selection_tool = None
        self.comparison_mode = False


class FakeSceneModeController:
    def __init__(self):
        self.is_active = False
        self.render_modes: list = []

    def set_render_mode(self, mode):
        self.render_modes.append(mode)


class FakeLabelOverlay:
    def __init__(self):
        self._illustration_mode = False
        self.enabled = None
        self.labels = None
        self.view_proj = None
        self.updates = 0

    def set_enabled(self, enabled):
        self.enabled = enabled

    def set_labels(self, labels):
        self.labels = labels

    def set_view_proj(self, vp):
        self.view_proj = vp

    def update(self):
        self.updates += 1
