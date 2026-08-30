"""Themed scene mode: reposing the whole body inside an environment.

Scene mode puts the body into a built environment (an examination room, a dance
studio) which means moving the body as a rigid whole -- supine on a table,
upright on a floor -- without disturbing the joint pose the user has set.  That
is done by inserting a *wrapper* node above ``bodyRoot`` and transforming that,
so the body's own hierarchy is untouched.

Two consequences are easy to get wrong and are handled here explicitly:

* **Double rotation.**  Soft-tissue skinning composes joint delta matrices in
  world space, so with the wrapper rotating everything the deltas would apply
  the rotation a second time.  The skinning system is told about the wrapper so
  it can cancel it.
* **Stale matrices on the first frame.**  Activation reparents nodes, so world
  matrices are rebuilt immediately rather than at the next scene update -- and
  again after seeking the auto-loaded clip, because the seek fires callbacks
  that move the wrapper.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from faceforge.core.events import EventType
from faceforge.core.math_utils import quat_from_axis_angle, quat_multiply, vec3

logger = logging.getLogger(__name__)

#: Clip auto-loaded when a scene type is entered.
SCENE_CLIPS = {"dance_studio": "Contemporary"}
DEFAULT_SCENE_CLIP = "Wake Up"

#: Empirically determined default supine placement of the wrapper.
SUPINE_POSITION = (-85.0, 105.0, 0.0)

#: Wrapper nudge axis -> position index / rotation axis.
_POSITION_AXES = {"px": 0, "py": 1, "pz": 2}
_ROTATION_AXES = {"rx": (1, 0, 0), "ry": (0, 1, 0), "rz": (0, 0, 1)}


def supine_quaternion():
    """The default supine orientation, as a quaternion.

    Composed from three axis rotations rather than written as literal
    components so the intent stays readable; the value was found empirically
    by nudging (see :meth:`SceneViewController.on_wrapper_nudge`).
    """
    return quat_multiply(
        quat_from_axis_angle(vec3(0, 0, 1), math.pi / 2),
        quat_multiply(
            quat_from_axis_angle(vec3(0, 1, 0), math.pi / 2),
            quat_from_axis_angle(vec3(1, 0, 0), -math.pi / 2),
        ),
    )


class SceneViewController:
    """Handlers for scene mode, scene cameras and wrapper nudging."""

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx

    def subscribe(self) -> None:
        bus = self.ctx.event_bus
        bus.subscribe(EventType.SCENE_MODE_TOGGLED, self.on_scene_mode_toggled)
        bus.subscribe(EventType.SCENE_CAMERA_CHANGED,
                      self.on_scene_camera_changed)
        bus.subscribe(EventType.SCENE_WRAPPER_NUDGE, self.on_wrapper_nudge)

    # -- Scene mode --------------------------------------------------------

    def on_scene_mode_toggled(self, enabled: bool = False,
                              scene_type: str = "examination", **kw) -> None:
        ctx = self.ctx
        body_root = ctx.node("bodyRoot")
        if body_root is None:
            return
        gl = ctx.gl_widget
        if enabled:
            self._activate(body_root, scene_type)
        else:
            self._deactivate(body_root)
        # The renderer no longer applies a scene transform of its own; the
        # wrapper node carries it.
        gl.renderer.scene_transform = None
        gl.orbit_controls.reset_from_camera()
        existing = ctx.scene.collect_meshes()
        if existing:
            ctx.scene_controller.set_render_mode(
                existing[0][0].material.render_mode)

    def _activate(self, body_root: Any, scene_type: str) -> None:
        ctx = self.ctx
        gl = ctx.gl_widget
        ctx.scene_controller.activate(
            body_root, ctx.scene, gl.camera, gl.lights, scene_type=scene_type)
        skinning = getattr(ctx.simulation, "soft_tissue", None)
        if skinning is not None:
            skinning.scene_wrapper = ctx.scene_controller.wrapper_node
        ctx.scene.update()

        clip_name = SCENE_CLIPS.get(scene_type, DEFAULT_SCENE_CLIP)
        clip = ctx.builtin_clips.get(clip_name)
        if clip is not None:
            ctx.anim_player.load(clip)
            ctx.anim_player.seek(0)
            ctx.control_panel.display_tab.transport.set_duration(clip.duration)
        # seek(0) fired callbacks that may have moved the wrapper.
        ctx.scene.update()

    def _deactivate(self, body_root: Any) -> None:
        ctx = self.ctx
        gl = ctx.gl_widget
        ctx.anim_player.stop()
        ctx.control_panel.display_tab.transport.set_playing(False)
        ctx.scene_controller.deactivate(
            body_root, ctx.scene, gl.camera, gl.lights)
        skinning = getattr(ctx.simulation, "soft_tissue", None)
        if skinning is not None:
            skinning.scene_wrapper = None

    def on_scene_camera_changed(self, preset: str = "", **kw) -> None:
        if not self.ctx.scene_controller.is_active:
            return
        self.ctx.scene_controller.set_camera_preset(self.ctx.camera, preset)
        self.ctx.gl_widget.orbit_controls.reset_from_camera()

    # -- Manual wrapper nudging -------------------------------------------

    def on_wrapper_nudge(self, axis: str = "", delta: float = 0.0,
                         **kw) -> None:
        """Nudge the wrapper transform, for establishing scene placements.

        This is a positioning tool, not a user feature: the supine placement in
        :func:`supine_quaternion` and :data:`SUPINE_POSITION` was found with it.
        It prints the resulting transform because the value being searched for
        is one to paste back into the source.
        """
        controller = self.ctx.scene_controller
        if not controller.is_active:
            return
        wrapper = controller.wrapper_node

        if axis == "reset":
            q = supine_quaternion()
            wrapper.set_position(*SUPINE_POSITION)
            wrapper.set_quaternion(q)
            print(f"[WRAPPER RESET] pos=(-85, 105, 0) quat={q.round(4)}")
        elif axis.startswith("p"):
            pos = wrapper.position.copy()
            pos[_POSITION_AXES[axis]] += delta
            wrapper.set_position(*pos)
            print(f"[WRAPPER POS] {axis}+={delta} → "
                  f"pos=({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
        elif axis.startswith("r"):
            ax_vec = vec3(*_ROTATION_AXES[axis])
            dq = quat_from_axis_angle(ax_vec, math.radians(delta))
            new_q = quat_multiply(dq, wrapper.quaternion)
            wrapper.set_quaternion(new_q)
            print(f"[WRAPPER ROT] {axis}+={delta}° → quat=({new_q[0]:.4f}, "
                  f"{new_q[1]:.4f}, {new_q[2]:.4f}, {new_q[3]:.4f})")

        self.ctx.scene.update()
        print(f"[WRAPPER] world_matrix diag="
              f"{wrapper.world_matrix.diagonal().round(3)}")
        print(f"[WRAPPER] world_matrix pos={wrapper.world_matrix[:3, 3].round(1)}")
