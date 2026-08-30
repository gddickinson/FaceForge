"""Display: render mode, background, camera presets, clip plane, skull mode.

Camera presets are absolute eye/target pairs rather than orbit offsets, so a
preset frames the same subject regardless of where the user had orbited to.
The body coordinate system is X=lateral, Y=depth (-Y anterior), Z=vertical with
the head near zero and the feet near -200, which is why the body presets target
(0, 0, -80) -- the body's visual centre -- and the head presets (0, 0, -5).
"""

from __future__ import annotations

import logging
from typing import Any

from faceforge.constants import set_jaw_pivot
from faceforge.coordination.render_mode_sync import apply_current_render_mode
from faceforge.core.events import EventType
from faceforge.core.material import RenderMode

logger = logging.getLogger(__name__)


#: Camera presets: ``name -> (eye position, target)``.  See module docstring
#: for the coordinate system these numbers live in.
CAMERA_PRESETS: dict[str, tuple[tuple[float, float, float],
                                tuple[float, float, float]]] = {
    # Body views (radius ~150, target at the body's visual centre)
    "body_front":         ((0, -150, -80),   (0, 0, -80)),
    "body_left":          ((-150, 0, -80),   (0, 0, -80)),
    "body_right":         ((150, 0, -80),    (0, 0, -80)),
    "body_top":           ((0, 0, 80),       (0, 0, -80)),
    "body_back":          ((0, 150, -80),    (0, 0, -80)),
    "body_three_quarter": ((80, -110, -50),  (0, 0, -80)),
    # Head views (radius ~35, target at head centre)
    "head_front":         ((0, -35, -5),     (0, 0, -5)),
    "head_left":          ((-35, 0, -5),     (0, 0, -5)),
    "head_right":         ((35, 0, -5),      (0, 0, -5)),
    "head_top":           ((0, 0, 30),       (0, 0, -5)),
    "head_back":          ((0, 35, -5),      (0, 0, -5)),
    "head_three_quarter": ((18, -26, 3),     (0, 0, -5)),
}


#: Background colour each stylised render mode switches to.  A mode absent from
#: this table keeps whatever background is set, and leaving a listed mode
#: restores the colour that was in force before the first one was entered.
MODE_BACKGROUNDS: dict[RenderMode, tuple[float, float, float, float]] = {
    RenderMode.ILLUSTRATION: (0.96, 0.94, 0.90, 1.0),  # warm paper
    RenderMode.SEPIA:        (0.92, 0.86, 0.74, 1.0),  # aged parchment
    RenderMode.COLOR_ATLAS:  (0.96, 0.94, 0.90, 1.0),  # paper
    RenderMode.PEN_INK:      (1.00, 1.00, 1.00, 1.0),  # pure white
    RenderMode.MEDICAL:      (0.12, 0.14, 0.18, 1.0),  # dark slate
    RenderMode.HOLOGRAM:     (0.02, 0.03, 0.06, 1.0),  # near-black
    RenderMode.CARTOON:      (0.18, 0.20, 0.25, 1.0),  # medium dark
    RenderMode.PORCELAIN:    (0.88, 0.87, 0.85, 1.0),  # soft grey
    RenderMode.BLUEPRINT:    (0.05, 0.12, 0.28, 1.0),  # blueprint blue
    RenderMode.THERMAL:      (0.02, 0.02, 0.04, 1.0),  # near-black
    RenderMode.ETHEREAL:     (0.04, 0.02, 0.08, 1.0),  # deep purple-black
}

#: Clip plane axis -> plane normal.
CLIP_NORMALS = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}


class DisplayController:
    """Handlers for how the scene is drawn, and from where."""

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        #: Background colour to restore when leaving the stylised modes, or
        #: ``None`` when no stylised mode has been entered.
        self.previous_background: tuple | None = None

    def subscribe(self) -> None:
        bus = self.ctx.event_bus
        bus.subscribe(EventType.RENDER_MODE_CHANGED, self.on_render_mode_changed)
        bus.subscribe(EventType.CAMERA_PRESET, self.on_camera_preset)

    def subscribe_late(self) -> None:
        """Colour and clip-plane handlers, subscribed after the animation ones.

        Split to preserve the original subscription order.
        """
        bus = self.ctx.event_bus
        bus.subscribe(EventType.COLOR_CHANGED, self.on_color_changed)

    # -- Render mode -------------------------------------------------------

    def on_render_mode_changed(self,
                               mode: RenderMode = RenderMode.WIREFRAME,
                               **kw) -> None:
        """Switch every mesh to *mode* and auto-switch the background.

        The mode is a per-material property, so it is written to every mesh in
        the scene plus, when a themed environment is active, that environment's
        own meshes.
        """
        for mesh, _ in self.ctx.scene.collect_meshes():
            mesh.material.render_mode = mode
        if self.ctx.scene_controller.is_active:
            self.ctx.scene_controller.set_render_mode(mode)
        self.apply_mode_background(mode)

    def apply_mode_background(self, mode: RenderMode) -> None:
        """Set (or restore) the clear colour for *mode*."""
        renderer = self.ctx.renderer
        if renderer is None:
            return
        auto_bg = MODE_BACKGROUNDS.get(mode)
        if auto_bg is not None:
            if self.previous_background is None:
                self.previous_background = renderer.CLEAR_COLOR
            renderer.CLEAR_COLOR = auto_bg
            renderer._bg_color_dirty = True
        elif self.previous_background is not None:
            renderer.CLEAR_COLOR = self.previous_background
            renderer._bg_color_dirty = True
            self.previous_background = None

    # -- Camera ------------------------------------------------------------

    def on_camera_preset(self, preset: str = "", **kw) -> None:
        cam_data = CAMERA_PRESETS.get(preset)
        if cam_data is None:
            return
        position, target = cam_data
        gl = self.ctx.gl_widget
        gl.camera.set_position(*position)
        gl.camera.set_target(*target)
        gl.orbit_controls.reset_from_camera()

    # -- Colour ------------------------------------------------------------

    def on_color_changed(self, target: str = "",
                         color: tuple = (0.8, 0.8, 0.8), **kw) -> None:
        """Recolour the background, or one named group of meshes.

        Mesh targets are matched by name because the colour picker offers
        anatomical groups ("skull", "face") rather than scene-graph paths.
        """
        if target == "background":
            renderer = self.ctx.renderer
            renderer.CLEAR_COLOR = (*color, 1.0)
            renderer._bg_color_dirty = True
            return
        for mesh, _ in self.ctx.scene.collect_meshes():
            if target == "skull" and mesh.name and "cranium" in mesh.name:
                mesh.material.color = color
            elif target == "face" and mesh.name == "face":
                mesh.material.color = color

    # -- Clip plane --------------------------------------------------------

    def on_clip_plane_changed(self, enabled: bool = False, axis: str = "x",
                              offset: float = 0.0, flip: bool = False,
                              **kw) -> None:
        renderer = self.ctx.renderer
        if not enabled:
            renderer.clear_clip_plane()
            return
        normal = list(CLIP_NORMALS.get(axis, (1, 0, 0)))
        if flip:
            normal = [-n for n in normal]
        renderer.set_clip_plane(tuple(normal), offset)

    # -- Skull mode --------------------------------------------------------

    def on_skull_mode_changed(self, mode: str = "original", **kw) -> None:
        """Rebuild the skull hierarchy in the requested mode.

        The skull is rebuilt rather than re-parented because "original" and
        "separated" are different hierarchies with a different jaw pivot, and
        the pivot has to be pushed back into three places that cached it: the
        global constant, the jaw muscle system and the head rotation system.
        """
        from faceforge.anatomy.skull import build_skull, get_jaw_pivot_node

        skull_grp = self.ctx.node("skullGroup")
        if skull_grp is None:
            return

        for child in list(skull_grp.children):
            skull_grp.remove(child)

        new_skull, new_meshes, new_pivot = build_skull(self.ctx.assets, mode=mode)
        for child in list(new_skull.children):
            new_skull.remove(child)
            skull_grp.add(child)

        pipeline = self.ctx.pipeline
        pipeline.skull_meshes = new_meshes
        pipeline.skull_mode = mode
        pipeline.jaw_pivot = new_pivot

        set_jaw_pivot(*new_pivot)
        if pipeline.jaw_muscles is not None:
            pipeline.jaw_muscles.set_jaw_pivot(*new_pivot)
        if pipeline.head_rotation is not None:
            pipeline.head_rotation.set_head_pivot(*new_pivot)

        self.ctx.simulation.jaw_pivot_node = get_jaw_pivot_node(skull_grp)

        apply_current_render_mode(
            self.ctx.scene,
            [m for m in new_meshes.values() if m is not None])

        logger.info("Skull mode switched to: %s, pivot: %s", mode, new_pivot)
