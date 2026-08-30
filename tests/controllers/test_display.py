"""Render mode, background auto-switching, camera presets, colour, clipping."""

from __future__ import annotations

import pytest

from faceforge.controllers.display import (
    CAMERA_PRESETS, MODE_BACKGROUNDS, DisplayController,
)
from faceforge.coordination.render_mode_sync import (
    apply_current_render_mode, current_render_mode,
)
from faceforge.core.events import EventType
from faceforge.core.material import RenderMode

from tests.controllers.fakes import FakeMesh


@pytest.fixture
def wired(ctx):
    ctx.scene.meshes = [FakeMesh("a"), FakeMesh("b"), FakeMesh("c")]
    controller = DisplayController(ctx)
    controller.subscribe()
    controller.subscribe_late()
    ctx.event_bus.subscribe(EventType.CLIP_PLANE_CHANGED,
                            controller.on_clip_plane_changed)
    return controller


# -- Render mode -----------------------------------------------------------

def test_render_mode_reaches_every_mesh(ctx, wired):
    ctx.event_bus.publish(EventType.RENDER_MODE_CHANGED, mode=RenderMode.XRAY)
    assert [m.material.render_mode for m in ctx.scene.meshes] == \
        [RenderMode.XRAY] * 3


def test_render_mode_reaches_the_themed_environment_only_when_active(ctx, wired):
    ctx.event_bus.publish(EventType.RENDER_MODE_CHANGED, mode=RenderMode.SOLID)
    assert ctx.scene_controller.render_modes == []
    ctx.scene_controller.is_active = True
    ctx.event_bus.publish(EventType.RENDER_MODE_CHANGED, mode=RenderMode.SEPIA)
    assert ctx.scene_controller.render_modes == [RenderMode.SEPIA]


# -- Background auto-switching ---------------------------------------------

def test_stylised_mode_switches_the_background(ctx, wired):
    ctx.event_bus.publish(EventType.RENDER_MODE_CHANGED,
                          mode=RenderMode.BLUEPRINT)
    assert ctx.renderer.CLEAR_COLOR == MODE_BACKGROUNDS[RenderMode.BLUEPRINT]
    assert ctx.renderer._bg_color_dirty is True


def test_leaving_the_stylised_modes_restores_the_original_background(ctx, wired):
    original = ctx.renderer.CLEAR_COLOR
    ctx.event_bus.publish(EventType.RENDER_MODE_CHANGED, mode=RenderMode.THERMAL)
    ctx.event_bus.publish(EventType.RENDER_MODE_CHANGED, mode=RenderMode.SOLID)
    assert ctx.renderer.CLEAR_COLOR == original


def test_the_original_background_is_captured_once_not_per_mode(ctx, wired):
    """Chaining stylised modes must not remember an auto-set colour as the original."""
    original = ctx.renderer.CLEAR_COLOR
    for mode in (RenderMode.BLUEPRINT, RenderMode.THERMAL, RenderMode.SEPIA):
        ctx.event_bus.publish(EventType.RENDER_MODE_CHANGED, mode=mode)
    ctx.event_bus.publish(EventType.RENDER_MODE_CHANGED, mode=RenderMode.SOLID)
    assert ctx.renderer.CLEAR_COLOR == original


def test_a_plain_mode_before_any_stylised_one_leaves_the_background_alone(ctx, wired):
    original = ctx.renderer.CLEAR_COLOR
    ctx.event_bus.publish(EventType.RENDER_MODE_CHANGED, mode=RenderMode.WIREFRAME)
    assert ctx.renderer.CLEAR_COLOR == original
    assert ctx.renderer._bg_color_dirty is False


# -- Camera presets --------------------------------------------------------

def test_camera_preset_sets_eye_and_target_and_resyncs_the_orbit(ctx, wired):
    ctx.event_bus.publish(EventType.CAMERA_PRESET, preset="head_left")
    position, target = CAMERA_PRESETS["head_left"]
    assert ctx.camera.position == position
    assert ctx.camera.target == target
    assert ctx.gl_widget.orbit_controls.resets == 1


def test_unknown_camera_preset_is_ignored(ctx, wired):
    ctx.event_bus.publish(EventType.CAMERA_PRESET, preset="nope")
    assert ctx.camera.position == (0.0, 0.0, 0.0)
    assert ctx.gl_widget.orbit_controls.resets == 0


def test_every_preset_targets_the_body_or_head_centre(wired):
    """Guards the coordinate-system convention documented on the module."""
    for name, (_, target) in CAMERA_PRESETS.items():
        expected = (0, 0, -80) if name.startswith("body") else (0, 0, -5)
        assert target == expected, name


# -- Colour ----------------------------------------------------------------

def test_background_colour_is_opaque(ctx, wired):
    ctx.event_bus.publish(EventType.COLOR_CHANGED, target="background",
                          color=(0.1, 0.2, 0.3))
    assert ctx.renderer.CLEAR_COLOR == (0.1, 0.2, 0.3, 1.0)


def test_skull_colour_matches_by_name_substring(ctx, wired):
    ctx.scene.meshes = [FakeMesh("cranium"), FakeMesh("face"), FakeMesh("jaw")]
    ctx.event_bus.publish(EventType.COLOR_CHANGED, target="skull",
                          color=(0.9, 0.8, 0.7))
    by_name = {m.name: m.material.color for m in ctx.scene.meshes}
    assert by_name["cranium"] == (0.9, 0.8, 0.7)
    assert by_name["jaw"] != (0.9, 0.8, 0.7)


def test_face_colour_matches_by_exact_name(ctx, wired):
    ctx.scene.meshes = [FakeMesh("face"), FakeMesh("face_features")]
    ctx.event_bus.publish(EventType.COLOR_CHANGED, target="face", color=(1, 0, 0))
    assert ctx.scene.meshes[0].material.color == (1, 0, 0)
    assert ctx.scene.meshes[1].material.color != (1, 0, 0)


# -- Clip plane ------------------------------------------------------------

@pytest.mark.parametrize("axis,normal", [
    ("x", (1, 0, 0)), ("y", (0, 1, 0)), ("z", (0, 0, 1)),
])
def test_clip_plane_normal_per_axis(ctx, wired, axis, normal):
    ctx.event_bus.publish(EventType.CLIP_PLANE_CHANGED, enabled=True,
                          axis=axis, offset=3.0, flip=False)
    assert ctx.renderer.clip == (normal, 3.0)


def test_clip_plane_flip_negates_the_normal(ctx, wired):
    ctx.event_bus.publish(EventType.CLIP_PLANE_CHANGED, enabled=True,
                          axis="y", offset=1.0, flip=True)
    assert ctx.renderer.clip == ((0, -1, 0), 1.0)


def test_disabling_the_clip_plane_clears_it(ctx, wired):
    ctx.event_bus.publish(EventType.CLIP_PLANE_CHANGED, enabled=True, axis="x")
    ctx.event_bus.publish(EventType.CLIP_PLANE_CHANGED, enabled=False)
    assert ctx.renderer.clip is None


# -- Render mode sync (used by the on-demand loaders) ----------------------

def test_current_mode_is_read_off_the_scene(ctx):
    ctx.scene.meshes = [FakeMesh("a", RenderMode.SEPIA)]
    assert current_render_mode(ctx.scene) is RenderMode.SEPIA


def test_an_empty_scene_reports_the_startup_mode(ctx):
    assert current_render_mode(ctx.scene) is RenderMode.WIREFRAME


def test_newly_loaded_meshes_adopt_the_mode_on_screen(ctx):
    """A layer ticked in XRAY must not arrive opaque."""
    ctx.scene.meshes = [FakeMesh("existing", RenderMode.XRAY)]
    fresh = [FakeMesh("new1", RenderMode.SOLID), FakeMesh("new2", RenderMode.SOLID)]
    returned = apply_current_render_mode(ctx.scene, fresh)
    assert returned is RenderMode.XRAY
    assert [m.material.render_mode for m in fresh] == [RenderMode.XRAY] * 2
