"""ExerciseController on the stub AppContext: what selecting an exercise does."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from faceforge.body.muscle_activation import MuscleActivationSystem
from faceforge.controllers.exercise import ExerciseController
from faceforge.core.events import EventType
from faceforge.scene.scene_animation import AnimationPlayer

from tests.body.test_ground_contact import _rig


class _Transport:
    def __init__(self):
        self.duration = None
        self.playing = None

    def set_duration(self, d):
        self.duration = d

    def set_playing(self, p):
        self.playing = p


class _DisplayTab:
    def __init__(self):
        self.transport = _Transport()
        self.synced = []

    def sync_scene_state(self, active, scene_type):
        self.synced.append((active, scene_type))


class _LayersTab:
    def __init__(self):
        self.visible = []

    def set_layer_visible(self, layer, visible):
        self.visible.append((layer, visible))


@pytest.fixture
def exercise_ctx(ctx):
    scene, wrapper, pivots = _rig()
    ctx.scene = scene
    ctx.scene_controller = SimpleNamespace(
        is_active=False, wrapper_node=wrapper, presets=[], targets=[],
        set_camera_preset=lambda cam, name, target=None: ctx.scene_controller.presets.append(name),
        set_camera_target_override=lambda target: ctx.scene_controller.targets.append(target))
    ctx.pipeline = SimpleNamespace(joint_setup=SimpleNamespace(pivots=pivots, joint_positions={}))
    ctx.anim_player = AnimationPlayer()
    ctx.anim_player.on_body_state = lambda d: ctx.state.target_body.set_from_js_dict(d)
    ctx.anim_player.on_wrapper_transform = lambda p, q: None
    ctx.muscle_activation = MuscleActivationSystem(dof_map={})
    ctx.simulation = SimpleNamespace(after_animation_hooks=[], after_scene_update_hooks=[])
    ctx.window = SimpleNamespace(control_panel=SimpleNamespace(display_tab=_DisplayTab(),
                                                               layers_tab=_LayersTab()))
    events = []
    ctx.event_bus.subscribe(EventType.SCENE_MODE_TOGGLED, lambda **kw: events.append(("scene", kw)))
    ctx.event_bus.subscribe(EventType.LAYER_TOGGLED, lambda **kw: events.append(("layer", kw)))
    ctx.event_bus.subscribe(EventType.EXERCISE_STATUS, lambda **kw: events.append(("status", kw)))
    controller = ExerciseController(ctx)
    controller.subscribe()
    return ctx, controller, events


def test_selecting_enters_the_gym_loads_muscles_and_starts(exercise_ctx):
    ctx, controller, events = exercise_ctx
    ctx.event_bus.publish(EventType.EXERCISE_SELECTED, exercise_id="bodyweight_squat", reps=2)
    scene_events = [kw for kind, kw in events if kind == "scene"]
    assert scene_events == [{"enabled": True, "scene_type": "gym"}]
    layers = [kw["layer"] for kind, kw in events if kind == "layer"]
    # The movers' regions plus the implied stance stabilisers (foot intrinsics).
    assert layers == ["back_muscles", "torso_muscles", "hip_muscles", "leg_muscles", "foot_muscles"]
    assert ctx.control_panel.display_tab.synced == [(True, "gym")]
    assert controller.runtime.active and ctx.anim_player.is_playing
    assert ctx.control_panel.display_tab.transport.playing is True
    assert ctx.muscle_activation.enabled
    assert ctx.scene_controller.presets == ["three_quarter"]
    assert controller.runtime.after_animation in ctx.simulation.after_animation_hooks
    assert controller.runtime.after_scene_update in ctx.simulation.after_scene_update_hooks


def test_the_presets_look_where_the_exercise_looks_until_it_stops(exercise_ctx):
    ctx, controller, events = exercise_ctx
    ctx.event_bus.publish(EventType.EXERCISE_SELECTED, exercise_id="pull_up")
    # The pull-up hangs the body: its definition carries a look-at override,
    # and the view buttons (SCENE_CAMERA_CHANGED) apply presets through it.
    assert ctx.scene_controller.targets == [(0.0, 190.0, 0.0)]
    ctx.event_bus.publish(EventType.EXERCISE_STOPPED)
    assert ctx.scene_controller.targets[-1] is None


def test_selecting_when_scene_mode_is_active_does_not_retoggle(exercise_ctx):
    ctx, controller, events = exercise_ctx
    ctx.scene_controller.is_active = True
    ctx.event_bus.publish(EventType.EXERCISE_SELECTED, exercise_id="push_up")
    assert not [kw for kind, kw in events if kind == "scene"]
    assert controller.runtime.active


def test_unknown_exercise_is_ignored(exercise_ctx):
    ctx, controller, events = exercise_ctx
    ctx.event_bus.publish(EventType.EXERCISE_SELECTED, exercise_id="nope")
    assert controller.runtime is None and events == []


def test_update_frame_publishes_status_and_stop_clears_everything(exercise_ctx):
    ctx, controller, events = exercise_ctx
    ctx.event_bus.publish(EventType.EXERCISE_SELECTED, exercise_id="bodyweight_squat")
    controller.update_frame()
    statuses = [kw for kind, kw in events if kind == "status"]
    assert statuses and statuses[-1]["phase"] == "Descent"
    ctx.event_bus.publish(EventType.EXERCISE_STOPPED)
    assert not controller.runtime.active
    assert ctx.simulation.after_animation_hooks == []
    assert not ctx.muscle_activation.enabled
    assert [kw for kind, kw in events if kind == "status"][-1]["exercise_id"] == ""


def test_options_change_palette_heatmap_and_restart_for_reps(exercise_ctx):
    ctx, controller, events = exercise_ctx
    ctx.event_bus.publish(EventType.EXERCISE_OPTION_CHANGED, option="palette", value="thermal")
    assert ctx.muscle_activation.palette == "thermal"
    ctx.event_bus.publish(EventType.EXERCISE_SELECTED, exercise_id="bodyweight_squat", reps=1)
    d1 = ctx.anim_player.duration
    ctx.event_bus.publish(EventType.EXERCISE_OPTION_CHANGED, option="reps", value=3)
    assert ctx.anim_player.duration == pytest.approx(3 * d1)
    ctx.event_bus.publish(EventType.EXERCISE_OPTION_CHANGED, option="heatmap", value=False)
    assert not ctx.muscle_activation.enabled


def test_all_muscles_option_loads_every_body_layer(exercise_ctx):
    from faceforge.exercise.muscle_groups import ALL_MUSCLE_REGIONS

    ctx, controller, events = exercise_ctx
    ctx.event_bus.publish(EventType.EXERCISE_OPTION_CHANGED, option="all_muscles", value=True)
    layers = [kw["layer"] for kind, kw in events if kind == "layer"]
    assert layers == list(ALL_MUSCLE_REGIONS)
    events.clear()
    ctx.event_bus.publish(EventType.EXERCISE_SELECTED, exercise_id="pull_up")
    layers = [kw["layer"] for kind, kw in events if kind == "layer"]
    assert layers == list(ALL_MUSCLE_REGIONS), "selecting an exercise keeps the whole body loaded"
