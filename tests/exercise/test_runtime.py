"""ExerciseRuntime on stub collaborators: hooks, callbacks, equipment lifecycle."""

from __future__ import annotations

import numpy as np
import pytest

from faceforge.body.muscle_activation import MuscleActivationSystem
from faceforge.core.math_utils import quat_from_axis_angle, vec3
from faceforge.core.scene_graph import Scene, SceneNode
from faceforge.exercise.catalog import get_exercise_catalog
from faceforge.exercise.runtime import ExerciseRuntime
from faceforge.scene.scene_animation import AnimationPlayer

from tests.body.test_ground_contact import _rig


@pytest.fixture(scope="module")
def catalog():
    return get_exercise_catalog()


def _runtime(with_activation=True):
    scene, wrapper, pivots = _rig()
    player = AnimationPlayer()
    seen = {"body": [], "wrapper": []}
    player.on_body_state = lambda d: seen["body"].append(d)
    player.on_wrapper_transform = lambda p, q: (wrapper.set_position(*p) if p else None,
                                                wrapper.set_quaternion(np.array(q)) if q else None,
                                                seen["wrapper"].append((p, q)))
    activation = MuscleActivationSystem(dof_map={}) if with_activation else None
    live = []
    rt = ExerciseRuntime(player=player, scene=scene, wrapper=wrapper, pivots=pivots,
                         joint_positions={}, muscle_activation=activation,
                         apply_live_body=live.append)
    return rt, scene, wrapper, pivots, player, seen, live, activation


def test_start_loads_the_clip_builds_equipment_and_plays(catalog):
    rt, scene, wrapper, pivots, player, seen, live, activation = _runtime()
    built = rt.start(catalog["barbell_back_squat"], reps=2)
    assert rt.active and player.is_playing
    assert player.duration == pytest.approx(built.duration)
    assert [n.name for n in rt.rig.nodes] == ["equip_barbell"]
    assert rt.rig.nodes[0].parent is scene
    assert activation.levels_are_external
    # seek(0) fired the callbacks: the original handler still ran, and the
    # live body was written too.
    assert seen["body"] and live
    assert seen["wrapper"]


def test_after_animation_samples_the_track_into_the_heatmap(catalog):
    rt, *_rest, activation = _runtime()
    rt.start(catalog["bodyweight_squat"], reps=1, autoplay=False)
    rt.player.seek(0.5)
    rt.after_animation(0.0)
    assert activation._levels and max(activation._levels.values()) > 0.3


def test_after_scene_update_reanchors_and_places_equipment(catalog):
    rt, scene, wrapper, pivots, player, *_ = _runtime()
    rt.start(catalog["barbell_back_squat"], reps=1, autoplay=False)
    for side in "RL":
        pivots[f"knee_{side}"].set_quaternion(quat_from_axis_angle(vec3(1, 0, 0), np.pi / 2))
    scene.update()
    rt.after_scene_update()
    lowest = min(n.get_world_position()[1] for k, n in pivots.items()
                 if k.startswith(("toe", "ankle")))
    assert lowest == pytest.approx(rt.ground_lock._target_y, abs=1e-6)
    bar = rt.rig.nodes[0]
    wr = pivots["wrist_R"].get_world_position()
    wl = pivots["wrist_L"].get_world_position()
    assert bar.position[1] == pytest.approx((wr[1] + wl[1]) / 2, abs=1e-6)


def test_status_reports_phase_cues_motions_and_group_levels(catalog):
    rt, *_ = _runtime()
    rt.start(catalog["bodyweight_squat"], reps=1, autoplay=False)
    rt.player.seek(0.2)
    status = rt.status()
    assert status.exercise_id == "bodyweight_squat" and status.rep == 1
    assert status.phase == "Descent" and status.kind == "eccentric"
    assert status.cues and any("Hips" in m for m in status.motions)
    assert 0 < status.group_levels["quadriceps"] <= 1.0
    event = status.as_event()
    assert set(event) >= {"exercise_id", "phase", "kind", "cue", "motions", "levels"}


def test_phase_changed_fires_once_per_boundary(catalog):
    rt, *_ = _runtime()
    rt.start(catalog["bodyweight_squat"], reps=1, autoplay=False)
    assert rt.phase_changed() is True
    assert rt.phase_changed() is False
    rt.player.seek(0.99)
    assert rt.phase_changed() is True


def test_stop_restores_the_callback_removes_equipment_and_releases_levels(catalog):
    rt, scene, wrapper, pivots, player, seen, live, activation = _runtime()
    original = player.on_body_state
    rt.start(catalog["kettlebell_swing"], reps=1)
    assert player.on_body_state is not original
    rt.stop()
    assert player.on_body_state is original
    assert not rt.active and rt.rig.items == []
    assert all(not n.name.startswith("equip_") for n in scene.children)
    assert not activation.levels_are_external
    assert not player.is_playing


def test_equipment_can_be_hidden(catalog):
    rt, *_ = _runtime()
    rt.show_equipment = False
    rt.start(catalog["barbell_back_squat"], reps=1, autoplay=False)
    assert rt.rig.items == []
