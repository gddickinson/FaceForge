"""The exercise tab's "Save OBJ and view" button, from event to file."""

from __future__ import annotations

import numpy as np
import pytest

import faceforge.constants as constants
import faceforge.ui.obj_viewer as obj_viewer
from faceforge.controllers.exercise import ExerciseController
from faceforge.core.events import EventBus, EventType
from faceforge.core.material import Material
from faceforge.core.mesh import BufferGeometry, MeshInstance
from faceforge.core.scene_graph import Scene, SceneNode


def _triangle_scene(extra_node: str | None = None) -> Scene:
    geom = BufferGeometry(
        positions=np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32),
        normals=np.array([0, 0, 1] * 3, dtype=np.float32),
        indices=np.array([0, 1, 2], dtype=np.uint32),
    )
    scene = Scene()
    node = SceneNode("body")
    node.mesh = MeshInstance(name="body", geometry=geom, material=Material.from_hex(0x888888))
    scene.add(node)
    if extra_node:
        room = SceneNode(extra_node)
        room.mesh = MeshInstance(name="floor", geometry=geom,
                                 material=Material.from_hex(0x222222))
        scene.add(room)
    scene.update()
    return scene


class _Ctx:
    def __init__(self, scene):
        self.event_bus = EventBus()
        self.scene = scene
        self.simulation = None
        self.animation_player = None


@pytest.fixture
def exported(tmp_path, monkeypatch):
    """Run the handler with PROJECT_ROOT redirected and the viewer stubbed."""
    monkeypatch.setattr(constants, "PROJECT_ROOT", tmp_path)
    opened: list = []
    monkeypatch.setattr(obj_viewer, "launch", lambda p: opened.append(p))

    def run(scene, on_progress=None):
        ctx = _Ctx(scene)
        got: dict = {}
        ctx.event_bus.subscribe(EventType.EXERCISE_EXPORTED, lambda **kw: got.update(kw))
        ExerciseController(ctx).on_export_obj(on_progress=on_progress)
        return got, opened
    return run


def test_it_writes_an_obj_and_opens_a_viewer_on_it(exported):
    got, opened = exported(_triangle_scene())
    assert got["ok"] is True
    path = constants.PROJECT_ROOT / "results" / "exercise_obj"
    assert (files := list(path.glob("*.obj")))
    assert files[0].read_text().count("\nv ") == 3
    assert opened == [files[0]], "the viewer is launched on the file just written"


def test_the_room_is_not_exported_with_the_athlete(exported):
    """`export_mesh` writes everything visible, and in scene mode that is the gym."""
    with_room, _ = exported(_triangle_scene(extra_node="scene_env_root"))
    assert "2 meshes" not in with_room["message"]
    assert "1 meshes" in with_room["message"]


def test_a_scene_with_nothing_in_it_reports_a_failure_rather_than_an_empty_file(exported):
    got, opened = exported(Scene())
    assert got["ok"] is False
    assert "failed" in got["message"].lower()
    assert opened == [], "nothing to view, so nothing is launched"


def test_the_progress_callback_reaches_the_bar(exported):
    """The tab's bar only moves because the exporter calls back into it."""
    seen: list[tuple[int, int, str]] = []
    got, _ = exported(_triangle_scene(), on_progress=lambda d, t, s: seen.append((d, t, s)))
    assert got["ok"] is True
    assert seen, "no progress was reported at all"
    assert seen[-1][0] == seen[-1][1], "the last report is not 100%"
    assert [s for _, _, s in seen] == sorted(
        [s for _, _, s in seen], key=["baking", "writing", "written"].index)
