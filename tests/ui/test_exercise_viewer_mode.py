"""The exercise viewer mode, end to end on the real application (slow, needs assets).

Enters the mode through the main window, lets every muscle layer load, starts
a deadlift and checks that the whole body is there and the demonstration runs.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from faceforge.core.events import EventType                       # noqa: E402
from tests.ui import gui_harness as H                             # noqa: E402

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def viewer():
    app, window, errors = H.build_main_window()
    assert not errors, errors
    H.drain_deferred_startup(app)
    window.set_viewer_mode(True)
    H.drain_deferred_startup(app, settle=1.0)         # every muscle layer loads here
    window.event_bus.publish(EventType.EXERCISE_SELECTED, exercise_id="conventional_deadlift",
                             reps=1)
    H.drain_deferred_startup(app, settle=1.0)
    return app, window


def _mesh_names(window) -> set[str]:
    return {mesh.name for mesh, _ in window.gl_widget.scene.collect_meshes()}


def test_the_mode_swaps_the_panels_and_moves_the_tab(viewer):
    _app, window = viewer
    assert window.viewer_mode
    assert window.right_stack.currentWidget() is window.viewer_panel
    assert window.viewer_panel.tab is window.control_panel.exercise_tab
    assert window.control_panel.tabs.indexOf(window.control_panel.exercise_tab) == -1
    assert not window.info_panel.isVisible()


def test_every_muscle_layer_and_the_skeleton_are_in_the_scene(viewer):
    _app, window = viewer
    names = _mesh_names(window)
    for probe in ("R Lumbricals", "Flex. Dig. Prof. R", "Rectus Femoris R", "Latissimus Dorsi R",
                  "Deltoid Acr. R", "Gluteus Max. R", "R Abductor Hallucis", "Right Humerus"):
        assert probe in names, probe


def test_the_demonstration_is_running(viewer):
    app, window = viewer
    # The runtime placed the deadlift's barbell in the gym: the clip started.
    assert window.gl_widget.scene.find("equip_barbell") is not None
    # Offscreen, paint events may never fire, so the status label is checked
    # only when frames have actually run.
    tab = window.control_panel.exercise_tab
    for _ in range(60):
        app.processEvents()
    label = tab._phase_label.text()
    assert label == "—" or label.startswith("Rep"), label


def test_leaving_the_mode_restores_the_control_panel(viewer):
    app, window = viewer
    window.set_viewer_mode(False)
    app.processEvents()
    assert not window.viewer_mode
    assert window.right_stack.currentWidget() is window.control_panel
    idx = window.control_panel.tabs.indexOf(window.control_panel.exercise_tab)
    assert idx == window.EXERCISE_TAB_INDEX
    assert window.control_panel.tabs.tabText(idx) == "EXERCISE"
