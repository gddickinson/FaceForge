"""The exercise viewer panel, headless: what it publishes and how it adopts the tab."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QTabWidget          # noqa: E402

from faceforge.core.events import EventBus, EventType             # noqa: E402
from faceforge.core.state import StateManager                     # noqa: E402
from faceforge.ui.exercise_viewer import VIEWS, ExerciseViewerPanel  # noqa: E402
from faceforge.ui.tabs.exercise_tab import ExerciseTab            # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _recorder(bus, event_type):
    got = []
    bus.subscribe(event_type, lambda **kw: got.append(kw))
    return got


def test_view_buttons_publish_gym_camera_presets(qapp):
    bus = EventBus()
    panel = ExerciseViewerPanel(bus)
    cams = _recorder(bus, EventType.SCENE_CAMERA_CHANGED)
    assert panel.view_names == tuple(p for _l, p in VIEWS)
    panel._view_buttons["back"].click()
    panel.select_view("overhead")
    assert [c["preset"] for c in cams] == ["back", "overhead"]


def test_display_toggles_publish_the_option_and_the_skin_layer(qapp):
    bus = EventBus()
    panel = ExerciseViewerPanel(bus)
    opts = _recorder(bus, EventType.EXERCISE_OPTION_CHANGED)
    layers = _recorder(bus, EventType.LAYER_TOGGLED)
    assert panel.all_muscles
    panel.request_all_muscles()
    assert opts[-1] == {"option": "all_muscles", "value": True}
    panel._skin.set_checked(True)
    assert layers[-1] == {"layer": "skin", "visible": True}


def test_the_tab_moves_between_the_control_panel_and_the_viewer(qapp):
    bus = EventBus()
    state = StateManager()
    tab = ExerciseTab(bus, state)
    tabs = QTabWidget()
    tabs.addTab(tab, "EXERCISE")
    panel = ExerciseViewerPanel(bus)
    idx = tabs.indexOf(tab)
    tabs.removeTab(idx)
    panel.adopt_tab(tab)
    assert panel.tab is tab and tab.parent() is not None
    back = panel.release_tab()
    assert back is tab and panel.tab is None
    tabs.insertTab(0, back, "EXERCISE")
    assert tabs.indexOf(tab) == 0


def test_exit_button_emits_the_signal(qapp):
    panel = ExerciseViewerPanel(EventBus())
    fired = []
    panel.exit_requested.connect(lambda: fired.append(True))
    buttons = [w for w in panel.findChildren(type(panel._view_buttons["front"]))
               if w.text().startswith("Back to")]
    assert len(buttons) == 1
    buttons[0].click()
    assert fired == [True]
