"""The application entry point: what ``main()`` is, and what wiring produces.

Two things are locked in here.

**The subscription map.**  ``build_controllers`` is run against a context of
stubs -- no ``QApplication``, no GL, no assets -- and the resulting event-bus
subscription counts are asserted exactly.  A handler that stops being wired, or
gets wired twice, changes this map and fails here rather than showing up as a
control that silently does nothing.

**The size of ``main()``.**  ``main()`` was ~2,209 lines holding 101 nested
functions, and the reason the whole application layer had no tests was that
everything lived inside it: there was no importable name to construct.  Summing
radon's cyclomatic complexity over ``main`` and every closure inside it gave
477 before this refactor and 1 after.  (A separate earlier audit of the same
function reported 417; the two numbers come from different tools, and only the
477 was measured here.)  The thresholds below are generous relative to what
``main()`` is now -- they exist to catch logic drifting back into the entry
point, not to police formatting.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

import faceforge.app as app_module
from faceforge.appcontext import AppContext
from faceforge.controllers import build_controllers
from faceforge.core.events import EventBus, EventType
from faceforge.core.state import StateManager

from tests.controllers.fakes import FakeGLWidget, FakeScene, FakeVisibility


class Signal:
    """A Qt-signal-shaped recorder."""

    def __init__(self):
        self.connected: list = []

    def connect(self, slot):
        self.connected.append(slot)


class StubTab:
    def __init__(self):
        self.label_style_changed = Signal()

    def on_structures_loaded(self, **kw):
        """Real method, not a lambda: the no-closures test inspects qualnames."""

    def __getattr__(self, name):
        # The display tab is built from a declarative spec; the controller
        # probes for optional buttons with hasattr.
        raise AttributeError(name)


class StubPanel:
    def __init__(self):
        self.display_tab = StubTab()
        self.layers_tab = StubTab()


class StubWindow:
    def __init__(self):
        self.scanner_requested = Signal()
        self.control_panel = StubPanel()


@pytest.fixture
def wired_ctx():
    scene = FakeScene()
    ctx = AppContext(
        event_bus=EventBus(),
        state=StateManager(),
        scene=scene,
        gl_widget=FakeGLWidget(scene),
        visibility=FakeVisibility(),
        window=StubWindow(),
        anim_player=SimpleNamespace(),
        named_nodes={},
    )
    controllers = build_controllers(ctx)
    return ctx, controllers


# -- The subscription map --------------------------------------------------

#: Event type name -> number of subscribed handlers after wiring.
EXPECTED_SUBSCRIPTIONS = {
    "AU_CHANGED": 1, "EXPRESSION_SET": 1, "HEAD_ROTATION_CHANGED": 1,
    "BODY_STATE_CHANGED": 1, "BODY_POSE_SET": 1,
    "GENDER_CHANGED": 1, "GENDER_RELEASED": 1,
    "LAYER_TOGGLED": 2,          # layer dispatch, then label invalidation
    "AUTO_BLINK_TOGGLED": 1, "AUTO_BREATHING_TOGGLED": 1,
    "EYE_TRACKING_TOGGLED": 1, "MICRO_EXPRESSIONS_TOGGLED": 1,
    "RENDER_MODE_CHANGED": 1, "CAMERA_PRESET": 1,
    "SCENE_MODE_TOGGLED": 1, "SCENE_CAMERA_CHANGED": 1, "SCENE_WRAPPER_NUDGE": 1,
    "ANIM_PLAY": 1, "ANIM_PAUSE": 1, "ANIM_STOP": 1, "ANIM_SEEK": 1,
    "ANIM_SPEED": 1, "ANIM_CLIP_SELECTED": 1,
    "COLOR_CHANGED": 1, "ALIGNMENT_CHANGED": 1, "LABELS_TOGGLED": 1,
    "CLIP_PLANE_CHANGED": 1, "SKULL_MODE_CHANGED": 1, "EYE_COLOR_SET": 1,
    "STRUCTURES_LOADED": 1, "HEATMAP_TOGGLED": 1, "STRUCTURE_SEARCH": 1,
    "SPEECH_PLAY": 1, "PATHOLOGY_CHANGED": 1,
}


def test_the_subscription_map_is_exactly_as_expected(wired_ctx):
    ctx, _ = wired_ctx
    got = {et.name: len(hs) for et, hs in ctx.event_bus._handlers.items() if hs}
    assert got == EXPECTED_SUBSCRIPTIONS


def test_layer_loading_is_subscribed_before_label_invalidation(wired_ctx):
    """Order is observable: the labels must see structures the toggle loaded."""
    ctx, controllers = wired_ctx
    handlers = ctx.event_bus._handlers[EventType.LAYER_TOGGLED]
    assert handlers[0].__self__ is controllers.layers
    assert handlers[1].__self__ is controllers.labels


def test_no_handler_is_a_closure(wired_ctx):
    """The closures were the defect.  Every handler must be a named bound method.

    A closure cannot be imported, constructed or called from a test, which is
    exactly why the application layer had no coverage before this refactor.
    """
    ctx, controllers = wired_ctx
    owned = {id(c) for c in vars(controllers).values()}
    for event_type, handlers in ctx.event_bus._handlers.items():
        for handler in handlers:
            qualname = getattr(handler, "__qualname__", repr(handler))
            assert "<locals>" not in qualname, \
                f"{event_type.name} is handled by a closure: {qualname}"
            owner = getattr(handler, "__self__", None)
            assert owner is not None, \
                f"{event_type.name} handler {qualname} is not a bound method"
            if event_type is not EventType.STRUCTURES_LOADED:
                assert id(owner) in owned, \
                    f"{event_type.name} handler is not owned by a controller"


def test_wiring_needs_no_qapplication_gl_or_assets(wired_ctx):
    """The whole handler layer is constructible from stubs -- the point of this refactor."""
    ctx, controllers = wired_ctx
    assert controllers.loaders is controllers.layers.loaders
    assert ctx.event_bus._handlers


def test_the_animation_player_callbacks_are_bound(wired_ctx):
    ctx, controllers = wired_ctx
    player = ctx.anim_player
    assert player.on_wrapper_transform.__self__ is controllers.animation
    assert player.on_body_state.__self__ is controllers.animation
    assert player.on_face.__self__ is controllers.animation
    assert player.on_complete.__self__ is controllers.animation
    assert not hasattr(player, "on_camera"), \
        "a clip must not take the camera away from the user"


def test_the_scanner_action_is_connected(wired_ctx):
    ctx, controllers = wired_ctx
    assert ctx.window.scanner_requested.connected == [
        controllers.tools.open_scanner]


def test_the_label_style_signal_is_connected(wired_ctx):
    ctx, controllers = wired_ctx
    assert ctx.control_panel.display_tab.label_style_changed.connected == [
        controllers.labels.apply_style]


# -- The shape of the entry point -----------------------------------------

APP_PATH = Path(inspect.getfile(app_module))


def test_main_is_short_enough_to_read_in_one_screen():
    source = inspect.getsource(app_module.main)
    body = [ln for ln in source.splitlines()
            if ln.strip() and not ln.strip().startswith("#")]
    assert len(body) < 30, f"main() has grown back to {len(body)} lines"


def test_main_defines_no_nested_functions():
    """The closures were the problem: nothing in ``main()`` may be a closure again."""
    tree = ast.parse(APP_PATH.read_text())
    main_def = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == "main")
    nested = [n.name for n in ast.walk(main_def)
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
              and n is not main_def]
    assert nested == []


def test_the_whole_module_stays_a_wiring_module():
    tree = ast.parse(APP_PATH.read_text())
    functions = [n for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    branches = [n for n in ast.walk(tree)
                if isinstance(n, (ast.If, ast.For, ast.While, ast.Try,
                                  ast.ExceptHandler))]
    assert len(functions) <= 10, f"app.py defines {len(functions)} functions"
    # The only branch in the module is the ``__main__`` guard.
    assert len(branches) <= 2, f"app.py has {len(branches)} branch points"


def test_the_entry_point_is_still_the_packaged_one():
    """``pyproject.toml`` ships ``faceforge = faceforge.app:main``."""
    assert callable(app_module.main)
    assert inspect.signature(app_module.main).parameters == {}
