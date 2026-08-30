"""Fixtures for the controller tests.  Stub classes live in :mod:`.fakes`."""

from __future__ import annotations

import pytest

from faceforge.appcontext import AppContext
from faceforge.core.events import EventBus
from faceforge.core.state import StateManager

from tests.controllers.fakes import (
    FakeGLWidget, FakeLabelOverlay, FakeScene, FakeSceneModeController,
    FakeVisibility,
)


@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def state():
    return StateManager()


@pytest.fixture
def ctx(bus, state):
    """A minimally populated real ``AppContext``.

    Individual tests attach the extra collaborators they need.  The context is
    the real dataclass, so a field a controller reads that a test forgot to set
    is ``None`` and fails loudly rather than being silently absent.
    """
    scene = FakeScene()
    return AppContext(
        event_bus=bus,
        state=state,
        scene=scene,
        gl_widget=FakeGLWidget(scene),
        visibility=FakeVisibility(),
        scene_controller=FakeSceneModeController(),
        label_overlay=FakeLabelOverlay(),
        named_nodes={},
    )
