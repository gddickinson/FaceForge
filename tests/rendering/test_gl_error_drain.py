"""The GL error a context arrives with, and who gets blamed for it.

macOS hands back an OpenGL 4.1 core context (over Metal) with a
``GL_INVALID_ENUM`` already sitting in its queue, left there by Qt while it
sets the surface up -- measured in every window shape: a bare widget, a widget
in a main window, and a widget created after a modal dialog.  PyOpenGL checks
``glGetError`` *after* every call and reports whatever it finds, so the flag
was charged to the first call FaceForge made::

    OpenGL.error.GLError(err = 1280, description = b'invalid enumerant',
        baseOperation = glClearColor, cArguments = (0.12, 0.12, 0.15, 1.0))

-- ``glClearColor``, which cannot raise ``GL_INVALID_ENUM`` at all.

The misattribution is the whole bug.  ``init_gl`` died on its first line, so
the shaders were never compiled, ``GLRenderer.render`` returned early forever
on ``_initialised``, the refresh timer never started, and the window stayed a
single flat colour without logging another word.  Measured on the standalone
OBJ viewer: 0 lit pixels before, 402,021 of 3.6 M after.

The drain lives in ``init_gl`` rather than in its callers because every caller
needs it -- the widget, the headless ``Session`` and the capture tools -- and
one that forgot would fail this silently.  These tests run without a GL
context; ``glGetError`` is substituted.
"""

from __future__ import annotations

import pytest

from faceforge.rendering.renderer import GLRenderer

GL_INVALID_ENUM = 1280
GL_INVALID_OPERATION = 1282


@pytest.fixture
def fake_errors(monkeypatch):
    """Queue GL errors for ``glGetError`` to hand back one at a time."""

    def queue(errors):
        pending = list(errors)
        calls = []

        def glGetError():
            calls.append(1)
            return pending.pop(0) if pending else 0

        monkeypatch.setattr("faceforge.rendering.renderer.glGetError", glGetError)
        return calls

    return queue


def test_drain_returns_the_errors_it_cleared(fake_errors):
    fake_errors([GL_INVALID_ENUM, GL_INVALID_OPERATION])
    assert GLRenderer().drain_errors() == [GL_INVALID_ENUM, GL_INVALID_OPERATION]


def test_drain_of_a_clean_queue_is_empty_and_cheap(fake_errors):
    calls = fake_errors([])
    assert GLRenderer().drain_errors() == []
    assert len(calls) == 1          # one poll, not sixteen


def test_drain_stops_at_the_limit(fake_errors):
    """A driver that returns an error unconditionally must not hang the init."""
    calls = fake_errors([GL_INVALID_ENUM] * 1000)
    assert len(GLRenderer().drain_errors(limit=4)) == 4
    assert len(calls) == 4


def test_init_gl_drains_before_it_touches_gl(monkeypatch):
    """Ordering is the point: a drain after ``glClearColor`` would fix nothing.

    A queued error stands in for the one macOS leaves behind, and the GL calls
    are substituted with an error checker of PyOpenGL's own shape -- raise if
    anything is pending afterwards.  Without the drain this test reproduces the
    original traceback; with it, ``init_gl`` completes.
    """
    pending = [GL_INVALID_ENUM]
    order = []

    def glGetError():
        return pending.pop(0) if pending else 0

    def checked(name):
        def call(*_a, **_k):
            order.append(name)
            if pending:
                raise RuntimeError(f"GLError 1280 charged to {name}")
        return call

    monkeypatch.setattr("faceforge.rendering.renderer.glGetError", glGetError)
    for fn in ("glClearColor", "glEnable", "glDepthFunc"):
        monkeypatch.setattr(f"faceforge.rendering.renderer.{fn}", checked(fn))
    renderer = GLRenderer()
    monkeypatch.setattr(renderer, "_compile_shaders", lambda: order.append("shaders"))

    renderer.init_gl()

    assert order[0] == "glClearColor"     # the drain left nothing to trip on
    assert "shaders" in order             # and initialisation ran to the end
    assert renderer._initialised
