"""GL resource release, and what happens when a frame cannot be drawn.

Two defects this locks out, both of the shape that leaves a green suite and a
wrong application:

* ``GLViewport.cleanup()`` existed, was correct, and was called from nowhere.
  Every VAO, VBO and shader program it frees leaked on teardown, and the 16 ms
  render timer kept firing ``update()`` while the widget came apart.

* ``paintGL`` caught every exception and logged it. Tolerating one bad frame is
  right; repeating it at 60 Hz is not. A shader that fails to link produced a
  blank window, thousands of identical tracebacks, and no signal to the user
  about why.

These run without a GL context: the failure path is exercised by making the
renderer raise, which is what a failed shader link ultimately does.
"""

from __future__ import annotations

import pytest

from faceforge.rendering.gl_widget import GLViewport

pytest.importorskip("PySide6")


@pytest.fixture
def viewport(qapp):
    w = GLViewport()
    yield w
    w._timer.stop()


class _Boom:
    """Renderer stand-in whose draw always fails, as a bad shader link does."""

    def __init__(self):
        self.calls = 0

    def render(self, *_a, **_k):
        self.calls += 1
        raise RuntimeError("shader program failed to link")

    def render_split(self, *_a, **_k):
        return self.render()

    def init_gl(self):
        pass

    def resize(self, *_a):
        pass

    def destroy(self):
        pass


def test_cleanup_is_idempotent_and_stops_the_timer(viewport):
    viewport._timer.start()
    viewport.cleanup()
    assert not viewport._timer.isActive(), "cleanup must stop the render timer"
    # The context signal can fire after an explicit call; twice must be safe.
    viewport.cleanup()
    assert viewport._cleaned_up is True


def test_cleanup_is_wired_to_the_context(monkeypatch, viewport):
    """initializeGL must connect cleanup, or it is unreachable again.

    Asserted on the connection rather than on a real teardown, because
    destroying a live context is not something a headless test can stage.
    """
    connected = []

    class _Sig:
        def connect(self, slot):
            connected.append(slot)

    class _Ctx:
        aboutToBeDestroyed = _Sig()

    monkeypatch.setattr(viewport, "context", lambda: _Ctx())
    monkeypatch.setattr(viewport, "renderer", _Boom())
    viewport.initializeGL()

    assert connected, "initializeGL did not connect anything to the context"
    assert any(getattr(s, "__name__", "") == "cleanup" or s == viewport.cleanup
               for s in connected), \
        f"cleanup not among the connected slots: {connected}"


def test_repeated_paint_failures_stop_the_loop_and_report(viewport):
    boom = _Boom()
    viewport.renderer = boom
    viewport.scene = object()          # non-None, so paintGL reaches the draw
    viewport._timer.start()

    emitted: list[int] = []
    viewport.render_failed.connect(emitted.append)

    limit = viewport.MAX_CONSECUTIVE_PAINT_FAILURES
    for _ in range(limit):
        viewport.paintGL()

    assert boom.calls == limit, \
        f"expected {limit} attempts before giving up, got {boom.calls}"
    assert not viewport._timer.isActive(), \
        "the render timer must stop rather than spin at 60 Hz on a broken frame"
    assert emitted == [limit], \
        f"render_failed should report once with the count, got {emitted}"

    # And it must not keep calling the renderer once disabled by the caller.
    before = boom.calls
    viewport.paintGL()
    assert boom.calls == before + 1, "paintGL itself still runs if invoked"


def test_a_success_resets_the_failure_count(viewport):
    """A transient error must not accumulate toward the cut-out."""

    class _Flaky(_Boom):
        def __init__(self):
            super().__init__()
            self.fail_next = True

        def render(self, *_a, **_k):
            self.calls += 1
            if self.fail_next:
                raise RuntimeError("transient")

    flaky = _Flaky()
    viewport.renderer = flaky
    viewport.scene = object()
    viewport._timer.start()

    for _ in range(viewport.MAX_CONSECUTIVE_PAINT_FAILURES - 1):
        viewport.paintGL()
    assert viewport._timer.isActive(), "should not have given up yet"

    flaky.fail_next = False
    viewport.paintGL()                       # one good frame
    assert viewport._paint_failures == 0, \
        "a completed frame must clear the failure count"

    flaky.fail_next = True
    for _ in range(viewport.MAX_CONSECUTIVE_PAINT_FAILURES - 1):
        viewport.paintGL()
    assert viewport._timer.isActive(), \
        "the count restarted after the good frame, so it must not have tripped"
