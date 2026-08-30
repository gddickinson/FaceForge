"""Fixtures for the session tests.

The one rule worth enforcing centrally: **no test may leak a live Session.**
A process holds at most one (see :mod:`faceforge.session`), so a test that
forgets to close its session does not fail itself -- it fails whichever test
runs next, with a message about a session it never created.  The autouse
fixture below turns that into a failure attributed to the test that caused it.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_leaked_session():
    """Fail the test that leaked a Session, not the next one."""
    from faceforge import session as fs

    before = fs.Session.active()
    assert before is None, (
        f"a Session was already live before this test started: {before!r}"
    )
    yield
    leaked = fs.Session.active()
    if leaked is not None:
        leaked.close()
        pytest.fail(f"this test left a Session open: {leaked!r}")


@pytest.fixture(scope="session")
def gl_or_skip():
    """A usable headless GL context, or skip.

    Skipping rather than failing keeps a contributor on a machine with no
    obtainable context from seeing a red suite for an environment reason.  The
    tests that make pixel claims all sit behind this.
    """
    from faceforge import session as fs

    try:
        glcontext = fs.glcontext_module()
        return glcontext.acquire_offscreen_gl("auto")
    except Exception as exc:                     # noqa: BLE001 - report and skip
        pytest.skip(f"no headless GL context available: {exc}")
