"""Suite-wide collection rules.

Everything here exists to keep one invariant true:

    `pytest -m "not slow"` passes on a checkout with no BodyParts3D dataset.

That is not a convenience.  It is the condition CI runs in — `assets/stl` is a
*committed symlink* to a path outside the repo, so a fresh clone always has a
dangling link — and it is the condition a new contributor is in before they
have downloaded 1.2 GB.

Keeping the invariant by hand does not work.  The asset-heavy modules were
marked `slow` one at a time by inspection, and within the hour a new module
(`tests/core/test_scene_state.py`) arrived with a `real_scene` fixture that
loads real STL meshes and asserts they loaded.  It was not marked, because
nobody marking modules by hand knew it was coming: eight tests went from
passing to *erroring at setup* the moment the dataset was absent, which is a
red CI build caused by a test that is perfectly correct.

So the rule is expressed once, against the thing that actually makes a test
asset-heavy — the fixture it requests.
"""

from __future__ import annotations

import pytest

# Fixtures that read the BodyParts3D dataset.  Any test whose fixture closure
# contains one of these is asset-heavy by definition and belongs in the slow
# tier, whoever wrote it and whenever they wrote it.
#
#   real_scene       tests/core/test_scene_state.py -- load_stl_batch(REAL_DEFS)
#   headless_scene   tests/tools/*, tests/body/*    -- the full headless load
#
# Add to this set when a new asset-reading fixture appears; do not mark the
# tests individually, or the next module to arrive will be missed the same way.
ASSET_FIXTURES = frozenset({"real_scene", "headless_scene"})


def pytest_collection_modifyitems(config, items):
    """Mark every test that requests an asset-reading fixture as ``slow``.

    ``item.fixturenames`` is the resolved closure, not just the direct
    parameters, so a fixture that itself depends on ``real_scene`` is covered
    too.  This hook lives in a conftest, which pytest registers after its own
    plugins and therefore calls *before* the ``-m`` deselection pass — the mark
    is in place by the time the expression is evaluated.
    """
    for item in items:
        if ASSET_FIXTURES.intersection(getattr(item, "fixturenames", ())):
            item.add_marker(pytest.mark.slow)
