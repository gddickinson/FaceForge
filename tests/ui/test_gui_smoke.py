"""End-to-end smoke test of the real application UI, headless.

This is the first coverage the UI layer has: it runs the product's own
``app.main()``, opens all six tabs, and interacts with every enabled control
in the main window and in every dialog the app constructs -- asserting both
that nothing raises and that nothing blocks the main thread.

Why timing is asserted
----------------------
The render loop is a 16 ms ``QTimer`` on the main thread, so a handler that
blocks for a second drops ~60 frames.  A control that silently triggers
on-demand asset loading or a full skinning rebuild is a responsiveness defect
even though it raises nothing, and only a timed sweep catches it.  This test
was written after exactly such a defect: ticking "Skin" ran a 792k-vertex
vertex-to-bone solve on the main thread, measured at 34.09 s.

The budget
----------
``INTERACTION_BUDGET_S`` is set from measurement, not aspiration.  On the
reference machine (Apple M1 Max, warm caches), three consecutive full sweeps
of 789 interactions gave a worst case of 0.364 s ("Stretch Heatmap"), a p99 of
8-14 ms, and a median of 0.03 ms.  The budget is 1.5 s: about 4x the worst
observed interaction, which absorbs a slower CI runner and incidental cache
variance, while still being ~23x below the 34 s regression it exists to catch.

Cold machines
-------------
The skin binding solve is memoised on disk (``faceforge.body.skinning_cache``,
the same pattern as the ``.npz`` welded-geometry cache).  The very first run
on a machine has to build that entry and legitimately spends ~34 s inside the
"Skin" interaction.  The fixture detects that case and the budget assertion
skips itself once, with a message; every later run asserts normally.  It never
passes silently on a warm machine.

Not hanging CI
--------------
``gui_harness.stub_blocking_calls`` neutralises ``QDialog.exec``,
``QApplication.exec`` and every *static* dialog helper
(``QColorDialog.getColor``, ``QFileDialog.getSaveFileName``,
``QMessageBox.*``).  Without those stubs the first colour-picker click blocks
forever, because headless there is nobody to dismiss the dialog.  Buttons
matching ``gui_harness.SKIP_BUTTON_WORDS`` (quit/close/save/export/record/...)
are deliberately not clicked: they would end the process or open a native
file dialog outside Qt's control.  ``QMessageBox.question`` is stubbed to
answer *No*, so a destructive confirmation is never confirmed.
"""

from __future__ import annotations

import pytest

from tests.ui import gui_harness as H

pytestmark = pytest.mark.slow

#: Wall seconds any single interaction may take.  See the module docstring.
INTERACTION_BUDGET_S = 1.5

#: Floors on interaction counts, so the sweep cannot silently stop covering
#: things.  Measured counts on the reference tree are roughly 2x these.
MIN_COUNTS = {
    "button": 60, "checkbox": 50, "slider": 150, "combo": 200, "tab": 6,
}


@pytest.fixture(scope="module")
def swept():
    """Build the real app once, drain startup, then sweep every control.

    Module-scoped: ``app.main()`` builds a QApplication and loads the full
    asset set, so it runs once and all assertions read the same result.
    """
    cache_glob = "binding.*.npz"
    from faceforge.body import skinning_cache

    def cache_entries() -> int:
        directory = skinning_cache.cache_dir()
        return len(list(directory.glob(cache_glob))) if directory.is_dir() else 0

    before = cache_entries()

    app, window, startup_errors = H.build_main_window()
    # Must happen before any timing: app.py defers the whole-scene load to a
    # 100 ms singleShot, which would otherwise be charged to whichever
    # interaction happens to be running when it fires.
    drain_s = H.drain_deferred_startup(app)

    tabs = H.walk_tabs(window, app)
    interactions = H.sweep(window, app) + H.sweep_open_dialogs(app, window)

    built_cache = cache_entries() > before
    return {
        "app": app,
        "window": window,
        "startup_errors": startup_errors,
        "drain_s": drain_s,
        "tabs": tabs,
        "interactions": interactions,
        "built_cache": built_cache,
    }


def test_app_constructs_without_exceptions(swept):
    """The real startup path raises nothing, in a slot or otherwise."""
    errors = swept["startup_errors"]
    assert errors == [], (
        f"{len(errors)} exception(s) during startup:\n"
        + "\n".join(e.strip().splitlines()[-1] for e in errors[:5])
    )
    assert swept["window"].windowTitle(), "main window has no title"


def test_all_six_tabs_open(swept):
    """Every named tab exists and selects without raising."""
    labels = [t.label for t in swept["tabs"]]
    for name in H.EXPECTED_TABS:
        assert f"tab({name})" in labels, f"tab {name!r} missing; got {labels}"
    failed = [(t.label, t.error) for t in swept["tabs"] if t.error]
    assert failed == [], f"tabs raised: {failed}"


def test_every_control_interacts_without_raising(swept):
    """No enabled control raises when clicked/toggled/moved/selected."""
    failed = [i for i in swept["interactions"] if i.error]
    assert failed == [], (
        f"{len(failed)} of {len(swept['interactions'])} interactions raised:\n"
        + "\n".join(f"  {i.label} -> {i.error[:160]}" for i in failed[:15])
    )


def test_interaction_coverage_has_not_regressed(swept):
    """The sweep still reaches every control family, in bulk.

    Guards against a refactor that makes ``findChildren`` return nothing and
    turns the whole module into a test that asserts on an empty list.
    """
    counts: dict[str, int] = {}
    for i in swept["interactions"] + swept["tabs"]:
        counts[i.kind] = counts.get(i.kind, 0) + 1
    for kind, floor in MIN_COUNTS.items():
        assert counts.get(kind, 0) >= floor, (
            f"only {counts.get(kind, 0)} {kind} interactions "
            f"(expected >= {floor}); full counts: {counts}"
        )


def test_no_interaction_blocks_the_render_thread(swept):
    """No single interaction exceeds the responsiveness budget."""
    if swept["built_cache"]:
        pytest.skip(
            "this run built the skin binding cache for the first time on this "
            "machine (~34 s of one-time solve inside the 'Skin' interaction); "
            "re-run to assert the steady-state budget"
        )
    over = [i for i in swept["interactions"] + swept["tabs"]
            if i.seconds > INTERACTION_BUDGET_S]
    assert over == [], (
        f"{len(over)} interaction(s) over the {INTERACTION_BUDGET_S:.2f} s "
        "budget — each blocks the 16 ms render timer:\n"
        + "\n".join(f"  {i.seconds:7.2f}s  {i.label}" for i in over)
        + "\n\n" + H.summarise(swept["interactions"] + swept["tabs"])
    )
