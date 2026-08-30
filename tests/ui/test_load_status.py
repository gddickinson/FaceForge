"""The load-failure surface, headless, with injected failing loaders.

The behaviour under test is the one that matters to a user: a partially loaded
body must look different from a complete one, and the two meshes genuinely
absent from BodyParts3D must not look like a broken install.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt                                  # noqa: E402
from PySide6.QtWidgets import (                                # noqa: E402
    QApplication,
    QMainWindow,
    QStatusBar,
)

from faceforge.coordination.loading_pipeline import LoadReport  # noqa: E402
from faceforge.ui.load_status import (                          # noqa: E402
    KNOWN_ABSENT,
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_OK,
    SEVERITY_WARNING,
    LoadStatusBadge,
    LoadStatusPanel,
    classify,
    install,
)


@pytest.fixture(scope="module")
def qapp():
    """A QApplication that is destroyed again on teardown.

    Load-bearing: ``faceforge.app.main()`` calls ``QApplication(sys.argv)``
    unconditionally, and shiboken refuses to build a second instance while one
    is alive ("Please destroy the QApplication singleton before creating a new
    QApplication instance").  A session-scoped app left running here therefore
    breaks tests/ui/test_gui_smoke.py, which runs the real ``main()``.  So this
    fixture creates an app only if none exists and shuts down the one it
    created, leaving a pre-existing app alone.
    """
    existing = QApplication.instance()
    if existing is not None:
        yield existing
        return
    app = QApplication([])
    try:
        yield app
    finally:
        app.processEvents()
        app.shutdown()


@pytest.fixture
def window(qapp):
    win = QMainWindow()
    win.setStatusBar(QStatusBar())
    yield win
    win.deleteLater()


# ── classification ───────────────────────────────────────────────────────

def test_no_report_is_silent():
    status = classify(None)
    assert status.severity == SEVERITY_OK
    assert status.badge == ""


def test_a_clean_report_is_ok_and_counts_are_shown():
    status = classify(LoadReport(), loaded=932, expected=932)
    assert status.severity == SEVERITY_OK
    assert status.badge == "932 of 932 structures loaded"
    assert status.actionable is False


def test_the_known_upstream_gap_is_informational_not_an_error():
    """The common case on a correct install: two meshes BodyParts3D does not
    ship.  It must be visible and it must not be red."""
    report = LoadReport(partial={"face_features": {
        "eyes": "FileNotFoundError: FMA49041.stl not found",
    }})
    status = classify(report, loaded=930, expected=932)
    assert status.severity == SEVERITY_INFO
    assert status.actionable is False
    assert "930 of 932 structures loaded" in status.badge
    assert "not in dataset" in status.badge
    assert "FMA49041" in status.detail
    assert "not a problem with this install" in status.detail


def test_both_known_absent_meshes_are_recognised():
    report = LoadReport(partial={"face_features": {
        "left": "FileNotFoundError: FMA49041.stl",
        "right": "FileNotFoundError: FMA49042.stl",
    }})
    status = classify(report, loaded=930, expected=932)
    assert status.severity == SEVERITY_INFO
    assert set(status.known_absent) == set(KNOWN_ABSENT)


def test_a_real_partial_failure_is_a_warning():
    report = LoadReport(partial={"body_muscles": {
        "left_arm": "RuntimeError: parse error",
        "left_leg": "RuntimeError: parse error",
    }})
    status = classify(report)
    assert status.severity == SEVERITY_WARNING
    assert status.actionable is True
    assert "2 structure group(s) incomplete" == status.badge


def test_a_whole_subsystem_failing_is_an_error():
    report = LoadReport(failures={
        "vasculature": "FileNotFoundError: assets/stl/vasculature",
    })
    status = classify(report)
    assert status.severity == SEVERITY_ERROR
    assert status.actionable is True
    assert "1 subsystem(s) failed to load" == status.badge
    assert "vasculature" in status.detail


def test_a_known_gap_alongside_a_real_failure_is_not_downgraded():
    """A report containing more than the upstream gap must not be excused by
    it -- that would hide a genuine failure behind an expected one."""
    report = LoadReport(
        failures={"organs": "FileNotFoundError: missing directory"},
        partial={"face_features": {"eyes": "FileNotFoundError: FMA49041.stl"}},
    )
    status = classify(report, loaded=800, expected=932)
    assert status.severity == SEVERITY_ERROR
    assert "FMA49041" in status.detail          # still explained
    assert "organs" in status.detail


def test_detail_lists_every_failed_group():
    report = LoadReport(partial={"skeleton": {
        "ribs": "A", "pelvis": "B", "hands": "C",
    }})
    detail = classify(report).detail
    for unit in ("ribs", "pelvis", "hands"):
        assert unit in detail


# ── the badge ────────────────────────────────────────────────────────────

def test_badge_starts_empty_and_hidden(qapp):
    badge = LoadStatusBadge()
    assert badge.text() == ""
    assert badge.isHidden() is True


def test_badge_shows_the_report(qapp):
    badge = LoadStatusBadge()
    badge.set_report(LoadReport(), loaded=932, expected=932)
    assert badge.isHidden() is False
    assert "932 of 932" in badge.text()
    assert badge.status.severity == SEVERITY_OK


def test_badge_severity_changes_its_style(qapp):
    badge = LoadStatusBadge()
    badge.set_report(LoadReport(failures={"organs": "boom"}))
    error_style = badge.styleSheet()
    badge.set_report(LoadReport(), loaded=1, expected=1)
    assert badge.styleSheet() != error_style


def test_badge_offers_a_pointer_only_when_there_is_detail(qapp):
    badge = LoadStatusBadge()
    badge.set_report(LoadReport(), loaded=1, expected=1)
    assert badge.cursor().shape() == Qt.ArrowCursor
    badge.set_report(LoadReport(failures={"organs": "boom"}))
    assert badge.cursor().shape() == Qt.PointingHandCursor


def test_badge_tooltip_carries_the_detail(qapp):
    badge = LoadStatusBadge()
    badge.set_report(LoadReport(failures={"organs": "FileNotFoundError: x"}))
    assert "organs" in badge.toolTip()


# ── the detail panel ─────────────────────────────────────────────────────

def test_clicking_opens_a_non_modal_panel(qapp):
    badge = LoadStatusBadge()
    badge.set_report(LoadReport(failures={"organs": "FileNotFoundError: x"}))
    panel = badge.show_panel()
    assert isinstance(panel, LoadStatusPanel)
    assert panel.isModal() is False
    assert "organs" in panel._detail.toPlainText()


def test_the_panel_is_reused_not_stacked(qapp):
    badge = LoadStatusBadge()
    badge.set_report(LoadReport(failures={"organs": "x"}))
    assert badge.show_panel() is badge.show_panel()


def test_no_panel_is_opened_for_a_clean_load(qapp):
    badge = LoadStatusBadge()
    badge.set_report(LoadReport(), loaded=1, expected=1)
    badge.mouseReleaseEvent(_fake_release())
    assert badge.panel is None


def test_an_open_panel_follows_a_later_report(qapp):
    badge = LoadStatusBadge()
    badge.set_report(LoadReport(failures={"organs": "x"}))
    panel = badge.show_panel()
    badge.set_report(LoadReport(failures={"brain": "y"}))
    assert "brain" in panel._detail.toPlainText()


def _fake_release():
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtCore import QPointF
    return QMouseEvent(QMouseEvent.Type.MouseButtonRelease, QPointF(1, 1),
                       QPointF(1, 1), Qt.LeftButton, Qt.LeftButton,
                       Qt.NoModifier)


# ── installation on a window ──────────────────────────────────────────────

def test_install_adds_a_badge_to_the_status_bar(window):
    badge = install(window, LoadReport(failures={"organs": "x"}))
    assert badge is not None
    assert window.load_status_badge is badge
    assert "1 subsystem(s) failed" in badge.text()


def test_install_is_idempotent(window):
    first = install(window, LoadReport())
    second = install(window, LoadReport(failures={"organs": "x"}))
    assert first is second


def test_install_on_a_window_without_a_status_bar_is_a_no_op(qapp):
    class Bare:
        pass
    assert install(Bare(), LoadReport()) is None


# ── the real MainWindow ──────────────────────────────────────────────────

def test_main_window_exposes_the_badge_and_the_setter(qapp, monkeypatch):
    """The surface must be reachable on the real window, with an injected
    failing loader standing in for a broken asset tree."""
    from faceforge.core.events import EventBus
    from faceforge.core.state import StateManager
    from faceforge.ui.main_window import MainWindow
    from PySide6.QtWidgets import QWidget

    window = MainWindow(EventBus(), StateManager(), QWidget())
    try:
        assert window.load_status_badge.isHidden() is True

        injected = LoadReport()
        injected.record("vasculature", FileNotFoundError("assets/stl/arteries"))
        window.set_load_report(injected, loaded=780, expected=932)

        badge = window.load_status_badge
        assert badge.isHidden() is False
        assert badge.status.severity == SEVERITY_ERROR
        assert "vasculature" in badge.status.detail
        assert "FileNotFoundError" in badge.status.detail

        # And the expected two-mesh case reads as informational.
        gap = LoadReport(partial={"face_features": {
            "eyes": "FileNotFoundError: FMA49042.stl"}})
        window.set_load_report(gap, loaded=930, expected=932)
        assert badge.status.severity == SEVERITY_INFO
        assert "930 of 932 structures loaded" in badge.text()
    finally:
        window.deleteLater()
