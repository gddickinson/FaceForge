"""Non-modal surface for asset-load failures.

The loaders report what did not load via ``LoadReport``, but until now nothing
showed it, so a body missing its vasculature looked exactly like a complete
one.  This module is the surface: a status-bar badge that opens a detail panel.

Three design decisions, each with a reason:

**Non-modal.**  A modal dialog on startup would be worse than silence.  The
common case on a correct installation is two meshes absent from the upstream
dataset; interrupting every launch for that trains the user to dismiss the
dialog unread, which is the state we are trying to leave.

**Severity is graded, not binary.**  ``LoadReport.degraded`` is true for both
"two known-absent meshes" and "the entire skeleton failed", and presenting
those identically is what makes a warning useless.  :func:`classify` maps a
report onto :data:`SEVERITY_*` using the known-absent list, so the expected
case reads as *informational* ("930 of 932 structures") in the normal
foreground colour, while an unexpected failure is amber and a total failure of
a subsystem is red.

**Known-absent meshes are named in the data, not in this file.**
``FMA49041``/``FMA49042`` are genuinely not in the BodyParts3D distribution, so
they are listed in :data:`KNOWN_ABSENT` with the reason -- and the badge says
so, rather than implying a broken install.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

SEVERITY_OK = "ok"
SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_ERROR = "error"

SEVERITY_ORDER = (SEVERITY_OK, SEVERITY_INFO, SEVERITY_WARNING, SEVERITY_ERROR)

#: Meshes that are absent from the upstream BodyParts3D distribution itself.
#: Their absence is a property of the dataset, not of the user's install, so a
#: report containing only these is informational.
KNOWN_ABSENT = {
    "FMA49041": "absent from the BodyParts3D distribution",
    "FMA49042": "absent from the BodyParts3D distribution",
}

_STYLES = {
    SEVERITY_OK: "color: #77aa77;",
    SEVERITY_INFO: "color: #aaaaaa;",
    SEVERITY_WARNING: "color: #ddaa44; font-weight: bold;",
    SEVERITY_ERROR: "color: #dd5555; font-weight: bold;",
}


@dataclass(frozen=True)
class LoadStatus:
    """A report reduced to what the UI needs to show."""

    severity: str
    badge: str
    detail: str
    failure_count: int = 0
    known_absent: tuple[str, ...] = ()

    @property
    def actionable(self) -> bool:
        """True when the user should look at this, i.e. it is not expected."""
        return self.severity in (SEVERITY_WARNING, SEVERITY_ERROR)


def classify(report: Optional[object], loaded: int = 0, expected: int = 0
             ) -> LoadStatus:
    """Reduce a ``LoadReport`` to a :class:`LoadStatus`.

    ``loaded``/``expected`` are mesh counts, used only for the badge text when
    they are supplied -- "930 of 932 structures loaded" is the sentence that
    makes the common case unalarming and still visible.  A report is:

    ``ok``
        no failures at all;
    ``info``
        every failure is a known-absent upstream mesh;
    ``warning``
        some subsystems partially failed;
    ``error``
        at least one whole subsystem failed to load.
    """
    if report is None:
        return LoadStatus(severity=SEVERITY_OK, badge="", detail="")

    failures = dict(getattr(report, "failures", {}) or {})
    partial = {k: dict(v) for k, v in
               (getattr(report, "partial", {}) or {}).items()}
    count = len(failures) + sum(len(v) for v in partial.values())

    absent = tuple(sorted(
        mesh_id for mesh_id in KNOWN_ABSENT
        if any(mesh_id in text for text in failures.values())
        or any(mesh_id in text
               for units in partial.values() for text in units.values())
    ))

    counts = (f"{loaded} of {expected} structures loaded"
              if expected else f"{count} issue(s)")

    if not failures and not partial:
        badge = counts if expected else "All structures loaded"
        return LoadStatus(severity=SEVERITY_OK, badge=badge,
                          detail="Scene loaded completely.")

    detail = _detail_text(failures, partial, absent)

    # Every recorded failure accounted for by a known-absent upstream mesh.
    if absent and count <= len(absent):
        return LoadStatus(
            severity=SEVERITY_INFO,
            badge=f"{counts} \u2014 {len(absent)} not in dataset",
            detail=detail, failure_count=count, known_absent=absent,
        )

    if failures:
        return LoadStatus(
            severity=SEVERITY_ERROR,
            badge=f"{len(failures)} subsystem(s) failed to load",
            detail=detail, failure_count=count, known_absent=absent,
        )

    return LoadStatus(
        severity=SEVERITY_WARNING,
        badge=f"{count} structure group(s) incomplete",
        detail=detail, failure_count=count, known_absent=absent,
    )


def _detail_text(failures: dict, partial: dict, absent: tuple) -> str:
    lines: list[str] = []
    if failures:
        lines.append("Subsystems that did not load:")
        lines += [f"  \u2022 {name}: {reason}"
                  for name, reason in sorted(failures.items())]
    if partial:
        lines.append("Partially loaded subsystems:")
        for name, units in sorted(partial.items()):
            lines.append(f"  \u2022 {name}: {len(units)} group(s) failed")
            lines += [f"      - {unit}: {reason}"
                      for unit, reason in sorted(units.items())]
    if absent:
        lines.append("")
        lines.append("Known upstream gaps (not a problem with this install):")
        lines += [f"  \u2022 {mesh_id}: {KNOWN_ABSENT[mesh_id]}"
                  for mesh_id in absent]
    return "\n".join(lines)


class LoadStatusBadge(QLabel):
    """A clickable status-bar label.  Click opens the detail panel.

    A label rather than a button so it sits quietly in the status bar next to
    the vertex and FPS counters; the cursor changes to a pointing hand when
    there is something to open, which is the only affordance a status bar
    should need.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._status = classify(None)
        self._panel: Optional[LoadStatusPanel] = None
        self.setTextFormat(Qt.PlainText)
        self.set_status(self._status)

    @property
    def status(self) -> LoadStatus:
        return self._status

    def set_status(self, status: LoadStatus) -> None:
        self._status = status
        self.setText(status.badge)
        self.setStyleSheet(_STYLES.get(status.severity, ""))
        self.setToolTip(status.detail or status.badge)
        has_detail = bool(status.detail) and status.severity != SEVERITY_OK
        self.setCursor(Qt.PointingHandCursor if has_detail else Qt.ArrowCursor)
        self.setVisible(bool(status.badge))
        if self._panel is not None and not self._panel.isHidden():
            self._panel.set_status(status)

    def set_report(self, report, loaded: int = 0, expected: int = 0) -> None:
        """Convenience: classify then display."""
        self.set_status(classify(report, loaded=loaded, expected=expected))

    # -- interaction -------------------------------------------------------

    def mouseReleaseEvent(self, event):                        # noqa: N802
        if self._status.detail and self._status.severity != SEVERITY_OK:
            self.show_panel()
        super().mouseReleaseEvent(event)

    def show_panel(self) -> "LoadStatusPanel":
        """Open (or re-show) the non-modal detail panel."""
        if self._panel is None:
            self._panel = LoadStatusPanel(self.window())
        self._panel.set_status(self._status)
        self._panel.show()
        self._panel.raise_()
        return self._panel

    @property
    def panel(self) -> Optional["LoadStatusPanel"]:
        return self._panel


class LoadStatusPanel(QDialog):
    """The detail view.  Explicitly non-modal -- see the module docstring."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Asset load report")
        self.setModal(False)
        self.resize(560, 380)

        layout = QVBoxLayout(self)
        self._headline = QLabel("")
        self._headline.setWordWrap(True)
        layout.addWidget(self._headline)

        self._detail = QPlainTextEdit()
        self._detail.setReadOnly(True)
        layout.addWidget(self._detail)

        buttons = QHBoxLayout()
        close = QPushButton("Close")
        close.clicked.connect(self.hide)
        buttons.addStretch()
        buttons.addWidget(close)
        layout.addLayout(buttons)

    def set_status(self, status: LoadStatus) -> None:
        self._headline.setText(status.badge)
        self._headline.setStyleSheet(_STYLES.get(status.severity, ""))
        self._detail.setPlainText(status.detail)


def install(window: QWidget, report=None, loaded: int = 0, expected: int = 0
            ) -> Optional[LoadStatusBadge]:
    """Attach a badge to ``window``'s status bar and show ``report``.

    Returns the badge, or None if the window has no status bar.  Idempotent:
    calling it again updates the existing badge rather than adding a second.

    This is the seam the application wires up.  ``MainWindow`` owns the badge
    (see :meth:`faceforge.ui.main_window.MainWindow.set_load_report`); this
    helper exists for callers that hold only a window.
    """
    badge = getattr(window, "load_status_badge", None)
    if badge is None:
        status_bar = getattr(window, "status_bar", None)
        if status_bar is None and hasattr(window, "statusBar"):
            status_bar = window.statusBar()
        if status_bar is None:
            return None
        badge = LoadStatusBadge(window)
        status_bar.addPermanentWidget(badge)
        window.load_status_badge = badge
    badge.set_report(report, loaded=loaded, expected=expected)
    return badge
