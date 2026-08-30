"""Drive FaceForge's real Qt UI headlessly, with per-interaction timing.

Why this works when the 3D viewport does not
--------------------------------------------
Qt's ``offscreen`` platform plugin refuses ``createPlatformOpenGLContext``, so
``GLViewport`` (a ``QOpenGLWidget``) never gets a context and never paints.
Everything *else* in the application -- the main window, the six tabs, every
button, slider, combo box and checkbox, the event bus, and the state manager
-- is ordinary Qt and constructs and responds normally.  So the UI layer is
fully drivable without a window server; only the rendered image is
unavailable.

Strategy
--------
Rather than reimplement ``app.main()``'s ~2,150 lines of wiring (which would
test a copy rather than the product), stub the calls that would block and
then run the real ``main()``:

* ``QDialog.exec``      -> returns Accepted immediately (StartupDialog runs a
                           nested event loop that would hang forever).
* ``QApplication.exec`` -> returns 0 (the main event loop).
* the STATIC dialog helpers (``QColorDialog.getColor``,
  ``QFileDialog.getSaveFileName``, ``QMessageBox.warning`` ...) -- these build
  and run their own dialog internally and never route through the
  ``QDialog.exec`` override, so headless the first one blocks forever.
* ``SystemExit`` is caught, since ``main()`` ends in ``sys.exit(app.exec())``.

The constructed window is then found via ``QApplication.topLevelWidgets()``.

Exceptions raised inside a Qt slot do not propagate to the caller -- Qt
prints them and continues -- so ``sys.excepthook`` is installed to capture
them, and every interaction is compared against the captured list.

Timing
------
Each interaction is timed.  The app's render loop is a 16 ms ``QTimer`` on the
main thread, so any handler that blocks for a second is a visible freeze
(~60 dropped frames per second of block).  Wall time per interaction is
therefore a first-class assertion, not a nicety.

One subtlety is load-bearing: ``app.py`` arms
``QTimer.singleShot(100, load_assets)``, so the whole-scene load (skull, body
skeleton, gender morph, face features) runs inside whichever
``processEvents()`` happens to be executing when that timer expires.  Left
undrained, that multi-second startup cost is charged to an arbitrary
interaction -- which is exactly how a 3.18 s reading was once attributed to
the "Skin" checkbox.  :func:`drain_deferred_startup` must be called before
any timing is collected.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from dataclasses import dataclass

# Must be set before QApplication is constructed.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import (  # noqa: E402  (import after the env var)
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QTabWidget,
)

#: The six tabs of the control panel, in order.
EXPECTED_TABS = ("ANIMATE", "BODY", "LAYERS", "ALIGN", "DISPLAY", "DEBUG")

#: Buttons whose label contains one of these are not clicked.  Each would
#: either terminate the process or open a native file dialog that the static
#: stubs below cannot fully neutralise on every platform.  Opening a picker is
#: CORRECT behaviour, not a defect -- these are skipped so the sweep can
#: continue past them.
SKIP_BUTTON_WORDS = (
    "quit", "exit", "close", "save", "load", "export", "import",
    "open", "browse", "record", "screenshot",
)

#: Exceptions raised inside Qt slots, captured by the excepthook.
SLOT_ERRORS: list[str] = []


@dataclass(frozen=True)
class Interaction:
    """One control interaction: what it was, how long it took, what it raised."""

    label: str
    kind: str
    seconds: float
    error: str | None = None


def _install_excepthook() -> None:
    def _hook(exc_type, exc, tb):
        SLOT_ERRORS.append("".join(traceback.format_exception(exc_type, exc, tb)))

    sys.excepthook = _hook


def stub_blocking_calls() -> None:
    """Neutralise every call that would block forever without a window server.

    This is what keeps the test from hanging CI.  ``QDialog.exec`` covers
    dialogs the app constructs itself; the static convenience helpers each
    build and run their own dialog internally and must be stubbed separately.
    """
    QDialog.exec = lambda self: QDialog.DialogCode.Accepted  # type: ignore[assignment]
    QApplication.exec = lambda self: 0                       # type: ignore[assignment]

    from PySide6.QtGui import QColor, QFont
    from PySide6.QtWidgets import (
        QColorDialog, QFileDialog, QFontDialog, QInputDialog, QMessageBox,
    )

    QColorDialog.getColor = staticmethod(lambda *a, **k: QColor("#808080"))
    QFontDialog.getFont = staticmethod(lambda *a, **k: (QFont(), True))
    QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: ("", ""))
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: ("", ""))
    QFileDialog.getOpenFileNames = staticmethod(lambda *a, **k: ([], ""))
    QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: "")
    QInputDialog.getText = staticmethod(lambda *a, **k: ("", False))
    QInputDialog.getInt = staticmethod(lambda *a, **k: (0, False))
    QInputDialog.getDouble = staticmethod(lambda *a, **k: (0.0, False))
    QInputDialog.getItem = staticmethod(lambda *a, **k: ("", False))
    for name in ("information", "warning", "critical", "about", "aboutQt"):
        setattr(QMessageBox, name, staticmethod(lambda *a, **k: None))
    # Never answer Yes: a destructive confirmation must not be confirmed.
    QMessageBox.question = staticmethod(
        lambda *a, **k: QMessageBox.StandardButton.No)


def build_main_window() -> tuple[QApplication, QMainWindow, list[str]]:
    """Run the product's real startup path.

    Returns ``(app, window, startup_errors)``.
    """
    SLOT_ERRORS.clear()
    _install_excepthook()
    stub_blocking_calls()

    import faceforge.app as app_mod

    try:
        app_mod.main()
    except SystemExit:
        pass          # main() ends in sys.exit(app.exec()); expected

    app = QApplication.instance()
    if app is None:                       # pragma: no cover - defensive
        raise RuntimeError("main() did not construct a QApplication")
    windows = [w for w in app.topLevelWidgets() if isinstance(w, QMainWindow)]
    if not windows:                       # pragma: no cover - defensive
        raise RuntimeError("no QMainWindow was constructed")
    errors = list(SLOT_ERRORS)
    SLOT_ERRORS.clear()
    return app, windows[0], errors


def drain_deferred_startup(
    app: QApplication, settle: float = 0.5, timeout: float = 300.0,
) -> float:
    """Pump the event loop until the deferred whole-scene load has finished.

    ``app.py`` arms ``QTimer.singleShot(100, load_assets)``, so the scene load
    runs inside a ``processEvents()`` call some time after ``main()`` returns.
    This pumps until *settle* seconds have passed with no pump exceeding
    50 ms, which cannot terminate before the 100 ms timer has fired.

    Returns the wall seconds spent draining.
    """
    t0 = time.perf_counter()
    quiet_since: float | None = None
    while time.perf_counter() - t0 < timeout:
        tick = time.perf_counter()
        app.processEvents()
        if time.perf_counter() - tick > 0.05:
            quiet_since = None            # something ran; restart the clock
        elif quiet_since is None:
            quiet_since = time.perf_counter()
        elif time.perf_counter() - quiet_since >= settle:
            break
        time.sleep(0.01)
    return time.perf_counter() - t0


def describe(widget) -> str:
    """Short human-readable identifier for a widget."""
    name = widget.objectName() or ""
    text = ""
    for attr in ("text", "currentText", "title"):
        if hasattr(widget, attr):
            try:
                text = str(getattr(widget, attr)()) or ""
            except (RuntimeError, TypeError):
                text = ""
            if text:
                break
    label = (text or name or "").strip().replace("\n", " ")[:38]
    return f"{type(widget).__name__}({label})"


def walk_tabs(window: QMainWindow, app: QApplication) -> list[Interaction]:
    """Select every tab of every ``QTabWidget`` in the window."""
    out: list[Interaction] = []
    for tab_widget in window.findChildren(QTabWidget):
        for i in range(tab_widget.count()):
            label = tab_widget.tabText(i)
            before = len(SLOT_ERRORS)
            t0 = time.perf_counter()
            err: str | None = None
            try:
                tab_widget.setCurrentIndex(i)
                app.processEvents()
            except Exception as exc:               # noqa: BLE001 - reported
                err = f"{type(exc).__name__}: {exc}"
            dt = time.perf_counter() - t0
            if err is None and len(SLOT_ERRORS) > before:
                err = SLOT_ERRORS[-1].strip().splitlines()[-1]
            out.append(Interaction(f"tab({label})", "tab", dt, err))
    return out


def _own_window(widget, root) -> bool:
    """True when *widget* lives in *root*'s own window, not a nested dialog.

    Clicking a button can create a dialog parented to the main window.  Qt
    still reports that dialog's children under ``window.findChildren``, so
    without this filter the main sweep's coverage would depend on which
    button happened to construct which dialog.  Dialogs get their own pass
    via :func:`sweep_open_dialogs` instead.
    """
    return widget.window() is root.window()


def sweep(
    window: QMainWindow, app: QApplication, root=None, prefix: str = "",
) -> list[Interaction]:
    """Interact with every enabled control under *root*, timing each one.

    *root* defaults to *window*.  Control lists are collected per type
    immediately before that type is swept, because earlier interactions
    legitimately create new controls (e.g. populating a combo box of loaded
    structures).
    """
    if root is None:
        root = window
    out: list[Interaction] = []

    def run(widget, action, kind: str) -> None:
        before = len(SLOT_ERRORS)
        err: str | None = None
        t0 = time.perf_counter()
        try:
            action()
            app.processEvents()
        except Exception as exc:                   # noqa: BLE001 - reported
            err = f"{type(exc).__name__}: {exc}"
        dt = time.perf_counter() - t0
        if err is None and len(SLOT_ERRORS) > before:
            err = SLOT_ERRORS[-1].strip().splitlines()[-1]
        out.append(Interaction(prefix + describe(widget), kind, dt, err))

    buttons = [
        b for b in root.findChildren(QPushButton)
        if b.isEnabled() and _own_window(b, root)
        and not any(w in (b.text() or "").lower() for w in SKIP_BUTTON_WORDS)
    ]
    for b in buttons:
        run(b, b.click, "button")

    boxes = [c for c in root.findChildren(QCheckBox)
             if c.isEnabled() and _own_window(c, root)]
    for c in boxes:
        run(c, lambda c=c: c.setChecked(not c.isChecked()), "checkbox")

    radios = [r for r in root.findChildren(QRadioButton)
              if r.isEnabled() and _own_window(r, root) and not r.isChecked()]
    for r in radios:
        run(r, r.click, "radio")

    sliders = [s for s in root.findChildren(QSlider)
               if s.isEnabled() and _own_window(s, root)]
    for s in sliders:
        lo, hi = s.minimum(), s.maximum()
        for value in (lo, (lo + hi) // 2, hi):
            run(s, lambda s=s, v=value: s.setValue(v), "slider")

    combos = [cb for cb in root.findChildren(QComboBox)
              if cb.isEnabled() and _own_window(cb, root)]
    for cb in combos:
        for i in range(cb.count()):
            run(cb, lambda cb=cb, i=i: cb.setCurrentIndex(i), "combo")

    # PySide6's findChildren rejects a tuple of types ("Subscripted generics
    # cannot be used with class and instance checks") -- query each separately.
    spins = [sp for sp in (root.findChildren(QSpinBox)
                           + root.findChildren(QDoubleSpinBox))
             if sp.isEnabled() and _own_window(sp, root)]
    for sp in spins:
        run(sp, lambda sp=sp: sp.setValue(sp.maximum()), "spinbox")

    return out


def sweep_open_dialogs(
    app: QApplication, window: QMainWindow,
) -> list[Interaction]:
    """Sweep the controls of every dialog the app has constructed.

    Clicking "Anatomy Quiz", "Compare" or "Edit Timeline" builds a real
    ``QDialog``; ``QDialog.exec`` is stubbed so those dialogs exist but never
    ran a nested event loop.  Their controls are real UI that would otherwise
    have no coverage at all, so they get their own pass -- kept separate from
    the main-window sweep because *which* dialogs exist depends on what the
    button pass clicked.
    """
    out: list[Interaction] = []
    dialogs = [w for w in app.topLevelWidgets()
               if isinstance(w, QDialog) and w is not window]
    for dialog in dialogs:
        name = type(dialog).__name__
        out.extend(sweep(window, app, root=dialog, prefix=f"{name}/"))
    return out


def summarise(interactions: list[Interaction]) -> str:
    """Multi-line report of counts, failures and the slowest interactions."""
    kinds: dict[str, int] = {}
    for i in interactions:
        kinds[i.kind] = kinds.get(i.kind, 0) + 1
    lines = [f"{len(interactions)} interactions: " +
             ", ".join(f"{k}={v}" for k, v in sorted(kinds.items()))]
    failed = [i for i in interactions if i.error]
    lines.append(f"failures: {len(failed)}")
    for i in failed[:15]:
        lines.append(f"  FAIL {i.label} -> {i.error[:140]}")
    lines.append("slowest:")
    for i in sorted(interactions, key=lambda x: -x.seconds)[:10]:
        lines.append(f"  {i.seconds:7.3f}s  {i.label}")
    return "\n".join(lines)
