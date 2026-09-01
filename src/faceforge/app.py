"""FaceForge application entry point.

``main()`` assembles the application and starts it.  It does four things and
nothing else:

1. build the collaborators -- :func:`faceforge.appcontext.build_app_context`;
2. wire the handlers -- :func:`faceforge.controllers.build_controllers`;
3. ask the user which preset to start in, then arm the asset load --
   :class:`faceforge.coordination.asset_load_sequence.AssetLoadSequence`;
4. install the per-frame loop and show the window.

Everything it used to do inline lives in those modules.  The behaviour is
unchanged; what changed is that each piece now has a name that can be imported
and tested without a ``QApplication``.
"""

# Disable PyOpenGL's per-call error checking BEFORE any GL imports.
# macOS Metal translation layer leaves stale GL errors that cause
# PyOpenGL's automatic error checker to raise on every GL call.
import OpenGL
OpenGL.ERROR_CHECKING = False

import logging
import sys

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from faceforge.appcontext import AppContext, build_app_context
from faceforge.controllers import Controllers, build_controllers
from faceforge.controllers.frame_loop import FrameLoop
from faceforge.coordination.asset_load_sequence import AssetLoadSequence, LoadStage
from faceforge.rendering.gl_widget import create_gl_format

logger = logging.getLogger(__name__)

#: Delay before the asset load sequence runs.
#:
#: The load needs the GL widget to have been shown and given a context, and Qt
#: offers no signal for "the first paint has happened", so the sequence is armed
#: on a short timer instead.  The cost of that deferral used to be invisible --
#: the load ran inside whichever ``processEvents()`` was executing when the
#: timer expired, which is how a GUI sweep once charged 3.18 s of startup work
#: to an unrelated checkbox.  The sequence now reports its stage as it runs, so
#: that time can be attributed to the stage that spends it.
ASSET_LOAD_DELAY_MS = 100


def configure_logging() -> None:
    """Send warnings somewhere visible.  Without this they are swallowed."""
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")


def configure_gl_format() -> None:
    """Set the default surface format.  Must precede ``QApplication``.

    Qt reads the default format when it creates the first OpenGL surface, and
    a ``QSurfaceFormat`` set after ``QApplication`` is constructed is ignored
    on some platforms -- silently producing a legacy context.
    """
    from PySide6.QtGui import QSurfaceFormat

    QSurfaceFormat.setDefaultFormat(create_gl_format())


def run_startup_dialog(ctx: AppContext) -> None:
    """Ask which preset to start in, and record the answer on the context.

    Must run *before* the asset load is armed.  ``QDialog.exec()`` runs a
    nested event loop, so a timer armed first would fire while the dialog is
    still open and the load would read the selection before it was made.
    """
    from faceforge.ui.startup_dialog import StartupDialog

    dialog = StartupDialog()
    dialog.exec()
    ctx.startup_preset = dialog.selected_preset
    ctx.startup_illustration = dialog.selected_illustration


def arm_asset_load(ctx: AppContext, controllers: Controllers) -> AssetLoadSequence:
    """Create the load sequence and schedule it.  See :data:`ASSET_LOAD_DELAY_MS`.

    The sequence is stored on the context before the timer is armed, and that
    is not bookkeeping: ``QTimer.singleShot`` does not keep the callable it is
    given alive, so a sequence referenced only by the timer is garbage
    collected before it fires and the whole scene silently fails to load --
    no exception, no log line, just an empty viewport.
    """
    sequence = AssetLoadSequence(ctx, controllers, on_stage=_log_stage)
    ctx.load_sequence = sequence
    QTimer.singleShot(ASSET_LOAD_DELAY_MS, sequence.run)
    return sequence


def _log_stage(stage: LoadStage) -> None:
    # Debug level: the startup log is already busy, and this is for whoever is
    # profiling the load rather than for every run.
    logger.debug("asset load stage: %s", stage.value)


def main() -> None:
    """Launch the FaceForge application."""
    configure_logging()
    configure_gl_format()

    # Reuse the running QApplication if there is one. Constructing a second is
    # a hard RuntimeError from libshiboken, which made main() unusable in two
    # legitimate situations: embedding FaceForge in a host Qt application, and
    # any test that needs a QApplication before calling main() (pytest-qt's
    # qapp fixture creates one, so mere test ordering could break the suite).
    app = QApplication.instance() or QApplication(sys.argv)

    ctx = build_app_context()
    controllers = build_controllers(ctx)

    run_startup_dialog(ctx)
    arm_asset_load(ctx, controllers)

    FrameLoop(ctx, controllers, ctx.gl_widget.paintGL).install()
    ctx.window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
