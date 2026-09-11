"""``python -m faceforge.exercise_viewer`` -- FaceForge opened straight into the exercise viewer.

The same application as :mod:`faceforge.app` (context, controllers, asset
load, frame loop), without the startup dialog: the skeleton preset is loaded,
and once the load sequence has run to its last stage the main window
switches to its exercise-viewer mode (see
:class:`faceforge.ui.exercise_viewer.ExerciseViewerPanel`), which enters the
gym, loads every muscle layer and offers the catalogue.  Pass ``--exercise ID``
to start a demonstration as soon as the body is ready.

The mode is entered on the load *sequence's* completion, not on the
``LOADING_COMPLETE`` event: that event is published by the skeleton pipeline
several stages before body animation, the rib pivots, the skinning and the
attachment systems are wired.  Entering there started the exercise with no
body animation (so no grip lock) and painted the skeleton before the ribs
were re-based under their breathing pivots.
"""

from __future__ import annotations

import argparse
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from faceforge.app import (
    arm_asset_load, configure_gl_format, configure_logging,
)
from faceforge.appcontext import build_app_context
from faceforge.controllers import build_controllers
from faceforge.controllers.frame_loop import FrameLoop
from faceforge.coordination.asset_load_sequence import LoadStage
from faceforge.core.events import EventType
from faceforge.exercise.catalog import get_exercise_catalog

#: Startup preset: bones only; the viewer mode adds every muscle layer itself.
STARTUP_PRESET = "Skeleton"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="faceforge.exercise_viewer",
                                 description="The exercise viewer, standalone.")
    ap.add_argument("--exercise", default=None,
                    help="catalogue id to demonstrate once loaded (see docs/exercises.md)")
    ap.add_argument("--list", action="store_true", help="print the catalogue ids and exit")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    catalog = get_exercise_catalog()
    if args.list:
        for exercise_id, defn in catalog.items():
            print(f"{exercise_id:32s} {defn.name}")
        return
    if args.exercise is not None and args.exercise not in catalog:
        sys.exit(f"unknown exercise {args.exercise!r}; try --list")

    configure_logging()
    configure_gl_format()
    app = QApplication.instance() or QApplication(sys.argv)

    ctx = build_app_context()
    controllers = build_controllers(ctx)
    ctx.startup_preset = STARTUP_PRESET
    ctx.startup_illustration = None
    sequence = arm_asset_load(ctx, controllers)
    FrameLoop(ctx, controllers, ctx.gl_widget.paintGL).install()
    watch_load_sequence(sequence, lambda: enter_viewer(ctx, args.exercise))
    ctx.window.setWindowTitle("FaceForge — Exercise viewer")
    ctx.window.show()
    sys.exit(app.exec())


def enter_viewer(ctx, exercise_id: str | None = None) -> None:
    """Switch the loaded application into the viewer mode and start ``exercise_id``."""
    ctx.window.set_viewer_mode(True)
    if exercise_id is not None:
        ctx.event_bus.publish(EventType.EXERCISE_SELECTED, exercise_id=exercise_id)


def watch_load_sequence(sequence, on_complete, *, defer=QTimer.singleShot) -> None:
    """Call ``on_complete`` once, from the event loop, when ``sequence`` reaches its last stage.

    Chains onto the sequence's existing stage observer.  The call is deferred
    (``defer(0, fn)``) so it runs after ``sequence.run`` has returned rather
    than inside its final stage.
    """
    previous = sequence.on_stage
    fired = {"done": False}

    def on_stage(stage) -> None:
        if previous is not None:
            previous(stage)
        if stage is LoadStage.COMPLETE and not fired["done"]:
            fired["done"] = True
            defer(0, on_complete)

    sequence.on_stage = on_stage


if __name__ == "__main__":
    main()
