"""What happens on every frame, and in what order.

The frame is driven by the GL widget's paint timer rather than a loop of our
own: ``paintGL`` is wrapped so simulation runs immediately before the draw it
feeds.  Anything that reads deformed vertex positions therefore sees this
frame's positions, not the previous frame's.

The order below is the order things depend on each other:

1. **Speech** writes AU targets, so it runs before the simulation that
   interpolates toward them.
2. **Simulation** advances every system by ``dt`` and deforms the meshes.
3. **Draw** -- the original ``paintGL``.
4. **Transport and labels** read what the frame produced.  Labels come last
   because they need the camera's final view-projection.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


class FrameLoop:
    """Wraps ``paintGL`` with the per-frame simulation and overlay work."""

    def __init__(self, ctx: Any, controllers: Any,
                 original_paint: Callable[[], None]) -> None:
        self.ctx = ctx
        self.controllers = controllers
        self.original_paint = original_paint

    def install(self) -> None:
        """Replace the GL widget's ``paintGL`` with this loop."""
        self.ctx.gl_widget.paintGL = self.paint

    def paint(self) -> None:
        dt = self.ctx.clock.get_delta()
        self.controllers.expression.advance_speech(dt)
        self.ctx.simulation.step(dt)
        self.original_paint()
        self.controllers.animation.update_frame()
        self.controllers.labels.update_frame()
