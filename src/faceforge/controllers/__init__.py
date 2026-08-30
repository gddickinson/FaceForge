"""Application event handlers, grouped by concern.

Each controller owns one area of the UI's behaviour, takes an
:class:`~faceforge.appcontext.AppContext` and nothing else, and subscribes its
own handlers.  They were previously ~100 closures inside ``app.main()``, which
is why the whole application layer had no tests: there was no importable name
to construct.

Subscription order is significant
---------------------------------
``EventBus.publish`` calls handlers in subscription order, and two event types
have more than one subscriber:

* ``LAYER_TOGGLED`` -- the layer controller (which may *load* structures) must
  run before the label controller marks its list stale, or the labels rebuild
  without the structures that were just loaded.
* ``STRUCTURES_LOADED`` -- published by the loaders, consumed by the layers tab.

:func:`build_controllers` therefore subscribes in a fixed order that reproduces
the original ``app.main()`` sequence exactly, and several controllers split
their subscriptions into more than one method purely to make that order
expressible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from faceforge.controllers.alignment import AlignmentController
from faceforge.controllers.animation import AnimationController
from faceforge.controllers.body import BodyController
from faceforge.controllers.diagnostics import DiagnosticsController
from faceforge.controllers.display import DisplayController
from faceforge.controllers.expression import ExpressionController
from faceforge.controllers.frame_loop import FrameLoop
from faceforge.controllers.labels import LabelController
from faceforge.controllers.layers import LayerController
from faceforge.controllers.overlays import OverlayController
from faceforge.controllers.scene_view import SceneViewController
from faceforge.controllers.tools import ToolsController
from faceforge.coordination.demand_loaders import DemandLoaders
from faceforge.core.events import EventType

__all__ = [
    "AlignmentController", "AnimationController", "BodyController",
    "Controllers", "DiagnosticsController", "DisplayController",
    "ExpressionController", "FrameLoop", "LabelController", "LayerController",
    "OverlayController", "SceneViewController", "ToolsController",
    "build_controllers",
]


@dataclass
class Controllers:
    """The assembled controllers, so callers can reach one by name."""

    expression: ExpressionController
    body: BodyController
    layers: LayerController
    display: DisplayController
    alignment: AlignmentController
    labels: LabelController
    scene_view: SceneViewController
    animation: AnimationController
    overlays: OverlayController
    diagnostics: DiagnosticsController
    tools: ToolsController
    loaders: DemandLoaders


def build_controllers(ctx: Any) -> Controllers:
    """Construct every controller, subscribe it, and connect the Qt signals.

    The subscription sequence below is deliberately explicit rather than a loop
    over the controllers: it is the application's handler ordering, and it
    reproduces the original ``main()`` order (see the module docstring).
    """
    loaders = DemandLoaders(ctx)
    controllers = Controllers(
        expression=ExpressionController(ctx),
        body=BodyController(ctx),
        layers=LayerController(ctx, loaders),
        display=DisplayController(ctx),
        alignment=AlignmentController(ctx),
        labels=LabelController(ctx),
        scene_view=SceneViewController(ctx),
        animation=AnimationController(ctx),
        overlays=OverlayController(ctx),
        diagnostics=DiagnosticsController(ctx),
        tools=ToolsController(ctx),
        loaders=loaders,
    )

    bus = ctx.event_bus
    controllers.tools.connect()
    controllers.animation.bind_player()

    # -- The main subscription block, in the original order ---------------
    controllers.expression.subscribe()          # AU, expression, head
    controllers.body.subscribe()                # body DOFs, pose, gender
    controllers.layers.subscribe()              # LAYER_TOGGLED (first)
    controllers.expression.subscribe_toggles()  # auto-blink and friends
    controllers.display.subscribe()             # render mode, camera preset
    controllers.scene_view.subscribe()          # scene mode, wrapper
    controllers.animation.subscribe()           # transport
    controllers.display.subscribe_late()        # colour
    controllers.alignment.subscribe()
    controllers.labels.subscribe()              # LABELS_TOGGLED
    controllers.labels.subscribe_invalidation()  # LAYER_TOGGLED (second)
    bus.subscribe(EventType.CLIP_PLANE_CHANGED,
                  controllers.display.on_clip_plane_changed)

    ctx.control_panel.display_tab.label_style_changed.connect(
        controllers.labels.apply_style)
    bus.subscribe(EventType.SKULL_MODE_CHANGED,
                  controllers.display.on_skull_mode_changed)
    bus.subscribe(EventType.EYE_COLOR_SET, controllers.expression.on_eye_color_set)
    bus.subscribe(EventType.STRUCTURES_LOADED,
                  ctx.control_panel.layers_tab.on_structures_loaded)

    controllers.overlays.subscribe_heatmap()
    controllers.overlays.subscribe_search()
    bus.subscribe(EventType.SPEECH_PLAY, controllers.expression.on_speech_play)
    controllers.overlays.subscribe_pathology()

    controllers.tools.connect_display_buttons()
    return controllers
