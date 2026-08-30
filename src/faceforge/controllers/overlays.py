"""Overlays that answer "where is it?": search highlight, heatmap, pathology.

Search highlighting works by opacity rather than colour, because the structures
being looked for are usually *inside* something else: dimming everything else to
0.15 makes a named muscle visible through the skin without changing its own
appearance, which colour tinting would.

The dimming is tracked with a flag so that clearing the query restores full
opacity exactly once.  Restoring on every empty-query event would fight the
pathology and quiz overlays, which also own mesh opacity.
"""

from __future__ import annotations

import logging
from typing import Any

from faceforge.core.events import EventType

logger = logging.getLogger(__name__)

#: Opacity a non-matching mesh is dimmed to while a search is active.
DIMMED_OPACITY = 0.15


class OverlayController:
    """Handlers for structure search, the muscle heatmap and pathology."""

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        #: True while non-matching meshes are dimmed by a search.
        self.highlighting = False

    def subscribe_heatmap(self) -> None:
        self.ctx.event_bus.subscribe(EventType.HEATMAP_TOGGLED,
                                     self.on_heatmap_toggled)

    def subscribe_search(self) -> None:
        self.ctx.event_bus.subscribe(EventType.STRUCTURE_SEARCH,
                                     self.on_structure_search)

    def subscribe_pathology(self) -> None:
        self.ctx.event_bus.subscribe(EventType.PATHOLOGY_CHANGED,
                                     self.on_pathology_changed)

    # -- Heatmap -----------------------------------------------------------

    def on_heatmap_toggled(self, enabled: bool = False, **kw) -> None:
        self.ctx.muscle_activation.set_enabled(enabled)

    # -- Search ------------------------------------------------------------

    def on_structure_search(self, query: str = "", **kw) -> None:
        """Dim everything that does not match *query*.

        An empty query restores opacity, but only if this controller was the
        one that dimmed it -- see the module docstring.  A query that matches
        nothing leaves the view untouched rather than blanking it, so a typo
        does not hide the model.
        """
        scene = self.ctx.scene
        if not query:
            if self.highlighting:
                for mesh, _ in scene.collect_meshes():
                    mesh.material.opacity = 1.0
                self.highlighting = False
            return

        results = self.ctx.search_index.search(query)
        if not results:
            return
        matched = {r.mesh_name for r in results}
        for mesh, _ in scene.collect_meshes():
            mesh.material.opacity = (
                1.0 if mesh.name in matched else DIMMED_OPACITY)
        self.highlighting = True

    # -- Pathology ---------------------------------------------------------

    def on_pathology_changed(self, condition: str = "none", target: str = "",
                             severity: float = 0.0, **kw) -> None:
        """Apply, replace or clear a pathological condition on one structure.

        The existing condition on *target* is removed before the new one is
        added, so dragging the severity slider replaces rather than stacks.
        Severity zero is a removal, which is why the add is conditional.
        """
        pathology = self.ctx.pathology
        if condition == "none":
            pathology.clear_all()
            return
        pathology.remove_condition(target)
        if severity > 0:
            pathology.add_condition(target, condition, severity)

    # -- Physiology particle overlays --------------------------------------

    def on_physiology_blood_flow(self, enabled: bool = False, **kw) -> None:
        """Clear the blood particle system when the overlay is turned off.

        Turning it *on* costs nothing here: the particle system is filled by
        the physiology simulation, not by this handler.
        """
        if not enabled:
            self.ctx.blood_particles.clear()

    def on_physiology_neural(self, enabled: bool = False, **kw) -> None:
        if not enabled:
            self.ctx.neural_particles.clear()
