"""Anatomical name labels drawn over the viewport.

Labels are rebuilt lazily, not on every frame: the list depends only on which
meshes are visible, so it is invalidated when a layer is toggled and rebuilt at
most once per frame, on the next frame that actually draws labels.  Rebuilding
per frame would mean a bounding-centre computation and a world transform per
visible mesh, sixty times a second, for a list that changes only when the user
ticks a checkbox.

The label *positions* do update every frame -- they follow the camera -- but
that is a matrix upload, not a rebuild.
"""

from __future__ import annotations

import logging
from typing import Any

from faceforge.core.events import EventType
from faceforge.core.math_utils import transform_point

logger = logging.getLogger(__name__)

#: Meshes never labelled.  The face is a single skin surface covering the whole
#: head, so a label on it lands in the middle of the face and covers what the
#: user is looking at.
UNLABELLED_MESHES = ("face",)


class LabelController:
    """Owns label enable state and the lazy rebuild of the label list."""

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        self.enabled = False
        #: Set when the visible mesh set may have changed.
        self.dirty = True

    def subscribe(self) -> None:
        self.ctx.event_bus.subscribe(EventType.LABELS_TOGGLED,
                                     self.on_labels_toggled)

    def subscribe_invalidation(self) -> None:
        """Invalidate the label list whenever a layer is toggled.

        Subscribed separately, and after the layer controller, because the
        layer handler may load new structures and this must see the result.
        """
        self.ctx.event_bus.subscribe(EventType.LAYER_TOGGLED, self.on_layer_toggled)

    def on_labels_toggled(self, enabled: bool = False, **kw) -> None:
        self.enabled = enabled
        self.dirty = True
        self.ctx.label_overlay.set_enabled(enabled)

    def on_layer_toggled(self, **kw) -> None:
        self.dirty = True

    def apply_style(self) -> None:
        """Push the display tab's label style controls into the overlay."""
        style = self.ctx.control_panel.display_tab.get_label_style()
        self.ctx.label_overlay.apply_style(style)

    # -- Rebuild -----------------------------------------------------------

    def rebuild(self) -> None:
        """Rebuild the label list from the currently visible meshes."""
        self.dirty = False
        labels = []
        for mesh, world_mat in self.ctx.scene.collect_meshes():
            name = mesh.name
            if not name or name in UNLABELLED_MESHES:
                continue
            centre = mesh.geometry.get_bounding_center()
            labels.append((name, transform_point(world_mat, centre)))
        self.ctx.label_overlay.set_labels(labels)

    def rebuild_illustration(self) -> None:
        """Rebuild labels in illustration mode, which curates its own list."""
        self.dirty = False
        from faceforge.ui.illustration_presets import rebuild_illustration_labels

        rebuild_illustration_labels(self.ctx.label_overlay, self.ctx.scene)

    def update_frame(self) -> None:
        """Per-frame label work: rebuild if dirty, then follow the camera."""
        if not self.enabled:
            return
        overlay = self.ctx.label_overlay
        if self.dirty:
            if overlay._illustration_mode:
                self.rebuild_illustration()
            else:
                self.rebuild()
        overlay.set_view_proj(self.ctx.camera.get_view_projection())
        overlay.update()
