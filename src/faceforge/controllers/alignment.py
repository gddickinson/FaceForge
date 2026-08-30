"""Alignment of the scanned face mesh onto the skull.

The face mesh and the skull come from different sources, so the face has to be
scaled, translated and rotated onto the cranium.  There are five degrees of
freedom and they interact, so a single slider cannot be applied on its own:
:func:`~faceforge.anatomy.face.update_alignment` takes the whole transform, and
this handler reads the other four values back off the sliders rather than
caching them, so the tab is the single source of truth for the alignment.

The defaults in :data:`ALIGNMENT_DEFAULTS` are the values the fit was
established at; a missing slider falls back to them rather than to zero, which
would throw the face off the skull entirely.
"""

from __future__ import annotations

import logging
from typing import Any

from faceforge.core.events import EventType

logger = logging.getLogger(__name__)

#: Fallback for any alignment field the tab does not report.
ALIGNMENT_DEFAULTS = {
    "scale": 1.14,
    "offset_x": -0.2,
    "offset_y": -10.6,
    "offset_z": 9.5,
    "rot_x": 88.5,
}


class AlignmentController:
    """Handler for the align tab's five face-fitting sliders."""

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx

    def subscribe(self) -> None:
        self.ctx.event_bus.subscribe(EventType.ALIGNMENT_CHANGED,
                                     self.on_alignment_changed)

    def on_alignment_changed(self, field: str = "", value: float = 0.0,
                             **kw) -> None:
        from faceforge.anatomy.face import update_alignment

        face_group = self.ctx.node("faceGroup")
        if face_group is None:
            return
        vals = dict(self.ctx.control_panel.align_tab.get_alignment())
        vals[field] = value
        update_alignment(
            face_group,
            scale=vals.get("scale", ALIGNMENT_DEFAULTS["scale"]),
            offset_x=vals.get("offset_x", ALIGNMENT_DEFAULTS["offset_x"]),
            offset_y=vals.get("offset_y", ALIGNMENT_DEFAULTS["offset_y"]),
            offset_z=vals.get("offset_z", ALIGNMENT_DEFAULTS["offset_z"]),
            rot_x_deg=vals.get("rot_x", ALIGNMENT_DEFAULTS["rot_x"]),
        )
