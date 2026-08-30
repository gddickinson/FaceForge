"""Layer and per-structure visibility, including load-on-first-enable.

One event, ``LAYER_TOGGLED``, carries every visibility change in the
application: whole layers ("organs"), individual structures
("organ_Heart"), face feature sub-categories ("eyes") and the two skull teeth
nodes.  This controller is the dispatcher.

Order matters here and is not incidental:

1. A per-structure toggle is handled and **returns**.  Those ids are
   registered directly with the visibility manager and must not fall through
   into the layer logic, where the prefix would not match any loader and the
   name would be registered a second time.
2. Everything else may match *several* of the remaining rules, so they all run
   in sequence rather than as alternatives -- ``visibility.set_visible`` at the
   end applies to every non-structure toggle regardless of which rules fired.
"""

from __future__ import annotations

import logging
from typing import Any

from faceforge.core.events import EventType

logger = logging.getLogger(__name__)


#: Per-structure toggle id prefixes.  A toggle id starting with one of these
#: addresses a single registered node, not a layer.
STRUCTURE_PREFIXES = (
    "organ_", "muscle_", "vasc_", "brain_",
    "pelvic_floor_", "ligaments_", "oral_",
    "cardiac_additional_", "intestinal_", "cns_additional_",
)

#: Layer id -> face feature category name.  The ids differ from the category
#: names because the UI names the toggle after the structure ("nose_cart") and
#: the feature system after the region ("nose").
FACE_FEATURE_CATEGORIES = {
    "eyes": "eyes",
    "ears": "ears",
    "nose_cart": "nose",
    "eyebrows": "eyebrows",
    "throat": "throat",
}

#: Nodes the single "teeth" toggle controls, inside ``skullGroup``.
TEETH_NODES = ("upper_teeth", "lower_teeth")


class LayerController:
    """Dispatches ``LAYER_TOGGLED`` to visibility, features and loaders."""

    def __init__(self, ctx: Any, loaders: Any) -> None:
        self.ctx = ctx
        self.loaders = loaders

    def subscribe(self) -> None:
        self.ctx.event_bus.subscribe(EventType.LAYER_TOGGLED,
                                     self.on_layer_toggled)

    def on_layer_toggled(self, layer: str = "", visible: bool = True,
                         **kw) -> None:
        if any(layer.startswith(p) for p in STRUCTURE_PREFIXES):
            self.ctx.visibility.set_visible(layer, visible)
            return

        self.set_face_feature_visible(layer, visible)

        if visible:
            loader = self.loaders.loader_for(layer)
            if loader is not None:
                loader()

        if layer == "teeth":
            self.set_teeth_visible(visible)

        self.ctx.visibility.set_visible(layer, visible)

    # -- Face features -----------------------------------------------------

    def set_face_feature_visible(self, layer: str, visible: bool) -> None:
        """Apply a face feature sub-category toggle, if *layer* is one."""
        features = getattr(self.ctx.pipeline, "face_features", None)
        if features is None or layer not in FACE_FEATURE_CATEGORIES:
            return
        features.set_category_visible(FACE_FEATURE_CATEGORIES[layer], visible)
        self.sync_face_feature_group()

    def sync_face_feature_group(self) -> None:
        """Show ``faceFeatureGroup`` when any sub-category has a visible node.

        The parent group gates the whole subtree, so it has to follow its
        children: left visible with everything off it costs a traversal, left
        hidden with a category on hides a structure the user just enabled.
        """
        group = self.ctx.node("faceFeatureGroup")
        features = getattr(self.ctx.pipeline, "face_features", None)
        if group is None or features is None:
            return
        group.visible = any(
            node.visible
            for nodes in features.categories.values()
            for node in nodes
        )

    # -- Teeth -------------------------------------------------------------

    def set_teeth_visible(self, visible: bool) -> None:
        """The teeth toggle drives two nodes inside the skull group."""
        skull_grp = self.ctx.node("skullGroup")
        if skull_grp is None:
            return
        for name in TEETH_NODES:
            node = skull_grp.find(name)
            if node is not None:
                node.visible = visible
