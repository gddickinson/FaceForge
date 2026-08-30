"""``LAYER_TOGGLED`` dispatch: the one event that carries every visibility change.

The dispatch order in :class:`LayerController` is the part worth testing.  It
is not a chain of alternatives -- one rule returns early and the rest all run
-- and getting that wrong produces bugs that are invisible until a specific
combination of toggles is used.
"""

from __future__ import annotations

import pytest

from faceforge.controllers.layers import LayerController
from faceforge.core.events import EventType

from tests.controllers.fakes import FakeNode


class RecordingLoaders:
    """Stands in for ``DemandLoaders``, recording which layers were loaded."""

    def __init__(self, known=("organs", "skin")):
        self.known = set(known)
        self.calls: list[str] = []

    def loader_for(self, layer):
        if layer not in self.known:
            return None
        return lambda: self.calls.append(layer)


class StubFaceFeatures:
    def __init__(self, categories=None):
        self.categories = categories or {}
        self.set_calls: list[tuple[str, bool]] = []

    def set_category_visible(self, category, visible):
        self.set_calls.append((category, visible))
        for node in self.categories.get(category, []):
            node.visible = visible


class StubPipeline:
    def __init__(self, face_features=None):
        self.face_features = face_features


@pytest.fixture
def wired(ctx):
    loaders = RecordingLoaders()
    controller = LayerController(ctx, loaders)
    controller.subscribe()
    return controller, loaders


# -- Per-structure toggles -------------------------------------------------

@pytest.mark.parametrize("toggle", [
    "organ_Heart", "muscle_arm_muscles_Biceps R", "vasc_Aorta", "brain_Pons",
    "ligaments_ACL", "oral_Tongue", "cns_additional_Dura",
])
def test_structure_toggles_go_straight_to_visibility(ctx, wired, toggle):
    """A structure id addresses one registered node and must not fall through."""
    _, loaders = wired
    ctx.pipeline = StubPipeline(StubFaceFeatures())
    ctx.event_bus.publish(EventType.LAYER_TOGGLED, layer=toggle, visible=False)
    assert ctx.visibility.set_calls == [(toggle, False)]
    assert loaders.calls == [], "a structure toggle must not trigger a load"


def test_structure_prefix_return_skips_the_face_feature_rules(ctx, wired):
    features = StubFaceFeatures()
    ctx.pipeline = StubPipeline(features)
    ctx.event_bus.publish(EventType.LAYER_TOGGLED, layer="organ_eyes", visible=True)
    assert features.set_calls == []


# -- Loading on first enable ----------------------------------------------

def test_enabling_a_layer_loads_it_then_shows_it(ctx, wired):
    controller, loaders = wired
    ctx.pipeline = StubPipeline()
    ctx.event_bus.publish(EventType.LAYER_TOGGLED, layer="organs", visible=True)
    assert loaders.calls == ["organs"]
    assert ctx.visibility.set_calls == [("organs", True)]


def test_hiding_a_layer_never_loads_it(ctx, wired):
    """Unticking a layer that was never loaded must not load it to hide it."""
    _, loaders = wired
    ctx.pipeline = StubPipeline()
    ctx.event_bus.publish(EventType.LAYER_TOGGLED, layer="organs", visible=False)
    assert loaders.calls == []
    assert ctx.visibility.set_calls == [("organs", False)]


def test_a_layer_with_no_loader_is_still_shown(ctx, wired):
    """Most layers are loaded at startup and only need the visibility call."""
    _, loaders = wired
    ctx.pipeline = StubPipeline()
    ctx.event_bus.publish(EventType.LAYER_TOGGLED, layer="ribs", visible=True)
    assert loaders.calls == []
    assert ctx.visibility.set_calls == [("ribs", True)]


# -- Face features ---------------------------------------------------------

def test_face_feature_layer_maps_to_its_category(ctx, wired):
    """The toggle id and the feature category deliberately differ."""
    features = StubFaceFeatures({"nose": []})
    ctx.pipeline = StubPipeline(features)
    ctx.event_bus.publish(EventType.LAYER_TOGGLED, layer="nose_cart", visible=False)
    assert features.set_calls == [("nose", False)]


def test_parent_group_follows_its_visible_children(ctx, wired):
    """``faceFeatureGroup`` gates the subtree, so it must track the categories."""
    eye, ear = FakeNode("eye"), FakeNode("ear")
    features = StubFaceFeatures({"eyes": [eye], "ears": [ear]})
    group = FakeNode("faceFeatureGroup")
    ctx.pipeline = StubPipeline(features)
    ctx.named_nodes["faceFeatureGroup"] = group

    ctx.event_bus.publish(EventType.LAYER_TOGGLED, layer="eyes", visible=False)
    assert group.visible is True, "ears are still on"
    ctx.event_bus.publish(EventType.LAYER_TOGGLED, layer="ears", visible=False)
    assert group.visible is False, "nothing left visible"
    ctx.event_bus.publish(EventType.LAYER_TOGGLED, layer="eyes", visible=True)
    assert group.visible is True


def test_face_feature_rules_are_skipped_before_the_head_loads(ctx, wired):
    ctx.pipeline = StubPipeline(face_features=None)
    ctx.event_bus.publish(EventType.LAYER_TOGGLED, layer="eyes", visible=True)
    assert ctx.visibility.set_calls == [("eyes", True)]


# -- Teeth -----------------------------------------------------------------

def test_teeth_toggle_drives_both_skull_nodes(ctx, wired):
    skull = FakeNode("skullGroup")
    upper, lower = FakeNode("upper_teeth"), FakeNode("lower_teeth")
    other = FakeNode("cranium")
    for node in (upper, lower, other):
        skull.add(node)
    ctx.named_nodes["skullGroup"] = skull
    ctx.pipeline = StubPipeline()

    ctx.event_bus.publish(EventType.LAYER_TOGGLED, layer="teeth", visible=False)
    assert (upper.visible, lower.visible) == (False, False)
    assert other.visible is True, "the teeth toggle must not touch the cranium"

    ctx.event_bus.publish(EventType.LAYER_TOGGLED, layer="teeth", visible=True)
    assert (upper.visible, lower.visible) == (True, True)


def test_teeth_toggle_survives_a_scene_with_no_skull(ctx, wired):
    ctx.pipeline = StubPipeline()
    ctx.event_bus.publish(EventType.LAYER_TOGGLED, layer="teeth", visible=True)
    assert ctx.visibility.set_calls == [("teeth", True)]
