"""Body pose, the two-stage gender morph, and the search/pathology overlays."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from faceforge.controllers.body import BodyController
from faceforge.controllers.labels import LabelController
from faceforge.controllers.overlays import DIMMED_OPACITY, OverlayController
from faceforge.core.events import EventType

from tests.controllers.fakes import FakeMesh, FakeNode


# ── Body pose ────────────────────────────────────────────────────────────

@pytest.fixture
def body(ctx):
    controller = BodyController(ctx)
    controller.subscribe()
    return controller


def test_js_field_names_are_translated(ctx, state, body):
    """The UI speaks camelCase; the state object speaks snake_case."""
    ctx.event_bus.publish(EventType.BODY_STATE_CHANGED,
                          field="spineFlex", value=15.0)
    assert state.target_body.spine_flex == 15.0


def test_a_python_field_name_also_works(ctx, state, body):
    ctx.event_bus.publish(EventType.BODY_STATE_CHANGED,
                          field="spine_flex", value=7.0)
    assert state.target_body.spine_flex == 7.0


def test_an_unknown_field_is_ignored_not_created(ctx, state, body):
    ctx.event_bus.publish(EventType.BODY_STATE_CHANGED,
                          field="not_a_dof", value=1.0)
    assert not hasattr(state.target_body, "not_a_dof")


def test_an_empty_field_name_is_ignored(ctx, state, body):
    """An empty field name must write nothing, not write somewhere arbitrary.

    Previously this only checked that publishing did not raise, which it would
    also have passed had the handler created an attribute named "" or written
    to an unrelated field.
    """
    before = {f: getattr(state.target_body, f)
              for f in vars(state.target_body)}

    ctx.event_bus.publish(EventType.BODY_STATE_CHANGED, field="", value=1.0)

    after = {f: getattr(state.target_body, f) for f in vars(state.target_body)}
    assert after == before, (
        "publishing an empty field name changed state: "
        f"{ {k: (before[k], after[k]) for k in after if after[k] != before.get(k)} }"
    )
    assert not hasattr(state.target_body, ""), "an empty attribute was created"


def test_a_bool_valued_flag_is_written_live_as_well_as_to_the_target(ctx, state,
                                                                     body):
    """Flags are copied, not interpolated, so the toggle takes effect at once."""
    flag = state.target_body.FLAG_FIELDS[0]
    ctx.event_bus.publish(EventType.BODY_STATE_CHANGED, field=flag, value=True)
    assert getattr(state.target_body, flag) is True
    assert getattr(state.body, flag) is True


def test_a_float_valued_flag_reaches_the_target_only(ctx, state, body):
    """The live write is keyed off the value's type, so 1.0 is not a bool."""
    flag = state.target_body.FLAG_FIELDS[0]
    ctx.event_bus.publish(EventType.BODY_STATE_CHANGED, field=flag, value=1.0)
    assert getattr(state.target_body, flag) == 1.0
    assert getattr(state.body, flag) is False


def test_a_pose_preset_applies_through_the_js_mapping(ctx, state, body):
    ctx.event_bus.publish(EventType.BODY_POSE_SET, name="T-pose",
                          values={"spineFlex": 5.0})
    assert state.target_body.spine_flex == 5.0


def test_an_empty_pose_preset_changes_nothing(ctx, state, body):
    state.target_body.spine_flex = 3.0
    ctx.event_bus.publish(EventType.BODY_POSE_SET, name="empty", values={})
    assert state.target_body.spine_flex == 3.0


# ── Gender morph: the cheap path and the expensive one ───────────────────

class StubMorph:
    def __init__(self, loaded=True):
        self.loaded = loaded
        self.gender_calls: list[float] = []
        self.scaled: list[int] = []

    def set_gender(self, gender):
        self.gender_calls.append(gender)

    def scale_skeleton(self, bone_meshes):
        self.scaled.append(len(bone_meshes))
        return len(bone_meshes)


def test_dragging_the_slider_morphs_the_surface_but_never_the_skeleton(ctx, state,
                                                                      body):
    """The expensive path per drag event would freeze the render loop."""
    morph = StubMorph()
    ctx.pipeline = SimpleNamespace(gender_morph=morph)
    ctx.event_bus.publish(EventType.GENDER_CHANGED, gender=0.5)
    assert morph.gender_calls == [0.5]
    assert morph.scaled == [], "bone scaling must wait for release"
    assert state.body.gender == 0.5
    assert state.target_body.gender == 0.5


def test_releasing_the_slider_scales_bones_and_restores_their_rest_pose(ctx, body):
    morph = StubMorph()
    ctx.pipeline = SimpleNamespace(gender_morph=morph)
    root = FakeNode("bodyRoot")
    femur = FakeNode("femur_R", FakeMesh("femur_R"))
    surface = FakeNode("body_surface", FakeMesh("body_surface"))
    root.add(femur)
    root.add(surface)
    ctx.named_nodes["bodyRoot"] = root
    ctx.simulation = SimpleNamespace(soft_tissue=None)

    ctx.event_bus.publish(EventType.GENDER_RELEASED, gender=1.0)
    assert morph.scaled == [1], "the body surface is skin, not bone"
    assert femur.mesh.rest_pose_stored == 1
    assert surface.mesh.rest_pose_stored == 0


def test_bone_collection_walks_the_reparented_hierarchy(ctx, body):
    """Bones live under pivots after setup, not in their original groups."""
    root = FakeNode("bodyRoot")
    pivot = FakeNode("shoulder_R")
    humerus = FakeNode("humerus_R", FakeMesh("humerus_R"))
    pivot.add(humerus)
    root.add(pivot)
    ctx.named_nodes["bodyRoot"] = root
    assert [n for n, _ in BodyController(ctx).collect_bone_meshes()] == ["humerus_R"]


def test_bone_collection_skips_unnamed_and_meshless_nodes(ctx, body):
    root = FakeNode("bodyRoot")
    root.add(FakeNode("", FakeMesh("anonymous")))
    root.add(FakeNode("group_only", None))
    ctx.named_nodes["bodyRoot"] = root
    assert BodyController(ctx).collect_bone_meshes() == []


def test_an_unloaded_morph_does_nothing_on_release(ctx, body):
    morph = StubMorph(loaded=False)
    ctx.pipeline = SimpleNamespace(gender_morph=morph)
    ctx.event_bus.publish(EventType.GENDER_RELEASED, gender=0.5)
    assert morph.gender_calls == []
    assert morph.scaled == []


# -- Re-registration after a rescale --------------------------------------

class Binding:
    """Stands in for ``SkinBinding`` in the rebind path.

    ``rebind_kwargs`` is part of that interface: the constraints a mesh was
    solved with (allowed_chains, spatial_limit, chain_z_margin) have to survive
    a skeleton rebuild, or the re-solve binds vertices to whatever chain is
    nearest and torso geometry starts following limb motion.  A double that
    omits it does not exercise the real contract.
    """

    def __init__(self, name, is_muscle=False, allowed_chains=None):
        self.mesh = FakeMesh(name)
        self.is_muscle = is_muscle
        self.muscle_name = name if is_muscle else None
        self.allowed_chains = allowed_chains
        self.spatial_limit = None
        self.chain_z_margin = None
        self.use_geodesic = True
        self.head_follow_config = None

    def rebind_kwargs(self):
        return {
            "is_muscle": self.is_muscle,
            "allowed_chains": self.allowed_chains,
            "spatial_limit": self.spatial_limit,
            "chain_z_margin": self.chain_z_margin,
            "use_geodesic": self.use_geodesic,
            "head_follow_config": self.head_follow_config,
            "muscle_name": self.muscle_name,
        }


class RebindSkinning:
    def __init__(self):
        self.bindings = [Binding("skin"), Binding("Biceps R", True)]
        self.cleared = 0
        self.rebuilt: list = []
        self.registered: list[str] = []

    def clear_bindings(self):
        self.cleared += 1

    def rebuild_skin_joints(self, chains):
        self.rebuilt.append(chains)

    def register_skin_mesh(self, mesh, **kw):
        self.registered.append(mesh.name)


def test_rescaling_rebuilds_the_chains_and_re_registers_every_mesh(ctx):
    skinning = RebindSkinning()
    ctx.simulation = SimpleNamespace(soft_tissue=skinning)
    ctx.named_nodes["bodyRoot"] = FakeNode("bodyRoot")
    ctx.joint_chain_builder = lambda: [["chain"]]

    BodyController(ctx).rebind_skinning()
    assert skinning.cleared == 1
    assert skinning.rebuilt == [[["chain"]]]
    assert skinning.registered == ["skin", "Biceps R"]
    assert ctx.scene.updates == 1


def test_a_mesh_that_fails_to_re_register_does_not_stop_the_others(ctx):
    skinning = RebindSkinning()

    def register(mesh, **kw):
        if mesh.name == "skin":
            raise RuntimeError("no matching joints")
        skinning.registered.append(mesh.name)

    skinning.register_skin_mesh = register
    ctx.simulation = SimpleNamespace(soft_tissue=skinning)
    ctx.named_nodes["bodyRoot"] = FakeNode("bodyRoot")
    ctx.joint_chain_builder = lambda: [["chain"]]

    BodyController(ctx).rebind_skinning()
    assert skinning.registered == ["Biceps R"]


def test_no_chains_means_no_destructive_rebuild(ctx):
    """Clearing bindings and then failing to rebuild would unbind the whole body."""
    skinning = RebindSkinning()
    ctx.simulation = SimpleNamespace(soft_tissue=skinning)
    ctx.named_nodes["bodyRoot"] = FakeNode("bodyRoot")
    ctx.joint_chain_builder = lambda: []

    BodyController(ctx).rebind_skinning()
    assert skinning.cleared == 0
    assert skinning.registered == []


# ── Search highlighting ──────────────────────────────────────────────────

class StubSearchIndex:
    def __init__(self, results):
        self.results = results

    def search(self, query):
        return self.results


@pytest.fixture
def overlays(ctx):
    ctx.scene.meshes = [FakeMesh("Mandible"), FakeMesh("Maxilla"), FakeMesh("Heart")]
    controller = OverlayController(ctx)
    controller.subscribe_search()
    controller.subscribe_pathology()
    return controller


def opacities(ctx):
    return {m.name: m.material.opacity for m in ctx.scene.meshes}


def test_a_search_dims_everything_that_does_not_match(ctx, overlays):
    """Dimming rather than tinting: the target is usually inside something."""
    ctx.search_index = StubSearchIndex([SimpleNamespace(mesh_name="Mandible")])
    ctx.event_bus.publish(EventType.STRUCTURE_SEARCH, query="mandible")
    assert opacities(ctx) == {"Mandible": 1.0, "Maxilla": DIMMED_OPACITY,
                              "Heart": DIMMED_OPACITY}


def test_clearing_the_query_restores_full_opacity(ctx, overlays):
    ctx.search_index = StubSearchIndex([SimpleNamespace(mesh_name="Mandible")])
    ctx.event_bus.publish(EventType.STRUCTURE_SEARCH, query="mandible")
    ctx.event_bus.publish(EventType.STRUCTURE_SEARCH, query="")
    assert set(opacities(ctx).values()) == {1.0}
    assert overlays.highlighting is False


def test_an_empty_query_does_not_stamp_over_another_overlay(ctx, overlays):
    """Pathology and the quiz also own opacity; only undo our own dimming."""
    ctx.scene.meshes[0].material.opacity = 0.3
    ctx.search_index = StubSearchIndex([])
    ctx.event_bus.publish(EventType.STRUCTURE_SEARCH, query="")
    assert ctx.scene.meshes[0].material.opacity == 0.3


def test_a_query_that_matches_nothing_leaves_the_view_alone(ctx, overlays):
    """A typo must not blank the model."""
    ctx.search_index = StubSearchIndex([])
    ctx.event_bus.publish(EventType.STRUCTURE_SEARCH, query="zzzz")
    assert set(opacities(ctx).values()) == {1.0}
    assert overlays.highlighting is False


# ── Pathology ────────────────────────────────────────────────────────────

class StubPathology:
    def __init__(self):
        self.calls: list[tuple] = []

    def clear_all(self):
        self.calls.append(("clear_all",))

    def remove_condition(self, target):
        self.calls.append(("remove", target))

    def add_condition(self, target, condition, severity):
        self.calls.append(("add", target, condition, severity))


def test_changing_severity_replaces_rather_than_stacks(ctx, overlays):
    ctx.pathology = StubPathology()
    ctx.event_bus.publish(EventType.PATHOLOGY_CHANGED, condition="atrophy",
                          target="Heart", severity=0.5)
    assert ctx.pathology.calls == [("remove", "Heart"),
                                   ("add", "Heart", "atrophy", 0.5)]


def test_zero_severity_removes_without_adding(ctx, overlays):
    ctx.pathology = StubPathology()
    ctx.event_bus.publish(EventType.PATHOLOGY_CHANGED, condition="atrophy",
                          target="Heart", severity=0.0)
    assert ctx.pathology.calls == [("remove", "Heart")]


def test_condition_none_clears_everything(ctx, overlays):
    ctx.pathology = StubPathology()
    ctx.event_bus.publish(EventType.PATHOLOGY_CHANGED, condition="none")
    assert ctx.pathology.calls == [("clear_all",)]


# ── Labels ───────────────────────────────────────────────────────────────

def test_labels_rebuild_lazily_and_only_once_per_change(ctx):
    """Rebuilding per frame would recompute bounding centres sixty times a second."""
    controller = LabelController(ctx)
    controller.subscribe()
    controller.subscribe_invalidation()
    ctx.scene.meshes = [FakeMesh("Mandible", centre=(1.0, 2.0, 3.0))]

    ctx.event_bus.publish(EventType.LABELS_TOGGLED, enabled=True)
    assert controller.dirty is True
    controller.update_frame()
    assert controller.dirty is False
    assert ctx.label_overlay.labels is not None
    assert ctx.label_overlay.view_proj == "view_proj"

    first = ctx.label_overlay.labels
    controller.update_frame()
    assert ctx.label_overlay.labels is first, "no rebuild while nothing changed"


def test_toggling_a_layer_invalidates_the_label_list(ctx):
    controller = LabelController(ctx)
    controller.subscribe()
    controller.subscribe_invalidation()
    ctx.event_bus.publish(EventType.LABELS_TOGGLED, enabled=True)
    controller.update_frame()
    ctx.event_bus.publish(EventType.LAYER_TOGGLED, layer="organs", visible=True)
    assert controller.dirty is True


def test_disabled_labels_do_no_per_frame_work(ctx):
    controller = LabelController(ctx)
    controller.update_frame()
    assert ctx.label_overlay.updates == 0
    assert ctx.label_overlay.labels is None


def test_the_face_skin_and_unnamed_meshes_are_never_labelled(ctx):
    """A label on the face lands in the middle of what the user is looking at."""
    controller = LabelController(ctx)
    ctx.scene.meshes = [FakeMesh("face"), FakeMesh(""), FakeMesh("Mandible")]
    controller.rebuild()
    assert [name for name, _ in ctx.label_overlay.labels] == ["Mandible"]
