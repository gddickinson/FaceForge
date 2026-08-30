"""The startup load sequence: ordering, observability, failure attribution.

The ordering is the point.  It used to be implicit in the line order of a
640-line closure, where the only way to check that chains are built after the
rib pivots exist was to read the whole function.  Here it is a list that can be
asserted without running anything.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from faceforge.coordination.asset_load_sequence import (
    LIMB_PIVOT_TOGGLES, SKELETON_TOGGLES, AssetLoadSequence, LoadStage,
    StageFailure,
)

from tests.controllers.fakes import FakeNode


# -- Ordering --------------------------------------------------------------

def test_stage_order_is_available_without_running_anything():
    order = AssetLoadSequence(ctx=None).stage_order()
    assert order[0] is LoadStage.LOAD_HEAD
    assert order[-1] is LoadStage.APPLY_STARTUP_PRESET
    assert len(order) == len(set(order)), "a stage must not run twice"


@pytest.mark.parametrize("earlier,later", [
    # Wiring reads what the phase before it built.
    (LoadStage.LOAD_HEAD, LoadStage.WIRE_HEAD),
    (LoadStage.LOAD_BODY_SKELETON, LoadStage.REGISTER_SKELETON_VISIBILITY),
    # The rib chain is built from pivots body animation collects.
    (LoadStage.WIRE_BODY_ANIMATION, LoadStage.BUILD_SKINNING),
    # The debug tab's tooling is constructed around the skinning system.
    (LoadStage.BUILD_SKINNING, LoadStage.ATTACH_DIAGNOSTICS),
    # A preset drives the UI through the bus, so everything must be wired.
    (LoadStage.FINALISE, LoadStage.APPLY_STARTUP_PRESET),
])
def test_dependency_ordering(earlier, later):
    order = AssetLoadSequence(ctx=None).stage_order()
    assert order.index(earlier) < order.index(later)


def test_iterating_the_sequence_yields_its_stages():
    assert list(AssetLoadSequence(ctx=None)) == \
        AssetLoadSequence(ctx=None).stage_order()


# -- Running ---------------------------------------------------------------

class RecordingSequence(AssetLoadSequence):
    """Replaces every stage body with a recorder, keeping the real ordering."""

    def __init__(self, failing=None, **kw):
        super().__init__(ctx=None, **kw)
        self.ran: list[str] = []
        self.failing = failing
        for stage, _ in AssetLoadSequence(ctx=None).steps():
            setattr(self, stage.value, self._recorder(stage))

    def _recorder(self, stage):
        def run():
            self.ran.append(stage.value)
            if stage is self.failing:
                raise ValueError("boom")
        return run


def test_run_executes_every_stage_in_order():
    seq = RecordingSequence()
    assert seq.run() is LoadStage.COMPLETE
    assert seq.ran == [s.value for s in seq.stage_order()]
    assert seq.completed == seq.stage_order()


def test_the_observer_sees_every_transition_ending_in_complete():
    seen: list[LoadStage] = []
    seq = RecordingSequence(on_stage=seen.append)
    seq.run()
    assert seen == [*seq.stage_order(), LoadStage.COMPLETE]


def test_the_current_stage_is_readable_while_running():
    seen: list[LoadStage] = []
    seq = RecordingSequence()
    seq.on_stage = lambda _: seen.append(seq.stage)
    seq.run()
    assert seen[0] is LoadStage.LOAD_HEAD
    assert seq.stage is LoadStage.COMPLETE


def test_a_failing_wiring_stage_names_itself_and_stops_the_sequence():
    """Attribution is the reason ``StageFailure`` exists."""
    seq = RecordingSequence(failing=LoadStage.WIRE_HEAD)
    with pytest.raises(StageFailure) as excinfo:
        seq.run()
    assert excinfo.value.stage is LoadStage.WIRE_HEAD
    assert isinstance(excinfo.value.cause, ValueError)
    assert seq.ran == ["load_head", "wire_head"], "later stages must not run"
    assert LoadStage.WIRE_HEAD not in seq.completed


# -- Asset phases degrade rather than abort --------------------------------

class StubReport:
    def __init__(self):
        self.records: list[tuple[str, str]] = []
        self.degraded = False

    def record(self, name, exc):
        self.records.append((name, str(exc)))
        self.degraded = True

    def summary(self):
        return "stub report"


def make_ctx(pipeline):
    return SimpleNamespace(pipeline=pipeline, named_nodes={}, simulation=None,
                           visibility=None, skin_chain_ids={})


def test_a_failed_head_load_is_recorded_and_the_load_continues():
    """A missing head STL must not cost the user the body."""
    report = StubReport()

    def explode():
        raise RuntimeError("no skull.stl")

    ctx = make_ctx(SimpleNamespace(load_head=explode, report=report))
    AssetLoadSequence(ctx).load_head()
    assert report.records == [("load_head", "no skull.stl")]


def test_a_failed_body_load_is_recorded_and_the_load_continues():
    report = StubReport()

    def explode():
        raise RuntimeError("no femur.stl")

    ctx = make_ctx(SimpleNamespace(load_body_skeleton=explode, report=report))
    AssetLoadSequence(ctx).load_body_skeleton()
    assert report.records == [("load_body_skeleton", "no femur.stl")]


# -- Chain construction ----------------------------------------------------

def pivot_table():
    """A joint-pivot table with both limbs and one full finger per side."""
    pivots = {}
    for side in ("R", "L"):
        for joint in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle"):
            pivots[f"{joint}_{side}"] = FakeNode(f"{joint}_{side}")
        for seg in ("mc", "prox", "mid", "dist"):
            pivots[f"finger_{side}_1_{seg}"] = FakeNode(f"finger_{side}_1_{seg}")
    return pivots


def chain_ctx(with_ribs=True):
    skeleton = SimpleNamespace(
        pivots={
            "thoracic": [{"level": i, "group": FakeNode(f"T{i}")} for i in (1, 2)],
            "lumbar": [{"level": i, "group": FakeNode(f"L{i}")} for i in (1,)],
        },
        rib_nodes=[], groups={})
    body_anim = SimpleNamespace(
        _rib_pivots=[FakeNode("rib0"), FakeNode("rib1")] if with_ribs else [])
    return SimpleNamespace(
        pipeline=SimpleNamespace(skeleton=skeleton,
                                 joint_setup=SimpleNamespace(pivots=pivot_table())),
        simulation=SimpleNamespace(body_animation=body_anim),
        skin_chain_ids={}, named_nodes={}, visibility=None)


def test_chain_ids_are_recorded_by_name():
    """The loaders name the chains they follow; the numbering is internal."""
    ctx = chain_ctx()
    chains = AssetLoadSequence(ctx).build_joint_chains()
    ids = ctx.skin_chain_ids
    assert ids["spine"] == 0, "the spine is always chain 0"
    for name in ("arm_R", "leg_R", "arm_L", "leg_L", "hand_R_1", "hand_L_1", "ribs"):
        assert name in ids, name
    assert len(chains) == len(ids)
    assert sorted(ids.values()) == list(range(len(chains)))


def test_the_spine_chain_runs_thoracic_then_lumbar():
    ctx = chain_ctx()
    chains = AssetLoadSequence(ctx).build_joint_chains()
    assert [name for name, _ in chains[ctx.skin_chain_ids["spine"]]] == \
        ["thoracic_1", "thoracic_2", "lumbar_1"]


def test_a_limb_chain_is_ordered_proximal_to_distal():
    ctx = chain_ctx()
    chains = AssetLoadSequence(ctx).build_joint_chains()
    assert [name for name, _ in chains[ctx.skin_chain_ids["arm_R"]]] == \
        ["shoulder_R", "elbow_R", "wrist_R"]


def test_empty_chains_are_not_registered():
    """A digit with no pivots must not take a chain id and freeze meshes to it."""
    ctx = chain_ctx()
    AssetLoadSequence(ctx).build_joint_chains()
    assert "hand_R_2" not in ctx.skin_chain_ids
    assert "foot_R_1" not in ctx.skin_chain_ids


def test_no_rib_pivots_means_no_rib_chain():
    ctx = chain_ctx(with_ribs=False)
    AssetLoadSequence(ctx).build_joint_chains()
    assert "ribs" not in ctx.skin_chain_ids


def test_rebuilding_chains_does_not_accumulate_ids():
    """The gender slider rebuilds on release; ids must be reassigned, not appended."""
    ctx = chain_ctx()
    seq = AssetLoadSequence(ctx)
    first = seq.build_joint_chains()
    ids_first = dict(ctx.skin_chain_ids)
    second = seq.build_joint_chains()
    assert len(second) == len(first)
    assert ctx.skin_chain_ids == ids_first


def test_a_degraded_skeleton_still_produces_a_usable_chain_table():
    ctx = SimpleNamespace(
        pipeline=SimpleNamespace(skeleton=None, joint_setup=None),
        simulation=SimpleNamespace(body_animation=None),
        skin_chain_ids={}, named_nodes={}, visibility=None)
    assert AssetLoadSequence(ctx).build_joint_chains() == []
    assert ctx.skin_chain_ids == {}


# -- Visibility registration ----------------------------------------------

class RecordingVisibility:
    def __init__(self):
        self.registered: list[tuple[str, object]] = []

    def register(self, toggle_id, node):
        self.registered.append((toggle_id, node))


def test_limb_toggles_register_pivots_not_the_emptied_bone_groups():
    """``setup_from_skeleton()`` reparents bones, so the pivot is what hides."""
    ctx = chain_ctx()
    ctx.visibility = RecordingVisibility()
    ctx.pipeline.skeleton.groups = {k: FakeNode(k) for k in SKELETON_TOGGLES}
    AssetLoadSequence(ctx).register_skeleton_visibility()

    registered = dict(ctx.visibility.registered)
    for toggle in SKELETON_TOGGLES.values():
        assert toggle in registered, toggle
    names = [(t, n.name) for t, n in ctx.visibility.registered]
    assert ("upper_limb_skel", "shoulder_R") in names
    assert ("upper_limb_skel", "shoulder_L") in names
    assert ("lower_limb_skel", "hip_R") in names


def test_every_limb_toggle_registers_both_sides():
    ctx = chain_ctx()
    ctx.visibility = RecordingVisibility()
    ctx.pipeline.skeleton.groups = {}
    AssetLoadSequence(ctx).register_skeleton_visibility()
    counts: dict[str, int] = {}
    for toggle, _ in ctx.visibility.registered:
        counts[toggle] = counts.get(toggle, 0) + 1
    for toggle, prefix in LIMB_PIVOT_TOGGLES.items():
        if prefix in ("shoulder", "hip", "wrist", "ankle"):
            assert counts.get(toggle) == 2, toggle
