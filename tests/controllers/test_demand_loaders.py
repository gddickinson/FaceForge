"""On-demand anatomy loading: chain assignment, the load-once guard, dispatch.

The chain-assignment functions are the part of the loading code that encodes
anatomy rather than plumbing -- getting a side wrong makes a muscle follow the
opposite arm -- so they are tested directly, with no scene at all.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from faceforge.coordination.demand_loaders import (
    INTEGUMENT_STLS, LIGAMENT_CHAIN_MAP, MUSCLE_CHAIN_MAP,
    MUSCLE_CHAIN_OVERRIDES, MUSCLE_REGIONS, DemandLoaders, digit_chain_ids,
    resolve_chain_set, resolve_sided_chains,
)
from faceforge.core.events import EventType

from tests.controllers.fakes import FakeMesh, FakeNode

#: A chain id table shaped like a fully built skeleton.
CHAIN_IDS = {
    "spine": 0, "arm_R": 1, "leg_R": 2, "arm_L": 3, "leg_L": 4, "ribs": 30,
    **{f"hand_{s}_{d}": 5 + i for i, (s, d) in enumerate(
        [(s, d) for s in ("R", "L") for d in range(1, 6)])},
    **{f"foot_{s}_{d}": 15 + i for i, (s, d) in enumerate(
        [(s, d) for s in ("R", "L") for d in range(1, 6)])},
}


# -- Chain resolution ------------------------------------------------------

def test_unresolvable_chains_yield_none_not_an_empty_set():
    """``None`` means "no restriction"; ``set()`` would freeze the mesh."""
    assert resolve_chain_set(["nonexistent"], CHAIN_IDS) is None
    assert resolve_chain_set([], CHAIN_IDS) is None


def test_known_chains_resolve_to_their_ids():
    assert resolve_chain_set(["spine", "ribs"], CHAIN_IDS) == {0, 30}


def test_unknown_names_are_dropped_not_fatal():
    assert resolve_chain_set(["spine", "tail"], CHAIN_IDS) == {0}


@pytest.mark.parametrize("name,expected_limb", [
    ("Biceps R", "arm_R"),
    ("Biceps L", "arm_L"),
])
def test_a_sided_muscle_binds_only_to_its_own_limb(name, expected_limb):
    got = resolve_sided_chains(["spine", "arm"], name, CHAIN_IDS)
    assert got == {CHAIN_IDS["spine"], CHAIN_IDS[expected_limb]}


def test_a_midline_structure_binds_to_both_sides():
    """Nothing tells a midline muscle which arm to follow, so it follows both."""
    got = resolve_sided_chains(["arm"], "Diaphragm", CHAIN_IDS)
    assert got == {CHAIN_IDS["arm_R"], CHAIN_IDS["arm_L"]}


def test_hand_token_expands_to_five_digit_chains_on_that_side():
    got = resolve_sided_chains(["hand"], "Lumbrical R", CHAIN_IDS)
    assert got == {CHAIN_IDS[f"hand_R_{d}"] for d in range(1, 6)}


def test_digit_chain_ids_covers_both_sides():
    assert digit_chain_ids("foot", CHAIN_IDS) == \
        {CHAIN_IDS[f"foot_{s}_{d}"] for s in ("R", "L") for d in range(1, 6)}


def test_digit_chain_ids_is_empty_when_no_digits_were_built():
    assert digit_chain_ids("hand", {"spine": 0}) == set()


# -- The anatomy tables ----------------------------------------------------

def test_every_muscle_region_has_a_default_chain_set():
    assert set(MUSCLE_REGIONS) == set(MUSCLE_CHAIN_MAP)


def test_overrides_are_all_sided_or_midline_names_and_non_empty():
    for name, chains in MUSCLE_CHAIN_OVERRIDES.items():
        assert chains, f"{name} has an empty override, which would mean 'no chains'"
        assert set(chains) <= {"spine", "ribs", "arm", "leg"}, name


def test_pectorals_follow_the_arm_and_intercostals_do_not():
    """The documented rationale, asserted: rib-to-humerus muscles need the arm."""
    assert "arm" in MUSCLE_CHAIN_OVERRIDES["Pect. Major Clav. R"]
    assert "arm" not in MUSCLE_CHAIN_OVERRIDES["Ext. Intercostal"]
    assert "arm" in MUSCLE_CHAIN_OVERRIDES["Latissimus Dorsi L"]
    assert "arm" not in MUSCLE_CHAIN_OVERRIDES["Serratus Ant. R"]


def test_ligament_categories_cover_the_config_vocabulary():
    assert set(LIGAMENT_CHAIN_MAP) == {"upper_limb", "lower_limb", "trunk", "hip"}


# -- Dispatch table --------------------------------------------------------

def test_loader_for_covers_every_muscle_region_and_integument(ctx):
    loaders = DemandLoaders(ctx)
    for layer in list(MUSCLE_REGIONS) + list(INTEGUMENT_STLS):
        assert loaders.loader_for(layer) is not None, layer


def test_loader_for_returns_none_for_startup_layers(ctx):
    """Layers loaded at startup have no on-demand loader, and must not fake one."""
    loaders = DemandLoaders(ctx)
    for layer in ("ribs", "pelvis", "thoracic", "teeth", "eyes"):
        assert loaders.loader_for(layer) is None, layer


@pytest.mark.parametrize("layer", [
    "organs", "vasculature", "brain", "skin", "hand_muscles", "foot_muscles",
    "pelvic_floor", "ligaments", "oral", "cardiac_additional", "intestinal",
    "cns_additional",
])
def test_every_named_on_demand_layer_has_a_loader(ctx, layer):
    assert DemandLoaders(ctx).loader_for(layer) is not None


# -- The load-once guard ---------------------------------------------------

def test_the_guard_is_claimed_before_the_load_runs(ctx):
    """A load that raises must not be retried on every tick of the checkbox."""
    loaders = DemandLoaders(ctx)
    attempts = []

    class ExplodingAssets:
        stl_dir = "."
        transform = None

        def load_organs(self):
            attempts.append(1)
            raise RuntimeError("missing STL")

    ctx.assets = ExplodingAssets()
    ctx.named_nodes["bodyRoot"] = FakeNode("bodyRoot")
    for _ in range(3):
        loaders.load_organs()
    assert attempts == [1], "a failed load must be attempted once, not repeatedly"
    assert "organs" in loaders.loaded


def test_a_load_with_no_body_root_is_not_retried_either(ctx):
    """The claim is taken before the parent check, matching the original."""
    loaders = DemandLoaders(ctx)
    loaders.load_organs()
    assert "organs" in loaders.loaded


# -- A full loader run -----------------------------------------------------

class StubResult:
    def __init__(self, meshes, nodes, group):
        self.meshes, self.nodes, self.group = meshes, nodes, group


class StubSkinning:
    def __init__(self):
        self.registered: list[dict] = []
        self.attachment_system = None
        self.bindings: list = []

    def register_skin_mesh(self, mesh, **kw):
        self.registered.append({"mesh": mesh.name, **kw})


def test_organs_bind_to_the_spine_only(ctx, monkeypatch):
    """A kidney that swings with the arm is a defect, so organs skip the limbs."""
    meshes = [FakeMesh("Heart"), FakeMesh("Liver")]
    nodes = [FakeNode("Heart"), FakeNode("Liver")]
    group = FakeNode("organs")
    ctx.assets = SimpleNamespace(load_organs=lambda: StubResult(meshes, nodes, group))
    ctx.named_nodes["bodyRoot"] = FakeNode("bodyRoot")
    ctx.skin_chain_ids.update(CHAIN_IDS)
    skinning = StubSkinning()
    ctx.simulation = SimpleNamespace(soft_tissue=skinning, physiology=None)

    monkeypatch.setattr("faceforge.coordination.demand_loaders.load_config",
                        lambda name: [{"name": "Heart", "category": "cardiac"},
                                      {"name": "Liver", "category": "digestive"}])

    announced = []
    ctx.event_bus.subscribe(EventType.STRUCTURES_LOADED,
                            lambda **kw: announced.append(kw))
    DemandLoaders(ctx).load_organs()

    assert [r["allowed_chains"] for r in skinning.registered] == \
        [{CHAIN_IDS["spine"]}] * 2
    assert all(r["is_muscle"] is False for r in skinning.registered)
    assert announced == [{"group_id": "organs", "items": [
        {"toggle_id": "organ_Heart", "name": "Heart", "category": "cardiac"},
        {"toggle_id": "organ_Liver", "name": "Liver", "category": "digestive"},
    ]}]
    assert ("organs", group) in ctx.visibility.registered


def test_post_registration_hooks_run_after_a_load(ctx, monkeypatch):
    """This is how saved chain overrides reach layers loaded later."""
    group = FakeNode("skin")
    ctx.assets = SimpleNamespace(
        load_skin=lambda: StubResult([FakeMesh("body_skin")], [FakeNode("s")], group))
    ctx.named_nodes["bodyRoot"] = FakeNode("bodyRoot")
    ctx.simulation = SimpleNamespace(soft_tissue=StubSkinning())
    fired = []
    ctx.after_registration_hooks.append(lambda: fired.append(1))

    DemandLoaders(ctx).load_skin()
    assert fired == [1]
