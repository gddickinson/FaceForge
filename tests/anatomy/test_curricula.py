"""Curricula: derivation from configs, tiering, ordering.

Two halves.  The synthetic half builds curricula from a temporary config
directory, so the derivation rules are tested without depending on the shipped
data.  The real half runs against ``assets/config`` -- those files are
committed (they are JSON, not the 1.2 GB STL dataset), so this is not an
asset-heavy test and belongs in the fast tier.
"""

import json

import pytest

from faceforge.anatomy.curricula import (
    TIER_ORDER,
    build_curricula,
    get_curricula,
    missing_topics,
    tier_for_label,
)


# ── tiering rule ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("label,tier", [
    ("Cerebellum", "foundation"),
    ("Frontal bone", "foundation"),
    ("Ninth thoracic vertebra", "intermediate"),
    ("Deep part of right masseter", "advanced"),
    ("", "unclassified"),
    ("   ", "unclassified"),
])
def test_tier_comes_from_preferred_label_token_count(label, tier):
    assert tier_for_label(label) == tier


# ── derivation from a synthetic config tree ──────────────────────────────

@pytest.fixture
def fake_configs(tmp_path):
    (tmp_path / "muscles").mkdir()
    (tmp_path / "skeleton").mkdir()
    (tmp_path / "skull_bones.json").write_text(json.dumps([
        {"name": "Frontal Bone", "stl": "FMA1", "group": "cranium"},
        {"name": "Mandible", "stl": "FMA2"},
        {"name": "Ninth Thoracic", "stl": "FMA3"},
    ]))
    (tmp_path / "muscles" / "expression_muscles.json").write_text(json.dumps([
        {"name": "Frontalis R", "stl": "FMA4", "auMap": {"AU1": 0.7}},
    ]))
    # A parameter file: no name/stl entries, so it is not a curriculum.
    (tmp_path / "joint_limits.json").write_text(json.dumps(
        {"elbow": {"min": 0, "max": 145}}))
    # A nested structure list, to prove the walk is not list-only.
    (tmp_path / "skeleton" / "rib_cage.json").write_text(json.dumps(
        {"left": [{"name": "Rib 1 L", "stl": "FMA5"}],
         "right": [{"name": "Rib 1 R", "stl": "FMA6"}]}))
    return tmp_path


FAKE_FMA = {
    "FMA1": {"display_name": "Frontal Bone", "preferred_label": "Frontal bone",
             "system": "skeletal", "category": "skull_bones"},
    "FMA2": {"display_name": "Mandible", "preferred_label": "Mandible",
             "system": "skeletal", "category": "skull_bones"},
    "FMA3": {"display_name": "Ninth Thoracic",
             "preferred_label": "Ninth thoracic vertebra",
             "system": "skeletal", "category": "skeleton"},
    "FMA4": {"display_name": "Frontalis R", "preferred_label": "Right frontalis",
             "system": "muscular", "category": "muscles"},
    "FMA5": {"display_name": "Rib 1 L", "preferred_label": "Left first rib",
             "system": "skeletal", "category": "skeleton"},
    # FMA6 deliberately absent -> unclassified tier.
}


def test_each_structure_config_becomes_one_curriculum(fake_configs):
    cur = build_curricula(fma=FAKE_FMA, config_dir=fake_configs)
    assert {"skull_bones", "expression_muscles", "rib_cage"} <= set(cur)
    assert "joint_limits" not in cur


def test_curriculum_membership_is_the_config_contents(fake_configs):
    cur = build_curricula(fma=FAKE_FMA, config_dir=fake_configs)
    assert set(cur["skull_bones"].item_ids()) == {"FMA1", "FMA2", "FMA3"}
    assert cur["expression_muscles"].item_ids() == ["FMA4"]


def test_nested_config_entries_are_found(fake_configs):
    cur = build_curricula(fma=FAKE_FMA, config_dir=fake_configs)
    assert set(cur["rib_cage"].item_ids()) == {"FMA5", "FMA6"}


def test_system_curricula_span_config_groups(fake_configs):
    cur = build_curricula(fma=FAKE_FMA, config_dir=fake_configs)
    assert set(cur["system:skeletal"].item_ids()) == {"FMA1", "FMA2", "FMA3", "FMA5"}
    assert cur["system:muscular"].item_ids() == ["FMA4"]


def test_structures_absent_from_the_crosswalk_are_unclassified(fake_configs):
    cur = build_curricula(fma=FAKE_FMA, config_dir=fake_configs)
    item = next(i for i in cur["rib_cage"].items if i.item_id == "FMA6")
    assert item.tier == "unclassified"
    assert item.preferred_label == ""
    assert item.label == "Rib 1 R"        # falls back to the display name


def test_items_are_ordered_by_tier_then_label(fake_configs):
    cur = build_curricula(fma=FAKE_FMA, config_dir=fake_configs)
    skull = cur["skull_bones"]
    assert [i.tier for i in skull.items] == ["foundation", "foundation",
                                             "intermediate"]
    # Within the foundation tier: 1-token "Mandible" before 2-token
    # "Frontal bone".
    assert skull.item_ids("foundation") == ["FMA2", "FMA1"]


def test_ordering_is_deterministic(fake_configs):
    a = build_curricula(fma=FAKE_FMA, config_dir=fake_configs)
    b = build_curricula(fma=FAKE_FMA, config_dir=fake_configs)
    assert {k: v.item_ids() for k, v in a.items()} == \
           {k: v.item_ids() for k, v in b.items()}


def test_tiers_and_counts_report_only_what_is_present(fake_configs):
    cur = build_curricula(fma=FAKE_FMA, config_dir=fake_configs)
    skull = cur["skull_bones"]
    assert skull.tiers == ("foundation", "intermediate")
    assert skull.counts() == {"foundation": 2, "intermediate": 1}
    assert len(skull) == 3


def test_unreadable_config_is_skipped_not_fatal(fake_configs):
    (fake_configs / "broken.json").write_text("{not json")
    cur = build_curricula(fma=FAKE_FMA, config_dir=fake_configs)
    assert "broken" not in cur
    assert "skull_bones" in cur


def test_empty_config_directory_yields_no_curricula(tmp_path):
    assert build_curricula(fma={}, config_dir=tmp_path) == {}


# ── against the shipped configs ──────────────────────────────────────────

def test_real_configs_produce_the_named_study_sets():
    cur = get_curricula()
    assert "skull_bones" in cur
    assert cur["skull_bones"].title == "Skull bones"
    assert cur["expression_muscles"].title == "Muscles of facial expression"
    assert len(cur["skull_bones"]) == 20
    assert len(cur["expression_muscles"]) == 43


def test_real_system_curricula_match_the_crosswalk_totals():
    from faceforge.loaders.stl_batch_loader import load_fma_labels
    fma = load_fma_labels()
    cur = get_curricula()
    for system in ("muscular", "skeletal"):
        expected = sum(1 for v in fma.values() if v.get("system") == system)
        assert len(cur[f"system:{system}"]) == expected, system


def test_every_real_item_has_a_tier_from_the_known_set():
    for curriculum in get_curricula().values():
        for item in curriculum.items:
            assert item.tier in TIER_ORDER


def test_cranial_nerves_are_reported_as_unsupported_by_the_dataset():
    gaps = missing_topics(get_curricula())
    assert "cranial nerves" in gaps
    assert "optic nerve" in gaps["cranial nerves"].lower()
