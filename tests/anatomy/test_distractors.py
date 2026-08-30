"""Distractor selection: the right neighbourhood, deterministically.

The claim under test is the one that makes an item discriminate -- that wrong
options are anatomically adjacent to the right one -- so the assertions are
about *which rung* each distractor came from and whether the provenance
sentence attached to it is true of the data.
"""

import pytest

from faceforge.anatomy.distractors import LADDER, DistractorPool
from faceforge.anatomy.fma_taxonomy import Taxonomy, SCHEMA_VERSION

# 1 bone organ -> {2 flat bone -> {4 frontal, 5 parietal, 7 occipital},
#                  3 irregular bone -> {6 mandible}}
PAYLOAD = {
    "schema_version": SCHEMA_VERSION,
    "_source": "synthetic",
    "nodes": {
        "1": {"label": "Bone organ", "parent": ""},
        "2": {"label": "Flat bone", "parent": "1"},
        "3": {"label": "Irregular bone", "parent": "1"},
        "4": {"label": "Frontal bone", "parent": "2"},
        "5": {"label": "Parietal bone", "parent": "2"},
        "6": {"label": "Mandible", "parent": "3"},
        "7": {"label": "Occipital bone", "parent": "2"},
    },
    "labels": {"FMA100": "neurocranium"},
    "part_of": {"FMA4": ["FMA100"], "FMA5": ["FMA100"], "FMA7": ["FMA100"]},
    "composite_of": {},
}

FMA = {
    "FMA4": {"display_name": "Frontal Bone", "preferred_label": "Frontal bone",
             "system": "skeletal", "category": "skull_bones"},
    "FMA5": {"display_name": "Parietal Bone",
             "preferred_label": "Parietal bone",
             "system": "skeletal", "category": "skull_bones"},
    "FMA6": {"display_name": "Mandible", "preferred_label": "Mandible",
             "system": "skeletal", "category": "skull_bones"},
    "FMA7": {"display_name": "Occipital Bone",
             "preferred_label": "Occipital bone",
             "system": "skeletal", "category": "skull_bones"},
    "FMA9": {"display_name": "Liver", "preferred_label": "Liver",
             "system": "digestive", "category": "organs"},
    "FMA10": {"display_name": "Unlabelled", "preferred_label": "",
              "system": "skeletal", "category": "skull_bones"},
}


@pytest.fixture
def pool():
    return DistractorPool(fma=FMA, taxonomy=Taxonomy(payload=PAYLOAD))


# ── the ladder ───────────────────────────────────────────────────────────

def test_siblings_are_preferred(pool):
    got = pool.choose("FMA4", 2, seed=1)
    assert {d.role for d in got} == {"is_a_sibling"}
    assert {d.label for d in got} <= {"Parietal bone", "Occipital bone"}


def test_ladder_falls_through_when_siblings_run_out(pool):
    roles = [d.role for d in pool.choose("FMA4", 4, seed=1)]
    assert roles[:2] == ["is_a_sibling", "is_a_sibling"]
    assert "is_a_cousin" in roles or "shares_whole" in roles
    # The last resort is a same-system structure, never something unrelated
    # with no recorded relation at all.
    assert all(r in {r0 for r0, _ in LADDER} for r in roles)


def test_cousins_come_from_the_grandparent_class(pool):
    got = pool.choose("FMA6", 1, seed=2)
    assert got[0].role == "is_a_cousin"
    assert got[0].label in {"Frontal bone", "Parietal bone", "Occipital bone"}


def test_the_focus_structure_is_never_offered(pool):
    got = pool.choose("FMA4", 5, seed=3)
    assert "FMA4" not in {d.item_id for d in got}


def test_excluded_ids_are_not_offered(pool):
    got = pool.choose("FMA4", 5, seed=3, exclude=["FMA5"])
    assert "FMA5" not in {d.item_id for d in got}


def test_structures_without_a_label_are_skipped(pool):
    got = pool.choose("FMA4", 5, seed=3)
    assert "FMA10" not in {d.item_id for d in got}


def test_duplicate_labels_are_not_offered_twice():
    fma = dict(FMA)
    fma["FMA11"] = {"display_name": "P", "preferred_label": "Parietal bone",
                    "system": "skeletal", "category": "skull_bones"}
    pool = DistractorPool(fma=fma, taxonomy=Taxonomy(payload=PAYLOAD))
    labels = [d.label for d in pool.choose("FMA4", 5, seed=1)]
    assert len(labels) == len(set(labels))


def test_restrict_to_keeps_an_exam_inside_the_material_studied():
    pool = DistractorPool(fma=FMA, taxonomy=Taxonomy(payload=PAYLOAD),
                          restrict_to={"FMA4", "FMA5"})
    got = pool.choose("FMA4", 4, seed=1)
    assert {d.item_id for d in got} == {"FMA5"}


def test_fewer_than_requested_is_reported_not_padded():
    pool = DistractorPool(fma=FMA, taxonomy=Taxonomy(payload=PAYLOAD),
                          restrict_to={"FMA4"})
    assert pool.choose("FMA4", 4, seed=1) == []


def test_a_structure_absent_from_the_crosswalk_can_still_be_a_distractor():
    """A real FMA class with a label is a legitimate wrong answer even when
    this app has no mesh for it -- an examiner offers structures a candidate
    should know, not only ones the software can draw."""
    pool = DistractorPool(fma={"FMA4": FMA["FMA4"]},
                          taxonomy=Taxonomy(payload=PAYLOAD))
    got = pool.choose("FMA4", 3, seed=1)
    assert {d.item_id for d in got} == {"FMA5", "FMA7", "FMA6"}
    assert all(d.label for d in got)


# ── determinism ──────────────────────────────────────────────────────────

def test_selection_is_reproducible_under_a_seed(pool):
    first = [d.item_id for d in pool.choose("FMA4", 3, seed=11)]
    assert first == [d.item_id for d in pool.choose("FMA4", 3, seed=11)]


def test_different_seeds_can_give_different_sets(pool):
    seeds = {tuple(d.item_id for d in pool.choose("FMA4", 2, seed=s))
             for s in range(12)}
    assert len(seeds) > 1


# ── provenance truth ─────────────────────────────────────────────────────

def test_sibling_provenance_names_the_shared_superclass(pool):
    got = pool.choose("FMA4", 1, seed=1)[0]
    assert got.provenance_kind == "fma_is_a"
    assert "FMA2" in got.provenance_reference


def test_shares_whole_provenance_names_a_genuinely_shared_whole():
    """Regression: the reference used to hard-code the focus structure's own
    narrowest whole, which became false once candidate widening pulled
    co-members out of broader wholes."""
    pool = DistractorPool()
    checked = 0
    for focus in ("FMA46759", "FMA49007", "FMA45740", "FMA46760"):
        for d in pool.choose(focus, 4, seed=7):
            if d.role != "shares_whole":
                continue
            checked += 1
            whole = pool._shared_whole(focus, d.item_id)
            assert whole, (focus, d.item_id)
            assert whole in pool._wholes[focus]
            assert whole in pool._wholes[d.item_id]
            assert pool._tax.label(whole).lower() in \
                d.provenance_reference.lower()
    assert checked, "expected at least one shares_whole distractor"


# ── against the real data ────────────────────────────────────────────────

def test_real_thoracic_vertebra_distractors_are_other_vertebrae():
    pool = DistractorPool()
    got = pool.choose("FMA10014", 4, seed=5)
    assert len(got) == 4
    assert all(d.role == "is_a_sibling" for d in got)
    assert all("vertebra" in d.label.lower() for d in got), \
        [d.label for d in got]


def test_real_skull_bone_is_not_offered_a_femur():
    pool = DistractorPool()
    labels = {d.label.lower() for d in pool.choose("FMA52734", 4, seed=5)}
    assert "femur" not in labels
    assert "right femur" not in labels


def test_rung_sizes_report_the_available_neighbourhood():
    sizes = DistractorPool().rung_sizes("FMA52734")
    assert set(sizes) == {role for role, _ in LADDER}
    assert sizes["is_a_sibling"] >= 4
