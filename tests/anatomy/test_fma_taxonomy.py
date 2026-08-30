"""The FMA relation graph, and the separation of is-a from part-of.

The synthetic half proves the graph algebra on a graph small enough to check by
hand.  The real half asserts facts about the shipped asset that are checkable
against anatomy: a frontal bone is a flat bone (is-a) and is part of the
neurocranium (part-of), and those are different statements.
"""

import json

import pytest

from faceforge.anatomy.fma_taxonomy import (
    SCHEMA_VERSION,
    Taxonomy,
    bare_id,
    get_taxonomy,
    prefixed_id,
)

# A hand-built graph:
#           1 bone organ
#         /            \
#    2 flat bone      3 irregular bone
#     /      \              \
#  4 frontal 5 parietal    6 mandible
PAYLOAD = {
    "schema_version": SCHEMA_VERSION,
    "_source": "synthetic",
    "_source_part_of": "synthetic-part-of",
    "_source_composite": "synthetic-composite",
    "nodes": {
        "1": {"label": "Bone organ", "parent": ""},
        "2": {"label": "Flat bone", "parent": "1"},
        "3": {"label": "Irregular bone", "parent": "1"},
        "4": {"label": "Frontal bone", "parent": "2"},
        "5": {"label": "Parietal bone", "parent": "2"},
        "6": {"label": "Mandible", "parent": "3"},
    },
    "labels": {"FMA100": "neurocranium", "FMA101": "skull",
               "FMA4": "Frontal bone"},
    # Note FMA2 ("flat bone") appears here, as it does in the real file: a
    # superclass wrongly listed as a whole.  strict=True must drop it.
    "part_of": {"FMA4": ["FMA100", "FMA2"], "FMA5": ["FMA100"]},
    # FMA100 aggregates two primitives, FMA101 only one, so FMA101 is the
    # more specific whole -- the property most_specific() must implement.
    "composite_of": {"FMA4": ["FMA101", "FMA100"], "FMA5": ["FMA100"]},
}


@pytest.fixture
def tax():
    return Taxonomy(payload=PAYLOAD)


# ── id normalisation ─────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,bare", [
    ("FMA52734", "52734"), ("52734", "52734"),
    ("FMA14543nsn", "14543"), ("", ""),
])
def test_bare_id(raw, bare):
    assert bare_id(raw) == bare


def test_prefixed_id_is_idempotent():
    assert prefixed_id("52734") == "FMA52734"
    assert prefixed_id("FMA52734") == "FMA52734"
    assert prefixed_id("BP48") == "BP48"


# ── is-a ─────────────────────────────────────────────────────────────────

def test_is_a_parent_and_chain(tax):
    assert tax.is_a_parent("FMA4") == "FMA2"
    assert tax.is_a_chain("FMA4") == ["FMA2", "FMA1"]
    assert tax.is_a_chain("FMA1") == []


def test_siblings_exclude_self_and_are_ordered(tax):
    assert tax.siblings("FMA4") == ["FMA5"]
    assert tax.siblings("FMA1") == []


def test_cousins_share_a_grandparent_but_not_a_parent(tax):
    assert tax.cousins("FMA4") == ["FMA6"]
    assert "FMA5" not in tax.cousins("FMA4")


def test_descendants_are_breadth_first(tax):
    assert tax.descendants_of("FMA1") == ["FMA2", "FMA3", "FMA4", "FMA5", "FMA6"]


def test_chain_is_cycle_safe():
    cyclic = dict(PAYLOAD, nodes={
        "1": {"label": "A", "parent": "2"},
        "2": {"label": "B", "parent": "1"},
    })
    assert len(Taxonomy(payload=cyclic).is_a_chain("FMA1")) <= 2


# ── part-of, and its separation from is-a ────────────────────────────────

def test_strict_part_of_drops_the_is_a_superclass(tax):
    assert tax.part_of("FMA4") == ["FMA100"]
    assert tax.part_of("FMA4", strict=False) == ["FMA100", "FMA2"]


def test_wholes_are_also_filtered(tax):
    assert tax.wholes("FMA4") == ["FMA101", "FMA100"]


def test_most_specific_prefers_the_whole_with_fewest_primitives(tax):
    assert tax.primitive_count("FMA100") == 2
    assert tax.primitive_count("FMA101") == 1
    assert tax.most_specific(["FMA101", "FMA100"]) == "FMA101"
    assert tax.most_specific([]) == ""


def test_relation_records_kind_and_source(tax):
    rel = tax.relation("part_of", "FMA4", "FMA100")
    assert rel.kind == "part_of"
    assert rel.subject_label == "Frontal bone"
    assert rel.object_label == "neurocranium"
    assert rel.source == "synthetic-part-of"
    assert tax.relation("is_a", "FMA4", "FMA2").source == "synthetic"


# ── degradation ──────────────────────────────────────────────────────────

def test_missing_file_degrades_to_no_relations(tmp_path, monkeypatch):
    empty = Taxonomy(payload={})
    assert empty.available is False
    assert empty.is_a_chain("FMA4") == []
    assert empty.part_of("FMA4") == []
    assert empty.label("FMA4") == ""


def test_wrong_schema_version_is_not_used(tmp_path, monkeypatch):
    from faceforge.anatomy import fma_taxonomy
    path = tmp_path / "fma_taxonomy.json"
    path.write_text(json.dumps({"schema_version": 999, "nodes": {"1": {}}}))
    monkeypatch.setattr(fma_taxonomy, "CONFIG_DIR", tmp_path)
    assert fma_taxonomy._load_payload() == {}


# ── against the shipped asset ────────────────────────────────────────────

def test_real_taxonomy_covers_every_loadable_structure():
    from faceforge.loaders.stl_batch_loader import load_fma_labels
    tax = get_taxonomy()
    assert tax.available
    missing = [k for k in load_fma_labels() if not tax.label(k)]
    assert missing == []


def test_real_is_a_and_part_of_are_different_relations():
    tax = get_taxonomy()
    # is-a: a frontal bone is a kind of flat bone.
    assert tax.label(tax.is_a_parent("FMA52734")) == "Flat bone"
    # part-of: it is part of the neurocranium, and "flat bone" must not appear
    # as a whole once the two relations are separated.
    wholes = [tax.label(w) for w in tax.part_of("FMA52734")]
    assert "neurocranium" in wholes
    assert "Flat bone" not in wholes
    assert "flat bone" not in [w.lower() for w in wholes]


def test_real_siblings_of_a_thoracic_vertebra_are_thoracic_vertebrae():
    tax = get_taxonomy()
    labels = [tax.label(s).lower() for s in tax.siblings("FMA10014")]
    assert labels, "T9 should have sibling vertebrae"
    assert all("thoracic vertebra" in lab for lab in labels), labels


def test_real_narrowest_whole_of_the_liver_is_the_abdomen():
    tax = get_taxonomy()
    wholes = list(tax.part_of("FMA7197")) + list(tax.wholes("FMA7197"))
    assert tax.label(tax.most_specific(wholes)) == "abdomen"
