"""Item generators: correct facts, true keys, no true distractors.

The load-bearing tests here are the negative ones.  An item whose distractor is
also a correct answer is unanswerable and teaches a learner that a true
statement is false, so ``test_*_distractors_are_never_true_*`` is checked
against the full unfiltered relation set, not against the answer alone.
"""

import json

import pytest

from faceforge.anatomy.exam_items import ItemRefused, present
from faceforge.anatomy.fma_taxonomy import SCHEMA_VERSION, Taxonomy
from faceforge.anatomy.item_generators import (
    ItemGenerator,
    VIGNETTE_SCHEMA,
    load_vignettes,
)

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
        "8": {"label": "Right frontalis", "parent": "9"},
        "9": {"label": "Frontalis", "parent": "10"},
        "10": {"label": "Muscle of head", "parent": "1"},
        "11": {"label": "Left frontalis", "parent": "9"},
    },
    "labels": {"FMA100": "neurocranium", "FMA101": "skull",
               "FMA102": "musculature of head"},
    "part_of": {
        "FMA4": ["FMA100", "FMA2"],       # FMA2 is the is-a parent: a trap
        "FMA5": ["FMA100"],
        "FMA7": ["FMA100"],
        "FMA6": ["FMA101"],
        "FMA8": ["FMA102"],
        "FMA11": ["FMA102"],
    },
    "composite_of": {},
}

FMA = {
    "FMA4": {"display_name": "Frontal Bone", "preferred_label": "Frontal bone",
             "system": "skeletal", "category": "skull_bones"},
    "FMA5": {"display_name": "Parietal Bone", "preferred_label": "Parietal bone",
             "system": "skeletal", "category": "skull_bones"},
    "FMA6": {"display_name": "Mandible", "preferred_label": "Mandible",
             "system": "skeletal", "category": "skull_bones"},
    "FMA7": {"display_name": "Occipital Bone",
             "preferred_label": "Occipital bone",
             "system": "skeletal", "category": "skull_bones"},
    "FMA8": {"display_name": "Frontalis R", "preferred_label": "Right frontalis",
             "system": "muscular", "category": "muscles"},
    "FMA11": {"display_name": "Frontalis L", "preferred_label": "Left frontalis",
              "system": "muscular", "category": "muscles"},
}


@pytest.fixture
def gen():
    return ItemGenerator(fma=FMA, taxonomy=Taxonomy(payload=PAYLOAD))


@pytest.fixture(scope="module")
def real_gen():
    return ItemGenerator()


# ── L1 identification ────────────────────────────────────────────────────

def test_identification_answer_is_the_fma_preferred_term(gen):
    item = present(gen.identification("FMA4", options=3, seed=1))
    assert item.level == "L1"
    assert item.answer.text == "Frontal bone"
    assert item.answer.item_id == "FMA4"
    assert item.verified is True


def test_identification_carries_the_crosswalk_row_as_provenance(gen):
    item = gen.identification("FMA4", options=3, seed=1)
    assert item.provenance[0].kind == "fma_label"
    assert item.provenance[0].reference == "FMA4"
    assert "fma_labels.json" in item.provenance[0].detail


def test_identification_is_none_for_an_unknown_structure(gen):
    assert gen.identification("FMA999", seed=1) is None


def test_answer_position_varies_with_the_seed(gen):
    positions = {gen.identification("FMA4", options=4, seed=s).answer_index
                 for s in range(12)}
    assert len(positions) > 1


def test_same_seed_gives_the_same_item_and_uid(gen):
    a = gen.identification("FMA4", options=4, seed=3)
    b = gen.identification("FMA4", options=4, seed=3)
    assert a.uid == b.uid
    assert [o.text for o in a.options] == [o.text for o in b.options]


# ── L2 system and laterality ─────────────────────────────────────────────

def test_system_answer_is_the_crosswalk_system(gen):
    item = present(gen.system_of("FMA4", options=2, seed=1))
    assert item.answer.text == "Skeletal system"
    assert "frontal bone" in item.stem.lower()


def test_system_distractors_are_never_the_true_system(gen):
    item = gen.system_of("FMA4", options=2, seed=1)
    wrong = [o.text for i, o in enumerate(item.options)
             if i != item.answer_index]
    assert "Skeletal system" not in wrong


def test_system_is_none_when_the_crosswalk_has_no_system():
    gen = ItemGenerator(fma={"FMA4": {"preferred_label": "X"}},
                        taxonomy=Taxonomy(payload=PAYLOAD))
    assert gen.system_of("FMA4", seed=1) is None


def test_laterality_reads_the_side_off_the_preferred_term(gen):
    item = present(gen.laterality("FMA8", seed=1))
    assert item.answer.text == "Right"
    assert item.fmt == "spot"
    assert "frontalis" in item.stem.lower()
    assert "right" not in item.stem.lower()      # must not give it away


def test_laterality_is_none_for_an_unpaired_structure(gen):
    assert gen.laterality("FMA4", seed=1) is None


# ── L3 is-a ──────────────────────────────────────────────────────────────

def test_is_a_answer_is_the_superclass_and_is_phrased_as_classification(gen):
    item = present(gen.is_a("FMA4", options=3, seed=1))
    assert item.answer.text == "Flat bone"
    assert "classified as a kind of" in item.stem
    assert "part of" not in item.stem


def test_is_a_distractors_are_never_ancestors(gen):
    tax = Taxonomy(payload=PAYLOAD)
    for focus in ("FMA4", "FMA6", "FMA8"):
        item = gen.is_a(focus, options=4, seed=2)
        if item is None:
            continue
        ancestors = set(tax.is_a_chain(focus))
        wrong = {o.item_id for i, o in enumerate(item.options)
                 if i != item.answer_index}
        assert not (wrong & ancestors), (focus, wrong & ancestors)


def test_is_a_is_none_at_the_root(gen):
    assert gen.is_a("FMA1", seed=1) is None


# ── L3 part-of ───────────────────────────────────────────────────────────

def test_part_of_answer_is_a_real_whole_not_the_is_a_parent(gen):
    item = present(gen.part_of("FMA4", options=3, seed=1))
    assert item.answer.text == "neurocranium"
    assert item.answer.text != "Flat bone"
    assert "a part?" in item.stem


def test_part_of_distractors_are_never_true_wholes_at_any_level(gen):
    tax = Taxonomy(payload=PAYLOAD)
    for focus in ("FMA4", "FMA5", "FMA6", "FMA8"):
        item = gen.part_of(focus, options=4, seed=4)
        if item is None:
            continue
        true_wholes = (set(tax.part_of(focus, strict=False))
                       | set(tax.wholes(focus, strict=False))
                       | set(tax.is_a_chain(focus)))
        wrong = {o.item_id for i, o in enumerate(item.options)
                 if i != item.answer_index}
        assert not (wrong & true_wholes), (focus, wrong & true_wholes)


def test_part_of_is_none_when_no_part_of_edge_exists(gen):
    assert gen.part_of("FMA7777", seed=1) is None


# ── L3 negative form ─────────────────────────────────────────────────────

def test_not_part_of_key_is_the_only_non_member(gen):
    item = gen.not_part_of("FMA100", options=3, seed=1)
    if item is None:
        pytest.skip("synthetic graph too small for a 3-option negative item")
    tax = Taxonomy(payload=PAYLOAD)
    answer_id = item.answer.item_id
    assert "FMA100" not in tax.part_of(answer_id, strict=False)
    for i, opt in enumerate(item.options):
        if i == item.answer_index:
            continue
        assert "FMA100" in tax.part_of(opt.item_id, strict=False)


def test_not_part_of_is_none_without_enough_true_members(gen):
    assert gen.not_part_of("FMA101", options=5, seed=1) is None


# ── batch and coverage ───────────────────────────────────────────────────

def test_generate_respects_the_count_and_the_level(gen):
    items = gen.generate("L1", ["FMA4", "FMA5", "FMA6"], count=2, options=3)
    assert len(items) == 2
    assert {i.level for i in items} == {"L1"}


def test_generate_skips_structures_the_data_cannot_serve(gen):
    items = gen.generate("L2", ["FMA4"], count=5, options=2)
    # Only system_of works for an unpaired structure; laterality returns None.
    assert [i.tags for i in items] == [("system", )]


def test_coverage_reports_what_each_generator_can_serve(gen):
    cov = gen.coverage(["FMA4", "FMA8"])
    assert cov["identification"] == 2
    assert cov["laterality"] == 1        # only the sided structure
    assert set(cov) >= {"identification", "system_of", "is_a", "part_of"}


# ── extended matching ────────────────────────────────────────────────────

def test_emq_shares_one_option_list(gen):
    items = gen.extended_matching(["FMA4", "FMA6", "FMA8"], seed=1)
    assert len(items) >= 2
    shared = items[0].options
    assert all(i.options == shared for i in items)
    assert {i.fmt for i in items} == {"emq"}


def test_emq_drops_stems_that_do_not_discriminate(gen):
    # Right and left frontalis share the superclass "Frontalis", so a stem
    # naming it identifies two options and must not be asked.
    items = gen.extended_matching(["FMA8", "FMA11", "FMA4", "FMA6"], seed=1)
    answered = {i.options[i.answer_index].item_id for i in items}
    assert "FMA8" not in answered
    assert "FMA11" not in answered
    assert answered == {"FMA4", "FMA6"}


def test_emq_returns_nothing_when_only_one_stem_survives(gen):
    assert gen.extended_matching(["FMA8", "FMA11", "FMA4"], seed=1) == []


def test_emq_needs_at_least_three_structures(gen):
    assert gen.extended_matching(["FMA4", "FMA6"], seed=1) == []


# ── L5 vignettes: schema and gate only ───────────────────────────────────

def test_no_vignette_content_is_shipped():
    from faceforge.constants import CONFIG_DIR
    assert not list(CONFIG_DIR.glob("*vignette*"))


def test_vignette_schema_documents_the_citation_requirement():
    assert "citation" in VIGNETTE_SCHEMA["required"]
    assert "citation" in VIGNETTE_SCHEMA["notes"]


def test_missing_vignette_file_is_not_an_error(tmp_path):
    assert load_vignettes(tmp_path / "absent.json") == ([], [])


def test_vignette_without_a_citation_is_rejected(tmp_path):
    path = tmp_path / "v.json"
    path.write_text(json.dumps({"items": [
        {"uid": "v1", "stem": "A 34-year-old...", "options": ["A", "B"],
         "answer_index": 0, "citation": ""},
        {"uid": "v2", "stem": "A 51-year-old...", "options": ["A", "B"],
         "answer_index": 0},
    ]}))
    items, rejected = load_vignettes(path)
    assert items == []
    assert {r.uid for r in rejected} == {"v1", "v2"}
    assert any("citation" in " ".join(r.reasons) for r in rejected)


def test_vignette_with_a_citation_loads_and_is_presentable(tmp_path):
    path = tmp_path / "v.json"
    path.write_text(json.dumps({"items": [
        {"uid": "v1", "stem": "A 34-year-old presents with...",
         "options": ["Facial nerve", "Trigeminal nerve"], "answer_index": 0,
         "citation": "Moore, Clinically Oriented Anatomy 8e, p. 900"},
    ]}))
    items, rejected = load_vignettes(path)
    assert rejected == []
    assert len(items) == 1
    item = items[0]
    assert item.level == "L5"
    assert item.verified is False        # authored, not derived
    assert present(item) is item         # but citable, so allowed


def test_malformed_vignette_rows_are_rejected_with_reasons(tmp_path):
    path = tmp_path / "v.json"
    path.write_text(json.dumps({"items": [
        "not an object",
        {"uid": "v3", "stem": "", "options": ["A"], "answer_index": 9,
         "citation": "x"},
    ]}))
    items, rejected = load_vignettes(path)
    assert items == []
    assert len(rejected) == 2


# ── against the real data ────────────────────────────────────────────────

def test_real_items_are_verified_and_presentable(real_gen):
    for focus in ("FMA52734", "FMA49007", "FMA7197", "FMA10014"):
        for name in ("identification", "system_of", "is_a", "part_of"):
            item = getattr(real_gen, name)(focus, options=5, seed=3)
            if item is None:
                continue
            assert present(item) is item
            assert item.verified, item.provenance_report()
            # Never more than requested, never fewer than two.  Some real
            # structures cannot fill five options -- right temporalis has one
            # is-a sibling and three parent-siblings -- and the generator
            # returns the short item rather than padding it with something
            # unrelated.  The shortfall is visible to the caller in len().
            assert 2 <= len(item.options) <= 5


def test_real_frontal_bone_is_classified_as_a_flat_bone(real_gen):
    item = real_gen.is_a("FMA52734", options=5, seed=3)
    assert item.answer.text == "Flat bone"


def test_real_frontal_bone_is_part_of_the_neurocranium(real_gen):
    item = real_gen.part_of("FMA52734", options=5, seed=3)
    assert item.answer.text == "neurocranium"


def test_real_coverage_over_the_whole_crosswalk(real_gen):
    from faceforge.loaders.stl_batch_loader import load_fma_labels
    ids = sorted(load_fma_labels())
    cov = real_gen.coverage(ids[:120])
    # Identification must work for essentially everything; the relation
    # generators are data-limited and that is reported, not asserted away.
    assert cov["identification"] >= 118
    assert cov["is_a"] > 0 and cov["part_of"] > 0


def test_real_option_shortfall_is_short_not_padded(real_gen):
    """Right temporalis has exactly one is-a sibling, so its classification
    item cannot offer five options.  The generator must produce a four-option
    item rather than reach for an unrelated structure."""
    item = real_gen.is_a("FMA49007", options=5, seed=3)
    assert len(item.options) < 5
    assert all(o.role in ("answer", "parent_sibling") for o in item.options)
