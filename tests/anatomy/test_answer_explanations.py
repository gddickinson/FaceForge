"""Wrong-answer explanations: resolution, difference computation, wording.

The synthetic fixture proves the rules; the last group runs against the real
crosswalk and asserts on structures that actually exist in it (checked, not
assumed -- see test_real_crosswalk_entries_used_here_exist).
"""

import pytest

from faceforge.anatomy.answer_explanations import ExplanationBuilder

FMA = {
    "FMA49007": {"display_name": "Temporalis R",
                 "preferred_label": "Right temporalis",
                 "system": "muscular", "category": "muscles"},
    "FMA49008": {"display_name": "Temporalis L",
                 "preferred_label": "Left temporalis",
                 "system": "muscular", "category": "muscles"},
    "FMA49004": {"display_name": "Masseter Deep R",
                 "preferred_label": "Deep part of right masseter",
                 "system": "muscular", "category": "muscles"},
    "FMA52734": {"display_name": "Frontal Bone",
                 "preferred_label": "Frontal bone",
                 "system": "skeletal", "category": "skull_bones"},
    "FMA7197": {"display_name": "Liver", "preferred_label": "Liver",
                "system": "digestive", "category": "organs"},
    "FMA_NOMETA": {"display_name": "Mystery", "preferred_label": "",
                   "system": "", "category": ""},
}


@pytest.fixture
def builder():
    return ExplanationBuilder(fma=FMA)


# ── resolution ───────────────────────────────────────────────────────────

def test_resolves_a_mesh_id(builder):
    facts = builder.facts("FMA49007")
    assert facts.item_id == "FMA49007"
    assert facts.preferred_label == "Right temporalis"
    assert facts.side == "right"


def test_resolves_a_display_name_and_the_fma_term(builder):
    assert builder.facts("Temporalis R").item_id == "FMA49007"
    assert builder.facts("right temporalis").item_id == "FMA49007"


def test_near_miss_spelling_still_resolves(builder):
    assert builder.facts("Temporalis  R").item_id == "FMA49007"
    assert builder.facts("Frontal bome").item_id == "FMA52734"


def test_unrelated_text_is_not_forced_onto_the_closest_entry(builder):
    facts = builder.facts("qqqqzzz")
    assert facts.item_id == ""
    assert facts.display_name == "qqqqzzz"


def test_empty_answer_is_unknown(builder):
    assert builder.facts("").known is False


# ── differences ──────────────────────────────────────────────────────────

def test_laterality_is_the_only_difference_between_a_pair(builder):
    e = builder.explain("Temporalis L", "Temporalis R")
    assert e.differences == (("side", "left", "right"),)
    assert e.same_system is True
    assert "Left temporalis" in e.text and "Right temporalis" in e.text
    assert "side (left vs right)" in e.text


def test_cross_system_confusion_names_both_systems(builder):
    e = builder.explain("Temporalis R", "Frontal Bone")
    attrs = {d[0] for d in e.differences}
    assert {"system", "category"} <= attrs
    assert e.same_system is False
    assert "muscular vs skeletal" in e.text


def test_unresolved_answer_still_describes_the_correct_structure(builder):
    e = builder.explain("qqqqzzz", "FMA7197")
    assert "does not match any structure" in e.text
    assert "Liver (digestive system, organs group" in e.text


def test_no_answer_given(builder):
    e = builder.explain("", "FMA7197")
    assert e.text.startswith("No answer given.")
    assert "Liver" in e.text


def test_same_structure_is_reported_as_such(builder):
    e = builder.explain("Temporalis R", "FMA49007")
    assert "same structure" in e.text


def test_missing_metadata_is_reported_as_missing_not_guessed(builder):
    e = builder.explain("Mystery", "FMA7197")
    assert "no system recorded" in e.text
    assert e.chosen.system == ""


def test_structured_fields_are_available_for_a_custom_layout(builder):
    e = builder.explain("Masseter Deep R", "Temporalis R")
    assert e.chosen.preferred_label == "Deep part of right masseter"
    assert e.correct.preferred_label == "Right temporalis"
    assert e.chosen.side == "right" and e.correct.side == "right"
    # Same system, group and side: the text must still say something concrete.
    assert e.differences == ()
    assert "FMA49004" in e.text and "FMA49007" in e.text


# ── against the real crosswalk ───────────────────────────────────────────

def test_real_crosswalk_entries_used_here_exist():
    from faceforge.loaders.stl_batch_loader import load_fma_labels
    real = load_fma_labels()
    for mesh_id in ("FMA49007", "FMA49004", "FMA52734", "FMA7197"):
        assert mesh_id in real, mesh_id
        assert real[mesh_id]["preferred_label"] == FMA[mesh_id]["preferred_label"]


def test_real_explanation_uses_real_terms():
    e = ExplanationBuilder().explain("Temporalis L", "Temporalis R")
    assert "Left temporalis" in e.text
    assert "Right temporalis" in e.text
