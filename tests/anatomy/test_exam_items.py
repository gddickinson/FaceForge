"""The provenance gate: what may and may not reach a learner.

These tests are the enforcement of the project's hard rule -- an anatomical
claim that is not traceable to real data must not be presented -- so they
assert on refusal, not just on acceptance.
"""

import pytest

from faceforge.anatomy.exam_items import (
    AUTHORED_SOURCES,
    DERIVED_SOURCES,
    ExamItem,
    ItemRefused,
    Option,
    Provenance,
    present,
    presentable,
)

DERIVED = Provenance(kind="fma_label", reference="FMA52734",
                     detail="preferred_label='Frontal bone'")
AUTHORED = Provenance(kind="citation", reference="Gray's Anatomy 42e p. 412")


def item(**kw) -> ExamItem:
    base = dict(
        level="L1", fmt="sba", stem="Identify the indicated structure.",
        options=(
            Option(text="Frontal bone", item_id="FMA52734", role="answer",
                   provenance=(DERIVED, )),
            Option(text="Parietal bone", item_id="FMA52735",
                   role="is_a_sibling",
                   provenance=(Provenance(kind="fma_is_a",
                                          reference="FMA52735"), )),
        ),
        answer_index=0, focus_id="FMA52734", provenance=(DERIVED, ),
    )
    base.update(kw)
    return ExamItem(**base)


# ── acceptance ───────────────────────────────────────────────────────────

def test_a_fully_derived_item_is_verified_and_presentable():
    it = item()
    assert it.verified is True
    assert it.problems() == []
    assert present(it) is it


def test_uid_is_deterministic_and_content_addressed():
    assert item().uid == item().uid
    assert item(stem="Different stem?").uid != item().uid
    assert len(item().uid) == 16


def test_answer_and_distractor_roles_are_exposed():
    it = item()
    assert it.answer.text == "Frontal bone"
    assert it.distractor_roles == ("is_a_sibling", )


# ── refusal ──────────────────────────────────────────────────────────────

def test_item_with_no_provenance_is_not_verified_and_is_refused():
    it = item(provenance=(), options=(
        Option(text="A"), Option(text="B"),
    ))
    assert it.verified is False
    assert "no provenance" in it.problems()
    with pytest.raises(ItemRefused) as excinfo:
        present(it)
    assert "no provenance" in excinfo.value.reasons


def test_authored_content_without_a_citation_is_refused():
    it = item(provenance=(Provenance(kind="citation", reference=""), ))
    assert it.verified is False
    assert "authored content without a citation" in it.problems()
    with pytest.raises(ItemRefused):
        present(it)


def test_authored_content_with_a_citation_is_allowed():
    it = item(level="L5", provenance=(AUTHORED, ),
              citation="Gray's Anatomy 42e p. 412",
              options=(Option(text="A", provenance=(AUTHORED, )),
                       Option(text="B", provenance=(AUTHORED, ))))
    assert it.verified is False          # authored, not machine-derived
    assert it.problems() == []          # but citable, so presentable
    assert present(it) is it


def test_revision_mode_still_requires_provenance():
    it = item(provenance=(), options=(Option(text="A"), Option(text="B")))
    with pytest.raises(ItemRefused):
        present(it, exam_mode=False)


def test_unverified_item_is_presentable_outside_exam_mode_if_sourced():
    it = item(provenance=(AUTHORED, ), citation="")
    assert it.problems(exam_mode=True) == ["authored content without a citation"]
    assert it.problems(exam_mode=False) == []


@pytest.mark.parametrize("kw,problem", [
    (dict(level="L9"), "unknown level 'L9'"),
    (dict(fmt="essay"), "unknown format 'essay'"),
    (dict(stem="   "), "empty stem"),
    (dict(answer_index=7), "answer_index out of range"),
])
def test_structural_problems_are_reported(kw, problem):
    assert problem in item(**kw).problems()


def test_duplicate_and_blank_option_text_are_refused():
    dup = item(options=(
        Option(text="Frontal bone", provenance=(DERIVED, )),
        Option(text="frontal bone", provenance=(DERIVED, )),
    ))
    assert "duplicate option text" in dup.problems()
    blank = item(options=(
        Option(text="Frontal bone", provenance=(DERIVED, )),
        Option(text="  ", provenance=(DERIVED, )),
    ))
    assert "blank option text" in blank.problems()


def test_single_option_item_is_refused():
    one = item(options=(Option(text="Frontal bone", provenance=(DERIVED, )), ))
    assert "1 option(s), need 2" in one.problems()


def test_five_option_requirement_can_be_raised():
    assert "2 option(s), need 5" in item().problems(min_options=5)


# ── batch splitting ──────────────────────────────────────────────────────

def test_presentable_splits_good_from_bad():
    good = item()
    bad = item(stem="", provenance=())
    ok, refused = presentable([good, bad])
    assert ok == [good]
    assert len(refused) == 1
    assert "empty stem" in refused[0][1]


# ── provenance vocabulary ────────────────────────────────────────────────

def test_derived_and_authored_source_sets_do_not_overlap():
    assert not (DERIVED_SOURCES & AUTHORED_SOURCES)


def test_provenance_report_names_every_fact_and_its_source():
    report = item().provenance_report()
    assert "fma_label:FMA52734" in report
    assert "fma_is_a:FMA52735" in report
    assert "verified=True" in report
