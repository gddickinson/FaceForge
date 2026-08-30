"""QuizEngine: curricula, grading, persistence, and the pre-existing API.

The engine is constructed with ``autoload=False`` or with explicit
collaborators everywhere here, so no test touches the real user data
directory.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from faceforge.anatomy.answer_explanations import ExplanationBuilder
from faceforge.anatomy.curricula import build_curricula
from faceforge.anatomy.quiz_engine import QuizEngine
from faceforge.anatomy.quiz_progress import ProgressStore
from faceforge.anatomy.structure_search import SearchEntry

T0 = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)

FMA = {
    "FMA1": {"display_name": "Frontal Bone", "preferred_label": "Frontal bone",
             "system": "skeletal", "category": "skull_bones"},
    "FMA2": {"display_name": "Mandible", "preferred_label": "Mandible",
             "system": "skeletal", "category": "skull_bones"},
    "FMA3": {"display_name": "Ninth Thoracic",
             "preferred_label": "Ninth thoracic vertebra",
             "system": "skeletal", "category": "skeleton"},
    "FMA4": {"display_name": "Frontalis R", "preferred_label": "Right frontalis",
             "system": "muscular", "category": "muscles"},
}


@pytest.fixture
def clock():
    state = {"now": T0}
    now = lambda: state["now"]                                  # noqa: E731
    now.advance = lambda **kw: state.__setitem__("now", state["now"] + timedelta(**kw))
    return now


@pytest.fixture
def curricula(tmp_path):
    cfg = tmp_path / "config"
    (cfg / "muscles").mkdir(parents=True)
    (cfg / "skeleton").mkdir(parents=True)
    (cfg / "skull_bones.json").write_text(json.dumps([
        {"name": "Frontal Bone", "stl": "FMA1"},
        {"name": "Mandible", "stl": "FMA2"},
        {"name": "Ninth Thoracic", "stl": "FMA3"},
    ]))
    (cfg / "muscles" / "expression_muscles.json").write_text(json.dumps([
        {"name": "Frontalis R", "stl": "FMA4"},
    ]))
    return build_curricula(fma=FMA, config_dir=cfg)


@pytest.fixture
def store(tmp_path, clock):
    return ProgressStore(user="t", path=tmp_path / "t.json", clock=clock)


@pytest.fixture
def engine(curricula, store, clock):
    return QuizEngine(
        progress=store, curricula=curricula,
        explanations=ExplanationBuilder(fma=FMA), clock=clock,
    )


# ── question generation from a curriculum ────────────────────────────────

def test_curriculum_questions_carry_provenance(engine):
    engine.start_quiz(count=3, difficulty="hard", curriculum="skull_bones")
    q = engine.current_question
    assert q.item_id in FMA
    assert q.preferred_label == FMA[q.item_id]["preferred_label"]
    assert q.curriculum == "skull_bones"
    assert q.tier in ("foundation", "intermediate")
    assert q.time_limit == 15.0


def test_difficulty_selects_tiers(engine):
    engine.start_quiz(count=10, difficulty="easy", curriculum="skull_bones")
    assert {q.tier for q in engine._questions} == {"foundation"}
    engine.start_quiz(count=10, difficulty="hard", curriculum="skull_bones")
    assert {q.tier for q in engine._questions} == {"foundation", "intermediate"}


def test_explicit_tier_overrides_difficulty(engine):
    engine.start_quiz(count=10, difficulty="easy", curriculum="skull_bones",
                      tier="intermediate")
    assert [q.item_id for q in engine._questions] == ["FMA3"]


def test_unknown_curriculum_leaves_the_quiz_inactive(engine):
    engine.start_quiz(count=5, curriculum="does_not_exist")
    assert engine.is_active is False
    assert engine.current_question is None


def test_empty_tier_leaves_the_quiz_inactive(engine):
    engine.start_quiz(count=5, curriculum="expression_muscles",
                      tier="intermediate")
    assert engine.is_active is False


def test_curriculum_keys_are_ordered_largest_first(engine):
    keys = engine.curriculum_keys()
    sizes = [len(engine.curricula[k]) for k in keys]
    assert sizes == sorted(sizes, reverse=True)
    # Ties break on the key, so the order is fully determined.
    assert keys[0] in ("skull_bones", "system:skeletal")


# ── grading and explanations ─────────────────────────────────────────────

def test_exact_answer_within_the_fast_threshold_grades_five(engine, clock):
    engine.start_quiz(count=1, difficulty="easy", curriculum="skull_bones")
    clock.advance(seconds=2)
    correct, _ = engine.check_answer(engine.current_question.display_name)
    assert correct is True
    assert engine.last_grade == 5


def test_a_slow_exact_answer_grades_four(engine, clock):
    engine.start_quiz(count=1, difficulty="easy", curriculum="skull_bones")
    clock.advance(seconds=45)
    engine.check_answer(engine.current_question.display_name)
    assert engine.last_grade == 4


def test_the_fma_preferred_term_is_accepted_as_exact(engine, clock):
    engine.start_quiz(count=1, difficulty="easy", curriculum="skull_bones",
                      tier="foundation")
    q = engine.current_question
    clock.advance(seconds=1)
    correct, _ = engine.check_answer(q.preferred_label)
    assert correct is True
    assert engine.last_grade == 5


def test_a_fuzzy_match_passes_at_the_lowest_passing_grade(engine, clock):
    engine.start_quiz(count=10, difficulty="hard", curriculum="skull_bones")
    q = next(q for q in engine._questions if q.item_id == "FMA1")
    engine._current_idx = engine._questions.index(q)
    clock.advance(seconds=1)
    correct, _ = engine.check_answer("Frontal Bne")
    assert correct is True
    assert engine.last_grade == 3


def test_a_wrong_answer_from_another_system_grades_one(engine, clock):
    engine.start_quiz(count=10, difficulty="hard", curriculum="skull_bones")
    engine.check_answer("Frontalis R")
    assert engine.last_grade in (1, 2)
    assert engine.last_explanation is not None
    assert engine.last_explanation.correct.preferred_label


def test_wrong_answer_explanation_names_both_structures(engine):
    engine.start_quiz(count=10, difficulty="easy", curriculum="skull_bones",
                      tier="foundation")
    target = engine.current_question
    engine.check_answer("Frontalis R")
    text = engine.last_explanation.text
    assert "Right frontalis" in text
    assert FMA[target.item_id]["preferred_label"] in text


# ── the skip path (regression) ───────────────────────────────────────────

def test_an_empty_answer_is_not_counted_as_correct(engine):
    """Regression: the substring branch of _fuzzy_match made "" match every
    structure, so the Skip button and the hard-mode timeout -- both of which
    call check_answer("") -- scored as correct."""
    engine.start_quiz(count=3, difficulty="hard", curriculum="skull_bones")
    correct, _ = engine.check_answer("")
    assert correct is False
    assert engine.score.correct == 0
    assert engine.score.incorrect == 1
    assert engine.last_grade == 0


def test_a_skip_lapses_the_card(engine, store):
    engine.start_quiz(count=1, difficulty="easy", curriculum="skull_bones")
    item = engine.current_question.item_id
    engine.check_answer("")
    assert store.scheduler.card(item).lapses == 1
    assert store.scheduler.card(item).interval_days == 1


# ── persistence and scheduling across sessions ───────────────────────────

def test_answers_are_recorded_with_full_context(engine, store, clock):
    engine.start_quiz(count=1, difficulty="easy", curriculum="skull_bones")
    q = engine.current_question
    clock.advance(seconds=3)
    engine.check_answer(q.display_name)
    attempt = store.attempts[-1]
    assert attempt.item_id == q.item_id
    assert attempt.curriculum == "skull_bones"
    assert attempt.tier == q.tier
    assert attempt.elapsed_s == pytest.approx(3.0)
    assert attempt.timestamp == (T0 + timedelta(seconds=3)).isoformat()
    assert attempt.grade == 5


def test_end_quiz_writes_the_progress_file(engine, store):
    engine.start_quiz(count=1, difficulty="easy", curriculum="skull_bones")
    engine.check_answer(engine.current_question.display_name)
    engine.end_quiz()
    assert store.path.exists()
    payload = json.loads(store.path.read_text())
    assert len(payload["attempts"]) == 1
    assert payload["cards"]


def test_a_later_session_asks_only_what_is_due(curricula, store, clock):
    """Two passes put an item 6 days out; a lapse puts it back to tomorrow.

    Both are SM-2 interval arithmetic, so after two days only the lapsed item
    should be asked -- which is the whole point of attaching a scheduler.
    """
    engine = QuizEngine(progress=store, curricula=curricula, clock=clock,
                        explanations=ExplanationBuilder(fma=FMA))
    engine.start_quiz(count=3, difficulty="hard", curriculum="skull_bones")
    asked = list(engine._questions)
    failed = asked[2]

    # Two passing reviews each for the first two items -> interval 6 days.
    for _ in range(2):
        for q in asked[:2]:
            clock.advance(seconds=1)
            store.record(q.item_id, q.display_name, True, 5)
    # The third item lapses -> interval 1 day.
    clock.advance(seconds=1)
    engine._current_idx = 2
    engine.check_answer("Frontalis R")
    engine.end_quiz()

    clock.advance(days=2)
    due = store.scheduler.due_items([q.item_id for q in asked])
    assert due == [failed.item_id]
    engine.start_quiz(count=1, difficulty="hard", curriculum="skull_bones")
    assert engine._questions[0].item_id == failed.item_id


def test_scheduling_prefers_due_items_over_unseen_ones(curricula, store, clock):
    engine = QuizEngine(progress=store, curricula=curricula, clock=clock,
                        autoload=False)
    engine.start_quiz(count=1, difficulty="easy", curriculum="skull_bones")
    first = engine.current_question.item_id
    engine.check_answer("")                     # lapse: due again in 1 day
    clock.advance(days=2)
    engine.start_quiz(count=1, difficulty="easy", curriculum="skull_bones")
    assert engine.current_question.item_id == first


def test_progress_failure_does_not_break_the_quiz(curricula, clock, caplog):
    class Broken:
        scheduler = None

        def record(self, *a, **kw):
            raise OSError("disk full")

    class Store(Broken):
        scheduler = ProgressStore(path=None, clock=clock).scheduler

        def __init__(self):
            self.path = "/nonexistent/x.json"

    engine = QuizEngine(progress=Store(), curricula=curricula, clock=clock,
                        autoload=False)
    engine.start_quiz(count=1, difficulty="easy", curriculum="skull_bones")
    correct, name = engine.check_answer(engine.current_question.display_name)
    assert correct is True                      # scoring survived
    assert engine.score.correct == 1


# ── the pre-existing search-index API still works ────────────────────────

def _index(entries):
    class Index:
        pass
    idx = Index()
    idx.entries = entries
    return idx


def test_search_index_path_is_unchanged(clock):
    entries = [
        SearchEntry(mesh_name="Biceps R", display_name="Biceps R",
                    category="muscle", region="arm", source_id="FMA9"),
        SearchEntry(mesh_name="Femur R", display_name="Femur R",
                    category="bone", region="leg"),
    ]
    engine = QuizEngine(_index(entries), autoload=False, clock=clock)
    engine.start_quiz(mode="identify", category="muscle", count=1)
    assert engine.is_active
    q = engine.current_question
    assert q.display_name == "Biceps R"
    assert q.item_id == "FMA9"
    correct, name = engine.check_answer("Biceps R")
    assert (correct, name) == (True, "Biceps R")
    assert engine.questions_remaining == 0
    assert engine.next_question() is None
    assert engine.is_active is False


def test_engine_with_no_index_and_no_curriculum_is_inactive(clock):
    engine = QuizEngine(autoload=False, clock=clock)
    engine.start_quiz(count=5)
    assert engine.is_active is False


def test_autoload_false_means_no_store_is_created(clock):
    engine = QuizEngine(autoload=False, clock=clock)
    assert engine.progress is None
    assert engine.explanations is None
    assert engine.save_progress() is False
