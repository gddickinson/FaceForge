"""SM-2 scheduling, driven by a fixed clock.

Every assertion here is a hand-computed SM-2 value, not a value read back off
the implementation, so the test fails if the algorithm drifts from the
published one:

    EF' = EF + (0.1 - (5-q)(0.08 + (5-q)0.02))
    I(1)=1, I(2)=6, I(n)=round(I(n-1) * EF)

The clock is a list cell the test advances by hand.  The point is not speed:
an interval-arithmetic bug (off-by-one day, interval measured from the wrong
epoch, a lapse that silently keeps its old due date) is only observable across
simulated days, and a test that sleeps cannot cover 20 of them.
"""

from datetime import datetime, timedelta, timezone

import pytest

from faceforge.anatomy.spaced_repetition import (
    INITIAL_EASINESS,
    MINIMUM_EASINESS,
    ReviewCard,
    Scheduler,
    grade_from_outcome,
    next_interval,
    review,
    update_easiness,
)

T0 = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def clock():
    """A manually advanced clock: call it for 'now', .advance(days=...)."""
    state = {"now": T0}

    def now():
        return state["now"]

    def advance(**kw):
        state["now"] = state["now"] + timedelta(**kw)

    now.advance = advance
    now.set = lambda t: state.__setitem__("now", t)
    return now


# ── easiness ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("grade,expected", [
    (5, 2.6),     # 2.5 + (0.1 - 0)                       = 2.6
    (4, 2.5),     # 2.5 + (0.1 - 1*(0.08 + 0.02))         = 2.5
    (3, 2.36),    # 2.5 + (0.1 - 2*(0.08 + 0.04))         = 2.36
    (2, 2.18),    # 2.5 + (0.1 - 3*(0.08 + 0.06))         = 2.18
    (1, 1.96),    # 2.5 + (0.1 - 4*(0.08 + 0.08))         = 1.96
    (0, 1.7),     # 2.5 + (0.1 - 5*(0.08 + 0.10))         = 1.70
])
def test_easiness_matches_published_sm2_formula(grade, expected):
    assert update_easiness(INITIAL_EASINESS, grade) == pytest.approx(expected)


def test_easiness_is_clamped_at_the_sm2_floor():
    ef = INITIAL_EASINESS
    for _ in range(20):
        ef = update_easiness(ef, 0)
    assert ef == pytest.approx(MINIMUM_EASINESS)


def test_grade_outside_zero_to_five_is_rejected():
    with pytest.raises(ValueError):
        update_easiness(2.5, 6)
    with pytest.raises(ValueError):
        review(ReviewCard("x"), -1, T0)


# ── intervals ────────────────────────────────────────────────────────────

def test_first_two_intervals_are_the_sm2_constants():
    assert next_interval(0, 0, 2.5) == 1
    assert next_interval(1, 1, 2.5) == 6


def test_later_intervals_are_previous_times_easiness_rounded():
    assert next_interval(2, 6, 2.5) == 15      # 6 * 2.5  = 15.0
    assert next_interval(3, 15, 2.5) == 38     # 15 * 2.5 = 37.5 -> 38
    assert next_interval(4, 38, 1.3) == 49     # 38 * 1.3 = 49.4 -> 49


def test_interval_never_stalls_at_the_easiness_floor():
    # 1 * 1.3 = 1.3 -> 1 would repeat the same day forever; SM-2 only reaches
    # I(n)=I(n-1)*EF for n>2, by which point the interval is at least 6.
    assert next_interval(2, 6, MINIMUM_EASINESS) == 8


# ── card progression over simulated days ─────────────────────────────────

def test_three_passes_walk_the_1_6_15_ladder(clock):
    card = ReviewCard("FMA52734")

    card = review(card, 5, clock())
    assert (card.repetitions, card.interval_days) == (1, 1)
    assert card.due == T0 + timedelta(days=1)
    assert card.easiness == pytest.approx(2.6)

    clock.advance(days=1)
    card = review(card, 5, clock())
    assert (card.repetitions, card.interval_days) == (2, 6)
    assert card.due == T0 + timedelta(days=7)

    clock.advance(days=6)
    card = review(card, 5, clock())
    # I(3) = I(2) * EF, with the EF the card carried into this review (2.7):
    # 6 * 2.7 = 16.2 -> 16.  The EF then becomes 2.8 for the next interval.
    assert card.interval_days == 16
    assert card.easiness == pytest.approx(2.8)
    assert card.due == T0 + timedelta(days=7 + 16)
    assert (card.reviews, card.lapses) == (3, 0)


def test_lapse_resets_interval_and_repetitions_but_keeps_easiness(clock):
    card = ReviewCard("FMA52734")
    for _ in range(3):
        card = review(card, 5, clock())
        clock.advance(days=card.interval_days)
    matured = card
    assert matured.interval_days == 16

    card = review(card, 1, clock())
    assert card.repetitions == 0
    assert card.interval_days == 1
    assert card.lapses == 1
    # EF is penalised (2.7 - 0.54) but not reset to 2.5.
    assert card.easiness == pytest.approx(matured.easiness - 0.54)
    assert card.due == clock() + timedelta(days=1)


def test_grade_three_passes_and_grade_two_fails(clock):
    passed = review(ReviewCard("a"), 3, clock())
    failed = review(ReviewCard("b"), 2, clock())
    assert passed.repetitions == 1 and passed.lapses == 0
    assert failed.repetitions == 0 and failed.lapses == 1


def test_review_does_not_mutate_its_input(clock):
    card = ReviewCard("a")
    review(card, 5, clock())
    assert card == ReviewCard("a")


def test_card_round_trips_through_dict(clock):
    card = review(ReviewCard("FMA1"), 4, clock())
    assert ReviewCard.from_dict(card.to_dict()) == card


def test_naive_timestamps_in_an_old_file_are_read_as_utc():
    card = ReviewCard.from_dict({"item_id": "a", "due": "2026-03-01T12:00:00"})
    assert card.due == T0
    assert card.is_due(T0)


# ── due selection ────────────────────────────────────────────────────────

def test_never_reviewed_card_is_due_and_new(clock):
    card = ReviewCard("a")
    assert card.is_new and card.is_due(clock())


def test_due_selection_puts_overdue_first_and_new_material_last(clock):
    sched = Scheduler(clock=clock)
    sched.record("old", 5)          # due in 1 day
    clock.advance(days=1)
    sched.record("recent", 5)       # due in 1 day, i.e. tomorrow
    clock.advance(days=5)           # "old" is 5 days overdue, "recent" 4

    due = sched.due_items(["fresh", "recent", "old"])
    assert due == ["old", "recent", "fresh"]


def test_not_yet_due_items_are_excluded(clock):
    sched = Scheduler(clock=clock)
    sched.record("a", 5)
    sched.record("a", 5)            # interval 6 days
    assert sched.due_items(["a"]) == []
    clock.advance(days=6)
    assert sched.due_items(["a"]) == ["a"]


def test_select_tops_up_with_unseen_items_when_nothing_is_due(clock):
    sched = Scheduler(clock=clock)
    sched.record("a", 5)
    sched.record("a", 5)            # a is not due for 6 days
    assert sched.select(["a", "b", "c"], 2) == ["b", "c"]


def test_select_is_capped_and_deterministic(clock):
    sched = Scheduler(clock=clock)
    ids = [f"i{n}" for n in range(10)]
    first = sched.select(ids, 4)
    assert len(first) == 4
    assert first == sched.select(ids, 4)


def test_scheduler_round_trips_through_dict(clock):
    sched = Scheduler(clock=clock)
    sched.record("a", 5)
    sched.record("b", 1)
    other = Scheduler(clock=clock)
    other.load_dict(sched.to_dict())
    assert other.cards == sched.cards


# ── outcome -> grade mapping ─────────────────────────────────────────────

def test_outcome_grade_mapping_is_the_documented_table():
    assert grade_from_outcome(True, exact=True, elapsed_s=2.0) == 5
    assert grade_from_outcome(True, exact=True, elapsed_s=30.0) == 4
    assert grade_from_outcome(True, exact=False, elapsed_s=2.0) == 3
    assert grade_from_outcome(False, same_system=True) == 2
    assert grade_from_outcome(False, same_system=False) == 1
    assert grade_from_outcome(False, skipped=True) == 0
    assert grade_from_outcome(True, skipped=True) == 0


def test_fast_threshold_is_a_parameter_not_a_constant():
    assert grade_from_outcome(True, elapsed_s=9.0, fast_threshold_s=10.0) == 5
    assert grade_from_outcome(True, elapsed_s=9.0, fast_threshold_s=5.0) == 4
