"""SM-2 spaced-repetition scheduling for the anatomy quiz.

The algorithm is **SM-2**, published by Piotr Woźniak as the scheduling
algorithm of SuperMemo 2 (1987-1989) and described in

    P. A. Woźniak and E. J. Gorzelańczyk, "Optimization of repetition spacing
    in the practice of learning", Acta Neurobiologiae Experimentalis 54 (1994)
    59-62.

with the reference pseudocode published at
https://super-memory.com/english/ol/sm2.htm .  Nothing here is invented; the
three update rules below are transcribed from that description:

1.  Grades are integers ``0..5``.  A grade ``>= 3`` is a *pass*.
2.  On a pass the repetition number ``n`` increments and the interval becomes
    ``I(1) = 1``, ``I(2) = 6``, ``I(n) = round(I(n-1) * EF)`` for ``n > 2``.
3.  On a fail the repetition number resets to 0 and the interval to 1 day;
    the easiness factor is *not* reset (SM-2 keeps EF across lapses).
4.  The easiness factor updates on every review as
    ``EF' = EF + (0.1 - (5-q) * (0.08 + (5-q) * 0.02))`` and is clamped at a
    floor of 1.3.  The initial value is 2.5.

The only project-specific part is :func:`grade_from_outcome`, which maps a
quiz answer onto SM-2's 0-5 scale; the mapping is stated in that docstring so
it can be argued with, and it is the *only* place a judgement call is made.

Time is injected, never read from the wall clock: every function takes an
explicit ``now`` and the scheduler takes a zero-argument ``clock``.  A
scheduler tested against ``time.time()`` can only be tested by sleeping, so
its interval arithmetic is in practice untested.  See
tests/anatomy/test_spaced_repetition.py, which drives 20-odd simulated days
through a fixed clock in microseconds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable, Optional

#: SM-2 initial easiness factor.
INITIAL_EASINESS = 2.5
#: SM-2 easiness floor.  Below this, intervals grow so slowly that the item is
#: effectively in daily rotation, which is the intended behaviour for an item
#: the learner keeps failing.
MINIMUM_EASINESS = 1.3
#: Lowest passing grade.
PASS_GRADE = 3


def utc_clock() -> datetime:
    """Default clock: timezone-aware UTC now."""
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ReviewCard:
    """SM-2 state for one quiz item.

    Attributes
    ----------
    item_id:
        Stable identifier of the structure being learned.  The quiz uses the
        BodyParts3D mesh id (``"FMA52734"``) where one is known and the config
        display name otherwise, so the id survives a display-name rewording.
    repetitions:
        SM-2 ``n`` -- consecutive passing reviews.  Reset to 0 by a lapse.
    easiness:
        SM-2 ``EF``.  Starts at 2.5, floor 1.3, no ceiling in SM-2.
    interval_days:
        Days until the next review, as computed at the last review.
    due:
        Absolute due time (last review + ``interval_days``).
    reviews / lapses:
        Lifetime counters, for reporting.  Not used by the algorithm.
    last_grade:
        The grade given at the most recent review, or ``None`` if never
        reviewed.
    """

    item_id: str
    repetitions: int = 0
    easiness: float = INITIAL_EASINESS
    interval_days: int = 0
    due: Optional[datetime] = None
    reviews: int = 0
    lapses: int = 0
    last_grade: Optional[int] = None
    last_reviewed: Optional[datetime] = None

    @property
    def is_new(self) -> bool:
        return self.reviews == 0

    def is_due(self, now: datetime) -> bool:
        """A never-reviewed card is always due; otherwise compare to ``due``."""
        if self.due is None:
            return True
        return now >= self.due

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "repetitions": self.repetitions,
            "easiness": round(self.easiness, 6),
            "interval_days": self.interval_days,
            "due": self.due.isoformat() if self.due else None,
            "reviews": self.reviews,
            "lapses": self.lapses,
            "last_grade": self.last_grade,
            "last_reviewed": (
                self.last_reviewed.isoformat() if self.last_reviewed else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReviewCard":
        def _dt(value: Optional[str]) -> Optional[datetime]:
            if not value:
                return None
            parsed = datetime.fromisoformat(value)
            # Files written by an older build may carry naive timestamps;
            # treat them as UTC rather than raising on comparison.
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed

        return cls(
            item_id=str(data["item_id"]),
            repetitions=int(data.get("repetitions", 0)),
            easiness=float(data.get("easiness", INITIAL_EASINESS)),
            interval_days=int(data.get("interval_days", 0)),
            due=_dt(data.get("due")),
            reviews=int(data.get("reviews", 0)),
            lapses=int(data.get("lapses", 0)),
            last_grade=(
                int(data["last_grade"]) if data.get("last_grade") is not None else None
            ),
            last_reviewed=_dt(data.get("last_reviewed")),
        )


def update_easiness(easiness: float, grade: int) -> float:
    """SM-2 easiness update, clamped at :data:`MINIMUM_EASINESS`.

    ``EF' = EF + (0.1 - (5-q) * (0.08 + (5-q) * 0.02))``
    """
    q = _validate_grade(grade)
    delta = 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)
    return max(MINIMUM_EASINESS, easiness + delta)


def next_interval(repetitions: int, previous_interval: int, easiness: float) -> int:
    """SM-2 interval for the review that has just *passed*.

    ``repetitions`` is the count *before* the increment, so the first pass
    (``repetitions == 0``) yields 1 day and the second yields 6.
    """
    if repetitions <= 0:
        return 1
    if repetitions == 1:
        return 6
    # SM-2 specifies I(n) = I(n-1) * EF.  Rounding is required because the
    # scheduler works in whole days; round-half-up keeps intervals monotonic
    # for EF >= 1.3 (1 * 1.3 -> 1 would stall, 6 * 1.3 -> 8).
    return max(1, int(math.floor(previous_interval * easiness + 0.5)))


def review(card: ReviewCard, grade: int, now: datetime) -> ReviewCard:
    """Apply one SM-2 review to ``card`` and return the new state.

    Pure: ``card`` is frozen and is not mutated.
    """
    q = _validate_grade(grade)

    # Order matters and follows the published listing: step 3 (compute I(n)
    # from EF) precedes step 5 (modify EF), so the interval uses the easiness
    # the item had *going into* this review.  Several ports apply the new EF
    # instead, which lengthens every interval by one EF-step; the difference
    # is small but it is a different algorithm, and this one is SM-2.
    interval_easiness = card.easiness
    easiness = update_easiness(card.easiness, q)

    if q >= PASS_GRADE:
        interval = next_interval(card.repetitions, card.interval_days,
                                 interval_easiness)
        repetitions = card.repetitions + 1
        lapses = card.lapses
    else:
        # SM-2: "repetitions start from the beginning without changing the
        # E-Factor" -- the interval resets, the easiness penalty above stands.
        interval = 1
        repetitions = 0
        lapses = card.lapses + 1

    return replace(
        card,
        repetitions=repetitions,
        easiness=easiness,
        interval_days=interval,
        due=now + timedelta(days=interval),
        reviews=card.reviews + 1,
        lapses=lapses,
        last_grade=q,
        last_reviewed=now,
    )


def grade_from_outcome(
    correct: bool,
    *,
    exact: bool = True,
    skipped: bool = False,
    elapsed_s: float = 0.0,
    fast_threshold_s: float = 5.0,
    same_system: bool = False,
) -> int:
    """Map a quiz answer onto an SM-2 grade.

    SM-2 assumes a self-graded 0-5 recall score, which a multiple-choice or
    free-text quiz does not directly produce.  This is the project's mapping,
    stated explicitly because it is the one judgement call in this module:

    ===== ===============================================================
    Grade Condition
    ===== ===============================================================
    5     correct, exact match, answered within ``fast_threshold_s``
    4     correct, exact match, slower
    3     correct only via fuzzy match (right structure, imperfect recall
          of the term) -- the lowest passing grade
    2     wrong, but the chosen structure is in the same body system as
          the answer (a confusion between neighbours)
    1     wrong, and in a different body system
    0     skipped or empty answer -- a complete blackout
    ===== ===============================================================
    """
    if skipped:
        return 0
    if correct:
        if not exact:
            return 3
        return 5 if 0.0 < elapsed_s <= fast_threshold_s else 4
    return 2 if same_system else 1


def _validate_grade(grade: int) -> int:
    q = int(grade)
    if not 0 <= q <= 5:
        raise ValueError(f"SM-2 grade must be in 0..5, got {grade!r}")
    return q


@dataclass
class Scheduler:
    """A collection of :class:`ReviewCard` with an injected clock.

    ``clock`` is a zero-argument callable returning a timezone-aware
    ``datetime``.  Tests pass a fixed or manually advanced clock; the
    application passes :func:`utc_clock`.
    """

    clock: Callable[[], datetime] = utc_clock
    cards: dict[str, ReviewCard] = field(default_factory=dict)

    def card(self, item_id: str) -> ReviewCard:
        """Existing card for ``item_id``, or a fresh one (not stored)."""
        return self.cards.get(item_id) or ReviewCard(item_id=item_id)

    def record(self, item_id: str, grade: int) -> ReviewCard:
        """Review ``item_id`` with ``grade`` at the clock's current time."""
        updated = review(self.card(item_id), grade, self.clock())
        self.cards[item_id] = updated
        return updated

    def due_items(self, candidates: Iterable[str]) -> list[str]:
        """Subset of ``candidates`` due now, most overdue first.

        New (never-reviewed) items sort last, so a session works through the
        backlog before introducing material.  Ordering within each group is by
        due time then item id, so the result is deterministic.
        """
        now = self.clock()
        due = [c for c in candidates if self.card(c).is_due(now)]

        def key(item_id: str):
            card = self.cards.get(item_id)
            if card is None or card.due is None:
                return (1, 0.0, item_id)
            return (0, (card.due - now).total_seconds(), item_id)

        return sorted(due, key=key)

    def select(self, candidates: Iterable[str], count: int) -> list[str]:
        """Up to ``count`` items to ask: due items first, then new material."""
        candidates = list(candidates)
        chosen = self.due_items(candidates)[:count]
        if len(chosen) < count:
            seen = set(chosen)
            for item_id in candidates:
                if item_id in seen:
                    continue
                chosen.append(item_id)
                if len(chosen) == count:
                    break
        return chosen

    def to_dict(self) -> dict[str, dict]:
        return {k: v.to_dict() for k, v in sorted(self.cards.items())}

    def load_dict(self, data: dict) -> None:
        self.cards = {
            k: ReviewCard.from_dict(v)
            for k, v in (data or {}).items()
            if isinstance(v, dict)
        }
