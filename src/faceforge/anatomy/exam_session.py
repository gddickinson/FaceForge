"""Exam sessions: assemble items, run a format, record the outcome.

Formats implemented here
------------------------
``sba``
    Single best answer.  ``options`` is configurable; 5 is the default because
    it is the USMLE convention.
``spot``
    Tag identification against the 3D render (L1/L2) or a simulated slice
    (L4).  Mechanically the same as ``sba`` from this module's point of view --
    the difference is that the stimulus is a tagged image, which the caller
    displays.
``emq``
    Extended matching: one option list, several stems.  Built by
    :meth:`~faceforge.anatomy.item_generators.ItemGenerator.extended_matching`,
    which drops any stem that does not identify exactly one option.
``station``
    Timed OSPE / steeplechase station: a fixed number of seconds per item and
    **no going back**.  Enforced structurally -- there is no ``previous()`` --
    and by :meth:`ExamSession.expired`, after which an answer is recorded as a
    skip.

Every item is routed through :func:`faceforge.anatomy.exam_items.present`
before the learner sees it, so an item without provenance, or an authored item
without a citation, is dropped rather than asked.  :attr:`ExamSession.refused`
lists what was dropped and why.

Timing and scheduling both use the injected ``clock``; nothing here reads the
wall clock.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Iterable, Optional, Sequence

from faceforge.anatomy.exam_items import ExamItem, ItemRefused, Option, present
from faceforge.anatomy.spaced_repetition import grade_from_outcome, utc_clock

logger = logging.getLogger(__name__)

DEFAULT_OPTIONS = 5
#: OSPE stations in UK/Australian medical schools commonly run 60-90 s per
#: station; 60 is the default here and is configurable per session.
DEFAULT_STATION_SECONDS = 60.0


@dataclass(frozen=True)
class ExamConfig:
    """What to assemble.

    ``levels`` are tried in order until ``count`` items exist, so an exam can
    be "L1 then L3" without the caller interleaving by hand.
    """

    levels: tuple[str, ...] = ("L1", )
    curriculum: str = ""
    tier: str = ""
    count: int = 10
    options: int = DEFAULT_OPTIONS
    fmt: str = "sba"
    seed: int = 0
    seconds_per_item: float = 0.0
    exam_mode: bool = True
    #: When set, focus structures are restricted to these ids (used by L4,
    #: where only structures present in the loaded scene can be scanned).
    focus_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnswerOutcome:
    """The result of answering one item."""

    item: ExamItem
    chosen: Optional[Option]
    correct: bool
    grade: int
    elapsed_s: float
    skipped: bool
    expired: bool
    explanation: str = ""
    provenance_report: str = ""


@dataclass
class ExamSession:
    """Runs one exam.

    Parameters
    ----------
    progress:
        A :class:`~faceforge.anatomy.quiz_progress.ProgressStore`, or None for
        an unrecorded session.
    curricula / generator / explanations:
        Collaborators; built from the project data when omitted.
    clock:
        Zero-argument callable returning a timezone-aware ``datetime``.
    """

    progress: object = None
    curricula: Optional[dict] = None
    generator: object = None
    explanations: object = None
    clock: Callable[[], datetime] = utc_clock

    items: list[ExamItem] = field(default_factory=list)
    refused: list[tuple[str, list[str]]] = field(default_factory=list)
    outcomes: list[AnswerOutcome] = field(default_factory=list)
    config: ExamConfig = field(default_factory=ExamConfig)
    _index: int = -1
    _shown_at: Optional[datetime] = None
    _answered: set[str] = field(default_factory=set)

    def __post_init__(self):
        if self.generator is None:
            from faceforge.anatomy.item_generators import ItemGenerator
            self.generator = ItemGenerator()
        if self.curricula is None:
            from faceforge.anatomy.curricula import get_curricula
            self.curricula = get_curricula()
        if self.explanations is None:
            from faceforge.anatomy.answer_explanations import ExplanationBuilder
            self.explanations = ExplanationBuilder()

    # -- assembly ----------------------------------------------------------

    def focus_order(self, config: ExamConfig) -> list[str]:
        """Structure ids to build items over, in the order they will be asked.

        Order of preference, so revision and examination behave sensibly:

        1. explicit ``config.focus_ids``;
        2. the named curriculum, tier-filtered, re-ordered by the SM-2
           scheduler (overdue first, then new material) when a progress store
           is attached;
        3. the curriculum's own tier ordering.
        """
        if config.focus_ids:
            return list(config.focus_ids)
        curriculum = self.curricula.get(config.curriculum)
        if curriculum is None:
            return []
        ids = curriculum.item_ids(config.tier) if config.tier \
            else curriculum.item_ids()
        if self.progress is not None and getattr(self.progress, "scheduler", None):
            # Ask for more focus structures than items, because a structure can
            # fail to yield an item at a given level.
            return self.progress.scheduler.select(ids, max(config.count * 4, 8))
        return ids

    def build(self, config: ExamConfig) -> list[ExamItem]:
        """Assemble presentable items for ``config``.

        Items that fail :func:`present` are dropped into :attr:`refused`
        instead of being asked, and the shortfall is visible in the returned
        length -- this module never pads an exam to length with something it
        cannot justify.
        """
        self.config = config
        self.refused = []
        focus = self.focus_order(config)
        if not focus:
            return []

        seconds = config.seconds_per_item
        if config.fmt == "station" and seconds <= 0:
            seconds = DEFAULT_STATION_SECONDS

        candidates: list[ExamItem] = []
        if config.fmt == "emq":
            candidates = list(self.generator.extended_matching(
                focus, seed=config.seed, seconds=seconds))
        else:
            for level in config.levels:
                if len(candidates) >= config.count:
                    break
                candidates.extend(self.generator.generate(
                    level, focus, config.count - len(candidates),
                    options=config.options, seed=config.seed, seconds=seconds))

        good: list[ExamItem] = []
        for item in candidates:
            if config.fmt in ("sba", "station") and item.fmt in ("sba", "spot"):
                # A generator emits its natural format; a station or a written
                # paper re-frames the same item.  Only the presentation
                # changes, so the option list and key are untouched.
                item = _reformat(item, config.fmt, seconds)
            try:
                good.append(present(item, exam_mode=config.exam_mode,
                                    min_options=2))
            except ItemRefused as exc:
                self.refused.append((exc.item_uid, exc.reasons))
        self.items = good[:config.count]
        return self.items

    def start(self, config: Optional[ExamConfig] = None) -> Optional[ExamItem]:
        """Build (if needed) and present the first item."""
        if config is not None:
            self.build(config)
        self.outcomes = []
        self._answered = set()
        self._index = 0 if self.items else -1
        self._shown_at = self.clock() if self.items else None
        return self.current

    # -- running -----------------------------------------------------------

    @property
    def current(self) -> Optional[ExamItem]:
        if 0 <= self._index < len(self.items):
            return self.items[self._index]
        return None

    @property
    def remaining(self) -> int:
        return max(0, len(self.items) - self._index - 1)

    @property
    def finished(self) -> bool:
        return self._index >= len(self.items)

    @property
    def score(self) -> tuple[int, int]:
        """(correct, answered)."""
        return (sum(1 for o in self.outcomes if o.correct), len(self.outcomes))

    def time_left(self) -> Optional[float]:
        """Seconds remaining on the current item, or None if untimed."""
        item = self.current
        if item is None or item.seconds <= 0 or self._shown_at is None:
            return None
        spent = (self.clock() - self._shown_at).total_seconds()
        return max(0.0, item.seconds - spent)

    def expired(self) -> bool:
        left = self.time_left()
        return left is not None and left <= 0.0

    def answer(self, option_index: Optional[int]) -> Optional[AnswerOutcome]:
        """Record an answer for the current item.

        ``option_index`` of ``None`` (or an out-of-range index) is a skip.  An
        answer submitted after the station clock has expired is recorded as a
        skip too, with ``expired=True``, rather than being silently accepted --
        a station that lets a late answer count is not a timed station.
        """
        item = self.current
        if item is None:
            return None
        now = self.clock()
        elapsed = 0.0 if self._shown_at is None else \
            max(0.0, (now - self._shown_at).total_seconds())
        expired = item.seconds > 0 and elapsed > item.seconds

        valid = (option_index is not None
                 and 0 <= option_index < len(item.options))
        chosen = item.options[option_index] if valid else None
        skipped = (not valid) or expired
        correct = bool(valid and not expired and option_index == item.answer_index)

        explanation, same_system = self._explain(item, chosen, correct)
        grade = grade_from_outcome(
            correct, exact=True, skipped=skipped, elapsed_s=elapsed,
            same_system=same_system,
        )
        outcome = AnswerOutcome(
            item=item, chosen=chosen, correct=correct, grade=grade,
            elapsed_s=elapsed, skipped=skipped, expired=expired,
            explanation=explanation,
            provenance_report=item.provenance_report(),
        )
        self.outcomes.append(outcome)
        self._answered.add(item.uid)
        self._record(item, outcome)
        return outcome

    def _explain(self, item: ExamItem, chosen: Optional[Option],
                 correct: bool) -> tuple[str, bool]:
        if correct or self.explanations is None:
            return ("", False)
        answer = item.answer
        chosen_ref = (chosen.item_id or chosen.text) if chosen else ""
        correct_ref = (answer.item_id or answer.text) if answer else ""
        if not correct_ref:
            return ("", False)
        try:
            expl = self.explanations.explain(chosen_ref, correct_ref)
        except Exception:                          # noqa: BLE001 - diagnostic
            logger.exception("Could not build exam explanation")
            return ("", False)
        return (expl.text, expl.same_system)

    def _record(self, item: ExamItem, outcome: AnswerOutcome) -> None:
        if self.progress is None:
            return
        try:
            self.progress.record(
                item.focus_id or item.uid,
                (item.answer.text if item.answer else item.uid),
                outcome.correct,
                outcome.grade,
                given_answer=(outcome.chosen.text if outcome.chosen else ""),
                mode=f"{item.level}/{item.fmt}",
                curriculum=self.config.curriculum,
                tier=self.config.tier,
                elapsed_s=outcome.elapsed_s,
                skipped=outcome.skipped,
            )
        except Exception:                          # noqa: BLE001 - diagnostic
            logger.exception("Could not record exam outcome")

    def next_item(self) -> Optional[ExamItem]:
        """Advance.  There is no way back -- see the module docstring."""
        self._index += 1
        self._shown_at = self.clock() if self.current is not None else None
        return self.current

    def finish(self) -> tuple[int, int]:
        """End the session, persist progress, return ``(correct, answered)``."""
        self._index = len(self.items)
        if self.progress is not None:
            try:
                self.progress.save()
            except OSError:
                logger.exception("Could not save progress after exam")
        return self.score

    # -- reporting ---------------------------------------------------------

    def audit(self) -> str:
        """Provenance for every item in the session, for review before use."""
        lines = [f"Exam: {len(self.items)} item(s), "
                 f"{len(self.refused)} refused, levels={self.config.levels}, "
                 f"format={self.config.fmt}"]
        lines += [item.provenance_report() for item in self.items]
        lines += [f"REFUSED {uid}: {', '.join(reasons)}"
                  for uid, reasons in self.refused]
        return "\n".join(lines)


def _reformat(item: ExamItem, fmt: str, seconds: float) -> ExamItem:
    """Same item, different presentation.  Never touches options or key."""
    from dataclasses import replace
    if item.fmt == fmt and item.seconds == seconds:
        return item
    # A spot item stays a spot item in a station: the stimulus is still a tag
    # on the render, only the clock changes.
    new_fmt = item.fmt if (fmt == "station" and item.fmt == "spot") else fmt
    return replace(item, fmt=new_fmt, seconds=seconds, uid=item.uid)
