"""Interactive anatomy quiz engine.

Generates questions from the AnatomySearchIndex or from a named curriculum,
checks answers with fuzzy matching, tracks scores, and -- when given a
:class:`~faceforge.anatomy.quiz_progress.ProgressStore` -- persists a per-user
history and schedules review with SM-2.

Three collaborators, each optional so the pre-existing constructor
``QuizEngine(search_index)`` keeps working unchanged:

``progress``
    :class:`~faceforge.anatomy.quiz_progress.ProgressStore`.  When present,
    every answer is appended to the user's history and fed to the SM-2
    scheduler, and question selection prefers items that are due.
``curricula``
    ``key -> Curriculum`` from :mod:`faceforge.anatomy.curricula`.  When a
    curriculum is named in :meth:`QuizEngine.start_quiz`, the pool is that
    curriculum's ordered items instead of the whole search index.
``explanations``
    :class:`~faceforge.anatomy.answer_explanations.ExplanationBuilder`.  Sets
    :attr:`QuizEngine.last_explanation` after every answer.

Time is injected as ``clock`` (a zero-argument callable returning a
timezone-aware ``datetime``) so the scheduling and the answer-speed grading
are testable without sleeping.
"""

import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from typing import Callable, Optional

from faceforge.anatomy.spaced_repetition import grade_from_outcome, utc_clock

logger = logging.getLogger(__name__)

#: How ``difficulty`` restricts a curriculum's tiers.  "hard" also imposes the
#: 15 s per-question limit that already existed.
DIFFICULTY_TIERS = {
    "easy": ("foundation",),
    "medium": ("foundation", "intermediate"),
    "hard": ("foundation", "intermediate", "advanced", "unclassified"),
}


@dataclass
class QuizQuestion:
    """A single quiz question."""
    mesh_name: str
    display_name: str
    category: str
    region: str
    mode: str  # "identify" or "locate"
    time_limit: float = 0.0  # 0 = no limit
    #: Stable structure id (BodyParts3D mesh id) when known; this is what the
    #: progress history and the SM-2 cards are keyed by, so a display-name
    #: rewording does not orphan a learner's history.
    item_id: str = ""
    #: FMA preferred term, when the crosswalk has one.
    preferred_label: str = ""
    #: Curriculum this question came from ("" for search-index questions).
    curriculum: str = ""
    tier: str = ""


@dataclass
class QuizScore:
    """Quiz score tracking."""
    correct: int = 0
    incorrect: int = 0
    total: int = 0
    streak: int = 0
    best_streak: int = 0
    start_time: float = 0.0

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time if self.start_time > 0 else 0.0


class QuizEngine:
    """Generates quiz questions and checks answers.

    Parameters
    ----------
    search_index : AnatomySearchIndex
        The search index to draw questions from.
    """

    def __init__(
        self,
        search_index=None,
        progress=None,
        curricula: Optional[dict] = None,
        explanations=None,
        clock: Callable[[], datetime] = utc_clock,
        autoload: bool = True,
    ):
        self._index = search_index
        self._progress = progress
        self._curricula = curricula
        self._explanations = explanations
        self._clock = clock
        # Default-on, lazily: the shipped construction site is
        # ``QuizEngine(ctx.search_index)`` in appcontext.py, and a learner's
        # progress should not depend on that call being updated.  Lazy because
        # constructing a store touches the filesystem; nothing is *written*
        # until save_progress()/end_quiz().  Pass autoload=False for a fully
        # in-memory engine.
        self._autoload = autoload
        self._questions: list[QuizQuestion] = []
        self._current_idx: int = -1
        self._score = QuizScore()
        self._mode = "identify"
        self._category_filter = ""
        self._difficulty = "medium"  # easy, medium, hard
        self._curriculum_key = ""
        self._active = False
        self._question_started: Optional[datetime] = None
        #: Explanation for the most recent answer, or None before the first.
        self.last_explanation = None
        #: SM-2 grade awarded for the most recent answer, or None.
        self.last_grade = None

    @property
    def progress(self):
        """The attached :class:`ProgressStore`, created on first use.

        Returns None when ``autoload=False`` and no store was injected.  A
        failure to construct or read the store is logged and downgraded to
        None: a corrupt history must not stop a quiz.
        """
        if self._progress is None and self._autoload:
            try:
                from faceforge.anatomy.quiz_progress import ProgressStore
                store = ProgressStore(clock=self._clock)
                store.load()
                self._progress = store
            except Exception:                      # noqa: BLE001 - diagnostic
                logger.exception("Quiz progress unavailable; running unrecorded")
                self._autoload = False
        return self._progress

    @property
    def explanations(self):
        """The :class:`ExplanationBuilder`, created on first use."""
        if self._explanations is None and self._autoload:
            try:
                from faceforge.anatomy.answer_explanations import ExplanationBuilder
                self._explanations = ExplanationBuilder()
            except Exception:                      # noqa: BLE001 - diagnostic
                logger.exception("Answer explanations unavailable")
                self._autoload = False
        return self._explanations

        # Common structures for "easy" mode
        self._easy_structures = {
            "muscle": [
                "Biceps", "Triceps", "Deltoid", "Pectoralis",
                "Gluteus", "Quadriceps", "Hamstring", "Gastrocnemius",
                "Trapezius", "Latissimus", "Rectus Abdominis",
            ],
            "bone": [
                "Femur", "Tibia", "Humerus", "Radius", "Ulna",
                "Scapula", "Clavicle", "Sternum", "Pelvis",
            ],
            "organ": [
                "Heart", "Lung", "Liver", "Kidney", "Stomach",
                "Brain", "Spleen", "Pancreas", "Bladder",
            ],
        }

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def score(self) -> QuizScore:
        return self._score

    @property
    def current_question(self) -> Optional[QuizQuestion]:
        if 0 <= self._current_idx < len(self._questions):
            return self._questions[self._current_idx]
        return None

    @property
    def questions_remaining(self) -> int:
        return max(0, len(self._questions) - self._current_idx - 1)

    @property
    def curricula(self) -> dict:
        """Available curricula, loaded lazily from the project's configs."""
        if self._curricula is None:
            from faceforge.anatomy.curricula import get_curricula
            self._curricula = get_curricula()
        return self._curricula

    def curriculum_keys(self) -> list[str]:
        """Curriculum keys ordered largest first -- what a UI should offer."""
        return [k for k, _ in sorted(self.curricula.items(),
                                     key=lambda kv: (-len(kv[1]), kv[0]))]

    def start_quiz(self, mode: str = "identify", category: str = "",
                   count: int = 10, difficulty: str = "medium",
                   curriculum: str = "", tier: str = "") -> None:
        """Start a new quiz session.

        Parameters
        ----------
        mode : str
            "identify" (name the highlighted structure) or
            "locate" (click on the named structure).
        category : str
            Filter by category ("muscle", "bone", "organ", etc.).
            Empty string = all categories.  Ignored when ``curriculum`` is
            given, since a curriculum already defines its membership.
        count : int
            Number of questions.
        difficulty : str
            "easy", "medium", "hard".  For a curriculum this selects tiers
            (see :data:`DIFFICULTY_TIERS`); for the search-index pool it keeps
            its previous meaning (a common-structure filter).  "hard" adds the
            15 s per-question limit in both cases.
        curriculum : str
            Key into :attr:`curricula`.  When given, questions come from that
            study set, ordered by the SM-2 scheduler (due items first) if a
            progress store is attached.
        tier : str
            Restrict to one tier ("foundation", "intermediate", "advanced").
            Overrides the ``difficulty`` tier selection.
        """
        self._mode = mode
        self._category_filter = category
        self._difficulty = difficulty
        self._curriculum_key = curriculum
        self._score = QuizScore(start_time=time.time())
        self._active = True
        self.last_explanation = None

        time_limit = 15.0 if difficulty == "hard" else 0.0

        if curriculum:
            questions = self._curriculum_questions(
                curriculum, tier, difficulty, count, mode, time_limit)
        else:
            questions = self._index_questions(
                category, difficulty, count, mode, time_limit)

        if not questions:
            logger.warning("No structures available for quiz "
                           "(curriculum=%r, category=%r, difficulty=%r)",
                           curriculum, category, difficulty)
            self._active = False
            self._questions = []
            self._current_idx = -1
            return

        self._questions = questions
        self._current_idx = 0
        self._question_started = self._clock()

    def _curriculum_questions(self, key: str, tier: str, difficulty: str,
                              count: int, mode: str, time_limit: float
                              ) -> list[QuizQuestion]:
        cur = self.curricula.get(key)
        if cur is None:
            logger.warning("Unknown curriculum %r", key)
            return []

        tiers = (tier,) if tier else DIFFICULTY_TIERS.get(
            difficulty, DIFFICULTY_TIERS["medium"])
        items = [i for i in cur.items if i.tier in tiers]
        if not items:
            return []

        by_id = {i.item_id: i for i in items}
        store = self.progress
        if store is not None:
            # Spaced repetition drives the order: overdue first, then new
            # material, then (only if still short) anything else.
            chosen_ids = store.scheduler.select(list(by_id), count)
        else:
            chosen_ids = [i.item_id for i in items][:count]

        return [
            QuizQuestion(
                mesh_name=by_id[i].display_name,
                display_name=by_id[i].display_name,
                category=by_id[i].category,
                region=by_id[i].system,
                mode=mode,
                time_limit=time_limit,
                item_id=by_id[i].item_id,
                preferred_label=by_id[i].preferred_label,
                curriculum=key,
                tier=by_id[i].tier,
            )
            for i in chosen_ids if i in by_id
        ]

    def _index_questions(self, category: str, difficulty: str, count: int,
                         mode: str, time_limit: float) -> list[QuizQuestion]:
        pool = self._build_pool(category, difficulty)
        if not pool:
            return []
        selected = random.sample(pool, min(count, len(pool)))
        return [
            QuizQuestion(
                mesh_name=entry.mesh_name,
                display_name=entry.display_name,
                category=entry.category,
                region=entry.region,
                mode=mode,
                time_limit=time_limit,
                item_id=getattr(entry, "source_id", "") or entry.mesh_name,
                preferred_label=getattr(entry, "preferred_label", ""),
            )
            for entry in selected
        ]

    def _build_pool(self, category: str, difficulty: str) -> list:
        """Build the question pool from the search index."""
        if self._index is None:
            return []

        entries = self._index.entries
        if category:
            entries = [e for e in entries if e.category == category]

        if difficulty == "easy":
            # Filter to common structures
            easy_names = set()
            for names in self._easy_structures.values():
                easy_names.update(n.lower() for n in names)
            entries = [
                e for e in entries
                if any(name in e.display_name.lower() for name in easy_names)
            ]

        return entries

    def check_answer(self, answer: str) -> tuple[bool, str]:
        """Check an answer against the current question.

        Parameters
        ----------
        answer : str
            User's answer (structure name for "identify" mode,
            mesh name for "locate" mode).

        Returns
        -------
        tuple[bool, str]
            (correct, correct_answer) where correct_answer is the
            expected answer string.
        """
        question = self.current_question
        if question is None:
            return False, ""

        correct_name = question.display_name
        self._score.total += 1

        # An empty answer is the skip path (the dialog's Skip button and the
        # hard-mode timeout both call through here with ""), which is SM-2
        # grade 0 rather than a wrong answer.
        skipped = not answer.strip()

        # Accept the FMA preferred term as well as the display name: a learner
        # who types "Right temporalis" for "Temporalis R" knows the structure.
        exact = self._exact_match(answer, correct_name) or (
            bool(question.preferred_label)
            and self._exact_match(answer, question.preferred_label)
        )
        is_correct = (not skipped) and (
            exact
            or self._fuzzy_match(answer, correct_name)
            or (bool(question.preferred_label)
                and self._fuzzy_match(answer, question.preferred_label))
        )

        self._record_outcome(question, answer, is_correct, exact, skipped)

        if is_correct:
            self._score.correct += 1
            self._score.streak += 1
            self._score.best_streak = max(self._score.best_streak,
                                          self._score.streak)
        else:
            self._score.incorrect += 1
            self._score.streak = 0

        return is_correct, correct_name

    def _record_outcome(self, question: QuizQuestion, answer: str,
                        is_correct: bool, exact: bool, skipped: bool) -> None:
        """Build the explanation, grade the answer and persist it.

        Failures here are logged, never raised: a full disk must lose the
        history, not the quiz in progress.
        """
        now = self._clock()
        elapsed = 0.0
        if self._question_started is not None:
            elapsed = max(0.0, (now - self._question_started).total_seconds())

        explanation = None
        builder = self.explanations
        if builder is not None:
            try:
                explanation = builder.explain(
                    answer, question.item_id or question.display_name)
            except Exception:                      # noqa: BLE001 - diagnostic
                logger.exception("Could not build answer explanation")
        self.last_explanation = explanation

        grade = grade_from_outcome(
            is_correct,
            exact=exact,
            skipped=skipped,
            elapsed_s=elapsed,
            same_system=bool(explanation is not None and explanation.same_system),
        )
        self.last_grade = grade

        store = self.progress
        if store is None:
            return
        try:
            store.record(
                question.item_id or question.display_name,
                question.display_name,
                is_correct,
                grade,
                given_answer=answer,
                mode=question.mode,
                curriculum=question.curriculum,
                tier=question.tier,
                elapsed_s=elapsed,
                skipped=skipped,
            )
        except Exception:                          # noqa: BLE001 - diagnostic
            logger.exception("Could not record quiz progress")

    def next_question(self) -> Optional[QuizQuestion]:
        """Advance to the next question.

        Returns None if quiz is complete.
        """
        self._current_idx += 1
        if self._current_idx >= len(self._questions):
            self._active = False
            return None
        self._question_started = self._clock()
        return self.current_question

    def end_quiz(self) -> QuizScore:
        """End the quiz, persist progress if a store is attached, return score."""
        self._active = False
        self.save_progress()
        return self._score

    def save_progress(self) -> bool:
        """Write the progress file.  Returns False if there is nothing to write.

        Never raises: an unwritable data directory must not take the app down
        at the end of a quiz.
        """
        store = self._progress          # never *create* a store just to save
        if store is None:
            return False
        try:
            store.save()
            return True
        except OSError:
            logger.exception("Could not save quiz progress to %s",
                             getattr(store, "path", "?"))
            return False

    @staticmethod
    def _exact_match(answer: str, correct: str) -> bool:
        """Case- and whitespace-insensitive equality."""
        return answer.strip().lower() == correct.strip().lower()

    @staticmethod
    def _fuzzy_match(answer: str, correct: str, threshold: float = 0.7) -> bool:
        """Check if answer is close enough to correct answer."""
        a = answer.lower().strip()
        c = correct.lower().strip()

        # Exact match
        if a == c:
            return True

        # Substring match
        if a in c or c in a:
            return True

        # Check if answer matches without side suffix (R/L)
        c_no_side = c.rsplit(" ", 1)[0] if c.endswith((" R", " L")) else c
        if a == c_no_side.lower():
            return True

        # Sequence matcher fuzzy match
        ratio = SequenceMatcher(None, a, c).ratio()
        return ratio >= threshold
