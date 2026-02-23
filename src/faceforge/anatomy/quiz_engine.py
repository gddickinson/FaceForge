"""Interactive anatomy quiz engine.

Generates questions from the AnatomySearchIndex, checks answers with
fuzzy matching, and tracks scores.
"""

import logging
import random
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class QuizQuestion:
    """A single quiz question."""
    mesh_name: str
    display_name: str
    category: str
    region: str
    mode: str  # "identify" or "locate"
    time_limit: float = 0.0  # 0 = no limit


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

    def __init__(self, search_index=None):
        self._index = search_index
        self._questions: list[QuizQuestion] = []
        self._current_idx: int = -1
        self._score = QuizScore()
        self._mode = "identify"
        self._category_filter = ""
        self._difficulty = "medium"  # easy, medium, hard
        self._active = False

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

    def start_quiz(self, mode: str = "identify", category: str = "",
                   count: int = 10, difficulty: str = "medium") -> None:
        """Start a new quiz session.

        Parameters
        ----------
        mode : str
            "identify" (name the highlighted structure) or
            "locate" (click on the named structure).
        category : str
            Filter by category ("muscle", "bone", "organ", etc.).
            Empty string = all categories.
        count : int
            Number of questions.
        difficulty : str
            "easy" (common structures), "medium" (all), "hard" (timed).
        """
        self._mode = mode
        self._category_filter = category
        self._difficulty = difficulty
        self._score = QuizScore(start_time=time.time())
        self._active = True

        # Build question pool
        pool = self._build_pool(category, difficulty)
        if not pool:
            logger.warning("No structures available for quiz")
            self._active = False
            return

        # Sample questions
        count = min(count, len(pool))
        selected = random.sample(pool, count)

        time_limit = 15.0 if difficulty == "hard" else 0.0

        self._questions = [
            QuizQuestion(
                mesh_name=entry.mesh_name,
                display_name=entry.display_name,
                category=entry.category,
                region=entry.region,
                mode=mode,
                time_limit=time_limit,
            )
            for entry in selected
        ]
        self._current_idx = 0

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

        if self._mode == "locate":
            # For locate mode, check if clicked mesh matches
            is_correct = self._fuzzy_match(answer, correct_name)
        else:
            # For identify mode, check typed answer
            is_correct = self._fuzzy_match(answer, correct_name)

        if is_correct:
            self._score.correct += 1
            self._score.streak += 1
            self._score.best_streak = max(self._score.best_streak,
                                          self._score.streak)
        else:
            self._score.incorrect += 1
            self._score.streak = 0

        return is_correct, correct_name

    def next_question(self) -> Optional[QuizQuestion]:
        """Advance to the next question.

        Returns None if quiz is complete.
        """
        self._current_idx += 1
        if self._current_idx >= len(self._questions):
            self._active = False
            return None
        return self.current_question

    def end_quiz(self) -> QuizScore:
        """End the quiz and return the final score."""
        self._active = False
        return self._score

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
