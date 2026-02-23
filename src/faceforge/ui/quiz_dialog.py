"""Interactive anatomy quiz dialog."""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QPushButton, QComboBox, QLabel, QLineEdit,
    QGroupBox, QSpinBox, QStackedWidget, QFrame,
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont

from faceforge.anatomy.quiz_engine import QuizEngine, QuizScore


class QuizDialog(QDialog):
    """Modal dialog for anatomy quiz.

    Signals
    -------
    highlight_requested(str)
        Request to highlight a mesh by name.
    clear_highlight()
        Request to clear all highlights.
    quiz_click_mode(bool)
        Enable/disable click-to-answer mode.
    """

    highlight_requested = Signal(str)
    clear_highlight = Signal()
    quiz_click_mode = Signal(bool)
    answer_submitted = Signal(str)

    def __init__(self, quiz_engine: QuizEngine, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Anatomy Quiz")
        self.setMinimumSize(400, 500)
        self._engine = quiz_engine
        self._timer_value = 0

        layout = QVBoxLayout(self)

        # ── Setup page ──
        self._pages = QStackedWidget()
        layout.addWidget(self._pages)

        # Page 0: Setup
        setup_page = QFrame()
        setup_layout = QVBoxLayout(setup_page)

        setup_group = QGroupBox("Quiz Settings")
        form = QFormLayout(setup_group)

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Identify", "Locate"])
        form.addRow("Mode:", self._mode_combo)

        self._category_combo = QComboBox()
        self._category_combo.addItems([
            "All", "Muscles", "Bones", "Organs", "Vessels",
        ])
        form.addRow("Category:", self._category_combo)

        self._difficulty_combo = QComboBox()
        self._difficulty_combo.addItems(["Easy", "Medium", "Hard"])
        self._difficulty_combo.setCurrentIndex(1)
        form.addRow("Difficulty:", self._difficulty_combo)

        self._count_spin = QSpinBox()
        self._count_spin.setRange(5, 50)
        self._count_spin.setValue(10)
        form.addRow("Questions:", self._count_spin)

        setup_layout.addWidget(setup_group)

        start_btn = QPushButton("Start Quiz")
        start_btn.setStyleSheet("font-size: 14px; padding: 8px;")
        start_btn.clicked.connect(self._start_quiz)
        setup_layout.addWidget(start_btn)
        setup_layout.addStretch()

        self._pages.addWidget(setup_page)

        # Page 1: Quiz
        quiz_page = QFrame()
        quiz_layout = QVBoxLayout(quiz_page)

        # Score bar
        score_row = QHBoxLayout()
        self._score_label = QLabel("Score: 0/0")
        self._score_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        score_row.addWidget(self._score_label)

        self._streak_label = QLabel("Streak: 0")
        score_row.addWidget(self._streak_label)

        self._timer_label = QLabel("")
        self._timer_label.setStyleSheet("color: #ff6666;")
        score_row.addWidget(self._timer_label)

        score_row.addStretch()
        self._remaining_label = QLabel("")
        score_row.addWidget(self._remaining_label)
        quiz_layout.addLayout(score_row)

        # Question display
        self._question_label = QLabel("Question")
        self._question_label.setFont(QFont("Arial", 14))
        self._question_label.setWordWrap(True)
        self._question_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._question_label.setStyleSheet(
            "background: rgba(40, 42, 50, 0.9); padding: 16px; "
            "border-radius: 8px; margin: 8px 0;"
        )
        quiz_layout.addWidget(self._question_label)

        # Answer input (for identify mode)
        self._answer_input = QLineEdit()
        self._answer_input.setPlaceholderText("Type the structure name...")
        self._answer_input.returnPressed.connect(self._submit_answer)
        quiz_layout.addWidget(self._answer_input)

        # Submit button
        btn_row = QHBoxLayout()
        self._submit_btn = QPushButton("Submit")
        self._submit_btn.clicked.connect(self._submit_answer)
        btn_row.addWidget(self._submit_btn)

        self._skip_btn = QPushButton("Skip")
        self._skip_btn.clicked.connect(self._skip_question)
        btn_row.addWidget(self._skip_btn)
        quiz_layout.addLayout(btn_row)

        # Feedback label
        self._feedback_label = QLabel("")
        self._feedback_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._feedback_label.setStyleSheet("font-size: 13px; padding: 8px;")
        quiz_layout.addWidget(self._feedback_label)

        quiz_layout.addStretch()

        # End quiz button
        end_btn = QPushButton("End Quiz")
        end_btn.clicked.connect(self._end_quiz)
        quiz_layout.addWidget(end_btn)

        self._pages.addWidget(quiz_page)

        # Page 2: Results
        results_page = QFrame()
        results_layout = QVBoxLayout(results_page)

        self._results_label = QLabel("")
        self._results_label.setFont(QFont("Arial", 14))
        self._results_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._results_label.setWordWrap(True)
        results_layout.addWidget(self._results_label)

        results_layout.addStretch()

        restart_btn = QPushButton("New Quiz")
        restart_btn.clicked.connect(lambda: self._pages.setCurrentIndex(0))
        results_layout.addWidget(restart_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        results_layout.addWidget(close_btn)

        self._pages.addWidget(results_page)

        # Timer for hard mode
        self._countdown_timer = QTimer(self)
        self._countdown_timer.setInterval(1000)
        self._countdown_timer.timeout.connect(self._on_timer_tick)

    def _start_quiz(self) -> None:
        mode_map = {0: "identify", 1: "locate"}
        cat_map = {0: "", 1: "muscle", 2: "bone", 3: "organ", 4: "vessel"}
        diff_map = {0: "easy", 1: "medium", 2: "hard"}

        mode = mode_map.get(self._mode_combo.currentIndex(), "identify")
        category = cat_map.get(self._category_combo.currentIndex(), "")
        difficulty = diff_map.get(self._difficulty_combo.currentIndex(), "medium")
        count = self._count_spin.value()

        self._engine.start_quiz(mode, category, count, difficulty)

        if mode == "locate":
            self.quiz_click_mode.emit(True)
            self._answer_input.setVisible(False)
            self._submit_btn.setVisible(False)
        else:
            self.quiz_click_mode.emit(False)
            self._answer_input.setVisible(True)
            self._submit_btn.setVisible(True)

        self._pages.setCurrentIndex(1)
        self._show_question()

    def _show_question(self) -> None:
        q = self._engine.current_question
        if q is None:
            self._end_quiz()
            return

        score = self._engine.score
        self._score_label.setText(f"Score: {score.correct}/{score.total}")
        self._streak_label.setText(f"Streak: {score.streak}")
        self._remaining_label.setText(f"Remaining: {self._engine.questions_remaining}")
        self._feedback_label.setText("")
        self._answer_input.clear()
        self._answer_input.setFocus()

        if q.mode == "identify":
            self._question_label.setText(
                "What is the name of the highlighted structure?"
            )
            self.highlight_requested.emit(q.mesh_name)
        else:
            self._question_label.setText(
                f"Click on: {q.display_name}"
            )
            self.clear_highlight.emit()

        # Timer for hard mode
        if q.time_limit > 0:
            self._timer_value = int(q.time_limit)
            self._timer_label.setText(f"Time: {self._timer_value}s")
            self._countdown_timer.start()
        else:
            self._timer_label.setText("")
            self._countdown_timer.stop()

    def _on_timer_tick(self) -> None:
        self._timer_value -= 1
        self._timer_label.setText(f"Time: {self._timer_value}s")
        if self._timer_value <= 0:
            self._countdown_timer.stop()
            self._skip_question()

    def _submit_answer(self) -> None:
        answer = self._answer_input.text().strip()
        if not answer:
            return
        self._check_answer(answer)

    def submit_click_answer(self, mesh_name: str) -> None:
        """Called when user clicks on a mesh in locate mode."""
        if self._engine.is_active and self._engine.current_question:
            if self._engine.current_question.mode == "locate":
                self._check_answer(mesh_name)

    def _check_answer(self, answer: str) -> None:
        self._countdown_timer.stop()
        correct, correct_name = self._engine.check_answer(answer)

        if correct:
            self._feedback_label.setText(f"Correct! {correct_name}")
            self._feedback_label.setStyleSheet(
                "font-size: 13px; padding: 8px; color: #66ff66;"
            )
        else:
            self._feedback_label.setText(
                f"Incorrect. Answer: {correct_name}"
            )
            self._feedback_label.setStyleSheet(
                "font-size: 13px; padding: 8px; color: #ff6666;"
            )

        score = self._engine.score
        self._score_label.setText(f"Score: {score.correct}/{score.total}")
        self._streak_label.setText(f"Streak: {score.streak}")

        # Move to next question after brief delay
        QTimer.singleShot(1500, self._next_question)

    def _skip_question(self) -> None:
        self._countdown_timer.stop()
        q = self._engine.current_question
        if q:
            self._feedback_label.setText(f"Skipped. Answer: {q.display_name}")
            self._feedback_label.setStyleSheet(
                "font-size: 13px; padding: 8px; color: #ffaa00;"
            )
            self._engine.check_answer("")  # Count as incorrect

        score = self._engine.score
        self._score_label.setText(f"Score: {score.correct}/{score.total}")
        QTimer.singleShot(1500, self._next_question)

    def _next_question(self) -> None:
        q = self._engine.next_question()
        if q is None:
            self._end_quiz()
        else:
            self._show_question()

    def _end_quiz(self) -> None:
        self._countdown_timer.stop()
        self.clear_highlight.emit()
        self.quiz_click_mode.emit(False)

        score = self._engine.end_quiz()
        elapsed = score.elapsed_time

        pct = f"{score.accuracy * 100:.0f}%" if score.total > 0 else "N/A"
        self._results_label.setText(
            f"Quiz Complete!\n\n"
            f"Score: {score.correct}/{score.total} ({pct})\n"
            f"Best Streak: {score.best_streak}\n"
            f"Time: {elapsed:.0f}s\n"
        )
        self._pages.setCurrentIndex(2)

    def set_quiz_engine(self, engine: QuizEngine) -> None:
        """Replace the quiz engine (for late binding)."""
        self._engine = engine

    def on_mesh_clicked(self, mesh_name: str) -> None:
        """Handle a mesh click from the GL viewport (locate mode)."""
        self.submit_click_answer(mesh_name)

    def closeEvent(self, event) -> None:
        self.clear_highlight.emit()
        self.quiz_click_mode.emit(False)
        self._countdown_timer.stop()
        super().closeEvent(event)
