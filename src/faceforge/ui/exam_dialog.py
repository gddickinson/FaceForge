"""Examination dialog: the exam tier, reachable from the UI.

Separate from :mod:`faceforge.ui.quiz_dialog` on purpose.  The quiz is a
free-text revision aid; an examination is a single-best-answer paper with a
fixed key, a per-item clock and an audit trail, and folding both into one
widget would make each worse.  They share the progress store, so a structure
revised in the quiz is scheduled in the exam and vice versa.

What the setup page exposes maps one-to-one onto
:class:`~faceforge.anatomy.exam_session.ExamConfig`; nothing is invented here.
The level list is built from :data:`faceforge.anatomy.exam_items.LEVEL_TITLES`
so a level added to the schema appears here without an edit, and each level's
tooltip names the data source that powers it -- a learner is entitled to know
that L3 items come from the FMA parent table rather than from an author.

L4 (radiological cross-sections) needs a loaded scene, which this dialog does
not own.  :meth:`ExamDialog.set_radiology_source` accepts a callable returning
``[(mesh, world_matrix, mesh_id), ...]``; until one is set, L4 is offered but
disabled with a tooltip saying why, rather than silently producing nothing.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from faceforge.anatomy.exam_items import LEVEL_SOURCES, LEVEL_TITLES
from faceforge.anatomy.exam_session import (
    DEFAULT_STATION_SECONDS,
    ExamConfig,
    ExamSession,
)

logger = logging.getLogger(__name__)

FORMAT_CHOICES = (
    ("Single best answer", "sba"),
    ("Timed station (OSPE)", "station"),
    ("Extended matching", "emq"),
)

#: Levels needing a loaded 3D scene, and why.
NEEDS_SCENE = {
    "L4": "requires a loaded scene to scan; open a body first",
}


class ExamDialog(QDialog):
    """Setup -> paper -> results, over a :class:`ExamSession`."""

    def __init__(self, session: Optional[ExamSession] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Anatomy Examination")
        self.resize(620, 620)

        self._session = session if session is not None else ExamSession()
        self._radiology_source: Optional[Callable[[], Sequence[tuple]]] = None
        self._radiology_items: dict[str, object] = {}
        self._level_boxes: dict[str, QCheckBox] = {}
        self._option_buttons: list[QRadioButton] = []
        self._answered = False

        self._pages = QStackedWidget()
        self._pages.addWidget(self._build_setup_page())
        self._pages.addWidget(self._build_paper_page())
        self._pages.addWidget(self._build_results_page())

        layout = QVBoxLayout(self)
        layout.addWidget(self._pages)

        # Station clock.  One shared timer, started only for timed items.
        self._timer = QTimer(self)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._on_tick)

    # -- setup page --------------------------------------------------------

    def _build_setup_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("<b>Anatomy examination</b>"))

        levels_box = QGroupBox("Levels")
        levels_layout = QVBoxLayout(levels_box)
        for level, title in LEVEL_TITLES.items():
            box = QCheckBox(f"{level} \u2014 {title}")
            box.setToolTip(f"Source: {LEVEL_SOURCES.get(level, 'unknown')}")
            if level in NEEDS_SCENE:
                box.setEnabled(False)
                box.setToolTip(f"{box.toolTip()}\n{NEEDS_SCENE[level]}")
            if level == "L5":
                # No vignette content ships; the level exists as a format.
                box.setEnabled(False)
                box.setToolTip(
                    f"{box.toolTip()}\nno vignette content is installed; "
                    "load a cited set to enable")
            box.setChecked(level == "L1")
            self._level_boxes[level] = box
            levels_layout.addWidget(box)
        layout.addWidget(levels_box)

        form = QFormLayout()
        self._curriculum_combo = QComboBox()
        self._curriculum_keys: list[str] = []
        self._populate_curricula()
        form.addRow("Curriculum:", self._curriculum_combo)

        self._tier_combo = QComboBox()
        self._tier_combo.addItem("All tiers")
        form.addRow("Tier:", self._tier_combo)

        self._format_combo = QComboBox()
        for title, _ in FORMAT_CHOICES:
            self._format_combo.addItem(title)
        form.addRow("Format:", self._format_combo)

        self._count_spin = QSpinBox()
        self._count_spin.setRange(1, 100)
        self._count_spin.setValue(10)
        form.addRow("Items:", self._count_spin)

        self._options_spin = QSpinBox()
        self._options_spin.setRange(2, 8)
        self._options_spin.setValue(5)
        self._options_spin.setToolTip(
            "5 is the USMLE single-best-answer convention. An item whose data "
            "cannot supply this many options is asked with fewer rather than "
            "padded with an unrelated structure.")
        form.addRow("Options per item:", self._options_spin)

        self._seconds_spin = QSpinBox()
        self._seconds_spin.setRange(10, 600)
        self._seconds_spin.setValue(int(DEFAULT_STATION_SECONDS))
        self._seconds_spin.setSuffix(" s")
        form.addRow("Seconds per station:", self._seconds_spin)

        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 99999)
        self._seed_spin.setToolTip(
            "Item and distractor selection is deterministic: the same seed "
            "reproduces the same paper.")
        form.addRow("Seed:", self._seed_spin)
        layout.addLayout(form)

        self._setup_status = QLabel("")
        self._setup_status.setWordWrap(True)
        layout.addWidget(self._setup_status)
        layout.addStretch()

        buttons = QHBoxLayout()
        self._start_button = QPushButton("Begin")
        self._start_button.clicked.connect(self.begin)
        buttons.addStretch()
        buttons.addWidget(self._start_button)
        layout.addLayout(buttons)
        return page

    def _populate_curricula(self) -> None:
        self._curriculum_combo.clear()
        self._curriculum_keys = []
        try:
            curricula = self._session.curricula or {}
            keys = sorted(curricula, key=lambda k: (-len(curricula[k]), k))
        except Exception:                          # noqa: BLE001 - diagnostic
            logger.exception("Curricula unavailable for the exam dialog")
            return
        for key in keys:
            self._curriculum_keys.append(key)
            cur = curricula[key]
            self._curriculum_combo.addItem(f"{cur.title} ({len(cur)})")

    # -- paper page --------------------------------------------------------

    def _build_paper_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        header = QHBoxLayout()
        self._progress_label = QLabel("")
        self._clock_label = QLabel("")
        header.addWidget(self._progress_label)
        header.addStretch()
        header.addWidget(self._clock_label)
        layout.addLayout(header)

        self._clock_bar = QProgressBar()
        self._clock_bar.setTextVisible(False)
        self._clock_bar.setVisible(False)
        layout.addWidget(self._clock_bar)

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setVisible(False)
        self._image_label.setMinimumHeight(220)
        layout.addWidget(self._image_label)

        self._stem_label = QLabel("")
        self._stem_label.setWordWrap(True)
        self._stem_label.setStyleSheet("font-size: 15px; padding: 6px;")
        layout.addWidget(self._stem_label)

        self._options_box = QGroupBox("")
        self._options_layout = QVBoxLayout(self._options_box)
        self._option_group = QButtonGroup(self)
        layout.addWidget(self._options_box)

        self._feedback_label = QLabel("")
        self._feedback_label.setWordWrap(True)
        self._feedback_label.setStyleSheet("padding: 6px;")
        layout.addWidget(self._feedback_label)
        layout.addStretch()

        buttons = QHBoxLayout()
        self._submit_button = QPushButton("Submit")
        self._submit_button.clicked.connect(self.submit)
        self._skip_button = QPushButton("Skip")
        self._skip_button.clicked.connect(lambda: self.submit(skip=True))
        self._next_button = QPushButton("Next")
        self._next_button.clicked.connect(self.advance)
        self._next_button.setEnabled(False)
        buttons.addWidget(self._skip_button)
        buttons.addStretch()
        buttons.addWidget(self._submit_button)
        buttons.addWidget(self._next_button)
        layout.addLayout(buttons)
        return page

    # -- results page ------------------------------------------------------

    def _build_results_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self._results_label = QLabel("")
        self._results_label.setWordWrap(True)
        self._results_label.setStyleSheet("font-size: 15px; padding: 8px;")
        layout.addWidget(self._results_label)

        self._audit_view = QPlainTextEdit()
        self._audit_view.setReadOnly(True)
        self._audit_view.setVisible(False)
        layout.addWidget(self._audit_view)

        buttons = QHBoxLayout()
        self._audit_button = QPushButton("Show provenance")
        self._audit_button.setCheckable(True)
        self._audit_button.toggled.connect(self._on_audit_toggled)
        again = QPushButton("New examination")
        again.clicked.connect(lambda: self._pages.setCurrentIndex(0))
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        buttons.addWidget(self._audit_button)
        buttons.addStretch()
        buttons.addWidget(again)
        buttons.addWidget(close)
        layout.addLayout(buttons)
        return page

    def _on_audit_toggled(self, checked: bool) -> None:
        if checked:
            self._audit_view.setPlainText(self._session.audit())
        self._audit_view.setVisible(checked)
        self._audit_button.setText(
            "Hide provenance" if checked else "Show provenance")

    # -- wiring ------------------------------------------------------------

    def set_radiology_source(self, source: Optional[Callable[[], Sequence[tuple]]]
                             ) -> None:
        """Supply the loaded scene, enabling L4.

        ``source`` returns ``[(mesh, world_matrix, mesh_id), ...]``.  Passing
        None disables L4 again (a scene can be unloaded).
        """
        self._radiology_source = source
        box = self._level_boxes.get("L4")
        if box is None:
            return
        box.setEnabled(source is not None)
        if source is None:
            box.setChecked(False)
            box.setToolTip(f"Source: {LEVEL_SOURCES['L4']}\n{NEEDS_SCENE['L4']}")
        else:
            box.setToolTip(f"Source: {LEVEL_SOURCES['L4']}")

    def selected_levels(self) -> tuple[str, ...]:
        return tuple(level for level, box in self._level_boxes.items()
                     if box.isChecked() and box.isEnabled())

    def build_config(self) -> ExamConfig:
        """The setup page's state as an :class:`ExamConfig`."""
        index = self._curriculum_combo.currentIndex()
        curriculum = self._curriculum_keys[index] \
            if 0 <= index < len(self._curriculum_keys) else ""
        fmt = FORMAT_CHOICES[self._format_combo.currentIndex()][1]
        return ExamConfig(
            levels=self.selected_levels() or ("L1", ),
            curriculum=curriculum,
            count=self._count_spin.value(),
            options=self._options_spin.value(),
            fmt=fmt,
            seed=self._seed_spin.value(),
            seconds_per_item=(float(self._seconds_spin.value())
                              if fmt == "station" else 0.0),
        )

    # -- running -----------------------------------------------------------

    def begin(self) -> bool:
        """Assemble and start the paper.  False if nothing could be built."""
        config = self.build_config()
        levels = list(config.levels)
        radiology_levels = [lv for lv in levels if lv == "L4"]
        other_levels = tuple(lv for lv in levels if lv != "L4")

        self._radiology_items = {}
        items = []
        if other_levels:
            items = list(self._session.build(
                ExamConfig(**{**config.__dict__, "levels": other_levels})))
        if radiology_levels and self._radiology_source is not None:
            items.extend(self._build_radiology(config))

        self._session.items = items[:config.count]
        if not self._session.items:
            refused = len(self._session.refused)
            self._setup_status.setText(
                "No items could be built for that selection. "
                f"{refused} item(s) were refused for missing provenance."
                if refused else
                "No items could be built for that selection: the data does "
                "not support these levels for this curriculum.")
            return False

        self._setup_status.setText("")
        self._session.start()
        self._pages.setCurrentIndex(1)
        self._show_current()
        return True

    def _build_radiology(self, config: ExamConfig) -> list:
        """L4 items over the live scene, if the source yields one."""
        try:
            scene = list(self._radiology_source() or ())
        except Exception:                          # noqa: BLE001 - diagnostic
            logger.exception("Radiology scene source failed")
            return []
        if not scene:
            return []
        try:
            from faceforge.anatomy.radiology_items import RadiologyItemBuilder
            from faceforge.scanner.engine import ScannerEngine
            from faceforge.scanner.tissue_map import TissueMapper
            builder = RadiologyItemBuilder(
                lambda: ScannerEngine(TissueMapper()),
                generator=self._session.generator)
            focus = self._session.focus_order(config) or \
                [entry[2] for entry in scene]
            built = builder.build_many(
                scene, focus, count=config.count, options=config.options,
                seed=config.seed, resolution=192,
                seconds=config.seconds_per_item)
        except Exception:                          # noqa: BLE001 - diagnostic
            logger.exception("Could not build radiology items")
            return []
        for result in built:
            self._radiology_items[result.item.uid] = result
        return [result.item for result in built]

    def _show_current(self) -> None:
        item = self._session.current
        if item is None:
            self._finish()
            return
        self._answered = False
        self._feedback_label.setText("")
        self._submit_button.setEnabled(True)
        self._skip_button.setEnabled(True)
        self._next_button.setEnabled(False)

        total = len(self._session.items)
        index = total - self._session.remaining
        self._progress_label.setText(
            f"Item {index} of {total} \u2014 {item.level} "
            f"({LEVEL_TITLES.get(item.level, '')})")
        self._stem_label.setText(item.stem)

        for button in self._option_buttons:
            self._option_group.removeButton(button)
            button.setParent(None)
        self._option_buttons = []
        for position, option in enumerate(item.options):
            button = QRadioButton(f"{chr(ord('A') + position)}. {option.text}")
            self._option_group.addButton(button, position)
            self._options_layout.addWidget(button)
            self._option_buttons.append(button)

        self._show_stimulus(item)

        if item.seconds > 0:
            self._clock_bar.setVisible(True)
            self._clock_bar.setRange(0, int(item.seconds * 4))
            self._timer.start()
            self._on_tick()
        else:
            self._clock_bar.setVisible(False)
            self._clock_label.setText("")
            self._timer.stop()

    def _show_stimulus(self, item) -> None:
        """Show the slice for an L4 item; hide the image otherwise."""
        result = self._radiology_items.get(item.uid)
        if result is None:
            self._image_label.setVisible(False)
            return
        pixmap = _scan_pixmap(result.image, result.tag_px)
        self._image_label.setPixmap(pixmap)
        self._image_label.setVisible(True)

    def _on_tick(self) -> None:
        left = self._session.time_left()
        if left is None:
            self._timer.stop()
            return
        self._clock_label.setText(f"{left:.0f} s remaining")
        self._clock_bar.setValue(int(left * 4))
        if left <= 0 and not self._answered:
            self._timer.stop()
            self.submit(skip=True)

    def submit(self, skip: bool = False) -> None:
        """Record the selected option (or a skip) and show the feedback."""
        if self._answered or self._session.current is None:
            return
        checked = self._option_group.checkedId()
        index = None if (skip or checked < 0) else checked
        outcome = self._session.answer(index)
        self._answered = True
        self._timer.stop()
        self._submit_button.setEnabled(False)
        self._skip_button.setEnabled(False)
        self._next_button.setEnabled(True)
        if outcome is None:
            return
        if outcome.correct:
            self._feedback_label.setText("Correct.")
            self._feedback_label.setStyleSheet(
                "padding: 6px; color: #66cc66;")
        else:
            prefix = "Out of time. " if outcome.expired else \
                ("Skipped. " if outcome.skipped else "Incorrect. ")
            answer = outcome.item.answer
            detail = outcome.explanation or (
                f"The answer is {answer.text}." if answer else "")
            self._feedback_label.setText(prefix + detail)
            self._feedback_label.setStyleSheet(
                "padding: 6px; color: #ffaa55;")

    def advance(self) -> None:
        if not self._answered:
            self.submit(skip=True)
        if self._session.next_item() is None:
            self._finish()
        else:
            self._show_current()

    def _finish(self) -> None:
        self._timer.stop()
        correct, answered = self._session.finish()
        pct = f"{100.0 * correct / answered:.0f}%" if answered else "n/a"
        refused = len(self._session.refused)
        lines = [
            f"Examination complete.",
            f"Score: {correct}/{answered} ({pct})",
        ]
        if refused:
            lines.append(f"{refused} item(s) were refused for lacking "
                         f"provenance and were not asked.")
        store = self._session.progress
        if store is not None:
            try:
                summary = store.summary()
                lines.append(
                    f"All time: {summary.correct}/{summary.attempts} correct "
                    f"over {summary.items_seen} structures; "
                    f"{summary.items_due} due for review.")
            except Exception:                      # noqa: BLE001 - diagnostic
                logger.exception("Could not summarise progress")
        self._results_label.setText("\n".join(lines))
        self._audit_button.setChecked(False)
        self._pages.setCurrentIndex(2)


def _scan_pixmap(image: np.ndarray, tag_px: tuple[int, int],
                 size: int = 320) -> QPixmap:
    """Grayscale scan as a pixmap with a crosshair on the tagged pixel.

    The tag is drawn as an open circle with a gap at the centre, not a filled
    marker: a marker that covers the structure it points at would hide the
    evidence the candidate is being asked to read.
    """
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 3:
        array = array.mean(axis=2)
    peak = float(array.max())
    scaled = (array / peak * 255.0) if peak > 0 else array
    buffer = np.ascontiguousarray(scaled.clip(0, 255).astype(np.uint8))
    height, width = buffer.shape
    qimage = QImage(buffer.data, width, height, width,
                    QImage.Format_Grayscale8).copy()
    pixmap = QPixmap.fromImage(qimage).scaled(
        size, size, Qt.KeepAspectRatio, Qt.FastTransformation)

    scale = pixmap.width() / max(width, 1)
    x, y = tag_px[0] * scale, tag_px[1] * scale
    painter = QPainter(pixmap)
    painter.setPen(QPen(Qt.yellow, 2))
    radius = max(6.0, pixmap.width() * 0.035)
    painter.drawEllipse(int(x - radius), int(y - radius),
                        int(radius * 2), int(radius * 2))
    painter.drawLine(int(x - radius * 2), int(y), int(x - radius), int(y))
    painter.drawLine(int(x + radius), int(y), int(x + radius * 2), int(y))
    painter.end()
    return pixmap
