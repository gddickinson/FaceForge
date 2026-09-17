"""Exercise tab: pick a demonstration, read the technique, watch the muscles work."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QApplication, QScrollArea, QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel,
    QListWidget, QListWidgetItem, QProgressBar, QPushButton, QSpinBox, QSizePolicy,
)
from PySide6.QtCore import Qt

from faceforge.core.events import EventBus, EventType
from faceforge.core.state import StateManager
from faceforge.exercise.catalog import get_exercise_catalog
from faceforge.exercise.model import Category
from faceforge.exercise.stabilisers import with_implied_stabilisers
from faceforge.ui.widgets.muscle_activation_list import MuscleActivationList
from faceforge.ui.widgets.section_label import SectionLabel
from faceforge.ui.widgets.toggle_row import ToggleRow
from faceforge.ui.widgets.transport_controls import TransportControls

_TEMPOS = [("0.5x (teaching)", 0.5), ("0.75x", 0.75), ("1x", 1.0), ("1.5x", 1.5)]
_PALETTES = [("Classic (blue-red)", "classic"), ("Thermal (colour-safe)", "thermal")]


def _wrap_label(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setObjectName("statsLabel")
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


class ExerciseTab(QScrollArea):
    """Publishes EXERCISE_SELECTED / EXERCISE_STOPPED / EXERCISE_OPTION_CHANGED;
    consumes EXERCISE_STATUS."""

    def __init__(self, event_bus: EventBus, state: StateManager, parent: QWidget | None = None):
        super().__init__(parent)
        self._bus = event_bus
        self._state = state
        self._catalog = get_exercise_catalog()
        self._current_id: str | None = None

        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QScrollArea.Shape.NoFrame)
        container = QWidget()
        self._layout = QVBoxLayout(container)
        self._layout.setContentsMargins(8, 4, 8, 8)
        self._layout.setSpacing(3)
        self.setWidget(container)

        # ── Exercise ──
        self._layout.addWidget(SectionLabel("Exercise"))
        filters = QHBoxLayout()
        self._category = QComboBox()
        self._category.addItem("All categories", None)
        for cat in Category:
            self._category.addItem(cat.value, cat)
        self._category.currentIndexChanged.connect(lambda _: self._refill())
        filters.addWidget(self._category, 1)
        self._equipment = QComboBox()
        self._equipment.addItem("Any equipment", None)
        kinds = sorted({e.kind for d in self._catalog.values() for e in d.equipment})
        self._equipment.addItem("No equipment", "none")
        for kind in kinds:
            self._equipment.addItem(kind.replace("_", " "), kind)
        self._equipment.currentIndexChanged.connect(lambda _: self._refill())
        filters.addWidget(self._equipment, 1)
        fw = QWidget()
        fw.setLayout(filters)
        self._layout.addWidget(fw)

        self._list = QListWidget()
        self._list.setMinimumHeight(150)
        self._list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._list.itemActivated.connect(self._on_item_activated)
        self._list.currentItemChanged.connect(lambda cur, _prev: self._show_definition(cur))
        self._layout.addWidget(self._list)

        buttons = QHBoxLayout()
        self._start_btn = QPushButton("Demonstrate")
        self._start_btn.setObjectName("poseButton")
        self._start_btn.clicked.connect(self._on_start)
        buttons.addWidget(self._start_btn)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setObjectName("resetButton")
        self._stop_btn.clicked.connect(self._on_stop)
        buttons.addWidget(self._stop_btn)
        bw = QWidget()
        bw.setLayout(buttons)
        self._layout.addWidget(bw)

        # ── Options ──
        self._layout.addWidget(SectionLabel("Options"))
        opts = QHBoxLayout()
        opts.addWidget(QLabel("Reps"))
        self._reps = QSpinBox()
        self._reps.setRange(1, 20)
        self._reps.setValue(3)
        self._reps.valueChanged.connect(lambda v: self._option("reps", int(v)))
        opts.addWidget(self._reps)
        opts.addWidget(QLabel("Tempo"))
        self._tempo = QComboBox()
        for label, value in _TEMPOS:
            self._tempo.addItem(label, value)
        self._tempo.setCurrentIndex(2)
        self._tempo.currentIndexChanged.connect(
            lambda _: self._option("tempo", float(self._tempo.currentData())))
        opts.addWidget(self._tempo, 1)
        ow = QWidget()
        ow.setLayout(opts)
        self._layout.addWidget(ow)

        self._heatmap = ToggleRow("Muscle activation heatmap", default=True)
        self._heatmap.toggled.connect(lambda v: self._option("heatmap", bool(v)))
        self._layout.addWidget(self._heatmap)
        self._palette = QComboBox()
        for label, value in _PALETTES:
            self._palette.addItem(label, value)
        self._palette.currentIndexChanged.connect(
            lambda _: self._option("palette", str(self._palette.currentData())))
        self._layout.addWidget(self._palette)
        self._equip_toggle = ToggleRow("Show equipment", default=True)
        self._equip_toggle.toggled.connect(lambda v: self._option("equipment", bool(v)))
        self._layout.addWidget(self._equip_toggle)
        self._load_toggle = ToggleRow("Load the muscles the exercise uses", default=True)
        self._load_toggle.toggled.connect(lambda v: self._option("load_muscles", bool(v)))
        self._layout.addWidget(self._load_toggle)

        # ── Playback ──
        self._layout.addWidget(SectionLabel("Playback"))
        self._transport = TransportControls(event_bus)
        self._layout.addWidget(self._transport)
        self._phase_label = QLabel("—")
        self._phase_label.setStyleSheet("font-size: 14px; font-weight: 700; color: #4fd1c5;")
        self._phase_label.setWordWrap(True)
        self._layout.addWidget(self._phase_label)
        self._cue_label = _wrap_label()
        self._layout.addWidget(self._cue_label)
        self._layout.addWidget(SectionLabel("Moving now"))
        self._motion_label = _wrap_label()
        self._layout.addWidget(self._motion_label)

        # ── Export ──
        # Below Playback, because it exports the frame the transport has
        # stopped on.  What the file contains is decided by the layer toggles:
        # with the skeleton showing a squat is 8.9 M vertices and 1.4 GB, so
        # the count and the size are reported back and the bar is not
        # decoration -- the export takes long enough to look like a hang.
        self._layout.addWidget(SectionLabel("Export"))
        self._obj_btn = QPushButton("Save OBJ && view")
        self._obj_btn.setObjectName("poseButton")
        self._obj_btn.setToolTip(
            "Write the pose currently on screen to an OBJ file and open it in a "
            "separate viewer window.  Everything visible is exported, so hide "
            "the layers you do not want first.")
        self._obj_btn.clicked.connect(self._on_export_obj)
        self._layout.addWidget(self._obj_btn)
        self._obj_progress = QProgressBar()
        self._obj_progress.setRange(0, 100)
        self._obj_progress.setValue(0)
        self._obj_progress.setTextVisible(True)
        self._obj_progress.setFormat("")
        self._obj_progress.setFixedHeight(14)
        self._layout.addWidget(self._obj_progress)
        self._obj_label = _wrap_label()
        self._obj_label.setStyleSheet("font-size: 10px; color: #999;")
        self._layout.addWidget(self._obj_label)

        # ── Muscles ──
        self._layout.addWidget(SectionLabel("Working muscles"))
        self._muscles = MuscleActivationList()
        self._layout.addWidget(self._muscles)

        # ── Technique ──
        self._layout.addWidget(SectionLabel("Technique"))
        self._description = _wrap_label()
        self._layout.addWidget(self._description)
        self._setup = _wrap_label()
        self._layout.addWidget(self._setup)
        self._layout.addWidget(SectionLabel("Common errors"))
        self._errors = _wrap_label()
        self._layout.addWidget(self._errors)
        self._layout.addWidget(SectionLabel("Physio notes"))
        self._notes = _wrap_label()
        self._layout.addWidget(self._notes)
        self._layout.addWidget(SectionLabel("Sources"))
        self._sources = _wrap_label()
        self._sources.setStyleSheet("font-size: 10px; color: #999;")
        self._layout.addWidget(self._sources)
        self._layout.addStretch()

        event_bus.subscribe(EventType.EXERCISE_STATUS, self.on_status)
        event_bus.subscribe(EventType.EXERCISE_EXPORTED, self.on_exported)
        self._refill()

    # ── OBJ export ──

    def _on_export_obj(self) -> None:
        self._obj_btn.setEnabled(False)
        self._obj_progress.setValue(0)
        self._obj_progress.setFormat("starting…")
        self._obj_label.setText("")
        QApplication.processEvents()          # paint the bar before the work
        try:
            self._bus.publish(EventType.EXERCISE_EXPORT_OBJ, on_progress=self.on_progress)
        finally:
            self._obj_btn.setEnabled(True)

    def on_progress(self, done: int, total: int, stage: str) -> None:
        """Called from inside the export, which holds this thread throughout.

        `processEvents` is what actually redraws the bar; without it the whole
        thing is a frozen window for the minute or so a full-skeleton export
        takes.  The controller stops the viewport's refresh timer first, so
        nothing repaints the 3D scene -- and therefore nothing moves the
        geometry -- while these events are pumped.
        """
        pct = int(100 * done / total) if total else 0
        self._obj_progress.setValue(pct)
        self._obj_progress.setFormat(f"{stage} {done}/{total}")
        QApplication.processEvents()

    def on_exported(self, path: str = "", message: str = "", ok: bool = True, **kw) -> None:
        self._obj_label.setText(message)
        self._obj_label.setStyleSheet(
            "font-size: 10px; color: %s;" % ("#999" if ok else "#e06c6c"))
        self._obj_progress.setValue(100 if ok else 0)
        self._obj_progress.setFormat("done" if ok else "failed")
        self._obj_btn.setEnabled(True)

    # ── list ──

    def _refill(self) -> None:
        self._list.clear()
        cat = self._category.currentData()
        kind = self._equipment.currentData()
        for defn in self._catalog.values():
            if cat is not None and defn.category is not cat:
                continue
            if kind == "none" and defn.equipment:
                continue
            if kind not in (None, "none") and kind not in defn.equipment_names:
                continue
            item = QListWidgetItem(defn.name)
            item.setData(Qt.ItemDataRole.UserRole, defn.id)
            item.setToolTip(defn.description)
            self._list.addItem(item)

    def _selected_id(self) -> str | None:
        item = self._list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item is not None else None

    def _show_definition(self, item) -> None:
        exercise_id = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        defn = self._catalog.get(exercise_id or "")
        if defn is None:
            return
        self._description.setText(defn.description)
        self._setup.setText("\n".join(f"• {c}" for c in defn.setup))
        self._errors.setText("\n".join(f"• {c}" for c in defn.errors))
        self._notes.setText("\n".join(f"• {c}" for c in defn.physio_notes))
        self._sources.setText("\n".join(f"• {s}" for s in defn.sources))
        self._muscles.set_muscles(with_implied_stabilisers(defn).muscles)
        self._reps.blockSignals(True)
        self._reps.setValue(defn.default_reps)
        self._reps.blockSignals(False)

    # ── slots ──

    def _on_item_activated(self, _item) -> None:
        self._on_start()

    def _on_start(self) -> None:
        exercise_id = self._selected_id()
        if not exercise_id:
            return
        self._current_id = exercise_id
        self._bus.publish(EventType.EXERCISE_SELECTED, exercise_id=exercise_id,
                          reps=int(self._reps.value()), tempo=float(self._tempo.currentData()))
        self._transport.set_playing(True)

    def _on_stop(self) -> None:
        self._current_id = None
        self._bus.publish(EventType.EXERCISE_STOPPED)
        self._transport.set_playing(False)
        self._phase_label.setText("—")
        self._cue_label.setText("")
        self._motion_label.setText("")

    def _option(self, option: str, value) -> None:
        self._bus.publish(EventType.EXERCISE_OPTION_CHANGED, option=option, value=value)

    # ── status ──

    def on_status(self, exercise_id: str = "", phase: str = "", kind: str = "", cue: str = "",
                  motions=(), levels=None, time: float = 0.0, rep: int = 0, **kw) -> None:
        if not exercise_id:
            return
        self._phase_label.setText(f"Rep {rep} · {phase} ({kind})")
        self._cue_label.setText(cue)
        self._motion_label.setText("\n".join(motions) if motions else "holding")
        self._muscles.set_levels(levels or {})

    def update_progress(self, progress: float, current_time: float, duration: float) -> None:
        self._transport.set_progress(progress, current_time, duration)

    @property
    def transport(self) -> TransportControls:
        return self._transport
