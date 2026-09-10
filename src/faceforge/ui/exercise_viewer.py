"""The exercise viewer: the whole body performing a demonstration, from any angle.

A *mode* of the main window rather than a second window.  A QOpenGLWidget can
live in one place only, so instead of a second viewport the right-hand
control panel is swapped for this panel, which adopts the control panel's
``ExerciseTab`` -- the same instance, so the transport, status and progress
plumbing that the controllers already drive keep working -- and adds what a
viewer needs: camera views all round the gym, whole-body muscle display and a
way back.  ``python -m faceforge.exercise_viewer`` opens the application
straight into this mode.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGridLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from faceforge.core.events import EventBus, EventType
from faceforge.ui.widgets.section_label import SectionLabel
from faceforge.ui.widgets.toggle_row import ToggleRow

#: (label, gym camera preset) -- see scene_mode_controller._GYM_CAMERA_PRESETS.
VIEWS: tuple[tuple[str, str], ...] = (
    ("Front", "front"), ("Three-quarter", "three_quarter"),
    ("Right side", "side"), ("Left side", "side_left"),
    ("Back", "back"), ("Back quarter", "back_quarter"),
    ("Low front", "low_front"), ("Overhead", "overhead"),
)

PANEL_WIDTH = 380


class ExerciseViewerPanel(QWidget):
    """Camera views, whole-body display toggles and the adopted exercise tab.

    Publishes ``SCENE_CAMERA_CHANGED`` (a view button), ``EXERCISE_OPTION_CHANGED``
    (``all_muscles``) and ``LAYER_TOGGLED`` (skin); emits ``exit_requested``
    when the user wants the full interface back.
    """

    exit_requested = Signal()

    def __init__(self, event_bus: EventBus, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._bus = event_bus
        self._tab: QWidget | None = None
        self.setFixedWidth(PANEL_WIDTH)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(4)

        title = QLabel("Exercise viewer")
        title.setStyleSheet("font-size: 16px; font-weight: 700; color: #4fd1c5;")
        layout.addWidget(title)
        hint = QLabel("Drag to orbit · scroll to zoom · right-drag to pan")
        hint.setObjectName("statsLabel")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        layout.addWidget(SectionLabel("View"))
        grid = QGridLayout()
        grid.setSpacing(4)
        self._view_buttons: dict[str, QPushButton] = {}
        for i, (label, preset) in enumerate(VIEWS):
            btn = QPushButton(label)
            btn.setObjectName("presetButton")
            btn.clicked.connect(lambda _checked=False, p=preset: self.select_view(p))
            grid.addWidget(btn, i // 2, i % 2)
            self._view_buttons[preset] = btn
        gw = QWidget()
        gw.setLayout(grid)
        layout.addWidget(gw)

        layout.addWidget(SectionLabel("Display"))
        self._all_muscles = ToggleRow("Show every muscle and the skeleton", default=True)
        self._all_muscles.toggled.connect(
            lambda v: self._bus.publish(EventType.EXERCISE_OPTION_CHANGED,
                                        option="all_muscles", value=bool(v)))
        layout.addWidget(self._all_muscles)
        self._skin = ToggleRow("Show skin", default=False)
        self._skin.toggled.connect(
            lambda v: self._bus.publish(EventType.LAYER_TOGGLED, layer="skin", visible=bool(v)))
        layout.addWidget(self._skin)

        self._tab_slot = QVBoxLayout()
        self._tab_slot.setContentsMargins(0, 0, 0, 0)
        slot = QWidget()
        slot.setLayout(self._tab_slot)
        slot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(slot, 1)

        back = QPushButton("Back to the full interface")
        back.setObjectName("resetButton")
        back.clicked.connect(self.exit_requested.emit)
        layout.addWidget(back)

    # -- the adopted tab --------------------------------------------------------------

    def adopt_tab(self, tab: QWidget) -> None:
        """Take the control panel's exercise tab into this panel."""
        if self._tab is not None:
            self.release_tab()
        self._tab = tab
        self._tab_slot.addWidget(tab)
        tab.show()

    def release_tab(self) -> QWidget | None:
        """Hand the exercise tab back (unparented) for the control panel to re-insert."""
        tab = self._tab
        if tab is None:
            return None
        self._tab_slot.removeWidget(tab)
        tab.setParent(None)
        self._tab = None
        return tab

    @property
    def tab(self) -> QWidget | None:
        return self._tab

    # -- actions -----------------------------------------------------------------------

    def select_view(self, preset: str) -> None:
        self._bus.publish(EventType.SCENE_CAMERA_CHANGED, preset=preset)

    def request_all_muscles(self) -> None:
        """Re-publish the whole-body option (used when the mode is entered)."""
        self._bus.publish(EventType.EXERCISE_OPTION_CHANGED, option="all_muscles",
                          value=bool(self._all_muscles.is_checked))

    @property
    def all_muscles(self) -> bool:
        return bool(self._all_muscles.is_checked)

    @property
    def view_names(self) -> tuple[str, ...]:
        return tuple(self._view_buttons)
