"""Left info panel showing active AUs and expression name."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QProgressBar, QSizePolicy,
)
from PySide6.QtCore import Qt

from faceforge.core.state import StateManager, AU_IDS
from faceforge.ui.widgets.section_label import SectionLabel


class _AUBar(QWidget):
    """Single AU indicator row: ID label + progress bar + value label."""

    def __init__(self, au_id: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._au_id = au_id

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        layout.setSpacing(1)

        # Top line: AU id + value
        top = QWidget()
        top_layout = __import__("PySide6.QtWidgets", fromlist=["QHBoxLayout"]).QHBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(4)

        self._id_label = QLabel(au_id)
        self._id_label.setObjectName("sliderLabel")
        self._id_label.setFixedWidth(40)
        top_layout.addWidget(self._id_label)

        self._value_label = QLabel("0.00")
        self._value_label.setObjectName("valueLabel")
        self._value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        top_layout.addWidget(self._value_label)

        layout.addWidget(top)

        # Progress bar
        self._bar = QProgressBar()
        self._bar.setObjectName("auBar")
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(4)
        layout.addWidget(self._bar)

    def set_value(self, val: float) -> None:
        clamped = max(0.0, min(1.0, val))
        self._bar.setValue(int(clamped * 100))
        self._value_label.setText(f"{clamped:.2f}")
        self.setVisible(clamped > 0.005)


class InfoPanel(QWidget):
    """Left-side info panel showing the current expression name and active AU values.

    Fixed width ~220px, updates per frame via :meth:`update_display`.
    """

    PANEL_WIDTH = 220

    def __init__(
        self,
        state: StateManager,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self.setObjectName("infoPanel")
        self.setFixedWidth(self.PANEL_WIDTH)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        # ── Expression Name ──
        layout.addWidget(SectionLabel("Expression"))
        self._expr_label = QLabel("neutral")
        self._expr_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._expr_label.setStyleSheet(
            "font-size: 16px; font-weight: 700; color: #4fd1c5; padding: 4px 0;"
        )
        layout.addWidget(self._expr_label)

        # ── Active AUs ──
        layout.addWidget(SectionLabel("Active AUs"))
        self._au_bars: dict[str, _AUBar] = {}
        for au_id in AU_IDS:
            bar = _AUBar(au_id)
            bar.setVisible(False)
            self._au_bars[au_id] = bar
            layout.addWidget(bar)

        # ── Head Rotation ──
        layout.addWidget(SectionLabel("Head"))
        self._head_label = QLabel("Y 0.0  P 0.0  R 0.0")
        self._head_label.setObjectName("statsLabel")
        self._head_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._head_label)

        layout.addStretch()

    # ── Public API ──

    def update_display(self) -> None:
        """Refresh the panel to reflect current state. Call once per frame."""
        face = self._state.face

        # Expression name
        self._expr_label.setText(face.current_expression.capitalize())

        # AU bars
        for au_id, bar in self._au_bars.items():
            bar.set_value(face.get_au(au_id))

        # Head rotation
        self._head_label.setText(
            f"Y {face.head_yaw:+.2f}  P {face.head_pitch:+.2f}  R {face.head_roll:+.2f}"
        )
