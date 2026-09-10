"""A list of muscle groups with a role tag and a live activation bar each."""

from __future__ import annotations

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar

from faceforge.body.muscle_activation import level_band
from faceforge.exercise.muscle_groups import group_label

_ROLE_COLORS = {"primary": "#e05a4f", "secondary": "#e0a44f", "stabiliser": "#5aa9e0"}
_ROLE_SHORT = {"primary": "P", "secondary": "S", "stabiliser": "St"}


class _GroupRow(QWidget):
    def __init__(self, key: str, role: str, note: str, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        layout.setSpacing(1)
        top = QWidget()
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(4)
        group, _, side = key.partition(":")
        tag = QLabel(_ROLE_SHORT.get(role, "?"))
        tag.setFixedWidth(18)
        tag.setStyleSheet(f"color: {_ROLE_COLORS.get(role, '#ccc')}; font-weight: 700;")
        tag.setToolTip(role)
        top_layout.addWidget(tag)
        name = group_label(group) + (f" ({side})" if side else "")
        self._name = QLabel(name)
        self._name.setObjectName("sliderLabel")
        if note:
            self._name.setToolTip(note)
        top_layout.addWidget(self._name, 1)
        self._value = QLabel("0 %")
        self._value.setObjectName("valueLabel")
        self._value.setFixedWidth(64)
        top_layout.addWidget(self._value)
        layout.addWidget(top)
        self._bar = QProgressBar()
        self._bar.setObjectName("auBar")
        self._bar.setRange(0, 100)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(4)
        layout.addWidget(self._bar)

    def set_level(self, level: float) -> None:
        pct = int(max(0.0, min(1.0, level)) * 100)
        self._bar.setValue(pct)
        self._value.setText(f"{pct} % {level_band(level)[:4]}")


class MuscleActivationList(QWidget):
    """Rows are rebuilt per exercise; levels are updated per status event."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self._rows: dict[str, _GroupRow] = {}

    def set_muscles(self, uses) -> None:
        """``uses``: iterable of MuscleUse, in display order (primary first)."""
        for row in self._rows.values():
            self._layout.removeWidget(row)
            row.deleteLater()
        self._rows = {}
        order = {"primary": 0, "secondary": 1, "stabiliser": 2}
        for use in sorted(uses, key=lambda u: order.get(u.role.value, 3)):
            key = use.group if use.side is None else f"{use.group}:{use.side}"
            if key in self._rows:
                continue
            row = _GroupRow(key, use.role.value, use.note)
            self._rows[key] = row
            self._layout.addWidget(row)

    def set_levels(self, levels: dict[str, float]) -> None:
        for key, row in self._rows.items():
            row.set_level(float(levels.get(key, 0.0)))

    def clear(self) -> None:
        self.set_muscles(())
