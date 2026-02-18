"""Collapsible section with master toggle and individual item toggles."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QPushButton, QLabel,
)
from PySide6.QtCore import Signal, Qt


class CollapsibleSection(QWidget):
    """A collapsible group with a master checkbox and per-item toggles.

    Layout::

        ▶ [✓] Section Title        (collapsed)
        ▼ [✓] Section Title        (expanded)
           [✓] Item A
           [✓] Item B
           ...

    Signals:
        master_toggled(bool): Emitted when the master checkbox is clicked.
        child_toggled(str, bool): Emitted when an individual item is toggled.
    """

    master_toggled = Signal(bool)
    child_toggled = Signal(str, bool)

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = title
        self._children: dict[str, QCheckBox] = {}
        self._updating_master = False

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Header row: arrow button + master checkbox
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 1, 0, 1)
        header_layout.setSpacing(4)

        self._arrow_btn = QPushButton("▶")
        self._arrow_btn.setFixedSize(16, 16)
        self._arrow_btn.setFlat(True)
        self._arrow_btn.setStyleSheet(
            "QPushButton { color: #8899aa; border: none; font-size: 10px; padding: 0; }"
        )
        self._arrow_btn.clicked.connect(self._toggle_expanded)
        header_layout.addWidget(self._arrow_btn)

        self._master_cb = QCheckBox(title)
        self._master_cb.setTristate(True)
        self._master_cb.setCheckState(Qt.CheckState.Unchecked)
        self._master_cb.clicked.connect(self._on_master_clicked)
        header_layout.addWidget(self._master_cb)
        header_layout.addStretch()

        root_layout.addWidget(header)

        # Content area (hidden by default)
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(1)
        self._content.setVisible(False)
        root_layout.addWidget(self._content)

    def add_item(self, toggle_id: str, label: str, default: bool = True) -> None:
        """Add an individual toggle item to the section."""
        cb = QCheckBox(label)
        cb.setChecked(default)
        cb.toggled.connect(lambda checked, tid=toggle_id: self._on_child_toggled(tid, checked))

        # Indent child items
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(20, 1, 0, 1)
        row_layout.setSpacing(6)
        row_layout.addWidget(cb)
        row_layout.addStretch()

        self._children[toggle_id] = cb
        self._content_layout.addWidget(row)
        self._update_master_state()

    def add_category_header(self, label: str) -> None:
        """Add a non-interactive category sub-header."""
        lbl = QLabel(label.upper())
        lbl.setObjectName("sectionLabel")
        lbl.setStyleSheet(
            "QLabel { color: #667788; font-size: 9px; font-weight: bold; "
            "padding: 4px 0 1px 20px; }"
        )
        self._content_layout.addWidget(lbl)

    def expand(self) -> None:
        self._content.setVisible(True)
        self._arrow_btn.setText("▼")

    def collapse(self) -> None:
        self._content.setVisible(False)
        self._arrow_btn.setText("▶")

    @property
    def is_expanded(self) -> bool:
        return self._content.isVisible()

    def set_master_checked(self, checked: bool) -> None:
        """Set master state without emitting signals (for programmatic use)."""
        self._updating_master = True
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        self._master_cb.setCheckState(state)
        self._updating_master = False

    def set_child_checked(self, toggle_id: str, checked: bool) -> None:
        """Set an individual child toggle state."""
        cb = self._children.get(toggle_id)
        if cb is not None:
            cb.setChecked(checked)

    # ── Internal ──

    def _toggle_expanded(self) -> None:
        if self.is_expanded:
            self.collapse()
        else:
            self.expand()

    def _on_master_clicked(self) -> None:
        if self._updating_master:
            return
        # On click, force to checked or unchecked (not partial)
        checked = self._master_cb.checkState() != Qt.CheckState.Unchecked
        self._updating_master = True
        # Set all children to match (suppress signals during batch update)
        for cb in self._children.values():
            cb.setChecked(checked)
        self._master_cb.setCheckState(
            Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        )
        self._updating_master = False
        # Emit master_toggled first (for group-level visibility)
        self.master_toggled.emit(checked)
        # Then emit child_toggled for each child so individual nodes update
        for toggle_id, cb in self._children.items():
            self.child_toggled.emit(toggle_id, checked)

    def _on_child_toggled(self, toggle_id: str, checked: bool) -> None:
        if not self._updating_master:
            self.child_toggled.emit(toggle_id, checked)
            self._update_master_state()

    def _update_master_state(self) -> None:
        """Recompute master checkbox tri-state from children."""
        if not self._children:
            return
        self._updating_master = True
        checked_count = sum(1 for cb in self._children.values() if cb.isChecked())
        total = len(self._children)
        if checked_count == 0:
            self._master_cb.setCheckState(Qt.CheckState.Unchecked)
        elif checked_count == total:
            self._master_cb.setCheckState(Qt.CheckState.Checked)
        else:
            self._master_cb.setCheckState(Qt.CheckState.PartiallyChecked)
        self._updating_master = False
