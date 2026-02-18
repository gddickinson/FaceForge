"""Section label with accent underline."""

from PySide6.QtWidgets import QLabel


class SectionLabel(QLabel):
    """A styled section heading with accent-colored bottom border.

    Uses the ``#sectionLabel`` QSS object name for styling:
    small caps, accent colour, 1px bottom border.
    """

    def __init__(self, text: str, parent=None) -> None:
        super().__init__(text.upper(), parent)
        self.setObjectName("sectionLabel")
