"""Semi-transparent loading overlay with progress bar."""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar, QSizePolicy
from PySide6.QtCore import Qt


class LoadingOverlay(QWidget):
    """A dark semi-transparent overlay showing a phase label and progress bar.

    Intended to be placed over the GL viewport while meshes are loading.
    Call :meth:`show_loading` to update phase text and progress (0.0--1.0),
    and :meth:`hide_overlay` when loading is complete.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("loadingOverlay")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAutoFillBackground(True)

        # Semi-transparent background via stylesheet
        self.setStyleSheet("background-color: rgba(10, 11, 14, 200);")

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(12)

        # Phase label
        self._phase_label = QLabel("Loading...")
        self._phase_label.setObjectName("loadingPhaseLabel")
        self._phase_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._phase_label)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setRange(0, 1000)
        self._progress.setValue(0)
        self._progress.setFixedWidth(300)
        self._progress.setFixedHeight(12)
        self._progress.setTextVisible(False)
        layout.addWidget(self._progress, alignment=Qt.AlignmentFlag.AlignCenter)

        # Percent label
        self._percent_label = QLabel("0%")
        self._percent_label.setObjectName("valueLabel")
        self._percent_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._percent_label)

        self.hide()

    # ── Public API ──

    def show_loading(self, phase: str, progress: float) -> None:
        """Update the overlay content and make it visible.

        Parameters
        ----------
        phase : str
            Human-readable description of the current loading phase.
        progress : float
            Value between 0.0 and 1.0 indicating overall progress.
        """
        clamped = max(0.0, min(1.0, progress))
        self._phase_label.setText(phase)
        self._progress.setValue(int(clamped * 1000))
        self._percent_label.setText(f"{int(clamped * 100)}%")
        if not self.isVisible():
            self.show()
            self.raise_()

    def hide_overlay(self) -> None:
        """Hide the loading overlay."""
        self.hide()

    # ── Geometry ──

    def resizeEvent(self, event) -> None:
        """Fill the parent widget."""
        super().resizeEvent(event)
        if self.parent():
            self.setGeometry(self.parent().rect())
