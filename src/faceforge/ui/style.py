"""QSS dark theme stylesheet matching the HTML version's design."""

# Color palette from the HTML CSS
COLORS = {
    "bg": "#0a0b0e",
    "surface": "#12141a",
    "surface2": "#1a1d26",
    "border": "#252830",
    "text": "#e8e9ed",
    "text_dim": "#8b8e99",
    "accent": "#4fd1c5",
    "accent2": "#f6ad55",
    "accent_hover": "#38b2ac",
    "accent_pressed": "#319795",
    "danger": "#e53e3e",
    "success": "#48bb78",
}

DARK_THEME = """
/* ── Global ── */
QMainWindow, QWidget {
    background-color: %(bg)s;
    color: %(text)s;
    font-family: -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: 12px;
}

/* ── QLabel ── */
QLabel {
    color: %(text)s;
    background: transparent;
    padding: 0px;
}

QLabel[dimmed="true"] {
    color: %(text_dim)s;
}

/* ── Section Label ── */
QLabel#sectionLabel {
    color: %(accent)s;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 4px 0px 2px 0px;
    border-bottom: 1px solid %(accent)s;
    margin-top: 8px;
    margin-bottom: 4px;
}

/* ── QPushButton ── */
QPushButton {
    background-color: %(surface2)s;
    color: %(text)s;
    border: 1px solid %(border)s;
    border-radius: 4px;
    padding: 5px 12px;
    font-size: 11px;
    min-height: 22px;
}

QPushButton:hover {
    background-color: %(border)s;
    border-color: %(accent)s;
}

QPushButton:pressed {
    background-color: %(accent_pressed)s;
    color: %(bg)s;
}

QPushButton[active="true"], QPushButton:checked {
    background-color: %(accent)s;
    color: %(bg)s;
    border-color: %(accent)s;
    font-weight: 600;
}

QPushButton#expressionButton {
    font-size: 10px;
    padding: 4px 8px;
    min-height: 20px;
    text-transform: capitalize;
}

QPushButton#poseButton {
    font-size: 10px;
    padding: 4px 8px;
    min-height: 20px;
    text-transform: capitalize;
}

QPushButton#renderModeButton {
    font-size: 10px;
    padding: 4px 10px;
    min-height: 24px;
}

QPushButton#cameraPresetButton {
    font-size: 10px;
    padding: 3px 8px;
    min-height: 20px;
}

QPushButton#resetButton {
    background-color: %(surface)s;
    color: %(accent2)s;
    border-color: %(accent2)s;
}

QPushButton#resetButton:hover {
    background-color: %(accent2)s;
    color: %(bg)s;
}

QPushButton#colorButton {
    border-radius: 3px;
    min-width: 28px;
    max-width: 28px;
    min-height: 22px;
    max-height: 22px;
    padding: 0px;
}

/* ── QSlider ── */
QSlider::groove:horizontal {
    height: 4px;
    background: %(surface2)s;
    border-radius: 2px;
    border: 1px solid %(border)s;
}

QSlider::handle:horizontal {
    background: %(accent)s;
    width: 12px;
    height: 12px;
    margin: -5px 0;
    border-radius: 6px;
    border: none;
}

QSlider::handle:horizontal:hover {
    background: %(accent_hover)s;
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 7px;
}

QSlider::sub-page:horizontal {
    background: %(accent)s;
    border-radius: 2px;
}

/* ── QCheckBox ── */
QCheckBox {
    color: %(text)s;
    spacing: 6px;
    font-size: 11px;
}

QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid %(border)s;
    border-radius: 3px;
    background-color: %(surface2)s;
}

QCheckBox::indicator:checked {
    background-color: %(accent)s;
    border-color: %(accent)s;
}

QCheckBox::indicator:hover {
    border-color: %(accent)s;
}

/* ── QTabWidget ── */
QTabWidget::pane {
    background-color: %(surface)s;
    border: 1px solid %(border)s;
    border-top: none;
}

QTabBar::tab {
    background-color: %(bg)s;
    color: %(text_dim)s;
    border: 1px solid %(border)s;
    border-bottom: none;
    padding: 5px 10px;
    font-size: 10px;
    min-width: 40px;
}

QTabBar::tab:selected {
    background-color: %(surface)s;
    color: %(accent)s;
    border-bottom: 2px solid %(accent)s;
    font-weight: 600;
}

QTabBar::tab:hover:!selected {
    color: %(text)s;
    background-color: %(surface2)s;
}

/* ── QScrollArea ── */
QScrollArea {
    background-color: %(surface)s;
    border: none;
}

QScrollArea > QWidget > QWidget {
    background-color: %(surface)s;
}

QScrollBar:vertical {
    background: %(bg)s;
    width: 8px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background: %(border)s;
    min-height: 30px;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background: %(text_dim)s;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    height: 0px;
}

/* ── QGroupBox ── */
QGroupBox {
    background-color: transparent;
    border: 1px solid %(border)s;
    border-radius: 4px;
    margin-top: 12px;
    padding: 8px 4px 4px 4px;
    font-size: 10px;
    font-weight: 600;
    color: %(text_dim)s;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    color: %(text_dim)s;
}

/* ── QProgressBar ── */
QProgressBar {
    background-color: %(surface2)s;
    border: 1px solid %(border)s;
    border-radius: 4px;
    height: 8px;
    text-align: center;
    font-size: 9px;
    color: %(text_dim)s;
}

QProgressBar::chunk {
    background-color: %(accent)s;
    border-radius: 3px;
}

/* ── QStatusBar ── */
QStatusBar {
    background-color: %(bg)s;
    border-top: 1px solid %(border)s;
    color: %(text_dim)s;
    font-size: 10px;
    font-family: "SF Mono", "Fira Code", "Consolas", monospace;
    padding: 2px 8px;
}

QStatusBar::item {
    border: none;
}

/* ── Info Panel ── */
QWidget#infoPanel {
    background-color: %(surface)s;
    border-right: 1px solid %(border)s;
}

/* ── Control Panel ── */
QWidget#controlPanel {
    background-color: %(surface)s;
    border-left: 1px solid %(border)s;
}

/* ── Value labels (monospace) ── */
QLabel#valueLabel {
    font-family: "SF Mono", "Fira Code", "Consolas", monospace;
    font-size: 10px;
    color: %(accent2)s;
}

QLabel#sliderLabel {
    font-size: 11px;
    color: %(text_dim)s;
}

/* ── Loading Overlay ── */
QWidget#loadingOverlay {
    background-color: rgba(10, 11, 14, 200);
}

QLabel#loadingPhaseLabel {
    color: %(accent)s;
    font-size: 14px;
    font-weight: 600;
}

/* ── Stats display ── */
QLabel#statsLabel {
    font-family: "SF Mono", "Fira Code", "Consolas", monospace;
    font-size: 10px;
    color: %(text_dim)s;
}

/* ── AU bar in info panel ── */
QProgressBar#auBar {
    background-color: %(surface2)s;
    border: none;
    border-radius: 2px;
    height: 4px;
}

QProgressBar#auBar::chunk {
    background-color: %(accent)s;
    border-radius: 2px;
}
""" % COLORS


def apply_theme(app) -> None:
    """Apply the dark theme to a QApplication."""
    app.setStyleSheet(DARK_THEME)
