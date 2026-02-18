"""Right control panel with QTabWidget containing 6 tabs."""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QTabWidget
from PySide6.QtCore import Qt

from faceforge.core.events import EventBus
from faceforge.core.state import StateManager
from faceforge.ui.tabs.animate_tab import AnimateTab
from faceforge.ui.tabs.body_tab import BodyTab
from faceforge.ui.tabs.layers_tab import LayersTab
from faceforge.ui.tabs.align_tab import AlignTab
from faceforge.ui.tabs.display_tab import DisplayTab
from faceforge.ui.tabs.debug_tab import DebugTab


class ControlPanel(QWidget):
    """Right-side control panel with tabbed interface.

    Contains 6 tabs: Animate, Body, Layers, Align, Display, Debug.
    Width ~330px, matching the HTML version's right panel.
    """

    def __init__(self, event_bus: EventBus, state: StateManager, parent=None):
        super().__init__(parent)
        self.event_bus = event_bus
        self.state = state

        self.setFixedWidth(330)
        self.setStyleSheet("background: rgba(18, 20, 26, 0.92);")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        # Create tabs
        self.animate_tab = AnimateTab(event_bus, state)
        self.body_tab = BodyTab(event_bus, state)
        self.layers_tab = LayersTab(event_bus, state)
        self.align_tab = AlignTab(event_bus, state)
        self.display_tab = DisplayTab(event_bus, state)
        self.debug_tab = DebugTab(event_bus, state)

        self.tabs.addTab(self.animate_tab, "ANIMATE")
        self.tabs.addTab(self.body_tab, "BODY")
        self.tabs.addTab(self.layers_tab, "LAYERS")
        self.tabs.addTab(self.align_tab, "ALIGN")
        self.tabs.addTab(self.display_tab, "DISPLAY")
        self.tabs.addTab(self.debug_tab, "DEBUG")

        layout.addWidget(self.tabs)
