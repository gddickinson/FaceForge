"""EventBus for decoupled publish/subscribe communication."""

from enum import Enum, auto
from typing import Any, Callable
from collections import defaultdict


class EventType(Enum):
    # State changes
    AU_CHANGED = auto()
    HEAD_ROTATION_CHANGED = auto()
    BODY_STATE_CHANGED = auto()
    EXPRESSION_SET = auto()
    BODY_POSE_SET = auto()

    # Loading events
    LOADING_STARTED = auto()
    LOADING_PROGRESS = auto()
    LOADING_PHASE = auto()
    LOADING_COMPLETE = auto()

    # Layer visibility
    LAYER_TOGGLED = auto()

    # Render mode
    RENDER_MODE_CHANGED = auto()
    COLOR_CHANGED = auto()

    # Camera
    CAMERA_PRESET = auto()

    # Alignment
    ALIGNMENT_CHANGED = auto()

    # Animation toggles
    AUTO_BLINK_TOGGLED = auto()
    AUTO_BREATHING_TOGGLED = auto()
    EYE_TRACKING_TOGGLED = auto()
    MICRO_EXPRESSIONS_TOGGLED = auto()

    # Frame events
    FRAME_UPDATE = auto()

    # Structure loading / labels
    STRUCTURES_LOADED = auto()
    LABELS_TOGGLED = auto()

    # Skull mode
    SKULL_MODE_CHANGED = auto()

    # Eye color
    EYE_COLOR_SET = auto()

    # Scene
    SCENE_READY = auto()

    # Debug visualization
    DEBUG_VIZ_CHANGED = auto()
    SELECTION_CHANGED = auto()


class EventBus:
    """Simple publish/subscribe event system."""

    def __init__(self):
        self._handlers: dict[EventType, list[Callable]] = defaultdict(list)

    def subscribe(self, event_type: EventType, handler: Callable) -> None:
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: Callable) -> None:
        handlers = self._handlers[event_type]
        if handler in handlers:
            handlers.remove(handler)

    def publish(self, event_type: EventType, **data: Any) -> None:
        for handler in self._handlers[event_type]:
            handler(**data)

    def clear(self) -> None:
        self._handlers.clear()
