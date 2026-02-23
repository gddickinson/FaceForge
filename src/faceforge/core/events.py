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

    # Scene view mode
    SCENE_MODE_TOGGLED = auto()
    SCENE_CAMERA_CHANGED = auto()
    SCENE_WRAPPER_NUDGE = auto()  # data: axis (str), delta (float)

    # Animation playback
    ANIM_PLAY = auto()
    ANIM_PAUSE = auto()
    ANIM_STOP = auto()
    ANIM_SEEK = auto()            # data: position (0-1)
    ANIM_SPEED = auto()           # data: speed (float)
    ANIM_CLIP_SELECTED = auto()   # data: clip_name (str)
    ANIM_PROGRESS = auto()        # data: progress (0-1), time (float), duration (float)

    # Clip plane
    CLIP_PLANE_CHANGED = auto()   # data: enabled (bool), axis (str), offset (float), flip (bool)

    # Muscle activation heatmap
    HEATMAP_TOGGLED = auto()      # data: enabled (bool)

    # Structure search
    STRUCTURE_SEARCH = auto()     # data: query (str)

    # Physiology controls
    PHYSIOLOGY_CHANGED = auto()   # data: field (str), value (any)

    # Speech animation
    SPEECH_PLAY = auto()          # data: text (str), speed (float)

    # Anatomy quiz
    QUIZ_START = auto()           # data: mode (str), category (str), count (int)
    QUIZ_ANSWER = auto()          # data: answer (str)

    # Pathology visualization
    PATHOLOGY_CHANGED = auto()    # data: condition (str), target (str), severity (float), enabled (bool)

    # Comparative anatomy views
    COMPARISON_MODE_CHANGED = auto()  # data: enabled (bool), left_config (dict), right_config (dict)

    # Gender dimorphism
    GENDER_CHANGED = auto()           # data: gender (float, 0=male, 1=female)
    GENDER_RELEASED = auto()          # data: gender (float) — slider released, trigger re-registration


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
