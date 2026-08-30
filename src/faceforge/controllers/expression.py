"""Face: Action Units, expression presets, head rotation, speech.

Every handler here writes to *target* state, never to live state, except where
noted.  The simulation interpolates live state toward the target each frame, so
writing the target is what makes a slider drag look like a face moving rather
than a face teleporting.

The documented exceptions are eye gaze and pupil dilation, which are live
fields the interpolator does not touch.
"""

from __future__ import annotations

import logging
from typing import Any

from faceforge.core.events import EventType

logger = logging.getLogger(__name__)

#: Mouth AUs reset when a speech sequence finishes, so the jaw closes rather
#: than holding the final viseme.
SPEECH_RESET_AUS = ("AU25", "AU26", "AU22", "AU20")


class ExpressionController:
    """Handlers for facial state: AUs, expressions, head pose, speech, eyes."""

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        #: Visemes for the sequence being spoken, or ``[]``.
        self.speech_sequence: list = []
        self.speech_playing = False
        self.speech_time = 0.0

    def subscribe(self) -> None:
        bus = self.ctx.event_bus
        bus.subscribe(EventType.AU_CHANGED, self.on_au_changed)
        bus.subscribe(EventType.EXPRESSION_SET, self.on_expression_set)
        bus.subscribe(EventType.HEAD_ROTATION_CHANGED, self.on_head_changed)

    def subscribe_toggles(self) -> None:
        """Subscribe the auto-behaviour toggles.

        Separate from :meth:`subscribe` only so the assembled subscription
        order matches the original, which interleaved these with the layer and
        body handlers.  Order is observable: several event types have more than
        one subscriber and the bus calls them in subscription order.
        """
        bus = self.ctx.event_bus
        bus.subscribe(EventType.AUTO_BLINK_TOGGLED, self.on_auto_blink)
        bus.subscribe(EventType.AUTO_BREATHING_TOGGLED, self.on_auto_breathing)
        bus.subscribe(EventType.EYE_TRACKING_TOGGLED, self.on_eye_tracking)
        bus.subscribe(EventType.MICRO_EXPRESSIONS_TOGGLED,
                      self.on_micro_expressions)

    # -- Action Units -----------------------------------------------------

    def on_au_changed(self, au_id: str = "", value: float = 0.0, **kw) -> None:
        """Route one slider to its state field.

        Gaze and pupil dilation are live fields the interpolator ignores, so
        they are written directly; everything else is an AU target.
        """
        state = self.ctx.state
        if au_id == "eye_look_x":
            state.face.eye_look_x = value
        elif au_id == "eye_look_y":
            state.face.eye_look_y = value
        elif au_id == "ear_wiggle":
            state.target_ear_wiggle = value
        elif au_id == "pupil_dilation":
            state.face.pupil_dilation = value
        else:
            state.target_au.set(au_id, value)

    def on_expression_set(self, name: str = "", values: dict | None = None,
                          **kw) -> None:
        """Apply a named expression preset.

        Every AU is written, not just the ones the preset names: a preset is a
        complete facial pose, so an AU the preset omits must go to zero rather
        than linger from whatever was set before.  Head rotation keys are
        accepted in both the JS (``headYaw``) and Python (``head_yaw``)
        spellings because presets exist in both config generations.
        """
        values = values or {}
        state = self.ctx.state
        for au_id in state.target_au.to_dict():
            state.target_au.set(au_id, values.get(au_id, 0.0))
        state.target_head.head_yaw = values.get(
            "headYaw", values.get("head_yaw", 0.0))
        state.target_head.head_pitch = values.get(
            "headPitch", values.get("head_pitch", 0.0))
        state.target_head.head_roll = values.get(
            "headRoll", values.get("head_roll", 0.0))
        state.face.current_expression = name

    def on_head_changed(self, head_yaw: float = 0.0, head_pitch: float = 0.0,
                        head_roll: float = 0.0, **kw) -> None:
        target = self.ctx.state.target_head
        target.head_yaw = head_yaw
        target.head_pitch = head_pitch
        target.head_roll = head_roll

    # -- Auto behaviours ---------------------------------------------------

    def on_auto_blink(self, enabled: bool = True, **kw) -> None:
        self.ctx.state.face.auto_blink = enabled

    def on_auto_breathing(self, enabled: bool = True, **kw) -> None:
        self.ctx.state.face.auto_breathing = enabled

    def on_eye_tracking(self, enabled: bool = False, **kw) -> None:
        self.ctx.state.face.eye_tracking = enabled

    def on_micro_expressions(self, enabled: bool = False, **kw) -> None:
        self.ctx.state.face.micro_expressions = enabled

    def on_eye_color_set(self, name: str = "brown",
                         color: tuple = (0.42, 0.26, 0.13), **kw) -> None:
        self.ctx.state.face.eye_color = name
        features = getattr(self.ctx.pipeline, "face_features", None)
        if features is not None:
            features.set_eye_color(color)

    # -- Speech ------------------------------------------------------------

    def on_speech_play(self, text: str = "", speed: float = 1.0, **kw) -> None:
        """Turn text into a timed viseme sequence and start playing it."""
        if not text:
            return
        self.speech_sequence = self.ctx.speech_engine.generate_au_sequence(
            text, speed=speed)
        self.speech_time = 0.0
        self.speech_playing = True

    def advance_speech(self, dt: float) -> None:
        """Drive mouth AU targets from the viseme sequence.  Called per frame.

        Visemes carry absolute start/end times, so playback is a scan for the
        viseme covering the current time rather than a step through the list --
        a dropped frame skips a viseme instead of desynchronising the rest.
        """
        if not self.speech_playing:
            return
        self.speech_time += dt
        state = self.ctx.state

        active = None
        for viseme in self.speech_sequence:
            if viseme.start_time <= self.speech_time <= viseme.end_time:
                active = viseme
                break

        if active is not None:
            for au_id, val in active.au_targets.items():
                state.target_au.set(au_id, val)
        elif self.speech_time > 0 and (
                not self.speech_sequence
                or self.speech_time > self.speech_sequence[-1].end_time):
            for au_id in SPEECH_RESET_AUS:
                state.target_au.set(au_id, 0.0)
            self.speech_playing = False
