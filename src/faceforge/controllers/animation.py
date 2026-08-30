"""Animation clip playback: transport controls and the player's callbacks.

The player pushes sampled values *out* through callbacks rather than reaching
into state itself, so the same clip can drive the body, the face and the scene
wrapper without the player knowing any of them exist.  This controller is where
those callbacks are bound.

The camera is deliberately left out: no ``on_camera`` callback is bound, so a
clip never takes the view away from the user mid-playback.
"""

from __future__ import annotations

import logging
from typing import Any

from faceforge.core.events import EventType

logger = logging.getLogger(__name__)


class AnimationController:
    """Transport handlers plus the animation player's output callbacks."""

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx

    def bind_player(self) -> None:
        """Bind the player's output callbacks to scene, body and face."""
        player = self.ctx.anim_player
        player.on_wrapper_transform = self.on_wrapper_transform
        player.on_body_state = self.on_body_state
        # Camera left under user control -- no on_camera callback.
        player.on_face = self.on_face
        player.on_complete = self.on_complete

    def subscribe(self) -> None:
        bus = self.ctx.event_bus
        bus.subscribe(EventType.ANIM_PLAY, self.on_play)
        bus.subscribe(EventType.ANIM_PAUSE, self.on_pause)
        bus.subscribe(EventType.ANIM_STOP, self.on_stop)
        bus.subscribe(EventType.ANIM_SEEK, self.on_seek)
        bus.subscribe(EventType.ANIM_SPEED, self.on_speed)
        bus.subscribe(EventType.ANIM_CLIP_SELECTED, self.on_clip_selected)

    # -- Player callbacks --------------------------------------------------

    def on_wrapper_transform(self, pos, quat) -> None:
        self.ctx.scene_controller.set_wrapper_transform(pos, quat)

    def on_body_state(self, state_dict: dict) -> None:
        self.ctx.state.target_body.set_from_js_dict(state_dict)

    def on_face(self, aus_dict: dict, head_dict: dict) -> None:
        state = self.ctx.state
        if aus_dict:
            for au_id, val in aus_dict.items():
                state.target_au.set(au_id, val)
        if head_dict:
            state.target_head.head_yaw = head_dict.get("headYaw", 0.0)
            state.target_head.head_pitch = head_dict.get("headPitch", 0.0)
            state.target_head.head_roll = head_dict.get("headRoll", 0.0)

    def on_complete(self) -> None:
        """Announce completion and put the transport back to a stopped state."""
        player = self.ctx.anim_player
        self.ctx.event_bus.publish(
            EventType.ANIM_PROGRESS, progress=1.0,
            time=player.duration, duration=player.duration)
        self.ctx.control_panel.display_tab.transport.set_playing(False)

    # -- Transport handlers ------------------------------------------------

    def on_play(self, **kw) -> None:
        self.ctx.anim_player.play()

    def on_pause(self, **kw) -> None:
        self.ctx.anim_player.pause()

    def on_stop(self, **kw) -> None:
        self.ctx.anim_player.stop()

    def on_seek(self, position: float = 0.0, **kw) -> None:
        self.ctx.anim_player.seek(position)

    def on_speed(self, speed: float = 1.0, **kw) -> None:
        self.ctx.anim_player.set_speed(speed)

    def on_clip_selected(self, clip_name: str = "", **kw) -> None:
        clip = self.ctx.builtin_clips.get(clip_name)
        if clip is None:
            return
        self.ctx.anim_player.load(clip)
        self.ctx.control_panel.display_tab.transport.set_duration(clip.duration)

    # -- Per-frame ---------------------------------------------------------

    def update_frame(self) -> None:
        """Push playback position into the transport widget.

        Skipped while stopped at zero, so a session that never plays anything
        does not touch the widget sixty times a second.
        """
        player = self.ctx.anim_player
        if player.is_playing or player.progress > 0:
            self.ctx.control_panel.display_tab.update_animation_progress(
                player.progress, player.current_time, player.duration)
