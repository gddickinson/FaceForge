"""Keyframe animation engine for scene-mode sequences.

Provides AnimationKeyframe, AnimationClip, and AnimationPlayer for
driving wrapper transforms, body state, camera, and face AUs over time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from faceforge.core.math_utils import (
    lerp,
    lerp_vec3,
    quat_slerp,
    vec3,
)


# ── Easing functions ─────────────────────────────────────────────────

def _ease_linear(t: float) -> float:
    return t


def _ease_in(t: float) -> float:
    return t * t


def _ease_out(t: float) -> float:
    return t * (2.0 - t)


def _ease_in_out(t: float) -> float:
    if t < 0.5:
        return 2.0 * t * t
    return -1.0 + (4.0 - 2.0 * t) * t


_EASING_MAP: dict[str, Callable[[float], float]] = {
    "linear": _ease_linear,
    "ease_in": _ease_in,
    "ease_out": _ease_out,
    "ease_in_out": _ease_in_out,
}


# ── Data classes ─────────────────────────────────────────────────────

@dataclass
class AnimationKeyframe:
    """A single keyframe in an animation clip."""

    time: float = 0.0

    wrapper_position: tuple[float, float, float] | None = None
    wrapper_quaternion: tuple[float, float, float, float] | None = None

    body_state: dict | None = None

    camera_position: tuple[float, float, float] | None = None
    camera_target: tuple[float, float, float] | None = None

    face_aus: dict | None = None
    head_rotation: dict | None = None

    easing: str = "ease_in_out"


@dataclass
class AnimationClip:
    """A sequence of keyframes that form a named animation."""

    name: str = ""
    keyframes: list[AnimationKeyframe] = field(default_factory=list)
    loop: bool = False

    @property
    def duration(self) -> float:
        if not self.keyframes:
            return 0.0
        return self.keyframes[-1].time


# ── Animation Player ─────────────────────────────────────────────────

class AnimationPlayer:
    """Plays an AnimationClip by interpolating between keyframes each tick.

    Callbacks are invoked with interpolated values:
      on_wrapper_transform(pos, quat)
      on_body_state(state_dict)
      on_camera(pos, target)
      on_face(aus_dict, head_dict)
      on_complete()
    """

    def __init__(self) -> None:
        self._clip: AnimationClip | None = None
        self._time: float = 0.0
        self._playing: bool = False
        self._speed: float = 1.0

        # Callbacks
        self.on_wrapper_transform: Callable | None = None
        self.on_body_state: Callable | None = None
        self.on_camera: Callable | None = None
        self.on_face: Callable | None = None
        self.on_complete: Callable | None = None

    # ── Properties ────────────────────────────────────────────────

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def current_time(self) -> float:
        return self._time

    @property
    def duration(self) -> float:
        return self._clip.duration if self._clip else 0.0

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def progress(self) -> float:
        d = self.duration
        if d <= 0.0:
            return 0.0
        return min(self._time / d, 1.0)

    # ── Control ───────────────────────────────────────────────────

    def load(self, clip: AnimationClip) -> None:
        """Load a clip and reset to the beginning."""
        self._clip = clip
        self._time = 0.0
        self._playing = False

    def play(self) -> None:
        if self._clip is not None:
            self._playing = True

    def pause(self) -> None:
        self._playing = False

    def stop(self) -> None:
        self._playing = False
        self._time = 0.0
        if self._clip and self._clip.keyframes:
            self._apply_keyframe_values(self._clip.keyframes[0])

    def seek(self, t: float) -> None:
        """Seek to a normalized position (0-1)."""
        if self._clip is None:
            return
        self._time = t * self._clip.duration
        self._evaluate()

    def set_speed(self, s: float) -> None:
        self._speed = max(0.01, s)

    # ── Per-frame tick ────────────────────────────────────────────

    def tick(self, dt: float) -> None:
        """Advance the animation by dt seconds and apply interpolated values."""
        if not self._playing or self._clip is None:
            return

        self._time += dt * self._speed
        duration = self._clip.duration

        if self._time >= duration:
            if self._clip.loop:
                self._time = self._time % duration if duration > 0 else 0.0
            else:
                self._time = duration
                self._playing = False
                self._evaluate()
                if self.on_complete:
                    self.on_complete()
                return

        self._evaluate()

    # ── Interpolation engine ──────────────────────────────────────

    def _evaluate(self) -> None:
        """Interpolate between the two surrounding keyframes and invoke callbacks."""
        clip = self._clip
        if clip is None or not clip.keyframes:
            return

        kfs = clip.keyframes
        t = self._time

        # Find surrounding keyframes
        if t <= kfs[0].time:
            self._apply_keyframe_values(kfs[0])
            return
        if t >= kfs[-1].time:
            self._apply_keyframe_values(kfs[-1])
            return

        # Binary search for the right interval
        lo, hi = 0, len(kfs) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if kfs[mid].time <= t:
                lo = mid
            else:
                hi = mid

        kf_a = kfs[lo]
        kf_b = kfs[hi]
        seg_duration = kf_b.time - kf_a.time
        if seg_duration <= 0:
            self._apply_keyframe_values(kf_b)
            return

        raw_t = (t - kf_a.time) / seg_duration
        easing_fn = _EASING_MAP.get(kf_b.easing, _ease_in_out)
        eased_t = easing_fn(raw_t)

        self._apply_interpolated(kf_a, kf_b, eased_t)

    def _apply_keyframe_values(self, kf: AnimationKeyframe) -> None:
        """Apply a single keyframe's values directly (no interpolation)."""
        if self.on_wrapper_transform and (kf.wrapper_position or kf.wrapper_quaternion):
            pos = kf.wrapper_position
            quat = kf.wrapper_quaternion
            self.on_wrapper_transform(pos, quat)

        if self.on_body_state and kf.body_state:
            self.on_body_state(kf.body_state)

        if self.on_camera and (kf.camera_position or kf.camera_target):
            self.on_camera(kf.camera_position, kf.camera_target)

        if self.on_face and (kf.face_aus or kf.head_rotation):
            self.on_face(kf.face_aus, kf.head_rotation)

    def _apply_interpolated(
        self, a: AnimationKeyframe, b: AnimationKeyframe, t: float,
    ) -> None:
        """Interpolate between two keyframes and invoke callbacks."""
        # Wrapper transform
        if self.on_wrapper_transform:
            pos = _interp_pos(a.wrapper_position, b.wrapper_position, t)
            quat = _interp_quat(a.wrapper_quaternion, b.wrapper_quaternion, t)
            if pos is not None or quat is not None:
                self.on_wrapper_transform(pos, quat)

        # Body state
        if self.on_body_state:
            bs = _interp_dict(a.body_state, b.body_state, t)
            if bs is not None:
                self.on_body_state(bs)

        # Camera
        if self.on_camera:
            cam_pos = _interp_pos(a.camera_position, b.camera_position, t)
            cam_tgt = _interp_pos(a.camera_target, b.camera_target, t)
            if cam_pos is not None or cam_tgt is not None:
                self.on_camera(cam_pos, cam_tgt)

        # Face
        if self.on_face:
            aus = _interp_dict(a.face_aus, b.face_aus, t)
            head = _interp_dict(a.head_rotation, b.head_rotation, t)
            if aus is not None or head is not None:
                self.on_face(aus, head)


# ── Interpolation helpers ─────────────────────────────────────────

def _interp_pos(
    a: tuple[float, float, float] | None,
    b: tuple[float, float, float] | None,
    t: float,
) -> tuple[float, float, float] | None:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    v = lerp_vec3(
        np.array(a, dtype=np.float64),
        np.array(b, dtype=np.float64),
        t,
    )
    return (float(v[0]), float(v[1]), float(v[2]))


def _interp_quat(
    a: tuple[float, float, float, float] | None,
    b: tuple[float, float, float, float] | None,
    t: float,
) -> tuple[float, float, float, float] | None:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    q = quat_slerp(
        np.array(a, dtype=np.float64),
        np.array(b, dtype=np.float64),
        t,
    )
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def _interp_dict(
    a: dict | None,
    b: dict | None,
    t: float,
) -> dict | None:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    # Merge keys from both
    keys = set(a.keys()) | set(b.keys())
    result = {}
    for k in keys:
        va = a.get(k, 0.0)
        vb = b.get(k, 0.0)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            result[k] = lerp(float(va), float(vb), t)
        else:
            result[k] = vb  # non-numeric: snap to b
    return result


# ── Serialization ─────────────────────────────────────────────────

def load_clip_from_dict(d: dict) -> AnimationClip:
    """Deserialize an AnimationClip from a JSON-compatible dict."""
    keyframes = []
    for kf_data in d.get("keyframes", []):
        kf = AnimationKeyframe(
            time=kf_data.get("time", 0.0),
            wrapper_position=tuple(kf_data["wrapper_position"]) if kf_data.get("wrapper_position") else None,
            wrapper_quaternion=tuple(kf_data["wrapper_quaternion"]) if kf_data.get("wrapper_quaternion") else None,
            body_state=kf_data.get("body_state"),
            camera_position=tuple(kf_data["camera_position"]) if kf_data.get("camera_position") else None,
            camera_target=tuple(kf_data["camera_target"]) if kf_data.get("camera_target") else None,
            face_aus=kf_data.get("face_aus"),
            head_rotation=kf_data.get("head_rotation"),
            easing=kf_data.get("easing", "ease_in_out"),
        )
        keyframes.append(kf)
    return AnimationClip(
        name=d.get("name", ""),
        keyframes=keyframes,
        loop=d.get("loop", False),
    )


def clip_to_dict(clip: AnimationClip) -> dict:
    """Serialize an AnimationClip to a JSON-compatible dict."""
    keyframes = []
    for kf in clip.keyframes:
        kf_data: dict = {"time": kf.time}
        if kf.wrapper_position is not None:
            kf_data["wrapper_position"] = list(kf.wrapper_position)
        if kf.wrapper_quaternion is not None:
            kf_data["wrapper_quaternion"] = list(kf.wrapper_quaternion)
        if kf.body_state is not None:
            kf_data["body_state"] = kf.body_state
        if kf.camera_position is not None:
            kf_data["camera_position"] = list(kf.camera_position)
        if kf.camera_target is not None:
            kf_data["camera_target"] = list(kf.camera_target)
        if kf.face_aus is not None:
            kf_data["face_aus"] = kf.face_aus
        if kf.head_rotation is not None:
            kf_data["head_rotation"] = kf.head_rotation
        if kf.easing != "ease_in_out":
            kf_data["easing"] = kf.easing
        keyframes.append(kf_data)
    return {
        "name": clip.name,
        "keyframes": keyframes,
        "loop": clip.loop,
    }
