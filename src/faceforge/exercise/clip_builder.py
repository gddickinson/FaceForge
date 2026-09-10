"""ExerciseDefinition -> a playable clip with phase spans and an activation track.

Keyframe semantics
------------------
Each :class:`~faceforge.exercise.model.Phase` holds the pose reached at its
END and how long that takes, so a rep of N phases is N keyframes, and the
clip starts from the last phase's pose (the start position).  The existing
:class:`~faceforge.scene.scene_animation.AnimationPlayer` interpolates between
keyframes with the easing named on the keyframe being approached, which is
exactly the per-phase easing an exercise wants.

Every keyframe carries a FULL pose: the player's dict interpolation treats a
missing key as zero, so a partial pose would snap unrelated joints to neutral.

Whole-body placement
--------------------
``wrapper_quaternion`` = orientation base (standing, supine, prone, ...)
composed with the phase's pitch / roll / yaw about the body's own axes.
``wrapper_position`` is the base placement for the orientation.  The ground
lock (:mod:`faceforge.body.ground_contact`) corrects the position every frame
so the anchored feet or hands stay put; the keyframe's ``lift`` (in the body
dict as ``ground_lift``) is the authored height *off* that anchor, which is
how a jump leaves the floor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from faceforge.core.math_utils import (
    quat_from_axis_angle, quat_multiply, quat_rotate_vec3, vec3,
)
from faceforge.exercise.activation import ActivationTrack, build_activation_track
from faceforge.exercise.model import ExerciseDefinition, Phase, PhaseKind
from faceforge.exercise.motion_description import JointMotion, describe_transition
from faceforge.exercise.pose_library import neutral
from faceforge.exercise.stabilisers import with_implied_stabilisers
from faceforge.scene.scene_animation import AnimationClip, AnimationKeyframe

#: Keys inside a keyframe's body dict that carry the authored lift and floor
#: travel of the anchor.  BodyState ignores them (no such attributes); the
#: exercise controller reads them.
LIFT_KEY = "ground_lift"
TRAVEL_X_KEY = "ground_travel_x"
TRAVEL_Z_KEY = "ground_travel_z"

#: Where the standing body's wrapper sits so the feet touch the floor (Y=0).
STAND_Y = 203.0
BENCH_TOP_Y = 58.0

_X = vec3(1.0, 0.0, 0.0)
_Y = vec3(0.0, 1.0, 0.0)
_Z = vec3(0.0, 0.0, 1.0)


def _rx(deg: float) -> np.ndarray:
    return quat_from_axis_angle(_X, math.radians(deg))


def _supine() -> np.ndarray:
    """Face up, head toward world -X (the examination table's placement)."""
    return quat_multiply(
        quat_from_axis_angle(_Z, math.pi / 2),
        quat_multiply(quat_from_axis_angle(_Y, math.pi / 2), _rx(-90.0)),
    )


def orientation_base(orientation: str) -> tuple[np.ndarray, tuple[float, float, float]]:
    """``(quaternion, position)`` placing the body for an orientation."""
    if orientation in ("standing", "seated", "hanging"):
        return _rx(-90.0), (0.0, STAND_Y, 0.0)
    if orientation == "supine":
        return _supine(), (-85.0, BENCH_TOP_Y + 15.0, 0.0)
    if orientation == "prone":
        # Roll the supine body half a turn about its long axis (world X).
        return quat_multiply(_rx(180.0), _supine()), (-85.0, 25.0, 0.0)
    if orientation == "side":
        return quat_multiply(_rx(90.0), _supine()), (-85.0, 25.0, 0.0)
    raise ValueError(f"unknown orientation {orientation!r}")


def body_rotation(phase: Phase) -> np.ndarray:
    """The phase's body-frame rotation: yaw (about Z), then roll (Y), then pitch (X).

    Pitch + leans the head anteriorly, roll + leans to the body's right, yaw +
    turns the body to its left.
    """
    q = np.array([0.0, 0.0, 0.0, 1.0])
    if phase.yaw:
        q = quat_multiply(q, quat_from_axis_angle(_Z, math.radians(phase.yaw)))
    if phase.roll:
        q = quat_multiply(q, quat_from_axis_angle(_Y, math.radians(phase.roll)))
    if phase.pitch:
        q = quat_multiply(q, quat_from_axis_angle(_X, math.radians(phase.pitch)))
    return q


def phase_quaternion(base: np.ndarray, phase: Phase) -> np.ndarray:
    """Base orientation with the phase's body-frame rotation applied first."""
    return quat_multiply(base, body_rotation(phase))


def phase_position(base: np.ndarray, base_pos, phase: Phase) -> tuple[float, float, float]:
    """Wrapper position that keeps ``phase.pivot`` (a body point) where it was.

    The wrapper rotates about the body origin near the head.  A sit-up turns
    about the hips and a bridge about the shoulders, so the position is
    pre-translated by ``R_base (p - R_body p)`` to hold the pivot still.
    """
    if phase.position is not None:
        return tuple(float(v) for v in phase.position)
    pos = np.asarray(base_pos, dtype=np.float64)
    if phase.pivot is not None:
        p = vec3(*phase.pivot)
        moved = quat_rotate_vec3(body_rotation(phase), p)
        pos = pos + quat_rotate_vec3(base, p - moved)
    return (float(pos[0]), float(pos[1]), float(pos[2]))


@dataclass(frozen=True)
class PhaseSpan:
    """One phase of one rep on the clip's timeline."""

    index: int
    rep: int
    phase: Phase
    t0: float
    t1: float
    motions: list[JointMotion] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.phase.name

    @property
    def kind(self) -> PhaseKind:
        return self.phase.kind

    @property
    def cues(self) -> tuple[str, ...]:
        return self.phase.cues


@dataclass
class ExerciseClip:
    """A built demonstration: clip, timeline, activation, placement."""

    definition: ExerciseDefinition
    clip: AnimationClip
    spans: list[PhaseSpan]
    activation: ActivationTrack
    reps: int
    tempo: float
    base_quaternion: np.ndarray
    base_position: tuple[float, float, float]

    @property
    def duration(self) -> float:
        return self.clip.duration

    @property
    def rep_duration(self) -> float:
        return self.duration / self.reps if self.reps else 0.0

    def span_at(self, t: float) -> PhaseSpan | None:
        if not self.spans:
            return None
        t = min(max(t, 0.0), self.duration)
        for span in self.spans:
            if span.t0 <= t < span.t1:
                return span
        return self.spans[-1]

    def rep_at(self, t: float) -> int:
        span = self.span_at(t)
        return span.rep if span is not None else 0


def _full(pose: dict[str, float]) -> dict[str, float]:
    out = neutral()
    out.update({k: float(v) for k, v in pose.items()})
    return out


def build_exercise_clip(defn: ExerciseDefinition, reps: int | None = None,
                        tempo: float = 1.0) -> ExerciseClip:
    """Build the clip for ``reps`` repetitions at ``tempo`` (1.0 = as authored).

    ``tempo`` scales every phase duration: 0.5 is twice as slow, for teaching.
    """
    # The built clip colours what the body is doing, not only what the
    # catalogue lists as movers: grip, carry and brace are implied from the
    # equipment, anchor and orientation (exercise/stabilisers.py).
    defn = with_implied_stabilisers(defn)
    reps = defn.default_reps if reps is None else max(1, int(reps))
    tempo = max(0.05, float(tempo))
    base_q, base_pos = orientation_base(defn.orientation)
    if defn.base_position is not None:
        base_pos = tuple(float(v) for v in defn.base_position)

    phases = list(defn.phases)
    start_phase = phases[-1]
    keyframes: list[AnimationKeyframe] = []
    spans: list[PhaseSpan] = []
    raw_spans: list[tuple[float, float, Phase]] = []

    def keyframe(t: float, ph: Phase) -> AnimationKeyframe:
        body = _full(ph.pose)
        body[LIFT_KEY] = float(ph.lift)
        body[TRAVEL_X_KEY] = float(ph.travel[0])
        body[TRAVEL_Z_KEY] = float(ph.travel[1])
        ph_base_q, ph_base_pos = base_q, base_pos
        if ph.orientation is not None and ph.orientation != defn.orientation:
            ph_base_q, ph_base_pos = orientation_base(ph.orientation)
        q = phase_quaternion(ph_base_q, ph)
        return AnimationKeyframe(
            time=round(t, 6), wrapper_position=phase_position(ph_base_q, ph_base_pos, ph),
            wrapper_quaternion=(float(q[0]), float(q[1]), float(q[2]), float(q[3])),
            body_state=body, easing=ph.easing,
        )

    keyframes.append(keyframe(0.0, start_phase))
    t = 0.0
    index = 0
    prev = start_phase
    for rep in range(1, reps + 1):
        for ph in phases:
            t0 = t
            t += ph.duration / tempo
            keyframes.append(keyframe(t, ph))
            motions = describe_transition(_full(prev.pose), _full(ph.pose))
            spans.append(PhaseSpan(index, rep, ph, t0, t, motions))
            raw_spans.append((t0, t, ph))
            prev = ph
            index += 1

    clip = AnimationClip(name=defn.name, keyframes=keyframes, loop=True)
    track = build_activation_track(defn, raw_spans)
    return ExerciseClip(defn, clip, spans, track, reps, tempo, base_q, base_pos)
