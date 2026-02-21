"""Built-in animation clips for scene mode.

Provides the "Wake Up" sequence (supine on table -> standing) and its
reverse "Lie Down" (standing -> supine on table).

Body model coordinate system (Z-up):
  +Z = head (superior), -Z = feet (inferior)
  -Y = face (anterior),  +Y = back (posterior)
  +X = right,            -X = left
  Head top at Z≈+28, feet at Z≈-93.

World coordinate system (Y-up):
  +Y = up, +X = right, +Z = toward camera.
  Table at Y=90, extends along X-axis.
"""

from __future__ import annotations

import math

import numpy as np

from faceforge.core.math_utils import (
    quat_from_axis_angle,
    quat_multiply,
    mat4_from_quaternion,
    vec3,
)
from faceforge.scene.scene_animation import AnimationClip, AnimationKeyframe
from faceforge.scene.scene_environment import TABLE_HEIGHT
from faceforge.scene.scene_mode_controller import _BODY_CENTER_X


# ── Quaternion presets ────────────────────────────────────────────────

# Supine: empirically determined rotation that places body on table.
# Maps body +Z (head) → world -X, body -Y (face) → world +Y (up).
# Matches scene_mode_controller._Q_SUPINE.
_Q_SUPINE = quat_multiply(
    quat_from_axis_angle(vec3(0, 0, 1), math.pi / 2),
    quat_multiply(
        quat_from_axis_angle(vec3(0, 1, 0), math.pi / 2),
        quat_from_axis_angle(vec3(1, 0, 0), -math.pi / 2),
    ),
)

# Standing upright, facing +X (beside the table).
# = Rz(-90°) * Q_SUPINE.  Maps body +Z → world +Y, body -Y → world +X.
_Q_STANDING_X = quat_multiply(
    quat_from_axis_angle(vec3(0, 0, 1), -math.pi / 2),
    _Q_SUPINE,
)

# Standing upright, facing +Z (toward camera).
# = Ry(-90°) * Q_STANDING_X = Rx(-90°).
# Maps body +Z → world +Y, body -Y → world +Z.
_Q_STANDING_Z = quat_from_axis_angle(vec3(1, 0, 0), -math.pi / 2)


def _q_sit_up(degrees: float) -> np.ndarray:
    """Quaternion at a given sit-up angle from supine.

    Rotates around world -Z axis.  0° = supine, 90° = upright facing +X.
    Face stays pointing upward throughout; feet swing toward +X (table edge).
    """
    dq = quat_from_axis_angle(vec3(0, 0, 1), math.radians(-degrees))
    return quat_multiply(dq, _Q_SUPINE)


def _sit_up_wrapper_pos(degrees: float) -> tuple[float, float, float]:
    """Compute wrapper position that keeps the pelvis anchored to the table.

    During the sit-up, the pelvis (body Z≈-50) should stay near the table
    surface at Y=TABLE_Y.  We compute the world offset of the pelvis at
    the current rotation and set wrapper_pos to cancel it.
    """
    q = _q_sit_up(degrees)
    M = mat4_from_quaternion(q)[:3, :3]
    pelvis_offset = M @ np.array([0, 0, -50], dtype=np.float64)
    # Keep pelvis at roughly (-35, TABLE_Y, 0) — same as supine pelvis position
    # But let pelvis X drift toward table edge as body rotates
    t = degrees / 90.0  # 0→1 over the sit-up range
    target_pelvis_x = -35.0 + t * 50.0  # drift from -35 to +15
    px = target_pelvis_x - pelvis_offset[0]
    py = _TABLE_Y - pelvis_offset[1]
    return (round(px, 1), round(py, 1), 0)


# Pre-computed rotation stages for the sit-up animation
_Q_30 = _q_sit_up(30)
_Q_60 = _q_sit_up(60)

# Table surface Y
_TABLE_Y = TABLE_HEIGHT + 15  # 105

# Standing wrapper Y: feet at body Z=-93, standing maps body Z → world Y.
# For feet on floor (Y=0): wrapper_Y = 93.
_STAND_Y = 93.0

# Body center X offset (matches scene_mode_controller)
_CX = _BODY_CENTER_X


def _q_tuple(q: np.ndarray) -> tuple[float, float, float, float]:
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


# ── Body state presets (camelCase matching body_poses.json) ──────────

_ANATOMICAL = {
    "spineFlex": 0, "spineLatBend": 0, "spineRotation": 0,
    "shoulderRAbduct": 0, "shoulderRFlex": 0,
    "shoulderLAbduct": 0, "shoulderLFlex": 0,
    "elbowRFlex": 0, "elbowLFlex": 0,
    "hipRFlex": 0, "hipLFlex": 0,
    "kneeRFlex": 0, "kneeLFlex": 0,
    "ankleRFlex": 0, "ankleLFlex": 0,
}

_SITTING_UP = {
    "spineFlex": 0.6,
    "hipRFlex": 0.5, "hipLFlex": 0.5,
    "kneeRFlex": 0.1, "kneeLFlex": 0.1,
}

_SITTING_EDGE = {
    "spineFlex": 0.1,
    "hipRFlex": 0.7, "hipLFlex": 0.7,
    "kneeRFlex": 0.7, "kneeLFlex": 0.7,
    "ankleRFlex": 0, "ankleLFlex": 0,
    "shoulderRAbduct": -0.05, "shoulderLAbduct": -0.05,
    "elbowRFlex": 0.2, "elbowLFlex": 0.2,
}

_SLIGHT_CROUCH = {
    "spineFlex": 0.05,
    "hipRFlex": 0.3, "hipLFlex": 0.3,
    "kneeRFlex": 0.4, "kneeLFlex": 0.4,
    "ankleRFlex": 0.1, "ankleLFlex": 0.1,
}

_RELAXED = {
    "spineFlex": 0.05,
    "shoulderRAbduct": -0.1, "shoulderRFlex": 0.05,
    "shoulderLAbduct": -0.1, "shoulderLFlex": 0.05,
    "elbowRFlex": 0.15, "elbowLFlex": 0.15,
    "hipRFlex": 0.05, "hipLFlex": 0.05,
    "kneeRFlex": 0.05, "kneeLFlex": 0.05,
}


# ── Wake Up clip ─────────────────────────────────────────────────────

def make_wake_up_clip() -> AnimationClip:
    """Create the "Wake Up" animation: supine on table -> standing.

    Body model is Z-up: head at Z≈+28, feet at Z≈-93, face toward -Y.

    Phase 1 (0-4s): Supine on table, eyes open, spine curls up.
    Phase 2 (4-10s): Sit-up — rotate -90° around world Z from supine.
      Pelvis stays anchored to table surface.  At 90°, body is upright
      facing +X with pelvis at table height.
    Phase 3 (10-14s): Slide off table edge, feet reach floor.
    Phase 4 (14-17s): Turn 90° to face camera (+Z).
    Phase 5 (17-20s): Hold relaxed standing pose.
    """
    # Pre-compute pelvis-anchored positions for sit-up keyframes
    pos_30 = _sit_up_wrapper_pos(30)
    pos_60 = _sit_up_wrapper_pos(60)
    pos_90 = _sit_up_wrapper_pos(90)

    keyframes = [
        # 0s: Supine on table, still
        AnimationKeyframe(
            time=0.0,
            wrapper_position=(_CX, _TABLE_Y, 0),
            wrapper_quaternion=_q_tuple(_Q_SUPINE),
            body_state=_ANATOMICAL.copy(),
            easing="linear",
        ),
        # 1s: Eyes open (hold position)
        AnimationKeyframe(
            time=1.0,
            wrapper_position=(_CX, _TABLE_Y, 0),
            wrapper_quaternion=_q_tuple(_Q_SUPINE),
            body_state=_ANATOMICAL.copy(),
            face_aus={"AU5": 0.3},
            easing="ease_in_out",
        ),
        # 4s: Spine curls up (still supine-rotated, body stays on table)
        AnimationKeyframe(
            time=4.0,
            wrapper_position=(_CX, _TABLE_Y, 0),
            wrapper_quaternion=_q_tuple(_Q_SUPINE),
            body_state=_SITTING_UP.copy(),
            easing="ease_in_out",
        ),
        # 6s: Sit up 30° — torso lifting, pelvis anchored to table
        AnimationKeyframe(
            time=6.0,
            wrapper_position=pos_30,
            wrapper_quaternion=_q_tuple(_Q_30),
            body_state=_SITTING_EDGE.copy(),
            easing="ease_in_out",
        ),
        # 8s: Sit up 60° — nearly upright, pelvis still on table
        AnimationKeyframe(
            time=8.0,
            wrapper_position=pos_60,
            wrapper_quaternion=_q_tuple(_Q_60),
            body_state=_SITTING_EDGE.copy(),
            easing="ease_in_out",
        ),
        # 10s: Fully upright 90° — seated on table, facing +X
        AnimationKeyframe(
            time=10.0,
            wrapper_position=pos_90,
            wrapper_quaternion=_q_tuple(_Q_STANDING_X),
            body_state=_SITTING_EDGE.copy(),
            easing="ease_in_out",
        ),
        # 12s: Sliding off table edge — feet approaching floor
        AnimationKeyframe(
            time=12.0,
            wrapper_position=(20, 115, 0),
            wrapper_quaternion=_q_tuple(_Q_STANDING_X),
            body_state=_SLIGHT_CROUCH.copy(),
            easing="ease_out",
        ),
        # 14s: Feet on floor — standing upright facing +X
        AnimationKeyframe(
            time=14.0,
            wrapper_position=(50, _STAND_Y, 0),
            wrapper_quaternion=_q_tuple(_Q_STANDING_X),
            body_state=_ANATOMICAL.copy(),
            easing="ease_in_out",
        ),
        # 17s: Turn to face camera (+Z), standing relaxed
        AnimationKeyframe(
            time=17.0,
            wrapper_position=(50, _STAND_Y, 0),
            wrapper_quaternion=_q_tuple(_Q_STANDING_Z),
            body_state=_RELAXED.copy(),
            easing="ease_in_out",
        ),
        # 20s: Hold (end)
        AnimationKeyframe(
            time=20.0,
            wrapper_position=(50, _STAND_Y, 0),
            wrapper_quaternion=_q_tuple(_Q_STANDING_Z),
            body_state=_RELAXED.copy(),
            easing="linear",
        ),
    ]
    return AnimationClip(name="Wake Up", keyframes=keyframes, loop=False)


# ── Lie Down clip ────────────────────────────────────────────────────

def make_lie_down_clip() -> AnimationClip:
    """Create the "Lie Down" animation: standing -> supine on table (reverse of Wake Up)."""
    wake = make_wake_up_clip()
    # Reverse the keyframes and remap times
    duration = wake.duration
    reversed_kfs = []
    for kf in reversed(wake.keyframes):
        new_kf = AnimationKeyframe(
            time=duration - kf.time,
            wrapper_position=kf.wrapper_position,
            wrapper_quaternion=kf.wrapper_quaternion,
            body_state=kf.body_state.copy() if kf.body_state else None,
            camera_position=kf.camera_position,
            camera_target=kf.camera_target,
            face_aus=kf.face_aus.copy() if kf.face_aus else None,
            head_rotation=kf.head_rotation.copy() if kf.head_rotation else None,
            easing=kf.easing,
        )
        reversed_kfs.append(new_kf)
    return AnimationClip(name="Lie Down", keyframes=reversed_kfs, loop=False)


# ── Registry ─────────────────────────────────────────────────────────

def get_builtin_clips() -> dict[str, AnimationClip]:
    """Return all built-in animation clips by name."""
    return {
        "Wake Up": make_wake_up_clip(),
        "Lie Down": make_lie_down_clip(),
    }
