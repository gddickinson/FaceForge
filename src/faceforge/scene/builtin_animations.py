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

# Standing wrapper Y: feet at body Z≈-200, standing maps body Z → world Y.
# For feet on floor (Y≈0): wrapper_Y ≈ 203.
_STAND_Y = 203.0

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


# ── Dance routine ─────────────────────────────────────────────────────

def _q_turn(degrees: float) -> tuple[float, float, float, float]:
    """Quaternion for standing upright facing *degrees* from +Z (camera).

    0° = facing camera (+Z).  Positive = turning left (counterclockwise
    when viewed from above).  The base standing quaternion is Rx(-90°)
    which maps body Z-up to world Y-up; we compose a world-Y rotation
    on top.
    """
    q_yaw = quat_from_axis_angle(vec3(0, 1, 0), math.radians(-degrees))
    return _q_tuple(quat_multiply(q_yaw, _Q_STANDING_Z))


def make_dance_clip() -> AnimationClip:
    """Create a ~64s looping contemporary dance routine.

    The choreography is a continuous flowing sequence using all 26 body
    DOFs, head rotation, and facial expression.  The body starts and ends
    at center stage facing the camera so the loop is seamless.

    Joint ranges (normalized -1 to 1 → mapped to degrees in body_animation):
      Spine:    flex ±45°, latBend ±30°, rotation ±30°
      Shoulder: abduct ±90°, flex ±90°, rotate ±70°
      Elbow:    flex ±145°
      Forearm:  rotate ±90°
      Wrist:    flex ±70°, deviate ±30°
      Hip:      flex ±90°, abduct ±45°, rotate ±45°
      Knee:     flex ±145°
      Ankle:    flex ±45°, invert ±30°
      Fingers:  curl, spread, thumb_op (0-1)
      Toes:     curl, spread (0-1)

    The wrapper position provides stage movement (X=side, Y=height, Z=depth).
    Wrapper quaternion rotates the whole body (turns, spins).

    Choreographic structure:
      0-4s    Opening: breath, arms float up
      4-8s    Port de bras: flowing arm circles
      8-12s   Weight shift + arabesque preparation
      12-16s  Arabesque right leg: full extension
      16-20s  Recovery + travelling step right
      20-24s  Contemporary floor reach + contract
      24-28s  Spiral turn preparation
      28-32s  Full turn (360°) with arms extended
      32-36s  Landing + ripple through body
      36-40s  Deep lunge left + arm sweep
      40-44s  Rise + développé (leg unfold)
      44-48s  Promenade turn with leg extended
      48-52s  Descent + floor work
      52-56s  Rise + expansive reach
      56-60s  Final turn back to front
      60-64s  Closing: settle to rest (matches opening for loop)
    """
    # Standing center stage, feet on floor
    SY = _STAND_Y  # 203
    face_z = _q_turn(0)  # facing camera
    face_r = _q_turn(-45)  # facing right-of-camera
    face_l = _q_turn(45)   # facing left-of-camera
    face_3q = _q_turn(90)  # facing stage left (profile)

    # ── Body state poses ──
    # Relaxed neutral
    neutral = {
        "spineFlex": 0.03, "spineLatBend": 0, "spineRotation": 0,
        "shoulderRAbduct": -0.08, "shoulderRFlex": 0.03, "shoulderRRotate": 0,
        "shoulderLAbduct": -0.08, "shoulderLFlex": 0.03, "shoulderLRotate": 0,
        "elbowRFlex": 0.1, "elbowLFlex": 0.1,
        "forearmRRotate": 0, "forearmLRotate": 0,
        "wristRFlex": 0, "wristRDeviate": 0,
        "wristLFlex": 0, "wristLDeviate": 0,
        "hipRFlex": 0.02, "hipRAbduct": 0, "hipRRotate": 0,
        "hipLFlex": 0.02, "hipLAbduct": 0, "hipLRotate": 0,
        "kneeRFlex": 0.03, "kneeLFlex": 0.03,
        "ankleRFlex": 0, "ankleLFlex": 0,
        "ankleRInvert": 0, "ankleLInvert": 0,
        "fingerRCurl": 0.15, "fingerLCurl": 0.15,
        "fingerRSpread": 0.2, "fingerLSpread": 0.2,
        "thumbROp": 0.1, "thumbLOp": 0.1,
        "toeRCurl": 0, "toeLCurl": 0,
    }

    # Arms floating up — breath opening
    arms_rise = {
        **neutral,
        "spineFlex": -0.05,  # slight arch back
        "shoulderRAbduct": 0.5, "shoulderRFlex": 0.3,
        "shoulderLAbduct": 0.5, "shoulderLFlex": 0.3,
        "elbowRFlex": 0.15, "elbowLFlex": 0.15,
        "wristRFlex": -0.2, "wristLFlex": -0.2,
        "fingerRSpread": 0.6, "fingerLSpread": 0.6,
        "fingerRCurl": 0, "fingerLCurl": 0,
    }

    # Port de bras: right arm high, left arm side (first position)
    port_de_bras_1 = {
        **neutral,
        "spineLatBend": 0.15, "spineRotation": 0.1,
        "shoulderRAbduct": 0.85, "shoulderRFlex": 0.6,
        "shoulderRRotate": -0.2,
        "shoulderLAbduct": 0.6, "shoulderLFlex": 0.1,
        "shoulderLRotate": 0.15,
        "elbowRFlex": 0.25, "elbowLFlex": 0.2,
        "forearmRRotate": 0.3, "forearmLRotate": -0.2,
        "wristRFlex": -0.15, "wristLFlex": -0.1,
        "hipRFlex": 0.05, "hipLFlex": 0.05,
        "kneeRFlex": 0.05, "kneeLFlex": 0.05,
        "fingerRSpread": 0.5, "fingerLSpread": 0.4,
    }

    # Port de bras: swap — left arm high, right arm side
    port_de_bras_2 = {
        **neutral,
        "spineLatBend": -0.15, "spineRotation": -0.1,
        "shoulderLAbduct": 0.85, "shoulderLFlex": 0.6,
        "shoulderLRotate": -0.2,
        "shoulderRAbduct": 0.6, "shoulderRFlex": 0.1,
        "shoulderRRotate": 0.15,
        "elbowLFlex": 0.25, "elbowRFlex": 0.2,
        "forearmLRotate": 0.3, "forearmRRotate": -0.2,
        "wristLFlex": -0.15, "wristRFlex": -0.1,
        "hipRFlex": 0.05, "hipLFlex": 0.05,
        "kneeRFlex": 0.05, "kneeLFlex": 0.05,
        "fingerLSpread": 0.5, "fingerRSpread": 0.4,
    }

    # Weight shift right — preparation for arabesque
    weight_shift_r = {
        **neutral,
        "spineFlex": 0.08, "spineLatBend": -0.12,
        "shoulderRAbduct": 0.3, "shoulderRFlex": 0.2,
        "shoulderLAbduct": 0.3, "shoulderLFlex": 0.4,
        "elbowRFlex": 0.15, "elbowLFlex": 0.2,
        "hipRFlex": 0.15, "hipLFlex": -0.1,
        "hipRAbduct": -0.05, "hipLAbduct": 0.15,
        "kneeRFlex": 0.2, "kneeLFlex": 0.05,
        "ankleRFlex": 0.05,
    }

    # Arabesque: right leg standing, left leg extended back
    arabesque = {
        **neutral,
        "spineFlex": -0.15, "spineLatBend": 0, "spineRotation": 0.05,
        "shoulderRAbduct": 0.7, "shoulderRFlex": 0.5,
        "shoulderLAbduct": 0.7, "shoulderLFlex": -0.2,
        "shoulderLRotate": 0.2,
        "elbowRFlex": 0.1, "elbowLFlex": 0.1,
        "forearmRRotate": 0.2, "forearmLRotate": -0.15,
        "wristRFlex": -0.15, "wristLFlex": -0.1,
        "hipRFlex": 0.05,  # standing leg
        "hipLFlex": -0.6,  # extended back
        "hipLAbduct": 0.05,
        "kneeRFlex": 0.08,  # slight bend standing leg
        "kneeLFlex": 0.02,  # straight working leg
        "ankleRFlex": 0.05,
        "ankleLFlex": -0.15,  # pointed foot
        "fingerRSpread": 0.6, "fingerLSpread": 0.6,
        "toeRCurl": 0.3,
    }

    # Travel step — walking step right with arms flowing
    travel_step = {
        **neutral,
        "spineFlex": 0.05, "spineLatBend": -0.08, "spineRotation": -0.15,
        "shoulderRAbduct": 0.4, "shoulderRFlex": -0.1,
        "shoulderLAbduct": 0.3, "shoulderLFlex": 0.3,
        "elbowRFlex": 0.12, "elbowLFlex": 0.3,
        "forearmLRotate": 0.2,
        "hipRFlex": -0.05, "hipLFlex": 0.25,
        "hipRAbduct": 0.1,
        "kneeRFlex": 0.05, "kneeLFlex": 0.35,
        "ankleRFlex": -0.1,  # push off
        "ankleLFlex": 0.1,
    }

    # Contemporary floor reach — deep forward bend
    floor_reach = {
        **neutral,
        "spineFlex": 0.7, "spineLatBend": 0.1, "spineRotation": 0,
        "shoulderRAbduct": 0.2, "shoulderRFlex": 0.8,
        "shoulderLAbduct": 0.15, "shoulderLFlex": 0.7,
        "elbowRFlex": 0.05, "elbowLFlex": 0.05,
        "forearmRRotate": 0.1, "forearmLRotate": -0.1,
        "wristRFlex": 0.3, "wristLFlex": 0.3,
        "hipRFlex": 0.6, "hipLFlex": 0.6,
        "kneeRFlex": 0.3, "kneeLFlex": 0.3,
        "ankleRFlex": 0.1, "ankleLFlex": 0.1,
        "fingerRCurl": 0.5, "fingerLCurl": 0.5,
        "fingerRSpread": 0, "fingerLSpread": 0,
    }

    # Contraction — Graham-style pulled center
    contraction = {
        **neutral,
        "spineFlex": 0.5, "spineLatBend": 0, "spineRotation": -0.1,
        "shoulderRAbduct": 0.2, "shoulderRFlex": 0.15,
        "shoulderLAbduct": 0.2, "shoulderLFlex": 0.15,
        "elbowRFlex": 0.5, "elbowLFlex": 0.5,
        "forearmRRotate": 0.3, "forearmLRotate": -0.3,
        "wristRFlex": 0.2, "wristLFlex": 0.2,
        "hipRFlex": 0.35, "hipLFlex": 0.35,
        "kneeRFlex": 0.4, "kneeLFlex": 0.4,
        "ankleRFlex": 0.05, "ankleLFlex": 0.05,
        "fingerRCurl": 0.7, "fingerLCurl": 0.7,
    }

    # Spiral prep — torso twist with arms wrapping
    spiral_prep = {
        **neutral,
        "spineFlex": 0.05, "spineLatBend": 0, "spineRotation": 0.4,
        "shoulderRAbduct": 0.3, "shoulderRFlex": 0.3,
        "shoulderRRotate": -0.3,
        "shoulderLAbduct": 0.4, "shoulderLFlex": 0.1,
        "shoulderLRotate": 0.2,
        "elbowRFlex": 0.4, "elbowLFlex": 0.25,
        "forearmRRotate": 0.4, "forearmLRotate": -0.2,
        "hipRFlex": 0.15, "hipLFlex": 0.15,
        "kneeRFlex": 0.25, "kneeLFlex": 0.25,
        "ankleRFlex": 0.05, "ankleLFlex": 0.05,
        "fingerRCurl": 0.3, "fingerLCurl": 0.1,
    }

    # Extended turn — arms wide, one leg in passé
    turn_pose = {
        **neutral,
        "spineFlex": -0.05, "spineLatBend": 0, "spineRotation": 0,
        "shoulderRAbduct": 0.8, "shoulderRFlex": 0.1,
        "shoulderRRotate": -0.1,
        "shoulderLAbduct": 0.8, "shoulderLFlex": 0.1,
        "shoulderLRotate": 0.1,
        "elbowRFlex": 0.12, "elbowLFlex": 0.12,
        "forearmRRotate": 0.15, "forearmLRotate": -0.15,
        "wristRFlex": -0.1, "wristLFlex": -0.1,
        "hipRFlex": 0.05,  # standing leg
        "hipLFlex": 0.35,  # passé
        "hipLAbduct": 0.3,
        "hipLRotate": -0.2,
        "kneeRFlex": 0.05,
        "kneeLFlex": 0.75,  # bent in passé
        "ankleRFlex": 0.1,  # relevé
        "ankleLFlex": -0.2,  # pointed
        "fingerRSpread": 0.7, "fingerLSpread": 0.7,
        "toeRCurl": 0.4,
    }

    # Landing ripple — soft knees, arms trailing
    landing_ripple = {
        **neutral,
        "spineFlex": 0.15, "spineLatBend": 0.05,
        "shoulderRAbduct": 0.5, "shoulderRFlex": -0.1,
        "shoulderLAbduct": 0.5, "shoulderLFlex": -0.1,
        "elbowRFlex": 0.2, "elbowLFlex": 0.2,
        "forearmRRotate": -0.2, "forearmLRotate": 0.2,
        "wristRFlex": 0.15, "wristLFlex": 0.15,
        "hipRFlex": 0.25, "hipLFlex": 0.25,
        "kneeRFlex": 0.35, "kneeLFlex": 0.35,
        "ankleRFlex": 0.08, "ankleLFlex": 0.08,
        "fingerRCurl": 0.3, "fingerLCurl": 0.3,
    }

    # Deep lunge left — right leg extended, left knee deep bend
    lunge_left = {
        **neutral,
        "spineFlex": -0.1, "spineLatBend": -0.2, "spineRotation": 0.15,
        "shoulderRAbduct": 0.9, "shoulderRFlex": 0.2,
        "shoulderRRotate": -0.15,
        "shoulderLAbduct": 0.3, "shoulderLFlex": 0.5,
        "shoulderLRotate": 0.1,
        "elbowRFlex": 0.08, "elbowLFlex": 0.15,
        "forearmRRotate": 0.2,
        "wristRFlex": -0.2, "wristLFlex": -0.1,
        "hipRFlex": -0.15, "hipRAbduct": 0.25,
        "hipLFlex": 0.7, "hipLAbduct": -0.1,
        "kneeRFlex": 0.05,
        "kneeLFlex": 0.85,
        "ankleRFlex": -0.1,
        "ankleLFlex": 0.2,
        "ankleLInvert": 0.1,
        "fingerRSpread": 0.8, "fingerLSpread": 0.3,
        "toeRCurl": 0.2,
    }

    # Développé — standing on right, left leg unfolds to front
    developpe = {
        **neutral,
        "spineFlex": -0.08, "spineLatBend": 0.05,
        "shoulderRAbduct": 0.6, "shoulderRFlex": 0.15,
        "shoulderLAbduct": 0.5, "shoulderLFlex": 0.3,
        "elbowRFlex": 0.15, "elbowLFlex": 0.2,
        "forearmRRotate": 0.1, "forearmLRotate": -0.1,
        "wristRFlex": -0.1, "wristLFlex": -0.15,
        "hipRFlex": 0.05,  # standing
        "hipLFlex": 0.65,  # extended front
        "hipLAbduct": 0.2,
        "kneeRFlex": 0.05,
        "kneeLFlex": 0.05,  # straight
        "ankleRFlex": 0.1,  # relevé
        "ankleLFlex": -0.2,  # pointed
        "fingerRSpread": 0.5, "fingerLSpread": 0.5,
        "toeRCurl": 0.3,
    }

    # Promenade — turning with extended leg, arms in crown
    promenade = {
        **neutral,
        "spineFlex": -0.05, "spineLatBend": -0.05, "spineRotation": 0.1,
        "shoulderRAbduct": 0.75, "shoulderRFlex": 0.55,
        "shoulderRRotate": -0.2,
        "shoulderLAbduct": 0.75, "shoulderLFlex": 0.55,
        "shoulderLRotate": 0.2,
        "elbowRFlex": 0.3, "elbowLFlex": 0.3,
        "forearmRRotate": 0.3, "forearmLRotate": -0.3,
        "wristRFlex": -0.1, "wristLFlex": -0.1,
        "hipRFlex": 0.05,
        "hipLFlex": 0.5,
        "hipLAbduct": 0.25,
        "kneeRFlex": 0.05,
        "kneeLFlex": 0.08,
        "ankleRFlex": 0.12,
        "ankleLFlex": -0.2,
        "fingerRSpread": 0.6, "fingerLSpread": 0.6,
    }

    # Low — sitting back into a deep fold
    low_fold = {
        **neutral,
        "spineFlex": 0.6, "spineLatBend": 0.1, "spineRotation": -0.15,
        "shoulderRAbduct": 0.2, "shoulderRFlex": 0.4,
        "shoulderLAbduct": 0.15, "shoulderLFlex": 0.3,
        "elbowRFlex": 0.35, "elbowLFlex": 0.4,
        "forearmRRotate": 0.2, "forearmLRotate": -0.2,
        "wristRFlex": 0.2, "wristLFlex": 0.2,
        "hipRFlex": 0.8, "hipLFlex": 0.7,
        "hipRAbduct": 0.15, "hipLAbduct": 0.1,
        "kneeRFlex": 0.9, "kneeLFlex": 0.85,
        "ankleRFlex": 0.15, "ankleLFlex": 0.15,
        "ankleRInvert": 0.1, "ankleLInvert": -0.1,
        "fingerRCurl": 0.6, "fingerLCurl": 0.5,
        "toeRCurl": 0.4, "toeLCurl": 0.4,
    }

    # Expansive reach — full extension, arms spread wide and high
    expansive = {
        **neutral,
        "spineFlex": -0.15, "spineLatBend": 0, "spineRotation": 0,
        "shoulderRAbduct": 0.95, "shoulderRFlex": 0.4,
        "shoulderRRotate": -0.2,
        "shoulderLAbduct": 0.95, "shoulderLFlex": 0.4,
        "shoulderLRotate": 0.2,
        "elbowRFlex": 0.05, "elbowLFlex": 0.05,
        "forearmRRotate": 0.2, "forearmLRotate": -0.2,
        "wristRFlex": -0.2, "wristLFlex": -0.2,
        "wristRDeviate": -0.15, "wristLDeviate": 0.15,
        "hipRFlex": 0, "hipLFlex": 0,
        "kneeRFlex": 0.02, "kneeLFlex": 0.02,
        "ankleRFlex": 0.1, "ankleLFlex": 0.1,  # relevé
        "fingerRSpread": 0.9, "fingerLSpread": 0.9,
        "fingerRCurl": 0, "fingerLCurl": 0,
        "thumbROp": 0.3, "thumbLOp": 0.3,
        "toeRCurl": 0.3, "toeLCurl": 0.3,
    }

    # Final turn — one arm high, other across body
    final_turn = {
        **neutral,
        "spineFlex": 0, "spineLatBend": 0.1, "spineRotation": 0.2,
        "shoulderRAbduct": 0.85, "shoulderRFlex": 0.6,
        "shoulderRRotate": -0.15,
        "shoulderLAbduct": 0.2, "shoulderLFlex": 0.3,
        "shoulderLRotate": 0.3,
        "elbowRFlex": 0.15, "elbowLFlex": 0.4,
        "forearmRRotate": 0.25, "forearmLRotate": 0.1,
        "wristRFlex": -0.1,
        "hipRFlex": 0.05, "hipLFlex": 0.15,
        "kneeRFlex": 0.08, "kneeLFlex": 0.2,
        "ankleRFlex": 0.08,
        "fingerRSpread": 0.7, "fingerLSpread": 0.3,
        "fingerLCurl": 0.3,
    }

    # ── Facial expression presets ──
    serene = {"AU5": 0.15, "AU6": 0.2, "AU12": 0.3}   # gentle smile
    focused = {"AU5": 0.25, "AU4": 0.1, "AU1": 0.15}   # intense focus
    joyful = {"AU6": 0.5, "AU12": 0.5, "AU5": 0.2}     # broad smile
    yearning = {"AU1": 0.4, "AU5": 0.3, "AU25": 0.2}   # open, reaching
    composed = {"AU5": 0.1, "AU12": 0.1}                # quiet

    # ── Head rotation presets ──
    head_center = {"headYaw": 0, "headPitch": 0, "headRoll": 0}
    head_tilt_r = {"headYaw": -0.15, "headPitch": 0.1, "headRoll": -0.15}
    head_tilt_l = {"headYaw": 0.15, "headPitch": 0.1, "headRoll": 0.15}
    head_up = {"headYaw": 0, "headPitch": -0.3, "headRoll": 0}
    head_down_r = {"headYaw": -0.2, "headPitch": 0.25, "headRoll": -0.1}
    head_look_r = {"headYaw": -0.35, "headPitch": 0.05, "headRoll": -0.05}
    head_look_l = {"headYaw": 0.35, "headPitch": 0.05, "headRoll": 0.05}

    keyframes = [
        # ── 0s: Opening — neutral, centered, quiet ──
        AnimationKeyframe(
            time=0.0,
            wrapper_position=(0, SY, 0),
            wrapper_quaternion=face_z,
            body_state=neutral.copy(),
            face_aus=composed,
            head_rotation=head_center,
            easing="linear",
        ),
        # ── 2s: Breath in — subtle lift ──
        AnimationKeyframe(
            time=2.0,
            wrapper_position=(0, SY + 1, 0),
            wrapper_quaternion=face_z,
            body_state={**neutral, "spineFlex": -0.03,
                        "shoulderRAbduct": 0.15, "shoulderLAbduct": 0.15},
            face_aus=serene,
            head_rotation={"headYaw": 0, "headPitch": -0.1, "headRoll": 0},
            easing="ease_in_out",
        ),
        # ── 4s: Arms float up ──
        AnimationKeyframe(
            time=4.0,
            wrapper_position=(0, SY + 1, 0),
            wrapper_quaternion=face_z,
            body_state=arms_rise.copy(),
            face_aus=serene,
            head_rotation=head_up,
            easing="ease_in_out",
        ),
        # ── 6s: Port de bras 1 — right arm high ──
        AnimationKeyframe(
            time=6.0,
            wrapper_position=(5, SY, 5),
            wrapper_quaternion=_q_turn(-10),
            body_state=port_de_bras_1.copy(),
            face_aus=serene,
            head_rotation=head_tilt_r,
            easing="ease_in_out",
        ),
        # ── 8s: Port de bras 2 — left arm high, counter-sway ──
        AnimationKeyframe(
            time=8.0,
            wrapper_position=(-5, SY, 5),
            wrapper_quaternion=_q_turn(10),
            body_state=port_de_bras_2.copy(),
            face_aus=serene,
            head_rotation=head_tilt_l,
            easing="ease_in_out",
        ),
        # ── 10s: Weight shift right — preparing arabesque ──
        AnimationKeyframe(
            time=10.0,
            wrapper_position=(10, SY, 0),
            wrapper_quaternion=_q_turn(-15),
            body_state=weight_shift_r.copy(),
            face_aus=focused,
            head_rotation=head_look_r,
            easing="ease_in_out",
        ),
        # ── 13s: Arabesque — full extension, hold ──
        AnimationKeyframe(
            time=13.0,
            wrapper_position=(15, SY + 2, -5),
            wrapper_quaternion=_q_turn(-25),
            body_state=arabesque.copy(),
            face_aus=yearning,
            head_rotation=head_up,
            easing="ease_in_out",
        ),
        # ── 15s: Hold arabesque peak ──
        AnimationKeyframe(
            time=15.0,
            wrapper_position=(15, SY + 2, -5),
            wrapper_quaternion=_q_turn(-25),
            body_state=arabesque.copy(),
            face_aus=yearning,
            head_rotation=head_up,
            easing="ease_in_out",
        ),
        # ── 17s: Recovery — land, travelling step right ──
        AnimationKeyframe(
            time=17.0,
            wrapper_position=(30, SY, 10),
            wrapper_quaternion=_q_turn(-30),
            body_state=travel_step.copy(),
            face_aus=serene,
            head_rotation=head_tilt_r,
            easing="ease_out",
        ),
        # ── 19s: Continue travelling ──
        AnimationKeyframe(
            time=19.0,
            wrapper_position=(40, SY, 15),
            wrapper_quaternion=_q_turn(-15),
            body_state={**travel_step,
                        "hipRFlex": 0.25, "hipLFlex": -0.05,
                        "kneeRFlex": 0.35, "kneeLFlex": 0.05,
                        "shoulderRFlex": 0.3, "shoulderLFlex": -0.1},
            face_aus=serene,
            head_rotation=head_center,
            easing="ease_in_out",
        ),
        # ── 21s: Floor reach — deep forward bend ──
        AnimationKeyframe(
            time=21.0,
            wrapper_position=(30, SY - 5, 10),
            wrapper_quaternion=_q_turn(0),
            body_state=floor_reach.copy(),
            face_aus=focused,
            head_rotation=head_down_r,
            easing="ease_in_out",
        ),
        # ── 23s: Contraction — pull center ──
        AnimationKeyframe(
            time=23.0,
            wrapper_position=(20, SY - 3, 5),
            wrapper_quaternion=_q_turn(10),
            body_state=contraction.copy(),
            face_aus=focused,
            head_rotation={"headYaw": 0.1, "headPitch": 0.2, "headRoll": 0},
            easing="ease_in_out",
        ),
        # ── 25s: Unwind from contraction ──
        AnimationKeyframe(
            time=25.0,
            wrapper_position=(10, SY, 0),
            wrapper_quaternion=_q_turn(20),
            body_state={**neutral,
                        "spineFlex": 0.1, "spineRotation": 0.2,
                        "shoulderRAbduct": 0.4, "shoulderLAbduct": 0.3,
                        "elbowRFlex": 0.2, "elbowLFlex": 0.25,
                        "hipRFlex": 0.1, "hipLFlex": 0.1,
                        "kneeRFlex": 0.15, "kneeLFlex": 0.15},
            face_aus=serene,
            head_rotation=head_look_l,
            easing="ease_in_out",
        ),
        # ── 27s: Spiral prep — wind up for turn ──
        AnimationKeyframe(
            time=27.0,
            wrapper_position=(5, SY, 0),
            wrapper_quaternion=_q_turn(30),
            body_state=spiral_prep.copy(),
            face_aus=focused,
            head_rotation=head_look_l,
            easing="ease_in",
        ),
        # ── 29s: Turn — 180° through ──
        AnimationKeyframe(
            time=29.0,
            wrapper_position=(0, SY + 2, 0),
            wrapper_quaternion=_q_turn(180),
            body_state=turn_pose.copy(),
            face_aus=joyful,
            head_rotation=head_center,
            easing="ease_in_out",
        ),
        # ── 31s: Complete turn — 360° (back to front) ──
        AnimationKeyframe(
            time=31.0,
            wrapper_position=(0, SY + 2, 0),
            wrapper_quaternion=_q_turn(360),
            body_state=turn_pose.copy(),
            face_aus=joyful,
            head_rotation=head_center,
            easing="ease_out",
        ),
        # ── 33s: Landing — absorb turn momentum ──
        AnimationKeyframe(
            time=33.0,
            wrapper_position=(0, SY - 2, 5),
            wrapper_quaternion=_q_turn(360 + 15),
            body_state=landing_ripple.copy(),
            face_aus=serene,
            head_rotation=head_tilt_l,
            easing="ease_out",
        ),
        # ── 35s: Recovery to neutral ──
        AnimationKeyframe(
            time=35.0,
            wrapper_position=(-5, SY, 5),
            wrapper_quaternion=_q_turn(360 + 10),
            body_state={**neutral,
                        "shoulderRAbduct": 0.3, "shoulderLAbduct": 0.3,
                        "elbowRFlex": 0.15, "elbowLFlex": 0.15},
            face_aus=serene,
            head_rotation=head_center,
            easing="ease_in_out",
        ),
        # ── 37s: Deep lunge left — dramatic reach ──
        AnimationKeyframe(
            time=37.0,
            wrapper_position=(-20, SY - 10, 0),
            wrapper_quaternion=_q_turn(360 - 10),
            body_state=lunge_left.copy(),
            face_aus=yearning,
            head_rotation=head_tilt_r,
            easing="ease_in_out",
        ),
        # ── 39s: Hold lunge ──
        AnimationKeyframe(
            time=39.0,
            wrapper_position=(-25, SY - 12, -5),
            wrapper_quaternion=_q_turn(360 - 15),
            body_state=lunge_left.copy(),
            face_aus=yearning,
            head_rotation={"headYaw": -0.25, "headPitch": -0.15, "headRoll": -0.1},
            easing="ease_in_out",
        ),
        # ── 41s: Rise from lunge — développé preparation ──
        AnimationKeyframe(
            time=41.0,
            wrapper_position=(-10, SY, 0),
            wrapper_quaternion=_q_turn(360),
            body_state={**neutral,
                        "spineFlex": -0.05,
                        "shoulderRAbduct": 0.5, "shoulderLAbduct": 0.5,
                        "shoulderRFlex": 0.2, "shoulderLFlex": 0.2,
                        "elbowRFlex": 0.15, "elbowLFlex": 0.15,
                        "hipRFlex": 0.05, "hipLFlex": 0.2,
                        "kneeLFlex": 0.5,
                        "ankleRFlex": 0.08},
            face_aus=focused,
            head_rotation=head_center,
            easing="ease_in_out",
        ),
        # ── 43s: Développé — leg unfolds to front ──
        AnimationKeyframe(
            time=43.0,
            wrapper_position=(-5, SY + 2, 0),
            wrapper_quaternion=_q_turn(360 + 10),
            body_state=developpe.copy(),
            face_aus=joyful,
            head_rotation=head_tilt_l,
            easing="ease_in_out",
        ),
        # ── 45s: Hold développé ──
        AnimationKeyframe(
            time=45.0,
            wrapper_position=(-5, SY + 2, 0),
            wrapper_quaternion=_q_turn(360 + 10),
            body_state=developpe.copy(),
            face_aus=joyful,
            head_rotation=head_look_l,
            easing="ease_in_out",
        ),
        # ── 47s: Promenade — turning with leg extended ──
        AnimationKeyframe(
            time=47.0,
            wrapper_position=(0, SY + 2, 0),
            wrapper_quaternion=_q_turn(360 + 90),
            body_state=promenade.copy(),
            face_aus=serene,
            head_rotation=head_center,
            easing="ease_in_out",
        ),
        # ── 49s: Descent — lower to floor ──
        AnimationKeyframe(
            time=49.0,
            wrapper_position=(5, SY - 15, 5),
            wrapper_quaternion=_q_turn(360 + 90),
            body_state=low_fold.copy(),
            face_aus=composed,
            head_rotation=head_down_r,
            easing="ease_in_out",
        ),
        # ── 51s: Floor moment — deep fold, stillness ──
        AnimationKeyframe(
            time=51.0,
            wrapper_position=(5, SY - 18, 5),
            wrapper_quaternion=_q_turn(360 + 80),
            body_state={**low_fold,
                        "spineFlex": 0.7,
                        "shoulderRFlex": 0.5, "shoulderLFlex": 0.35,
                        "wristRFlex": 0.3, "wristLFlex": 0.25},
            face_aus=composed,
            head_rotation={"headYaw": -0.15, "headPitch": 0.3, "headRoll": -0.1},
            easing="ease_in_out",
        ),
        # ── 53s: Begin rise — unfurl from floor ──
        AnimationKeyframe(
            time=53.0,
            wrapper_position=(5, SY - 5, 5),
            wrapper_quaternion=_q_turn(360 + 60),
            body_state={**neutral,
                        "spineFlex": 0.3,
                        "shoulderRAbduct": 0.4, "shoulderLAbduct": 0.35,
                        "shoulderRFlex": 0.3, "shoulderLFlex": 0.25,
                        "elbowRFlex": 0.2, "elbowLFlex": 0.25,
                        "hipRFlex": 0.3, "hipLFlex": 0.3,
                        "kneeRFlex": 0.4, "kneeLFlex": 0.4,
                        "ankleRFlex": 0.1, "ankleLFlex": 0.1},
            face_aus=serene,
            head_rotation=head_center,
            easing="ease_in",
        ),
        # ── 55s: Full rise — expansive reach ──
        AnimationKeyframe(
            time=55.0,
            wrapper_position=(0, SY + 3, 0),
            wrapper_quaternion=_q_turn(360 + 30),
            body_state=expansive.copy(),
            face_aus=joyful,
            head_rotation=head_up,
            easing="ease_in_out",
        ),
        # ── 57s: Hold expansive — peak moment ──
        AnimationKeyframe(
            time=57.0,
            wrapper_position=(0, SY + 3, 0),
            wrapper_quaternion=_q_turn(360 + 15),
            body_state=expansive.copy(),
            face_aus=joyful,
            head_rotation={"headYaw": 0, "headPitch": -0.25, "headRoll": 0.05},
            easing="ease_in_out",
        ),
        # ── 59s: Final turn — arm high, coming back to front ──
        AnimationKeyframe(
            time=59.0,
            wrapper_position=(0, SY + 1, 0),
            wrapper_quaternion=_q_turn(720 - 30),
            body_state=final_turn.copy(),
            face_aus=serene,
            head_rotation=head_tilt_r,
            easing="ease_in_out",
        ),
        # ── 61s: Settling — arms lowering ──
        AnimationKeyframe(
            time=61.0,
            wrapper_position=(0, SY, 0),
            wrapper_quaternion=_q_turn(720),
            body_state={**neutral,
                        "shoulderRAbduct": 0.2, "shoulderLAbduct": 0.2,
                        "shoulderRFlex": 0.1, "shoulderLFlex": 0.1,
                        "elbowRFlex": 0.15, "elbowLFlex": 0.15,
                        "fingerRSpread": 0.3, "fingerLSpread": 0.3},
            face_aus=composed,
            head_rotation={"headYaw": 0, "headPitch": -0.05, "headRoll": 0},
            easing="ease_in_out",
        ),
        # ── 64s: Return to opening pose (seamless loop point) ──
        AnimationKeyframe(
            time=64.0,
            wrapper_position=(0, SY, 0),
            wrapper_quaternion=_q_turn(720),
            body_state=neutral.copy(),
            face_aus=composed,
            head_rotation=head_center,
            easing="ease_in_out",
        ),
    ]
    return AnimationClip(name="Dance", keyframes=keyframes, loop=True)


# ── Shared neutral base for all dance clips ──────────────────────────

def _neutral() -> dict:
    """Baseline neutral standing pose — all joints near zero."""
    return {
        "spineFlex": 0.03, "spineLatBend": 0, "spineRotation": 0,
        "shoulderRAbduct": -0.08, "shoulderRFlex": 0.03, "shoulderRRotate": 0,
        "shoulderLAbduct": -0.08, "shoulderLFlex": 0.03, "shoulderLRotate": 0,
        "elbowRFlex": 0.1, "elbowLFlex": 0.1,
        "forearmRRotate": 0, "forearmLRotate": 0,
        "wristRFlex": 0, "wristRDeviate": 0,
        "wristLFlex": 0, "wristLDeviate": 0,
        "hipRFlex": 0.02, "hipRAbduct": 0, "hipRRotate": 0,
        "hipLFlex": 0.02, "hipLAbduct": 0, "hipLRotate": 0,
        "kneeRFlex": 0.03, "kneeLFlex": 0.03,
        "ankleRFlex": 0, "ankleLFlex": 0,
        "ankleRInvert": 0, "ankleLInvert": 0,
        "fingerRCurl": 0.15, "fingerLCurl": 0.15,
        "fingerRSpread": 0.2, "fingerLSpread": 0.2,
        "thumbROp": 0.1, "thumbLOp": 0.1,
        "toeRCurl": 0, "toeLCurl": 0,
    }


def _pose(overrides: dict) -> dict:
    """Create a pose by overlaying *overrides* onto neutral."""
    p = _neutral()
    p.update(overrides)
    return p


# ── Tap Dance ─────────────────────────────────────────────────────────

def make_tap_dance_clip() -> AnimationClip:
    """Create a ~48s looping tap dance routine.

    Characterised by fast, percussive footwork with the upper body staying
    relatively upright and arms loose.  Includes shuffles, flaps, buffalo
    turns, time steps, and a Shim-Sham finale.

    Choreographic structure:
      0-2s    Preparation: weight settle, arms relaxed
      2-6s    Basic time step (R lead)
      6-10s   Shuffle-ball-change combos
      10-14s  Traveling flaps stage right
      14-18s  Cramp rolls + stamps
      18-22s  Buffalo turn (360°)
      22-26s  Paddle turns
      26-30s  Wings (jump with feet clicking)
      30-34s  Pullbacks + toe stands
      34-38s  Maxie Ford combinations
      38-42s  Shim-Sham break
      42-46s  Final flourish + freeze
      46-48s  Return to opening (loop)
    """
    SY = _STAND_Y
    face_z = _q_turn(0)

    # Face presets
    cheerful = {"AU6": 0.4, "AU12": 0.5, "AU5": 0.15}
    grin = {"AU6": 0.6, "AU12": 0.7, "AU25": 0.2}
    cool = {"AU12": 0.2, "AU5": 0.1}
    wink_r = {"AU6": 0.3, "AU12": 0.4, "AU46": 0.8}  # right eye close

    head_c = {"headYaw": 0, "headPitch": 0, "headRoll": 0}
    head_nod = {"headYaw": 0, "headPitch": 0.1, "headRoll": 0}
    head_tilt = {"headYaw": -0.1, "headPitch": 0, "headRoll": -0.1}

    # Time step: stomp R, hop L, step R, step L (quick feet, loose arms)
    time_step_1 = _pose({
        "hipRFlex": 0.15, "kneeRFlex": 0.3, "ankleRFlex": 0.15,
        "hipLFlex": 0.05, "kneeLFlex": 0.1,
        "shoulderRAbduct": 0.1, "shoulderLAbduct": 0.1,
        "elbowRFlex": 0.3, "elbowLFlex": 0.25,
        "toeRCurl": 0.4,
    })
    time_step_2 = _pose({
        "hipLFlex": 0.15, "kneeLFlex": 0.3, "ankleLFlex": 0.15,
        "hipRFlex": 0.05, "kneeRFlex": 0.1,
        "shoulderRAbduct": 0.1, "shoulderLAbduct": 0.1,
        "elbowRFlex": 0.25, "elbowLFlex": 0.3,
        "toeLCurl": 0.4,
    })

    # Shuffle: quick brush forward + back
    shuffle_r = _pose({
        "hipRFlex": 0.25, "kneeRFlex": 0.15, "ankleRFlex": -0.15,
        "hipLFlex": 0.08, "kneeLFlex": 0.2,
        "shoulderRAbduct": 0.15, "shoulderLAbduct": 0.15,
        "elbowRFlex": 0.2, "elbowLFlex": 0.2,
        "spineRotation": 0.05,
    })
    shuffle_l = _pose({
        "hipLFlex": 0.25, "kneeLFlex": 0.15, "ankleLFlex": -0.15,
        "hipRFlex": 0.08, "kneeRFlex": 0.2,
        "shoulderRAbduct": 0.15, "shoulderLAbduct": 0.15,
        "elbowRFlex": 0.2, "elbowLFlex": 0.2,
        "spineRotation": -0.05,
    })

    # Flap: brush forward + step down (travelling)
    flap_r = _pose({
        "hipRFlex": 0.3, "kneeRFlex": 0.1, "ankleRFlex": -0.2,
        "hipLFlex": 0.05, "kneeLFlex": 0.15,
        "shoulderRAbduct": 0.2, "shoulderLAbduct": 0.1,
        "elbowRFlex": 0.15, "elbowLFlex": 0.2,
        "spineRotation": 0.08,
    })
    flap_l = _pose({
        "hipLFlex": 0.3, "kneeLFlex": 0.1, "ankleLFlex": -0.2,
        "hipRFlex": 0.05, "kneeRFlex": 0.15,
        "shoulderLAbduct": 0.2, "shoulderRAbduct": 0.1,
        "elbowLFlex": 0.15, "elbowRFlex": 0.2,
        "spineRotation": -0.08,
    })

    # Cramp roll: toe-toe-heel-heel
    cramp_high = _pose({
        "ankleRFlex": 0.2, "ankleLFlex": 0.2,
        "kneeRFlex": 0.15, "kneeLFlex": 0.15,
        "toeRCurl": 0.5, "toeLCurl": 0.5,
        "shoulderRAbduct": 0.2, "shoulderLAbduct": 0.2,
        "elbowRFlex": 0.3, "elbowLFlex": 0.3,
    })
    cramp_stamp = _pose({
        "ankleRFlex": -0.1, "ankleLFlex": -0.1,
        "kneeRFlex": 0.25, "kneeLFlex": 0.25,
        "hipRFlex": 0.1, "hipLFlex": 0.1,
        "shoulderRAbduct": 0.15, "shoulderLAbduct": 0.15,
    })

    # Buffalo turn: shuffle into a turn
    buffalo_prep = _pose({
        "spineRotation": 0.2,
        "shoulderRAbduct": 0.3, "shoulderLAbduct": 0.2,
        "elbowRFlex": 0.2, "elbowLFlex": 0.3,
        "hipRFlex": 0.15, "kneeRFlex": 0.25,
        "hipLFlex": 0.1, "kneeLFlex": 0.15,
    })
    buffalo_air = _pose({
        "hipRFlex": 0.2, "hipLFlex": 0.2,
        "kneeRFlex": 0.35, "kneeLFlex": 0.35,
        "ankleRFlex": -0.15, "ankleLFlex": -0.15,
        "shoulderRAbduct": 0.4, "shoulderLAbduct": 0.4,
        "elbowRFlex": 0.15, "elbowLFlex": 0.15,
        "toeRCurl": 0.3, "toeLCurl": 0.3,
    })

    # Wing: jump, feet click mid-air
    wing_jump = _pose({
        "hipRAbduct": 0.25, "hipLAbduct": 0.25,
        "hipRFlex": 0.1, "hipLFlex": 0.1,
        "kneeRFlex": 0.2, "kneeLFlex": 0.2,
        "ankleRInvert": -0.2, "ankleLInvert": 0.2,
        "shoulderRAbduct": 0.5, "shoulderLAbduct": 0.5,
        "elbowRFlex": 0.1, "elbowLFlex": 0.1,
        "fingerRSpread": 0.6, "fingerLSpread": 0.6,
    })
    wing_land = _pose({
        "hipRFlex": 0.15, "hipLFlex": 0.15,
        "kneeRFlex": 0.35, "kneeLFlex": 0.35,
        "ankleRFlex": 0.1, "ankleLFlex": 0.1,
        "shoulderRAbduct": 0.2, "shoulderLAbduct": 0.2,
        "elbowRFlex": 0.25, "elbowLFlex": 0.25,
    })

    # Toe stand (pullback landing)
    toe_stand = _pose({
        "ankleRFlex": 0.25, "ankleLFlex": 0.25,
        "kneeRFlex": 0.02, "kneeLFlex": 0.02,
        "spineFlex": -0.05,
        "shoulderRAbduct": 0.3, "shoulderLAbduct": 0.3,
        "shoulderRFlex": 0.2, "shoulderLFlex": 0.2,
        "elbowRFlex": 0.15, "elbowLFlex": 0.15,
        "toeRCurl": 0.6, "toeLCurl": 0.6,
        "fingerRSpread": 0.5, "fingerLSpread": 0.5,
    })

    # Shim-sham: classic 8-count break
    shimsham_1 = _pose({
        "hipRFlex": 0.2, "kneeRFlex": 0.25,
        "hipLFlex": 0.05, "kneeLFlex": 0.1,
        "ankleRFlex": -0.1, "ankleLFlex": 0.05,
        "shoulderRAbduct": 0.25, "shoulderLAbduct": 0.2,
        "elbowRFlex": 0.35, "elbowLFlex": 0.2,
        "spineRotation": 0.1, "spineLatBend": -0.05,
    })
    shimsham_2 = _pose({
        "hipLFlex": 0.2, "kneeLFlex": 0.25,
        "hipRFlex": 0.05, "kneeRFlex": 0.1,
        "ankleLFlex": -0.1, "ankleRFlex": 0.05,
        "shoulderLAbduct": 0.25, "shoulderRAbduct": 0.2,
        "elbowLFlex": 0.35, "elbowRFlex": 0.2,
        "spineRotation": -0.1, "spineLatBend": 0.05,
    })

    # Final freeze: one foot forward, arms spread
    freeze = _pose({
        "hipRFlex": 0.15, "kneeRFlex": 0.1,
        "hipLFlex": -0.05, "kneeLFlex": 0.02,
        "ankleRFlex": 0.05,
        "shoulderRAbduct": 0.7, "shoulderRFlex": 0.15,
        "shoulderLAbduct": 0.5, "shoulderLFlex": -0.1,
        "elbowRFlex": 0.15, "elbowLFlex": 0.2,
        "fingerRSpread": 0.8, "fingerLSpread": 0.7,
        "spineFlex": -0.05, "spineLatBend": 0.08,
    })

    keyframes = [
        # 0s: Start neutral
        AnimationKeyframe(time=0.0, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=face_z, body_state=_neutral(),
                          face_aus=cool, head_rotation=head_c, easing="linear"),
        # 2s: Settle into groove
        AnimationKeyframe(time=2.0, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=face_z, body_state=time_step_1,
                          face_aus=cheerful, head_rotation=head_nod, easing="ease_in_out"),
        # 3s: Time step kick L
        AnimationKeyframe(time=3.0, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=face_z, body_state=time_step_2,
                          face_aus=cheerful, head_rotation=head_c, easing="ease_in_out"),
        # 4s: Time step R
        AnimationKeyframe(time=4.0, wrapper_position=(2, SY, 0),
                          wrapper_quaternion=face_z, body_state=time_step_1,
                          face_aus=cheerful, head_rotation=head_nod, easing="ease_in_out"),
        # 5s: Time step L
        AnimationKeyframe(time=5.0, wrapper_position=(2, SY, 0),
                          wrapper_quaternion=face_z, body_state=time_step_2,
                          face_aus=grin, head_rotation=head_c, easing="ease_in_out"),
        # 6.5s: Shuffle R
        AnimationKeyframe(time=6.5, wrapper_position=(5, SY, 0),
                          wrapper_quaternion=_q_turn(-5), body_state=shuffle_r,
                          face_aus=cheerful, head_rotation=head_tilt, easing="ease_in_out"),
        # 7.5s: Shuffle L
        AnimationKeyframe(time=7.5, wrapper_position=(5, SY, 0),
                          wrapper_quaternion=_q_turn(5), body_state=shuffle_l,
                          face_aus=cheerful, head_rotation=head_c, easing="ease_in_out"),
        # 8.5s: Shuffle R again
        AnimationKeyframe(time=8.5, wrapper_position=(8, SY, 0),
                          wrapper_quaternion=_q_turn(-5), body_state=shuffle_r,
                          face_aus=grin, head_rotation=head_tilt, easing="ease_in_out"),
        # 9.5s: Ball change
        AnimationKeyframe(time=9.5, wrapper_position=(8, SY, 0),
                          wrapper_quaternion=face_z, body_state=time_step_2,
                          face_aus=cheerful, head_rotation=head_c, easing="ease_in_out"),
        # 11s: Flap R traveling
        AnimationKeyframe(time=11.0, wrapper_position=(20, SY, 5),
                          wrapper_quaternion=_q_turn(-10), body_state=flap_r,
                          face_aus=cheerful, head_rotation=head_tilt, easing="ease_in_out"),
        # 12s: Flap L traveling
        AnimationKeyframe(time=12.0, wrapper_position=(30, SY, 5),
                          wrapper_quaternion=_q_turn(-5), body_state=flap_l,
                          face_aus=grin, head_rotation=head_c, easing="ease_in_out"),
        # 13s: Flap R traveling
        AnimationKeyframe(time=13.0, wrapper_position=(40, SY, 5),
                          wrapper_quaternion=_q_turn(-10), body_state=flap_r,
                          face_aus=cheerful, head_rotation=head_tilt, easing="ease_in_out"),
        # 14.5s: Cramp roll high
        AnimationKeyframe(time=14.5, wrapper_position=(40, SY + 2, 0),
                          wrapper_quaternion=face_z, body_state=cramp_high,
                          face_aus=grin, head_rotation=head_nod, easing="ease_in"),
        # 15.5s: Cramp stamp
        AnimationKeyframe(time=15.5, wrapper_position=(40, SY - 2, 0),
                          wrapper_quaternion=face_z, body_state=cramp_stamp,
                          face_aus=cheerful, head_rotation=head_c, easing="ease_out"),
        # 16.5s: Cramp roll high again
        AnimationKeyframe(time=16.5, wrapper_position=(40, SY + 2, 0),
                          wrapper_quaternion=face_z, body_state=cramp_high,
                          face_aus=grin, head_rotation=head_nod, easing="ease_in"),
        # 17.5s: Stamp!
        AnimationKeyframe(time=17.5, wrapper_position=(40, SY - 2, 0),
                          wrapper_quaternion=face_z, body_state=cramp_stamp,
                          face_aus=grin, head_rotation=head_c, easing="ease_out"),
        # 18.5s: Buffalo prep — wind up
        AnimationKeyframe(time=18.5, wrapper_position=(35, SY, 0),
                          wrapper_quaternion=_q_turn(15), body_state=buffalo_prep,
                          face_aus=cheerful, head_rotation=head_c, easing="ease_in"),
        # 20s: Buffalo mid-turn 180°
        AnimationKeyframe(time=20.0, wrapper_position=(30, SY + 3, 0),
                          wrapper_quaternion=_q_turn(180), body_state=buffalo_air,
                          face_aus=grin, head_rotation=head_c, easing="ease_in_out"),
        # 21.5s: Buffalo land 360°
        AnimationKeyframe(time=21.5, wrapper_position=(25, SY - 2, 0),
                          wrapper_quaternion=_q_turn(360), body_state=wing_land,
                          face_aus=cheerful, head_rotation=head_nod, easing="ease_out"),
        # 23s: Paddle turn — step around
        AnimationKeyframe(time=23.0, wrapper_position=(20, SY, 0),
                          wrapper_quaternion=_q_turn(360 + 90), body_state=time_step_1,
                          face_aus=cool, head_rotation=head_c, easing="ease_in_out"),
        # 24.5s: Paddle 270°
        AnimationKeyframe(time=24.5, wrapper_position=(15, SY, 0),
                          wrapper_quaternion=_q_turn(360 + 270), body_state=time_step_2,
                          face_aus=cheerful, head_rotation=head_tilt, easing="ease_in_out"),
        # 25.5s: Paddle complete 360°
        AnimationKeyframe(time=25.5, wrapper_position=(10, SY, 0),
                          wrapper_quaternion=_q_turn(720), body_state=time_step_1,
                          face_aus=grin, head_rotation=head_c, easing="ease_in_out"),
        # 27s: Wing jump!
        AnimationKeyframe(time=27.0, wrapper_position=(5, SY + 8, 0),
                          wrapper_quaternion=_q_turn(720), body_state=wing_jump,
                          face_aus=grin, head_rotation=head_c, easing="ease_out"),
        # 28s: Wing land
        AnimationKeyframe(time=28.0, wrapper_position=(5, SY - 3, 0),
                          wrapper_quaternion=_q_turn(720), body_state=wing_land,
                          face_aus=cheerful, head_rotation=head_nod, easing="ease_out"),
        # 29.5s: Second wing
        AnimationKeyframe(time=29.5, wrapper_position=(0, SY + 8, 0),
                          wrapper_quaternion=_q_turn(720), body_state=wing_jump,
                          face_aus=grin, head_rotation=head_c, easing="ease_out"),
        # 30.5s: Land
        AnimationKeyframe(time=30.5, wrapper_position=(0, SY - 3, 0),
                          wrapper_quaternion=_q_turn(720), body_state=wing_land,
                          face_aus=cheerful, head_rotation=head_c, easing="ease_out"),
        # 32s: Toe stand
        AnimationKeyframe(time=32.0, wrapper_position=(0, SY + 3, 0),
                          wrapper_quaternion=_q_turn(720), body_state=toe_stand,
                          face_aus=cool, head_rotation=head_c, easing="ease_in_out"),
        # 33.5s: Down from toes
        AnimationKeyframe(time=33.5, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=_q_turn(720), body_state=time_step_1,
                          face_aus=cheerful, head_rotation=head_tilt, easing="ease_in_out"),
        # 35s: Maxie Ford R
        AnimationKeyframe(time=35.0, wrapper_position=(-5, SY, 5),
                          wrapper_quaternion=_q_turn(720 - 15), body_state=shuffle_r,
                          face_aus=cheerful, head_rotation=head_c, easing="ease_in_out"),
        # 36s: Maxie Ford L
        AnimationKeyframe(time=36.0, wrapper_position=(-10, SY, 5),
                          wrapper_quaternion=_q_turn(720 + 15), body_state=shuffle_l,
                          face_aus=grin, head_rotation=head_tilt, easing="ease_in_out"),
        # 37.5s: Back to center
        AnimationKeyframe(time=37.5, wrapper_position=(-5, SY, 0),
                          wrapper_quaternion=_q_turn(720), body_state=time_step_2,
                          face_aus=cheerful, head_rotation=head_c, easing="ease_in_out"),
        # 39s: Shim-sham R
        AnimationKeyframe(time=39.0, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=_q_turn(720), body_state=shimsham_1,
                          face_aus=grin, head_rotation=head_nod, easing="ease_in_out"),
        # 40s: Shim-sham L
        AnimationKeyframe(time=40.0, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=_q_turn(720), body_state=shimsham_2,
                          face_aus=grin, head_rotation=head_c, easing="ease_in_out"),
        # 41s: Shim-sham R
        AnimationKeyframe(time=41.0, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=_q_turn(720), body_state=shimsham_1,
                          face_aus=grin, head_rotation=head_nod, easing="ease_in_out"),
        # 42s: Shim-sham break — stomp!
        AnimationKeyframe(time=42.0, wrapper_position=(0, SY - 2, 0),
                          wrapper_quaternion=_q_turn(720), body_state=cramp_stamp,
                          face_aus=grin, head_rotation=head_c, easing="ease_out"),
        # 43.5s: Flourish — big finish arms
        AnimationKeyframe(time=43.5, wrapper_position=(0, SY + 2, 0),
                          wrapper_quaternion=_q_turn(720 + 15), body_state=freeze,
                          face_aus=grin, head_rotation=head_tilt, easing="ease_in_out"),
        # 45s: Hold freeze
        AnimationKeyframe(time=45.0, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=_q_turn(720), body_state=freeze,
                          face_aus=wink_r, head_rotation=head_nod, easing="ease_in_out"),
        # 47s: Settle back
        AnimationKeyframe(time=47.0, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=_q_turn(720), body_state=_neutral(),
                          face_aus=cool, head_rotation=head_c, easing="ease_in_out"),
        # 48s: Loop point
        AnimationKeyframe(time=48.0, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=_q_turn(720), body_state=_neutral(),
                          face_aus=cool, head_rotation=head_c, easing="ease_in_out"),
    ]
    return AnimationClip(name="Tap Dance", keyframes=keyframes, loop=True)


# ── Waltz ─────────────────────────────────────────────────────────────

def make_waltz_clip() -> AnimationClip:
    """Create a ~48s looping waltz (3/4 time, slow and elegant).

    Graceful, sweeping box-step pattern with an invisible partner.
    Arms held in dance frame, flowing rises and falls in 3/4 time.

    Structure (each 'measure' is ~2s at waltz tempo ≈ 90 BPM):
      0-2s    Opening — frame position, breath
      2-8s    Forward box step (R lead) x2
      8-14s   Backward box step (L lead) x2
      14-20s  Natural turn (CW 360°)
      20-26s  Reverse turn (CCW 360°)
      26-32s  Promenade — travelling side-by-side
      32-38s  Underarm turn (imaginary partner)
      38-42s  Hesitation step — suspended rise
      42-46s  Final natural turn + close
      46-48s  Return to start (loop)
    """
    SY = _STAND_Y
    face_z = _q_turn(0)

    elegant = {"AU6": 0.15, "AU12": 0.2, "AU5": 0.1}
    tender = {"AU1": 0.2, "AU6": 0.2, "AU12": 0.25}
    serene = {"AU5": 0.15, "AU12": 0.15}

    head_c = {"headYaw": 0, "headPitch": 0, "headRoll": 0}
    head_left = {"headYaw": 0.2, "headPitch": -0.05, "headRoll": 0.05}
    head_slight_l = {"headYaw": 0.1, "headPitch": 0, "headRoll": 0}

    # Dance frame — arms up as if holding partner
    frame = _pose({
        "shoulderRAbduct": 0.5, "shoulderRFlex": 0.35, "shoulderRRotate": -0.2,
        "shoulderLAbduct": 0.45, "shoulderLFlex": 0.4, "shoulderLRotate": 0.15,
        "elbowRFlex": 0.55, "elbowLFlex": 0.6,
        "forearmRRotate": 0.2, "forearmLRotate": -0.3,
        "wristRFlex": -0.05, "wristLFlex": 0.1,
        "fingerRCurl": 0.3, "fingerLCurl": 0.4,
        "spineFlex": -0.03,
    })

    # Rise: on toes, legs straight, slight lift
    frame_rise = _pose({
        **{k: v for k, v in frame.items() if k != "spineFlex"},
        "spineFlex": -0.06,
        "ankleRFlex": 0.12, "ankleLFlex": 0.12,
        "toeRCurl": 0.2, "toeLCurl": 0.2,
    })

    # Forward step R: right foot forward, left trails
    fwd_r = _pose({
        **{k: v for k, v in frame.items()},
        "hipRFlex": 0.2, "kneeRFlex": 0.05,
        "hipLFlex": -0.05, "kneeLFlex": 0.1,
        "ankleRFlex": 0.05, "ankleLFlex": -0.05,
        "spineRotation": 0.05,
    })

    # Side step R: feet apart to right
    side_r = _pose({
        **{k: v for k, v in frame.items()},
        "hipRAbduct": 0.15, "hipLAbduct": -0.05,
        "kneeRFlex": 0.08, "kneeLFlex": 0.08,
        "spineLatBend": -0.05,
    })

    # Close: feet together
    close_step = _pose({
        **{k: v for k, v in frame.items()},
        "kneeRFlex": 0.05, "kneeLFlex": 0.05,
        "ankleRFlex": 0.08, "ankleLFlex": 0.08,
    })

    # Back step L
    back_l = _pose({
        **{k: v for k, v in frame.items()},
        "hipLFlex": -0.15,
        "hipRFlex": 0.08, "kneeRFlex": 0.15,
        "ankleLFlex": -0.1,
        "spineRotation": -0.05,
    })

    # Side step L
    side_l = _pose({
        **{k: v for k, v in frame.items()},
        "hipLAbduct": 0.15, "hipRAbduct": -0.05,
        "kneeRFlex": 0.08, "kneeLFlex": 0.08,
        "spineLatBend": 0.05,
    })

    # Promenade — both facing same direction, open position
    promenade_open = _pose({
        "spineFlex": -0.03, "spineRotation": 0.15,
        "shoulderRAbduct": 0.55, "shoulderRFlex": 0.3, "shoulderRRotate": -0.2,
        "shoulderLAbduct": 0.3, "shoulderLFlex": 0.1,
        "elbowRFlex": 0.5, "elbowLFlex": 0.3,
        "forearmRRotate": 0.2,
        "hipRFlex": 0.15, "kneeRFlex": 0.1,
        "hipLFlex": 0.05, "kneeLFlex": 0.05,
        "fingerRCurl": 0.3, "fingerLCurl": 0.2,
    })

    # Underarm turn — right arm high
    underarm = _pose({
        "shoulderRAbduct": 0.9, "shoulderRFlex": 0.7,
        "shoulderLAbduct": 0.3, "shoulderLFlex": 0.2,
        "elbowRFlex": 0.2, "elbowLFlex": 0.4,
        "forearmRRotate": 0.4,
        "spineFlex": -0.05, "spineRotation": 0.1,
        "kneeRFlex": 0.05, "kneeLFlex": 0.05,
        "ankleRFlex": 0.1, "ankleLFlex": 0.1,
        "fingerRSpread": 0.5,
    })

    # Hesitation — suspended high on one foot
    hesitation = _pose({
        **{k: v for k, v in frame.items()},
        "spineFlex": -0.08,
        "ankleRFlex": 0.15,
        "hipLFlex": 0.15, "kneeLFlex": 0.2,
        "ankleLFlex": -0.15,
        "toeRCurl": 0.3,
    })

    keyframes = [
        AnimationKeyframe(time=0.0, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=face_z, body_state=frame,
                          face_aus=elegant, head_rotation=head_left, easing="linear"),
        # Box step 1: forward R (count 1)
        AnimationKeyframe(time=2.0, wrapper_position=(0, SY, 8),
                          wrapper_quaternion=face_z, body_state=fwd_r,
                          face_aus=elegant, head_rotation=head_left, easing="ease_in_out"),
        # Side R (count 2) — rise
        AnimationKeyframe(time=3.0, wrapper_position=(8, SY + 2, 8),
                          wrapper_quaternion=face_z, body_state=side_r,
                          face_aus=elegant, head_rotation=head_left, easing="ease_in_out"),
        # Close (count 3) — fall
        AnimationKeyframe(time=4.0, wrapper_position=(8, SY, 8),
                          wrapper_quaternion=face_z, body_state=close_step,
                          face_aus=elegant, head_rotation=head_left, easing="ease_in_out"),
        # Back L (count 1)
        AnimationKeyframe(time=5.0, wrapper_position=(8, SY, 0),
                          wrapper_quaternion=face_z, body_state=back_l,
                          face_aus=tender, head_rotation=head_left, easing="ease_in_out"),
        # Side L (count 2) — rise
        AnimationKeyframe(time=6.0, wrapper_position=(0, SY + 2, 0),
                          wrapper_quaternion=face_z, body_state=side_l,
                          face_aus=tender, head_rotation=head_left, easing="ease_in_out"),
        # Close (count 3) — fall
        AnimationKeyframe(time=7.0, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=face_z, body_state=close_step,
                          face_aus=elegant, head_rotation=head_left, easing="ease_in_out"),
        # Repeat box step
        AnimationKeyframe(time=8.5, wrapper_position=(0, SY, 10),
                          wrapper_quaternion=face_z, body_state=fwd_r,
                          face_aus=elegant, head_rotation=head_left, easing="ease_in_out"),
        AnimationKeyframe(time=9.5, wrapper_position=(8, SY + 2, 10),
                          wrapper_quaternion=face_z, body_state=frame_rise,
                          face_aus=tender, head_rotation=head_left, easing="ease_in_out"),
        AnimationKeyframe(time=10.5, wrapper_position=(8, SY, 5),
                          wrapper_quaternion=face_z, body_state=back_l,
                          face_aus=elegant, head_rotation=head_left, easing="ease_in_out"),
        AnimationKeyframe(time=11.5, wrapper_position=(0, SY + 2, 5),
                          wrapper_quaternion=face_z, body_state=frame_rise,
                          face_aus=elegant, head_rotation=head_left, easing="ease_in_out"),
        AnimationKeyframe(time=12.5, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=face_z, body_state=close_step,
                          face_aus=elegant, head_rotation=head_left, easing="ease_in_out"),
        # Natural turn: CW 360° in ~6s
        AnimationKeyframe(time=14.0, wrapper_position=(5, SY, 5),
                          wrapper_quaternion=_q_turn(-45), body_state=fwd_r,
                          face_aus=tender, head_rotation=head_left, easing="ease_in_out"),
        AnimationKeyframe(time=15.5, wrapper_position=(8, SY + 2, 0),
                          wrapper_quaternion=_q_turn(-135), body_state=frame_rise,
                          face_aus=elegant, head_rotation=head_slight_l, easing="ease_in_out"),
        AnimationKeyframe(time=17.0, wrapper_position=(3, SY, -5),
                          wrapper_quaternion=_q_turn(-225), body_state=back_l,
                          face_aus=tender, head_rotation=head_left, easing="ease_in_out"),
        AnimationKeyframe(time=18.5, wrapper_position=(-3, SY + 2, 0),
                          wrapper_quaternion=_q_turn(-315), body_state=frame_rise,
                          face_aus=elegant, head_rotation=head_left, easing="ease_in_out"),
        AnimationKeyframe(time=19.5, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=_q_turn(-360), body_state=close_step,
                          face_aus=elegant, head_rotation=head_left, easing="ease_in_out"),
        # Reverse turn: CCW 360°
        AnimationKeyframe(time=21.0, wrapper_position=(-5, SY, 5),
                          wrapper_quaternion=_q_turn(-360 + 45), body_state=back_l,
                          face_aus=tender, head_rotation=head_left, easing="ease_in_out"),
        AnimationKeyframe(time=22.5, wrapper_position=(-8, SY + 2, 0),
                          wrapper_quaternion=_q_turn(-360 + 135), body_state=frame_rise,
                          face_aus=elegant, head_rotation=head_slight_l, easing="ease_in_out"),
        AnimationKeyframe(time=24.0, wrapper_position=(-3, SY, -5),
                          wrapper_quaternion=_q_turn(-360 + 225), body_state=fwd_r,
                          face_aus=tender, head_rotation=head_left, easing="ease_in_out"),
        AnimationKeyframe(time=25.5, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=_q_turn(-360 + 360), body_state=close_step,
                          face_aus=elegant, head_rotation=head_left, easing="ease_in_out"),
        # Promenade
        AnimationKeyframe(time=27.0, wrapper_position=(10, SY, 10),
                          wrapper_quaternion=_q_turn(-30), body_state=promenade_open,
                          face_aus=elegant, head_rotation=head_slight_l, easing="ease_in_out"),
        AnimationKeyframe(time=29.0, wrapper_position=(25, SY, 15),
                          wrapper_quaternion=_q_turn(-30), body_state=promenade_open,
                          face_aus=tender, head_rotation=head_slight_l, easing="ease_in_out"),
        AnimationKeyframe(time=31.0, wrapper_position=(30, SY, 10),
                          wrapper_quaternion=_q_turn(0), body_state=frame,
                          face_aus=elegant, head_rotation=head_left, easing="ease_in_out"),
        # Underarm turn
        AnimationKeyframe(time=33.0, wrapper_position=(25, SY + 3, 5),
                          wrapper_quaternion=_q_turn(0), body_state=underarm,
                          face_aus=tender, head_rotation=head_c, easing="ease_in_out"),
        AnimationKeyframe(time=35.0, wrapper_position=(20, SY + 3, 0),
                          wrapper_quaternion=_q_turn(180), body_state=underarm,
                          face_aus=elegant, head_rotation=head_left, easing="ease_in_out"),
        AnimationKeyframe(time=37.0, wrapper_position=(15, SY, 0),
                          wrapper_quaternion=_q_turn(360), body_state=frame,
                          face_aus=elegant, head_rotation=head_left, easing="ease_in_out"),
        # Hesitation step
        AnimationKeyframe(time=39.0, wrapper_position=(10, SY + 3, 5),
                          wrapper_quaternion=_q_turn(360), body_state=hesitation,
                          face_aus=tender, head_rotation=head_left, easing="ease_in_out"),
        AnimationKeyframe(time=41.0, wrapper_position=(10, SY + 3, 5),
                          wrapper_quaternion=_q_turn(360), body_state=hesitation,
                          face_aus=serene, head_rotation=head_left, easing="ease_in_out"),
        # Final turn + close
        AnimationKeyframe(time=43.0, wrapper_position=(5, SY + 2, 0),
                          wrapper_quaternion=_q_turn(360 - 180), body_state=frame_rise,
                          face_aus=elegant, head_rotation=head_left, easing="ease_in_out"),
        AnimationKeyframe(time=45.0, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=_q_turn(360 - 360), body_state=close_step,
                          face_aus=tender, head_rotation=head_left, easing="ease_in_out"),
        # Settle
        AnimationKeyframe(time=47.0, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=_q_turn(0), body_state=frame,
                          face_aus=elegant, head_rotation=head_left, easing="ease_in_out"),
        AnimationKeyframe(time=48.0, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=_q_turn(0), body_state=frame,
                          face_aus=elegant, head_rotation=head_left, easing="ease_in_out"),
    ]
    return AnimationClip(name="Waltz", keyframes=keyframes, loop=True)


# ── Disco ─────────────────────────────────────────────────────────────

def make_disco_clip() -> AnimationClip:
    """Create a ~48s looping disco/funk routine.

    High-energy, rhythmic movements: pointing, hip thrusts, shoulder rolls,
    the hustle, spinning, and iconic Saturday Night Fever poses.

    Structure:
      0-2s    Groove start — bounce
      2-8s    Classic point-up / point-down
      8-14s   Hustle side steps
      14-18s  Hip rolls + shoulder shimmy
      18-22s  Travolta pointing sequence
      22-26s  Spinning on the floor
      26-30s  The bump (hip isolations)
      30-34s  Arm rolls + body wave
      34-38s  High kicks
      38-42s  Double spin
      42-46s  Final Travolta freeze
      46-48s  Return (loop)
    """
    SY = _STAND_Y
    face_z = _q_turn(0)

    excited = {"AU6": 0.5, "AU12": 0.6, "AU5": 0.2, "AU25": 0.2}
    fierce = {"AU6": 0.3, "AU12": 0.4, "AU4": 0.15, "AU5": 0.2}
    party = {"AU6": 0.6, "AU12": 0.7, "AU25": 0.3}
    cool_d = {"AU12": 0.3, "AU5": 0.15}

    head_c = {"headYaw": 0, "headPitch": 0, "headRoll": 0}
    head_groove = {"headYaw": 0, "headPitch": 0.08, "headRoll": 0}
    head_look_up = {"headYaw": 0, "headPitch": -0.25, "headRoll": 0}

    # Bounce groove — knees pumping
    bounce = _pose({
        "hipRFlex": 0.15, "hipLFlex": 0.15,
        "kneeRFlex": 0.25, "kneeLFlex": 0.25,
        "ankleRFlex": 0.05, "ankleLFlex": 0.05,
        "shoulderRAbduct": 0.15, "shoulderLAbduct": 0.15,
        "elbowRFlex": 0.4, "elbowLFlex": 0.4,
        "spineRotation": 0.05,
    })
    bounce_up = _pose({
        "hipRFlex": 0.05, "hipLFlex": 0.05,
        "kneeRFlex": 0.08, "kneeLFlex": 0.08,
        "shoulderRAbduct": 0.15, "shoulderLAbduct": 0.15,
        "elbowRFlex": 0.3, "elbowLFlex": 0.3,
    })

    # Point up (Travolta!) — right arm diagonal up
    point_up_r = _pose({
        "shoulderRAbduct": 0.85, "shoulderRFlex": 0.6,
        "shoulderLAbduct": 0.1, "shoulderLFlex": -0.1,
        "elbowRFlex": 0.05, "elbowLFlex": 0.5,
        "forearmLRotate": -0.3,
        "fingerRCurl": 0.7, "fingerRSpread": 0,
        "hipRFlex": 0.12, "hipLFlex": 0.12,
        "kneeRFlex": 0.2, "kneeLFlex": 0.2,
        "spineLatBend": 0.1, "spineRotation": 0.1,
    })

    # Point down — right arm diagonal down across body
    point_down_r = _pose({
        "shoulderRAbduct": 0.1, "shoulderRFlex": 0.4,
        "shoulderRRotate": 0.3,
        "shoulderLAbduct": 0.3, "shoulderLFlex": 0.1,
        "elbowRFlex": 0.05, "elbowLFlex": 0.4,
        "fingerRCurl": 0.7, "fingerRSpread": 0,
        "hipRFlex": 0.15, "hipLFlex": 0.15,
        "kneeRFlex": 0.25, "kneeLFlex": 0.25,
        "spineLatBend": -0.15, "spineRotation": -0.1,
    })

    # Hustle side step R
    hustle_r = _pose({
        "hipRAbduct": 0.2,
        "hipRFlex": 0.05, "hipLFlex": 0.1,
        "kneeRFlex": 0.1, "kneeLFlex": 0.2,
        "shoulderRAbduct": 0.3, "shoulderLAbduct": 0.3,
        "elbowRFlex": 0.5, "elbowLFlex": 0.5,
        "forearmRRotate": 0.3, "forearmLRotate": -0.3,
        "spineLatBend": -0.1,
    })
    hustle_l = _pose({
        "hipLAbduct": 0.2,
        "hipLFlex": 0.05, "hipRFlex": 0.1,
        "kneeLFlex": 0.1, "kneeRFlex": 0.2,
        "shoulderRAbduct": 0.3, "shoulderLAbduct": 0.3,
        "elbowRFlex": 0.5, "elbowLFlex": 0.5,
        "forearmRRotate": -0.3, "forearmLRotate": 0.3,
        "spineLatBend": 0.1,
    })

    # Hip roll — spine isolation
    hip_roll_r = _pose({
        "spineLatBend": -0.2, "spineRotation": 0.15,
        "hipRAbduct": 0.1, "hipRFlex": 0.05,
        "shoulderRAbduct": 0.2, "shoulderLAbduct": 0.2,
        "elbowRFlex": 0.4, "elbowLFlex": 0.4,
        "kneeRFlex": 0.15, "kneeLFlex": 0.2,
    })
    hip_roll_l = _pose({
        "spineLatBend": 0.2, "spineRotation": -0.15,
        "hipLAbduct": 0.1, "hipLFlex": 0.05,
        "shoulderRAbduct": 0.2, "shoulderLAbduct": 0.2,
        "elbowRFlex": 0.4, "elbowLFlex": 0.4,
        "kneeRFlex": 0.2, "kneeLFlex": 0.15,
    })

    # Body wave — sequential flex top to bottom
    wave_top = _pose({
        "spineFlex": -0.15,
        "shoulderRAbduct": 0.4, "shoulderLAbduct": 0.4,
        "shoulderRFlex": 0.3, "shoulderLFlex": 0.3,
        "elbowRFlex": 0.2, "elbowLFlex": 0.2,
        "hipRFlex": 0.02, "hipLFlex": 0.02,
        "kneeRFlex": 0.05, "kneeLFlex": 0.05,
    })
    wave_mid = _pose({
        "spineFlex": 0.15,
        "shoulderRAbduct": 0.2, "shoulderLAbduct": 0.2,
        "elbowRFlex": 0.4, "elbowLFlex": 0.4,
        "hipRFlex": 0.2, "hipLFlex": 0.2,
        "kneeRFlex": 0.1, "kneeLFlex": 0.1,
    })
    wave_bottom = _pose({
        "spineFlex": 0.05,
        "shoulderRAbduct": 0.15, "shoulderLAbduct": 0.15,
        "elbowRFlex": 0.3, "elbowLFlex": 0.3,
        "hipRFlex": 0.25, "hipLFlex": 0.25,
        "kneeRFlex": 0.3, "kneeLFlex": 0.3,
        "ankleRFlex": 0.1, "ankleLFlex": 0.1,
    })

    # High kick R
    kick_r = _pose({
        "hipRFlex": 0.7, "kneeRFlex": 0.05,
        "ankleRFlex": -0.15,
        "hipLFlex": 0.05, "kneeLFlex": 0.1,
        "shoulderRAbduct": 0.5, "shoulderLAbduct": 0.5,
        "shoulderRFlex": 0.2, "shoulderLFlex": 0.2,
        "elbowRFlex": 0.15, "elbowLFlex": 0.15,
        "spineFlex": -0.1,
    })
    kick_l = _pose({
        "hipLFlex": 0.7, "kneeLFlex": 0.05,
        "ankleLFlex": -0.15,
        "hipRFlex": 0.05, "kneeRFlex": 0.1,
        "shoulderRAbduct": 0.5, "shoulderLAbduct": 0.5,
        "shoulderRFlex": 0.2, "shoulderLFlex": 0.2,
        "elbowRFlex": 0.15, "elbowLFlex": 0.15,
        "spineFlex": -0.1,
    })

    # Travolta freeze — both arms up, weight on one leg
    travolta = _pose({
        "shoulderRAbduct": 0.9, "shoulderRFlex": 0.65,
        "shoulderLAbduct": 0.3, "shoulderLFlex": -0.1,
        "elbowRFlex": 0.05, "elbowLFlex": 0.4,
        "fingerRCurl": 0.7, "fingerRSpread": 0,
        "hipRFlex": 0.15, "hipLFlex": 0.02,
        "kneeRFlex": 0.2, "kneeLFlex": 0.02,
        "spineFlex": -0.08, "spineLatBend": 0.1,
    })

    keyframes = [
        AnimationKeyframe(time=0.0, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=face_z, body_state=_neutral(),
                          face_aus=cool_d, head_rotation=head_c, easing="linear"),
        # Groove bounce
        AnimationKeyframe(time=1.0, wrapper_position=(0, SY - 3, 0),
                          wrapper_quaternion=face_z, body_state=bounce,
                          face_aus=excited, head_rotation=head_groove, easing="ease_in_out"),
        AnimationKeyframe(time=1.5, wrapper_position=(0, SY + 1, 0),
                          wrapper_quaternion=face_z, body_state=bounce_up,
                          face_aus=excited, head_rotation=head_c, easing="ease_in_out"),
        # Point up R!
        AnimationKeyframe(time=2.5, wrapper_position=(0, SY - 2, 0),
                          wrapper_quaternion=face_z, body_state=point_up_r,
                          face_aus=party, head_rotation=head_look_up, easing="ease_in_out"),
        # Point down
        AnimationKeyframe(time=3.5, wrapper_position=(0, SY - 3, 0),
                          wrapper_quaternion=face_z, body_state=point_down_r,
                          face_aus=fierce, head_rotation=head_groove, easing="ease_in_out"),
        # Point up again
        AnimationKeyframe(time=4.5, wrapper_position=(0, SY - 1, 0),
                          wrapper_quaternion=face_z, body_state=point_up_r,
                          face_aus=party, head_rotation=head_look_up, easing="ease_in_out"),
        # Bounce
        AnimationKeyframe(time=5.5, wrapper_position=(0, SY - 3, 0),
                          wrapper_quaternion=face_z, body_state=bounce,
                          face_aus=excited, head_rotation=head_groove, easing="ease_in_out"),
        # Point up with left
        AnimationKeyframe(time=6.5, wrapper_position=(0, SY - 1, 0),
                          wrapper_quaternion=face_z,
                          body_state=_pose({
                              "shoulderLAbduct": 0.85, "shoulderLFlex": 0.6,
                              "shoulderRAbduct": 0.1, "shoulderRFlex": -0.1,
                              "elbowLFlex": 0.05, "elbowRFlex": 0.5,
                              "fingerLCurl": 0.7, "fingerLSpread": 0,
                              "hipRFlex": 0.12, "hipLFlex": 0.12,
                              "kneeRFlex": 0.2, "kneeLFlex": 0.2,
                              "spineLatBend": -0.1, "spineRotation": -0.1,
                          }),
                          face_aus=party, head_rotation=head_look_up, easing="ease_in_out"),
        # Bounce
        AnimationKeyframe(time=7.5, wrapper_position=(0, SY - 3, 0),
                          wrapper_quaternion=face_z, body_state=bounce,
                          face_aus=excited, head_rotation=head_groove, easing="ease_in_out"),
        # Hustle R
        AnimationKeyframe(time=9.0, wrapper_position=(15, SY - 2, 0),
                          wrapper_quaternion=face_z, body_state=hustle_r,
                          face_aus=excited, head_rotation=head_c, easing="ease_in_out"),
        # Hustle L
        AnimationKeyframe(time=10.0, wrapper_position=(5, SY - 2, 0),
                          wrapper_quaternion=face_z, body_state=hustle_l,
                          face_aus=excited, head_rotation=head_c, easing="ease_in_out"),
        # Hustle R
        AnimationKeyframe(time=11.0, wrapper_position=(20, SY - 2, 0),
                          wrapper_quaternion=face_z, body_state=hustle_r,
                          face_aus=party, head_rotation=head_c, easing="ease_in_out"),
        # Hustle L
        AnimationKeyframe(time=12.0, wrapper_position=(10, SY - 2, 0),
                          wrapper_quaternion=face_z, body_state=hustle_l,
                          face_aus=excited, head_rotation=head_c, easing="ease_in_out"),
        # Center
        AnimationKeyframe(time=13.0, wrapper_position=(10, SY - 3, 0),
                          wrapper_quaternion=face_z, body_state=bounce,
                          face_aus=excited, head_rotation=head_groove, easing="ease_in_out"),
        # Hip rolls
        AnimationKeyframe(time=14.5, wrapper_position=(10, SY - 2, 0),
                          wrapper_quaternion=face_z, body_state=hip_roll_r,
                          face_aus=fierce, head_rotation=head_c, easing="ease_in_out"),
        AnimationKeyframe(time=15.5, wrapper_position=(10, SY - 2, 0),
                          wrapper_quaternion=face_z, body_state=hip_roll_l,
                          face_aus=fierce, head_rotation=head_c, easing="ease_in_out"),
        AnimationKeyframe(time=16.5, wrapper_position=(10, SY - 2, 0),
                          wrapper_quaternion=face_z, body_state=hip_roll_r,
                          face_aus=party, head_rotation=head_c, easing="ease_in_out"),
        AnimationKeyframe(time=17.5, wrapper_position=(10, SY - 2, 0),
                          wrapper_quaternion=face_z, body_state=hip_roll_l,
                          face_aus=party, head_rotation=head_c, easing="ease_in_out"),
        # Travolta sequence
        AnimationKeyframe(time=19.0, wrapper_position=(5, SY - 2, 5),
                          wrapper_quaternion=_q_turn(-15), body_state=point_up_r,
                          face_aus=party, head_rotation=head_look_up, easing="ease_in_out"),
        AnimationKeyframe(time=20.0, wrapper_position=(5, SY - 3, 5),
                          wrapper_quaternion=_q_turn(15), body_state=point_down_r,
                          face_aus=fierce, head_rotation=head_groove, easing="ease_in_out"),
        AnimationKeyframe(time=21.0, wrapper_position=(5, SY - 1, 5),
                          wrapper_quaternion=_q_turn(-15), body_state=travolta,
                          face_aus=party, head_rotation=head_look_up, easing="ease_in_out"),
        # Spin
        AnimationKeyframe(time=23.0, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=_q_turn(180), body_state=bounce_up,
                          face_aus=excited, head_rotation=head_c, easing="ease_in_out"),
        AnimationKeyframe(time=25.0, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=_q_turn(360), body_state=bounce,
                          face_aus=party, head_rotation=head_groove, easing="ease_in_out"),
        # Body wave
        AnimationKeyframe(time=27.0, wrapper_position=(0, SY + 1, 0),
                          wrapper_quaternion=_q_turn(360), body_state=wave_top,
                          face_aus=fierce, head_rotation=head_c, easing="ease_in_out"),
        AnimationKeyframe(time=28.5, wrapper_position=(0, SY - 2, 0),
                          wrapper_quaternion=_q_turn(360), body_state=wave_mid,
                          face_aus=fierce, head_rotation=head_groove, easing="ease_in_out"),
        AnimationKeyframe(time=30.0, wrapper_position=(0, SY - 4, 0),
                          wrapper_quaternion=_q_turn(360), body_state=wave_bottom,
                          face_aus=excited, head_rotation=head_c, easing="ease_in_out"),
        AnimationKeyframe(time=31.0, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=_q_turn(360), body_state=bounce_up,
                          face_aus=party, head_rotation=head_c, easing="ease_in_out"),
        # High kicks
        AnimationKeyframe(time=33.0, wrapper_position=(-5, SY + 2, 0),
                          wrapper_quaternion=_q_turn(360), body_state=kick_r,
                          face_aus=party, head_rotation=head_c, easing="ease_in_out"),
        AnimationKeyframe(time=34.0, wrapper_position=(-5, SY - 2, 0),
                          wrapper_quaternion=_q_turn(360), body_state=bounce,
                          face_aus=excited, head_rotation=head_groove, easing="ease_out"),
        AnimationKeyframe(time=35.0, wrapper_position=(-10, SY + 2, 0),
                          wrapper_quaternion=_q_turn(360), body_state=kick_l,
                          face_aus=party, head_rotation=head_c, easing="ease_in_out"),
        AnimationKeyframe(time=36.0, wrapper_position=(-10, SY - 2, 0),
                          wrapper_quaternion=_q_turn(360), body_state=bounce,
                          face_aus=excited, head_rotation=head_groove, easing="ease_out"),
        # Double spin
        AnimationKeyframe(time=38.0, wrapper_position=(-5, SY + 2, 0),
                          wrapper_quaternion=_q_turn(360 + 180), body_state=bounce_up,
                          face_aus=excited, head_rotation=head_c, easing="ease_in_out"),
        AnimationKeyframe(time=40.0, wrapper_position=(0, SY + 2, 0),
                          wrapper_quaternion=_q_turn(360 + 540), body_state=bounce_up,
                          face_aus=party, head_rotation=head_c, easing="ease_in_out"),
        AnimationKeyframe(time=41.0, wrapper_position=(0, SY - 3, 0),
                          wrapper_quaternion=_q_turn(360 + 720), body_state=bounce,
                          face_aus=excited, head_rotation=head_groove, easing="ease_out"),
        # Final Travolta
        AnimationKeyframe(time=43.0, wrapper_position=(0, SY - 1, 0),
                          wrapper_quaternion=_q_turn(360 + 720), body_state=travolta,
                          face_aus=party, head_rotation=head_look_up, easing="ease_in_out"),
        AnimationKeyframe(time=45.0, wrapper_position=(0, SY - 1, 0),
                          wrapper_quaternion=_q_turn(360 + 720), body_state=travolta,
                          face_aus=party, head_rotation=head_look_up, easing="ease_in_out"),
        # Settle
        AnimationKeyframe(time=47.0, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=_q_turn(360 + 720), body_state=_neutral(),
                          face_aus=cool_d, head_rotation=head_c, easing="ease_in_out"),
        AnimationKeyframe(time=48.0, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=_q_turn(360 + 720), body_state=_neutral(),
                          face_aus=cool_d, head_rotation=head_c, easing="ease_in_out"),
    ]
    return AnimationClip(name="Disco", keyframes=keyframes, loop=True)


# ── Tai Chi ───────────────────────────────────────────────────────────

def make_tai_chi_clip() -> AnimationClip:
    """Create a ~56s looping tai chi / qigong flow.

    Ultra-slow, meditative, continuous movement.  Every transition is
    glacially smooth.  Emphasises breath, weight transfer, and whole-body
    coordination with minimal foot movement.

    Structure:
      0-4s    Opening: Wu Ji (standing meditation)
      4-10s   Commencement: arms float up and down
      10-16s  Ward Off Left
      16-22s  Ward Off Right + Roll Back
      22-28s  Press + Push (An)
      28-34s  Single Whip
      34-40s  Cloud Hands (x3)
      40-46s  Golden Rooster Stands on One Leg
      46-52s  Closing: Cross Hands
      52-56s  Return to Wu Ji (loop)
    """
    SY = _STAND_Y
    face_z = _q_turn(0)

    peaceful = {"AU5": 0.05, "AU6": 0.1, "AU12": 0.1}
    focused_tc = {"AU5": 0.1, "AU4": 0.05}
    breath_in = {"AU5": 0.15, "AU25": 0.1}

    head_c = {"headYaw": 0, "headPitch": 0, "headRoll": 0}
    head_down = {"headYaw": 0, "headPitch": 0.1, "headRoll": 0}
    head_look_l = {"headYaw": 0.2, "headPitch": 0.05, "headRoll": 0}
    head_look_r = {"headYaw": -0.2, "headPitch": 0.05, "headRoll": 0}

    # Wu Ji — still standing, weight centered
    wu_ji = _pose({
        "shoulderRAbduct": 0.05, "shoulderLAbduct": 0.05,
        "elbowRFlex": 0.05, "elbowLFlex": 0.05,
        "kneeRFlex": 0.08, "kneeLFlex": 0.08,
        "fingerRCurl": 0.1, "fingerLCurl": 0.1,
        "fingerRSpread": 0.3, "fingerLSpread": 0.3,
    })

    # Commencement rise — arms float up to shoulder height
    commence_up = _pose({
        "shoulderRAbduct": 0.05, "shoulderLAbduct": 0.05,
        "shoulderRFlex": 0.5, "shoulderLFlex": 0.5,
        "elbowRFlex": 0.05, "elbowLFlex": 0.05,
        "wristRFlex": -0.1, "wristLFlex": -0.1,
        "kneeRFlex": 0.08, "kneeLFlex": 0.08,
        "fingerRSpread": 0.4, "fingerLSpread": 0.4,
    })

    # Commencement press down — hands press to waist
    commence_down = _pose({
        "shoulderRAbduct": 0.05, "shoulderLAbduct": 0.05,
        "shoulderRFlex": 0.15, "shoulderLFlex": 0.15,
        "elbowRFlex": 0.4, "elbowLFlex": 0.4,
        "wristRFlex": 0.15, "wristLFlex": 0.15,
        "kneeRFlex": 0.15, "kneeLFlex": 0.15,
        "fingerRSpread": 0.3, "fingerLSpread": 0.3,
    })

    # Ward off left — left arm curved forward, right at hip
    ward_off_l = _pose({
        "spineRotation": -0.1,
        "shoulderLAbduct": 0.2, "shoulderLFlex": 0.4,
        "elbowLFlex": 0.4,
        "shoulderRAbduct": 0.05, "shoulderRFlex": 0.1,
        "elbowRFlex": 0.3,
        "wristLFlex": 0.05, "wristRFlex": 0.1,
        "hipLFlex": 0.15, "kneeLFlex": 0.2,
        "hipRFlex": 0.05, "kneeRFlex": 0.1,
        "fingerLSpread": 0.4, "fingerRSpread": 0.3,
    })

    # Ward off right
    ward_off_r = _pose({
        "spineRotation": 0.1,
        "shoulderRAbduct": 0.2, "shoulderRFlex": 0.4,
        "elbowRFlex": 0.4,
        "shoulderLAbduct": 0.05, "shoulderLFlex": 0.1,
        "elbowLFlex": 0.3,
        "wristRFlex": 0.05, "wristLFlex": 0.1,
        "hipRFlex": 0.15, "kneeRFlex": 0.2,
        "hipLFlex": 0.05, "kneeLFlex": 0.1,
        "fingerRSpread": 0.4, "fingerLSpread": 0.3,
    })

    # Roll back — both arms sweep back
    roll_back = _pose({
        "spineRotation": -0.15, "spineFlex": 0.05,
        "shoulderRAbduct": 0.15, "shoulderRFlex": -0.1,
        "shoulderLAbduct": 0.1, "shoulderLFlex": 0.2,
        "elbowRFlex": 0.35, "elbowLFlex": 0.4,
        "forearmRRotate": -0.2, "forearmLRotate": 0.1,
        "hipRFlex": 0.05, "hipLFlex": 0.15,
        "kneeRFlex": 0.1, "kneeLFlex": 0.2,
        "fingerRSpread": 0.3, "fingerLSpread": 0.3,
    })

    # Press — both hands come together forward
    press = _pose({
        "spineFlex": 0.05, "spineRotation": 0,
        "shoulderRAbduct": 0.1, "shoulderRFlex": 0.35,
        "shoulderLAbduct": 0.1, "shoulderLFlex": 0.35,
        "elbowRFlex": 0.3, "elbowLFlex": 0.3,
        "wristRFlex": 0.05, "wristLFlex": 0.05,
        "hipRFlex": 0.12, "hipLFlex": 0.12,
        "kneeRFlex": 0.15, "kneeLFlex": 0.15,
        "fingerRSpread": 0.2, "fingerLSpread": 0.2,
    })

    # Push (An) — palms push forward
    push = _pose({
        "spineFlex": 0.08,
        "shoulderRAbduct": 0.05, "shoulderRFlex": 0.5,
        "shoulderLAbduct": 0.05, "shoulderLFlex": 0.5,
        "elbowRFlex": 0.15, "elbowLFlex": 0.15,
        "wristRFlex": -0.15, "wristLFlex": -0.15,
        "hipRFlex": 0.15, "hipLFlex": 0.15,
        "kneeRFlex": 0.2, "kneeLFlex": 0.2,
        "fingerRSpread": 0.4, "fingerLSpread": 0.4,
    })

    # Single whip — right arm extends, left hooks
    single_whip = _pose({
        "spineRotation": -0.2,
        "shoulderRAbduct": 0.6, "shoulderRFlex": 0.2,
        "shoulderLAbduct": 0.1, "shoulderLFlex": 0.3,
        "elbowRFlex": 0.1, "elbowLFlex": 0.5,
        "wristRFlex": 0.3,  # hook hand
        "fingerRCurl": 0.6,
        "wristLFlex": -0.1,
        "fingerLSpread": 0.5,
        "hipRFlex": 0.05, "hipLFlex": 0.15,
        "kneeRFlex": 0.1, "kneeLFlex": 0.2,
    })

    # Cloud hands — weight shifting side to side, arms circling
    cloud_r = _pose({
        "spineRotation": 0.15,
        "shoulderRAbduct": 0.3, "shoulderRFlex": 0.4,
        "shoulderLAbduct": 0.1, "shoulderLFlex": 0.15,
        "elbowRFlex": 0.35, "elbowLFlex": 0.3,
        "forearmRRotate": 0.2,
        "hipRFlex": 0.05, "hipLFlex": 0.12,
        "kneeRFlex": 0.1, "kneeLFlex": 0.18,
        "fingerRSpread": 0.4, "fingerLSpread": 0.3,
    })
    cloud_l = _pose({
        "spineRotation": -0.15,
        "shoulderLAbduct": 0.3, "shoulderLFlex": 0.4,
        "shoulderRAbduct": 0.1, "shoulderRFlex": 0.15,
        "elbowLFlex": 0.35, "elbowRFlex": 0.3,
        "forearmLRotate": -0.2,
        "hipLFlex": 0.05, "hipRFlex": 0.12,
        "kneeLFlex": 0.1, "kneeRFlex": 0.18,
        "fingerLSpread": 0.4, "fingerRSpread": 0.3,
    })

    # Golden rooster — standing on one leg
    rooster_r = _pose({
        "hipLFlex": 0.5, "kneeLFlex": 0.6,
        "ankleLFlex": -0.1,
        "hipRFlex": 0.05, "kneeRFlex": 0.08,
        "shoulderLAbduct": 0.1, "shoulderLFlex": 0.55,
        "elbowLFlex": 0.4,
        "shoulderRAbduct": 0.05, "shoulderRFlex": 0.1,
        "elbowRFlex": 0.3,
        "spineFlex": -0.03,
        "fingerLSpread": 0.5,
    })
    rooster_l = _pose({
        "hipRFlex": 0.5, "kneeRFlex": 0.6,
        "ankleRFlex": -0.1,
        "hipLFlex": 0.05, "kneeLFlex": 0.08,
        "shoulderRAbduct": 0.1, "shoulderRFlex": 0.55,
        "elbowRFlex": 0.4,
        "shoulderLAbduct": 0.05, "shoulderLFlex": 0.1,
        "elbowLFlex": 0.3,
        "spineFlex": -0.03,
        "fingerRSpread": 0.5,
    })

    # Cross hands — closing
    cross_hands = _pose({
        "shoulderRAbduct": 0.2, "shoulderRFlex": 0.35,
        "shoulderLAbduct": 0.2, "shoulderLFlex": 0.35,
        "elbowRFlex": 0.35, "elbowLFlex": 0.35,
        "forearmRRotate": 0.2, "forearmLRotate": -0.2,
        "wristRFlex": 0.05, "wristLFlex": 0.05,
        "kneeRFlex": 0.1, "kneeLFlex": 0.1,
        "fingerRSpread": 0.3, "fingerLSpread": 0.3,
    })

    keyframes = [
        # Wu Ji
        AnimationKeyframe(time=0.0, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=face_z, body_state=wu_ji,
                          face_aus=peaceful, head_rotation=head_down, easing="linear"),
        AnimationKeyframe(time=4.0, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=face_z, body_state=wu_ji,
                          face_aus=peaceful, head_rotation=head_down, easing="ease_in_out"),
        # Commencement rise
        AnimationKeyframe(time=7.0, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=face_z, body_state=commence_up,
                          face_aus=breath_in, head_rotation=head_c, easing="ease_in_out"),
        # Commencement press down
        AnimationKeyframe(time=10.0, wrapper_position=(0, SY - 2, 0),
                          wrapper_quaternion=face_z, body_state=commence_down,
                          face_aus=peaceful, head_rotation=head_down, easing="ease_in_out"),
        # Ward off left
        AnimationKeyframe(time=13.0, wrapper_position=(-3, SY - 2, 3),
                          wrapper_quaternion=_q_turn(15), body_state=ward_off_l,
                          face_aus=focused_tc, head_rotation=head_look_l, easing="ease_in_out"),
        AnimationKeyframe(time=16.0, wrapper_position=(-3, SY - 1, 5),
                          wrapper_quaternion=_q_turn(15), body_state=ward_off_l,
                          face_aus=peaceful, head_rotation=head_look_l, easing="ease_in_out"),
        # Ward off right
        AnimationKeyframe(time=19.0, wrapper_position=(3, SY - 2, 5),
                          wrapper_quaternion=_q_turn(-15), body_state=ward_off_r,
                          face_aus=focused_tc, head_rotation=head_look_r, easing="ease_in_out"),
        # Roll back
        AnimationKeyframe(time=22.0, wrapper_position=(0, SY - 3, 0),
                          wrapper_quaternion=_q_turn(10), body_state=roll_back,
                          face_aus=peaceful, head_rotation=head_look_l, easing="ease_in_out"),
        # Press
        AnimationKeyframe(time=25.0, wrapper_position=(0, SY - 2, 5),
                          wrapper_quaternion=face_z, body_state=press,
                          face_aus=focused_tc, head_rotation=head_c, easing="ease_in_out"),
        # Push
        AnimationKeyframe(time=28.0, wrapper_position=(0, SY - 1, 8),
                          wrapper_quaternion=face_z, body_state=push,
                          face_aus=breath_in, head_rotation=head_c, easing="ease_in_out"),
        # Single whip
        AnimationKeyframe(time=31.0, wrapper_position=(-5, SY - 2, 5),
                          wrapper_quaternion=_q_turn(30), body_state=single_whip,
                          face_aus=peaceful, head_rotation=head_look_l, easing="ease_in_out"),
        AnimationKeyframe(time=34.0, wrapper_position=(-5, SY - 2, 5),
                          wrapper_quaternion=_q_turn(30), body_state=single_whip,
                          face_aus=focused_tc, head_rotation=head_look_l, easing="ease_in_out"),
        # Cloud hands R
        AnimationKeyframe(time=36.0, wrapper_position=(3, SY - 2, 3),
                          wrapper_quaternion=_q_turn(15), body_state=cloud_r,
                          face_aus=peaceful, head_rotation=head_look_r, easing="ease_in_out"),
        # Cloud hands L
        AnimationKeyframe(time=38.0, wrapper_position=(-3, SY - 2, 3),
                          wrapper_quaternion=_q_turn(-15), body_state=cloud_l,
                          face_aus=peaceful, head_rotation=head_look_l, easing="ease_in_out"),
        # Cloud hands R
        AnimationKeyframe(time=40.0, wrapper_position=(3, SY - 2, 3),
                          wrapper_quaternion=_q_turn(15), body_state=cloud_r,
                          face_aus=peaceful, head_rotation=head_look_r, easing="ease_in_out"),
        # Golden rooster R
        AnimationKeyframe(time=43.0, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=face_z, body_state=rooster_r,
                          face_aus=focused_tc, head_rotation=head_c, easing="ease_in_out"),
        # Golden rooster L
        AnimationKeyframe(time=46.0, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=face_z, body_state=rooster_l,
                          face_aus=focused_tc, head_rotation=head_c, easing="ease_in_out"),
        # Cross hands
        AnimationKeyframe(time=49.0, wrapper_position=(0, SY - 1, 0),
                          wrapper_quaternion=face_z, body_state=cross_hands,
                          face_aus=peaceful, head_rotation=head_down, easing="ease_in_out"),
        # Lower to Wu Ji
        AnimationKeyframe(time=52.0, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=face_z, body_state=commence_down,
                          face_aus=peaceful, head_rotation=head_down, easing="ease_in_out"),
        # Wu Ji return
        AnimationKeyframe(time=55.0, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=face_z, body_state=wu_ji,
                          face_aus=peaceful, head_rotation=head_down, easing="ease_in_out"),
        AnimationKeyframe(time=56.0, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=face_z, body_state=wu_ji,
                          face_aus=peaceful, head_rotation=head_down, easing="ease_in_out"),
    ]
    return AnimationClip(name="Tai Chi", keyframes=keyframes, loop=True)


# ── Power Dance (high energy, jumps, dynamic) ────────────────────────

def make_power_dance_clip() -> AnimationClip:
    """Create a ~52s looping high-energy power/street dance.

    Explosive, athletic movements: jumps, drops, power moves, freezes.
    Maximum dynamic range from floor to full extension.

    Structure:
      0-2s    Ready stance
      2-6s    Explosive jump + spread eagle
      6-10s   Drop to deep squat + sweep
      10-14s  Power lunge R → L rapid switch
      14-18s  Air kick sequence
      18-22s  Windmill arms + twist
      22-26s  Drop to ground → pop up
      26-30s  Isolation hits (robot-style)
      30-34s  Running man
      34-38s  Big leap + split landing
      38-42s  Barrel turn (fast 720°)
      42-46s  Explosive reach + freeze
      46-50s  Cool-down undulation
      50-52s  Return (loop)
    """
    SY = _STAND_Y
    face_z = _q_turn(0)

    intense = {"AU4": 0.3, "AU5": 0.3, "AU25": 0.2}
    fierce_p = {"AU4": 0.2, "AU5": 0.25, "AU12": 0.15, "AU25": 0.15}
    explosive = {"AU5": 0.4, "AU25": 0.3, "AU4": 0.15}
    triumphant = {"AU6": 0.4, "AU12": 0.5, "AU5": 0.2, "AU25": 0.2}

    head_c = {"headYaw": 0, "headPitch": 0, "headRoll": 0}
    head_snap_r = {"headYaw": -0.3, "headPitch": 0, "headRoll": -0.1}
    head_snap_l = {"headYaw": 0.3, "headPitch": 0, "headRoll": 0.1}
    head_up_power = {"headYaw": 0, "headPitch": -0.3, "headRoll": 0}
    head_down_power = {"headYaw": 0, "headPitch": 0.25, "headRoll": 0}

    # Ready stance — athletic crouch
    ready = _pose({
        "spineFlex": 0.1,
        "hipRFlex": 0.2, "hipLFlex": 0.2,
        "hipRAbduct": 0.1, "hipLAbduct": 0.1,
        "kneeRFlex": 0.3, "kneeLFlex": 0.3,
        "shoulderRAbduct": 0.2, "shoulderLAbduct": 0.2,
        "elbowRFlex": 0.5, "elbowLFlex": 0.5,
        "fingerRCurl": 0.6, "fingerLCurl": 0.6,
    })

    # Spread eagle — full extension mid-air
    spread_eagle = _pose({
        "spineFlex": -0.15,
        "shoulderRAbduct": 0.95, "shoulderLAbduct": 0.95,
        "shoulderRFlex": 0.3, "shoulderLFlex": 0.3,
        "elbowRFlex": 0.05, "elbowLFlex": 0.05,
        "hipRAbduct": 0.35, "hipLAbduct": 0.35,
        "hipRFlex": 0.05, "hipLFlex": 0.05,
        "kneeRFlex": 0.05, "kneeLFlex": 0.05,
        "ankleRFlex": -0.1, "ankleLFlex": -0.1,
        "fingerRSpread": 0.9, "fingerLSpread": 0.9,
        "toeRCurl": 0.3, "toeLCurl": 0.3,
    })

    # Deep squat
    deep_squat = _pose({
        "spineFlex": 0.3,
        "hipRFlex": 0.8, "hipLFlex": 0.8,
        "hipRAbduct": 0.2, "hipLAbduct": 0.2,
        "kneeRFlex": 0.9, "kneeLFlex": 0.9,
        "ankleRFlex": 0.15, "ankleLFlex": 0.15,
        "shoulderRAbduct": 0.3, "shoulderLAbduct": 0.3,
        "shoulderRFlex": 0.2, "shoulderLFlex": 0.2,
        "elbowRFlex": 0.4, "elbowLFlex": 0.4,
    })

    # Power lunge R
    lunge_r = _pose({
        "spineFlex": -0.1, "spineRotation": 0.15,
        "hipRFlex": 0.7, "kneeRFlex": 0.8,
        "hipLFlex": -0.15, "kneeLFlex": 0.05,
        "ankleLFlex": -0.1,
        "shoulderRAbduct": 0.7, "shoulderRFlex": 0.3,
        "shoulderLAbduct": 0.3, "shoulderLFlex": -0.1,
        "elbowRFlex": 0.15, "elbowLFlex": 0.3,
        "fingerRCurl": 0.7, "fingerLCurl": 0.7,
    })
    lunge_l = _pose({
        "spineFlex": -0.1, "spineRotation": -0.15,
        "hipLFlex": 0.7, "kneeLFlex": 0.8,
        "hipRFlex": -0.15, "kneeRFlex": 0.05,
        "ankleRFlex": -0.1,
        "shoulderLAbduct": 0.7, "shoulderLFlex": 0.3,
        "shoulderRAbduct": 0.3, "shoulderRFlex": -0.1,
        "elbowLFlex": 0.15, "elbowRFlex": 0.3,
        "fingerRCurl": 0.7, "fingerLCurl": 0.7,
    })

    # Air kick
    air_kick = _pose({
        "hipRFlex": 0.8, "kneeRFlex": 0.05,
        "ankleRFlex": -0.2,
        "hipLFlex": 0.1, "kneeLFlex": 0.2,
        "shoulderRAbduct": 0.5, "shoulderRFlex": -0.2,
        "shoulderLAbduct": 0.5, "shoulderLFlex": 0.3,
        "elbowRFlex": 0.2, "elbowLFlex": 0.2,
        "spineFlex": -0.15,
    })

    # Robot hit — sharp isolated position
    robot_r = _pose({
        "spineRotation": 0.2, "spineLatBend": -0.1,
        "shoulderRAbduct": 0.6, "shoulderRFlex": 0.3,
        "elbowRFlex": 0.7,
        "shoulderLAbduct": 0.1, "shoulderLFlex": 0,
        "elbowLFlex": 0.3,
        "hipRFlex": 0.05, "hipLFlex": 0.1,
        "kneeRFlex": 0.1, "kneeLFlex": 0.15,
        "fingerRCurl": 0.8, "fingerLCurl": 0.8,
    })
    robot_l = _pose({
        "spineRotation": -0.2, "spineLatBend": 0.1,
        "shoulderLAbduct": 0.6, "shoulderLFlex": 0.3,
        "elbowLFlex": 0.7,
        "shoulderRAbduct": 0.1, "shoulderRFlex": 0,
        "elbowRFlex": 0.3,
        "hipLFlex": 0.05, "hipRFlex": 0.1,
        "kneeLFlex": 0.1, "kneeRFlex": 0.15,
        "fingerRCurl": 0.8, "fingerLCurl": 0.8,
    })

    # Running man — alternating leg pumps
    run_r = _pose({
        "hipRFlex": 0.4, "kneeRFlex": 0.6,
        "hipLFlex": -0.1, "kneeLFlex": 0.1,
        "ankleRFlex": 0.05, "ankleLFlex": -0.1,
        "shoulderRFlex": -0.2, "shoulderLFlex": 0.3,
        "elbowRFlex": 0.5, "elbowLFlex": 0.5,
        "spineRotation": -0.1,
    })
    run_l = _pose({
        "hipLFlex": 0.4, "kneeLFlex": 0.6,
        "hipRFlex": -0.1, "kneeRFlex": 0.1,
        "ankleLFlex": 0.05, "ankleRFlex": -0.1,
        "shoulderLFlex": -0.2, "shoulderRFlex": 0.3,
        "elbowRFlex": 0.5, "elbowLFlex": 0.5,
        "spineRotation": 0.1,
    })

    # Split landing
    split_land = _pose({
        "spineFlex": 0.1,
        "hipRFlex": 0.6, "hipRAbduct": 0.3,
        "hipLFlex": -0.3, "hipLAbduct": 0.15,
        "kneeRFlex": 0.15, "kneeLFlex": 0.1,
        "ankleRFlex": 0.1, "ankleLFlex": -0.15,
        "shoulderRAbduct": 0.8, "shoulderLAbduct": 0.6,
        "shoulderRFlex": 0.2, "shoulderLFlex": -0.1,
        "elbowRFlex": 0.1, "elbowLFlex": 0.15,
        "fingerRSpread": 0.8, "fingerLSpread": 0.6,
    })

    # Explosive reach — both arms thrust up
    power_reach = _pose({
        "spineFlex": -0.2,
        "shoulderRAbduct": 0.6, "shoulderRFlex": 0.8,
        "shoulderLAbduct": 0.6, "shoulderLFlex": 0.8,
        "elbowRFlex": 0.05, "elbowLFlex": 0.05,
        "wristRFlex": -0.2, "wristLFlex": -0.2,
        "hipRFlex": 0.05, "hipLFlex": 0.05,
        "kneeRFlex": 0.08, "kneeLFlex": 0.08,
        "ankleRFlex": 0.12, "ankleLFlex": 0.12,
        "fingerRSpread": 0.9, "fingerLSpread": 0.9,
        "toeRCurl": 0.4, "toeLCurl": 0.4,
    })

    # Power freeze — asymmetric dramatic pose
    power_freeze = _pose({
        "spineFlex": -0.1, "spineLatBend": -0.15, "spineRotation": 0.2,
        "shoulderRAbduct": 0.9, "shoulderRFlex": 0.5,
        "shoulderLAbduct": 0.4, "shoulderLFlex": 0.6,
        "elbowRFlex": 0.1, "elbowLFlex": 0.4,
        "forearmRRotate": 0.3,
        "hipRFlex": 0.3, "kneeRFlex": 0.4,
        "hipLFlex": 0.02, "kneeLFlex": 0.05,
        "ankleRFlex": 0.05,
        "fingerRSpread": 0.9, "fingerLCurl": 0.5,
    })

    keyframes = [
        # Ready
        AnimationKeyframe(time=0.0, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=face_z, body_state=ready,
                          face_aus=intense, head_rotation=head_down_power, easing="linear"),
        AnimationKeyframe(time=2.0, wrapper_position=(0, SY - 3, 0),
                          wrapper_quaternion=face_z, body_state=ready,
                          face_aus=intense, head_rotation=head_c, easing="ease_in"),
        # Explosive jump!
        AnimationKeyframe(time=3.0, wrapper_position=(0, SY + 15, 0),
                          wrapper_quaternion=face_z, body_state=spread_eagle,
                          face_aus=explosive, head_rotation=head_up_power, easing="ease_out"),
        # Land in squat
        AnimationKeyframe(time=4.5, wrapper_position=(0, SY - 15, 0),
                          wrapper_quaternion=face_z, body_state=deep_squat,
                          face_aus=fierce_p, head_rotation=head_down_power, easing="ease_in"),
        # Recover
        AnimationKeyframe(time=6.0, wrapper_position=(0, SY - 5, 0),
                          wrapper_quaternion=face_z, body_state=ready,
                          face_aus=intense, head_rotation=head_c, easing="ease_out"),
        # Floor sweep (stay low, spin)
        AnimationKeyframe(time=8.0, wrapper_position=(0, SY - 15, 0),
                          wrapper_quaternion=_q_turn(180), body_state=deep_squat,
                          face_aus=fierce_p, head_rotation=head_snap_r, easing="ease_in_out"),
        AnimationKeyframe(time=10.0, wrapper_position=(0, SY - 10, 0),
                          wrapper_quaternion=_q_turn(360), body_state=ready,
                          face_aus=intense, head_rotation=head_c, easing="ease_out"),
        # Power lunge R
        AnimationKeyframe(time=11.0, wrapper_position=(10, SY - 8, 0),
                          wrapper_quaternion=_q_turn(360 - 20), body_state=lunge_r,
                          face_aus=fierce_p, head_rotation=head_snap_r, easing="ease_in_out"),
        # Quick switch — lunge L
        AnimationKeyframe(time=12.0, wrapper_position=(-10, SY - 8, 0),
                          wrapper_quaternion=_q_turn(360 + 20), body_state=lunge_l,
                          face_aus=fierce_p, head_rotation=head_snap_l, easing="ease_in_out"),
        # Lunge R again
        AnimationKeyframe(time=13.0, wrapper_position=(15, SY - 8, 0),
                          wrapper_quaternion=_q_turn(360 - 20), body_state=lunge_r,
                          face_aus=explosive, head_rotation=head_snap_r, easing="ease_in_out"),
        # Center recovery
        AnimationKeyframe(time=14.0, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=_q_turn(360), body_state=ready,
                          face_aus=intense, head_rotation=head_c, easing="ease_out"),
        # Air kick R
        AnimationKeyframe(time=15.5, wrapper_position=(0, SY + 8, 5),
                          wrapper_quaternion=_q_turn(360), body_state=air_kick,
                          face_aus=explosive, head_rotation=head_c, easing="ease_in_out"),
        # Land + air kick L (mirror)
        AnimationKeyframe(time=17.0, wrapper_position=(0, SY + 8, -5),
                          wrapper_quaternion=_q_turn(360),
                          body_state=_pose({
                              "hipLFlex": 0.8, "kneeLFlex": 0.05, "ankleLFlex": -0.2,
                              "hipRFlex": 0.1, "kneeRFlex": 0.2,
                              "shoulderLAbduct": 0.5, "shoulderLFlex": -0.2,
                              "shoulderRAbduct": 0.5, "shoulderRFlex": 0.3,
                              "elbowRFlex": 0.2, "elbowLFlex": 0.2,
                              "spineFlex": -0.15,
                          }),
                          face_aus=explosive, head_rotation=head_c, easing="ease_in_out"),
        # Land
        AnimationKeyframe(time=18.0, wrapper_position=(0, SY - 3, 0),
                          wrapper_quaternion=_q_turn(360), body_state=ready,
                          face_aus=intense, head_rotation=head_down_power, easing="ease_out"),
        # Windmill arms + twist
        AnimationKeyframe(time=20.0, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=_q_turn(360 + 90),
                          body_state=_pose({
                              "spineRotation": 0.3,
                              "shoulderRAbduct": 0.9, "shoulderRFlex": 0.5,
                              "shoulderLAbduct": 0.2, "shoulderLFlex": -0.3,
                              "elbowRFlex": 0.05, "elbowLFlex": 0.1,
                              "hipRFlex": 0.1, "hipLFlex": 0.1,
                              "kneeRFlex": 0.15, "kneeLFlex": 0.15,
                          }),
                          face_aus=fierce_p, head_rotation=head_snap_r, easing="ease_in_out"),
        AnimationKeyframe(time=22.0, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=_q_turn(360 + 270),
                          body_state=_pose({
                              "spineRotation": -0.3,
                              "shoulderLAbduct": 0.9, "shoulderLFlex": 0.5,
                              "shoulderRAbduct": 0.2, "shoulderRFlex": -0.3,
                              "elbowLFlex": 0.05, "elbowRFlex": 0.1,
                              "hipRFlex": 0.1, "hipLFlex": 0.1,
                              "kneeRFlex": 0.15, "kneeLFlex": 0.15,
                          }),
                          face_aus=fierce_p, head_rotation=head_snap_l, easing="ease_in_out"),
        # Drop to ground
        AnimationKeyframe(time=24.0, wrapper_position=(0, SY - 20, 0),
                          wrapper_quaternion=_q_turn(720), body_state=deep_squat,
                          face_aus=intense, head_rotation=head_down_power, easing="ease_in"),
        # Pop up!
        AnimationKeyframe(time=25.5, wrapper_position=(0, SY + 5, 0),
                          wrapper_quaternion=_q_turn(720), body_state=power_reach,
                          face_aus=explosive, head_rotation=head_up_power, easing="ease_out"),
        # Robot hits
        AnimationKeyframe(time=27.0, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=_q_turn(720), body_state=robot_r,
                          face_aus=fierce_p, head_rotation=head_snap_r, easing="linear"),
        AnimationKeyframe(time=28.0, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=_q_turn(720), body_state=robot_l,
                          face_aus=fierce_p, head_rotation=head_snap_l, easing="linear"),
        AnimationKeyframe(time=29.0, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=_q_turn(720), body_state=robot_r,
                          face_aus=fierce_p, head_rotation=head_snap_r, easing="linear"),
        AnimationKeyframe(time=30.0, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=_q_turn(720), body_state=ready,
                          face_aus=intense, head_rotation=head_c, easing="ease_out"),
        # Running man
        AnimationKeyframe(time=31.0, wrapper_position=(0, SY, -5),
                          wrapper_quaternion=_q_turn(720), body_state=run_r,
                          face_aus=fierce_p, head_rotation=head_c, easing="ease_in_out"),
        AnimationKeyframe(time=31.7, wrapper_position=(0, SY, -10),
                          wrapper_quaternion=_q_turn(720), body_state=run_l,
                          face_aus=fierce_p, head_rotation=head_c, easing="ease_in_out"),
        AnimationKeyframe(time=32.4, wrapper_position=(0, SY, -15),
                          wrapper_quaternion=_q_turn(720), body_state=run_r,
                          face_aus=intense, head_rotation=head_c, easing="ease_in_out"),
        AnimationKeyframe(time=33.0, wrapper_position=(0, SY, -20),
                          wrapper_quaternion=_q_turn(720), body_state=run_l,
                          face_aus=fierce_p, head_rotation=head_c, easing="ease_in_out"),
        # Big leap
        AnimationKeyframe(time=35.0, wrapper_position=(0, SY + 18, 0),
                          wrapper_quaternion=_q_turn(720), body_state=spread_eagle,
                          face_aus=explosive, head_rotation=head_up_power, easing="ease_out"),
        # Split landing
        AnimationKeyframe(time=36.5, wrapper_position=(0, SY - 10, 10),
                          wrapper_quaternion=_q_turn(720), body_state=split_land,
                          face_aus=triumphant, head_rotation=head_c, easing="ease_in"),
        # Recover
        AnimationKeyframe(time=38.0, wrapper_position=(0, SY, 5),
                          wrapper_quaternion=_q_turn(720), body_state=ready,
                          face_aus=intense, head_rotation=head_c, easing="ease_out"),
        # Barrel turn 720°
        AnimationKeyframe(time=40.0, wrapper_position=(0, SY + 3, 0),
                          wrapper_quaternion=_q_turn(720 + 360), body_state=_pose({
                              "shoulderRAbduct": 0.3, "shoulderLAbduct": 0.3,
                              "elbowRFlex": 0.5, "elbowLFlex": 0.5,
                              "hipRFlex": 0.1, "hipLFlex": 0.1,
                              "kneeRFlex": 0.15, "kneeLFlex": 0.15,
                          }),
                          face_aus=fierce_p, head_rotation=head_c, easing="ease_in_out"),
        AnimationKeyframe(time=42.0, wrapper_position=(0, SY - 3, 0),
                          wrapper_quaternion=_q_turn(720 + 720), body_state=ready,
                          face_aus=intense, head_rotation=head_c, easing="ease_out"),
        # Explosive reach + freeze
        AnimationKeyframe(time=43.5, wrapper_position=(0, SY + 5, 0),
                          wrapper_quaternion=_q_turn(720 + 720), body_state=power_reach,
                          face_aus=explosive, head_rotation=head_up_power, easing="ease_out"),
        # Power freeze
        AnimationKeyframe(time=45.0, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=_q_turn(720 + 720), body_state=power_freeze,
                          face_aus=triumphant, head_rotation=head_snap_r, easing="ease_in_out"),
        # Hold freeze
        AnimationKeyframe(time=46.5, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=_q_turn(720 + 720), body_state=power_freeze,
                          face_aus=triumphant, head_rotation=head_snap_r, easing="ease_in_out"),
        # Cool-down
        AnimationKeyframe(time=49.0, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=_q_turn(720 + 720), body_state=ready,
                          face_aus=intense, head_rotation=head_c, easing="ease_in_out"),
        AnimationKeyframe(time=51.0, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=_q_turn(720 + 720), body_state=_neutral(),
                          face_aus={"AU12": 0.2, "AU5": 0.1}, head_rotation=head_c, easing="ease_in_out"),
        AnimationKeyframe(time=52.0, wrapper_position=(0, SY, 0),
                          wrapper_quaternion=_q_turn(720 + 720), body_state=_neutral(),
                          face_aus={"AU12": 0.2, "AU5": 0.1}, head_rotation=head_c, easing="ease_in_out"),
    ]
    return AnimationClip(name="Power Dance", keyframes=keyframes, loop=True)


# ── Registry ─────────────────────────────────────────────────────────

def get_builtin_clips() -> dict[str, AnimationClip]:
    """Return all built-in animation clips by name."""
    return {
        "Wake Up": make_wake_up_clip(),
        "Lie Down": make_lie_down_clip(),
        "Contemporary": make_dance_clip(),
        "Tap Dance": make_tap_dance_clip(),
        "Waltz": make_waltz_clip(),
        "Disco": make_disco_clip(),
        "Tai Chi": make_tai_chi_clip(),
        "Power Dance": make_power_dance_clip(),
    }
