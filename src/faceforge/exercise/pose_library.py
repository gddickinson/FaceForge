"""Pose authoring for exercises: degrees in, normalised BodyState DOFs out.

Conventions (see :mod:`faceforge.body.dof_ranges` for the full table):

* ``hip_flex`` 110 means both hips flexed 110 degrees; ``hip_r_flex`` means one.
* positive = flexion / abduction / external rotation / dorsiflexion / supination.
* Trunk lean is not a DOF.  :func:`squat` and :func:`hinge` return the whole-
  body pitch alongside the pose; the clip builder turns it into a wrapper
  rotation.  The hips are flexed by *more* than the pitch so the thighs end
  up where the description says.

The foot-flat rule
------------------
With the body pitched forward by ``pitch`` and the thigh swung anteriorly by
``hip`` and the shank back by ``knee`` (all degrees), the foot lies flat on
the floor when the ankle is dorsiflexed by ``pitch - hip + knee``.  Every
standing pose here uses :func:`flat_foot_ankle` so the ground lock has a flat
sole to stand on.
"""

from __future__ import annotations

from faceforge.body.dof_ranges import POSE_DOF_FIELDS, degrees_to_dof, is_pose_dof

_BILATERAL_PREFIXES = ("shoulder", "elbow", "forearm", "wrist", "hip", "knee", "ankle")
_BILATERAL_SUFFIXED = ("finger_curl", "finger_spread", "thumb_op", "toe_curl", "toe_spread")


def _expand_key(key: str) -> list[str]:
    """``"hip_flex"`` -> both sides; ``"hip_r_flex"`` / ``"spine_flex"`` -> itself."""
    if is_pose_dof(key) and key in POSE_DOF_FIELDS:
        return [key]
    for prefix in _BILATERAL_PREFIXES:
        if key.startswith(prefix + "_"):
            rest = key[len(prefix) + 1:]
            return [f"{prefix}_r_{rest}", f"{prefix}_l_{rest}"]
    if key in _BILATERAL_SUFFIXED:
        return [f"{key}_r", f"{key}_l"]
    raise KeyError(f"{key!r} is not a body DOF or a bilateral shorthand")


def neutral() -> dict[str, float]:
    """Every pose DOF at zero: the anatomical position."""
    return {f: 0.0 for f in POSE_DOF_FIELDS}


def pose(**degrees: float) -> dict[str, float]:
    """A full pose from keyword DOFs in degrees, on top of :func:`neutral`.

    ``pose(hip_flex=100, knee_flex=110, shoulder_r_flex=90)``
    """
    out = neutral()
    for key, deg in degrees.items():
        for field_name in _expand_key(key):
            out[field_name] = degrees_to_dof(field_name, float(deg))
    return out


def merge(*poses: dict[str, float]) -> dict[str, float]:
    """Later poses override earlier ones; the result is always a full pose."""
    out = neutral()
    for p in poses:
        out.update(p)
    return out


def combine(*parts: dict[str, float]) -> dict[str, float]:
    """Union of PARTIAL poses, without neutral fill (later parts win).

    Use this, not :func:`merge`, to bundle arm and grip fragments: ``merge``
    returns a full pose, and merging a full "arms" pose over a squat resets
    every leg DOF to zero.
    """
    out: dict[str, float] = {}
    for p in parts:
        out.update(p)
    return out


def only(**degrees: float) -> dict[str, float]:
    """The given DOFs in degrees, with NO neutral fill (for merging)."""
    out: dict[str, float] = {}
    for key, deg in degrees.items():
        for field_name in _expand_key(key):
            out[field_name] = degrees_to_dof(field_name, float(deg))
    return out


def flat_foot_ankle(pitch: float, hip: float, knee: float) -> float:
    """Ankle dorsiflexion (degrees) that keeps the sole flat, see module doc."""
    return pitch - hip + knee


def squat(depth_hip: float, knee: float, pitch: float, **extra: float) -> tuple[dict, float]:
    """A bilateral squat position.

    ``depth_hip`` is the hip flexion angle between trunk and thigh, ``knee``
    the knee flexion, ``pitch`` the trunk lean from vertical.  Returns
    ``(pose, pitch)``.  Extra keyword DOFs (arms, spine) are merged in.
    """
    ankle = flat_foot_ankle(pitch, depth_hip, knee)
    p = pose(hip_flex=depth_hip, knee_flex=knee, ankle_flex=ankle, **extra)
    return p, pitch


def hinge(pitch: float, knee: float, **extra: float) -> tuple[dict, float]:
    """A hip hinge: trunk pitched forward by ``pitch``, thighs kept vertical."""
    ankle = flat_foot_ankle(pitch, pitch, knee)
    p = pose(hip_flex=pitch, knee_flex=knee, ankle_flex=ankle, **extra)
    return p, pitch


def stand(**extra: float) -> tuple[dict, float]:
    """Upright, feet flat, arms as given."""
    return pose(**extra), 0.0


def lunge(front: str, hip_front: float, knee_front: float, hip_back: float,
          knee_back: float, pitch: float = 10.0, ankle_back: float = -20.0,
          **extra: float) -> tuple[dict, float]:
    """A split stance: ``front`` is ``"r"`` or ``"l"``."""
    back = "l" if front == "r" else "r"
    kw = {
        f"hip_{front}_flex": hip_front, f"knee_{front}_flex": knee_front,
        f"ankle_{front}_flex": flat_foot_ankle(pitch, hip_front, knee_front),
        f"hip_{back}_flex": hip_back, f"knee_{back}_flex": knee_back,
        f"ankle_{back}_flex": ankle_back,
    }
    kw.update(extra)
    return pose(**kw), pitch


def arms(flex: float = 0.0, abduct: float = 0.0, rotate: float = 0.0, elbow: float = 0.0,
         forearm: float = 0.0, wrist: float = 0.0, side: str | None = None) -> dict[str, float]:
    """Bilateral (or one-sided) arm DOFs in degrees, without neutral fill."""
    sides = ("r", "l") if side is None else (side.lower(),)
    kw: dict[str, float] = {}
    for s in sides:
        kw[f"shoulder_{s}_flex"] = flex
        kw[f"shoulder_{s}_abduct"] = abduct
        kw[f"shoulder_{s}_rotate"] = rotate
        kw[f"elbow_{s}_flex"] = elbow
        kw[f"forearm_{s}_rotate"] = forearm
        kw[f"wrist_{s}_flex"] = wrist
    return only(**kw)


def grip(curl: float = 85.0, thumb: float = 45.0) -> dict[str, float]:
    """Fingers closed round a handle.

    ``curl`` is the metacarpophalangeal flexion in degrees; the
    interphalangeal joints follow in proportion (``BodyAnimationSystem``
    ``_FINGER_CURL_MAX``), so 85 closes the hand round a bar.
    """
    return only(finger_curl=curl, thumb_op=thumb)
