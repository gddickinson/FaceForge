"""Shared constants and paths for FaceForge."""

import os
from pathlib import Path


def _env_path(var: str, default: Path) -> Path:
    """Return ``$var`` as a path if set and non-empty, else *default*.

    The paths below are anchored on this file's location, which is correct for
    a source checkout but has two consequences worth overriding:

    * The BodyParts3D dataset is 1.32 GB and ships separately (``assets/stl``
      is a symlink in the repo).  Without an override, a user who keeps it on
      another volume has to create that symlink rather than point at it.
    * Because the anchor is the *installed* package, an editable install always
      resolves to the original checkout's assets.  A copy of the tree therefore
      reads the original's dataset, which makes the "no dataset present"
      condition -- the one CI actually runs in -- untestable locally.  That is
      not hypothetical: it silently invalidated two attempts to reproduce a CI
      failure, each of which reported the with-dataset result instead.

    An unset or empty variable falls through to the default, so existing
    checkouts behave exactly as before.
    """
    value = os.environ.get(var, "").strip()
    return Path(value).expanduser() if value else default


# Project paths.  Each may be redirected by the matching FACEFORGE_* variable.
PROJECT_ROOT = _env_path("FACEFORGE_PROJECT_ROOT",
                         Path(__file__).parent.parent.parent)
ASSETS_DIR = _env_path("FACEFORGE_ASSETS_DIR", PROJECT_ROOT / "assets")
CONFIG_DIR = _env_path("FACEFORGE_CONFIG_DIR", ASSETS_DIR / "config")
MESHDATA_DIR = ASSETS_DIR / "meshdata"
#: BodyParts3D STL directory.  Set FACEFORGE_STL_DIR to keep the 1.32 GB
#: dataset outside the repository instead of symlinking it into assets/stl.
STL_DIR = _env_path("FACEFORGE_STL_DIR", ASSETS_DIR / "stl")
MUSCLE_CONFIG_DIR = CONFIG_DIR / "muscles"
SKELETON_CONFIG_DIR = CONFIG_DIR / "skeleton"
MAKEHUMAN_DIR = ASSETS_DIR / "makehuman"
BODY_MESHES_DIR = ASSETS_DIR / "body_meshes"

# Face mesh constants
FACE_VERT_COUNT = 468  # Original MediaPipe face landmarks
BACK_RING_COUNT = 9
BACK_VERTS_PER_RING = 36
BACK_VERT_COUNT = BACK_RING_COUNT * BACK_VERTS_PER_RING + 1  # +1 for pole
OUTER_VERT_COUNT = FACE_VERT_COUNT + BACK_VERT_COUNT  # 793
FACE_TOTAL_VERT_COUNT = OUTER_VERT_COUNT * 2  # ~1586 (outer + inner shell)

# Skin mesh constants
SKIN_THICKNESS = 0.3
SKIN_OFFSET = 0.5

# Skull/face alignment defaults
DEFAULT_FACE_SCALE = 1.14
DEFAULT_FACE_OFFSET_X = -0.2
DEFAULT_FACE_OFFSET_Y = -10.6
DEFAULT_FACE_OFFSET_Z = 9.5
DEFAULT_FACE_ROT_X_DEG = 88.5

# Jaw pivot (TMJ hinge) — original embedded skull position
JAW_PIVOT_ORIGINAL = (0.0, -1.5, 10.4)
JAW_PIVOT = JAW_PIVOT_ORIGINAL  # Backward compat alias

# Mutable active pivot (updated when BP3D skull computes TMJ dynamically)
_active_jaw_pivot = list(JAW_PIVOT_ORIGINAL)


def get_jaw_pivot() -> tuple[float, float, float]:
    """Return the currently active jaw pivot position."""
    return tuple(_active_jaw_pivot)


def set_jaw_pivot(x: float, y: float, z: float) -> None:
    """Set the active jaw pivot (e.g. after computing TMJ from BP3D mandible)."""
    _active_jaw_pivot[:] = [x, y, z]

# Camera defaults (front view, Z-up body)
# Body Z-vertical: head≈0, feet≈-200; Y-depth: anterior≈-5, posterior≈+10
DEFAULT_CAMERA_POS = (0.0, -120.0, -30.0)
DEFAULT_CAMERA_TARGET = (0.0, 0.0, -50.0)

# Head rotation limits (degrees)
HEAD_YAW_MAX = 35.0
HEAD_PITCH_MAX = 30.0
HEAD_ROLL_MAX = 30.0

# Animation defaults
TARGET_FPS = 60
MAX_DELTA_TIME = 0.1  # Clamp dt to avoid large jumps

# STL loading tiers
TIER_SKULL = 0      # Skull mesh (embedded data)
TIER_HEAD = 1       # Head muscles, features, vertebrae
TIER_SKELETON = 2   # Body skeleton
TIER_MUSCLES = 3    # Body muscles (on demand)
TIER_ORGANS = 4     # Organs (on demand)
TIER_VASCULAR = 5   # Vasculature + brain (on demand)
