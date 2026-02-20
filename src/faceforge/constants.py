"""Shared constants and paths for FaceForge."""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
CONFIG_DIR = ASSETS_DIR / "config"
MESHDATA_DIR = ASSETS_DIR / "meshdata"
STL_DIR = ASSETS_DIR / "stl"
MUSCLE_CONFIG_DIR = CONFIG_DIR / "muscles"
SKELETON_CONFIG_DIR = CONFIG_DIR / "skeleton"

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

# Camera defaults (full body view)
DEFAULT_CAMERA_POS = (0.0, -40.0, 120.0)
DEFAULT_CAMERA_TARGET = (0.0, -30.0, 0.0)

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
