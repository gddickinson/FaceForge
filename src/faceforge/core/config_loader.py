"""JSON config file loading utilities."""

import json
from pathlib import Path
from typing import Any

from faceforge.constants import CONFIG_DIR, MESHDATA_DIR, MUSCLE_CONFIG_DIR, SKELETON_CONFIG_DIR


def load_json(path: Path) -> Any:
    """Load and return parsed JSON from a file."""
    with open(path) as f:
        return json.load(f)


def load_config(name: str) -> Any:
    """Load a config file from assets/config/."""
    return load_json(CONFIG_DIR / name)


def load_meshdata(name: str) -> Any:
    """Load a mesh data file from assets/meshdata/."""
    return load_json(MESHDATA_DIR / name)


def load_muscle_config(name: str) -> Any:
    """Load a muscle config from assets/config/muscles/."""
    return load_json(MUSCLE_CONFIG_DIR / name)


def load_skeleton_config(name: str) -> Any:
    """Load a skeleton config from assets/config/skeleton/."""
    return load_json(SKELETON_CONFIG_DIR / name)
