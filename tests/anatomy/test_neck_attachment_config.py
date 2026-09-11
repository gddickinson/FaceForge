"""Every neck muscle names attachment bones that actually exist.

``lowerBones`` is resolved against the live scene by name, and a name that
matches nothing fails silently: ``bone_displacement`` returns ``None`` and
the muscle simply gets no pinning and no bone-follow.  Ten muscles shipped
that way, naming "Thoracic Vertebra T1" where the scene node is "T1".

The skeleton configs are the source of the node names, so this checks the
two files against each other without needing the STL set.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from faceforge.constants import CONFIG_DIR

SKELETON_DIR = Path(CONFIG_DIR) / "skeleton"
NECK_CONFIG = Path(CONFIG_DIR) / "muscles" / "neck_muscles.json"


def _skeleton_names() -> set[str]:
    names: set[str] = set()
    for path in SKELETON_DIR.glob("*.json"):
        entries = json.loads(path.read_text())
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict) and entry.get("name"):
                names.add(entry["name"])
    return names


def _neck_muscles() -> list[dict]:
    return json.loads(NECK_CONFIG.read_text())


def test_the_skeleton_configs_were_found():
    names = _skeleton_names()
    assert len(names) > 100, "skeleton configs missing or empty"
    assert "T1" in names and "Right Clavicle" in names


@pytest.mark.parametrize("muscle", _neck_muscles(), ids=lambda m: m["name"])
def test_every_lower_bone_is_a_real_skeleton_node(muscle):
    names = _skeleton_names()
    for bone in muscle.get("lowerBones", []):
        assert bone in names, (
            f"{muscle['name']} names {bone!r}, which is not a skeleton node; "
            "the pinning and bone-follow would silently do nothing"
        )


def test_every_neck_muscle_names_its_attachment_bones():
    """A muscle with no bones falls back to a coarse regional average."""
    missing = [m["name"] for m in _neck_muscles() if not m.get("lowerBones")]
    assert missing == [], missing
