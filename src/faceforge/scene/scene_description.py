"""Serializable scene description for future Blender export.

Data-only module -- no rendering logic -- so a Blender export script can
consume it independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SceneDescription:
    """Captures all scene parameters in a renderer-agnostic format.

    A future ``tools/blender_export.py`` would read this description,
    export body meshes as OBJ/glTF, and generate a Blender Python script
    that recreates the room, camera, lighting, and animation keyframes.
    """

    # Room dimensions (width, height, depth)
    room_dims: tuple[float, float, float] = (300.0, 250.0, 200.0)

    # Table
    table_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    table_dims: tuple[float, float, float] = (200.0, 90.0, 60.0)  # length, height, width

    # Lamp
    lamp_pos: tuple[float, float, float] = (0.0, 200.0, 0.0)

    # Lighting
    light_color: tuple[float, float, float] = (1.0, 0.95, 0.85)
    light_intensity: float = 1.5

    # Body transform (4x4 row-major)
    body_transform: list[list[float]] = field(default_factory=lambda: [
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
    ])

    # Camera
    camera_pos: tuple[float, float, float] = (0.0, 200.0, 0.0)
    camera_target: tuple[float, float, float] = (0.0, 90.0, 0.0)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "room_dims": list(self.room_dims),
            "table_pos": list(self.table_pos),
            "table_dims": list(self.table_dims),
            "lamp_pos": list(self.lamp_pos),
            "light_color": list(self.light_color),
            "light_intensity": self.light_intensity,
            "body_transform": self.body_transform,
            "camera_pos": list(self.camera_pos),
            "camera_target": list(self.camera_target),
        }

    @classmethod
    def from_dict(cls, d: dict) -> SceneDescription:
        """Deserialize from a dictionary."""
        return cls(
            room_dims=tuple(d.get("room_dims", (300.0, 250.0, 200.0))),
            table_pos=tuple(d.get("table_pos", (0.0, 0.0, 0.0))),
            table_dims=tuple(d.get("table_dims", (200.0, 90.0, 60.0))),
            lamp_pos=tuple(d.get("lamp_pos", (0.0, 200.0, 0.0))),
            light_color=tuple(d.get("light_color", (1.0, 0.95, 0.85))),
            light_intensity=d.get("light_intensity", 1.5),
            body_transform=d.get("body_transform", [
                [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
            ]),
            camera_pos=tuple(d.get("camera_pos", (0.0, 200.0, 0.0))),
            camera_target=tuple(d.get("camera_target", (0.0, 90.0, 0.0))),
        )
