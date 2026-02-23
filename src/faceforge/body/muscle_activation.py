"""Muscle Activation Heatmap system.

Color-codes muscles by computed activation intensity as the body moves.
Activation = max absolute DOF value across a muscle's associated joints.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from faceforge.core.mesh import MeshInstance

logger = logging.getLogger(__name__)

# Cool→hot colour ramp (blue→cyan→green→yellow→red)
# 256-entry LUT, each entry is (R, G, B) in [0, 1].
_COLOR_LUT = np.zeros((256, 3), dtype=np.float32)

def _build_lut():
    """Build a 256-entry cool→hot colour lookup table."""
    for i in range(256):
        t = i / 255.0
        if t < 0.25:
            s = t / 0.25
            _COLOR_LUT[i] = (0.0, s, 1.0)          # blue → cyan
        elif t < 0.5:
            s = (t - 0.25) / 0.25
            _COLOR_LUT[i] = (0.0, 1.0, 1.0 - s)    # cyan → green
        elif t < 0.75:
            s = (t - 0.5) / 0.25
            _COLOR_LUT[i] = (s, 1.0, 0.0)           # green → yellow
        else:
            s = (t - 0.75) / 0.25
            _COLOR_LUT[i] = (1.0, 1.0 - s, 0.0)    # yellow → red

_build_lut()


def _load_dof_map() -> dict[str, list[str]]:
    """Load DOF→muscle mapping from config."""
    config_dir = Path(__file__).resolve().parents[2] / "assets" / "config"
    path = config_dir / "muscle_dof_map.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


class MuscleActivationSystem:
    """Computes per-muscle activation from DOF values and applies vertex colours.

    Parameters
    ----------
    dof_map : dict[str, list[str]]
        Maps DOF field names (e.g. ``"elbow_r_flex"``) to lists of muscle
        names that cross that joint.
    """

    def __init__(self, dof_map: Optional[dict[str, list[str]]] = None):
        if dof_map is None:
            dof_map = _load_dof_map()
        self._dof_map = dof_map
        # Invert: muscle_name → list of DOF fields
        self._muscle_dofs: dict[str, list[str]] = {}
        for dof, muscles in self._dof_map.items():
            for m in muscles:
                self._muscle_dofs.setdefault(m, []).append(dof)

        # Registered muscles: name → (mesh, original_colors)
        self._muscles: dict[str, tuple[MeshInstance, Optional[np.ndarray]]] = {}
        self._enabled = False

    def register_muscle(self, mesh: MeshInstance, name: str) -> None:
        """Register a muscle mesh for heatmap colouring."""
        # Store original vertex colours (or None if mesh doesn't have them)
        original = None
        if mesh.geometry.vertex_colors is not None:
            original = mesh.geometry.vertex_colors.copy()
        self._muscles[name] = (mesh, original)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        """Toggle heatmap on/off. When off, restore original colours."""
        was_enabled = self._enabled
        self._enabled = enabled
        if not enabled and was_enabled:
            self._restore_colors()

    def _restore_colors(self) -> None:
        """Restore original vertex colours on all registered muscles."""
        for name, (mesh, original) in self._muscles.items():
            if original is not None:
                mesh.geometry.vertex_colors = original.copy()
            else:
                mesh.geometry.vertex_colors = None
            mesh.geometry.vertex_colors_active = False
            mesh.needs_update = True

    def update(self, body_state) -> None:
        """Compute activation for each muscle and apply vertex colours.

        Parameters
        ----------
        body_state : BodyState
            Current body state with DOF values as attributes.
        """
        if not self._enabled:
            return

        for name, (mesh, _original) in self._muscles.items():
            # Compute activation: max |DOF value| across associated joints
            dofs = self._muscle_dofs.get(name, [])
            activation = 0.0
            for dof_field in dofs:
                val = getattr(body_state, dof_field, 0.0)
                if isinstance(val, (int, float)):
                    activation = max(activation, abs(float(val)))
            activation = min(activation, 1.0)

            # Map activation to colour via LUT
            idx = int(activation * 255)
            color = _COLOR_LUT[idx]

            # Apply uniform colour to all vertices
            n_verts = mesh.geometry.vertex_count
            if n_verts <= 0:
                continue

            colors = np.broadcast_to(color, (n_verts, 3)).copy()
            mesh.geometry.vertex_colors = colors.astype(np.float32)
            mesh.geometry.vertex_colors_active = True
            mesh.needs_update = True
