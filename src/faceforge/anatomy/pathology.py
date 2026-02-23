"""Pathology visualization system.

Applies visual effects to anatomical structures to simulate pathological
conditions: fracture, tumor, inflammation, osteoarthritis, edema.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from faceforge.core.mesh import MeshInstance

logger = logging.getLogger(__name__)


@dataclass
class PathologyCondition:
    """A pathological condition applied to a structure."""
    condition: str      # "fracture", "tumor", "inflammation", "osteoarthritis", "edema"
    severity: float     # 0.0 to 1.0
    mesh_name: str = ""
    mesh: Optional[MeshInstance] = None
    # Saved state for restoration
    original_color: Optional[tuple] = None
    original_scale: Optional[np.ndarray] = None
    original_positions: Optional[np.ndarray] = None
    tumor_mesh: Optional[MeshInstance] = None


# Condition colour overlays (R, G, B)
_CONDITION_COLORS = {
    "inflammation": (0.9, 0.2, 0.1),    # Red
    "edema":        (0.3, 0.5, 0.9),    # Blue tint
    "osteoarthritis": (0.7, 0.6, 0.3),  # Yellow-brown
    "fracture":     (0.8, 0.8, 0.8),    # Light grey
    "tumor":        (0.6, 0.3, 0.5),    # Purple-ish
}


class PathologySystem:
    """Manages pathological condition visualizations on anatomical meshes.

    Supports:
    - Inflammation: red colour overlay + slight scale increase
    - Edema: blue tint + 10% scale increase
    - Osteoarthritis: vertex noise displacement + yellow-brown colour
    - Fracture: mesh split + fragment displacement
    - Tumor: parametric sphere mesh at surface
    """

    def __init__(self):
        self._conditions: dict[str, PathologyCondition] = {}  # keyed by mesh_name
        self._available_meshes: dict[str, MeshInstance] = {}

    def register_mesh(self, mesh: MeshInstance, name: str) -> None:
        """Register a mesh as available for pathology effects."""
        self._available_meshes[name] = mesh

    def get_available_targets(self) -> list[str]:
        """Return list of available target structure names."""
        return sorted(self._available_meshes.keys())

    def add_condition(self, mesh_name: str, condition: str,
                      severity: float = 0.5) -> bool:
        """Apply a pathological condition to a structure.

        Parameters
        ----------
        mesh_name : str
            Target structure name.
        condition : str
            Condition type.
        severity : float
            Severity from 0.0 to 1.0.

        Returns
        -------
        bool
            True if condition was applied.
        """
        mesh = self._available_meshes.get(mesh_name)
        if mesh is None:
            logger.warning("Mesh not found for pathology: %s", mesh_name)
            return False

        # Remove existing condition on this mesh first
        if mesh_name in self._conditions:
            self.remove_condition(mesh_name)

        # Save original state
        cond = PathologyCondition(
            condition=condition,
            severity=min(max(severity, 0.0), 1.0),
            mesh_name=mesh_name,
            mesh=mesh,
            original_color=mesh.material.color,
            original_positions=(
                mesh.geometry.positions.copy()
                if mesh.geometry.positions is not None else None
            ),
        )

        self._conditions[mesh_name] = cond
        self._apply_effect(cond)
        return True

    def remove_condition(self, mesh_name: str) -> None:
        """Remove pathology condition and restore original appearance."""
        cond = self._conditions.pop(mesh_name, None)
        if cond is None:
            return

        mesh = cond.mesh
        if mesh is None:
            return

        # Restore original colour
        if cond.original_color is not None:
            mesh.material.color = cond.original_color

        # Restore original positions
        if cond.original_positions is not None and mesh.geometry.positions is not None:
            mesh.geometry.positions[:] = cond.original_positions
            mesh.needs_update = True

        # Remove vertex colours
        mesh.geometry.vertex_colors = None
        mesh.geometry.vertex_colors_active = False
        mesh.needs_update = True

    def update_severity(self, mesh_name: str, severity: float) -> None:
        """Update the severity of an existing condition."""
        cond = self._conditions.get(mesh_name)
        if cond is None:
            return

        # Restore then re-apply with new severity
        mesh = cond.mesh
        if mesh is not None and cond.original_positions is not None:
            mesh.geometry.positions[:] = cond.original_positions
        if mesh is not None and cond.original_color is not None:
            mesh.material.color = cond.original_color

        cond.severity = min(max(severity, 0.0), 1.0)
        self._apply_effect(cond)

    def get_active_conditions(self) -> list[PathologyCondition]:
        """Return list of currently active conditions."""
        return list(self._conditions.values())

    def clear_all(self) -> None:
        """Remove all pathology conditions."""
        for name in list(self._conditions.keys()):
            self.remove_condition(name)

    def update(self) -> None:
        """Per-frame update (for animated effects like pulsing tumors)."""
        # Currently no per-frame animation; reserved for future use
        pass

    def _apply_effect(self, cond: PathologyCondition) -> None:
        """Apply the visual effect for a condition."""
        dispatch = {
            "inflammation": self._apply_inflammation,
            "edema": self._apply_edema,
            "osteoarthritis": self._apply_osteoarthritis,
            "fracture": self._apply_fracture,
            "tumor": self._apply_tumor,
        }
        fn = dispatch.get(cond.condition)
        if fn:
            fn(cond)

    def _apply_inflammation(self, cond: PathologyCondition) -> None:
        """Red colour overlay + slight scale increase."""
        mesh = cond.mesh
        if mesh is None:
            return

        s = cond.severity
        # Lerp colour toward red
        base = cond.original_color or (0.8, 0.7, 0.6)
        target = _CONDITION_COLORS["inflammation"]
        color = tuple(
            base[i] * (1 - s * 0.8) + target[i] * s * 0.8
            for i in range(3)
        )
        mesh.material.color = color

        # Scale increase (up to 5% at full severity)
        if cond.original_positions is not None and mesh.geometry.positions is not None:
            center = cond.original_positions.mean(axis=0)
            scale = 1.0 + s * 0.05
            mesh.geometry.positions[:] = (
                center + (cond.original_positions - center) * scale
            )
            mesh.needs_update = True

    def _apply_edema(self, cond: PathologyCondition) -> None:
        """Blue tint + 10% scale increase."""
        mesh = cond.mesh
        if mesh is None:
            return

        s = cond.severity
        base = cond.original_color or (0.8, 0.7, 0.6)
        target = _CONDITION_COLORS["edema"]
        color = tuple(
            base[i] * (1 - s * 0.6) + target[i] * s * 0.6
            for i in range(3)
        )
        mesh.material.color = color

        # Scale increase (up to 10% at full severity)
        if cond.original_positions is not None and mesh.geometry.positions is not None:
            center = cond.original_positions.mean(axis=0)
            scale = 1.0 + s * 0.10
            mesh.geometry.positions[:] = (
                center + (cond.original_positions - center) * scale
            )
            mesh.needs_update = True

    def _apply_osteoarthritis(self, cond: PathologyCondition) -> None:
        """Vertex noise displacement + yellow-brown colour."""
        mesh = cond.mesh
        if mesh is None:
            return

        s = cond.severity
        base = cond.original_color or (0.8, 0.7, 0.6)
        target = _CONDITION_COLORS["osteoarthritis"]
        color = tuple(
            base[i] * (1 - s * 0.7) + target[i] * s * 0.7
            for i in range(3)
        )
        mesh.material.color = color

        # Add surface noise (erosion effect)
        if cond.original_positions is not None and mesh.geometry.positions is not None:
            n_verts = len(cond.original_positions)
            rng = np.random.RandomState(42)  # Deterministic noise
            noise = rng.randn(n_verts, 3).astype(np.float32) * s * 0.3
            mesh.geometry.positions[:] = cond.original_positions + noise
            mesh.needs_update = True

    def _apply_fracture(self, cond: PathologyCondition) -> None:
        """Mesh fragment displacement (simple split along midline)."""
        mesh = cond.mesh
        if mesh is None or cond.original_positions is None:
            return

        s = cond.severity
        positions = cond.original_positions.copy()
        center = positions.mean(axis=0)

        # Split by X midline: displace one half
        left_mask = positions[:, 0] < center[0]
        displacement = np.array([s * 2.0, s * 0.5, s * 0.3], dtype=np.float32)
        positions[left_mask] -= displacement

        mesh.geometry.positions[:] = positions
        mesh.needs_update = True

        base = cond.original_color or (0.8, 0.7, 0.6)
        target = _CONDITION_COLORS["fracture"]
        color = tuple(
            base[i] * (1 - s * 0.3) + target[i] * s * 0.3
            for i in range(3)
        )
        mesh.material.color = color

    def _apply_tumor(self, cond: PathologyCondition) -> None:
        """Colour overlay for tumor (sphere generation deferred to scene)."""
        mesh = cond.mesh
        if mesh is None:
            return

        s = cond.severity
        base = cond.original_color or (0.8, 0.7, 0.6)
        target = _CONDITION_COLORS["tumor"]
        color = tuple(
            base[i] * (1 - s * 0.5) + target[i] * s * 0.5
            for i in range(3)
        )
        mesh.material.color = color

        # Scale: slight bulge
        if cond.original_positions is not None and mesh.geometry.positions is not None:
            center = cond.original_positions.mean(axis=0)
            scale = 1.0 + s * 0.08
            mesh.geometry.positions[:] = (
                center + (cond.original_positions - center) * scale
            )
            mesh.needs_update = True
