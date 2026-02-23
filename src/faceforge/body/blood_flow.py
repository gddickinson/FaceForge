"""Blood flow particle effect module.

Extracts vessel centrelines and manages blood flow particle spawning.
Wrapper around the ParticleSystem for vascular-specific logic.
"""

import logging
from typing import Optional

import numpy as np

from faceforge.rendering.particle_system import ParticleSystem, BloodFlowParticles

logger = logging.getLogger(__name__)


class BloodFlowSystem:
    """Manages blood flow particles along vasculature.

    Call ``register_vessels`` after vascular meshes are loaded, then
    ``update(dt)`` each frame.
    """

    def __init__(self, particle_system: ParticleSystem):
        self._particles = BloodFlowParticles(particle_system)
        self._enabled = False
        self._heart_rate = 72.0  # BPM
        self._cardiac_time = 0.0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def set_heart_rate(self, bpm: float) -> None:
        self._heart_rate = max(30.0, min(200.0, bpm))

    def register_vessels(self, vascular_meshes: list, vascular_defs: list) -> None:
        """Extract centrelines from vascular meshes and register paths.

        Parameters
        ----------
        vascular_meshes : list[MeshInstance]
            Loaded vascular mesh instances.
        vascular_defs : list[dict]
            Vascular definitions with type info.
        """
        for mesh, defn in zip(vascular_meshes, vascular_defs):
            if mesh.geometry.positions is None:
                continue

            positions = mesh.geometry.positions
            if len(positions) < 10:
                continue

            # Extract centreline by Z-slice averaging
            centreline = self._extract_centreline(positions)
            vtype = defn.get("type", "artery")

            if len(centreline) >= 2:
                self._particles.register_vessel(centreline, vtype)

        logger.info("Registered %d vessel paths for blood flow",
                     len(self._particles._vessel_paths))

    def update(self, dt: float) -> None:
        """Update blood flow particles."""
        if not self._enabled:
            return

        # Advance cardiac cycle
        beats_per_sec = self._heart_rate / 60.0
        self._cardiac_time += dt * beats_per_sec
        phase = self._cardiac_time % 1.0

        self._particles.set_cardiac_phase(phase)
        self._particles.update(dt)

    @staticmethod
    def _extract_centreline(positions: np.ndarray,
                            n_slices: int = 20) -> np.ndarray:
        """Extract a rough centreline by averaging positions in Z slices."""
        z_min, z_max = positions[:, 2].min(), positions[:, 2].max()
        if z_max - z_min < 1.0:
            return np.array([positions.mean(axis=0)])

        z_edges = np.linspace(z_min, z_max, n_slices + 1)
        centreline = []

        for i in range(n_slices):
            mask = (positions[:, 2] >= z_edges[i]) & (positions[:, 2] < z_edges[i + 1])
            if mask.any():
                centreline.append(positions[mask].mean(axis=0))

        return np.array(centreline, dtype=np.float32) if centreline else np.array([])
