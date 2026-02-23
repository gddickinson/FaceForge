"""Neural impulse particle effect module.

Generates electrical pulse visualisations along CNS structures.
"""

import logging
from typing import Optional

import numpy as np

from faceforge.rendering.particle_system import ParticleSystem, NeuralImpulseParticles

logger = logging.getLogger(__name__)


class NeuralImpulseSystem:
    """Manages neural impulse particles along brain/CNS meshes.

    Call ``register_nerves`` after brain meshes are loaded, then
    ``update(dt)`` each frame.
    """

    def __init__(self, particle_system: ParticleSystem):
        self._particles = NeuralImpulseParticles(particle_system)
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def register_nerves(self, brain_meshes: list) -> None:
        """Register brain/CNS meshes for neural impulse effects.

        Parameters
        ----------
        brain_meshes : list[MeshInstance]
            Loaded brain mesh instances.
        """
        for mesh in brain_meshes:
            if mesh.geometry.positions is None:
                continue

            positions = mesh.geometry.positions
            if len(positions) < 5:
                continue

            # Subsample surface points for efficiency
            n = len(positions)
            if n > 200:
                indices = np.random.choice(n, 200, replace=False)
                points = positions[indices].copy()
            else:
                points = positions.copy()

            self._particles.register_nerve(points)

        logger.info("Registered %d nerve surfaces for impulse effects",
                     len(self._particles._nerve_points))

    def update(self, dt: float) -> None:
        """Update neural impulse particles."""
        if not self._enabled:
            return
        self._particles.update(dt)
