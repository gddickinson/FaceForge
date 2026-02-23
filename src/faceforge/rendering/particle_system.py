"""Lightweight particle system for physiological effects.

Manages point-sprite particles for blood flow, neural impulses,
and airflow visualisation. Uses numpy for bulk updates.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ParticleSystem:
    """Simple CPU-side particle system with spawn/update cycle.

    Particles are stored as parallel arrays for vectorised updates.
    Rendering is done via vertex colour arrays on a point-cloud mesh.

    Parameters
    ----------
    max_particles : int
        Maximum number of concurrent particles.
    """

    def __init__(self, max_particles: int = 5000):
        self._max = max_particles
        self._count = 0
        self._enabled = False

        # Particle data arrays
        self._positions = np.zeros((max_particles, 3), dtype=np.float32)
        self._velocities = np.zeros((max_particles, 3), dtype=np.float32)
        self._colors = np.ones((max_particles, 3), dtype=np.float32)
        self._lifetimes = np.zeros(max_particles, dtype=np.float32)
        self._ages = np.zeros(max_particles, dtype=np.float32)
        self._alive = np.zeros(max_particles, dtype=bool)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        if not enabled:
            self.clear()

    @property
    def count(self) -> int:
        return self._count

    @property
    def positions(self) -> np.ndarray:
        """Active particle positions (N x 3)."""
        return self._positions[:self._count]

    @property
    def colors(self) -> np.ndarray:
        """Active particle colours (N x 3)."""
        return self._colors[:self._count]

    def spawn(self, position: np.ndarray, velocity: np.ndarray,
              lifetime: float, color: np.ndarray) -> None:
        """Spawn a single particle.

        Parameters
        ----------
        position : ndarray (3,)
        velocity : ndarray (3,)
        lifetime : float (seconds)
        color : ndarray (3,) RGB [0,1]
        """
        if self._count >= self._max:
            # Recycle oldest particle
            idx = np.argmax(self._ages[:self._count])
        else:
            idx = self._count
            self._count += 1

        self._positions[idx] = position
        self._velocities[idx] = velocity
        self._lifetimes[idx] = lifetime
        self._ages[idx] = 0.0
        self._colors[idx] = color
        self._alive[idx] = True

    def spawn_batch(self, positions: np.ndarray, velocities: np.ndarray,
                    lifetimes: np.ndarray, colors: np.ndarray) -> None:
        """Spawn multiple particles at once."""
        n = len(positions)
        for i in range(n):
            self.spawn(positions[i], velocities[i], lifetimes[i], colors[i])

    def update(self, dt: float) -> None:
        """Advance all particles by dt seconds."""
        if not self._enabled or self._count == 0:
            return

        n = self._count

        # Age particles
        self._ages[:n] += dt

        # Kill expired particles
        expired = self._ages[:n] >= self._lifetimes[:n]

        # Move particles
        self._positions[:n] += self._velocities[:n] * dt

        # Fade colour based on age/lifetime ratio
        life_ratio = np.clip(
            1.0 - self._ages[:n] / np.maximum(self._lifetimes[:n], 0.001),
            0.0, 1.0,
        )
        # Apply alpha-like fade by scaling colour toward zero
        self._colors[:n, 0] *= life_ratio
        self._colors[:n, 1] *= life_ratio
        self._colors[:n, 2] *= life_ratio

        # Remove expired particles by compacting
        if np.any(expired):
            keep = ~expired
            n_keep = np.count_nonzero(keep)
            if n_keep < n:
                self._positions[:n_keep] = self._positions[:n][keep]
                self._velocities[:n_keep] = self._velocities[:n][keep]
                self._colors[:n_keep] = self._colors[:n][keep]
                self._lifetimes[:n_keep] = self._lifetimes[:n][keep]
                self._ages[:n_keep] = self._ages[:n][keep]
                self._count = n_keep

    def clear(self) -> None:
        """Remove all particles."""
        self._count = 0
        self._alive[:] = False


class BloodFlowParticles:
    """Blood flow particle effect along vasculature paths.

    Spawns red (arterial) and blue (venous) particles that travel
    along vessel centrelines.
    """

    def __init__(self, particle_system: ParticleSystem):
        self._ps = particle_system
        self._vessel_paths: list[np.ndarray] = []
        self._vessel_types: list[str] = []  # "artery" or "vein"
        self._spawn_timer = 0.0
        self._spawn_rate = 20.0  # particles per second
        self._cardiac_phase = 0.0

    def register_vessel(self, centreline: np.ndarray,
                        vessel_type: str = "artery") -> None:
        """Register a vessel centreline path."""
        self._vessel_paths.append(centreline)
        self._vessel_types.append(vessel_type)

    def set_cardiac_phase(self, phase: float) -> None:
        """Set current cardiac phase (0-1) for pulse-modulated speed."""
        self._cardiac_phase = phase

    def update(self, dt: float) -> None:
        """Spawn new particles along vessel paths."""
        if not self._ps.enabled or not self._vessel_paths:
            return

        self._spawn_timer += dt
        spawn_interval = 1.0 / self._spawn_rate

        while self._spawn_timer >= spawn_interval:
            self._spawn_timer -= spawn_interval
            self._spawn_particle()

    def _spawn_particle(self) -> None:
        """Spawn a single particle at a random vessel start point."""
        if not self._vessel_paths:
            return

        idx = np.random.randint(len(self._vessel_paths))
        path = self._vessel_paths[idx]
        vtype = self._vessel_types[idx]

        if len(path) < 2:
            return

        # Start at beginning of path
        pos = path[0].copy()

        # Direction along path
        direction = path[1] - path[0]
        length = np.linalg.norm(direction)
        if length > 0:
            direction /= length

        # Speed modulated by cardiac phase
        base_speed = 5.0
        speed = base_speed * (0.5 + 0.5 * np.sin(self._cardiac_phase * np.pi * 2))
        velocity = direction * speed

        # Colour: red for arteries, blue for veins
        if vtype == "artery":
            color = np.array([0.9, 0.15, 0.1], dtype=np.float32)
        else:
            color = np.array([0.2, 0.2, 0.8], dtype=np.float32)

        lifetime = length / max(speed, 0.1) if speed > 0 else 2.0
        self._ps.spawn(pos, velocity, lifetime, color)


class NeuralImpulseParticles:
    """Neural impulse particle effect along CNS structures.

    Random pulses travel along brain/nerve mesh surfaces.
    """

    def __init__(self, particle_system: ParticleSystem):
        self._ps = particle_system
        self._nerve_points: list[np.ndarray] = []
        self._spawn_timer = 0.0
        self._spawn_rate = 5.0

    def register_nerve(self, surface_points: np.ndarray) -> None:
        """Register nerve surface sample points."""
        self._nerve_points.append(surface_points)

    def update(self, dt: float) -> None:
        """Spawn impulse particles."""
        if not self._ps.enabled or not self._nerve_points:
            return

        self._spawn_timer += dt
        spawn_interval = 1.0 / self._spawn_rate

        while self._spawn_timer >= spawn_interval:
            self._spawn_timer -= spawn_interval

            # Pick random nerve
            idx = np.random.randint(len(self._nerve_points))
            points = self._nerve_points[idx]
            if len(points) == 0:
                continue

            # Random start point
            pi = np.random.randint(len(points))
            pos = points[pi].copy()

            # Random direction
            direction = np.random.randn(3).astype(np.float32)
            direction /= np.linalg.norm(direction) + 1e-6
            velocity = direction * 8.0

            # Electric blue-white colour
            color = np.array([0.5, 0.7, 1.0], dtype=np.float32)
            self._ps.spawn(pos, velocity, 0.5, color)
