"""Physiological simulation systems: heartbeat, blood flow, breathing, digestion, fasciculation.

Each sub-system registers meshes from on-demand loaders, gates on group visibility,
and applies per-frame position or color deltas. All math is vectorized numpy.

Position-modifying systems apply incremental deltas to current geometry positions
(which already include soft tissue skinning output). To prevent accumulation across
frames, PhysiologySystem.step() returns True when it modified positions, and the
Simulation invalidates the soft tissue signature so positions are recalculated
fresh on the next frame.
"""

import math
import logging
from typing import Optional

import numpy as np

from faceforge.core.state import BodyState

logger = logging.getLogger(__name__)


# ── Cardiac System ──────────────────────────────────────────────────────

# Mesh names that belong to the heart (from organs.json cardiac category)
_CARDIAC_WALL_NAMES = {"Heart Wall"}
_CARDIAC_VALVE_NAMES = {"Tricuspid Valve", "Mitral Valve", "Pulmonary Valve"}
_CARDIAC_PAPILLARY_NAMES = {
    "R Vent. Ant. Pap.", "R Vent. Post. Pap.",
    "R Vent. Sept. Pap.", "L Vent. Post. Pap.",
}
_CARDIAC_NAMES = _CARDIAC_WALL_NAMES | _CARDIAC_VALVE_NAMES | _CARDIAC_PAPILLARY_NAMES


class _CardiacMesh:
    """Registered cardiac mesh with cached rest data."""
    __slots__ = ("mesh", "rest_positions", "centroid", "directions", "kind")

    def __init__(self, mesh, kind: str):
        self.mesh = mesh
        self.kind = kind  # "wall", "valve", or "papillary"
        pos = mesh.geometry.positions.reshape(-1, 3)
        self.rest_positions = pos.copy()
        self.centroid = pos.mean(axis=0)
        diff = pos - self.centroid
        norms = np.linalg.norm(diff, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        self.directions = diff / norms


class CardiacSystem:
    """Heart contraction: radial scale toward centroid during systole."""

    SYSTOLE_FRACTION = 0.35  # first 35% of cycle is systole
    WALL_CONTRACTION = 0.06  # 6% inward at peak
    VALVE_EXPANSION = 0.08   # 8% outward at peak (opening)

    def __init__(self):
        self._meshes: list[_CardiacMesh] = []
        self._phase: float = 0.0  # 0-1 within cardiac cycle

    def register(self, mesh, name: str) -> bool:
        if name in _CARDIAC_WALL_NAMES:
            self._meshes.append(_CardiacMesh(mesh, "wall"))
            return True
        elif name in _CARDIAC_VALVE_NAMES:
            self._meshes.append(_CardiacMesh(mesh, "valve"))
            return True
        elif name in _CARDIAC_PAPILLARY_NAMES:
            self._meshes.append(_CardiacMesh(mesh, "papillary"))
            return True
        return False

    @property
    def phase(self) -> float:
        return self._phase

    def step(self, dt: float, heart_rate: float) -> bool:
        """Advance cardiac cycle and apply contraction. Returns True if positions modified."""
        if not self._meshes:
            return False
        # Advance phase: heart_rate in BPM → cycles/sec
        cps = heart_rate / 60.0
        self._phase = (self._phase + dt * cps) % 1.0

        # Cardiac envelope: sin²(t_norm * π) during systole, exp decay during diastole
        t = self._phase
        if t < self.SYSTOLE_FRACTION:
            t_norm = t / self.SYSTOLE_FRACTION
            envelope = math.sin(t_norm * math.pi) ** 2
        else:
            # Exponential decay from end of systole
            t_dias = (t - self.SYSTOLE_FRACTION) / (1.0 - self.SYSTOLE_FRACTION)
            envelope = math.exp(-4.0 * t_dias)

        for cm in self._meshes:
            if cm.kind == "wall" or cm.kind == "papillary":
                # Contract inward
                scale = -self.WALL_CONTRACTION * envelope
            else:
                # Valves expand outward
                scale = self.VALVE_EXPANSION * envelope

            # Apply radial displacement to current positions (fresh from soft tissue)
            disp = cm.directions * scale
            pos = cm.mesh.geometry.positions.reshape(-1, 3)
            np.add(pos, disp, out=pos)
            cm.mesh.geometry.positions = pos.ravel()
            cm.mesh.needs_update = True

        return True

    def reset(self) -> None:
        self._phase = 0.0


# ── Pulse Wave System ───────────────────────────────────────────────────

class _VascularMesh:
    """Registered vascular mesh with distance-from-heart per vertex."""
    __slots__ = ("mesh", "distances", "vtype", "base_color", "pulse_color")

    def __init__(self, mesh, vtype: str, heart_z: float):
        self.mesh = mesh
        self.vtype = vtype  # "artery" or "vein"
        pos = mesh.geometry.positions.reshape(-1, 3)
        # Distance from heart centroid along Z
        self.distances = np.abs(pos[:, 2] - heart_z).astype(np.float32)
        # Normalize to 0-1 range
        d_max = self.distances.max()
        if d_max > 1e-6:
            self.distances /= d_max

        if vtype == "artery":
            self.base_color = np.array([0.55, 0.08, 0.08], dtype=np.float32)  # dark red
            self.pulse_color = np.array([1.0, 0.15, 0.1], dtype=np.float32)   # bright red
        else:
            self.base_color = np.array([0.15, 0.15, 0.45], dtype=np.float32)  # dark blue
            self.pulse_color = np.array([0.25, 0.3, 0.7], dtype=np.float32)   # brighter blue


class PulseWaveSystem:
    """Per-vertex color pulse traveling outward from heart along vasculature."""

    PULSE_WIDTH = 0.15      # Gaussian width of pulse
    VEIN_DELAY = 0.4        # fraction of beat delayed for veins
    PULSE_SPEED = 1.0       # normalized distance per beat

    def __init__(self):
        self._meshes: list[_VascularMesh] = []
        self._heart_z: float = 0.0
        self._heart_z_set: bool = False

    def set_heart_centroid_z(self, z: float) -> None:
        self._heart_z = z
        self._heart_z_set = True

    def register(self, mesh, name: str, vtype: str) -> None:
        self._meshes.append(_VascularMesh(mesh, vtype, self._heart_z))

    def step(self, cardiac_phase: float) -> None:
        if not self._meshes:
            return

        for vm in self._meshes:
            phase = cardiac_phase
            if vm.vtype == "vein":
                phase = (phase - self.VEIN_DELAY) % 1.0

            # Pulse position travels outward: at phase=0 pulse is at d=0 (heart)
            pulse_center = phase * self.PULSE_SPEED

            # Gaussian intensity per vertex
            diff = vm.distances - pulse_center
            intensity = np.exp(-0.5 * (diff / self.PULSE_WIDTH) ** 2)

            # Lerp between base and pulse color
            nv = len(intensity)
            colors = np.empty((nv, 3), dtype=np.float32)
            for c in range(3):
                colors[:, c] = vm.base_color[c] + intensity * (vm.pulse_color[c] - vm.base_color[c])

            vm.mesh.geometry.vertex_colors = colors.ravel()
            vm.mesh.geometry.colors_dirty = True
            vm.mesh.material.vertex_colors_active = True

    def reset(self) -> None:
        for vm in self._meshes:
            vm.mesh.material.vertex_colors_active = False


# ── Lung Expansion System ───────────────────────────────────────────────

_LUNG_LOBE_NAMES = {
    "R Lung Upper Lobe", "R Lung Lower Lobe", "R Lung Middle Lobe",
    "L Lung Upper Lobe", "L Lung Lower Lobe",
}


class _LungMesh:
    """Registered lung lobe mesh with cached rest data."""
    __slots__ = ("mesh", "rest_positions", "centroid", "directions")

    def __init__(self, mesh):
        self.mesh = mesh
        pos = mesh.geometry.positions.reshape(-1, 3)
        self.rest_positions = pos.copy()
        self.centroid = pos.mean(axis=0)
        diff = pos - self.centroid
        norms = np.linalg.norm(diff, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        self.directions = diff / norms


class LungExpandSystem:
    """Lung lobe radial expansion synced with existing breath_phase_body."""

    EXPANSION_SCALE = 0.04  # max radial expansion factor

    def __init__(self):
        self._meshes: list[_LungMesh] = []

    def register(self, mesh, name: str) -> bool:
        if name in _LUNG_LOBE_NAMES:
            self._meshes.append(_LungMesh(mesh))
            return True
        return False

    def step(self, breath_phase: float, breath_depth: float) -> bool:
        """Apply lung expansion. Returns True if positions modified."""
        if not self._meshes:
            return False
        # Expansion factor: positive during inhale (sin > 0)
        expansion = math.sin(breath_phase) * breath_depth * self.EXPANSION_SCALE

        for lm in self._meshes:
            disp = lm.directions * expansion
            pos = lm.mesh.geometry.positions.reshape(-1, 3)
            np.add(pos, disp, out=pos)
            lm.mesh.geometry.positions = pos.ravel()
            lm.mesh.needs_update = True

        return True

    def reset(self) -> None:
        pass


# ── Peristalsis System ──────────────────────────────────────────────────

_DIGESTIVE_RATES: dict[str, float] = {
    "Esophagus": 0.1,
    "Stomach": 0.05,
    "Duodenum": 0.15,
    "Jejunum": 0.2,
    "Ileum": 0.2,
    "Colon": 0.05,
    "Rectum": 0.05,
}


class _DigestiveMesh:
    """Registered digestive mesh with per-vertex normalized Z and radial directions."""
    __slots__ = ("mesh", "rest_positions", "centroid", "radial_dirs",
                 "z_normalized", "base_rate", "phase")

    def __init__(self, mesh, name: str):
        self.mesh = mesh
        pos = mesh.geometry.positions.reshape(-1, 3)
        self.rest_positions = pos.copy()
        self.centroid = pos.mean(axis=0)

        # Radial direction from centroid axis (XY plane only for tube organs)
        diff_xy = pos[:, :2] - self.centroid[:2]
        radial_norms = np.linalg.norm(diff_xy, axis=1, keepdims=True)
        radial_norms[radial_norms < 1e-8] = 1.0
        self.radial_dirs = np.zeros_like(pos)
        self.radial_dirs[:, :2] = diff_xy / radial_norms

        # Normalize Z along organ length
        z_vals = pos[:, 2]
        z_min, z_max = z_vals.min(), z_vals.max()
        z_range = z_max - z_min
        if z_range > 1e-6:
            self.z_normalized = ((z_vals - z_min) / z_range).astype(np.float32)
        else:
            self.z_normalized = np.zeros(len(z_vals), dtype=np.float32)

        self.base_rate = _DIGESTIVE_RATES.get(name, 0.1)
        self.phase = 0.0


class PeristalsisSystem:
    """Traveling radial compression wave along digestive organs."""

    COMPRESSION_AMPLITUDE = 0.06  # 6% radial compression at peak
    WAVE_WIDTH = 0.2              # width of compression wave in normalized Z

    def __init__(self):
        self._meshes: list[_DigestiveMesh] = []

    def register(self, mesh, name: str) -> bool:
        if name in _DIGESTIVE_RATES:
            self._meshes.append(_DigestiveMesh(mesh, name))
            return True
        return False

    def step(self, dt: float, rate_scale: float) -> bool:
        """Apply peristalsis wave. Returns True if positions modified."""
        if not self._meshes:
            return False

        for dm in self._meshes:
            # Advance phase at organ-specific rate, scaled by user slider
            effective_rate = dm.base_rate * (0.5 + rate_scale * 1.5)
            dm.phase = (dm.phase + dt * effective_rate) % 1.0

            # Traveling wave: compression peak at wave_center
            wave_center = dm.phase
            diff = dm.z_normalized - wave_center
            # Wrap around for continuous wave
            diff = np.minimum(np.abs(diff), np.abs(diff + 1.0))
            diff = np.minimum(diff, np.abs(diff - 1.0))
            # Gaussian compression profile
            compression = np.exp(-0.5 * (diff / self.WAVE_WIDTH) ** 2)

            # Radial compression (negative = inward)
            scale = -self.COMPRESSION_AMPLITUDE * compression
            disp = dm.radial_dirs * scale[:, np.newaxis]
            pos = dm.mesh.geometry.positions.reshape(-1, 3)
            np.add(pos, disp, out=pos)
            dm.mesh.geometry.positions = pos.ravel()
            dm.mesh.needs_update = True

        return True

    def reset(self) -> None:
        for dm in self._meshes:
            dm.phase = 0.0


# ── Fasciculation System ────────────────────────────────────────────────

class _FasciculationMesh:
    """Registered muscle mesh with pre-computed random twitch data."""
    __slots__ = ("mesh", "active_mask", "directions", "frequencies", "phases")

    def __init__(self, mesh, rng: np.random.Generator):
        self.mesh = mesh
        nv = mesh.geometry.positions.shape[0] // 3

        # ~12% of vertices are active fasciculation sites
        self.active_mask = rng.random(nv) < 0.12
        n_active = self.active_mask.sum()

        # Random displacement directions (unit vectors)
        dirs = rng.standard_normal((n_active, 3)).astype(np.float32)
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        self.directions = dirs / norms

        # Random frequencies 2-8 Hz per active vertex
        self.frequencies = rng.uniform(2.0, 8.0, size=n_active).astype(np.float32)

        # Random initial phases
        self.phases = rng.uniform(0.0, 2.0 * np.pi, size=n_active).astype(np.float32)


class FasciculationSystem:
    """Subtle random muscle twitching via per-vertex sinusoidal displacement."""

    AMPLITUDE = 0.3  # world units at intensity=1.0

    def __init__(self):
        self._meshes: list[_FasciculationMesh] = []
        self._rng = np.random.default_rng(42)
        self._time: float = 0.0

    def register(self, mesh, name: str) -> None:
        self._meshes.append(_FasciculationMesh(mesh, self._rng))

    def step(self, dt: float, intensity: float) -> bool:
        """Apply fasciculation. Returns True if positions modified."""
        if not self._meshes or intensity < 1e-4:
            return False
        self._time += dt
        amp = self.AMPLITUDE * intensity

        for fm in self._meshes:
            n_active = fm.active_mask.sum()
            if n_active == 0:
                continue

            # Sinusoidal displacement per active vertex
            t_arr = self._time * fm.frequencies + fm.phases
            sin_vals = np.sin(t_arr) * amp

            # Build displacement array for all vertices
            pos = fm.mesh.geometry.positions.reshape(-1, 3)
            disp = fm.directions * sin_vals[:, np.newaxis]
            pos[fm.active_mask] += disp
            fm.mesh.geometry.positions = pos.ravel()
            fm.mesh.needs_update = True

        return True

    def reset(self) -> None:
        self._time = 0.0


# ── Physiology System Orchestrator ──────────────────────────────────────

class PhysiologySystem:
    """Orchestrates all physiological sub-systems.

    Each sub-system is gated on its respective group visibility.

    Position-modifying sub-systems add incremental deltas to current geometry
    positions (which are fresh from soft tissue skinning).  When any sub-system
    modifies positions, ``step()`` returns ``True`` so the caller can invalidate
    the soft tissue signature — ensuring soft tissue recalculates from scratch
    on the next frame rather than skipping and letting deltas accumulate.
    """

    def __init__(self):
        self.cardiac = CardiacSystem()
        self.pulse_wave = PulseWaveSystem()
        self.lung_expand = LungExpandSystem()
        self.peristalsis = PeristalsisSystem()
        self.fasciculation = FasciculationSystem()

        # Group references for visibility gating (set by app.py)
        self.organ_group = None    # SceneNode
        self.vascular_group = None  # SceneNode
        self.muscle_groups: list = []  # list of SceneNode

    @staticmethod
    def _is_visible(node) -> bool:
        """Check if a node and all ancestors are visible."""
        while node is not None:
            if not node.visible:
                return False
            node = node.parent
        return True

    def register_organ(self, mesh, name: str, category: str) -> None:
        """Register an organ mesh for appropriate sub-systems."""
        self.cardiac.register(mesh, name)
        self.lung_expand.register(mesh, name)
        self.peristalsis.register(mesh, name)
        # Set heart centroid Z for pulse wave system from Heart Wall
        if name == "Heart Wall":
            pos = mesh.geometry.positions.reshape(-1, 3)
            heart_z = pos[:, 2].mean()
            self.pulse_wave.set_heart_centroid_z(heart_z)

    def register_vascular(self, mesh, name: str, vtype: str) -> None:
        """Register a vascular mesh for pulse wave system."""
        self.pulse_wave.register(mesh, name, vtype)

    def register_muscle(self, mesh, name: str) -> None:
        """Register a muscle mesh for fasciculation system."""
        self.fasciculation.register(mesh, name)

    def step(self, body: BodyState, dt: float) -> bool:
        """Run all active sub-systems for one frame.

        Returns True if any position-modifying sub-system ran, signaling
        the caller to invalidate the soft tissue skinning cache.
        """
        organs_visible = self._is_visible(self.organ_group)
        vasc_visible = self._is_visible(self.vascular_group)
        any_muscle_visible = any(self._is_visible(g) for g in self.muscle_groups)

        modified_positions = False

        # Cardiac (heart contraction)
        if body.auto_heartbeat and organs_visible:
            if self.cardiac.step(dt, body.heart_rate):
                modified_positions = True

        # Pulse wave (per-vertex vascular color — does NOT modify positions)
        if body.auto_pulse_wave and vasc_visible:
            self.pulse_wave.step(self.cardiac.phase)
        elif not body.auto_pulse_wave:
            # Ensure colors are reset when toggled off
            self.pulse_wave.reset()

        # Lung expansion (synced with existing breathing)
        if body.auto_lung_expand and organs_visible:
            if self.lung_expand.step(body.breath_phase_body, body.breath_depth):
                modified_positions = True

        # Peristalsis (digestive wave)
        if body.auto_peristalsis and organs_visible:
            if self.peristalsis.step(dt, body.peristalsis_rate):
                modified_positions = True

        # Fasciculation (muscle twitching)
        if body.auto_fasciculation and any_muscle_visible:
            if self.fasciculation.step(dt, body.fasciculation_intensity):
                modified_positions = True

        return modified_positions
