"""Muscle activation heatmap: colour each muscle by how hard it is working.

Two sources of activation, one colouring path:

* **Pose-derived** (the default).  Activation = the largest normalised DOF
  value across the joints the muscle crosses, from
  ``assets/config/muscle_dof_map.json``.  A crude but honest proxy for a
  static pose.
* **External levels**, set by :meth:`MuscleActivationSystem.set_levels`.  The
  exercise system samples a per-muscle activation track built from the
  exercise's muscle roles and phase (concentric / eccentric / isometric) and
  pushes it here every frame.  While levels are set, muscles the track does
  not mention are drawn at zero, so the heatmap reads as "this exercise".

Colour is written as per-vertex colours and switched on through the
material's ``vertex_colors_active`` flag, which is what the renderer's shader
uniform reads (``gl_material.py``).  ``geometry.colors_dirty`` is what makes
the renderer re-upload the colour buffer.  An earlier version of this module
set a ``vertex_colors_active`` attribute on the *geometry* (which nothing
reads) and never set ``colors_dirty``, so the heatmap toggle changed nothing
on screen.

Levels follow the DiGiovine et al. (1992) EMG bands used throughout the
exercise literature: low 0-20 %MVIC, moderate 21-40, high 41-60, very high
>60.  :func:`level_band` names the band for the UI.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from faceforge.core.mesh import MeshInstance

logger = logging.getLogger(__name__)

#: DiGiovine et al. (1992) activation bands, as (upper bound, name).
ACTIVATION_BANDS: tuple[tuple[float, str], ...] = (
    (0.20, "low"), (0.40, "moderate"), (0.60, "high"), (1.01, "very high"),
)


def level_band(level: float) -> str:
    """Name the DiGiovine band a 0-1 activation level falls in."""
    for upper, name in ACTIVATION_BANDS:
        if level <= upper:
            return name
    return ACTIVATION_BANDS[-1][1]


def _lerp_ramp(stops: list[tuple[float, tuple[float, float, float]]]) -> np.ndarray:
    """A 256-entry RGB lookup table interpolated between ``(t, rgb)`` stops."""
    lut = np.zeros((256, 3), dtype=np.float32)
    ts = np.array([s[0] for s in stops])
    cols = np.array([s[1] for s in stops], dtype=np.float32)
    for i in range(256):
        t = i / 255.0
        k = int(np.searchsorted(ts, t, side="right")) - 1
        k = max(0, min(k, len(stops) - 2))
        t0, t1 = ts[k], ts[k + 1]
        f = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        lut[i] = cols[k] * (1.0 - f) + cols[k + 1] * f
    return lut


#: Colour ramps by name.  ``classic`` is the original blue-to-red rainbow;
#: ``thermal`` is a sequential ramp (dark plum -> magenta -> orange -> yellow)
#: that stays ordered in luminance, so it reads correctly in greyscale and
#: for the common red-green colour-vision deficiencies.
PALETTES: dict[str, np.ndarray] = {
    "classic": _lerp_ramp([
        (0.00, (0.0, 0.0, 1.0)), (0.25, (0.0, 1.0, 1.0)), (0.50, (0.0, 1.0, 0.0)),
        (0.75, (1.0, 1.0, 0.0)), (1.00, (1.0, 0.0, 0.0)),
    ]),
    "thermal": _lerp_ramp([
        (0.00, (0.13, 0.08, 0.22)), (0.25, (0.45, 0.10, 0.50)),
        (0.50, (0.80, 0.20, 0.35)), (0.75, (0.98, 0.55, 0.15)),
        (1.00, (1.00, 0.92, 0.45)),
    ]),
}


def _load_dof_map() -> dict[str, list[str]]:
    """Load DOF -> muscle mapping from config."""
    config_dir = Path(__file__).resolve().parents[2] / "assets" / "config"
    path = config_dir / "muscle_dof_map.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


class MuscleActivationSystem:
    """Computes per-muscle activation and applies vertex colours.

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
        self._muscle_dofs: dict[str, list[str]] = {}
        for dof, muscles in self._dof_map.items():
            for m in muscles:
                self._muscle_dofs.setdefault(m, []).append(dof)

        # name -> (mesh, original vertex colours or None)
        self._muscles: dict[str, tuple[MeshInstance, Optional[np.ndarray]]] = {}
        self._enabled = False
        self._levels: Optional[dict[str, float]] = None
        self._current: dict[str, float] = {}
        self._palette = "classic"

    # -- registration ------------------------------------------------------

    def register_muscle(self, mesh: MeshInstance, name: str) -> None:
        """Register a muscle mesh for heatmap colouring (idempotent by name)."""
        original = None
        if mesh.geometry.vertex_colors is not None:
            original = mesh.geometry.vertex_colors.copy()
        self._muscles[name] = (mesh, original)

    @property
    def muscle_names(self) -> list[str]:
        return list(self._muscles)

    # -- switches ----------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        """Toggle the heatmap.  Turning it off restores the original colours."""
        was_enabled = self._enabled
        self._enabled = enabled
        if not enabled and was_enabled:
            self._restore_colors()

    @property
    def palette(self) -> str:
        return self._palette

    def set_palette(self, name: str) -> None:
        if name not in PALETTES:
            raise KeyError(f"unknown heatmap palette {name!r}; have {sorted(PALETTES)}")
        self._palette = name

    def set_levels(self, levels: Optional[dict[str, float]]) -> None:
        """Drive the heatmap from external per-muscle levels (0-1).

        ``None`` returns to the pose-derived estimate.  Muscles absent from
        ``levels`` are drawn at zero while levels are set.
        """
        self._levels = None if levels is None else dict(levels)

    @property
    def levels_are_external(self) -> bool:
        return self._levels is not None

    @property
    def current_levels(self) -> dict[str, float]:
        """The activation applied on the last :meth:`update`, by muscle name."""
        return dict(self._current)

    # -- per frame ---------------------------------------------------------

    def activation_for(self, name: str, body_state) -> float:
        """Activation of one registered muscle for the current source."""
        if self._levels is not None:
            return float(min(max(self._levels.get(name, 0.0), 0.0), 1.0))
        activation = 0.0
        for dof_field in self._muscle_dofs.get(name, []):
            val = getattr(body_state, dof_field, 0.0)
            if isinstance(val, (int, float)):
                activation = max(activation, abs(float(val)))
        return min(activation, 1.0)

    def update(self, body_state) -> None:
        """Compute activation for each muscle and apply vertex colours."""
        if not self._enabled:
            return
        lut = PALETTES[self._palette]
        current: dict[str, float] = {}
        for name, (mesh, _original) in self._muscles.items():
            activation = self.activation_for(name, body_state)
            current[name] = activation
            n_verts = mesh.geometry.vertex_count
            if n_verts <= 0:
                continue
            color = lut[int(activation * 255)]
            colors = np.broadcast_to(color, (n_verts, 3)).copy().astype(np.float32)
            mesh.geometry.vertex_colors = colors
            mesh.geometry.colors_dirty = True
            mesh.material.vertex_colors_active = True
        self._current = current

    def _restore_colors(self) -> None:
        """Restore original vertex colours on all registered muscles."""
        for _name, (mesh, original) in self._muscles.items():
            mesh.geometry.vertex_colors = original.copy() if original is not None else None
            mesh.geometry.colors_dirty = original is not None
            mesh.material.vertex_colors_active = False
        self._current = {}
