"""From muscle roles and phase kind to a per-frame activation level.

The model is deliberately simple and stated, so a physiotherapist can read
what the colours mean:

* each muscle group has a *peak* level from its role (primary 0.95,
  secondary 0.6, stabiliser 0.3 unless the definition says otherwise);
* the peak is reached in a **concentric** phase;
* an **eccentric** phase runs the agonists at 75 % of that -- surface EMG is
  7-31 % lower in eccentric than in velocity-matched concentric actions
  (see ``docs/exercise_animation.md``);
* an **isometric** hold sits at 85 %;
* a **transition** (unloaded repositioning) drops to 20 %;
* stabilisers hold their level through every loaded phase.

Per-phase overrides (``Phase.activation``) replace the computed level for a
group, which is how a rowing stroke can peak the legs in the drive and the
arms only at the finish.  Levels ramp linearly across the first and last
15 % of each phase so the heatmap never snaps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from faceforge.exercise.model import ExerciseDefinition, Phase, PhaseKind, Role
from faceforge.exercise.muscle_groups import expand_group

#: Fraction of a group's peak reached in each phase kind, by role.
KIND_SCALE: dict[PhaseKind, dict[Role, float]] = {
    PhaseKind.CONCENTRIC: {Role.PRIMARY: 1.0, Role.SECONDARY: 1.0, Role.STABILISER: 1.0},
    PhaseKind.ECCENTRIC: {Role.PRIMARY: 0.75, Role.SECONDARY: 0.75, Role.STABILISER: 1.0},
    PhaseKind.ISOMETRIC: {Role.PRIMARY: 0.85, Role.SECONDARY: 0.8, Role.STABILISER: 1.0},
    PhaseKind.TRANSITION: {Role.PRIMARY: 0.2, Role.SECONDARY: 0.2, Role.STABILISER: 0.6},
}

#: Fraction of a phase spent ramping in and out of its level.
RAMP_FRACTION = 0.15


def phase_group_levels(defn: ExerciseDefinition, phase: Phase) -> dict[str, float]:
    """Activation per muscle *group* during ``phase``."""
    levels: dict[str, float] = {}
    scale = KIND_SCALE[phase.kind]
    for use in defn.muscles:
        level = use.level * scale[use.role]
        key = use.group if use.side is None else f"{use.group}:{use.side}"
        levels[key] = max(levels.get(key, 0.0), level)
    for group, override in phase.activation.items():
        levels[group] = float(override)
    return levels


def group_levels_to_muscles(levels: dict[str, float]) -> dict[str, float]:
    """Expand group levels (``"group"`` or ``"group:R"``) to mesh-name levels."""
    out: dict[str, float] = {}
    for key, level in levels.items():
        group, _, side = key.partition(":")
        for name in expand_group(group, side or None):
            out[name] = max(out.get(name, 0.0), level)
    return out


@dataclass
class ActivationTrack:
    """Piecewise-linear activation per muscle mesh over the clip's time."""

    times: np.ndarray                      # (K,) breakpoints, ascending
    levels: dict[str, np.ndarray]          # mesh name -> (K,) levels
    duration: float

    def sample(self, t: float) -> dict[str, float]:
        """Levels at time ``t`` (clamped to the clip)."""
        t = float(min(max(t, 0.0), self.duration))
        return {name: float(np.interp(t, self.times, arr)) for name, arr in self.levels.items()}

    def sample_groups(self, defn: ExerciseDefinition, t: float) -> dict[str, float]:
        """Mean level per muscle group at ``t``, for the UI list."""
        per_muscle = self.sample(t)
        out: dict[str, float] = {}
        for use in defn.muscles:
            names = expand_group(use.group, use.side)
            vals = [per_muscle.get(n, 0.0) for n in names]
            key = use.group if use.side is None else f"{use.group}:{use.side}"
            out[key] = float(max(vals)) if vals else 0.0
        return out

    @property
    def muscle_names(self) -> list[str]:
        return list(self.levels)


def build_activation_track(defn: ExerciseDefinition,
                           spans: list[tuple[float, float, Phase]]) -> ActivationTrack:
    """Build the track from ``(t0, t1, phase)`` spans covering the clip."""
    if not spans:
        return ActivationTrack(np.zeros(1), {}, 0.0)
    all_names: set[str] = set()
    per_span_levels: list[dict[str, float]] = []
    for _t0, _t1, phase in spans:
        m = group_levels_to_muscles(phase_group_levels(defn, phase))
        per_span_levels.append(m)
        all_names.update(m)

    times: list[float] = []
    columns: dict[str, list[float]] = {n: [] for n in all_names}
    for (t0, t1, _phase), m in zip(spans, per_span_levels, strict=True):
        ramp = (t1 - t0) * RAMP_FRACTION
        for t in (t0 + ramp, t1 - ramp):
            times.append(t)
            for n in all_names:
                columns[n].append(m.get(n, 0.0))
    duration = spans[-1][1]
    times_arr = np.array(times, dtype=np.float64)
    levels = {n: np.array(v, dtype=np.float64) for n, v in columns.items()}
    return ActivationTrack(times_arr, levels, float(duration))
