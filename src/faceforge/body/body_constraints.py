"""Body joint limit enforcement via simple clamping.

Clamps BodyState DOF values to physiological limits loaded from config.
Unlike the neck constraint solver, body constraints are pure clamp-based —
muscle activation coloring comes from soft tissue stretch ratios instead.
"""

import logging
from typing import Any

from faceforge.core.config_loader import load_config
from faceforge.core.state import BodyState

logger = logging.getLogger(__name__)


class BodyConstraints:
    """Clamp body DOFs to physiological joint limits."""

    def __init__(self):
        self._limits: dict[str, tuple[float, float]] = {}

    def load(self) -> None:
        """Load joint limits from config.  Graceful no-op on failure."""
        try:
            data = load_config("body_joint_limits.json")
        except (FileNotFoundError, ValueError) as e:
            logger.warning("Body joint limits config not found, constraints disabled: %s", e)
            return

        raw = data.get("limits", {})
        for key, bounds in raw.items():
            lo = float(bounds.get("min", -1.0))
            hi = float(bounds.get("max", 1.0))
            if "{s}" in key:
                # Expand bilateral template for both sides
                for side in ("r", "l"):
                    expanded = key.replace("{s}", side)
                    self._limits[expanded] = (lo, hi)
            else:
                self._limits[key] = (lo, hi)

    def clamp(self, state: BodyState) -> None:
        """Clamp all DOF values in *state* to their configured limits."""
        if not self._limits:
            return
        for attr, (lo, hi) in self._limits.items():
            val = getattr(state, attr, None)
            if val is None:
                continue
            clamped = max(lo, min(hi, val))
            if clamped != val:
                setattr(state, attr, clamped)
