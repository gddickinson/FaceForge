"""Micro-expression generator: random subtle AU flickers."""

import random

from faceforge.core.state import FaceState, TargetAU, AU_IDS


class MicroExpressionGen:
    """Adds subtle random AU fluctuations for lifelike appearance.

    Periodically selects a random AU and applies a small target change,
    which the interpolation system smoothly applies.
    """

    INTERVAL_MIN = 1.0
    INTERVAL_MAX = 4.0
    INTENSITY_MAX = 0.15

    def __init__(self):
        self._timer = 0.0
        self._next_fire = self._random_interval()
        self._active_au: str | None = None
        self._original_value: float = 0.0
        self._duration = 0.0
        self._phase = 0.0

    def update(self, face: FaceState, target_au: TargetAU, dt: float) -> None:
        if not face.micro_expressions:
            return

        # If currently showing a micro-expression, track its duration
        if self._active_au is not None:
            self._phase += dt
            if self._phase >= self._duration:
                # Restore original target
                target_au.set(self._active_au, self._original_value)
                self._active_au = None
                self._timer = 0.0
                self._next_fire = self._random_interval()
            return

        self._timer += dt
        if self._timer >= self._next_fire:
            # Fire a micro-expression
            au = random.choice(AU_IDS)
            self._active_au = au
            self._original_value = target_au.get(au)
            self._duration = random.uniform(0.2, 0.5)
            self._phase = 0.0

            # Small random addition
            intensity = random.uniform(0.02, self.INTENSITY_MAX)
            new_val = min(1.0, self._original_value + intensity)
            target_au.set(au, new_val)

    def _random_interval(self) -> float:
        return random.uniform(self.INTERVAL_MIN, self.INTERVAL_MAX)
