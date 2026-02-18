"""Delta clock for frame timing."""

import time

from faceforge.constants import MAX_DELTA_TIME


class DeltaClock:
    """Tracks elapsed time between frames."""

    def __init__(self):
        self._last_time = time.perf_counter()

    def get_delta(self) -> float:
        """Return seconds elapsed since last call, clamped to MAX_DELTA_TIME."""
        now = time.perf_counter()
        dt = now - self._last_time
        self._last_time = now
        return min(dt, MAX_DELTA_TIME)

    def reset(self) -> None:
        self._last_time = time.perf_counter()
