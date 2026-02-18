"""Automatic blinking system."""

import random

from faceforge.core.state import FaceState


class AutoBlink:
    """Generates natural-looking blink patterns.

    Blink cycle: wait → close → open
    Random interval between 2-6 seconds.
    Blink duration ~0.15s close + ~0.15s open.
    """

    BLINK_CLOSE_DURATION = 0.08
    BLINK_OPEN_DURATION = 0.12
    MIN_INTERVAL = 2.0
    MAX_INTERVAL = 6.0

    def __init__(self):
        self._timer = 0.0
        self._next_blink = self._random_interval()
        self._blinking = False
        self._blink_phase = 0.0

    def update(self, face: FaceState, dt: float) -> None:
        if not face.auto_blink:
            return

        if self._blinking:
            self._blink_phase += dt
            total = self.BLINK_CLOSE_DURATION + self.BLINK_OPEN_DURATION
            if self._blink_phase < self.BLINK_CLOSE_DURATION:
                # Closing
                face.blink_amount = self._blink_phase / self.BLINK_CLOSE_DURATION
            elif self._blink_phase < total:
                # Opening
                t = (self._blink_phase - self.BLINK_CLOSE_DURATION) / self.BLINK_OPEN_DURATION
                face.blink_amount = 1.0 - t
            else:
                # Done
                face.blink_amount = 0.0
                self._blinking = False
                self._timer = 0.0
                self._next_blink = self._random_interval()
        else:
            self._timer += dt
            if self._timer >= self._next_blink:
                self._blinking = True
                self._blink_phase = 0.0

    def _random_interval(self) -> float:
        return random.uniform(self.MIN_INTERVAL, self.MAX_INTERVAL)
