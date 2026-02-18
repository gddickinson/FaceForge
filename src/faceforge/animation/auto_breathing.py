"""Automatic breathing cycle for face (nostril flare + subtle jaw)."""

import math

from faceforge.core.state import FaceState


class AutoBreathing:
    """Generates subtle breathing animation on the face.

    Cycles AU9 (nose wrinkle/flare) and AU25 (lips part) slightly
    with a sinusoidal breathing rhythm.
    """

    # Subtle breathing amplitudes — small so they don't override expressions
    NOSTRIL_AMPLITUDE = 0.06   # AU9 peak
    LIP_PART_AMPLITUDE = 0.03  # AU25 peak

    def update(self, face: FaceState, dt: float) -> None:
        if not face.auto_breathing:
            return

        face.breath_phase += dt * 2.5  # ~0.4 Hz breathing rate
        if face.breath_phase > 2 * math.pi:
            face.breath_phase -= 2 * math.pi

        breath = (math.sin(face.breath_phase) + 1.0) * 0.5  # 0..1

        # Add subtle nostril flare and lip parting during inhale
        face.AU9 = max(face.AU9, breath * self.NOSTRIL_AMPLITUDE)
        face.AU25 = max(face.AU25, breath * self.LIP_PART_AMPLITUDE)
