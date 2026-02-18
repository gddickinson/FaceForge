"""Tests for auto-blink system."""

from faceforge.core.state import FaceState
from faceforge.animation.auto_blink import AutoBlink


def test_blink_eventually_triggers():
    face = FaceState()
    face.auto_blink = True
    blinker = AutoBlink()

    # Force the blink to trigger immediately
    blinker._next_blink = 0.001
    blinker._timer = 0.0
    blinker.update(face, 0.01)  # This triggers the blink
    blinker.update(face, 0.01)  # This advances into closing phase
    assert face.blink_amount > 0.0


def test_blink_disabled():
    face = FaceState()
    face.auto_blink = False
    blinker = AutoBlink()

    for _ in range(1000):
        blinker.update(face, 0.016)
    assert face.blink_amount == 0.0


def test_blink_completes():
    face = FaceState()
    face.auto_blink = True
    blinker = AutoBlink()

    # Force a blink
    blinker._blinking = True
    blinker._blink_phase = 0.0

    # Run through the blink
    for _ in range(50):
        blinker.update(face, 0.01)

    # After the blink completes, amount should return to 0
    assert face.blink_amount == 0.0
    assert blinker._blinking is False
