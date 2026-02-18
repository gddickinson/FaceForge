"""Tests for state interpolation."""

import pytest

from faceforge.core.state import FaceState, BodyState, TargetAU, TargetHead
from faceforge.animation.interpolation import StateInterpolator


def test_au_interpolation():
    face = FaceState()
    target = TargetAU()
    target.set("AU12", 1.0)
    interp = StateInterpolator()

    # After one step, should move toward target
    interp._interpolate_aus(face, target, 0.016)
    assert face.AU12 > 0.0
    assert face.AU12 < 1.0

    # After many steps, should be close to target
    for _ in range(100):
        interp._interpolate_aus(face, target, 0.016)
    assert abs(face.AU12 - 1.0) < 0.01


def test_head_interpolation():
    face = FaceState()
    target = TargetHead()
    target.head_yaw = 0.5
    interp = StateInterpolator()

    for _ in range(100):
        interp._interpolate_head(face, target, 0.016)
    assert abs(face.head_yaw - 0.5) < 0.01


def test_body_interpolation():
    body = BodyState()
    target = BodyState()
    target.spine_flex = 0.8
    interp = StateInterpolator()

    for _ in range(100):
        interp._interpolate_body(body, target, 0.016)
    assert abs(body.spine_flex - 0.8) < 0.01


def test_zero_dt():
    face = FaceState()
    target = TargetAU()
    target.set("AU1", 1.0)
    interp = StateInterpolator()

    interp._interpolate_aus(face, target, 0.0)
    assert face.AU1 == 0.0  # No movement with zero dt
