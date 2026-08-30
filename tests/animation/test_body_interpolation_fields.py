"""Field-ownership tests for :meth:`StateInterpolator._interpolate_body`.

``_interpolate_body`` used to lerp *every* key returned by
``BodyState.to_dict()`` toward ``target_body``.  ``to_dict()`` filters on
``isinstance(v, (int, float))`` -- and ``bool`` is a subclass of ``int`` -- so
the dict contains all 52 numeric fields including the six ``auto_*`` toggles,
the ``gender`` slider, and the ``breath_phase_body`` accumulator.  Those three
groups are *not* pose targets: they are written directly on ``state.body`` by
the UI or by the animation systems, and nothing ever writes the corresponding
``target_body`` field.  The interpolator therefore dragged them back to their
defaults every frame (defects ``interp-saturates-breath-phase``,
``interp-clobbers-gender``, ``interp-bools-become-floats``).

FIXED: ``BodyState`` now carries an explicit allowlist -- ``POSE_FIELDS`` (39
joint angles) plus ``SETTING_FIELDS`` (5 continuous physiology settings) make up
``INTERPOLATED_FIELDS``; ``FLAG_FIELDS`` (the 6 booleans) are *copied* from the
target rather than lerped; and ``LIVE_ONLY_FIELDS``
(``breath_phase_body``, ``gender``) are never touched.  The tests below are the
regression guards; the numbers the defect report measured before the fix are
quoted inline so they cannot silently come back.
"""

import itertools
import math

import pytest

from faceforge.animation.interpolation import StateInterpolator
from faceforge.core.state import BodyState

DT = 1.0 / 60.0        # the app's QTimer interval is 16 ms


# ----------------------------------------------------------------------
# Root cause
# ----------------------------------------------------------------------

def test_to_dict_leaks_bool_fields_because_bool_subclasses_int():
    """Documents the root cause shared by all three defects below.

    ``to_dict()`` is unchanged -- it is a serialisation helper and other callers
    may want the flags.  The fix is that ``_interpolate_body`` no longer
    consumes it; see :func:`test_interpolator_does_not_consume_to_dict`.
    """
    d = BodyState().to_dict()
    bools = sorted(k for k, v in d.items() if isinstance(v, bool))
    assert bools == [
        "auto_breath_body",
        "auto_fasciculation",
        "auto_heartbeat",
        "auto_lung_expand",
        "auto_peristalsis",
        "auto_pulse_wave",
    ], "set of bool fields leaking into to_dict() changed"
    assert "gender" in d
    assert "breath_phase_body" in d


def test_allowlist_partitions_every_numeric_field_exactly_once():
    """The allowlist must stay exhaustive, or a new field silently goes unowned.

    This is the guard that stops the defect recurring a fourth time: adding a
    field to ``BodyState`` without classifying it fails here rather than being
    silently adopted by the interpolator.
    """
    numeric = sorted(BodyState().to_dict())
    classified = (
        BodyState.POSE_FIELDS
        + BodyState.SETTING_FIELDS
        + BodyState.FLAG_FIELDS
        + BodyState.LIVE_ONLY_FIELDS
    )
    assert len(classified) == len(set(classified)), "a field is classified twice"
    assert sorted(classified) == numeric, (
        f"unclassified={sorted(set(numeric) - set(classified))} "
        f"phantom={sorted(set(classified) - set(numeric))}"
    )
    assert len(numeric) == 52
    assert len(BodyState.POSE_FIELDS) == 39
    assert BodyState.LIVE_ONLY_FIELDS == ("breath_phase_body", "gender")
    assert BodyState.INTERPOLATED_FIELDS == (
        BodyState.POSE_FIELDS + BodyState.SETTING_FIELDS
    )


def test_interpolator_does_not_consume_to_dict():
    """A new numeric field must not be picked up just by existing."""
    body, target = BodyState(), BodyState()
    body.to_dict = lambda: pytest.fail("_interpolate_body still calls to_dict()")
    target.to_dict = body.to_dict
    StateInterpolator()._interpolate_body(body, target, DT)


# ----------------------------------------------------------------------
# DEFECT interp-clobbers-gender
# ----------------------------------------------------------------------

def test_gender_is_not_dragged_back_toward_the_default():
    body, target = BodyState(), BodyState()
    body.gender = 1.0                       # user drags the slider fully female

    for _ in range(60):                     # one second of frames
        StateInterpolator()._interpolate_body(body, target, DT)

    assert body.gender == pytest.approx(1.0, abs=1e-3), (
        f"gender decayed to {body.gender:.6f} after 1 s -- the slider value is "
        "destroyed by the pose interpolator"
    )


def test_gender_does_not_decay_at_the_rate_we_measured_before_the_fix():
    """Guard against the specific pre-fix decay curve coming back.

    Before the allowlist, ``body.gender`` fell to 0.355264 / 0.126213 /
    0.015930 / 0.000254 after 15 / 30 / 60 / 120 frames (measured, 60 fps).
    """
    body, target = BodyState(), BodyState()
    body.gender = 1.0
    samples = {}
    for frame in range(1, 121):
        StateInterpolator()._interpolate_body(body, target, DT)
        if frame in (15, 30, 60, 120):
            samples[frame] = body.gender

    assert samples == {15: 1.0, 30: 1.0, 60: 1.0, 120: 1.0}


def test_gender_is_not_in_the_interpolated_set():
    assert "gender" not in BodyState.INTERPOLATED_FIELDS
    assert "gender" in BodyState.LIVE_ONLY_FIELDS


# ----------------------------------------------------------------------
# DEFECT interp-saturates-breath-phase
# ----------------------------------------------------------------------

def _run_breathing(seconds: float, *, with_interpolator: bool):
    """Advance breath phase the way body_animation.update does, and sample it."""
    body, target = BodyState(), BodyState()
    body.auto_breath_body = True
    target.auto_breath_body = True     # app.py:217 sets both for bool fields
    interp = StateInterpolator()
    phases = []
    for _ in range(int(seconds / DT)):
        # faceforge/body/body_animation.py:62-64
        body.breath_phase_body += DT * body.breath_rate * 2 * math.pi
        if body.breath_phase_body > 2 * math.pi:
            body.breath_phase_body -= 2 * math.pi
        if with_interpolator:
            interp._interpolate_body(body, target, DT)
        phases.append(body.breath_phase_body)
    return phases


def _cycles(phases: list[float]) -> float:
    """Completed respiratory cycles: wrap events plus the partial last cycle."""
    wraps = sum(1 for a, b in itertools.pairwise(phases) if b < a)
    return wraps + phases[-1] / (2 * math.pi)


def test_breathing_completes_full_cycles():
    """At breath_rate=0.25 Hz, 10 s must contain ~2.5 full cycles."""
    phases = _run_breathing(10.0, with_interpolator=True)
    sin_vals = [math.sin(p) for p in phases]

    assert max(phases) > 2 * math.pi * 0.9, (
        f"phase never approached a full cycle: max {max(phases):.4f} rad"
    )
    assert min(sin_vals) < -0.9, (
        f"chest never exhaled: sin(phase) stayed in "
        f"[{min(sin_vals):.4f}, {max(sin_vals):.4f}]"
    )
    assert _cycles(phases) == pytest.approx(2.5, abs=0.05), (
        f"expected ~2.5 cycles in 10 s at 0.25 Hz, got {_cycles(phases):.3f}"
    )


def test_interpolator_no_longer_perturbs_the_breath_accumulator():
    """The interpolated run must now be bit-identical to the free-running one.

    Before the fix the interpolated phase saturated at 0.3665 rad with
    sin(phase) confined to [+0.024, +0.358] -- 0 of the expected 2.5 cycles,
    i.e. the chest inflated ~36% and never exhaled.
    """
    with_interp = _run_breathing(10.0, with_interpolator=True)
    without = _run_breathing(10.0, with_interpolator=False)

    assert with_interp == without
    assert max(without) == pytest.approx(6.257, abs=1e-2)

    sin_with = [math.sin(p) for p in with_interp]
    assert min(sin_with) < -0.99
    assert max(sin_with) > 0.99
    assert "breath_phase_body" not in BodyState.INTERPOLATED_FIELDS


# ----------------------------------------------------------------------
# DEFECT interp-bools-become-floats
# ----------------------------------------------------------------------

AUTO_FLAGS = [
    "auto_breath_body", "auto_heartbeat", "auto_pulse_wave",
    "auto_lung_expand", "auto_peristalsis", "auto_fasciculation",
]


@pytest.mark.parametrize("flag", AUTO_FLAGS)
def test_auto_flags_stay_boolean(flag):
    body, target = BodyState(), BodyState()
    setattr(body, flag, True)
    setattr(target, flag, True)          # app.py sets both for bool fields

    StateInterpolator()._interpolate_body(body, target, DT)

    value = getattr(body, flag)
    assert isinstance(value, bool), (
        f"{flag} became {type(value).__name__} ({value!r}) after one frame"
    )


def test_auto_flag_turns_off_promptly_when_only_the_target_is_cleared():
    body, target = BodyState(), BodyState()
    body.auto_heartbeat = True
    target.auto_heartbeat = False        # e.g. preset_manager.set_from_js_dict

    interp = StateInterpolator()
    frames = 0
    while body.auto_heartbeat and frames < 20_000:
        interp._interpolate_body(body, target, DT)
        frames += 1

    assert frames < 60, (
        f"flag still truthy after {frames} frames ({frames * DT:.1f} s)"
    )


@pytest.mark.parametrize("flag", AUTO_FLAGS)
def test_auto_flag_stays_a_bool_over_a_long_run(flag):
    """Before the fix the flag decayed as 0.9333**n and was still truthy after
    20 000 frames (333 s of wall clock) while never becoming a ``bool`` again."""
    body, target = BodyState(), BodyState()
    setattr(body, flag, True)
    setattr(target, flag, False)

    interp = StateInterpolator()
    for _ in range(20_000):
        interp._interpolate_body(body, target, DT)

    value = getattr(body, flag)
    assert value is False, f"{flag} is {type(value).__name__} {value!r}"


@pytest.mark.parametrize("flag", AUTO_FLAGS)
def test_target_only_flag_enable_also_propagates(flag):
    """Symmetric case: a preset switching a toggle *on* must take effect too."""
    body, target = BodyState(), BodyState()
    setattr(target, flag, True)
    StateInterpolator()._interpolate_body(body, target, DT)
    assert getattr(body, flag) is True
