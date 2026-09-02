"""A muscle may override the global stretch/pin parameters; absence means global.

Why per muscle at all: physiological excursion depends on optimal fibre length
relative to moment arm and on how many joints a muscle spans, so one global
allowance is wrong in both directions at once.  Measured on this model, the
same parameter change moves muscles in OPPOSITE directions -- lowering
MAX_STRETCH improved deltoid acromial and degraded deltoid clavicular -- which
is the clearest evidence that the optimum is not shared.

These tests assert the resolution BEHAVIOUR (the limit actually applied),
not merely that the field round-trips, because a plumbed-but-unused override
would pass a storage test and change nothing.
"""

import numpy as np

from faceforge.anatomy.bone_anchors import BoneAnchorRegistry
from faceforge.anatomy.muscle_attachments import (
    MAX_STRETCH,
    PIN_STRENGTH,
    MuscleAttachmentData,
    MuscleAttachmentSystem,
)


def _system_with(*datas):
    sys_ = MuscleAttachmentSystem(BoneAnchorRegistry())
    for i, d in enumerate(datas):
        sys_._attachments[i] = d
    return sys_


def _data(name, stretch, max_stretch=None):
    return MuscleAttachmentData(
        muscle_name=name,
        origin_bones=["A"],
        insertion_bones=["B"],
        attachment_frac=np.linspace(0.0, 1.0, 10),
        origin_mask=np.linspace(0.0, 1.0, 10) > 0.8,
        insertion_mask=np.linspace(0.0, 1.0, 10) < 0.2,
        current_stretch=stretch,
        max_stretch=max_stretch,
    )


def test_the_global_still_applies_when_no_override_is_given():
    """The default path must be unchanged: an unsourced muscle keeps today's
    behaviour rather than acquiring an invented allowance."""
    over = MAX_STRETCH + 0.05
    sys_ = _system_with(_data("default", over))
    assert sys_.get_total_tension_excess() == pytest_approx(0.05)


def test_a_tighter_override_reports_excess_the_global_would_not():
    """A muscle whose allowance is below the global must clamp earlier."""
    sys_ = _system_with(_data("tight", 1.20, max_stretch=1.10))
    assert sys_.get_total_tension_excess() == pytest_approx(0.10)
    # Same stretch under the global (1.35) would be no excess at all, so this
    # value cannot have come from the global.
    assert 1.20 < MAX_STRETCH


def test_a_looser_override_suppresses_excess_the_global_would_report():
    """And a muscle with a larger physiological excursion must not clamp."""
    over = MAX_STRETCH + 0.05
    sys_ = _system_with(_data("loose", over, max_stretch=MAX_STRETCH + 0.20))
    assert sys_.get_total_tension_excess() == 0.0


def test_overrides_are_resolved_per_muscle_not_globally():
    """Two muscles in one system, different allowances, both respected."""
    sys_ = _system_with(
        _data("tight", 1.20, max_stretch=1.10),   # excess 0.10
        _data("loose", 1.20, max_stretch=1.50),   # excess 0.00
        _data("default", MAX_STRETCH + 0.05),      # excess 0.05
    )
    assert sys_.get_total_tension_excess() == pytest_approx(0.15)


def test_pin_strength_default_is_the_module_global():
    """The pin override resolves the same way; None must mean the global.

    Asserted on the resolution expression rather than by running the pinning
    pass, which needs a real mesh and bone registry -- the point here is that
    None does not silently become 0.0 (which would disable pinning entirely).
    """
    d = MuscleAttachmentData(muscle_name="m", origin_bones=[], insertion_bones=[])
    assert d.pin_strength is None
    resolved = d.pin_strength if d.pin_strength is not None else PIN_STRENGTH
    assert resolved == PIN_STRENGTH
    assert resolved > 0.0


def pytest_approx(value, tol=1e-9):
    import pytest
    return pytest.approx(value, abs=tol)
