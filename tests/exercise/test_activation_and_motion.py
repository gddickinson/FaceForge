"""The activation model and the movement description."""

from __future__ import annotations

import pytest

from faceforge.exercise.activation import (
    KIND_SCALE, build_activation_track, group_levels_to_muscles, phase_group_levels,
)
from faceforge.exercise.catalog import get_exercise_catalog
from faceforge.exercise.clip_builder import build_exercise_clip
from faceforge.exercise.model import Phase, PhaseKind, Role
from faceforge.exercise.motion_description import describe_transition, summarise_motions
from faceforge.exercise.pose_library import pose


@pytest.fixture(scope="module")
def squat():
    return get_exercise_catalog()["bodyweight_squat"]


def test_concentric_beats_eccentric_beats_transition(squat):
    con = next(p for p in squat.phases if p.kind is PhaseKind.CONCENTRIC)
    ecc = next(p for p in squat.phases if p.kind is PhaseKind.ECCENTRIC)
    lc, le = phase_group_levels(squat, con), phase_group_levels(squat, ecc)
    assert lc["quadriceps"] > le["quadriceps"]
    assert le["quadriceps"] == pytest.approx(lc["quadriceps"] * KIND_SCALE[PhaseKind.ECCENTRIC][Role.PRIMARY])
    # A stabiliser holds its level regardless of the phase kind.
    assert lc["erector_spinae"] == le["erector_spinae"]


def test_roles_order_the_levels(squat):
    con = next(p for p in squat.phases if p.kind is PhaseKind.CONCENTRIC)
    levels = phase_group_levels(squat, con)
    assert levels["quadriceps"] > levels["adductors"] > levels["rectus_abdominis"]


def test_phase_overrides_and_sided_keys():
    levels = group_levels_to_muscles({"quadriceps:R": 0.9, "quadriceps": 0.2})
    assert levels["Vastus Lat. R"] == 0.9
    assert levels["Vastus Lat. L"] == 0.2


def test_track_samples_ramp_between_phases(squat):
    built = build_exercise_clip(squat, reps=1)
    track = built.activation
    descent, ascent = built.spans[0], built.spans[2]
    mid_descent = track.sample((descent.t0 + descent.t1) / 2)["Vastus Lat. R"]
    mid_ascent = track.sample((ascent.t0 + ascent.t1) / 2)["Vastus Lat. R"]
    assert mid_ascent > mid_descent
    boundary = track.sample(ascent.t0)["Vastus Lat. R"]
    assert mid_descent <= boundary <= mid_ascent


def test_track_covers_the_clip_and_clamps(squat):
    built = build_exercise_clip(squat, reps=2)
    assert built.activation.duration == pytest.approx(built.duration)
    assert built.activation.sample(-5.0) == built.activation.sample(0.0)
    assert built.activation.sample(1e9) == built.activation.sample(built.duration)


def test_describe_squat_descent_in_anatomical_terms():
    stand = pose()
    bottom = pose(hip_flex=120, knee_flex=120, ankle_flex=35)
    motions = describe_transition(stand, bottom)
    terms = {(m.joint, m.side, m.term) for m in motions}
    assert ("hip", "R", "flexion") in terms and ("knee", "L", "flexion") in terms
    assert ("ankle", "R", "dorsiflexion") in terms
    lines = summarise_motions(motions)
    assert lines[0].startswith("Hips: flexion 0° → 120°")
    assert any(line.startswith("Ankles: dorsiflexion") for line in lines)
    assert len(lines) == 3, lines


def test_describe_reverse_direction_uses_the_negative_term():
    motions = describe_transition(pose(hip_flex=120), pose())
    assert motions[0].term == "extension"
    assert motions[0].start_deg == pytest.approx(120)


def test_small_changes_are_ignored():
    assert describe_transition(pose(), pose(knee_flex=2)) == []
