"""Deformation quality as a gate, not a printout.

The suite had 1813 passing tests and none measured deformation quality, which
is how a muscle distorting to 501% of its own rest span coexisted with a green
suite, and how raising an arm dragging trunk geometry went unnoticed until a
human saw it on screen.

Marked slow: this loads real BodyParts3D meshes and runs the full deformation.
"""

import pytest

from tools import deformation_quality as dq

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def measured():
    return dq.measure()


def test_the_pose_actually_moved_something(measured):
    """Control: every other assertion here is meaningless without this."""
    assert measured["moved_joints"] > 0, (
        f"pose {measured['pose']} moved no joints"
    )
    assert measured["seam_edges"] > 0, "no seam edges to measure"


def test_the_rest_pose_is_reproduced_exactly(measured):
    """Control: the metrics are all differences from the rest reference."""
    assert measured["rest_deviation"] <= 1e-6, (
        f"rest pose not reproduced: max |pos - ref| = "
        f"{measured['rest_deviation']:.6g}"
    )


def test_no_vertex_moves_when_its_own_joints_do_not(measured):
    """The containment invariant -- a hard invariant, so zero tolerance.

    Raising an arm must not move trunk or leg geometry. This held only after
    the neighbour clamp and boundary smoothing were retired: both pull a vertex
    toward its mesh-neighbour average, so a static vertex beside a moving one
    was dragged by construction, up to 42.6 units.
    """
    assert measured["containment"] <= dq.THRESHOLDS["containment"], (
        f"{measured['containment_worst']} moved "
        f"{measured['containment']:.3f} units with none of its own joints moving"
    )


@pytest.mark.parametrize("metric", ["seam_p99", "seam_max", "bulk_p99"])
def test_distortion_does_not_regress(measured, metric):
    """A ratchet on the shipped configuration, not a quality bar.

    The seam figures in THRESHOLDS are poor and are the open defect; they are
    pinned so they cannot get worse while that is resolved.
    """
    assert measured[metric] <= dq.THRESHOLDS[metric], (
        f"{metric} {measured[metric]:.4f} exceeds {dq.THRESHOLDS[metric]}"
    )


def test_the_gate_is_sensitive_to_a_broken_engine(monkeypatch):
    """Negative control: break the engine and require the gate to say so.

    Without this, every assertion above could be passing vacuously.

    The mechanism matters and has had to be replaced.  The original control
    re-enabled ``USE_NEIGHBOR_CLAMP`` with the hull bound off, which used to
    drag static vertices along with their moving neighbours.  It no longer
    does: measured, containment stays at 0.000 with that switch on, with the
    per-binding skip disabled, and with ``CONTAIN_CORRECTIONS`` or
    ``USE_BONE_OFFSET_PROJECTION`` on.  The control had been passing an engine
    that was no longer broken.

    Opting every muscle into the soft-body path is a breaker that still works,
    and for a reason the metric is meant to catch: the physics pass relaxes
    edges, which couples vertices, so a vertex none of whose own joints moved
    is pulled by a neighbour whose joints did.  Measured, it puts 6.000 units
    of drift on Triceps Long R.
    """
    from faceforge.body.soft_tissue import SoftTissueSkinning as S
    from faceforge.coordination import demand_loaders as dl

    # The hull bound corrects the drift away again, so it comes off, exactly
    # as the original control took it off.
    monkeypatch.setattr(S, "USE_HULL_BOUND", False)
    monkeypatch.setattr(S, "CONTAIN_CORRECTIONS", False)
    real = dl.register_muscle_layer

    def force_physics(skinning, layer, meshes, defs, chain_ids, **kw):
        for defn in defs:
            defn["physicsDeform"] = True
        return real(skinning, layer, meshes, defs, chain_ids, **kw)

    monkeypatch.setattr(dl, "register_muscle_layer", force_physics)
    failures = dq.check(dq.measure())
    assert any("containment" in f for f in failures), (
        f"the gate passed a deliberately broken engine; failures were {failures}"
    )
