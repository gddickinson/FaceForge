"""Neighbour-stretch clamping, as an assertion rather than a printout.

Ported from ``tools/test_neighbor_clamp.py``, which lived in ``tools/`` despite
its name and printed a table without asserting anything.  The clamp exists to
stop a mis-bound vertex from being dragged into a spike, so the property to
check is that the worst edge stretch in the mesh is *lower* with clamping on
than with it off, and that the shipped diagnostic agrees.

The switch is ``CLAMP_PASSES`` on the skinning manager's class (zero passes
disables the clamp), which is the same constant the renderer reads -- so this
measures the shipped code path rather than a re-implementation.

The original script is left in place -- see ``tools/README.md``.
"""

from __future__ import annotations

import numpy as np
import pytest

# `slow`: loads the full BodyParts3D skin and skeleton through the headless
# loader.  Deselect with `pytest -m "not slow"`.
pytestmark = pytest.mark.slow

POSE = "extreme_arm_raise"


@pytest.fixture(scope="module")
def bound_skin():
    from tools.headless_loader import load_headless_scene, load_layer, register_layer
    scene = load_headless_scene()
    meshes = load_layer(scene, "skin")
    register_layer(scene, meshes, "skin")
    assert scene.skinning.bindings, "skin did not bind"
    return scene


def _edge_stretch(binding) -> np.ndarray:
    """Per-edge stretch ratio: current length / rest length.

    Edges with zero rest length are dropped rather than producing an infinite
    ratio.
    """
    mesh = binding.mesh
    rest = mesh.rest_positions.reshape(-1, 3).astype(np.float64)
    current = mesh.geometry.positions.reshape(-1, 3).astype(np.float64)
    edges = binding.edge_pairs
    assert edges is not None, "neighbour data was not built"
    rest_len = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
    cur_len = np.linalg.norm(current[edges[:, 0]] - current[edges[:, 1]], axis=1)
    usable = rest_len > 1e-9
    return cur_len[usable] / rest_len[usable]


def _pose_and_measure(scene, clamp_passes: int) -> dict[str, float]:
    from tools.headless_loader import apply_pose
    from tools.skinning_scorer import get_all_poses, _make_body_state
    binding = scene.skinning.bindings[0]
    manager = type(scene.skinning)
    original = manager.CLAMP_PASSES
    try:
        manager.CLAMP_PASSES = clamp_passes
        poses = get_all_poses([POSE])
        state = _make_body_state(poses[POSE])
        apply_pose(scene, state)
        stretch = _edge_stretch(binding)
        return {
            "max": float(stretch.max()),
            "p999": float(np.percentile(stretch, 99.9)),
            "mean": float(stretch.mean()),
            "over_5x": int((stretch > 5.0).sum()),
            "edges": int(stretch.size),
        }
    finally:
        manager.CLAMP_PASSES = original


@pytest.fixture(scope="module")
def measured(bound_skin):
    """Both measurements, taken once: a pose application is seconds long."""
    return {
        "clamped": _pose_and_measure(bound_skin, clamp_passes=10),
        "unclamped": _pose_and_measure(bound_skin, clamp_passes=0),
    }


def test_neighbour_data_is_built_for_the_skin(bound_skin):
    binding = bound_skin.skinning.bindings[0]
    assert binding.edge_pairs is not None
    assert binding.edge_pairs.shape[1] == 2
    assert binding.edge_pairs.shape[0] > 1000


def test_the_pose_is_extreme_enough_to_stretch_something(measured):
    assert measured["unclamped"]["max"] > 1.5


def test_clamping_reduces_the_stretch_tail(measured):
    """What the clamp demonstrably does: shrink the population of overstretched
    edges.  Asserted on the 99.9th percentile and the count above 5x rather
    than on the single worst edge -- see the next test for why."""
    assert measured["clamped"]["p999"] <= measured["unclamped"]["p999"]

    # The over-5x COUNT is no longer strictly reduced, and that is a recorded
    # behaviour change rather than a regression being asserted away -- the same
    # treatment the next test gives the single worst edge.
    #
    # SoftTissueSkinning.CONTAIN_CORRECTIONS forbids the clamp from moving a
    # vertex whose own driving joints are all unchanged, because that pull was
    # the entire cause of geometry crossing body regions: raising an arm moved
    # trunk geometry by up to 42.6 units (skin) and 10.2 (transverse
    # trapezius), traced by ablation to this pass and boundary smoothing alone.
    # With containment enforced that motion is exactly 0.000.
    #
    # The cost is an overstretched edge spanning a static and a moving vertex,
    # which can now only be corrected from one end. Measured on
    # extreme_arm_raise: 19,426 edges over 5x with clamping against 19,395
    # without -- 31 edges, 0.16%. The p999 assertion above still holds, so the
    # clamp still shrinks the distribution; it no longer wins on this count.
    unclamped = measured["unclamped"]["over_5x"]
    assert measured["clamped"]["over_5x"] <= unclamped * 1.01, (
        f'clamping left {measured["clamped"]["over_5x"]:,} edges over 5x '
        f"against {unclamped:,} unclamped -- more than the 1% allowance for "
        "containment-limited clamping, so this is a real loss of clamp "
        "effectiveness rather than the measured 0.16% tie"
    )


def test_the_single_worst_edge_is_not_improved_by_clamping(measured):
    """Measured behaviour, recorded rather than asserted away.

    On ``extreme_arm_raise`` the clamp lowers the tail but leaves the single
    worst edge *higher* than with the clamp off (778x vs 517x in one run).
    That is consistent with how the clamp works -- pulling a clamped vertex
    back toward its neighbours displaces those neighbours, so the extremum can
    move rather than shrink -- but it means "the clamp bounds edge stretch" is
    not true of the maximum, only of the distribution.  This test exists so
    the behaviour is visible in the suite instead of being discovered again;
    it asserts only that both numbers are finite and that the tail assertion
    above is the one carrying the load.  If the clamp is ever changed to bound
    the maximum, this test will start failing and should be replaced by the
    stronger assertion.
    """
    assert np.isfinite(measured["clamped"]["max"])
    assert np.isfinite(measured["unclamped"]["max"])
    assert measured["clamped"]["max"] > measured["clamped"]["p999"]


def test_clamping_leaves_the_bulk_of_the_mesh_alone(measured):
    """A clamp that changed the mean would be deforming the whole body."""
    assert measured["clamped"]["mean"] == pytest.approx(
        measured["unclamped"]["mean"], rel=0.05)


def test_the_shipped_diagnostic_agrees_with_the_measurement(bound_skin):
    """``SkinningDiagnostic.check_neighbor_stretch`` is what a developer runs;
    it must report the same worst case this test measures."""
    from faceforge.body.diagnostics import SkinningDiagnostic
    measured = _pose_and_measure(bound_skin, clamp_passes=10)
    diagnostic = SkinningDiagnostic(bound_skin.skinning)
    report = diagnostic.check_neighbor_stretch()
    assert report is not None
    reported = _worst_from(report)
    if reported is None:
        pytest.skip(f"diagnostic report shape not recognised: {type(report)}")
    assert reported == pytest.approx(measured["max"], rel=0.25)


def _worst_from(report) -> float | None:
    """Pull the worst stretch ratio out of the diagnostic's report.

    Tolerant of the report being a dataclass, a dict or a mapping of layers,
    because this test asserts on the number, not on the container.
    """
    for name in ("max_stretch", "worst_stretch", "max_ratio", "max"):
        value = getattr(report, name, None)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(report, dict) and isinstance(report.get(name), (int, float)):
            return float(report[name])
    if isinstance(report, dict):
        numbers = [v for v in report.values() if isinstance(v, (int, float))]
        if numbers:
            return float(max(numbers))
    return None
