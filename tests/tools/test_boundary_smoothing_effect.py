"""Cross-chain seam smoothing, as an assertion rather than a printout.

Ported from ``tools/test_boundary_smoothing.py``, which lived in ``tools/``
despite its name, printed a table and asserted nothing -- so a regression in
boundary smoothing could only be caught by a human reading the numbers.  The
measurement is unchanged; what is new is that the properties it was written to
demonstrate are now checked.

The property under test: at a chain boundary (a skin edge whose two vertices
are bound to different joint chains) the displacement discontinuity is larger
than within a chain, and boundary smoothing reduces it.  Both are statements
about the *ratio* of cross-chain to same-chain discontinuity, not about an
absolute millimetre figure, because the absolute value depends on the pose
amplitude.

The original script is left in place -- see ``tools/README.md`` for the
recommendation to retire it.
"""

from __future__ import annotations

import numpy as np
import pytest

# `slow`: the module-scoped fixture loads the full BodyParts3D skin and
# skeleton through the headless loader.  Deselect with `pytest -m "not slow"`.
pytestmark = pytest.mark.slow

POSE = "sitting"


@pytest.fixture(scope="module")
def bound_skin():
    """The headless scene with the skin layer registered and bound."""
    from tools.headless_loader import load_headless_scene, load_layer, register_layer
    scene = load_headless_scene()
    meshes = load_layer(scene, "skin")
    register_layer(scene, meshes, "skin")
    assert scene.skinning.bindings, "skin did not bind"
    return scene


def _discontinuity(scene, binding) -> dict[str, float]:
    """Mean edge-displacement difference, split by same- vs cross-chain.

    Transcribed from the original script: an edge is cross-chain when its two
    vertices are bound to joints in different chains, and the discontinuity is
    the norm of the difference between the two vertices' displacements.
    """
    mesh = binding.mesh
    rest = mesh.rest_positions.reshape(-1, 3).astype(np.float64)
    current = mesh.geometry.positions.reshape(-1, 3).astype(np.float64)
    edges = binding.edge_pairs
    if edges is None:
        return {}
    displacement = current - rest
    chains = np.array(
        [scene.skinning.joints[idx].chain_id for idx in binding.joint_indices],
        dtype=np.int32)
    cross = chains[edges[:, 0]] != chains[edges[:, 1]]
    diff = np.linalg.norm(
        displacement[edges[:, 0]] - displacement[edges[:, 1]], axis=1)
    out: dict[str, float] = {}
    if np.any(~cross):
        out["same_chain_mean"] = float(diff[~cross].mean())
    if np.any(cross):
        out["cross_chain_mean"] = float(diff[cross].mean())
        out["cross_chain_max"] = float(diff[cross].max())
    out["cross_chain_edges"] = int(cross.sum())
    return out


def _pose_and_measure(scene, passes: int) -> dict[str, float]:
    """Pose the scene with ``passes`` smoothing iterations and measure.

    The switch is ``BOUNDARY_SMOOTH_PASSES`` on the skinning manager's class
    (``faceforge.body.soft_tissue``), which is where the Laplacian
    displacement smoothing is configured; zero passes disables it.  The
    original script had no switch at all -- it printed one column.
    """
    from tools.headless_loader import apply_pose
    from tools.skinning_scorer import get_all_poses, _make_body_state
    binding = scene.skinning.bindings[0]
    manager = type(scene.skinning)
    original = manager.BOUNDARY_SMOOTH_PASSES
    try:
        manager.BOUNDARY_SMOOTH_PASSES = passes
        state = _make_body_state(get_all_poses([POSE])[POSE])
        apply_pose(scene, state)
        return _discontinuity(scene, binding)
    finally:
        manager.BOUNDARY_SMOOTH_PASSES = original


@pytest.fixture(scope="module")
def measured(bound_skin):
    """Both measurements, taken once: each pose application is seconds long."""
    return {
        "smoothed": _pose_and_measure(bound_skin, passes=5),
        "unsmoothed": _pose_and_measure(bound_skin, passes=0),
    }


def test_the_mesh_has_cross_chain_edges_to_measure(measured):
    assert measured["smoothed"]["cross_chain_edges"] > 0


def test_cross_chain_edges_are_the_discontinuous_ones(measured):
    """The premise of the smoothing system: seams live at chain boundaries."""
    smoothed = measured["smoothed"]
    assert smoothed["cross_chain_mean"] > smoothed["same_chain_mean"]


def test_boundary_smoothing_reduces_the_cross_chain_discontinuity(measured):
    assert measured["smoothed"]["cross_chain_mean"] < \
        measured["unsmoothed"]["cross_chain_mean"]


def test_smoothing_does_not_smear_the_whole_mesh(measured):
    """It must fix the seam without inflating within-chain displacement."""
    assert measured["smoothed"]["same_chain_mean"] <= \
        measured["unsmoothed"]["same_chain_mean"] * 1.5
