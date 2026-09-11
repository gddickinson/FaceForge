"""What moves a neck muscle's body end, and what puts it back.

Three defects are pinned here, all measured on the real rig first:

* A muscle followed the *regional* anchor even when it named the bone it
  attaches to.  At full thoracic flexion the top thoracic pivot travels
  4.58 units while T1 -- where longus colli and longus capitis originate --
  does not move at all, because the cervical chain carrying T1 hangs off
  ``bodyRoot``.  The suboccipitals, whose origins are on C1 and C2, were
  dragged 3.76 units by a thorax they are not attached to.
* ``update`` skipped its work whenever the body delta was zero.  Returning
  to rest *is* a zero delta, so the frame that should have put the muscles
  back was the frame that was skipped, and a neck bent by a sit-up stayed
  bent.
* Muscles that name no bones still have to follow their region.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceforge.anatomy import neck_body_follow as follow
from faceforge.anatomy.bone_anchors import BoneAnchorRegistry
from faceforge.core.math_utils import quat_identity
from faceforge.core.scene_graph import SceneNode

from tests.anatomy.test_neck_muscle_pinning import (
    _build_system_with_muscle,
    _make_synthetic_muscle,
)


def _registry(positions: dict[str, tuple[float, float, float]]) -> tuple:
    reg = BoneAnchorRegistry()
    nodes = {}
    for name, pos in positions.items():
        n = SceneNode(name=name)
        n.set_position(*pos)
        n.mark_dirty()
        n.update_world_matrix(force=True)
        nodes[name] = n
    reg.register_bones(nodes)
    reg.snapshot_rest_positions()
    return reg, nodes


def _move(node: SceneNode, pos) -> None:
    node.set_position(*pos)
    node.mark_dirty()
    node.update_world_matrix(force=True)


def _lower_end(system):
    md = system.muscle_data[0]
    fracs = md.spine_fracs
    mask = fracs <= np.percentile(fracs, 15)
    pos = md.mesh.geometry.positions.reshape(-1, 3)
    rest = md.rest_positions.reshape(-1, 3)
    return (pos[mask] - rest[mask]).mean(axis=0)


def test_a_named_bone_beats_the_regional_anchor():
    """The muscle attaches to T1, which has not moved: it must not move."""
    defn, mesh, node = _make_synthetic_muscle(
        lower_attach="thoracic", lower_bones=["T1"])
    reg, _ = _registry({"T1": (0.0, -30.0, 0.0)})
    system = _build_system_with_muscle(defn, mesh, node, bone_registry=reg)

    # The thoracic region has swung forward by 5 units; T1 has not.
    system.set_body_anchors_rest({"thoracic": np.zeros(3)})
    system.set_body_anchors_current({"thoracic": np.array([0.0, 0.0, 5.0])})
    system.update(quat_identity())

    assert np.linalg.norm(_lower_end(system)) < 1e-6


def test_a_muscle_without_bones_still_follows_its_region():
    defn, mesh, node = _make_synthetic_muscle(lower_attach="thoracic")
    system = _build_system_with_muscle(defn, mesh, node)

    system.set_body_anchors_rest({"thoracic": np.zeros(3)})
    system.set_body_anchors_current({"thoracic": np.array([0.0, 0.0, 5.0])})
    system.update(quat_identity())

    moved = _lower_end(system)
    assert moved[2] > 3.0, moved


def test_the_body_end_follows_the_bone_that_does_move():
    defn, mesh, node = _make_synthetic_muscle(
        lower_attach="thoracic", lower_bones=["T3"])
    reg, nodes = _registry({"T3": (0.0, -30.0, 0.0)})
    system = _build_system_with_muscle(defn, mesh, node, bone_registry=reg)

    _move(nodes["T3"], (0.0, -30.0, 5.0))
    system.update(quat_identity())

    moved = _lower_end(system)
    assert moved[2] > 3.0, moved


def test_the_muscles_return_to_rest_when_the_body_does():
    """The regression: a zero delta used to read as 'nothing to do'."""
    defn, mesh, node = _make_synthetic_muscle(
        lower_attach="thoracic", lower_bones=["T3"])
    reg, nodes = _registry({"T3": (0.0, -30.0, 0.0)})
    system = _build_system_with_muscle(defn, mesh, node, bone_registry=reg)

    _move(nodes["T3"], (0.0, -30.0, 5.0))
    system.update(quat_identity())
    assert np.linalg.norm(_lower_end(system)) > 1.0

    _move(nodes["T3"], (0.0, -30.0, 0.0))
    system.update(quat_identity())

    md = system.muscle_data[0]
    assert md.mesh.geometry.positions == pytest.approx(md.rest_positions, abs=1e-5)


def test_an_unchanged_pose_is_still_skipped():
    """The early exit has to keep working, or every frame redoes the mesh."""
    defn, mesh, node = _make_synthetic_muscle(
        lower_attach="thoracic", lower_bones=["T3"])
    reg, nodes = _registry({"T3": (0.0, -30.0, 0.0)})
    system = _build_system_with_muscle(defn, mesh, node, bone_registry=reg)

    _move(nodes["T3"], (0.0, -30.0, 5.0))
    system.update(quat_identity())
    md = system.muscle_data[0]

    md.mesh.needs_update = False
    system.update(quat_identity())
    assert md.mesh.needs_update is False


def test_bone_displacement_is_none_when_nothing_resolves():
    reg, _ = _registry({"T3": (0.0, -30.0, 0.0)})
    assert follow.bone_displacement(reg, "x", ["Thoracic Vertebra T1"]) is None
    assert follow.bone_displacement(reg, "x", None) is None
    assert follow.bone_displacement(None, "x", ["T3"]) is None


def test_regional_deltas_are_zero_for_an_anchor_with_no_rest():
    deltas = follow.regional_deltas({}, {"thoracic": np.array([1.0, 2.0, 3.0])})
    assert set(deltas) == set(follow.ANCHOR_REGIONS)
    for value in deltas.values():
        assert value == pytest.approx(np.zeros(3))
