"""Vertebral pivots must form a chain, rooted at the caudal end.

The loader parents every pivot to the batch group, making them siblings.
``BodyAnimationSystem`` rotates each by ``fraction * total`` and the fraction
tables sum to exactly 1.0, which only yields the intended bend if the rotations
ACCUMULATE.  As siblings they did not: measured on the real skeleton, the
cranial end of the thoracic spine moved 0.000 units at full flexion, and the
skin displaced 2.48 units -- individual vertebrae tilting in place, no curve.

Two further things this locks down, both learned the hard way:

* The chain must be rooted at the CAUDAL end.  ``pivots[0]`` is the most
  cranial vertebra in this dataset, so nesting in list order builds the chain
  upside down: the chain's caudal end swings while its cranial end -- the
  vertebrae the shoulders and head ride on -- stays put.
* Reparenting must not disturb the rest pose.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceforge.body.skeleton import nest_spine_pivots
from faceforge.core.scene_graph import SceneNode


def _chain(z_values):
    """Sibling pivots at the given heights, all under one parent."""
    root = SceneNode(name="spine_group")
    pivots = []
    for i, z in enumerate(z_values):
        n = SceneNode(name=f"pivot_{i}")
        n.set_position(0.0, 0.0, float(z))
        root.add(n)
        pivots.append({"group": n, "level": i, "fraction": 0.1})
    root.update_world_matrix(force=True)
    return root, pivots


def _world(node):
    node.update_world_matrix(force=True)
    return node.world_matrix[:3, 3].copy()


def test_pivots_become_a_chain():
    root, pivots = _chain([-10, -20, -30, -40])
    moved = nest_spine_pivots(pivots)
    nodes = [p["group"] for p in pivots]
    assert moved == 3
    nested = [n for n in nodes if n.parent in nodes]
    assert len(nested) == 3, "every pivot but the root must hang off another"


def test_the_chain_is_rooted_at_the_caudal_end():
    """List order is cranial-first here, as it is in the real dataset."""
    root, pivots = _chain([-10, -20, -30, -40])   # index 0 is the HIGHEST
    nest_spine_pivots(pivots)
    nodes = [p["group"] for p in pivots]
    chain_root = [n for n in nodes if n.parent not in nodes]
    assert len(chain_root) == 1
    z = [float(_world(n)[2]) for n in nodes]
    assert float(_world(chain_root[0])[2]) == pytest.approx(min(z)), (
        "chain is upside down: rotations would accumulate toward the pelvis "
        "and flexion would swing the chain's caudal end rather than the "
        "cranial vertebrae the shoulders ride on"
    )


def test_reparenting_does_not_move_the_rest_pose():
    root, pivots = _chain([-10, -20, -30, -40])
    nodes = [p["group"] for p in pivots]
    before = [_world(n) for n in nodes]
    nest_spine_pivots(pivots)
    after = [_world(n) for n in nodes]
    drift = max(float(np.linalg.norm(a - b))
                for a, b in zip(after, before, strict=True))
    assert drift < 1e-9, f"rest pose shifted by {drift}"


def test_list_order_is_preserved():
    """The animator pairs fracs[i] with pivots[i]; reordering silently
    reassigns every fraction to the wrong vertebra."""
    root, pivots = _chain([-10, -20, -30, -40])
    names = [p["group"].name for p in pivots]
    nest_spine_pivots(pivots)
    assert [p["group"].name for p in pivots] == names


def test_rotation_accumulates_down_the_chain():
    """The property the whole change exists for."""
    root, pivots = _chain([-10, -20, -30, -40])
    nodes = [p["group"] for p in pivots]
    nest_spine_pivots(pivots)

    cranial = max(nodes, key=lambda n: float(_world(n)[2]))
    caudal = min(nodes, key=lambda n: float(_world(n)[2]))
    rest_cranial, rest_caudal = _world(cranial), _world(caudal)

    # Same small rotation on every vertebra, as the animator does.
    from faceforge.core.math_utils import quat_from_euler
    for n in nodes:
        n.set_quaternion(quat_from_euler(0.15, 0.0, 0.0, "XYZ"))
    root.update_world_matrix(force=True)

    moved_cranial = float(np.linalg.norm(_world(cranial) - rest_cranial))
    moved_caudal = float(np.linalg.norm(_world(caudal) - rest_caudal))
    assert moved_cranial > moved_caudal, (
        "the cranial end must travel further than the caudal end -- if they "
        "match, rotations are not accumulating and the pivots are siblings"
    )
    assert moved_caudal < 1e-9, "the caudal root should stay put"


def test_a_single_pivot_is_left_alone():
    root, pivots = _chain([-10])
    assert nest_spine_pivots(pivots) == 0
