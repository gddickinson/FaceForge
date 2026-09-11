"""Re-snapshotting the rest pose after the skeleton itself moves.

A sex morph moves every joint.  The binding -- which bone a vertex follows --
does not change, so re-solving it is 85 seconds spent to get the same answer;
what must change is the joints' rest transforms and everything cached from
them.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceforge.body.soft_tissue import SoftTissueSkinning
from faceforge.core.math_utils import quat_from_axis_angle, vec3
from faceforge.core.state import BodyState
from tests.body.test_skinning_under_scene_wrapper import _grid_mesh, _rig


def _chains(nodes):
    return [[(n.name, n) for n in nodes]]


def test_the_inverse_rest_cache_does_not_survive_a_rebuild():
    """It is keyed by joint INDEX and described as constant for the rig's life.

    It is constant only while the joint list is.  Rebuilding it left index i
    pointing at a new joint while the cache held the old joint's inverse, so
    every delta read through ``_joint_delta`` was ``current_new x rest_old``
    inverted instead of the identity.  The hull bound, the one pass that reads
    those deltas, then clamped 554875 skin vertices by up to 10.4 units.
    """
    scene, wrapper, nodes, sk = _rig()
    sk.build_skin_joints(_chains(nodes))
    sk._begin_frame()
    first = sk._joint_delta(1).copy()
    np.testing.assert_allclose(first, np.eye(4), atol=1e-9)

    nodes[1].set_position(0.0, 0.0, 55.0)          # the skeleton changes size
    scene.update()
    sk.rebuild_skin_joints(_chains(nodes))
    sk._begin_frame()
    np.testing.assert_allclose(sk._joint_delta(1), np.eye(4), atol=1e-9), \
        "the rebuilt joint is its own rest pose"


class TestResnapshot:
    def test_it_keeps_the_bindings_and_refreshes_the_rest_metrics(self):
        scene, wrapper, nodes, sk = _rig()
        mesh = _grid_mesh()
        sk.register_skin_mesh(mesh, is_muscle=True, muscle_name=mesh.name)
        binding = sk.bindings[0]
        before_indices = binding.joint_indices.copy()
        before_dist = binding.rest_neighbor_dist.copy()

        # The morph shrinks the mesh's rest pose and moves the skeleton with it.
        mesh.rest_positions = (np.asarray(mesh.rest_positions) * 0.9).astype(np.float32)
        nodes[2].set_position(0.0, 0.0, 27.0)
        scene.update()

        assert sk.resnapshot_rest(_chains(nodes)) is True
        assert len(sk.bindings) == 1, "the binding is kept, not re-solved"
        np.testing.assert_array_equal(sk.bindings[0].joint_indices, before_indices)
        assert not np.allclose(sk.bindings[0].rest_neighbor_dist, before_dist), \
            "the neighbour baseline is measured on the new rest pose"

    def test_it_refuses_when_the_joint_list_changed(self):
        """Then the per-vertex indices really would point at the wrong joints."""
        scene, wrapper, nodes, sk = _rig()
        mesh = _grid_mesh()
        sk.register_skin_mesh(mesh, is_muscle=True, muscle_name=mesh.name)
        assert sk.resnapshot_rest([[(n.name, n) for n in nodes[:2]]]) is False

    def test_it_drops_the_caches_derived_from_the_old_rest_pose(self):
        scene, wrapper, nodes, sk = _rig()
        mesh = _grid_mesh()
        sk.register_skin_mesh(mesh, is_muscle=True, muscle_name=mesh.name)
        binding = sk.bindings[0]
        for attr in ("_rest_f64", "_pos_h", "_captured_ref", "_hull_used"):
            setattr(binding, attr, "stale")
        assert sk.resnapshot_rest(_chains(nodes)) is True
        for attr in ("_rest_f64", "_pos_h", "_captured_ref", "_hull_used"):
            assert not hasattr(binding, attr), attr

    def test_the_morphed_rest_pose_is_what_gets_drawn(self):
        """After a re-snapshot the body IS the morphed body, with no delta left."""
        scene, wrapper, nodes, sk = _rig()
        mesh = _grid_mesh()
        sk.register_skin_mesh(mesh, is_muscle=True, muscle_name=mesh.name)
        morphed = (np.asarray(mesh.rest_positions).reshape(-1, 3) * 0.9)
        mesh.rest_positions = morphed.reshape(-1).astype(np.float32)
        scene.update()
        assert sk.resnapshot_rest(_chains(nodes)) is True
        sk._last_signature = ()
        sk.update(BodyState())
        drawn = np.asarray(mesh.geometry.positions, dtype=np.float64).reshape(-1, 3)
        np.testing.assert_allclose(drawn, morphed, atol=1e-3)
