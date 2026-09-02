"""A skeleton rebuild must not silently rebind meshes to other chains.

Gender scaling moves bone positions, so ``rebuild_skin_joints`` has to
re-snapshot the rest matrices and every mesh has to be re-solved.  The bug this
module locks out: the re-registration passed only ``is_muscle`` and
``muscle_name``, dropping ``allowed_chains`` / ``spatial_limit`` /
``chain_z_margin``.  Each mesh was then solved against ALL chains and bound to
whatever bone was nearest, so a torso mesh constrained to the spine picked up
the arm chain and torso geometry followed arm motion.

The defect was invisible to the suite: nothing raised, and the geometry was
merely wrong.  These tests therefore assert on the SOLVE RESULT (which chain
each vertex ended up on), not on the arguments passed.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceforge.body.soft_tissue import SoftTissueSkinning
from faceforge.core.material import Material
from faceforge.core.mesh import BufferGeometry, MeshInstance
from faceforge.core.scene_graph import SceneNode

SPINE_CHAIN = 0
ARM_CHAIN = 1


def _node(name: str, pos) -> SceneNode:
    n = SceneNode(name=name)
    n.position = np.array(pos, dtype=np.float64)
    n.update_world_matrix(force=True)
    return n


@pytest.fixture
def chains():
    """A vertical spine and a horizontal arm chain leaving the shoulder."""
    spine = [(f"spine_{i}", _node(f"spine_{i}", (0.0, 0.0, z)))
             for i, z in enumerate([-40, -20, 0, 20, 40])]
    arm = [(f"arm_{i}", _node(f"arm_{i}", (x, 0.0, 38.0)))
           for i, x in enumerate([12, 30, 50, 68])]
    return [spine, arm]


@pytest.fixture
def shoulder_region_mesh():
    """Upper-lateral torso, where the arm chain really is the nearest bone.

    This placement is the point of the fixture: for a mesh sitting medial to
    the shoulder the nearest-bone solve gets the right answer by accident, so
    it cannot detect a lost constraint.
    """
    gx, gy, gz = np.meshgrid(np.linspace(4, 22, 14), [0.0],
                             np.linspace(28, 40, 10))
    pos = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3).astype(np.float32)
    geom = BufferGeometry(
        positions=pos.ravel(),
        normals=np.tile([0.0, 1.0, 0.0], len(pos)).astype(np.float32),
        indices=np.arange((len(pos) // 3) * 3, dtype=np.uint32),
    )
    mesh = MeshInstance(name="shoulder_region", geometry=geom,
                        material=Material())
    mesh.store_rest_pose()
    return mesh


def _chain_ids(sk: SoftTissueSkinning, binding) -> np.ndarray:
    return np.array([sk.joints[j].chain_id for j in binding.joint_indices])


def test_the_fixture_can_actually_detect_the_bug(chains, shoulder_region_mesh):
    """Negative control: unconstrained, this mesh DOES bind to the arm chain.

    Without this, a test asserting "0 vertices on the arm chain" would pass
    just as happily on a mesh nowhere near the arm, and would lock in nothing.
    """
    sk = SoftTissueSkinning()
    sk.build_skin_joints(chains)
    sk.register_skin_mesh(shoulder_region_mesh, is_muscle=True,
                          allowed_chains=None)
    bled = int((_chain_ids(sk, sk.bindings[0]) == ARM_CHAIN).sum())
    assert bled > 0, (
        "fixture no longer exercises the defect: an unconstrained solve put no "
        "vertices on the arm chain, so the assertions below prove nothing"
    )


def test_constraints_are_recorded_on_the_binding(chains, shoulder_region_mesh):
    sk = SoftTissueSkinning()
    sk.build_skin_joints(chains)
    sk.register_skin_mesh(shoulder_region_mesh, is_muscle=False,
                          allowed_chains={SPINE_CHAIN},
                          spatial_limit=25.0, chain_z_margin=15.0)
    b = sk.bindings[0]
    assert b.allowed_chains == {SPINE_CHAIN}
    assert b.spatial_limit == 25.0
    assert b.chain_z_margin == 15.0


def test_skeleton_rebuild_preserves_the_chain_assignment(
        chains, shoulder_region_mesh):
    """The regression itself: rebind after a rebuild must not move vertices."""
    sk = SoftTissueSkinning()
    sk.build_skin_joints(chains)
    sk.register_skin_mesh(shoulder_region_mesh, is_muscle=True,
                          allowed_chains={SPINE_CHAIN})
    before = sk.bindings[0]
    chains_before = _chain_ids(sk, before)
    joints_before = before.joint_indices.copy()

    # Exactly what the gender path does.
    old = list(sk.bindings)
    sk.clear_bindings()
    sk.rebuild_skin_joints(chains)
    for b in old:
        sk.register_skin_mesh(b.mesh, **b.rebind_kwargs())

    after = sk.bindings[0]
    chains_after = _chain_ids(sk, after)

    assert int((chains_after == ARM_CHAIN).sum()) == 0, (
        "a skeleton rebuild rebound torso vertices to the arm chain: "
        f"{int((chains_after == ARM_CHAIN).sum())} of {len(chains_after)}"
    )
    np.testing.assert_array_equal(chains_before, chains_after)
    np.testing.assert_array_equal(joints_before, after.joint_indices)


def test_rebind_kwargs_round_trips_every_constraint(chains,
                                                    shoulder_region_mesh):
    """Guards against a new constraint being added and not carried over."""
    sk = SoftTissueSkinning()
    sk.build_skin_joints(chains)
    sk.register_skin_mesh(shoulder_region_mesh, is_muscle=True,
                          allowed_chains={SPINE_CHAIN}, spatial_limit=25.0,
                          chain_z_margin=15.0, use_geodesic=False,
                          head_follow_config={"upperFrac": 0.3},
                          muscle_name="test_muscle", side="R",
                          physics_deform=True)
    kw = sk.bindings[0].rebind_kwargs()
    assert kw == {
        "is_muscle": True,
        "allowed_chains": {SPINE_CHAIN},
        "spatial_limit": 25.0,
        "chain_z_margin": 15.0,
        "use_geodesic": False,
        "head_follow_config": {"upperFrac": 0.3},
        "muscle_name": "test_muscle",
        # Side eligibility joined the constraint set when `ribs` turned out to
        # be one unsided chain: chain filtering could not express "right ribs
        # only", and ~600 vertices of the right pectoralis major clavicular
        # head were bound to the LEFT 12th rib. It has to survive a rebuild
        # like the others, or a gender change silently restores the bug.
        "side": "R",
        # Physics opt-in must survive too: a muscle that needs the pass in the
        # authored config must still get it after a gender rebuild.
        "physics_deform": True,
    }
