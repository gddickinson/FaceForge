"""The flesh of a body part, as a distance the skin binding can rank by.

Nearest-bone binding fails wherever the bone is not under the skin.  With the
arms hanging in the rest pose the forearm is about 3 units from the flank and
the lumbar spine about 19, so the flank binds to the arm and is drawn out
along it when the arm lifts.  Muscle fills the soft tissue, so the flesh
nearest a patch of skin is the flesh that skin sits on: measured on the spikes
that survived every other fix, the nearest muscle is a trunk muscle for 78% of
them while their nearest bone says arm.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceforge.body.muscle_field import (
    GROUPS,
    MuscleChainField,
    group_of_joint,
)


@pytest.mark.parametrize("name,expected", [
    ("shoulder_R", "arm_R"),
    ("elbow_L", "arm_L"),
    ("wrist_R", "arm_R"),
    ("clavicle_L", "arm_L"),
    ("scapula_R", "arm_R"),
    ("finger_R_2_mid", "hand_R"),
    ("toe_L_3_prox", "foot_L"),
    ("hip_R", "leg_R"),
    ("knee_L", "leg_L"),
    ("ankle_R", "leg_R"),
    ("rib_12", "trunk"),
    ("lumbar_3", "trunk"),
    ("thoracic_0", "trunk"),
])
def test_a_joint_is_assigned_to_its_body_part(name, expected):
    assert group_of_joint(name) == expected


def test_every_group_a_joint_maps_to_is_one_the_field_knows():
    for name in ("shoulder_R", "finger_L_1_mc", "toe_R_2_mt", "hip_L", "rib_0"):
        assert group_of_joint(name) in GROUPS


def _field():
    # Trunk flesh along the midline, arm flesh out to one side.
    trunk = np.stack([np.zeros(40), np.zeros(40), np.linspace(-80, 0, 40)], axis=1)
    arm = np.stack([np.full(40, 25.0), np.zeros(40), np.linspace(-80, 0, 40)], axis=1)
    return MuscleChainField({"trunk": trunk, "arm_R": arm})


def test_the_flesh_nearest_a_point_is_the_flesh_it_sits_on():
    f = _field()
    q = np.array([[3.0, 0.0, -40.0], [22.0, 0.0, -40.0]])
    d_trunk = f.distance(q, "trunk")
    d_arm = f.distance(q, "arm_R")
    assert d_trunk[0] < d_arm[0]          # near the midline: trunk flesh
    assert d_arm[1] < d_trunk[1]          # out at the arm: arm flesh


def test_a_body_part_with_no_muscles_says_so_rather_than_guessing():
    f = _field()
    d = f.distance(np.zeros((3, 3)), "leg_L")
    assert np.isinf(d).all()


def test_it_survives_a_round_trip_through_disk(tmp_path):
    f = _field()
    path = tmp_path / "field.npz"
    f.save(path)
    back = MuscleChainField.load(path)
    assert back is not None
    assert set(back.groups) == set(f.groups)
    q = np.array([[3.0, 0.0, -40.0]])
    assert back.distance(q, "trunk") == pytest.approx(f.distance(q, "trunk"))


def test_a_missing_file_is_not_an_error():
    assert MuscleChainField.load("/nonexistent/muscle_field.npz") is None


def test_the_weight_is_in_the_binding_cache_key():
    from faceforge.body import skinning_cache
    from faceforge.body.soft_tissue import SoftTissueSkinning

    sk = SoftTissueSkinning()
    a = dict(skinning_cache.scalar_tunables(sk))
    assert "MUSCLE_FIELD_WEIGHT" in a
    sk.MUSCLE_FIELD_WEIGHT = 0.25
    b = dict(skinning_cache.scalar_tunables(sk))
    assert a["MUSCLE_FIELD_WEIGHT"] != b["MUSCLE_FIELD_WEIGHT"]


def test_the_field_digest_is_in_the_binding_cache_key():
    """A rebuilt field must not be served a binding solved against the old one."""
    from faceforge.body import skinning_cache
    from faceforge.body.soft_tissue import SoftTissueSkinning

    sk = SoftTissueSkinning()
    assert "muscle_field_id" in dict(skinning_cache.scalar_tunables(sk))
    a = _field()
    b = MuscleChainField({"trunk": a.points["trunk"] + 1.0,
                          "arm_R": a.points["arm_R"]})
    assert a.digest != b.digest
    assert a.digest == MuscleChainField(dict(a.points)).digest


def test_the_field_gives_a_direction_into_the_body():
    """The inward test needs a point, not just a distance."""
    f = _field()
    q = np.array([[3.0, 0.0, -40.0]])
    p = f.nearest_point(q, "trunk")
    assert p is not None and p.shape == (1, 3)
    inward = p[0] - q[0]
    assert inward[0] < 0.0                 # from x = 3 toward the midline
    assert f.nearest_point(q, "leg_L") is None


def test_a_chain_seeds_only_where_its_bone_is_inward():
    """The flank and the forearm beside it are told apart by direction.

    Both bones are close to the skin between them and both have flesh close,
    so neither distance decides it.  Two sheets stand in for the trunk and
    the arm hanging beside it.  The arm bone is genuinely the nearer of the
    two to the trunk sheet -- 3.5 against 8 -- so no confidence margin saves
    it and the trunk sheet seeds the arm.  The arm keeps its own sheet either
    way, so the only thing the test can remove is its hold on the trunk's.
    """
    from faceforge.body.muscle_field import MuscleChainField
    from faceforge.body.soft_tissue import SoftTissueSkinning

    def sheet(x0):
        z = np.linspace(-20.0, 20.0, 41)
        y = np.linspace(-6.0, 6.0, 7)
        gz, gy = np.meshgrid(z, y, indexing="ij")
        return np.stack([np.full(gz.size, x0), gy.ravel(), gz.ravel()], axis=1)

    trunk_skin, arm_skin = sheet(0.0), sheet(6.0)
    positions = np.vstack([trunk_skin, arm_skin])
    n = len(trunk_skin)

    def grid_edges(off):
        idx = np.arange(n).reshape(41, 7) + off
        return np.concatenate([
            np.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()], axis=1),
            np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], axis=1)])

    # Joined only at the top, as the arm joins the trunk at the shoulder.
    edges = np.vstack([grid_edges(0), grid_edges(n), np.array([[6, n + 6]])])
    lengths = np.linalg.norm(
        positions[edges[:, 0]] - positions[edges[:, 1]], axis=1)

    sk = SoftTissueSkinning()
    sk.muscle_field = MuscleChainField({
        "trunk": trunk_skin + np.array([-3.0, 0.0, 0.0]),
        "arm_R": arm_skin + np.array([-1.0, 0.0, 0.0]),
    })
    # The arm bone sits in the gap, 3.5 from the trunk sheet -- inside the
    # seed radius, as the hanging forearm is from the flank.
    seg_starts = np.array([[-8.0, 0.0, -20.0], [3.5, 0.0, -20.0]])
    seg_ends = np.array([[-8.0, 0.0, 20.0], [3.5, 0.0, 20.0]])
    seg_chains = np.array([0, 1], dtype=np.int32)

    # Isolate the direction rule: the own-flesh rule, which ships on, would
    # stop the arm seeding the trunk sheet on its own and hide what this is
    # testing.
    sk.SEED_ON_OWN_FLESH = False
    sk.SEED_INWARD_ONLY = False
    loose = sk._geodesic_chain_dists(positions, edges, lengths,
                                     seg_starts, seg_ends, seg_chains)
    sk.SEED_INWARD_ONLY = True
    strict = sk._geodesic_chain_dists(positions, edges, lengths,
                                      seg_starts, seg_ends, seg_chains)

    mid = n // 2                       # a trunk vertex far from the join
    # Without the test the arm reaches the trunk sheet directly across the gap.
    assert loose[mid, 1] < 6.0
    # With it, the arm can only reach it the long way round, over the join.
    assert strict[mid, 1] > loose[mid, 1] + 5.0
    # The trunk's own field is unchanged, and the arm still owns its own sheet.
    assert strict[mid, 0] == pytest.approx(loose[mid, 0])
    arm_mid = n + n // 2
    assert strict[arm_mid, 1] < strict[arm_mid, 0]


def test_a_chain_seeds_only_skin_that_sits_on_its_own_flesh():
    """The rule that finished the flank, where direction alone was not enough.

    Skin on the lateral chest sits further out than the humerus, so the
    humerus is inward of it and the direction test lets the arm seed chest
    skin.  The flesh under that skin is the trunk's, and says so.
    """
    from faceforge.body.muscle_field import MuscleChainField
    from faceforge.body.soft_tissue import SoftTissueSkinning

    def sheet(x0):
        z = np.linspace(-20.0, 20.0, 41)
        y = np.linspace(-6.0, 6.0, 7)
        gz, gy = np.meshgrid(z, y, indexing="ij")
        return np.stack([np.full(gz.size, x0), gy.ravel(), gz.ravel()], axis=1)

    chest, arm = sheet(0.0), sheet(10.0)
    positions = np.vstack([chest, arm])
    n = len(chest)

    def grid_edges(off):
        idx = np.arange(n).reshape(41, 7) + off
        return np.concatenate([
            np.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()], axis=1),
            np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], axis=1)])

    edges = np.vstack([grid_edges(0), grid_edges(n), np.array([[6, n + 6]])])
    lengths = np.linalg.norm(
        positions[edges[:, 0]] - positions[edges[:, 1]], axis=1)

    from faceforge.body.soft_tissue import SkinJoint

    sk = SoftTissueSkinning()
    sk.SEED_INWARD_ONLY = False
    sk.joints = [SkinJoint(name="lumbar_0", node=None, chain_id=0),
                 SkinJoint(name="shoulder_R", node=None, chain_id=1)]
    sk.muscle_field = MuscleChainField({
        "trunk": chest + np.array([-2.0, 0.0, 0.0]),
        "arm_R": arm + np.array([2.0, 0.0, 0.0]),
    })
    # The arm bone sits between the two sheets, 4 from the chest: inward of
    # it, within the seed radius, and on the wrong body part.
    seg_starts = np.array([[-6.0, 0.0, -20.0], [4.0, 0.0, -20.0]])
    seg_ends = np.array([[-6.0, 0.0, 20.0], [4.0, 0.0, 20.0]])
    seg_chains = np.array([0, 1], dtype=np.int32)

    sk.SEED_ON_OWN_FLESH = False
    loose = sk._geodesic_chain_dists(positions, edges, lengths,
                                     seg_starts, seg_ends, seg_chains)
    sk.SEED_ON_OWN_FLESH = True
    strict = sk._geodesic_chain_dists(positions, edges, lengths,
                                      seg_starts, seg_ends, seg_chains)

    mid = n // 2
    assert loose[mid, 1] < 6.0                       # the arm reaches straight over
    assert strict[mid, 1] > loose[mid, 1] + 5.0      # now only the long way
    arm_mid = n + n // 2
    assert strict[arm_mid, 1] < strict[arm_mid, 0]   # the arm keeps its own sheet
