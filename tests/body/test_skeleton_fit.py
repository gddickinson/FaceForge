"""Fitting the skeleton into a body-surface mesh: what the transform must keep.

These run on a hand-built three-bone leg, not on the asset set, so what is
asserted is the *rule* -- the articulations stay shut, the fit is exactly
undoable, the soft tissue and the target surface are never written to -- and
not a number that happens to fall out of one pair of meshes.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceforge.body.fit_regions import (
    REGION_NAMES, RegionTransforms, anchors, blend_tables, region_of,
    rotation_matrix,
)
from faceforge.body.skeleton_field import compose
from faceforge.body.skeleton_fit import SkeletonFit, node_offset
from faceforge.core.mesh import BufferGeometry, MeshInstance
from faceforge.core.scene_graph import SceneNode


def bone(name: str, points: np.ndarray) -> SceneNode:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    geo = BufferGeometry(positions=pts.reshape(-1).copy(),
                         normals=np.zeros(pts.size, dtype=np.float32))
    node = SceneNode(name=name)
    node.mesh = MeshInstance(name=name, geometry=geo)
    node.mesh.store_rest_pose()
    return node


@pytest.fixture
def leg():
    """bodyRoot -> hip_R_pivot -> knee_R_pivot -> ankle_R_pivot, one bone each.

    Each bone runs from its own pivot down to the next, so a segment that
    scales without its child pivot following shows up immediately as a gap.
    """
    root = SceneNode(name="bodyRoot")

    hip = SceneNode(name="hip_R_pivot")
    hip.set_position(10.0, 0.0, -80.0)
    hip.add(bone("Right Femur", [[0, 0, 0], [0, 0, -30], [0, 0, -60]]))
    root.add(hip)

    knee = SceneNode(name="knee_R_pivot")
    knee.set_position(0.0, 0.0, -60.0)
    knee.add(bone("Right Tibia", [[0, 0, 0], [0, 0, -20], [0, 0, -40]]))
    hip.add(knee)

    ankle = SceneNode(name="ankle_R_pivot")
    ankle.set_position(0.0, 0.0, -40.0)
    ankle.add(bone("R Calcaneus", [[0, 0, 0], [0, -8, -5]]))
    knee.add(ankle)

    # The surface the skeleton is fitted to, and a muscle on the femur:
    # neither may be written to.
    surface = SceneNode(name="bodyMeshGroup")
    surface.add(bone("body_surface", [[0, 0, 0], [1, 1, 1]]))
    root.add(surface)
    muscle = bone("Quadriceps", [[0, 0, -10], [0, 0, -40]])
    hip.add(muscle)

    joints = {"hip_R": np.array([10.0, 0.0, -80.0]),
              "knee_R": np.array([10.0, 0.0, -140.0]),
              "ankle_R": np.array([10.0, 0.0, -180.0])}
    return root, joints, muscle


def table(**regions) -> dict[str, dict[str, dict]]:
    both = {name: {} for name in REGION_NAMES}
    both.update(regions)
    return {"male": both, "female": both}


def world_points(node: SceneNode) -> np.ndarray:
    geo = node.mesh.geometry
    pts = np.asarray(geo.positions, dtype=np.float64).reshape(-1, 3)
    return pts + node_offset(node)


def find(root: SceneNode, name: str) -> SceneNode:
    node = root.find(name)
    assert node is not None, name
    return node


# -- membership --------------------------------------------------------------


def test_a_node_inherits_its_parents_region_unless_it_names_one():
    assert region_of("knee_R_pivot", "thigh_R") == "shank_R"
    assert region_of("Right Tibia", "shank_R") == "shank_R"
    assert region_of("clavicle_L_pivot", "thorax") == "girdle_L"
    assert region_of("Right 4th Rib", "thorax") == "thorax"


def test_the_knuckle_is_where_the_hand_stops_and_the_fingers_start():
    assert region_of("finger_R_3_mc_pivot", "hand_R") == "hand_R"
    assert region_of("finger_R_3_prox_pivot", "hand_R") == "fingers_R"
    assert region_of("toe_L_1_mt_pivot", "foot_L") == "foot_L"
    assert region_of("toe_L_1_dist_pivot", "toes_L") == "toes_L"


# -- the transform -----------------------------------------------------------


def test_amount_zero_leaves_the_skeleton_exactly_where_it_was(leg):
    root, joints, _ = leg
    before = world_points(find(root, "Right Femur"))
    SkeletonFit(table(thigh_R={"scale": [1.0, 1.0, 0.5]})
                ).apply(root, 0.0, 0.0, joints)
    assert np.allclose(world_points(find(root, "Right Femur")), before)


def test_a_shortened_thigh_takes_the_knee_with_it(leg):
    """The whole point: a bone may change length without the joint opening."""
    root, joints, _ = leg
    fit = SkeletonFit(table(thigh_R={"scale": [1.0, 1.0, 0.5]}))
    fit.apply(root, 1.0, 0.0, joints)

    femur_end = world_points(find(root, "Right Femur"))[-1]
    knee = node_offset(find(root, "knee_R_pivot"))
    assert np.allclose(femur_end, knee, atol=1e-9), \
        "the femur's distal end must still be at the knee"
    assert knee[2] == pytest.approx(-80.0 - 30.0), \
        "a 0.5 scale on a 60-unit thigh puts the knee 30 units below the hip"
    assert joints["knee_R"][2] == pytest.approx(-110.0)


def test_a_child_region_is_carried_by_its_parent_unchanged(leg):
    """The shank is not scaled, so it keeps its length -- lower down."""
    root, joints, _ = leg
    fit = SkeletonFit(table(thigh_R={"scale": [1.0, 1.0, 0.5]}))
    fit.apply(root, 1.0, 0.0, joints)
    tibia = world_points(find(root, "Right Tibia"))
    assert np.linalg.norm(tibia[-1] - tibia[0]) == pytest.approx(40.0)
    assert node_offset(find(root, "ankle_R_pivot"))[2] == pytest.approx(-150.0)


def test_the_fit_is_exactly_undoable(leg):
    root, joints, _ = leg
    before = {name: world_points(find(root, name))
              for name in ("Right Femur", "Right Tibia", "R Calcaneus")}
    joints_before = {k: v.copy() for k, v in joints.items()}
    fit = SkeletonFit(table(
        thigh_R={"scale": [0.9, 1.1, 0.5]},
        shank_R={"scale": [0.8, 0.8, 0.8]}))
    fit.apply(root, 1.0, 0.0, joints)
    assert fit.applied
    fit.reset(root, joints)
    assert not fit.applied
    for name, pts in before.items():
        assert np.allclose(world_points(find(root, name)), pts, atol=1e-9)
    for key, value in joints_before.items():
        assert np.allclose(joints[key], value, atol=1e-9)


def test_applying_twice_is_the_same_as_applying_once(leg):
    """The slider and the checkbox may be used in any order."""
    root, joints, _ = leg
    t = table(thigh_R={"scale": [0.9, 1.0, 0.7]})
    fit = SkeletonFit(t)
    fit.apply(root, 1.0, 0.0, joints)
    once = world_points(find(root, "Right Tibia"))
    fit.apply(root, 1.0, 0.0, joints)
    assert np.allclose(world_points(find(root, "Right Tibia")), once, atol=1e-9)


def test_the_surface_it_is_fitted_to_is_never_moved(leg):
    root, joints, _ = leg
    surface = find(root, "body_surface")
    before = np.array(surface.mesh.geometry.positions, copy=True)
    SkeletonFit(table(pelvis={"scale": [2.0, 2.0, 2.0],
                              "offset": [5.0, 5.0, 5.0]})
                ).apply(root, 1.0, 0.0, joints)
    assert np.array_equal(surface.mesh.geometry.positions, before)


def test_soft_tissue_the_skinning_owns_is_left_to_the_skinning(leg):
    root, joints, muscle = leg
    before = np.array(muscle.mesh.geometry.positions, copy=True)
    SkeletonFit(table(thigh_R={"scale": [1.0, 1.0, 0.5]})
                ).apply(root, 1.0, 0.0, joints, exclude={id(muscle.mesh)})
    assert np.array_equal(muscle.mesh.geometry.positions, before)


def test_an_offset_moves_a_region_bodily(leg):
    root, joints, _ = leg
    SkeletonFit(table(pelvis={"offset": [0.0, 0.0, -7.0]})).apply(
        root, 1.0, 0.0, joints)
    assert node_offset(find(root, "hip_R_pivot"))[2] == pytest.approx(-87.0)


def test_amount_blends_smoothly_between_the_two_skeletons(leg):
    root, joints, _ = leg
    t = table(thigh_R={"scale": [1.0, 1.0, 0.5]})
    fit = SkeletonFit(t)
    fit.apply(root, 0.5, 0.0, joints)
    assert node_offset(find(root, "knee_R_pivot"))[2] == pytest.approx(-125.0)


def test_a_missing_config_leaves_the_option_inert(leg):
    root, joints, _ = leg
    before = world_points(find(root, "Right Femur"))
    fit = SkeletonFit({"male": {}, "female": {}})
    assert not fit.available
    fit.apply(root, 1.0, 0.0, joints)
    assert np.allclose(world_points(find(root, "Right Femur")), before)


# -- the two tables ----------------------------------------------------------


def test_the_male_and_female_fits_are_lerped_by_the_sex_slider():
    male = {"pelvis": {"scale": [1.0, 1.0, 1.0], "offset": [0.0, 0.0, 0.0]}}
    female = {"pelvis": {"scale": [0.5, 0.5, 0.5], "rotation": [0.0, 8.0, 0.0],
                         "offset": [0.0, 0.0, -10.0]}}
    blended = blend_tables(male, female, 0.5)
    assert blended["pelvis"]["scale"] == [0.75, 0.75, 0.75]
    assert blended["pelvis"]["rotation"] == [0.0, 4.0, 0.0]
    assert blended["pelvis"]["offset"] == [0.0, 0.0, -5.0]


def test_every_region_has_an_anchor_on_a_real_skeleton(leg):
    root, joints, _ = leg
    points = anchors(root, joints, node_offset)
    assert np.allclose(points["hip_R"], [10.0, 0.0, -80.0])
    assert "pelvic_centre" in points and "cervicothoracic" in points


def test_identity_is_recognised_so_nothing_is_rewritten():
    points = {"pelvic_centre": np.zeros(3)}
    t = RegionTransforms({name: {} for name in REGION_NAMES}, points, 1.0)
    assert t.is_identity()


# -- composing two skeleton changes -----------------------------------------


def test_two_warps_are_chained_not_added():
    """The second was measured on a body the first had already moved."""
    first = lambda p: np.tile([0.0, 0.0, 10.0], (len(p), 1))       # noqa: E731
    second = lambda p: p * 0.0 + np.where(                          # noqa: E731
        p[:, 2:3] > 5.0, 1.0, -1.0) * np.array([1.0, 0.0, 0.0])
    both = compose(first, second)
    at_zero = both(np.zeros((1, 3)))
    # After the first warp the point is at z = 10, so the second sees it above
    # the threshold; adding the two fields would have seen z = 0 and gone the
    # other way.
    assert at_zero[0][0] == pytest.approx(1.0)
    assert at_zero[0][2] == pytest.approx(10.0)


def test_compose_of_nothing_is_nothing():
    assert compose(None, None) is None
    warp = lambda p: p * 0.0                                        # noqa: E731
    assert compose(None, warp) is warp


# -- rotations compose down the chain ---------------------------------------


def test_turning_the_thigh_carries_the_shank_and_the_foot(leg):
    """A pose, not a set of loose rotations: the limb turns as a limb.

    Before this, each region's rotation was absolute in body coordinates, so
    a turn at the hip moved the knee but left the shank pointing where it
    always had -- the leg bent instead of swinging, and the hand had to
    express the whole arm's turn by itself, which is why it was the one region
    that ended against the rotation bound.
    """
    root, joints, _ = leg
    SkeletonFit(table(thigh_R={"rotation": [0.0, 30.0, 0.0]})).apply(
        root, 1.0, 0.0, joints)

    hip = np.array([10.0, 0.0, -80.0])
    rot = rotation_matrix([0.0, 30.0, 0.0])
    for name, rest_end in (("Right Femur", np.array([10.0, 0.0, -140.0])),
                           ("Right Tibia", np.array([10.0, 0.0, -180.0]))):
        expected = hip + rot @ (rest_end - hip)
        assert np.allclose(world_points(find(root, name))[-1], expected, atol=1e-9)


def test_a_child_turns_relative_to_the_parent_it_hangs_from(leg):
    root, joints, _ = leg
    SkeletonFit(table(thigh_R={"rotation": [0.0, 20.0, 0.0]},
                      shank_R={"rotation": [0.0, 20.0, 0.0]})).apply(
        root, 1.0, 0.0, joints)
    hip = np.array([10.0, 0.0, -80.0])
    knee = hip + rotation_matrix([0.0, 20.0, 0.0]) @ np.array([0.0, 0.0, -60.0])
    ankle = knee + rotation_matrix([0.0, 40.0, 0.0]) @ np.array([0.0, 0.0, -40.0])
    assert np.allclose(node_offset(find(root, "knee_R_pivot")), knee, atol=1e-9)
    assert np.allclose(node_offset(find(root, "ankle_R_pivot")), ankle, atol=1e-9)


def test_half_a_rotation_is_half_a_rotation_not_a_squashed_one(leg):
    """``amount`` blends the parameters, so the limb stays the length it is."""
    root, joints, _ = leg
    SkeletonFit(table(thigh_R={"rotation": [0.0, 40.0, 0.0]})).apply(
        root, 0.5, 0.0, joints)
    hip = np.array([10.0, 0.0, -80.0])
    knee = node_offset(find(root, "knee_R_pivot"))
    assert np.linalg.norm(knee - hip) == pytest.approx(60.0)
    expected = hip + rotation_matrix([0.0, 20.0, 0.0]) @ np.array([0.0, 0.0, -60.0])
    assert np.allclose(knee, expected, atol=1e-9)
