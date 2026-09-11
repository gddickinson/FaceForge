"""The sex morph scales the skeleton as a hierarchy, so the joints stay together."""

from __future__ import annotations

import numpy as np
import pytest

from faceforge.body.bone_scaling import BoneScaler
from faceforge.body.skeleton_field import displacement_warp, sampled_warp
from faceforge.body.skeleton_morph import SkeletonMorph
from faceforge.core.mesh import BufferGeometry, MeshInstance
from faceforge.core.scene_graph import SceneNode


def _bone(name: str, pts: np.ndarray) -> SceneNode:
    geom = BufferGeometry(positions=pts.astype(np.float32).ravel(),
                          normals=np.zeros(pts.size, dtype=np.float32))
    node = SceneNode(name=name)
    node.mesh = MeshInstance(name=name, geometry=geom)
    node.mesh.store_rest_pose()
    return node


def _cylinder(length: float, radius: float = 2.0, n: int = 24) -> np.ndarray:
    """A bone: a ring at each end of a segment running down -Z from the origin."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([np.cos(th) * radius, np.sin(th) * radius, np.zeros(n)], axis=1)
    return np.concatenate([ring, ring + np.array([0.0, 0.0, -length])])


def _arm_rig():
    """bodyRoot -> shoulder pivot -> humerus, elbow pivot -> radius, wrist pivot."""
    root = SceneNode(name="bodyRoot")
    shoulder = SceneNode(name="shoulder_R_pivot")
    shoulder.set_position(20.0, 0.0, -15.0)
    root.add(shoulder)
    shoulder.add(_bone("Right Humerus", _cylinder(30.0)))
    elbow = SceneNode(name="elbow_R_pivot")
    elbow.set_position(0.0, 0.0, -30.0)
    shoulder.add(elbow)
    elbow.add(_bone("Right Radius", _cylinder(25.0)))
    wrist = SceneNode(name="wrist_R_pivot")
    wrist.set_position(0.0, 0.0, -25.0)
    elbow.add(wrist)
    wrist.add(_bone("R 1st Metacarpal", _cylinder(8.0, radius=1.0)))
    root.update_world_matrix(force=True)
    return root, shoulder, elbow, wrist


def _world(node: SceneNode) -> np.ndarray:
    g = node.mesh.geometry
    p = np.asarray(g.positions, dtype=np.float64).reshape(-1, 3)
    m = np.asarray(node.world_matrix, dtype=np.float64)
    return p @ m[:3, :3].T + m[:3, 3]


def _gap(a: SceneNode, b: SceneNode) -> float:
    pa, pb = _world(a), _world(b)
    return float(np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2).min())


def _find(root: SceneNode, name: str) -> SceneNode:
    if root.name == name:
        return root
    for c in root.children:
        hit = _find(c, name)
        if hit is not None:
            return hit
    return None


class TestHierarchy:
    def test_scaling_a_limb_does_not_open_its_joints(self):
        """A bone scaled about its own centroid pulls away from the next one.

        Measured on the shipped skeleton before this: the knee opened from
        0.07 to 3.55 units and the elbow from 0.59 to 3.89.
        """
        root, shoulder, elbow, wrist = _arm_rig()
        humerus = _find(root, "Right Humerus")
        radius = _find(root, "Right Radius")
        before = _gap(humerus, radius)
        assert before < 1e-6, "the test rig's bones start in contact"

        morph = SkeletonMorph(BoneScaler())
        morph.apply(root, 1.0)
        root.update_world_matrix(force=True)
        assert _gap(humerus, radius) < 1e-6, "the elbow must stay shut"

    def test_the_chain_shortens_and_the_distal_joints_follow(self):
        root, shoulder, elbow, wrist = _arm_rig()
        elbow_before = np.asarray(elbow.get_world_position(), dtype=float).copy()
        wrist_before = np.asarray(wrist.get_world_position(), dtype=float).copy()
        shoulder_before = np.asarray(shoulder.get_world_position(), dtype=float).copy()

        morph = SkeletonMorph(BoneScaler())
        morph.apply(root, 1.0)
        root.update_world_matrix(force=True)

        upper_before = np.linalg.norm(elbow_before - shoulder_before)
        upper_after = np.linalg.norm(np.asarray(elbow.get_world_position()) - shoulder_before)
        assert upper_after == pytest.approx(upper_before * 0.92, rel=1e-3)
        # The wrist follows the elbow rather than staying put.
        assert np.linalg.norm(np.asarray(wrist.get_world_position()) - wrist_before) > 1.0

    def test_gender_zero_restores_the_skeleton_exactly(self):
        root, *_ = _arm_rig()
        humerus = _find(root, "Right Humerus")
        before = np.asarray(humerus.mesh.geometry.positions).copy()
        morph = SkeletonMorph(BoneScaler())
        morph.apply(root, 1.0)
        morph.apply(root, 0.4)
        morph.apply(root, 0.0)
        np.testing.assert_allclose(humerus.mesh.geometry.positions, before, atol=1e-6)

    def test_soft_tissue_is_excluded_from_scaling(self):
        """A muscle is deformed by the morph, never scaled as though it were bone."""
        root, shoulder, *_ = _arm_rig()
        muscle = _bone("Tibialis Ant. R", _cylinder(20.0))
        shoulder.add(muscle)
        before = np.asarray(muscle.mesh.geometry.positions).copy()
        morph = SkeletonMorph(BoneScaler())
        morph.apply(root, 1.0, exclude={id(muscle.mesh)})
        np.testing.assert_allclose(muscle.mesh.geometry.positions, before, atol=1e-9)

    def test_joint_positions_are_refreshed(self):
        root, shoulder, elbow, wrist = _arm_rig()
        jp = {"shoulder_R": np.asarray(shoulder.get_world_position(), dtype=float).copy(),
              "elbow_R": np.asarray(elbow.get_world_position(), dtype=float).copy()}
        morph = SkeletonMorph(BoneScaler())
        morph.apply(root, 1.0, joint_positions=jp)
        root.update_world_matrix(force=True)
        np.testing.assert_allclose(jp["elbow_R"], elbow.get_world_position(), atol=1e-6)


class TestDisplacementField:
    def test_the_spline_reproduces_an_affine_change_exactly(self):
        """A uniformly scaled skeleton must scale its soft tissue uniformly."""
        rng = np.random.default_rng(0)
        pts = rng.normal(size=(30, 3)) * 25
        warp = displacement_warp(pts, pts * -0.1)
        q = rng.normal(size=(200, 3)) * 25
        np.testing.assert_allclose(warp(q), q * -0.1, atol=1e-6)

    def test_it_interpolates_the_control_points(self):
        rng = np.random.default_rng(1)
        pts = rng.normal(size=(20, 3)) * 30
        d = rng.normal(size=(20, 3))
        np.testing.assert_allclose(displacement_warp(pts, d)(pts), d, atol=1e-3)

    def test_the_lattice_matches_the_spline_and_falls_back_outside_it(self):
        rng = np.random.default_rng(2)
        pts = rng.normal(size=(24, 3)) * 20
        d = pts * -0.08
        exact = displacement_warp(pts, d)
        fast = sampled_warp(exact, pts, spacing=2.0)
        q = rng.normal(size=(500, 3)) * 20
        assert np.abs(fast(q) - exact(q)).max() < 0.05
        far = np.array([[900.0, 900.0, 900.0]])
        np.testing.assert_allclose(fast(far), exact(far), atol=1e-6)

    def test_no_control_points_is_a_no_op(self):
        warp = displacement_warp(np.zeros((0, 3)), np.zeros((0, 3)))
        np.testing.assert_allclose(warp(np.ones((5, 3))), np.zeros((5, 3)))
