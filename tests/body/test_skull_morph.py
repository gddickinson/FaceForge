"""Cranial sex: the head used to be the one part the slider did not touch.

Measured on the shipped configuration before this existed, exactly five bone
meshes were unchanged between gender 0 and gender 1 -- the cranium, the jaw,
both sets of teeth and the atlas -- which is to say the whole head.  The
merged skull matched none of the per-bone cranial patterns, because those name
bones (``frontal bone``, ``zygomatic``) and the asset is one mesh called
``cranium``.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceforge.body import skull_morph
from faceforge.body.bone_scaling import BoneScaler
from faceforge.core.mesh import BufferGeometry, MeshInstance
from faceforge.core.scene_graph import SceneNode


@pytest.fixture
def skull():
    """A crude skull: a column of vertices from crown to chin, off the midline."""
    z = np.linspace(10.0, -10.0, 21)
    pts = np.stack([np.full_like(z, 8.0), np.zeros_like(z), z], axis=1)
    pts = np.vstack([pts, np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)])
    geo = BufferGeometry(positions=pts.reshape(-1).astype(np.float32),
                         normals=np.zeros(pts.size, dtype=np.float32))
    node = SceneNode(name="cranium")
    node.mesh = MeshInstance(name="cranium", geometry=geo)
    node.mesh.store_rest_pose()
    root = SceneNode(name="bodyRoot")
    root.add(node)
    return root, node


def positions(node) -> np.ndarray:
    return np.asarray(node.mesh.geometry.positions,
                      dtype=np.float64).reshape(-1, 3)


# -- the patterns ------------------------------------------------------------


@pytest.mark.parametrize("name,key", [
    ("cranium", "cranium"),
    ("jaw", "mandible"),
    ("upper_teeth", "tooth"),
    ("lower_teeth", "tooth"),
    ("Atlas (C1)", "vertebra"),
    ("Axis (C2)", "vertebra"),
    ("R Sesamoid", "phalanx_foot"),
])
def test_every_head_mesh_now_has_a_sex(name, key):
    assert BoneScaler()._match_bone(name) == key


@pytest.mark.parametrize("name,key", [
    ("Upper Jaw Gingiva", "maxilla"),
    ("Lower Jaw Gingiva", "mandible"),
])
def test_the_gums_follow_the_bone_they_sit_on(name, key):
    """"jaw" as a word also appears in the gingivae, which are not the jaw."""
    assert BoneScaler()._match_bone(name) == key


def test_the_published_head_ratios_are_what_the_config_says():
    scales = BoneScaler()._bone_scales["cranium"]
    assert scales == pytest.approx([14.5 / 15.2, 18.1 / 18.9, 0.95], abs=0.001)


# -- the graded narrowing ----------------------------------------------------


def test_the_grade_is_nothing_at_the_crown_and_everything_at_the_face():
    z = np.array([[0, 0, 10.0], [0, 0, 0.0], [0, 0, -10.0]])
    t = skull_morph.facial_grade(z, 10.0, -10.0)
    assert t[0] == pytest.approx(0.0)
    assert t[-1] == pytest.approx(1.0)
    assert 0.0 < t[1] < 1.0


def test_the_male_skull_is_left_exactly_as_it_is(skull):
    root, node = skull
    before = positions(node).copy()
    assert skull_morph.apply(root, 0.0) == 1
    assert np.allclose(positions(node), before)


def test_the_face_narrows_further_than_the_vault(skull):
    root, node = skull
    before = positions(node).copy()
    skull_morph.apply(root, 1.0)
    after = positions(node)
    crown = np.argmax(before[:, 2])
    chin = np.argmin(before[:, 2])
    assert after[crown, 0] == pytest.approx(before[crown, 0]), \
        "the vault already has its own factor; this must not touch it"
    assert after[chin, 0] == pytest.approx(
        before[chin, 0] * skull_morph.FACE_NARROWING)


def test_it_narrows_about_the_midline_so_nothing_drifts_off_centre(skull):
    root, node = skull
    before = positions(node).copy()
    skull_morph.apply(root, 1.0)
    after = positions(node)
    midline = np.abs(before[:, 0]) < 1e-9
    assert np.allclose(after[midline, 0], 0.0)
    assert np.allclose(after[:, 1:], before[:, 1:]), "only breadth changes"


def test_it_is_applied_in_proportion_to_the_slider(skull):
    root, node = skull
    before = positions(node).copy()
    skull_morph.apply(root, 0.5)
    after = positions(node)
    chin = np.argmin(before[:, 2])
    half = 1.0 + 0.5 * (skull_morph.FACE_NARROWING - 1.0)
    assert after[chin, 0] == pytest.approx(before[chin, 0] * half)


def test_soft_tissue_the_skinning_owns_is_not_touched(skull):
    root, node = skull
    before = positions(node).copy()
    assert skull_morph.apply(root, 1.0, exclude={id(node.mesh)}) == 0
    assert np.allclose(positions(node), before)
