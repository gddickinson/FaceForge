"""A female model must not keep the male organs, and should have breast tissue.

The BodyParts3D set is a male cadaver: eight of the configured organs are male
reproductive structures and there is no female equivalent of any of them in the
asset set, nor a breast.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceforge.body import breast_tissue, sex_specific
from faceforge.core.mesh import BufferGeometry, MeshInstance
from faceforge.core.scene_graph import SceneNode


# -- sex-specific structures -------------------------------------------------


@pytest.fixture
def organs():
    root = SceneNode(name="bodyRoot")
    for name in ("Prostate", "Left Testis", "Urethra", "Liver", "Heart"):
        node = SceneNode(name=name)
        node.mesh = MeshInstance(
            name=name,
            geometry=BufferGeometry(positions=np.zeros(9, dtype=np.float32),
                                    normals=np.zeros(9, dtype=np.float32)))
        root.add(node)
    return root


def visible(root) -> dict[str, bool]:
    return {c.name: c.visible for c in root.children}


def test_a_male_model_keeps_everything(organs):
    sex_specific.apply(organs, 0.0)
    assert all(visible(organs).values())


def test_a_female_model_keeps_no_male_organ(organs):
    sex_specific.apply(organs, 1.0)
    seen = visible(organs)
    assert seen["Prostate"] is False
    assert seen["Left Testis"] is False
    assert seen["Liver"] is True and seen["Heart"] is True


def test_the_male_urethra_goes_too(organs):
    """Both sexes have one; this mesh is 20 cm and runs through the penis."""
    sex_specific.apply(organs, 1.0)
    assert visible(organs)["Urethra"] is False


def test_they_come_back_when_the_slider_does(organs):
    sex_specific.apply(organs, 1.0)
    sex_specific.apply(organs, 0.0)
    assert all(visible(organs).values())


def test_the_change_happens_where_the_body_is_ambiguous():
    assert 0.0 < sex_specific.HIDDEN_BY < 1.0


def test_a_name_is_matched_whole_not_as_a_substring():
    """"Prostate" must not take the prostatic urethra with it."""
    root = SceneNode(name="bodyRoot")
    node = SceneNode(name="Prostatic Utricle")
    node.mesh = MeshInstance(
        name="Prostatic Utricle",
        geometry=BufferGeometry(positions=np.zeros(9, dtype=np.float32),
                                normals=np.zeros(9, dtype=np.float32)))
    root.add(node)
    sex_specific.apply(root, 1.0)
    assert node.visible is True


# -- breast tissue -----------------------------------------------------------


def chest_mesh():
    """A coarse dome over the chest, on both sides, with inward normals."""
    u = np.linspace(-1.0, 1.0, 13)
    grid = np.array([(x, y) for x in u for y in u])
    pts, faces = [], []
    for sign in (1.0, -1.0):
        base = len(pts)
        for gx, gy in grid:
            r = np.hypot(gx, gy)
            pts.append([sign * 12.0 + gx * 11.0,
                        -16.0 - 3.0 * max(0.0, 1.0 - r * r),
                        -55.0 + gy * 11.0])
        n = len(u)
        for i in range(n - 1):
            for j in range(n - 1):
                a = base + i * n + j
                faces.append([a, a + 1, a + n])
                faces.append([a + 1, a + n + 1, a + n])
    pts = np.asarray(pts)
    return pts, np.asarray(faces, dtype=np.int64)


def test_it_finds_a_breast_on_each_side():
    male, faces = chest_mesh()
    female = male.copy()
    female[:, 1] -= 1.0
    normals = np.tile([0.0, 1.0, 0.0], (len(male), 1))   # inward, as the asset is
    tissue = breast_tissue.build(male, female, faces, normals)
    assert tissue is not None
    centres = tissue.positions(1.0)[: len(tissue.source)]
    assert (centres[:, 0] > 0).any() and (centres[:, 0] < 0).any()


def test_the_lens_is_deepest_at_the_nipple_and_nothing_at_its_base():
    male, faces = chest_mesh()
    normals = np.tile([0.0, 1.0, 0.0], (len(male), 1))
    tissue = breast_tissue.build(male, male.copy(), faces, normals)
    assert tissue is not None
    assert tissue.profile.max() == pytest.approx(1.0, abs=0.05)
    assert tissue.profile.min() < 0.4 * tissue.profile.max(), \
        "the lens must taper away from the nipple, not sit on a plinth"
    assert (tissue.profile >= 0.0).all()


def test_a_man_has_a_little_and_a_woman_a_lot():
    male, faces = chest_mesh()
    normals = np.tile([0.0, 1.0, 0.0], (len(male), 1))
    tissue = breast_tissue.build(male, male.copy(), faces, normals)
    assert tissue is not None
    assert 0.0 < tissue.volume(0.0) < tissue.volume(0.5) < tissue.volume(1.0)


def test_the_shell_is_closed_and_wound_outward():
    male, faces = chest_mesh()
    normals = np.tile([0.0, 1.0, 0.0], (len(male), 1))
    tissue = breast_tissue.build(male, male.copy(), faces, normals)
    assert tissue is not None
    edges = np.vstack([tissue.faces[:, [0, 1]], tissue.faces[:, [1, 2]],
                       tissue.faces[:, [2, 0]]])
    _, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    assert (counts == 2).all(), "a shell with a hole in it is not a volume"
    _, directed = np.unique(edges, axis=0, return_counts=True)
    assert (directed == 1).all(), "inconsistent winding renders as holes"
    assert tissue.signed_volume(1.0) > 0.0


def test_no_chest_means_no_tissue():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    normals = np.tile([0.0, 1.0, 0.0], (3, 1))
    assert breast_tissue.build(pts, pts.copy(), faces, normals) is None
