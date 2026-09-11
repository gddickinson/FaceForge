"""The soft-tissue half of the sex morph: muscle bulk, fat distribution, no tearing."""

from __future__ import annotations

import numpy as np
import pytest

from faceforge.body.bone_scaling import BoneScaler
from faceforge.body.edge_relaxation import enforce_edge_range
from faceforge.body.muscle_morph import MuscleMorph
from faceforge.body.skin_morph import SkinShapeMorph, radial_component
from faceforge.body.soft_tissue_morph import SoftTissueMorph
from faceforge.core.mesh import BufferGeometry, MeshInstance


def _tube(length=20.0, radius=3.0, rings=9, seg=16):
    """A muscle-shaped tube along Z, with triangles."""
    th = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    zs = np.linspace(-length / 2, length / 2, rings)
    pts = np.concatenate([
        np.stack([np.cos(th) * radius, np.sin(th) * radius, np.full(seg, z)], axis=1)
        for z in zs])
    tris = []
    for r in range(rings - 1):
        for s in range(seg):
            a = r * seg + s
            b = r * seg + (s + 1) % seg
            tris += [(a, b, a + seg), (b, b + seg, a + seg)]
    return pts, np.asarray(tris, dtype=np.uint32).ravel()


def _mesh(name, pts, tris=None):
    geom = BufferGeometry(positions=pts.astype(np.float32).ravel(),
                          normals=np.zeros(pts.size, dtype=np.float32),
                          indices=tris)
    m = MeshInstance(name=name, geometry=geom)
    m.store_rest_pose()
    return m


class _Binding:
    def __init__(self, mesh, is_muscle=False):
        self.mesh = mesh
        self.is_muscle = is_muscle
        self.joint_indices = np.zeros(len(mesh.rest_positions) // 3, dtype=np.int32)


class TestMuscleBulk:
    def test_the_belly_thins_and_the_length_is_untouched(self):
        """Length comes from the bones; girth is the sex difference."""
        pts, _ = _tube()
        out = MuscleMorph().thin_belly(object(), pts, 0.6)
        assert out[:, 2].max() - out[:, 2].min() == pytest.approx(
            pts[:, 2].max() - pts[:, 2].min())
        mid = np.abs(pts[:, 2]) < 1e-9
        r_before = np.hypot(pts[mid, 0], pts[mid, 1])
        r_after = np.hypot(out[mid, 0], out[mid, 1])
        assert (r_after / r_before).mean() == pytest.approx(0.6, abs=0.02)

    def test_the_ends_stay_put_so_attachments_stay_attached(self):
        pts, _ = _tube()
        out = MuscleMorph().thin_belly(object(), pts, 0.5)
        end = np.abs(pts[:, 2]) >= pts[:, 2].max() - 1e-9
        np.testing.assert_allclose(out[end], pts[end], atol=1e-6)

    def test_the_upper_limb_is_the_more_dimorphic(self):
        m = MuscleMorph()
        assert m.bulk_factor("arm_R", 1.0) < m.bulk_factor("leg_R", 1.0) < 1.0
        assert m.bulk_factor("arm_R", 0.0) == 1.0
        assert m.bulk_factor("unknown", 1.0) == pytest.approx(0.70)


class TestSoftTissueField:
    def test_only_the_outward_component_survives(self):
        """A fat distribution is radial; a twist or a lift is not."""
        th = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        pts = np.stack([np.cos(th) * 10, np.sin(th) * 10, np.zeros(32)], axis=1)
        out = np.stack([np.cos(th), np.sin(th), np.zeros(32)], axis=1)
        np.testing.assert_allclose(radial_component(pts, out), out, atol=1e-9)
        tang = np.stack([-np.sin(th), np.cos(th), np.zeros(32)], axis=1)
        np.testing.assert_allclose(radial_component(pts, tang), 0.0, atol=1e-9)
        lift = np.tile([0.0, 0.0, 1.0], (32, 1))
        np.testing.assert_allclose(radial_component(pts, lift), 0.0, atol=1e-9)

    def test_the_uniform_size_change_is_removed_from_the_pair(self):
        """The skeleton already carries stature; applying it again shrinks twice."""
        pts, _ = _tube(length=40.0, radius=6.0)
        shrunk = pts * 0.8
        field = SkinShapeMorph.from_pair(pts, shrunk)
        assert np.abs(field._delta).max() < 0.2

    def test_a_real_bulge_survives_the_size_normalisation(self):
        pts, _ = _tube(length=40.0, radius=6.0)
        bulged = pts.copy()
        front = pts[:, 1] > 3
        bulged[front, 1] += 2.0
        field = SkinShapeMorph.from_pair(pts, bulged)
        assert np.linalg.norm(field._delta, axis=1).max() > 0.5

    def test_the_field_fades_with_distance_rather_than_cutting_off(self):
        """A hard cut-off tore the skin along its boundary."""
        src = np.zeros((8, 3))
        src[:, 0] = np.linspace(-4, 4, 8)
        field = SkinShapeMorph(src, np.tile([0.0, 1.0, 0.0], (8, 1)))
        near = field.delta_for(np.array([[0.0, 0.0, 1.0]]))[0, 1]
        mid = field.delta_for(np.array([[0.0, 0.0, 9.0]]))[0, 1]
        far = field.delta_for(np.array([[0.0, 0.0, 40.0]]))[0, 1]
        assert near > mid > far
        assert far == pytest.approx(0.0, abs=1e-6)


class TestNoTearing:
    def test_the_edge_band_reopens_a_collapsed_face(self):
        pos = np.array([[0.0, 0, 0], [1, 0, 0], [1, 0.02, 0], [0, 0.02, 0]])
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
        rest = np.ones(4)
        report = enforce_edge_range(pos, edges, rest, max_compression=0.2, iterations=40)
        d = np.linalg.norm(pos[edges[:, 0]] - pos[edges[:, 1]], axis=1)
        assert report["converged"]
        assert d.min() == pytest.approx(0.8, abs=1e-3)

    def test_it_also_pulls_an_over_stretched_edge_in_and_keeps_the_midpoint(self):
        pos = np.array([[0.0, 0, 0], [3.0, 0, 0]])
        enforce_edge_range(pos, np.array([[0, 1]]), np.array([1.0]),
                           max_stretch=0.25, iterations=20)
        assert np.linalg.norm(pos[1] - pos[0]) == pytest.approx(1.25, abs=1e-3)
        np.testing.assert_allclose((pos[0] + pos[1]) / 2, [1.5, 0, 0], atol=1e-9)


class TestComposition:
    def test_the_rest_pose_is_rebuilt_from_the_original_every_time(self):
        pts, tris = _tube()
        mesh = _mesh("Biceps Long R", pts, tris)
        binding = _Binding(mesh, is_muscle=True)
        original = np.asarray(mesh.rest_positions).copy()
        tissue = SoftTissueMorph()
        warp = lambda q: np.tile([0.0, 0.0, -1.0], (len(q), 1))
        tissue.apply([binding], 1.0, warp=warp)
        moved = np.asarray(mesh.rest_positions).copy()
        assert not np.allclose(moved, original)
        tissue.apply([binding], 0.6, warp=warp)
        tissue.apply([binding], 0.0, warp=None)
        np.testing.assert_allclose(mesh.rest_positions, original, atol=1e-5)

    def test_the_skinning_caches_are_dropped_so_the_change_reaches_a_frame(self):
        pts, tris = _tube()
        mesh = _mesh("Skin", pts, tris)
        binding = _Binding(mesh)
        binding._rest_f64 = np.zeros((3, 3))
        binding._pos_h = np.zeros((3, 4))
        SoftTissueMorph().apply([binding], 1.0,
                                warp=lambda q: np.zeros_like(q))
        assert binding._rest_f64 is None and binding._pos_h is None

    def test_a_rest_pose_of_a_different_length_is_refused(self):
        """Resizing a mesh desynchronises everything holding vertex indices."""
        pts, tris = _tube()
        mesh = _mesh("Skin", pts, tris)
        binding = _Binding(mesh)
        tissue = SoftTissueMorph()
        tissue.base_of(mesh)
        mesh.rest_positions = np.zeros(9, dtype=np.float32)
        tissue.apply([binding], 1.0, warp=lambda q: np.zeros_like(q))
        assert len(mesh.rest_positions) == 9, "the short array was left alone"


class TestBoneNameMatching:
    @pytest.mark.parametrize("name,key", [
        ("Right 5th Rib", "rib"), ("Costal Cartilage 1 R", "costal_cartilage"),
        ("Body of Sternum", "sternum"), ("Right Scapula", "scapula"),
        ("Right Tibia", "tibia"), ("Right Fibula", "fibula"),
        ("R 1st Metacarpal", "metacarpal"), ("Left Talus", "talus"),
    ])
    def test_bones_are_matched(self, name, key):
        assert BoneScaler()._match_bone(name) == key

    @pytest.mark.parametrize("name", [
        "Tibialis Ant. R", "Fibularis Long. R", "Subscapularis R",
        "Iliocostalis Lumb. R", "Iliotibial Tract R", "Biceps Long R", "Skin",
    ])
    def test_soft_tissue_is_not_mistaken_for_bone(self, name):
        """A substring test read 23 muscles as bones and scaled them."""
        assert BoneScaler()._match_bone(name) is None
