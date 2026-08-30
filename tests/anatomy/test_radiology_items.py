"""L4: tagged structures on simulated cross-sections.

Driven by synthetic box meshes rather than the BodyParts3D dataset, so these
run in the fast tier and the expected tag position is computable by hand: a
6-unit cube centred on the origin projects to a square centred in the frame,
so the tag must land in the middle.  The real scanner engine is used unmodified
-- it is called, not stubbed, because the thing under test is precisely whether
the double-scan tagging works against the real ray caster.
"""

import numpy as np
import pytest

from faceforge.anatomy.fma_taxonomy import SCHEMA_VERSION, Taxonomy
from faceforge.anatomy.item_generators import ItemGenerator
from faceforge.anatomy.radiology_items import (
    GRAYSCALE_MODES,
    REDUCTIONS,
    RadiologyItemBuilder,
    ScanPlane,
    mesh_world_bounds,
    plane_through_mesh,
)
from faceforge.core.mesh import BufferGeometry, Material, MeshInstance
from faceforge.scanner.engine import ScannerEngine
from faceforge.scanner.tissue_map import TissueMapper

PAYLOAD = {
    "schema_version": SCHEMA_VERSION,
    "_source": "synthetic",
    "nodes": {
        "1": {"label": "Bone organ", "parent": ""},
        "2": {"label": "Flat bone", "parent": "1"},
        "4": {"label": "Frontal bone", "parent": "2"},
        "5": {"label": "Parietal bone", "parent": "2"},
        "7": {"label": "Occipital bone", "parent": "2"},
    },
    "labels": {}, "part_of": {}, "composite_of": {},
}
FMA = {
    "FMA4": {"display_name": "Frontal Bone", "preferred_label": "Frontal bone",
             "system": "skeletal", "category": "skull_bones"},
    "FMA5": {"display_name": "Parietal Bone", "preferred_label": "Parietal bone",
             "system": "skeletal", "category": "skull_bones"},
    "FMA7": {"display_name": "Occipital Bone",
             "preferred_label": "Occipital bone",
             "system": "skeletal", "category": "skull_bones"},
}

IDENTITY = np.eye(4, dtype=np.float32)

_CUBE_FACES = (
    (0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5),
    (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4),
    (1, 5, 7), (1, 7, 3),
)


def cube(centre=(0.0, 0.0, 0.0), size=6.0, name="mesh") -> MeshInstance:
    """A closed, non-indexed 12-triangle cube."""
    d = size / 2.0
    corners = np.array([[x, y, z] for x in (-d, d) for y in (-d, d)
                        for z in (-d, d)], dtype=np.float32)
    tris = np.array([corners[i] for f in _CUBE_FACES for i in f],
                    dtype=np.float32) + np.asarray(centre, dtype=np.float32)
    geom = BufferGeometry(positions=tris.reshape(-1),
                          normals=np.zeros_like(tris).reshape(-1))
    return MeshInstance(name=name, geometry=geom, material=Material())


@pytest.fixture
def builder():
    return RadiologyItemBuilder(
        lambda: ScannerEngine(TissueMapper()),
        ItemGenerator(fma=FMA, taxonomy=Taxonomy(payload=PAYLOAD)),
    )


@pytest.fixture
def scene():
    return [
        (cube((0.0, 0.0, 0.0), 6.0, "Frontal Bone"), IDENTITY, "FMA4"),
        (cube((20.0, 0.0, 0.0), 6.0, "Parietal Bone"), IDENTITY, "FMA5"),
    ]


# ── plane derivation ─────────────────────────────────────────────────────

def test_world_bounds_of_a_cube():
    lo, hi = mesh_world_bounds(cube((1.0, 2.0, 3.0), 4.0), IDENTITY)
    assert lo == pytest.approx([-1.0, 0.0, 1.0])
    assert hi == pytest.approx([3.0, 4.0, 5.0])


def test_plane_spans_the_structure_so_rays_enter_from_outside():
    """Regression: a thin slab centred on the mesh centre starts *inside* a
    closed shell, so every ray exits past the far wall and the scan is empty."""
    mesh = cube((0.0, 0.0, 0.0), 6.0)
    plane = plane_through_mesh(mesh, IDENTITY, "axial")
    lo, hi = mesh_world_bounds(mesh, IDENTITY)
    normal = np.asarray(plane.normal)
    origin = np.asarray(plane.origin)
    # The ray origin must be outside the structure along the view axis...
    assert float(origin @ normal) < float(lo @ np.abs(normal)) or \
        float(origin @ normal) < float(hi @ np.abs(normal))
    # ...and the slab must be deep enough to cross it.
    assert plane.depth >= (hi - lo) @ np.abs(normal)


def test_plane_field_of_view_leaves_context_around_the_structure():
    plane = plane_through_mesh(cube(size=6.0), IDENTITY, "axial", margin=1.35)
    assert plane.width == pytest.approx(6.0 * 1.35)
    assert plane.height == pytest.approx(6.0 * 1.35)


def test_explicit_depth_is_honoured():
    plane = plane_through_mesh(cube(size=6.0), IDENTITY, "axial", depth=2.0)
    assert plane.depth == 2.0


@pytest.mark.parametrize("name", ["axial", "coronal", "sagittal"])
def test_the_three_standard_planes_are_orthonormal(name):
    plane = plane_through_mesh(cube(size=6.0), IDENTITY, name)
    n, r, u = (np.asarray(v) for v in (plane.normal, plane.right, plane.up))
    for vec in (n, r, u):
        assert np.linalg.norm(vec) == pytest.approx(1.0)
    assert n @ r == pytest.approx(0.0)
    assert n @ u == pytest.approx(0.0)
    assert r @ u == pytest.approx(0.0)


def test_scan_args_match_the_engine_signature():
    plane = plane_through_mesh(cube(), IDENTITY, "axial")
    import inspect
    params = set(inspect.signature(ScannerEngine.scan).parameters)
    assert set(plane.as_scan_args()) <= params


# ── item construction ────────────────────────────────────────────────────

def test_builds_an_l4_spot_item_tagged_on_the_structure(builder, scene):
    result = builder.build(scene, "FMA4", plane_name="axial", resolution=64,
                           mode="ct", seed=1)
    assert result is not None
    item = result.item
    assert (item.level, item.fmt) == ("L4", "spot")
    assert item.answer.item_id == "FMA4"
    assert item.answer.text == "Frontal bone"
    assert "CT" in item.stem and "axial" in item.stem
    # A centred cube projects to a centred square, so the tag is mid-frame.
    assert item.tag_xy == pytest.approx((0.5, 0.5), abs=0.05)


def test_the_tag_pixel_is_inside_the_structure(builder, scene):
    result = builder.build(scene, "FMA4", resolution=64, mode="ct", seed=1)
    x, y = result.tag_px
    focus_only = builder._scan([scene[0]], result.plane, 64, "ct", "max")
    assert focus_only[y, x] > 0, "tag must point at the structure itself"


def test_the_image_is_the_whole_scene_not_just_the_focus(builder):
    scene = [
        (cube((-3.0, 0.0, 0.0), 4.0, "Frontal Bone"), IDENTITY, "FMA4"),
        (cube((3.0, 0.0, 0.0), 4.0, "Parietal Bone"), IDENTITY, "FMA5"),
    ]
    plane = ScanPlane(origin=(0.0, 6.0, 0.0), normal=(0.0, -1.0, 0.0),
                      right=(1.0, 0.0, 0.0), up=(0.0, 0.0, -1.0),
                      width=20.0, height=20.0, depth=12.0, label="axial")
    result = builder.build(scene, "FMA4", plane=plane, resolution=64,
                           mode="ct", seed=1)
    focus_only = builder._scan([scene[0]], plane, 64, "ct", "max")
    assert int((result.image > 0).sum()) > int((focus_only > 0).sum())


def test_provenance_records_the_render_and_the_label(builder, scene):
    item = builder.build(scene, "FMA4", resolution=64, mode="mri", seed=1).item
    kinds = [p.kind for p in item.provenance]
    assert kinds == ["fma_label", "scanner_render"]
    render = item.provenance[1]
    assert "mri" in render.reference
    assert "nearest the centroid" in render.detail
    assert item.verified is True


def test_distractors_come_from_the_same_neighbourhood(builder, scene):
    item = builder.build(scene, "FMA4", resolution=64, mode="ct", seed=1).item
    roles = set(item.distractor_roles)
    assert roles <= {"is_a_sibling", "is_a_cousin", "shares_whole",
                     "same_system_and_category", "same_system", "same_category"}
    assert "is_a_sibling" in roles


# ── refusals and guards ──────────────────────────────────────────────────

def test_structure_absent_from_the_scene_yields_nothing(builder, scene):
    assert builder.build(scene, "FMA999", resolution=32) is None


def test_plane_that_misses_the_structure_yields_nothing(builder, scene):
    far = ScanPlane(origin=(0.0, 500.0, 0.0), normal=(0.0, -1.0, 0.0),
                    right=(1.0, 0.0, 0.0), up=(0.0, 0.0, -1.0),
                    width=10.0, height=10.0, depth=2.0, label="axial")
    assert builder.build(scene, "FMA4", plane=far, resolution=32) is None


def test_a_structure_covering_too_few_pixels_yields_nothing(builder, scene):
    assert builder.build(scene, "FMA4", resolution=64, mode="ct",
                         min_mask_pixels=10 ** 6) is None


def test_anatomical_mode_is_rejected(builder, scene):
    with pytest.raises(ValueError, match="colour matching"):
        builder.build(scene, "FMA4", mode="anatomical", resolution=32)


def test_unknown_reduction_is_rejected(builder, scene):
    with pytest.raises(ValueError, match="reduction"):
        builder.build(scene, "FMA4", reduction="mip", resolution=32)


@pytest.mark.parametrize("mode", GRAYSCALE_MODES)
def test_every_grayscale_mode_produces_an_item(builder, scene, mode):
    assert builder.build(scene, "FMA4", resolution=48, mode=mode) is not None


@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_every_reduction_produces_an_item(builder, scene, reduction):
    assert builder.build(scene, "FMA4", resolution=48,
                         reduction=reduction) is not None


# ── determinism and batching ─────────────────────────────────────────────

def test_the_same_seed_gives_the_same_item(builder, scene):
    a = builder.build(scene, "FMA4", resolution=48, seed=5)
    b = builder.build(scene, "FMA4", resolution=48, seed=5)
    assert a.item.uid == b.item.uid
    assert a.tag_px == b.tag_px
    assert np.array_equal(a.image, b.image)


def test_build_many_skips_what_it_cannot_render(builder, scene):
    got = builder.build_many(scene, ["FMA999", "FMA4", "FMA5"], count=5,
                             resolution=48)
    assert [r.item.focus_id for r in got] == ["FMA4", "FMA5"]


def test_build_many_respects_the_count(builder, scene):
    assert len(builder.build_many(scene, ["FMA4", "FMA5"], count=1,
                                  resolution=48)) == 1
