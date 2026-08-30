"""Volume geometry: the numbers a DICOM or NIfTI export is only as good as.

The important tests here compare :class:`~faceforge.export.volume.
VolumeGeometry` against the scanner's *own* ray-grid formula, recomputed
independently from :data:`faceforge.session.SCAN_ORIENTATIONS`.  If the two
ever disagree, every exported voxel position is wrong by that disagreement, and
nothing downstream would notice.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceforge.export import volume as vol
from faceforge.session import plane_frame


def ray_grid(origin, orientation, width, height, resolution):
    """The scanner's ray origins, recomputed here from first principles.

    Deliberately a transcription of ``ScannerEngine.scan``'s own grid
    construction rather than a call into it: this is the reference the export
    geometry is checked against, so it must not share code with the thing under
    test.
    """
    _normal, right, up = plane_frame(orientation)
    u = np.linspace(-0.5, 0.5, resolution)
    v = np.linspace(0.5, -0.5, resolution)
    return (
        np.asarray(origin, dtype=np.float64)
        + u[None, :, None] * (right * width)
        + v[:, None, None] * (up * height)
    )


# ---------------------------------------------------------------------------
# In-plane spacing
# ---------------------------------------------------------------------------


def test_spacing_is_field_over_resolution_minus_one():
    """The classic off-by-one that puts a silent scale error in every export."""
    g = vol.build_geometry(
        orientation="axial", centre=(0, 0, 0), field_width=200.0,
        field_height=100.0, resolution=101, slices=1, slice_spacing=1.0,
        slab_depth=1.0,
    )
    assert g.column_spacing == pytest.approx(2.0)
    assert g.row_spacing == pytest.approx(1.0)
    assert g.pixel_spacing == pytest.approx((1.0, 2.0)), (
        "DICOM PixelSpacing is (row spacing, column spacing), in that order"
    )
    # The wrong answer, spelled out so nobody reintroduces it.
    assert g.column_spacing != pytest.approx(200.0 / 101)


@pytest.mark.parametrize("orientation", ["axial", "coronal", "sagittal"])
@pytest.mark.parametrize("resolution", [8, 33, 128])
def test_voxel_positions_match_the_scanner_ray_grid(orientation, resolution):
    """The whole export rests on this: the tags describe where rays actually went."""
    centre = (12.0, -34.0, 1500.0)
    g = vol.build_geometry(
        orientation=orientation, centre=centre, field_width=180.0,
        field_height=240.0, resolution=resolution, slices=3,
        slice_spacing=4.0, slab_depth=4.0,
    )
    for k in range(g.slices):
        expected = ray_grid(g.slice_centre(k), orientation,
                            g.field_width, g.field_height, resolution)
        got = np.stack([
            np.stack([g.voxel_position(k, r, c) for c in range(resolution)])
            for r in range(resolution)
        ])
        assert np.allclose(got, expected, atol=1e-9), (
            f"{orientation} slice {k}: max error "
            f"{np.abs(got - expected).max():.3e} mm"
        )


def test_the_field_spans_exactly_the_requested_extent():
    g = vol.build_geometry(
        orientation="axial", centre=(0, 0, 0), field_width=200.0,
        field_height=200.0, resolution=64, slices=1, slice_spacing=1.0,
        slab_depth=1.0,
    )
    first = g.voxel_position(0, 0, 0)
    last = g.voxel_position(0, 63, 63)
    assert np.linalg.norm(last - first) == pytest.approx(
        np.hypot(200.0, 200.0))


# ---------------------------------------------------------------------------
# Slice stacking
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("orientation", ["axial", "coronal", "sagittal"])
def test_the_slice_stack_is_right_handed(orientation):
    g = vol.build_geometry(
        orientation=orientation, centre=(0, 0, 0), field_width=100.0,
        field_height=100.0, resolution=16, slices=5, slice_spacing=3.0,
        slab_depth=3.0,
    )
    assert g.right_handed
    assert np.allclose(np.cross(g.row_dir, g.col_dir), g.stack_dir, atol=1e-12)


def test_slices_are_evenly_spaced_and_centred_on_the_requested_centre():
    g = vol.build_geometry(
        orientation="axial", centre=(0.0, 0.0, 100.0), field_width=100.0,
        field_height=100.0, resolution=16, slices=4, slice_spacing=2.5,
        slab_depth=2.5,
    )
    centres = np.stack([g.slice_centre(k) for k in range(4)])
    assert np.allclose(centres.mean(axis=0), [0.0, 0.0, 100.0])
    steps = np.diff(centres, axis=0)
    assert np.allclose(np.linalg.norm(steps, axis=1), 2.5)


def test_the_scan_origin_puts_the_slab_centre_on_the_reported_position():
    """The scanner sweeps forward from the origin, so the origin sits behind."""
    g = vol.build_geometry(
        orientation="axial", centre=(0, 0, 0), field_width=100.0,
        field_height=100.0, resolution=16, slices=3, slice_spacing=6.0,
        slab_depth=6.0,
    )
    for k in range(3):
        origin = g.scan_origin(k)
        slab_centre = origin + 0.5 * g.slab_depth * g.ray_dir
        assert np.allclose(slab_centre, g.slice_centre(k)), (
            "a slice's reported position must be the centre of the slab that "
            "was actually sampled, not its near face"
        )


def test_a_slice_index_outside_the_volume_is_refused():
    g = vol.build_geometry(
        orientation="axial", centre=(0, 0, 0), field_width=100.0,
        field_height=100.0, resolution=16, slices=2, slice_spacing=1.0,
        slab_depth=1.0,
    )
    with pytest.raises(IndexError):
        g.slice_centre(2)


# ---------------------------------------------------------------------------
# Patient-coordinate conventions
# ---------------------------------------------------------------------------


def test_axial_image_orientation_patient_is_the_standard_axial_pair():
    g = vol.build_geometry(
        orientation="axial", centre=(0, 0, 0), field_width=100.0,
        field_height=100.0, resolution=16, slices=1, slice_spacing=1.0,
        slab_depth=1.0,
    )
    # Rows run to patient left (+X), columns run posterior (+Y): the textbook
    # axial ImageOrientationPatient.
    assert g.image_orientation_patient() == pytest.approx(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


@pytest.mark.parametrize("orientation,expected_stack", [
    ("axial", (0.0, 0.0, 1.0)),        # superior
    ("coronal", (0.0, -1.0, 0.0)),     # anterior
    ("sagittal", (-1.0, 0.0, 0.0)),    # patient right
])
def test_stack_directions_are_the_anatomical_axes(orientation, expected_stack):
    g = vol.build_geometry(
        orientation=orientation, centre=(0, 0, 0), field_width=100.0,
        field_height=100.0, resolution=16, slices=2, slice_spacing=1.0,
        slab_depth=1.0,
    )
    assert np.allclose(g.stack_dir, expected_stack)


def test_lps_to_ras_flips_x_and_y_only():
    g = vol.build_geometry(
        orientation="axial", centre=(0, 0, 0), field_width=100.0,
        field_height=100.0, resolution=16, slices=1, slice_spacing=1.0,
        slab_depth=1.0,
    )
    assert np.allclose(g.to_ras((10.0, 20.0, 30.0)), (-10.0, -20.0, 30.0))


def test_nifti_affine_voxel_sizes_match_the_spacings():
    g = vol.build_geometry(
        orientation="axial", centre=(0, 0, 0), field_width=200.0,
        field_height=100.0, resolution=101, slices=5, slice_spacing=3.0,
        slab_depth=3.0,
    )
    affine = g.nifti_affine()
    sizes = [np.linalg.norm(affine[:3, c]) for c in range(3)]
    assert sizes == pytest.approx([2.0, 1.0, 3.0])
    assert np.linalg.det(affine[:3, :3]) != 0


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs,match", [
    ({"resolution": 4}, "too small"),
    ({"slices": 0}, "not a volume"),
    ({"slice_spacing": 0.0}, "must be positive"),
    ({"slab_depth": -1.0}, "must be positive"),
    ({"field_width": 0.0}, "must be positive"),
])
def test_impossible_geometry_is_refused(kwargs, match):
    base = dict(orientation="axial", centre=(0, 0, 0), field_width=100.0,
                field_height=100.0, resolution=16, slices=2,
                slice_spacing=1.0, slab_depth=1.0)
    base.update(kwargs)
    with pytest.raises(vol.VolumeError, match=match):
        vol.build_geometry(**base)


@pytest.mark.parametrize("mode", ["xray", "anatomical"])
def test_non_tomographic_modes_are_refused(synthetic_scene, mode):
    with pytest.raises(vol.VolumeError, match="does not describe a tomographic"):
        vol.scan_volume(synthetic_scene, mode=mode)


def test_an_empty_volume_is_refused_rather_than_returned(synthetic_scene):
    """A plane outside the subject gives an all-zero stack; that is an error."""
    with pytest.raises(vol.VolumeError, match="no ray hit any geometry"):
        vol.scan_volume(
            synthetic_scene, centre=(0.0, 0.0, 5000.0), field_width=50.0,
            field_height=50.0, resolution=16, slices=2, slice_spacing=1.0,
        )


# ---------------------------------------------------------------------------
# The stacked data
# ---------------------------------------------------------------------------


def test_scan_volume_stacks_slices_with_the_expected_tissue_values(ct_volume):
    """Two known boxes at known x, so the values in the stack are predictable."""
    from faceforge.scanner.tissue_map import TissueMapper

    assert ct_volume.shape == (4, 32, 32)
    present = sorted(np.unique(ct_volume.data).tolist())
    bone = float(TissueMapper.get_value("bone", "ct"))
    organ = float(TissueMapper.get_value("organ", "ct"))
    assert present == pytest.approx([0.0, organ, bone], abs=1e-6), present

    # The mandible box is at x = -50 and the heart box at x = +50, so the
    # densest value must sit on the negative-x half of every slice.  Columns
    # increase along +X, so that is the left half of the array.
    mid = ct_volume.shape[2] // 2
    left = ct_volume.data[:, :, :mid]
    right = ct_volume.data[:, :, mid:]
    assert left.max() == pytest.approx(bone)
    assert right.max() == pytest.approx(organ)


def test_nifti_array_is_the_transpose_the_affine_expects(ct_volume):
    array = ct_volume.nifti_array()
    assert array.shape == (32, 32, 4)
    assert array[3, 5, 1] == ct_volume.data[1, 5, 3]


# ---------------------------------------------------------------------------
# The one claim that needs real anatomy: BodyParts3D coordinates are LPS
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_bodyparts3d_coordinates_are_lps_millimetres():
    """The measurement :data:`faceforge.export.volume.SCENE_TO_LPS` rests on.

    ``SCENE_TO_LPS`` is the identity because BodyParts3D coordinates already
    are (Left, Posterior, Superior).  That was measured from the meshes, and it
    is re-measured here so that a change to the STL loader breaks this test
    rather than silently rotating every exported volume.
    """
    from pathlib import Path

    from faceforge.constants import STL_DIR
    from faceforge.loaders.stl_parser import load_stl_file

    landmarks = {
        "right_femur": "FMA24474", "left_femur": "FMA24475",
        "sternum": "FMA7487", "t9": "FMA10014",
        "mandible": "FMA52748", "sacrum": "FMA16202",
    }
    centroids = {}
    for key, source_id in landmarks.items():
        path = Path(STL_DIR) / f"{source_id}.stl"
        if not path.is_file():
            pytest.skip(f"{source_id}.stl is not present")
        geom = load_stl_file(path)
        centroids[key] = geom.positions.reshape(-1, 3).mean(axis=0)

    # +X is patient LEFT.
    assert centroids["right_femur"][0] < 0 < centroids["left_femur"][0]
    # +Y is patient POSTERIOR (the sternum is anterior to a thoracic vertebra).
    assert centroids["sternum"][1] < centroids["t9"][1]
    # +Z is patient SUPERIOR.
    assert centroids["mandible"][2] > centroids["sacrum"][2]
    # Millimetres: a femur is a few hundred mm long, not a few, nor a few
    # hundred thousand.
    femur = load_stl_file(Path(STL_DIR) / "FMA24474.stl").positions.reshape(-1, 3)
    span = float(femur[:, 2].max() - femur[:, 2].min())
    assert 300.0 < span < 600.0, f"femur spans {span} along z; not millimetres?"

    assert np.allclose(vol.SCENE_TO_LPS, np.eye(3))
