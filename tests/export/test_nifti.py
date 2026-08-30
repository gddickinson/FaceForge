"""NIfTI output, read back with nibabel.

The affine is the whole ballgame.  A NIfTI with a wrong affine loads happily,
looks right and measures wrong, so the tests here reload the file and check the
affine against voxel positions computed independently from the scanner's ray
grid -- not against the affine that was written.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

from faceforge.export.hounsfield import (                  # noqa: E402
    HUMappingError, TISSUE_NOMINAL_HU,
)
from faceforge.export.nifti import (                       # noqa: E402
    NIFTI_COMMENT_ECODE, export_nifti,
)
from faceforge.export.provenance import (                  # noqa: E402
    BODYPARTS3D_ATTRIBUTION, collect_provenance,
)
from faceforge.export.volume import LPS_TO_RAS, scan_volume  # noqa: E402
from tests.export.test_volume import ray_grid              # noqa: E402


# ---------------------------------------------------------------------------
# It loads, with the shape and dtype claimed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".nii", ".nii.gz"])
def test_nibabel_loads_what_was_written(ct_volume, tmp_path, suffix):
    out = tmp_path / f"vol{suffix}"
    result = export_nifti(ct_volume, out)

    img = nib.load(out)
    assert img.shape == (32, 32, 4) == result.shape
    assert img.get_fdata().dtype == np.float64          # get_fdata always upcasts
    assert img.get_data_dtype() == np.float32
    assert np.allclose(img.get_fdata(), ct_volume.nifti_array(), atol=1e-6)


def test_the_array_axes_are_column_row_slice(ct_volume, tmp_path):
    """A transposed volume is the classic silent NIfTI bug."""
    out = tmp_path / "vol.nii"
    export_nifti(ct_volume, out)
    data = np.asarray(nib.load(out).dataobj)
    for k in range(ct_volume.shape[0]):
        assert np.allclose(data[:, :, k], ct_volume.data[k].T, atol=1e-6)


def test_units_are_millimetres(ct_volume, tmp_path):
    out = tmp_path / "vol.nii"
    export_nifti(ct_volume, out)
    header = nib.load(out).header
    assert header.get_xyzt_units()[0] == "mm"


# ---------------------------------------------------------------------------
# The affine
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("orientation", ["axial", "coronal", "sagittal"])
def test_the_affine_maps_voxels_to_where_the_rays_actually_were(
    synthetic_scene, tmp_path, orientation,
):
    resolution, slices = 16, 3
    volume = scan_volume(
        synthetic_scene, orientation=orientation, centre=(5.0, -3.0, 2.0),
        field_width=180.0, field_height=140.0, resolution=resolution,
        slices=slices, slice_spacing=4.0, slab_depth=50.0,
        mode="ct", reduction="max",
    )
    out = tmp_path / f"{orientation}.nii"
    export_nifti(volume, out)

    affine = nib.load(out).affine
    i, j, k = np.meshgrid(np.arange(resolution), np.arange(resolution),
                          np.arange(slices), indexing="ij")
    voxels = np.stack([i, j, k, np.ones_like(i)], axis=-1).astype(np.float64)
    ras = voxels @ affine.T

    worst = 0.0
    for slice_index in range(slices):
        grid = ray_grid(volume.geometry.slice_centre(slice_index), orientation,
                        volume.geometry.field_width,
                        volume.geometry.field_height, resolution)
        # grid is [row, col, xyz] in scene (LPS) coordinates; the NIfTI array
        # is [col, row, slice] in RAS.
        expected = np.einsum("ij,rcj->rci", LPS_TO_RAS, grid)
        got = ras[:, :, slice_index, :3].transpose(1, 0, 2)
        worst = max(worst, float(np.abs(got - expected).max()))

    assert worst < 1e-4, (
        f"{orientation}: the affine places voxels up to {worst:.3e} mm away "
        "from where the scanner cast its rays"
    )


def test_voxel_sizes_read_back_as_the_scan_spacings(ct_volume, tmp_path):
    out = tmp_path / "vol.nii"
    export_nifti(ct_volume, out)
    zooms = nib.load(out).header.get_zooms()[:3]
    in_plane = 200.0 / 31
    assert zooms == pytest.approx((in_plane, in_plane, 5.0), rel=1e-5)


def test_axis_codes_describe_an_axial_stack(ct_volume, tmp_path):
    """The axis labels a viewer will show, from nibabel's own reading."""
    out = tmp_path / "vol.nii"
    result = export_nifti(ct_volume, out)
    codes = nib.aff2axcodes(nib.load(out).affine)
    assert codes == result.axis_codes
    # Columns run to patient left, so i decreases in RAS x: 'L'.  Rows run
    # posterior, so j is 'P'.  Slices stack superior: 'S'.
    assert codes == ("L", "P", "S")


def test_the_affine_is_not_degenerate(ct_volume, tmp_path):
    out = tmp_path / "vol.nii"
    export_nifti(ct_volume, out)
    affine = nib.load(out).affine
    assert abs(np.linalg.det(affine[:3, :3])) > 1e-6


# ---------------------------------------------------------------------------
# Values and their declaration
# ---------------------------------------------------------------------------


def test_index_mode_writes_the_model_values_unscaled(ct_volume, tmp_path):
    out = tmp_path / "vol.nii"
    export_nifti(ct_volume, out, hu_mode="index")
    img = nib.load(out)
    slope, inter = img.header.get_slope_inter()
    assert (slope, inter) in ((1.0, 0.0), (None, None)), (slope, inter)
    values = np.unique(np.asarray(img.dataobj))
    assert values.min() >= 0.0 and values.max() <= 1.0


def test_class_mode_writes_int16_nominal_hu(ct_volume, tmp_path):
    out = tmp_path / "hu.nii"
    result = export_nifti(ct_volume, out, hu_mode="class")
    assert result.dtype == "int16"

    img = nib.load(out)
    assert img.get_data_dtype() == np.int16
    values = set(np.unique(np.asarray(img.dataobj)).tolist())
    assert values == {
        TISSUE_NOMINAL_HU["air"],
        TISSUE_NOMINAL_HU["organ"],
        TISSUE_NOMINAL_HU["bone"],
    }, values


def test_class_mode_is_refused_where_it_is_not_defensible(
    synthetic_scene, tmp_path,
):
    volume = scan_volume(
        synthetic_scene, resolution=16, slices=2, slice_spacing=5.0,
        slab_depth=50.0, field_width=200.0, field_height=200.0,
        mode="ct", reduction="mean",
    )
    out = tmp_path / "bad.nii"
    with pytest.raises(HUMappingError, match="reduction='max'"):
        export_nifti(volume, out, hu_mode="class")
    assert not out.exists()


# ---------------------------------------------------------------------------
# Provenance: retrievable, not just present
# ---------------------------------------------------------------------------


def test_the_comment_extension_round_trips_through_nibabel(ct_volume, tmp_path):
    """Attribution a library can hand back, rather than an unreadable comment."""
    out = tmp_path / "vol.nii"
    export_nifti(ct_volume, out)

    extensions = nib.load(out).header.extensions
    payloads = [e for e in extensions if e.get_code() == NIFTI_COMMENT_ECODE]
    assert len(payloads) == 1

    doc = json.loads(payloads[0].get_content().decode("utf-8"))
    assert doc["attribution"] == BODYPARTS3D_ATTRIBUTION
    assert doc["simulated"] is True
    assert doc["simulation"].startswith("SIMULATED IMAGE")
    assert "NOT Hounsfield units" in doc["value_description"]
    assert doc["geometry"]["right_handed_stack"] is True
    assert np.allclose(np.asarray(doc["affine_voxel_to_ras_mm"]),
                       nib.load(out).affine, atol=1e-5), (
        "the affine recorded in the extension must be the affine in the header"
    )


def test_descrip_marks_the_file_as_simulated(ct_volume, tmp_path):
    out = tmp_path / "vol.nii"
    export_nifti(ct_volume, out)
    descrip = bytes(nib.load(out).header["descrip"]).rstrip(b"\x00").decode()
    assert descrip.startswith("FaceForge SIMULATED")
    assert len(descrip) <= 80


def test_the_sidecar_lists_the_structures(synthetic_scene, ct_volume, tmp_path):
    records = collect_provenance(synthetic_scene.collect_meshes())
    out = tmp_path / "vol.nii.gz"
    result = export_nifti(ct_volume, out, provenance=records)

    assert result.sidecar is not None
    doc = json.loads(result.sidecar.read_text(encoding="utf-8"))
    assert {s["source_id"] for s in doc["structures"]} == {"FMA52748", "FMA7088"}
    assert doc["nifti"]["axis_codes"] == ["L", "P", "S"]
    assert doc["pixel_values"]["rescale_type"] == "US"
