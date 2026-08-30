"""DICOM output, read back with pydicom.

Every assertion here comes from a file that has been written to disk and
reparsed -- nothing checks the in-memory dataset that produced it.  The
geometry test is the one that matters most: it reconstructs each voxel's
patient coordinate from the tags alone (ImagePositionPatient,
ImageOrientationPatient, PixelSpacing) and compares against the scanner's ray
grid recomputed from :data:`faceforge.session.SCAN_ORIENTATIONS`.  A flipped
axis or a wrong voxel size fails there.

Not tested here: conformance to the DICOM standard's IOD requirements.  No
conformance validator was available in this environment, and asserting
conformance from inside the code that wrote the file would be circular.  What
is established is that pydicom parses every file, that the attributes this
exporter writes are present with the values it wrote, and that the geometry
round-trips.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")

from faceforge.export.dicom import (                       # noqa: E402
    DicomExportError, SIMULATION_STATEMENT, export_dicom_series,
)
from faceforge.export.hounsfield import (                  # noqa: E402
    HUMappingError, TISSUE_NOMINAL_HU,
)
from faceforge.export.provenance import (                  # noqa: E402
    BODYPARTS3D_ATTRIBUTION, collect_provenance,
)
from faceforge.export.volume import scan_volume            # noqa: E402
from tests.export.test_volume import ray_grid              # noqa: E402


# ---------------------------------------------------------------------------
# The series exists and parses
# ---------------------------------------------------------------------------


def test_one_file_per_slice_and_pydicom_reads_them_all(ct_volume, tmp_path):
    result = export_dicom_series(ct_volume, tmp_path / "series")
    assert result.slices == 4
    assert len(list((tmp_path / "series").glob("*.dcm"))) == 4

    for k, path in enumerate(result.files):
        ds = pydicom.dcmread(path)
        assert ds.InstanceNumber == k + 1
        assert ds.Rows == 32 and ds.Columns == 32
        assert ds.pixel_array.shape == (32, 32)


def test_the_series_shares_one_frame_of_reference(ct_volume, tmp_path):
    result = export_dicom_series(ct_volume, tmp_path / "series")
    frames = set()
    series = set()
    studies = set()
    instances = set()
    for path in result.files:
        ds = pydicom.dcmread(path)
        frames.add(ds.FrameOfReferenceUID)
        series.add(ds.SeriesInstanceUID)
        studies.add(ds.StudyInstanceUID)
        instances.add(ds.SOPInstanceUID)
    assert len(frames) == 1, "slices in different frames of reference cannot stack"
    assert len(series) == 1 and len(studies) == 1
    assert len(instances) == 4, "each slice needs its own SOPInstanceUID"


@pytest.mark.parametrize("mode,expected", [
    ("ct", ("CT", "1.2.840.10008.5.1.4.1.1.2")),
    ("mri_t1", ("MR", "1.2.840.10008.5.1.4.1.1.4")),
])
def test_modality_and_sop_class_follow_the_scan_mode(
    synthetic_scene, tmp_path, mode, expected,
):
    volume = scan_volume(
        synthetic_scene, resolution=16, slices=2, slice_spacing=5.0,
        slab_depth=50.0, field_width=200.0, field_height=200.0,
        mode=mode, reduction="max",
    )
    result = export_dicom_series(volume, tmp_path / mode)
    ds = pydicom.dcmread(result.files[0])
    assert (ds.Modality, ds.SOPClassUID) == expected
    if expected[0] == "MR":
        # Type 1 for the MR Image IOD; absent, many readers reject the file.
        assert ds.ScanningSequence == "RM"
        assert ds.SequenceVariant == "NONE"
        assert ds.MRAcquisitionType == "2D"


# ---------------------------------------------------------------------------
# Geometry, reconstructed from the tags alone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("orientation", ["axial", "coronal", "sagittal"])
def test_voxel_positions_reconstructed_from_the_tags_match_the_ray_grid(
    synthetic_scene, tmp_path, orientation,
):
    """The claim: a reader that trusts the tags lands where the rays were.

    BodyParts3D coordinates are LPS (measured; see
    ``tests/export/test_volume.py``), so a patient coordinate reconstructed
    from the tags is directly comparable to a scene coordinate.
    """
    resolution, slices = 16, 3
    volume = scan_volume(
        synthetic_scene, orientation=orientation, centre=(5.0, -3.0, 2.0),
        field_width=180.0, field_height=140.0, resolution=resolution,
        slices=slices, slice_spacing=4.0, slab_depth=50.0,
        mode="ct", reduction="max",
    )
    result = export_dicom_series(volume, tmp_path / orientation)

    worst = 0.0
    for k, path in enumerate(result.files):
        ds = pydicom.dcmread(path)
        ipp = np.array([float(v) for v in ds.ImagePositionPatient])
        iop = np.array([float(v) for v in ds.ImageOrientationPatient])
        row_dir, col_dir = iop[:3], iop[3:]
        row_spacing, col_spacing = (float(v) for v in ds.PixelSpacing)

        r = np.arange(int(ds.Rows))[:, None, None]
        c = np.arange(int(ds.Columns))[None, :, None]
        from_tags = (ipp + c * col_spacing * row_dir
                     + r * row_spacing * col_dir)

        expected = ray_grid(volume.geometry.slice_centre(k), orientation,
                            volume.geometry.field_width,
                            volume.geometry.field_height, resolution)
        worst = max(worst, float(np.abs(from_tags - expected).max()))

    assert worst < 1e-4, (
        f"{orientation}: voxel positions from the DICOM tags differ from the "
        f"scanner's ray grid by up to {worst:.3e} mm"
    )


def test_pixel_spacing_is_field_over_resolution_minus_one(ct_volume, tmp_path):
    result = export_dicom_series(ct_volume, tmp_path / "series")
    ds = pydicom.dcmread(result.files[0])
    expected = 200.0 / 31
    assert [float(v) for v in ds.PixelSpacing] == pytest.approx(
        [expected, expected], rel=1e-9)


def test_slice_thickness_and_spacing_describe_the_slabs(ct_volume, tmp_path):
    result = export_dicom_series(ct_volume, tmp_path / "series")
    positions = []
    for path in result.files:
        ds = pydicom.dcmread(path)
        assert float(ds.SliceThickness) == pytest.approx(50.0)
        assert float(ds.SpacingBetweenSlices) == pytest.approx(5.0)
        positions.append(np.array([float(v) for v in ds.ImagePositionPatient]))

    steps = np.diff(np.stack(positions), axis=0)
    assert np.allclose(np.linalg.norm(steps, axis=1), 5.0), (
        "consecutive ImagePositionPatient values must be one slice spacing apart"
    )


def test_the_stack_is_right_handed_with_respect_to_its_own_orientation(
    ct_volume, tmp_path,
):
    """Several converters silently reorder a left-handed series."""
    result = export_dicom_series(ct_volume, tmp_path / "series")
    first, second = (pydicom.dcmread(p) for p in result.files[:2])
    iop = np.array([float(v) for v in first.ImageOrientationPatient])
    normal = np.cross(iop[:3], iop[3:])
    step = (np.array([float(v) for v in second.ImagePositionPatient])
            - np.array([float(v) for v in first.ImagePositionPatient]))
    assert float(np.dot(normal, step)) > 0


# ---------------------------------------------------------------------------
# Pixel values: index mode
# ---------------------------------------------------------------------------


def test_index_mode_values_rescale_back_to_the_model_values(ct_volume, tmp_path):
    result = export_dicom_series(ct_volume, tmp_path / "series", hu_mode="index")
    for k, path in enumerate(result.files):
        ds = pydicom.dcmread(path)
        assert ds.RescaleType == "US", (
            "index mode must not claim Hounsfield units"
        )
        real = (ds.pixel_array.astype(np.float64) * float(ds.RescaleSlope)
                + float(ds.RescaleIntercept))
        assert np.allclose(real, ct_volume.data[k], atol=1e-6), (
            "applying the DICOM rescale must return the scanner's own values"
        )


def test_index_mode_says_in_the_file_that_it_is_not_hu(ct_volume, tmp_path):
    result = export_dicom_series(ct_volume, tmp_path / "series", hu_mode="index")
    ds = pydicom.dcmread(result.files[0])
    assert "NOT Hounsfield units" in ds.DerivationDescription


# ---------------------------------------------------------------------------
# Pixel values: class mode
# ---------------------------------------------------------------------------


def test_class_mode_writes_nominal_hu_and_labels_them_hu(ct_volume, tmp_path):
    result = export_dicom_series(ct_volume, tmp_path / "hu", hu_mode="class")
    seen = set()
    for path in result.files:
        ds = pydicom.dcmread(path)
        assert ds.RescaleType == "HU"
        assert float(ds.RescaleIntercept) == -1024.0
        assert float(ds.RescaleSlope) == 1.0
        real = (ds.pixel_array.astype(np.float64) * float(ds.RescaleSlope)
                + float(ds.RescaleIntercept))
        seen.update(np.unique(real).tolist())

    assert seen == {
        float(TISSUE_NOMINAL_HU["air"]),
        float(TISSUE_NOMINAL_HU["organ"]),
        float(TISSUE_NOMINAL_HU["bone"]),
    }, seen


def test_class_mode_carries_its_limitations_into_the_file(ct_volume, tmp_path):
    result = export_dicom_series(ct_volume, tmp_path / "hu", hu_mode="class")
    ds = pydicom.dcmread(result.files[0])
    text = ds.DerivationDescription
    assert text.startswith("SIMULATED")
    for phrase in ("nominal", "no noise", "Do not use for dosimetry"):
        assert phrase in text, phrase


def test_class_mode_is_refused_for_a_mean_reduction(synthetic_scene, tmp_path):
    volume = scan_volume(
        synthetic_scene, resolution=16, slices=2, slice_spacing=5.0,
        slab_depth=50.0, field_width=200.0, field_height=200.0,
        mode="ct", reduction="mean",
    )
    out = tmp_path / "bad"
    with pytest.raises(HUMappingError, match="reduction='max'"):
        export_dicom_series(volume, out, hu_mode="class")
    # The refusal happens before anything is written, so there is no partial
    # series left behind claiming to be HU.
    assert not out.exists() or not list(out.glob("*.dcm"))


def test_class_mode_is_refused_for_mri(synthetic_scene, tmp_path):
    volume = scan_volume(
        synthetic_scene, resolution=16, slices=2, slice_spacing=5.0,
        slab_depth=50.0, field_width=200.0, field_height=200.0,
        mode="mri_t1", reduction="max",
    )
    with pytest.raises(HUMappingError, match="no HU equivalent"):
        export_dicom_series(volume, tmp_path / "bad", hu_mode="class")


# ---------------------------------------------------------------------------
# "This is simulated" has to be in the file, not only in the docs
# ---------------------------------------------------------------------------


def test_every_file_declares_itself_derived_secondary_and_simulated(
    ct_volume, tmp_path,
):
    result = export_dicom_series(ct_volume, tmp_path / "series")
    for path in result.files:
        ds = pydicom.dcmread(path)
        assert list(ds.ImageType)[:2] == ["DERIVED", "SECONDARY"]
        assert ds.ConversionType == "SYN"
        assert ds.DerivationDescription.startswith("SIMULATED")
        assert SIMULATION_STATEMENT in ds.ImageComments
        assert str(ds.PatientName) == "FaceForge^Simulated"
        assert ds.Manufacturer == "FaceForge"


def test_the_attribution_is_in_every_file(ct_volume, tmp_path):
    """A licence condition: it travels with the pixels, not just the sidecar."""
    result = export_dicom_series(ct_volume, tmp_path / "series")
    for path in result.files:
        ds = pydicom.dcmread(path)
        assert BODYPARTS3D_ATTRIBUTION in ds.ImageComments


def test_the_sidecar_records_the_contributing_structures(
    synthetic_scene, ct_volume, tmp_path,
):
    records = collect_provenance(synthetic_scene.collect_meshes())
    result = export_dicom_series(ct_volume, tmp_path / "series",
                                 provenance=records)
    assert result.sidecar is not None
    doc = json.loads(result.sidecar.read_text(encoding="utf-8"))
    assert doc["attribution"] == BODYPARTS3D_ATTRIBUTION
    assert {s["ontology_id"] for s in doc["structures"]} == {
        "FMA:52748", "FMA:7088"}
    assert doc["conformance_validated"] is False, (
        "the sidecar must not imply a conformance check that was not run"
    )
    assert doc["volume"]["geometry"]["right_handed_stack"] is True


def test_the_result_does_not_claim_conformance(ct_volume, tmp_path):
    result = export_dicom_series(ct_volume, tmp_path / "series")
    assert result.as_dict()["conformance_validated"] is False
    assert any("conformance validator" in n for n in result.notes)


def test_a_non_tomographic_sop_class_is_refused(ct_volume, tmp_path):
    """A mode with no DICOM IOD must fail rather than pick one at random."""
    import dataclasses

    broken = dataclasses.replace(ct_volume, mode="xray")
    with pytest.raises(DicomExportError, match="no DICOM SOP class"):
        export_dicom_series(broken, tmp_path / "series")
