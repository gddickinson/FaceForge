"""DICOM export of virtual-scanner volumes.

A picture of a cross-section saved as ``.dcm`` is not a medical image; it is a
screenshot in a DICOM wrapper.  Three things separate the two, and this module
is organised around them.

**1. The pixel values have to mean something, and be labelled with what they
mean.**  See :mod:`faceforge.export.hounsfield`, which is where the argument
lives.  Short version: the scanner's model cannot produce Hounsfield units, so
the default (``hu_mode="index"``) stores the model's own dimensionless value
with ``RescaleSlope=0.001``, ``RescaleIntercept=0`` and ``RescaleType="US"`` --
tags that describe what the numbers actually are.  ``hu_mode="class"`` is
available for the one configuration where an HU mapping is defensible
(``mode="ct"`` with ``reduction="max"``, where the value inverts exactly to a
tissue class) and writes nominal per-class HU with the limitation spelled out
in DerivationDescription.  Nothing here fabricates plausible-looking HU.

**2. The geometry has to be right.**  PixelSpacing, SliceThickness,
SpacingBetweenSlices, ImagePositionPatient, ImageOrientationPatient and a
single FrameOfReferenceUID across the series, all derived in
:mod:`faceforge.export.volume` from the scanner's actual ray grid -- including
the two easy-to-miss details (spacing is ``field/(resolution-1)``; the slab is
cast from half a thickness behind its reported centre).
``tests/export/test_dicom.py`` reads the written files back with pydicom and
reconstructs each voxel position from the tags alone, then compares against the
ray grid.

**3. It has to say it is simulated.**  Every file carries
``ImageType = DERIVED\\SECONDARY``, a DerivationDescription beginning with
"SIMULATED", ``PatientName = FaceForge^Simulated`` and the BodyParts3D
attribution in ImageComments.  A synthetic study that could be mistaken for an
acquisition is a hazard in a teaching PACS.

What is *not* claimed
---------------------
The output has not been checked against a DICOM conformance validator -- no
such validator was available in this environment, and the standard's IOD
definitions are not bundled here.  What *is* verified is that pydicom reads
every file back, that the required Type 1/Type 2 attributes this module writes
are present with the values it wrote, and that the geometry round-trips.  That
is a weaker statement than "conformant", and it is deliberately the only one
made.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from faceforge.export.hounsfield import RescaleSpec, to_hounsfield
from faceforge.export.provenance import (
    BODYPARTS3D_ATTRIBUTION,
    GENERATOR,
    provenance_document,
)
from faceforge.export.volume import ScanVolume

logger = logging.getLogger(__name__)

#: SOP Class per scan mode.  CT and MR are the two tomographic IODs; the
#: scanner's other modes are refused upstream by
#: :data:`faceforge.export.volume.VOLUME_MODES`.
_SOP_CLASS = {
    "ct": ("1.2.840.10008.5.1.4.1.1.2", "CT"),
    "mri_t1": ("1.2.840.10008.5.1.4.1.1.4", "MR"),
    "mri_t2": ("1.2.840.10008.5.1.4.1.1.4", "MR"),
}

#: ImageType value 3.  The CT IOD's defined terms for it are AXIAL and
#: LOCALIZER; a coronal or sagittal stack is neither, so it gets REFORMATTED,
#: which is what a reformatted plane is called elsewhere in the standard.  This
#: is flagged in the result notes because it is a judgement call, not a checked
#: conformance claim.
_PLANE_TERM = {"axial": "AXIAL", "coronal": "REFORMATTED",
               "sagittal": "REFORMATTED"}

SIMULATION_STATEMENT = (
    "SIMULATED IMAGE. Not an acquisition from any scanner and not derived "
    "from any patient. Produced by the FaceForge virtual scanner by casting "
    "rays through BodyParts3D triangle surface meshes."
)


class DicomExportError(RuntimeError):
    """A DICOM series that cannot be written correctly."""


@dataclass(frozen=True)
class DicomSeriesResult:
    """What was written, and what was and was not verified."""

    directory: Path
    files: tuple[Path, ...]
    modality: str
    sop_class_uid: str
    study_uid: str
    series_uid: str
    frame_of_reference_uid: str
    rescale: RescaleSpec
    sidecar: Path | None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def slices(self) -> int:
        return len(self.files)

    def as_dict(self) -> dict[str, Any]:
        return {
            "directory": str(self.directory),
            "slices": self.slices,
            "files": [f.name for f in self.files],
            "modality": self.modality,
            "sop_class_uid": self.sop_class_uid,
            "study_instance_uid": self.study_uid,
            "series_instance_uid": self.series_uid,
            "frame_of_reference_uid": self.frame_of_reference_uid,
            "pixel_values": self.rescale.as_dict(),
            "sidecar": None if self.sidecar is None else str(self.sidecar),
            "simulated": True,
            "conformance_validated": False,
            "notes": list(self.notes),
        }


def _dicom_datetime() -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y%m%d"), now.strftime("%H%M%S.%f")[:13]


def _window(spec: RescaleSpec) -> tuple[float, float]:
    """Window centre/width covering the data actually present."""
    lo, hi = spec.real_min, spec.real_max
    width = max(hi - lo, 1e-6)
    return ((lo + hi) / 2.0, width)


def export_dicom_series(
    volume: ScanVolume,
    directory: Path | str,
    *,
    hu_mode: str = "index",
    patient_name: str = "FaceForge^Simulated",
    patient_id: str = "FACEFORGE-SIM",
    series_description: str | None = None,
    study_uid: str | None = None,
    series_uid: str | None = None,
    frame_of_reference_uid: str | None = None,
    provenance: Any = None,
    sidecar: bool = True,
) -> DicomSeriesResult:
    """Write *volume* as a DICOM series, one file per slice.

    Parameters that matter
    ----------------------
    hu_mode
        ``"index"`` (default) or ``"class"``.  See
        :mod:`faceforge.export.hounsfield`.
    provenance
        Optional sequence of
        :class:`~faceforge.export.provenance.StructureProvenance` for the
        contributing structures, written to the sidecar.

    All UIDs default to freshly generated ones, so two exports of the same
    volume are two distinct studies -- which is correct: they are not the same
    acquisition, because there was no acquisition.
    """
    try:
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    except ImportError as exc:                           # pragma: no cover
        raise DicomExportError(
            "DICOM export needs pydicom: pip install pydicom"
        ) from exc

    directory = Path(directory)
    geometry = volume.geometry
    try:
        sop_class_uid, modality = _SOP_CLASS[volume.mode]
    except KeyError:
        raise DicomExportError(
            f"no DICOM SOP class for scan mode {volume.mode!r}; "
            f"supported: {sorted(_SOP_CLASS)}"
        ) from None

    spec = to_hounsfield(
        volume.data, hu_mode=hu_mode, mode=volume.mode,
        reduction=volume.reduction,
    )
    if spec.rescale_type == "HU" and modality != "CT":   # pragma: no cover
        raise DicomExportError(
            "refusing to write RescaleType=HU on a non-CT modality"
        )

    directory.mkdir(parents=True, exist_ok=True)
    study_uid = study_uid or generate_uid()
    series_uid = series_uid or generate_uid()
    frame_uid = frame_of_reference_uid or generate_uid()
    date, time = _dicom_datetime()
    centre, width = _window(spec)
    iop = geometry.image_orientation_patient()
    description = series_description or (
        f"FaceForge SIMULATED {volume.mode} {geometry.orientation}")

    files: list[Path] = []
    for k in range(geometry.slices):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = sop_class_uid
        instance_uid = generate_uid()
        meta.MediaStorageSOPInstanceUID = instance_uid
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.ImplementationVersionName = "FACEFORGE"

        ds = FileDataset({}, {}, file_meta=meta, preamble=b"\0" * 128)

        # -- identity --------------------------------------------------
        ds.SOPClassUID = sop_class_uid
        ds.SOPInstanceUID = instance_uid
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = frame_uid
        ds.PositionReferenceIndicator = ""
        ds.SeriesNumber = 1
        ds.InstanceNumber = k + 1
        ds.AcquisitionNumber = 1
        ds.StudyID = "1"

        # -- "this is not a patient" -----------------------------------
        ds.PatientName = patient_name
        ds.PatientID = patient_id
        ds.PatientBirthDate = ""
        ds.PatientSex = ""
        ds.StudyDate = ds.SeriesDate = ds.ContentDate = date
        ds.StudyTime = ds.SeriesTime = ds.ContentTime = time
        ds.Modality = modality
        ds.Manufacturer = "FaceForge"
        ds.ManufacturerModelName = "FaceForge virtual scanner"
        ds.SoftwareVersions = GENERATOR
        ds.StudyDescription = "FaceForge simulated anatomical study"
        ds.SeriesDescription = description
        ds.ImageType = ["DERIVED", "SECONDARY",
                        _PLANE_TERM.get(geometry.orientation, "REFORMATTED")]
        ds.ConversionType = "SYN"           # synthetic image
        ds.DerivationDescription = (
            f"{SIMULATION_STATEMENT} {spec.description}"
        )[:1024]
        ds.ImageComments = (
            f"{SIMULATION_STATEMENT} | {BODYPARTS3D_ATTRIBUTION} | "
            f"scan mode={volume.mode} reduction={volume.reduction} "
            f"hu_mode={spec.mode} unit={spec.unit}"
        )[:10240]

        if modality == "MR":
            # Type 1 for the MR Image IOD.  'RM' (research mode) is the honest
            # defined term for a sequence that does not exist.
            ds.ScanningSequence = "RM"
            ds.SequenceVariant = "NONE"
            ds.ScanOptions = ""
            ds.MRAcquisitionType = "2D"

        # -- geometry --------------------------------------------------
        row_spacing, col_spacing = geometry.pixel_spacing
        ds.PixelSpacing = [f"{row_spacing:.10g}", f"{col_spacing:.10g}"]
        ds.SliceThickness = f"{geometry.slab_depth:.10g}"
        ds.SpacingBetweenSlices = f"{geometry.slice_spacing:.10g}"
        position = geometry.image_position_patient(k)
        ds.ImagePositionPatient = [f"{v:.10g}" for v in position]
        ds.ImageOrientationPatient = [f"{v:.10g}" for v in iop]
        ds.SliceLocation = f"{float(np.dot(geometry.to_lps(geometry.slice_centre(k)), geometry.direction_to_lps(geometry.stack_dir))):.10g}"

        # -- pixels ----------------------------------------------------
        plane = spec.stored[k]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Rows, ds.Columns = int(plane.shape[0]), int(plane.shape[1])
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1 if spec.signed else 0
        ds.RescaleIntercept = f"{spec.intercept:.10g}"
        ds.RescaleSlope = f"{spec.slope:.10g}"
        ds.RescaleType = spec.rescale_type
        ds.WindowCenter = f"{centre:.10g}"
        ds.WindowWidth = f"{width:.10g}"
        ds.PixelData = np.ascontiguousarray(plane).tobytes()

        path = directory / f"{volume.mode}_{geometry.orientation}_{k:04d}.dcm"
        ds.save_as(path, enforce_file_format=True)
        files.append(path)

    notes = [
        f"pixel values: {spec.unit}; RescaleType={spec.rescale_type}",
        "ImageType DERIVED/SECONDARY, ConversionType SYN, and a "
        "DerivationDescription beginning SIMULATED mark every file as "
        "synthetic.",
        f"one FrameOfReferenceUID ({frame_uid}) across all "
        f"{geometry.slices} slices; the stack is right-handed with respect to "
        "ImageOrientationPatient.",
        "not checked against a DICOM conformance validator: none was "
        "available here.  Verified only that pydicom reads the files back and "
        "the geometry round-trips.",
    ]
    notes.extend(spec.notes)
    if geometry.orientation != "axial":
        notes.append(
            f"ImageType value 3 is REFORMATTED for a {geometry.orientation} "
            "stack; the CT IOD's defined terms are AXIAL and LOCALIZER, "
            "neither of which describes this plane."
        )
    if geometry.transform_applied != "none":
        notes.append(
            f"mesh transform {geometry.transform_applied!r} was applied to the "
            "scene, so scene coordinates are NOT BodyParts3D LPS and the "
            "patient-coordinate tags inherit that transform."
        )

    sidecar_path: Path | None = None
    if sidecar:
        sidecar_path = directory / "provenance.json"
        doc = provenance_document(
            list(provenance or []), fmt="dicom",
            target=directory.name,
            extra={
                "simulation": SIMULATION_STATEMENT,
                "volume": volume.as_dict(),
                "pixel_values": spec.as_dict(),
                "dicom": {
                    "modality": modality,
                    "sop_class_uid": sop_class_uid,
                    "study_instance_uid": study_uid,
                    "series_instance_uid": series_uid,
                    "frame_of_reference_uid": frame_uid,
                    "files": [f.name for f in files],
                },
                "conformance_validated": False,
            },
        )
        sidecar_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False),
                                encoding="utf-8")

    logger.info("wrote %d DICOM slices to %s (%s, %s)", len(files), directory,
                modality, spec.rescale_type)
    return DicomSeriesResult(
        directory=directory,
        files=tuple(files),
        modality=modality,
        sop_class_uid=sop_class_uid,
        study_uid=study_uid,
        series_uid=series_uid,
        frame_of_reference_uid=frame_uid,
        rescale=spec,
        sidecar=sidecar_path,
        notes=tuple(notes),
    )
